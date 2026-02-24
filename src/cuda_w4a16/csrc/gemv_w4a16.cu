#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cstdint>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kTileOC = 16;

__device__ inline int min_int(int a, int b) {
  return a < b ? a : b;
}

__device__ inline uint8_t unpack_u4_from_i32(uint32_t packed, int idx) {
  return static_cast<uint8_t>((packed >> (idx * 4)) & 0x0F);
}

__device__ inline uint8_t load_w4_code_from_i32_row(
    const int32_t* __restrict__ qweight_row,
    int k) {
  const uint32_t pack = static_cast<uint32_t>(qweight_row[k >> 3]);
  return unpack_u4_from_i32(pack, k & 7);
}

__device__ inline uint8_t load_zero_u4_from_i32_row(
    const int32_t* __restrict__ qzero_row,
    int gid) {
  const uint32_t pack = static_cast<uint32_t>(qzero_row[gid >> 3]);
  return unpack_u4_from_i32(pack, gid & 7);
}

__global__ void gemv_w4a16_asym_i32_kernel(
    const half* __restrict__ inputs,
    const int32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const int32_t* __restrict__ qzeros,
    half* __restrict__ outputs,
    int M,
    int IC,
    int OC,
    int groups,
    int group_size,
    int weight_stride_i32,
    int zero_stride_i32,
    int packed_iters) {
  const int row = static_cast<int>(blockIdx.y);
  const int oc_base = static_cast<int>(blockIdx.x) * kTileOC;
  if (row >= M || oc_base >= OC) {
    return;
  }

  const int tile_cols = min_int(kTileOC, OC - oc_base);
  const int32_t* weight_rows[kTileOC];
  const half* scale_rows[kTileOC];
  const int32_t* zero_rows[kTileOC];

  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    if (t < tile_cols) {
      const int oc = oc_base + t;
      weight_rows[t] = qweight + oc * weight_stride_i32;
      scale_rows[t] = scales + oc * groups;
      zero_rows[t] = qzeros + oc * zero_stride_i32;
    } else {
      weight_rows[t] = nullptr;
      scale_rows[t] = nullptr;
      zero_rows[t] = nullptr;
    }
  }

  float acc[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    acc[t] = 0.0f;
  }

  // GlowQ stores 8 int4 codes per int32. group_size is quantizer-constrained to
  // multiples of 8, so each packed word belongs to a single quant group.
  for (int pack = threadIdx.x; pack < packed_iters; pack += blockDim.x) {
    const int k0 = pack * 8;
    const int gid = k0 / group_size;
    if (gid >= groups) {
      continue;
    }

    float in8[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int k = k0 + i;
      in8[i] = (k < IC) ? __half2float(inputs[row * IC + k]) : 0.0f;
    }

    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      if (t >= tile_cols) {
        break;
      }
      const float s = __half2float(scale_rows[t][gid]);
      const float z = static_cast<float>(load_zero_u4_from_i32_row(zero_rows[t], gid));
      uint32_t packed_w = static_cast<uint32_t>(weight_rows[t][pack]);

      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float code = static_cast<float>(packed_w & 0x0F);
        partial += (code - z) * s * in8[i];
        packed_w >>= 4;
      }
      acc[t] += partial;
    }
  }

  __shared__ float shm[kThreads][kTileOC];
  const int tid = threadIdx.x;
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    shm[tid][t] = acc[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t = 0; t < kTileOC; ++t) {
        shm[tid][t] += shm[tid + stride][t];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      const int oc = oc_base + t;
      if (t < tile_cols && oc < OC) {
        outputs[row * OC + oc] = __float2half(shm[0][t]);
      }
    }
  }
}

}  // namespace

torch::Tensor w4a16_gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size) {
  TORCH_CHECK(_in_feats.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(_kernel.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(_scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(_zeros.is_cuda(), "qzeros must be CUDA");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "inputs must be fp16");
  TORCH_CHECK(_kernel.dtype() == torch::kInt, "qweight must be int32 (8x int4 packed)");
  TORCH_CHECK(_scales.dtype() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(_zeros.dtype() == torch::kInt, "qzeros must be int32 (8x int4 packed)");
  TORCH_CHECK(_in_feats.dim() == 2, "inputs must be [M, IC]");
  TORCH_CHECK(_kernel.dim() == 2, "qweight must be [OC, packed_i32]");
  TORCH_CHECK(_scales.dim() == 2, "scales must be [OC, groups]");
  TORCH_CHECK(_zeros.dim() == 2, "qzeros must be [OC, groups_packed]");
  TORCH_CHECK(group_size > 0, "group_size must be > 0");
  TORCH_CHECK(group_size % 8 == 0, "group_size must be a multiple of 8");

  auto inputs = _in_feats.contiguous();
  auto kernel = _kernel.contiguous();
  auto scales = _scales.contiguous();
  auto zeros = _zeros.contiguous();

  const int64_t M = inputs.size(0);
  const int64_t IC = inputs.size(1);
  const int64_t OC = kernel.size(0);
  const int groups = static_cast<int>((IC + group_size - 1) / group_size);
  const int packed_iters = static_cast<int>((IC + 7) / 8);
  const int weight_stride_i32 = static_cast<int>(kernel.size(1));
  const int zero_stride_i32 = static_cast<int>(zeros.size(1));
  const int expected_zero_stride = (groups + 7) / 8;

  TORCH_CHECK(scales.size(0) == OC, "scales OC mismatch");
  TORCH_CHECK(zeros.size(0) == OC, "qzeros OC mismatch");
  TORCH_CHECK(scales.size(1) >= groups, "scales groups mismatch");
  TORCH_CHECK(weight_stride_i32 >= packed_iters, "qweight packed cols too small");
  TORCH_CHECK(zero_stride_i32 >= expected_zero_stride, "qzeros packed cols too small");

  auto out = torch::empty({M, OC}, inputs.options().dtype(torch::kHalf));
  const half* in_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  const int32_t* w_ptr = reinterpret_cast<const int32_t*>(kernel.data_ptr<int>());
  const half* s_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const int32_t* z_ptr = reinterpret_cast<const int32_t*>(zeros.data_ptr<int>());
  half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  dim3 grid(static_cast<unsigned int>((OC + kTileOC - 1) / kTileOC),
            static_cast<unsigned int>(M));
  dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_w4a16_asym_i32_kernel<<<grid, block, 0, stream>>>(
      in_ptr,
      w_ptr,
      s_ptr,
      z_ptr,
      out_ptr,
      static_cast<int>(M),
      static_cast<int>(IC),
      static_cast<int>(OC),
      groups,
      group_size,
      weight_stride_i32,
      zero_stride_i32,
      packed_iters);
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

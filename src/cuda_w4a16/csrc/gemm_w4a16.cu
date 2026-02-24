#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cstdint>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;

__host__ __device__ inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

__device__ inline uint8_t unpack_u4_from_i32(uint32_t packed, int idx) {
  return static_cast<uint8_t>((packed >> (idx * 4)) & 0x0F);
}

__global__ void dequant_w4a16_asym_i32_kernel(
    const int32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const int32_t* __restrict__ qzeros,
    half* __restrict__ out_w,  // [rows, IC]
    int IC,
    int OC,
    int rows,
    int groups,
    int group_size,
    int weight_stride_i32,
    int zero_stride_i32,
    int row_start) {
  const int row_local = static_cast<int>(blockIdx.x);
  const int row = row_start + row_local;
  const int k = static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
  if (row_local >= rows || row >= OC || k >= IC) {
    return;
  }

  const int gid = k / group_size;
  if (gid >= groups) {
    return;
  }

  const uint32_t wpack = static_cast<uint32_t>(qweight[row * weight_stride_i32 + (k >> 3)]);
  const uint32_t zpack = static_cast<uint32_t>(qzeros[row * zero_stride_i32 + (gid >> 3)]);
  const float code = static_cast<float>(unpack_u4_from_i32(wpack, k & 7));
  const float z = static_cast<float>(unpack_u4_from_i32(zpack, gid & 7));
  const float s = __half2float(scales[row * groups + gid]);
  out_w[row_local * IC + k] = __float2half((code - z) * s);
}

static inline void check_w4a16_common(
    const torch::Tensor& inputs,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& qzeros,
    int group_size) {
  TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(qzeros.is_cuda(), "qzeros must be CUDA");
  TORCH_CHECK(inputs.dtype() == torch::kHalf, "inputs must be fp16");
  TORCH_CHECK(qweight.dtype() == torch::kInt, "qweight must be int32");
  TORCH_CHECK(scales.dtype() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(qzeros.dtype() == torch::kInt, "qzeros must be int32");
  TORCH_CHECK(inputs.dim() == 2, "inputs must be [M, IC]");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be [OC, packed_i32]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [OC, groups]");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be [OC, groups_packed]");
  TORCH_CHECK(group_size > 0, "group_size must be > 0");
  TORCH_CHECK(group_size % 8 == 0, "group_size must be a multiple of 8");
}

}  // namespace

torch::Tensor w4a16_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size,
    int split_k_iters) {
  (void)split_k_iters;
  check_w4a16_common(_in_feats, _kernel, _scales, _zeros, group_size);

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

  auto deq_w = torch::empty({OC, IC}, scales.options().dtype(torch::kHalf));
  const int32_t* w_ptr = reinterpret_cast<const int32_t*>(kernel.data_ptr<int>());
  const half* s_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const int32_t* z_ptr = reinterpret_cast<const int32_t*>(zeros.data_ptr<int>());
  half* deq_ptr = reinterpret_cast<half*>(deq_w.data_ptr<at::Half>());

  dim3 block(kThreads);
  dim3 grid(
      static_cast<unsigned int>(OC),
      static_cast<unsigned int>(ceil_div_int(static_cast<int>(IC), kThreads)));
  auto stream = at::cuda::getCurrentCUDAStream();
  dequant_w4a16_asym_i32_kernel<<<grid, block, 0, stream>>>(
      w_ptr,
      s_ptr,
      z_ptr,
      deq_ptr,
      static_cast<int>(IC),
      static_cast<int>(OC),
      static_cast<int>(OC),
      groups,
      group_size,
      weight_stride_i32,
      zero_stride_i32,
      0);
  AT_CUDA_CHECK(cudaGetLastError());

  auto out = at::matmul(inputs, deq_w.transpose(0, 1));
  TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == OC, "unexpected GEMM output shape");
  return out;
}

#include <torch/extension.h>
#include <cuda_fp16.h>


__global__ void gemv_rowmajor_tile8(
    const half* __restrict__ inputs,
    const int* __restrict__ qweight,
    const half* __restrict__ scales,
    const int* __restrict__ qzeros,
    half* __restrict__ outputs,
    int B, int IC, int OC, int G) {
  constexpr int TILE_OC = 16;
  int b = blockIdx.y;
  int oc0 = blockIdx.x * TILE_OC;
  if (b >= B || oc0 >= OC) return;

  int groups = (IC + G - 1) / G;
  float psum[TILE_OC];
  #pragma unroll
  for (int t=0;t<TILE_OC;++t) psum[t] = 0.0f;

  for (int pack = threadIdx.x; pack < (IC + 7) / 8; pack += blockDim.x) {
    int k0 = pack * 8;
    int gid0 = k0 / G;

    float in8[8];
    #pragma unroll
    for (int i=0;i<8;++i) {
      int k = k0 + i;
      in8[i] = (k < IC) ? __half2float(inputs[b * IC + k]) : 0.0f;
    }

    #pragma unroll
    for (int t=0; t<TILE_OC; ++t) {
      int oc = oc0 + t;
      if (oc >= OC) continue;
      float s = __half2float(scales[oc * groups + gid0]);
      int zpack = qzeros[oc * ((groups + 7) / 8) + gid0 / 8];
      float z = float((zpack >> ((gid0 % 8) * 4)) & 0xF);
      unsigned int w = reinterpret_cast<const unsigned int*>(qweight + oc * (IC / 8))[pack];
      float acc = 0.0f;
      #pragma unroll
      for (int i=0;i<8;++i) {
        float wq = float(w & 0xF);
        acc += (s * (wq - z)) * in8[i];
        w >>= 4;
      }
      psum[t] += acc;
    }
  }

  __shared__ float shm[256][TILE_OC];
  int tid = threadIdx.x;
  #pragma unroll
  for (int t=0;t<TILE_OC;++t) shm[tid][t] = psum[t];
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t=0;t<TILE_OC;++t) shm[tid][t] += shm[tid + stride][t];
    }
    __syncthreads();
  }
  if (tid == 0) {
    #pragma unroll
    for (int t=0;t<TILE_OC;++t) {
      int oc = oc0 + t;
      if (oc < OC) outputs[b * OC + oc] = __float2half(shm[0][t]);
    }
  }
}


torch::Tensor w4a16_gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size) {
  TORCH_CHECK(_in_feats.is_cuda() && _kernel.is_cuda() && _scales.is_cuda() && _zeros.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf && _scales.dtype() == torch::kHalf, "in_feats/scales must be fp16");
  TORCH_CHECK(_kernel.dtype() == torch::kInt && _zeros.dtype() == torch::kInt, "kernel/zeros must be int32");
  int B = _in_feats.size(0);
  int IC = _in_feats.size(1);
  int OC = _kernel.size(0);
  int groups = (_in_feats.size(1) + group_size - 1) / group_size;
  auto options = _in_feats.options();
  at::Tensor _out = torch::empty({B, OC}, options);
  const half* in_ptr = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
  const int* w_ptr = reinterpret_cast<const int*>(_kernel.data_ptr<int>());
  const half* s_ptr = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
  const int* z_ptr = reinterpret_cast<const int*>(_zeros.data_ptr<int>());
  half* out_ptr = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  dim3 grid((OC + 15)/16, B);
  int threads = 256;
  gemv_rowmajor_tile8<<<grid, threads>>>(in_ptr, w_ptr, s_ptr, z_ptr, out_ptr, B, IC, OC, group_size);
  return _out;
}


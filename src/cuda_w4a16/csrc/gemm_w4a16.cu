#include <torch/extension.h>
#include <cuda_fp16.h>

extern torch::Tensor w4a16_gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size);

torch::Tensor w4a16_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size,
    int  ) {
  return w4a16_gemv_forward_cuda(_in_feats, _kernel, _scales, _zeros, group_size);
}


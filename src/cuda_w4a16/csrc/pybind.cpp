#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor w4a16_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size,
    int split_k_iters);

torch::Tensor w4a16_gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm_forward_cuda", &w4a16_gemm_forward_cuda, "W4A16 GEMM kernel");
    m.def("w4a16_gemv_forward_cuda", &w4a16_gemv_forward_cuda, "W4A16 GEMV kernel");
}


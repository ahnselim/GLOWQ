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
    m.def(
        "w4a16_gemv_forward_cuda",
        &w4a16_gemv_forward_cuda,
        "Asymmetric W4A16 GEMV (CUDA, int32-packed int4)");
    m.def(
        "w4a16_gemm_forward_cuda",
        &w4a16_gemm_forward_cuda,
        "Asymmetric W4A16 GEMM fallback (CUDA dequant + matmul)");

    // Compatibility aliases for QM-style naming where useful.
    m.def(
        "gemv_w4a16_asym",
        &w4a16_gemv_forward_cuda,
        "Asymmetric W4A16 GEMV (CUDA, alias)");
}

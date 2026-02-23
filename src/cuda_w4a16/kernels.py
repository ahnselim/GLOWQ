import os
import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "w4a16_kernels"
_ext = None


def load_w4a16_extension(verbose: bool = False):
    global _ext
    if _ext is not None:
        return _ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(this_dir, "csrc")

    sources = [
        os.path.join(csrc_dir, "pybind.cpp"),
        os.path.join(csrc_dir, "gemm_w4a16.cu"),
        os.path.join(csrc_dir, "gemv_w4a16.cu"),
    ]

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-Xptxas",
        "-O3,verbose" if verbose else "-O3",
    ]

    _ext = load(
        name=_EXT_NAME,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    return _ext


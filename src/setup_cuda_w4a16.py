import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = Path(__file__).resolve().parent
CSRC_DIR = THIS_DIR / "cuda_w4a16" / "csrc"


def _nvcc_flags():
    return [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-Xptxas",
        "-O3",
    ]


setup(
    name="glowq-cuda-w4a16-kernels",
    version="0.0.1",
    description="Prebuilt CUDA extension for GlowQ W4A16 kernels",
    ext_modules=[
        CUDAExtension(
            name="cuda_w4a16.w4a16_kernels",
            sources=[
                str(CSRC_DIR / "pybind.cpp"),
                str(CSRC_DIR / "gemm_w4a16.cu"),
                str(CSRC_DIR / "gemv_w4a16.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": _nvcc_flags(),
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)

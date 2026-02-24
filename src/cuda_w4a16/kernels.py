import importlib
import os
from typing import Optional

_EXT_NAME = "w4a16_kernels"
_ext = None


def _want_verbose(user_flag: Optional[bool] = None) -> bool:
    if user_flag:
        return True
    env = os.environ.get("W4A16_VERBOSE", os.environ.get("CUDA_MIXED_VERBOSE", ""))
    return env.lower() in {"1", "true", "yes", "y", "on"}


def load_w4a16_extension(verbose: Optional[bool] = None):
    global _ext
    if _ext is not None:
        return _ext
    try:
        # Default path: prebuilt extension installed via build_ext --inplace.
        _ext = importlib.import_module(".w4a16_kernels", package=__package__)
        return _ext
    except Exception as import_exc:
        allow_jit = os.environ.get("W4A16_ALLOW_JIT", "").lower() in {"1", "true", "yes", "y", "on"}
        if not allow_jit:
            raise RuntimeError(
                "Failed to import prebuilt cuda_w4a16 kernel extension. "
                "Build it first with:\n"
                "  cd GlowQ/src && python setup_cuda_w4a16.py build_ext --inplace\n"
                "If you intentionally want runtime JIT fallback, set W4A16_ALLOW_JIT=1."
            ) from import_exc

        from torch.utils.cpp_extension import load

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
            "-O3",
        ]
        _ext = load(
            name=_EXT_NAME,
            sources=sources,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=_want_verbose(verbose),
        )
        return _ext


__all__ = ["load_w4a16_extension"]

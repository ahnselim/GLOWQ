import math
import os
from typing import Optional

import torch
import torch.nn as nn

from .kernels import load_w4a16_extension


def _env_bool(*names: str, default: bool = False) -> bool:
    for name in names:
        val = os.environ.get(name)
        if val is None:
            continue
        return val.lower() in {"1", "true", "yes", "y", "on"}
    return default


def _env_int(*names: str, default: int) -> int:
    for name in names:
        val = os.environ.get(name)
        if val is None or val == "":
            continue
        return int(val)
    return int(default)


def _get_kernel_out_dtype() -> torch.dtype:
    # Match QM baseline behavior but keep W4A16-specific override compatibility.
    val = os.environ.get(
        "W4A16_KERNEL_OUT_DTYPE",
        os.environ.get("MIXED_KERNEL_OUT_DTYPE", "fp32"),
    ).lower()
    if val in ("fp16", "float16", "half"):
        return torch.float16
    if val in ("fp32", "float32"):
        return torch.float32
    return torch.float32


def _pack_int4_to_int32(u4: torch.Tensor) -> torch.Tensor:
    assert u4.dtype == torch.uint8
    shifts = torch.arange(8, device=u4.device, dtype=torch.int32) * 4
    u4_i32 = (u4.to(torch.int32) << shifts).sum(dim=-1)
    return u4_i32.to(torch.int32)


def _unpack_int4_i32(packed_i32: torch.Tensor, total_vals: int) -> torch.Tensor:
    packed_i32 = packed_i32.contiguous()
    rows = packed_i32.size(0)
    shifts = (torch.arange(8, device=packed_i32.device, dtype=torch.int32) * 4).view(1, 1, 8)
    vals = ((packed_i32.to(torch.int32).unsqueeze(-1) >> shifts) & 0x0F).to(torch.uint8)
    return vals.reshape(rows, -1)[:, :total_vals].contiguous()


def _quantize_per_group_w4(weight: torch.Tensor, group_size: int):
    oc, ic = weight.shape
    if ic % group_size != 0:
        pad = group_size - (ic % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        ic = weight.shape[1]

    groups = ic // group_size
    wg = weight.reshape(oc, groups, group_size)
    w_min = wg.min(dim=-1).values
    w_max = wg.max(dim=-1).values
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zeros_fp = (-w_min / scales).round().clamp(0, 15)

    q = torch.round(wg / scales.unsqueeze(-1) + zeros_fp.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)

    if group_size % 8 != 0:
        raise ValueError("group_size must be multiple of 8 for int32 int4 packing")
    q8 = q.view(oc, groups, group_size // 8, 8)
    q_packed = _pack_int4_to_int32(q8)

    zeros_u8 = zeros_fp.to(torch.uint8)
    groups_packed = (groups + 7) // 8
    pad_groups = groups_packed * 8 - groups
    zeros_base = torch.nn.functional.pad(zeros_u8, (0, pad_groups), value=8)
    zeros_u8_8 = zeros_base.view(oc, groups_packed, 8)
    zeros_packed = _pack_int4_to_int32(zeros_u8_8)

    kernel_int32 = q_packed.reshape(oc, ic // 8).to(torch.int32)
    zeros_int32 = zeros_packed.to(torch.int32)
    scales_f16 = scales.to(torch.float16)
    return kernel_int32.contiguous(), zeros_int32.contiguous(), scales_f16.contiguous(), ic


class CudaW4A16Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        if self.group_size <= 0 or (self.group_size % 8) != 0:
            raise ValueError("group_size must be > 0 and a multiple of 8 for GlowQ int32 int4 packing")

        groups = math.ceil(self.in_features / self.group_size)
        groups_packed = (groups + 7) // 8
        weight_cols = (self.in_features + 7) // 8

        self.register_buffer("qweight_i32", torch.empty(self.out_features, weight_cols, dtype=torch.int32))
        self.register_buffer("qzeros_i32", torch.empty(self.out_features, max(1, groups_packed), dtype=torch.int32))
        self.register_buffer("scales_f16", torch.empty(self.out_features, groups, dtype=torch.float16))

        gid = torch.arange(self.in_features, dtype=torch.long)
        gid = torch.div(gid, self.group_size, rounding_mode="floor")
        self.register_buffer("gid", gid, persistent=False)
        self.groups = groups
        self.groups_packed = groups_packed

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._ext = None
        self._gemv_threshold = _env_int("W4A16_GEMV_M_MAX", "CUDA_MIXED_GEMV_M_MAX", default=128)
        self._w_cache: Optional[torch.Tensor] = None
        self._w_cache_failed = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )

    def _get_ext(self):
        if self._ext is None:
            self._ext = load_w4a16_extension()
        return self._ext

    def _run_cuda_gemv(self, x2d: torch.Tensor) -> torch.Tensor:
        return self._get_ext().w4a16_gemv_forward_cuda(
            x2d,
            self.qweight_i32,
            self.scales_f16,
            self.qzeros_i32,
            int(self.group_size),
        )

    def _run_cuda_gemm(self, x2d: torch.Tensor) -> torch.Tensor:
        split_k_iters = _env_int("W4A16_SPLIT_K_ITERS", default=1)
        return self._get_ext().w4a16_gemm_forward_cuda(
            x2d,
            self.qweight_i32,
            self.scales_f16,
            self.qzeros_i32,
            int(self.group_size),
            int(split_k_iters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            m = orig_shape[0] * orig_shape[1]
            k = orig_shape[2]
            x2d = x.reshape(m, k)
        else:
            x2d = x.reshape(-1, x.shape[-1])
            m, k = x2d.shape

        if k != self.in_features:
            raise ValueError(f"Input dim mismatch: got {k}, expected {self.in_features}")

        x2d = x2d.contiguous()
        if x2d.dtype != torch.float16:
            x2d = x2d.to(torch.float16)

        force_gemm = _env_bool("W4A16_FORCE_GEMM", "CUDA_MIXED_FORCE_GEMM", default=False)
        force_gemv = _env_bool("W4A16_FORCE_GEMV", "CUDA_MIXED_FORCE_GEMV", default=False)
        legacy_use_gemv = _env_bool("W4A16_USE_GEMV", default=False)
        prefer_gemv = (not force_gemm) and (force_gemv or legacy_use_gemv or m <= self._gemv_threshold)

        out_dtype = _get_kernel_out_dtype()
        if prefer_gemv:
            out = self._run_cuda_gemv(x2d)
        else:
            out = self._matmul_with_dequant(x2d, out_dtype)

        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _matmul_with_dequant(self, x2d: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        use_cuda_gemm = x2d.is_cuda and _env_bool(
            "W4A16_GEMM_CUDA",
            "CUDA_MIXED_DEQUANT_CUDA",
            default=True,
        )
        use_cache = _env_bool(
            "W4A16_DEQUANT_CACHE",
            "CUDA_MIXED_DEQUANT_CACHE",
            default=False,
        )
        cache_m_min = _env_int(
            "W4A16_DEQUANT_CACHE_M_MIN",
            "CUDA_MIXED_DEQUANT_CACHE_M_MIN",
            default=256,
        )

        if use_cuda_gemm and not use_cache:
            out = self._run_cuda_gemm(x2d)
            if out.dtype != out_dtype:
                out = out.to(out_dtype)
            return out

        if use_cache and (not self._w_cache_failed) and x2d.size(0) >= cache_m_min:
            w = self._w_cache
            if (w is None) or (not torch.is_tensor(w)) or (w.device != x2d.device):
                try:
                    w = self._dequant_full_weight_fallback()
                    self._w_cache = w
                except RuntimeError as e:
                    self._w_cache_failed = True
                    self._w_cache = None
                    if "out of memory" not in str(e).lower():
                        raise
            if w is not None:
                tmp = torch.matmul(x2d, w.t())
                if tmp.dtype != out_dtype:
                    tmp = tmp.to(out_dtype)
                return tmp

        chunk = _env_int("W4A16_DEQUANT_CHUNK", "CUDA_MIXED_DEQUANT_CHUNK", default=512)
        out = torch.empty((x2d.size(0), self.out_features), device=x2d.device, dtype=out_dtype)
        for start in range(0, self.out_features, chunk):
            end = min(start + chunk, self.out_features)
            w_chunk = self._dequant_chunk(start, end)
            tmp = torch.matmul(x2d, w_chunk.t())
            if tmp.dtype != out_dtype:
                tmp = tmp.to(out_dtype)
            out[:, start:end] = tmp
        return out

    def _dequant_full_weight_fallback(self) -> torch.Tensor:
        chunks = []
        chunk = _env_int("W4A16_DEQUANT_CHUNK", "CUDA_MIXED_DEQUANT_CHUNK", default=512)
        for start in range(0, self.out_features, chunk):
            end = min(start + chunk, self.out_features)
            chunks.append(self._dequant_chunk(start, end))
        return torch.cat(chunks, dim=0).contiguous()

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        qvals = _unpack_int4_i32(self.qweight_i32[start:end, :], self.in_features).to(torch.float16)
        zvals = _unpack_int4_i32(self.qzeros_i32[start:end, : self.groups_packed], self.groups).to(torch.float16)
        zeros_k = zvals.index_select(1, self.gid)
        scales_k = self.scales_f16[start:end, :].to(torch.float16).index_select(1, self.gid)
        return (qvals - zeros_k) * scales_k

    @classmethod
    def from_float(cls, linear_layer: nn.Linear, group_size: int):
        dev = linear_layer.weight.device
        mod = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            group_size,
            bias=(linear_layer.bias is not None),
        ).to(dev)

        weight = linear_layer.weight.data.to(torch.float16)
        kernel_i32, zeros_i32, scales_f16, _ = _quantize_per_group_w4(weight, group_size)

        mod.qweight_i32.resize_(kernel_i32.shape)
        mod.qweight_i32.copy_(kernel_i32)
        mod.qzeros_i32.resize_(zeros_i32.shape)
        mod.qzeros_i32.copy_(zeros_i32)
        mod.scales_f16.resize_(scales_f16.shape)
        mod.scales_f16.copy_(scales_f16)
        if linear_layer.bias is not None:
            mod.bias.data.copy_(linear_layer.bias.data.to(torch.float16))
        return mod


def convert_to_cuda_w4a16(model: nn.Module, group_size: int = 128) -> nn.Module:
    skip_keywords = ["lm_head"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(kw in name for kw in skip_keywords):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                new_mod = CudaW4A16Linear.from_float(getattr(parent, parts[-1]), group_size)
                setattr(parent, parts[-1], new_mod)
            except Exception:
                continue
    torch.cuda.empty_cache()
    return model

import math
import os
import torch
import torch.nn as nn
from .kernels import load_w4a16_extension


def _pack_int4_to_int32(u4: torch.Tensor) -> torch.Tensor:

    assert u4.dtype == torch.uint8
    shifts = torch.arange(8, device=u4.device, dtype=torch.int32) * 4
    u4_i32 = (u4.to(torch.int32) << shifts).sum(dim=-1)
    return u4_i32.to(torch.int32)


def _make_divisible(c: int, divisor: int) -> int:
    return (c + divisor - 1) // divisor


def _quantize_per_group_w4(weight: torch.Tensor, group_size: int):

    OC, IC = weight.shape
    if IC % group_size != 0:
        pad = group_size - (IC % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        IC = weight.shape[1]

    groups = IC // group_size
    Wg = weight.reshape(OC, groups, group_size)
    w_min = Wg.min(dim=-1).values
    w_max = Wg.max(dim=-1).values
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zeros_fp = (-w_min / scales).round().clamp(0, 15)

    q = torch.round(Wg / scales.unsqueeze(-1) + zeros_fp.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)

    if group_size % 8 != 0:
        raise ValueError("group_size must be multiple of 8 for packing")
    q8 = q.view(OC, groups, group_size // 8, 8)
    q_packed = _pack_int4_to_int32(q8)

    zeros_u8 = zeros_fp.to(torch.uint8)
    groups_packed = (groups + 7) // 8
    pad_groups = groups_packed * 8 - groups
    zeros_base = torch.nn.functional.pad(zeros_u8, (0, pad_groups), value=8)
    zeros_u8_8 = zeros_base.view(OC, groups_packed, 8)
    zeros_packed = _pack_int4_to_int32(zeros_u8_8)

    kernel_int32 = q_packed.reshape(OC, IC // 8).to(torch.int32)
    zeros_int32 = zeros_packed.to(torch.int32)
    scales_f16 = scales.to(torch.float16)
    return (
        kernel_int32.contiguous(),
        zeros_int32.contiguous(),
        scales_f16.contiguous(),
        IC,
    )


class CudaW4A16Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer("qweight_i32", torch.empty(out_features, (in_features + 7) // 8, dtype=torch.int32))
        self.register_buffer("qzeros_i32", torch.empty(out_features, max(1, (math.ceil(in_features / group_size) + 7) // 8), dtype=torch.int32))
        self.register_buffer("scales_f16", torch.empty(out_features, math.ceil(in_features / group_size), dtype=torch.float16))

        groups = math.ceil(in_features / group_size)
        gid = torch.arange(in_features, dtype=torch.int32).div(group_size, rounding_mode='floor')
        self.register_buffer("gid", gid)
        self.groups = groups
        self.groups_packed = (groups + 7) // 8

        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16)) if bias else None

        self._ext = None
        self._tuned = False
        self._prefer_gemv = None
        self._best_chunk = 512

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}, bias={self.bias is not None}"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ext is None:
            self._ext = load_w4a16_extension()

        orig_shape = x.shape
        if x.dim() == 3:
            M, K = orig_shape[0] * orig_shape[1], orig_shape[2]
            x2d = x.reshape(M, K)
        else:
            x2d = x
            M, K = x2d.shape

        if (not self._tuned) and bool(int(os.environ.get("W4A16_AUTOTUNE", "0"))) and x2d.is_cuda:
            self._autotune(x2d)
        use_gemv_env = bool(int(os.environ.get("W4A16_USE_GEMV", "0")))
        gemv_m_max = int(os.environ.get("W4A16_GEMV_M_MAX", "128"))
        prefer_gemv = self._prefer_gemv if self._prefer_gemv is not None else (M <= gemv_m_max)
        if (use_gemv_env and M <= max(1, gemv_m_max)) or prefer_gemv:
            out2d = self._ext.w4a16_gemv_forward_cuda(
                x2d, self.qweight_i32, self.scales_f16, self.qzeros_i32, int(self.group_size)
            )
        else:
            device = x2d.device
            dtype = x2d.dtype
            OC = self.out_features
            IC = self.in_features
            groups = self.groups
            groups_packed = self.groups_packed
            gid = self.gid.to(device=device)
            out2d = torch.empty((M, OC), device=device, dtype=dtype)
            chunk = int(os.environ.get("W4A16_DEQUANT_CHUNK", str(self._best_chunk)))
            for start in range(0, OC, chunk):
                end = min(start + chunk, OC)
                rows = end - start
                packed = self.qweight_i32[start:end, :]
                q_u8_parts = [(packed >> (4 * i)) & 0xF for i in range(8)]


                q_u8_full = torch.stack(q_u8_parts, dim=-1).reshape(rows, -1)
                q_u8 = q_u8_full[:, :IC].to(torch.float16)
                zp = self.qzeros_i32[start:end, :groups_packed]
                zp_parts = [(zp >> (4 * i)) & 0xF for i in range(8)]


                zp_u8_full = torch.stack(zp_parts, dim=-1).reshape(rows, groups_packed * 8)
                zp_unpack = zp_u8_full[:, :groups].to(torch.float16)
                zeros_k = zp_unpack.index_select(1, gid.to(torch.long))
                scales_k = self.scales_f16[start:end, :].to(torch.float16).index_select(1, gid.to(torch.long))
                w_chunk = (q_u8 - zeros_k) * scales_k
                out2d[:, start:end] = x2d.matmul(w_chunk.t())
        if self.bias is not None:
            out2d = out2d + self.bias.unsqueeze(0)
        return out2d.reshape(*orig_shape[:-1], self.out_features)

    @torch.no_grad()
    def _autotune(self, x2d: torch.Tensor):
        device = x2d.device
        if device.type != 'cuda':
            self._tuned = True
            return
        torch.cuda.synchronize(device)
        times = {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = load_w4a16_extension().w4a16_gemv_forward_cuda(x2d, self.qweight_i32, self.scales_f16, self.qzeros_i32, int(self.group_size))
        end.record()
        torch.cuda.synchronize(device)
        t_gemv = start.elapsed_time(end)
        times[("gemv", 0)] = t_gemv
        for chunk in (256, 512, 768):
            OC = self.out_features
            IC = self.in_features
            groups = self.groups
            groups_packed = self.groups_packed
            gid = self.gid.to(device=device)
            out = torch.empty((x2d.shape[0], OC), device=device, dtype=x2d.dtype)
            torch.cuda.synchronize(device)
            start.record()
            for s in range(0, OC, chunk):
                e = min(s + chunk, OC)
                packed = self.qweight_i32[s:e, :]
                q_u8_full = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(e - s, -1)
                q_u8 = q_u8_full[:, :IC].to(torch.float16)
                zp = self.qzeros_i32[s:e, :groups_packed]
                zp_u8_full = torch.stack([(zp >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(e - s, groups_packed * 8)
                zp_unpack = zp_u8_full[:, :groups].to(torch.float16)
                zeros_k = zp_unpack.index_select(1, gid.to(torch.long))
                scales_k = self.scales_f16[s:e, :].to(torch.float16).index_select(1, gid.to(torch.long))
                w_chunk = (q_u8 - zeros_k) * scales_k
                out[:, s:e] = x2d.matmul(w_chunk.t())
            end.record()
            torch.cuda.synchronize(device)
            times[("gemm", chunk)] = start.elapsed_time(end)
        best_mode, best_param = min(times.items(), key=lambda kv: kv[1])[0]
        if best_mode == "gemv":
            self._prefer_gemv = True
        else:
            self._prefer_gemv = False
            self._best_chunk = best_param
        self._tuned = True

    @classmethod
    def from_float(cls, linear_layer: nn.Linear, group_size: int):
        dev = linear_layer.weight.device
        dtype = linear_layer.weight.dtype
        mod = cls(linear_layer.in_features, linear_layer.out_features, group_size, bias=(linear_layer.bias is not None)).to(dev)

        W = linear_layer.weight.data.to(torch.float16)
        kernel_i32, zeros_i32, scales_f16, IC_padded = _quantize_per_group_w4(W, group_size)
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
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                new_mod = CudaW4A16Linear.from_float(getattr(parent, parts[-1]), group_size)
                setattr(parent, parts[-1], new_mod)
            except Exception:

                continue
    torch.cuda.empty_cache()
    return model

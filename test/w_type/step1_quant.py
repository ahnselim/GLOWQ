"""
w_type/step1_quant.py
Computes Step 1 quantization errors for multiple weight formats (INT, MXFP, NVFP, and block-FP variants).
output :
(user-specified output paths)
|-- <out_original_weights>    (.pt)
|-- <out_quant_err>           (.pt)
`-- <out_quantized_weights>   (.pt, optional)
"""

import os
import gc
import re
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer




_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False





@torch.no_grad()
def fake_quantize_asymmetric_anybit(weight: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:

    assert bits in (2, 3, 4), "bits must be 2, 3, or 4"
    qmax = (1 << bits) - 1
    O, I = weight.shape
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    pad = (group_size - (I % group_size)) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))
    Wf = weight.to(torch.float32)
    groups = Wf.reshape(O, -1, group_size)

    min_vals = groups.min(dim=-1).values
    max_vals = groups.max(dim=-1).values
    scales = torch.clamp((max_vals - min_vals) / qmax, min=1e-8)
    zeros = torch.round(-min_vals / scales).clamp(0, qmax)

    q = torch.round(groups / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).clamp(0, qmax)
    deq = (q - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)

    Wq = deq.reshape(O, -1)
    if pad:
        Wq = Wq[:, :I]
    return Wq.to(weight.dtype)






@torch.no_grad()
def quantize_fp_em(x: torch.Tensor, exp_bits: int, man_bits: int, bias: int) -> torch.Tensor:

    x32 = x.to(torch.float32)
    sign = torch.sign(x32)
    a = x32.abs()


    mant_pow = float(2 ** man_bits)
    mant_max = float(2 ** man_bits - 1)



    min_subnorm = float(2.0 ** (1 - bias) * (1.0 / mant_pow))

    min_normal = float(2.0 ** (1 - bias))

    emax_real = float((2 ** exp_bits - 1) - bias)
    max_normal = float(2.0 ** emax_real * (1.0 + mant_max / mant_pow))

    out_mag = torch.zeros_like(a)


    finite_mask = torch.isfinite(a)
    a = torch.where(finite_mask, a, torch.zeros_like(a))


    a_clamped = torch.clamp(a, max=max_normal)


    mask_sub = (a_clamped > 0) & (a_clamped < min_normal)
    if mask_sub.any():
        a_sub = a_clamped[mask_sub]
        val_unscaled = a_sub / float(2.0 ** (1 - bias))
        M_real = val_unscaled * mant_pow
        M = torch.round(M_real).clamp(1.0, mant_max)
        sub_val = float(2.0 ** (1 - bias)) * (M / mant_pow)
        out_mag[mask_sub] = sub_val


    mask_norm = a_clamped >= min_normal
    if mask_norm.any():
        a_norm = a_clamped[mask_norm]

        e_real = torch.floor(torch.log2(a_norm))
        e_min_real = float(1 - bias)
        e_max_real = emax_real
        e_real = e_real.clamp(e_min_real, e_max_real)
        scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=a_norm.device), e_real)
        t = a_norm / scale
        frac_real = (t - 1.0) * mant_pow
        M = torch.round(frac_real).clamp(0.0, mant_max)
        norm_val = scale * (1.0 + M / mant_pow)
        out_mag[mask_norm] = norm_val


    out = out_mag * sign
    return out.to(x.dtype)





@torch.no_grad()
def fake_quantize_mx_block(weight: torch.Tensor, mode: str, group_size: int) -> torch.Tensor:

    mode = mode.lower()
    if mode == "mxfp4":

        e_bits, m_bits, bias, emax_elem = 2, 1, 1, 2
    elif mode == "mxfp6":



        e_bits, m_bits, bias, emax_elem = 2, 3, 1, 2
    else:
        raise ValueError(f"fake_quantize_mx_block: unknown mode {mode}")

    O, I = weight.shape
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    pad = (group_size - (I % group_size)) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))
        I_pad = I + pad
    else:
        I_pad = I

    Wf = weight.to(torch.float32)
    groups = Wf.reshape(O, -1, group_size)


    amax = groups.abs().amax(dim=-1)
    eps = 1e-30
    amax_safe = torch.where(amax == 0, torch.full_like(amax, eps), amax)


    shared_exp = torch.floor(torch.log2(amax_safe)) - float(emax_elem)
    shared_exp = shared_exp.clamp(-126.0, 127.0)
    X = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=Wf.device),
        shared_exp
    )


    y = groups / X.unsqueeze(-1)
    y_q = quantize_fp_em(y, exp_bits=e_bits, man_bits=m_bits, bias=bias)


    y_q = torch.where(amax.unsqueeze(-1) == 0, torch.zeros_like(y_q), y_q)

    deq = y_q * X.unsqueeze(-1)
    Wq = deq.reshape(O, I_pad)
    if pad:
        Wq = Wq[:, :I]

    return Wq.to(weight.dtype)







@torch.no_grad()
def fake_quantize_nvfp4_block(weight: torch.Tensor, group_size: int) -> torch.Tensor:

    O, I = weight.shape
    if group_size <= 0:
        raise ValueError("group_size must be positive")


    pad = (group_size - (I % group_size)) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))
        I_pad = I + pad
    else:
        I_pad = I

    Wf = weight.to(torch.float32)
    groups = Wf.reshape(O, -1, group_size)



    amax_global = groups.abs().amax()
    eps = 1e-30
    if amax_global <= eps:

        return torch.zeros_like(weight)

    FP4_MAX = 6.0
    FP8_MAX = 448.0
    gamma = (FP4_MAX * FP8_MAX) / amax_global


    amax_block = groups.abs().amax(dim=-1, keepdim=True)
    amax_safe = torch.where(amax_block == 0,
                            torch.full_like(amax_block, eps),
                            amax_block)

    d_real = amax_safe / FP4_MAX


    d_scaled = d_real * gamma
    d_scaled_q = quantize_fp_em(
        d_scaled,
        exp_bits=4, man_bits=3, bias=7
    )


    d_hat = d_scaled_q / gamma


    s_enc = 1.0 / d_hat.clamp_min(1e-30)



    y = groups * s_enc

    y_q = quantize_fp_em(
        y,
        exp_bits=2, man_bits=1, bias=1
    )


    zero_block = (amax_block == 0)
    y_q = torch.where(zero_block, torch.zeros_like(y_q), y_q)


    v_q = y_q / s_enc

    Wq = v_q.reshape(O, I_pad)
    if pad:
        Wq = Wq[:, :I]

    return Wq.to(weight.dtype)






@torch.no_grad()
def fake_quantize_blockfloat(weight: torch.Tensor, mode: str, group_size: int) -> torch.Tensor:

    mode_l = mode.lower()
    if mode_l in ("mxfp4", "mxfp6"):
        return fake_quantize_mx_block(weight, mode_l, group_size)
    elif mode_l == "nvfp4":
        return fake_quantize_nvfp4_block(weight, group_size)
    else:
        raise ValueError(f"Unknown block-fp mode: {mode}")





if _HAS_TRITON:
    @triton.jit
    def _quant_linear_kernel(
        x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, bias_ptr, output_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_qwm, stride_qwk, stride_qzm, stride_qzk,
        stride_sm, stride_sk, stride_om, stride_on,
        group_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        HAS_ZEROS: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N)
        pid_m, pid_n = pid // num_pid_n, pid % num_pid_n

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        qweight_ptrs = qweight_ptr + (offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 2) * stride_qwk)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_start = k * BLOCK_SIZE_K
            k_offs = k_start + offs_k

            x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
            packed = tl.load(qweight_ptrs, mask=q_mask, other=0)

            is_low = k_offs % 2 == 0
            nibbles = tl.where(is_low[:, None], packed & 0x0F, packed >> 4)

            group_id = k_offs // group_size
            scales_ptrs = scales_ptr + (offs_bn[None, :] * stride_sm + group_id[:, None] * stride_sk)
            scales = tl.load(scales_ptrs, mask=q_mask, other=0.0)

            if HAS_ZEROS:
                zeros_gid = group_id // 2
                qzeros_ptrs = qzeros_ptr + (offs_bn[None, :] * stride_qzm + zeros_gid[:, None] * stride_qzk)
                packed_zeros = tl.load(qzeros_ptrs, mask=q_mask, other=0)
                is_low_zero = group_id % 2 == 0
                zeros = tl.where(is_low_zero[:, None], packed_zeros & 0x0F, packed_zeros >> 4)
            else:
                zeros = 8

            deq = (nibbles.to(tl.float32) - zeros.to(tl.float32)) * scales.to(x.dtype)
            acc += tl.dot(x, deq)

            x_ptrs += BLOCK_SIZE_K * stride_xk
            qweight_ptrs += (BLOCK_SIZE_K // 2) * stride_qwk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
            acc = acc + bias[None, :]

        c = acc.to(output_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(out_ptrs, c, mask=c_mask)

    def _quant_linear(x, qweight, qzeros, scales, bias, group_size):
        orig_shape = x.shape
        if x.dim() == 3:
            M, K = orig_shape[0] * orig_shape[1], orig_shape[2]
        else:
            M, K = x.shape
        N = scales.shape[0]
        x = x.reshape(M, K)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        stride_qz0, stride_qz1 = ((qzeros.stride(0), qzeros.stride(1)) if qzeros is not None else (0, 0))

        _quant_linear_kernel[grid](
            x, qweight, qzeros, scales, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            stride_qz0, stride_qz1,
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            group_size=group_size,
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
            HAS_ZEROS=(qzeros is not None), HAS_BIAS=(bias is not None),
            num_warps=4, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)

    class TritonTrue4BitLinear(nn.Module):
        def __init__(self, in_features, out_features, group_size=128, bias=False):
            super().__init__()
            self.in_features, self.out_features, self.group_size = in_features, out_features, group_size
            self.register_buffer("qweight", torch.empty((out_features, in_features // 2), dtype=torch.uint8))
            self.register_buffer("qzeros",  torch.empty((out_features, math.ceil(in_features / group_size) // 2), dtype=torch.uint8))
            self.register_buffer("scales",  torch.empty((out_features, math.ceil(in_features / group_size)), dtype=torch.float16))
            self.bias = (nn.Parameter(torch.empty(out_features, dtype=torch.float16)) if bias else None)

        def forward(self, x):
            return _quant_linear(x, self.qweight, self.qzeros, self.scales, self.bias, self.group_size)

        @classmethod
        def from_float(cls, linear_layer: nn.Linear, group_size: int):
            qlayer = cls(linear_layer.in_features, linear_layer.out_features, group_size, linear_layer.bias is not None
                        ).to(linear_layer.weight.device, dtype=linear_layer.weight.dtype)
            W = linear_layer.weight.data.clone()
            O, I = W.shape
            if I % group_size != 0:
                W = F.pad(W, (0, group_size - (I % group_size)))
            I_pad = W.shape[1]
            Wg = W.reshape(O, I_pad // group_size, group_size)
            min_vals = Wg.min(dim=-1).values
            max_vals = Wg.max(dim=-1).values
            scales = ((max_vals - min_vals) / 15.0).clamp(min=1e-8)
            zeros_f = (-min_vals / scales).round()

            qvals = torch.round(Wg / scales.unsqueeze(-1) + zeros_f.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)
            low_w, high_w = qvals[:, :, 0::2], qvals[:, :, 1::2]
            packed_w = (high_w << 4) | low_w
            qlayer.qweight.data.copy_(packed_w.reshape(O, I_pad // 2))

            qlayer.scales.data.copy_(scales.to(torch.float16))
            zeros_u8 = zeros_f.to(torch.uint8)
            low_z, high_z = zeros_u8[:, 0::2], zeros_u8[:, 1::2]
            packed_z = (high_z << 4) | low_z
            qlayer.qzeros.data.copy_(packed_z)

            if linear_layer.bias is not None:
                qlayer.bias.data.copy_(linear_layer.bias.data)
            return qlayer

    @torch.no_grad()
    def triton_dequantize_w(weight: torch.Tensor, group_size: int) -> torch.Tensor:
        in_f, out_f = weight.shape[1], weight.shape[0]
        tmp = nn.Linear(in_f, out_f, bias=False, device=weight.device, dtype=torch.float16)
        tmp.weight.data = weight
        qlayer = TritonTrue4BitLinear.from_float(tmp, group_size=group_size)
        qweight, scales, qzeros = qlayer.qweight, qlayer.scales, qlayer.qzeros
        O, I_packed = qweight.shape
        I_pad = I_packed * 2

        low_w = qweight & 0x0F
        high_w = qweight >> 4
        unpack_w = torch.stack((low_w, high_w), dim=-1).view(O, I_pad)

        low_z = qzeros & 0x0F
        high_z = qzeros >> 4
        unpack_z = torch.stack((low_z, high_z), dim=-1).view(O, -1)

        Wg = unpack_w.reshape(O, I_pad // group_size, group_size)
        deq_g = (Wg.to(scales.dtype) - unpack_z.unsqueeze(-1)) * scales.unsqueeze(-1)
        Wq = deq_g.reshape(O, I_pad)[:, :in_f].contiguous()
        del tmp, qlayer
        gc.collect()
        if weight.is_cuda:
            torch.cuda.empty_cache()
        return Wq.to(torch.float16)





TARGET_KEYWORDS = [

    "q_proj", "k_proj", "v_proj", "o_proj",

    "gate_proj", "up_proj", "down_proj",

    "fc1", "fc2", "w1", "w2", "w3",
]


def is_target_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    if not (name.endswith(".weight") and tensor.ndim == 2 and "layers" in name):
        return False
    return any(kw in name for kw in TARGET_KEYWORDS)


def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)





def main():
    ap = argparse.ArgumentParser(
        description="Step 1 — Compute quantization error (INT2/3/4 + MXFP4/NVFP4/MXFP6, block-FP; MX+/NVFP4 paper-style)."
    )
    ap.add_argument("--model_name", type=str, required=True,
                    help="HF model id (e.g., mistralai/Mistral-7B-v0.3)")
    ap.add_argument("--quant_dtype",
                    type=str,
                    choices=["int2", "int3", "int4", "mxfp4", "nvfp4", "mxfp6"],
                    default="int4",
                    help="Quantization datatype/precision.")
    ap.add_argument("--bits", type=int, choices=[2, 3, 4], default=None,
                    help="(Compatibility) If provided, force quant_dtype to int{bits}.")
    ap.add_argument("--group_size", type=int, default=128,
                    help="Per-group size (used for both INT-family and block-FP formats; "
                         "MXFP4/MXFP6 typically use 32, and 16 is recommended for NVFP4).")
    ap.add_argument("--device", type=str, default="cuda", help="Compute device for per-layer ops (cuda|cpu)")
    ap.add_argument("--trust_remote_code", action="store_true", help="Required for some Qwen/others")
    ap.add_argument("--out_quant_err", required=True, help="Path to save error dict (.pt)")
    ap.add_argument("--out_original_weights", required=True, help="Path to save original state_dict (.pt)")
    ap.add_argument("--out_quantized_weights", default=None, help="(Optional) Path to save Wq dict (.pt)")
    ap.add_argument("--w4_backend", choices=["fake", "triton"], default="fake",
                    help="When quant_dtype=int4, choose backend: fake(default) or triton (if installed).")
    args = ap.parse_args()


    qdtype = args.quant_dtype.lower()
    if args.bits is not None:
        if qdtype.startswith("int") and qdtype != f"int{args.bits}":
            print(f"[WARN] --bits={args.bits} overrides --quant_dtype={qdtype}. Using int{args.bits}.")
        qdtype = f"int{args.bits}"


    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Load] tokenizer: {args.model_name}")
    _ = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)

    print(f"[Load] model (FP16, CPU): {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )

    print("\n[Info] Using ORIGINAL weights .")
    eff_bits = (int(qdtype.replace("int", "")) if qdtype.startswith("int") else None)
    print(f"[Config] quant_dtype={qdtype}, eff_bits={eff_bits}, group_size={args.group_size}, "
          f"w4_backend={args.w4_backend}, device={device.type}")


    print("[Dump] collecting original state_dict to CPU...")
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ensure_dir(args.out_original_weights)
    torch.save(original_state_dict, args.out_original_weights)
    print(f"  -> saved original weights: {args.out_original_weights}")

    quant_err_dict = {}
    quantized_weight_dict = {} if args.out_quantized_weights else None

    layer_dims = {}
    processed = 0

    print(f"\n[Run] Building Wq and Eq = W - Wq ...")
    for name, W_cpu in tqdm(original_state_dict.items(), desc="Layers"):
        if not is_target_linear_weight(name, W_cpu):
            continue


        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_id = m.group(1)
            module_type = name.split(".")[-2]
            layer_dims.setdefault(layer_id, {})[module_type] = tuple(W_cpu.shape)


        W = W_cpu.to(device, non_blocking=True)


        if qdtype.startswith("int"):
            b = int(qdtype.replace("int", ""))
            if b == 4 and args.w4_backend == "triton":
                if not _HAS_TRITON or device.type != "cuda":
                    print(f"  [WARN] Triton not available; falling back to fake 4-bit for {name}")
                    Wq = fake_quantize_asymmetric_anybit(W, group_size=args.group_size, bits=4)
                else:
                    Wq = triton_dequantize_w(W, group_size=args.group_size)
            else:
                Wq = fake_quantize_asymmetric_anybit(W, group_size=args.group_size, bits=b)
        else:

            Wq = fake_quantize_blockfloat(W, mode=qdtype, group_size=args.group_size)


        Wq_cpu = Wq.cpu()
        Eq = (W_cpu - Wq_cpu).to(torch.float32)
        quant_err_dict[name] = Eq
        if quantized_weight_dict is not None:
            quantized_weight_dict[name] = Wq_cpu.to(torch.float16).contiguous()
        processed += 1


        try:
            on = W.norm().item()
            qn = Wq.norm().item()
            en = Eq.norm().item()
            er = en / max(on, 1e-12)
            print(f"  • {name}: shape={tuple(W_cpu.shape)} | "
                  f"||W||={on:.4f} ||Wq||={qn:.4f} ||E||={en:.4f} (ratio {er:.4f})")
        except Exception:
            pass


        del W, Wq, Wq_cpu, Eq
        if device.type == "cuda":
            torch.cuda.empty_cache()


    print(f"\n[Dims] First 3 layers discovered for {args.model_name}:")
    for lid in sorted(layer_dims.keys())[:3]:
        print(f"  - layer {lid}:")
        for mod, shp in sorted(layer_dims[lid].items()):
            print(f"      {mod}: {shp}")


    ensure_dir(args.out_quant_err)
    torch.save(quant_err_dict, args.out_quant_err)
    print(f"\n[Save] quantization errors: {args.out_quant_err}")

    if quantized_weight_dict is not None:
        ensure_dir(args.out_quantized_weights)
        torch.save(quantized_weight_dict, args.out_quantized_weights)
        print(f"[Save] fake-quant (dequantized) weights: {args.out_quantized_weights}")


    if quant_err_dict:
        total_elems = sum(t.numel() for t in quant_err_dict.values())
        avg_abs = torch.cat([t.flatten() for t in quant_err_dict.values()]).abs().mean().item()
        print("\n[Stats]")
        print(f"  • Processed layers: {processed}")
        print(f"  • Total error elements: {total_elems:,}")
        print(f"  • Average |error|: {avg_abs:.6f}")

    print("\n[Done] Ready for Step 2/3 (GSVD/correction & evaluation).")


if __name__ == "__main__":
    main()

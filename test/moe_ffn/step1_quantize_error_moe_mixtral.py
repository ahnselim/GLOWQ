"""
moe_ffn/step1_quantize_error_moe_mixtral.py
Computes fake-quantization error tensors for Mixtral MoE expert FFN weights as Step 1 of the MoE pipeline.
output :
(user-specified output paths)
|-- <out_quant_err>           (.pt)
|-- <out_original_weights>    (.pt)
`-- <out_quantized_weights>   (.pt, optional)
"""

import os
import gc
import math
import argparse
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer





@torch.no_grad()
def fake_quantize_w4_groupwise(
    W: torch.Tensor,
    group_size: int = 128,
    num_bits: int = 4,
) -> torch.Tensor:

    assert W.ndim == 2, f"Expected 2D weight, got shape {tuple(W.shape)}"
    out_dim, in_dim = W.shape
    qmax = (1 << num_bits) - 1


    pad = (group_size - (in_dim % group_size)) % group_size
    if pad:
        W_padded = F.pad(W, (0, pad), mode="constant", value=0.0)
    else:
        W_padded = W
    out_dim, in_dim_padded = W_padded.shape
    n_group = in_dim_padded // group_size

    W_groups = W_padded.view(out_dim, n_group, group_size).to(torch.float32)


    min_vals = W_groups.amin(dim=-1, keepdim=True)
    max_vals = W_groups.amax(dim=-1, keepdim=True)
    ranges = max_vals - min_vals


    tiny_mask = ranges < 1e-8

    scales = (ranges / max(qmax, 1)).clamp(min=1e-8)
    zeros = torch.round(-min_vals / scales).clamp(0, qmax)

    q = torch.round(W_groups / scales + zeros).clamp(0, qmax)
    dq = (q - zeros) * scales


    dq = torch.where(tiny_mask.expand_as(dq), W_groups, dq)

    dq = dq.view(out_dim, in_dim_padded)
    if pad:
        dq = dq[:, :in_dim]
    return dq.to(W.dtype)








def is_mixtral_moe_attn_ffn_related_weight(name: str, tensor: torch.Tensor) -> bool:

    if not (name.endswith(".weight") and tensor.ndim == 2):
        return False
    if "layers." not in name and "model.layers." not in name:
        return False

    lower = name.lower()


    if "block_sparse_moe" in lower:

        if lower.endswith("block_sparse_moe.gate.weight"):
            return True


        if ".experts." in lower:
            parts = name.split(".")
            module_name = parts[-2].lower() if len(parts) >= 2 else ""
            if module_name in {"w1", "w2", "w3"}:
                return True


    parts = name.split(".")
    module_name = parts[-2].lower() if len(parts) >= 2 else ""


    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return True


    if module_name in {"wq", "wk", "wv", "wo"}:
        return True

    return False





def main():
    parser = argparse.ArgumentParser(
        description="STEP 1 (Mixtral MoE only) - FakeQuant-based quantization error computation (no Triton)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="HF model id (default: mistralai/Mixtral-8x7B-Instruct-v0.1; change if needed)",
    )
    parser.add_argument(
        "--out_quant_err",
        type=str,
        required=True,
        help="Path to save the quantization error (Eq) dictionary (.pt)",
    )
    parser.add_argument(
        "--out_original_weights",
        type=str,
        required=True,
        help="Path to save the full original FP16 model state_dict (.pt)",
    )
    parser.add_argument(
        "--out_quantized_weights",
        type=str,
        required=False,
        help="Path to save FakeQuant (dequant) Wq for selected layers (.pt, optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for FakeQuant operations (cuda or cpu)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size along the last dimension (default: 128)",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=4,
        help="FakeQuant bit width (default: 4-bit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="torch random seed",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set if needed when loading models that use custom code",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"📚 Loading tokenizer for '{args.model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📦 Loading FP16 model: {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )


    print("💾 Extracting original state_dict (FP16) to CPU...")
    raw_state_dict = model.state_dict()
    original_state_dict: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().contiguous() for k, v in raw_state_dict.items()
    }
    del model, raw_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    quant_err_dict: Dict[str, torch.Tensor] = {}
    fake_quant_weights: Dict[str, torch.Tensor] = (
        {} if args.out_quantized_weights is not None else None
    )


    targeted_names = [
        name
        for name, tensor in original_state_dict.items()
        if is_mixtral_moe_attn_ffn_related_weight(name, tensor)
    ]
    print(f"\n🎯 Targeted Mixtral-MoE+Attention-related weights: {len(targeted_names)} tensors")

    print(
        f"🚀 Running FakeQuant (num_bits={args.num_bits}, group_size={args.group_size}) for targeted tensors..."
    )
    for name in tqdm(targeted_names, desc="FakeQuant & Eq"):
        W_cpu = original_state_dict[name]
        W = W_cpu.to(device=device, dtype=torch.float32)


        Wq = fake_quantize_w4_groupwise(
            W, group_size=args.group_size, num_bits=args.num_bits
        )
        Wq_cpu = Wq.to(dtype=torch.float16).cpu().contiguous()


        Eq = W_cpu.to(torch.float32) - Wq_cpu.to(torch.float32)
        quant_err_dict[name] = Eq.to(torch.float32)

        if fake_quant_weights is not None:
            fake_quant_weights[name] = Wq_cpu


        orig_norm = W_cpu.norm().item()
        q_norm = Wq_cpu.norm().item()
        err_norm = Eq.norm().item()
        err_ratio = err_norm / max(orig_norm, 1e-12)
        print(
            f"[{name}] shape={tuple(W_cpu.shape)}, "
            f"||W||={orig_norm:.4f}, ||Wq||={q_norm:.4f}, "
            f"||E||={err_norm:.4f}, ||E||/||W||={err_ratio:.4f}"
        )

        del W, Wq, Wq_cpu, Eq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    os.makedirs(os.path.dirname(args.out_quant_err) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_original_weights) or ".", exist_ok=True)
    if args.out_quantized_weights:
        os.makedirs(os.path.dirname(args.out_quantized_weights) or ".", exist_ok=True)

    print("\n💾 Saving outputs ...")
    torch.save(quant_err_dict, args.out_quant_err)
    torch.save(original_state_dict, args.out_original_weights)
    if fake_quant_weights is not None and args.out_quantized_weights:
        torch.save(fake_quant_weights, args.out_quantized_weights)

    print(f"  • Quantization errors (Eq): {args.out_quant_err}")
    print(f"  • Original weights (W):     {args.out_original_weights}")
    if args.out_quantized_weights:
        print(f"  • Fake-quant weights (Wq): {args.out_quantized_weights}")


    if quant_err_dict:
        all_errors = torch.cat([t.flatten() for t in quant_err_dict.values()])
        total_elems = all_errors.numel()
        avg_abs_err = all_errors.abs().mean().item()
        max_abs_err = all_errors.abs().max().item()
        print("\n📊 Error Statistics:")
        print(f"  • Total error elements: {total_elems:,}")
        print(f"  • Mean |E|:             {avg_abs_err:.6f}")
        print(f"  • Max  |E|:             {max_abs_err:.6f}")

    print(
        "\n✅ DONE. This output (err_dict + fake_quant_weights + original_state_dict)"
    )
    print("   - can be shared across both Design A (uniform E_cat) and Design B (usage-weighted E_cat).")
    print("   - In Step 2, split into two versions by applying different usage pi_i values.")


if __name__ == "__main__":
    main()

"""
moe_ffn/step1_moe_export_usage_qwen15.py
Exports expert usage statistics from Qwen1.5-MoE calibration data for grouping and analysis.
output :
(user-specified output path)
`-- <output_json>   (flat expert usage JSON)
"""

import os
import json
import argparse
from typing import Optional, List, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Not a boolean: {v}")


def build_calibration_tokens(
    tokenizer,
    nsamples: int = 64,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = None,
) -> torch.Tensor:
    print(
        f"[build_calibration_tokens] dataset={dataset_name}, nsamples={nsamples}, seqlen={seqlen}"
    )

    try:
        ds = load_dataset(
            dataset_name,
            name=dataset_config,
            split="train",
            streaming=True,
        )
    except Exception:
        ds = load_dataset(dataset_name, name=dataset_config, split="train")

    sample_budget = max(nsamples * 5, nsamples)
    if hasattr(ds, "take"):
        iterator = ds.take(sample_budget)
    else:
        total = len(ds)
        subset = min(sample_budget, total)
        iterator = ds.select(range(subset))

    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None):
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    token_buffer: List[int] = []
    samples: List[torch.Tensor] = []

    for row in iterator:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = (
            tokenizer(text, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        if not ids:
            continue
        if eos_id is not None:
            ids.append(eos_id)
        token_buffer.extend(ids)

        while len(token_buffer) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(token_buffer[:seqlen], dtype=torch.long))
            token_buffer = token_buffer[seqlen:]
            if len(samples) >= nsamples:
                break
        if len(samples) >= nsamples:
            break

    if len(samples) < nsamples:
        print(
            f"[build_calibration_tokens] WARNING: collected only {len(samples)}/{nsamples} sequences."
        )

    if samples:
        return torch.stack(samples, dim=0)
    return torch.empty(0, seqlen, dtype=torch.long)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[MoE usage export] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    tokens = build_calibration_tokens(
        tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        dataset_name=args.calib_dataset,
        dataset_config=args.calib_config,
    )
    if tokens.numel() == 0:
        raise RuntimeError(
            f"Failed to build calibration tokens from {args.calib_dataset}"
        )


    router_usage_counts: Dict[str, torch.Tensor] = {}
    router_num_experts: Dict[str, int] = {}

    def make_router_hook(name: str, module: nn.Linear):
        num_experts = module.out_features
        router_num_experts[name] = num_experts

        router_usage_counts[name] = torch.zeros(
            num_experts, dtype=torch.long, device="cpu"
        )
        top_k = min(args.top_k, num_experts)

        def hook(mod, inp, out):

            logits = out
            if isinstance(logits, tuple):
                logits = logits[0]

            if logits.dim() == 3:
                b, t, e = logits.shape
                logits = logits.reshape(b * t, e)
            elif logits.dim() == 2:
                _, e = logits.shape
            else:
                return

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                _, idx = probs.topk(top_k, dim=-1)
                idx_flat = idx.reshape(-1)
                counts = torch.bincount(
                    idx_flat.cpu(), minlength=num_experts
                )
                router_usage_counts[name] += counts

        return hook


    handles = []
    for name, module in model.named_modules():
        if name.endswith("router") and isinstance(module, nn.Linear):
            print(f"[MoE usage export] Register router hook on: {name}")
            h = module.register_forward_hook(make_router_hook(name, module))
            handles.append(h)

    if not handles:
        print(
            "[MoE usage export] WARNING: No modules ending with '.router' found. "
            "Check model architecture or adjust matching rule."
        )

    model.eval()
    for i in tqdm(range(tokens.shape[0]), desc="Collecting MoE routing usage"):
        input_ids = tokens[i : i + 1].to(device)
        with torch.no_grad():
            _ = model(input_ids)

    for h in handles:
        h.remove()








    flat_usage: Dict[str, float] = {}

    for router_name, counts in router_usage_counts.items():
        num_experts = router_num_experts[router_name]


        if "." in router_name:
            prefix = router_name.rsplit(".", 1)[0]
        else:
            prefix = router_name

        for expert_idx in range(num_experts):
            count_e = float(counts[expert_idx].item())
            expert_prefix = f"{prefix}.experts.{expert_idx}"


            for sub in ("w1", "w2", "w3"):
                weight_name = f"{expert_prefix}.{sub}.weight"
                flat_usage[weight_name] = count_e

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(flat_usage, f, indent=2, ensure_ascii=False)

    print(
        f"[MoE usage export] Saved FLAT usage stats for {len(flat_usage)} weights to {args.output_json}"
    )
    print("  (format: { 'model.layers.L.mlp.experts.e.w{1,2,3}.weight': count, ... })")
    print("  -> You can pass this directly to step2 --usage_stats_path and use usage_mode=usage_weighted.")


if __name__ == "__main__":
    p = argparse.ArgumentParser("STEP 1 (MoE) - Export expert usage stats for Qwen1.5-MoE (flat JSON)")
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B",
        help="HF model name (MoE).",
    )
    p.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to save usage JSON (e.g., ./output/qwen15_moe/expert_usage_flat.json)",
    )
    p.add_argument(
        "--nsamples",
        type=int,
        default=64,
        help="Number of sequences for routing statistics.",
    )
    p.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for each sample.",
    )
    p.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset name (HF datasets).",
    )
    p.add_argument(
        "--calib_config",
        type=str,
        default=None,
        help="Optional dataset config name.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Top-k experts used per token (Qwen MoE typically uses 2).",
    )
    p.add_argument(
        "--trust_remote_code",
        type=str2bool,
        default=True,
        help="Qwen MoE typically requires remote code.",
    )

    args = p.parse_args()
    main(args)

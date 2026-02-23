"""
w4ax/step4_evaluate_cumulative.py
Cumulatively evaluates restoration quality as important Shared-B groups are progressively enabled.
output :
./cumulative_results/   (default; or --output_dir)
|-- cumulative_results.csv
`-- *_performance_plot.png   (one per selected metric/method)
"""

import argparse
import gc
import json
import math
import os
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from step3_eval_integrated import (
    apply_activation_fake_quant,
    apply_quantized_weights,
    evaluate,
    measure_generation_metrics,
    patch_svd_correction_wrappers,
)

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The quick brown fox",
    "In a shocking finding, scientists discovered that",
]


def derive_group_key(bkey: str) -> str:
    if "B_shared" in bkey:
        return bkey.replace(".B_shared", "")
    return bkey.rsplit(".", 1)[0]


def filter_bmap(bmap: Mapping[str, str], groups: Iterable[str]) -> dict[str, str]:
    groups = set(groups)
    if not groups:
        return {}
    return {
        weight_name: bkey
        for weight_name, bkey in bmap.items()
        if derive_group_key(bkey) in groups
    }


def load_fake_quant_model(
    model_name: str,
    original_weights_path: str,
    quantized_weights_path: str,
    device: torch.device,
    trust_remote_code: bool,
) -> AutoModelForCausalLM:
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    original_weights = torch.load(
        original_weights_path, map_location="cpu", weights_only=True
    )
    model_fp16.load_state_dict(original_weights)
    del original_weights

    fake_quant_weights = torch.load(
        quantized_weights_path, map_location="cpu", weights_only=True
    )
    apply_quantized_weights(model_fp16, fake_quant_weights)
    del fake_quant_weights

    model = model_fp16.to(device)
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    return model


def run_single_evaluation(args, groups_to_patch: set[str]):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_fake_quant_model(
        args.model_name,
        args.original_weights_path,
        args.quantized_weights_path,
        device,
        args.trust_remote_code,
    )

    shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f:
        full_bmap = json.load(f)

    module_names = [name.replace(".weight", "") for name in full_bmap.keys()]
    if args.activation_bits > 0:
        apply_activation_fake_quant(
            model,
            module_names,
            act_bits=args.activation_bits,
            group_size=args.activation_group_size,
        )

    filtered_bmap = filter_bmap(full_bmap, groups_to_patch)
    if filtered_bmap:
        model = patch_svd_correction_wrappers(model, shared, filtered_bmap, alpha_svd=1.0)

    ppl, _ = evaluate(
        model,
        tokenizer,
        device,
        f"Cumulative Restore ({len(groups_to_patch)} groups, {args.model_name})",
    )

    gen_metrics = None
    if not args.skip_gen:
        gen_metrics = measure_generation_metrics(
            model,
            tokenizer,
            device,
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=args.gen_do_sample,
            num_beams=args.gen_num_beams,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            repeats=args.gen_repeats,
        )

    del model, shared, full_bmap, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return ppl, gen_metrics


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.rankings_json, "r") as f:
        rankings = json.load(f)

    if args.methods:
        missing = [m for m in args.methods if m not in rankings]
        if missing:
            raise ValueError(f"Requested methods not found in rankings: {missing}")
        metrics_to_use = args.methods
    else:
        metrics_to_use = list(rankings.keys())

    results = []

    for metric in metrics_to_use:
        ranked_groups = rankings[metric]
        print(f"\n{'='*20} Metric: {metric} ({len(ranked_groups)} groups) {'='*20}")
        pbar = tqdm(range(0, len(ranked_groups) + 1), desc=f"Metric: {metric}")
        for i in pbar:
            groups_to_patch = set(ranked_groups[:i])
            pbar.set_description(
                f"Metric: {metric} | Restoring {i}/{len(ranked_groups)} groups"
            )

            ppl, gen_metrics = run_single_evaluation(args, groups_to_patch)
            ttfb = gen_metrics["ttfb_ms_median"] if gen_metrics else math.nan
            tokps = gen_metrics["tok_s_median"] if gen_metrics else math.nan

            results.append(
                {
                    "metric": metric,
                    "restored_count": i,
                    "ppl": ppl,
                    "ttfb_ms_median": ttfb,
                    "tok_s_median": tokps,
                }
            )

            df = pd.DataFrame(results)
            df.to_csv(os.path.join(args.output_dir, "cumulative_results.csv"), index=False)

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "cumulative_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ All evaluation results saved to {csv_path}")

    for metric in metrics_to_use:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Number of Restored Groups")
        ax1.set_ylabel("Perplexity (PPL)", color="tab:blue")
        ax1.plot(
            metric_df["restored_count"],
            metric_df["ppl"],
            color="tab:blue",
            marker="o",
            label="PPL",
        )
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Throughput (tok/s)", color="tab:red")
        ax2.plot(
            metric_df["restored_count"],
            metric_df["tok_s_median"],
            color="tab:red",
            marker="x",
            label="Throughput (tok/s)",
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title(f"Performance vs. Restored Groups ({metric})")
        fig.tight_layout()
        plt.grid(True)

        plot_path = os.path.join(args.output_dir, f"{metric}_performance_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Plot saved for {metric}: {plot_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Step 4: Cumulatively evaluate fake-quant model restoration."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--original_weights_path", type=str, required=True)
    parser.add_argument("--quantized_weights_path", type=str, required=True)
    parser.add_argument("--shared_path", type=str, required=True)
    parser.add_argument("--bmap_path", type=str, required=True)
    parser.add_argument("--rankings_json", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cumulative_results",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Required for some model families",
    )
    parser.add_argument(
        "--activation_bits",
        type=int,
        default=8,
        help="Bit-width for fake activation quantization (set <=0 to disable).",
    )
    parser.add_argument(
        "--activation_group_size",
        type=int,
        default=128,
        help="Group size for activation fake quantization.",
    )
    parser.add_argument(
        "--skip_gen",
        action="store_true",
        help="Skip generation latency/throughput measurement",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=50,
        help="New tokens for throughput measurement",
    )
    parser.add_argument(
        "--gen_repeats",
        type=int,
        default=1,
        help="Repeat generation measurement this many times",
    )
    parser.add_argument(
        "--gen_do_sample", action="store_true", help="Use sampling for generation"
    )
    parser.add_argument(
        "--gen_num_beams", type=int, default=1, help="Beam size for generation"
    )
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_p", type=float, default=1.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of ranking metrics to evaluate (default: all).",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())

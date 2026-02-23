"""
w4ax/step3_eval.py
Runs baseline and SVD-corrected W4Ax evaluations using the integrated evaluation utilities.
output :
(no output files)
`-- stdout/stderr metrics and logs only
"""

import argparse
import gc
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from step3_eval_integrated import (
    AddSVDCorrection,
    apply_activation_fake_quant,
    apply_quantized_weights,
    patch_svd_correction_wrappers,
    evaluate,
    measure_generation_metrics,
)


default_prompts = [
    "Hello, my name is",
    "The quick brown fox",
    "In a shocking finding, scientists discovered that",
]


def run_fake_quant_evaluation(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📥 Loading original FP16 model for baseline comparison: {args.model_name}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    print(f"📥 Loading original weights from: {args.original_weights_path}")
    original_weights = torch.load(
        args.original_weights_path, map_location="cpu", weights_only=True
    )
    model_fp16.load_state_dict(original_weights)
    del original_weights

    print(f"📦 Loading fake-quant weights from: {args.quantized_weights_path}")
    fake_quant_weights = torch.load(
        args.quantized_weights_path, map_location="cpu", weights_only=True
    )
    apply_quantized_weights(model_fp16, fake_quant_weights)
    del fake_quant_weights

    model = model_fp16.to(device)
    method_label = "Fake-Quant W4A8 GEMM"
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"🧩 Loading correction artifacts and patching wrappers for {args.model_name} ..."
    )
    shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)
    module_names = [name.replace(".weight", "") for name in bmap.keys()]
    if args.activation_bits > 0:
        apply_activation_fake_quant(
            model,
            module_names,
            act_bits=args.activation_bits,
            group_size=args.activation_group_size,
        )
    model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)

    results = {}

    print("\n=== BASELINE EVALUATION (NO SVD CORRECTION) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 0.0
    ppl_base, time_base = evaluate(
        model, tokenizer, device, f"{method_label} (No SVD, {args.model_name})"
    )
    gen_metrics_base = None
    if not args.skip_gen:
        print("Measuring generation metrics for baseline...")
        try:
            gen_metrics_base = measure_generation_metrics(
                model,
                tokenizer,
                device,
                prompts=default_prompts,
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=args.gen_do_sample,
                num_beams=args.gen_num_beams,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repeats=args.gen_repeats,
            )
            print(
                f"   • Baseline TTFB: {gen_metrics_base['ttfb_ms_median']:.1f}ms (median)"
            )
            print(
                f"   • Baseline Throughput: {gen_metrics_base['tok_s_median']:.2f} tok/s (median)"
            )
        except Exception as e:
            print(f"Generation measurement failed for baseline: {e}")
            gen_metrics_base = None
    results["baseline"] = {
        "ppl": ppl_base,
        "time": time_base,
        "generation_metrics": gen_metrics_base,
    }

    print("\n=== SVD CORRECTION EVALUATION (ALPHA=1.0) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 1.0
    ppl, time_val = evaluate(
        model,
        tokenizer,
        device,
        f"{method_label} + SVD Correction (α=1.0, {args.model_name})",
    )
    gen_metrics_svd = None
    if not args.skip_gen:
        print("Measuring generation metrics for SVD correction...")
        try:
            gen_metrics_svd = measure_generation_metrics(
                model,
                tokenizer,
                device,
                prompts=default_prompts,
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=args.gen_do_sample,
                num_beams=args.gen_num_beams,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repeats=args.gen_repeats,
            )
            print(f"   • SVD TTFB: {gen_metrics_svd['ttfb_ms_median']:.1f}ms (median)")
            print(
                f"   • SVD Throughput: {gen_metrics_svd['tok_s_median']:.2f} tok/s (median)"
            )
        except Exception as e:
            print(f"Generation measurement failed for SVD: {e}")
            gen_metrics_svd = None
    results["svd"] = {
        "ppl": ppl,
        "time": time_val,
        "generation_metrics": gen_metrics_svd,
    }

    print(
        f"\n{'='*15} FINAL SUMMARY ({args.model_name} + Fake-Quant W4A8 + Shared-B) {'='*15}"
    )
    print(f"Model: {args.model_name} (Fake-Quant W4A8 GEMM, )")
    print("-" * 120)
    print(
        f"{'Method':<50} | {'Perplexity':<10} | {'Time (s)':<8} | {'TTFB(ms)':<10} | {'tok/s':<10}"
    )
    print("-" * 120)
    base_data = results["baseline"]
    base_gen = base_data.get("generation_metrics")
    ttfb_base_str = f"{base_gen['ttfb_ms_median']:.1f}" if base_gen else "-"
    tok_s_base_str = f"{base_gen['tok_s_median']:.2f}" if base_gen else "-"
    print(
        f"{method_label + ' Baseline (no SVD)':<50} | {base_data['ppl']:<10.4f} | {base_data['time']:<8.2f} | {ttfb_base_str:<10} | {tok_s_base_str:<10}"
    )
    svd_data = results["svd"]
    svd_gen = svd_data.get("generation_metrics")
    ttfb_svd_str = f"{svd_gen['ttfb_ms_median']:.1f}" if svd_gen else "-"
    tok_s_svd_str = f"{svd_gen['tok_s_median']:.2f}" if svd_gen else "-"
    print(
        f"{method_label + ' + SVD Correction (α=1.0)':<50} | {svd_data['ppl']:<10.4f} | {svd_data['time']:<8.2f} | {ttfb_svd_str:<10} | {tok_s_svd_str:<10}"
    )
    print("=" * 120)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate fake-quant (W4A8) model with Shared-B SVD correction."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--shared_path", required=True)
    parser.add_argument("--bmap_path", required=True)
    parser.add_argument(
        "--original_weights_path",
        required=True,
        help="Path to original weights (from step1)",
    )
    parser.add_argument(
        "--quantized_weights_path",
        required=True,
        help="Path to fake-quant (dequantized) weights saved in step1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Required for some model families",
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
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_fake_quant_evaluation(args)


if __name__ == "__main__":
    main()

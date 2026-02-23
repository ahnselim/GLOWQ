"""
step4_evaluate_cumulative.py
Cumulatively evaluates model performance while progressively restoring groups based on importance rankings.
output :
/home/tako/asl/GLSR/EXP-C/pt_files/Qwen_Qwen2.5_7B/cumulative_results/   (default; or --output_dir)
|-- cumulative_results.csv
`-- *_performance_plot.png   (one per ranking metric)
"""

import argparse, json, torch, torch.nn as nn, gc, os, time, re, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer













_here = os.path.dirname(__file__)
_root = os.path.abspath(os.path.join(_here, ".."))
_cuda4bit_path = os.path.join(_root, "CUDA4BIT")
if _cuda4bit_path not in sys.path:
    sys.path.insert(0, _cuda4bit_path)

from step3_eval import (
    TritonTrue4BitLinear,
    AddSVDCorrection,
    MiniGroupCache,
    evaluate,
    measure_generation_metrics,
    get_parent_module,
)

try:
    from cuda_w4a16.linear import CudaW4A16Linear, convert_to_cuda_w4a16

    HAS_CUDA_W4A16 = True
except Exception:
    HAS_CUDA_W4A16 = False


def _role_from_suffix(sfx: str) -> str:
    if sfx.endswith("q_proj"):
        return "q"
    if sfx.endswith("k_proj"):
        return "k"
    if sfx.endswith("v_proj"):
        return "v"
    if sfx.endswith("gate_proj"):
        return "gate"
    if sfx.endswith("up_proj"):
        return "up"
    return "solo"


def patch_svd_correction_wrappers_cumulative(
    model, shared, bmap, groups_to_patch: set, alpha_svd=1.0
):
    
    patched_count = 0
    gkey2cache: dict[str, MiniGroupCache] = {}
    for weight_name, bkey in bmap.items():

        gkey = None
        if "B_shared" in bkey:
            gkey = bkey.replace(".B_shared", "")
        else:
            gkey = bkey.rsplit(".", 1)[0]


        if gkey not in groups_to_patch:
            continue

        module_name = weight_name.replace(".weight", "")
        B_q = shared.get(bkey)


        is_group = "B_shared" in bkey
        if is_group:
            module_suffix = module_name.split(".")[-1]
            a_key = f"{gkey}.{module_suffix}.A"
            role = _role_from_suffix(module_suffix)
            cache = gkey2cache.setdefault(gkey, MiniGroupCache())
        else:
            a_key = gkey + ".A"
            role = "solo"
            cache = None
        A_q = shared.get(a_key)

        if A_q is None or B_q is None:
            continue

        try:
            parent, attr_name = get_parent_module(model, module_name)
            inner = getattr(parent, attr_name)
            valid_types = (TritonTrue4BitLinear,)
            if HAS_CUDA_W4A16:
                valid_types = (TritonTrue4BitLinear, CudaW4A16Linear)
            if not isinstance(inner, valid_types):
                continue

            wrapped = AddSVDCorrection(
                inner, A_q, B_q, role, is_group, cache, alpha_svd
            )
            setattr(parent, attr_name, wrapped)
            patched_count += 1
        except AttributeError:
            continue


    return model







def run_evaluation(
    model_name,
    original_weights_path,
    shared_path,
    bmap_path,
    groups_to_patch,
    device,
    trust_remote_code=False,
):
    

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
    if HAS_CUDA_W4A16:
        model = convert_to_cuda_w4a16(model_fp16).to(device)
    else:

        from step3_eval import convert_to_triton_4bit

        model = convert_to_triton_4bit(model_fp16).to(device)
    del model_fp16, original_weights
    gc.collect()
    torch.cuda.empty_cache()


    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    shared = torch.load(shared_path, map_location=device, weights_only=True)
    with open(bmap_path, "r") as f:
        bmap = json.load(f)

    model = patch_svd_correction_wrappers_cumulative(
        model, shared, bmap, groups_to_patch
    )


    ppl, _ = evaluate(model, tokenizer, device, "cumulative_eval")

    prompts = ["Hello, my name is", "The quick brown fox"]
    gen_metrics = measure_generation_metrics(
        model, tokenizer, device, prompts=prompts, max_new_tokens=50
    )


    del model, shared, bmap, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return ppl, gen_metrics


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.rankings_json, "r") as f:
        rankings = json.load(f)

    results = []

    for metric, ranked_groups in rankings.items():
        print(f"\n{'='*20} Starting Evaluation for Metric: {metric} {'='*20}")

        pbar = tqdm(range(0, len(ranked_groups) + 1), desc=f"Metric: {metric}")
        for i in pbar:
            groups_to_patch = set(ranked_groups[:i])

            pbar.set_description(
                f"Metric: {metric} | Restoring {i}/{len(ranked_groups)} groups"
            )

            ppl, gen_metrics = run_evaluation(
                args.model_name,
                args.original_weights_path,
                args.shared_path,
                args.bmap_path,
                groups_to_patch,
                device,
                trust_remote_code=args.trust_remote_code,
            )

            results.append(
                {
                    "metric": metric,
                    "restored_count": i,
                    "ppl": ppl,
                    "ttfb_ms_median": gen_metrics["ttfb_ms_median"],
                    "tok_s_median": gen_metrics["tok_s_median"],
                }
            )


            df = pd.DataFrame(results)
            df.to_csv(
                os.path.join(args.output_dir, "cumulative_results.csv"), index=False
            )


    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "cumulative_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ All evaluation results saved to {csv_path}")


    for metric in rankings.keys():
        metric_df = df[df["metric"] == metric]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Cumulatively evaluate model performance."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--original_weights_path", type=str, required=True)
    parser.add_argument("--shared_path", type=str, required=True)
    parser.add_argument("--bmap_path", type=str, required=True)
    parser.add_argument("--rankings_json", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/tako/asl/GLSR/EXP-C/pt_files/Qwen_Qwen2.5_7B/cumulative_results",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set for models requiring custom Hugging Face code (e.g., some Qwen variants).",
    )
    args = parser.parse_args()
    main(args)



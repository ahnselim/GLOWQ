"""
w4ax/step3_1_calculate_importance.py
Calculates and ranks Shared-B group importance scores to prioritize cumulative restoration experiments.
output :
<dirname(output_json) or ./output>/
`-- <basename(output_json)>   (default: importance_rankings.json)
"""

import json
import torch
import argparse
import re
from typing import Optional
from tqdm import tqdm
from collections import defaultdict




def load_data(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def get_all_groups(b_ref_map: dict) -> dict:
    groups = defaultdict(list)

    for weight_name, bkey in b_ref_map.items():
        if "B_shared" in bkey:
            gkey = bkey.replace(".B_shared", "")
        else:
            gkey = bkey.rsplit(".", 1)[0]
        groups[gkey].append(weight_name)


    def get_layer_num(name: str) -> int:
        match = re.search(r"layers\.(\d+)", name)
        if match:
            return int(match.group(1))

        return float("inf")


    sorted_group_keys = sorted(groups.keys(), key=get_layer_num)


    print(f"✅ Found {len(groups)} unique groups (qkv, mlp, o_proj, down_proj, etc.).")
    return {k: groups[k] for k in sorted_group_keys}


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg and device_arg.lower() != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main(args):
    device = _resolve_device(args.device)
    print(f"Using device: {device}")


    err_T = load_data(args.err_path)
    original_weights = load_data(args.original_weights_path)
    shared = load_data(args.shared_path, device=device)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)


    groups = get_all_groups(bmap)

    scores = {
        "gsvd_singular_value_sum": {},
        "normalized_error_ratio": {},
        "frobenius_norm_error": {},
        "cosine_similarity": {},
    }

    print("Calculating importance scores for each group...")
    for gkey, weight_names in tqdm(groups.items(), desc="Processing Groups"):

        a_matrices = []
        if "qkv" in gkey or "mlp" in gkey and "down_proj" not in gkey:
            for name in weight_names:
                module_suffix = name.split(".")[-2]
                a_key = f"{gkey}.{module_suffix}.A"
                if a_key in shared:
                    a_matrices.append(shared[a_key])
        else:
            a_key = f"{gkey}.A"
            if a_key in shared:
                a_matrices.append(shared[a_key])

        if a_matrices:
            gsvd_score = sum([(A.norm() ** 2).item() for A in a_matrices])
            scores["gsvd_singular_value_sum"][gkey] = gsvd_score


        err_tensors = [err_T[name] for name in weight_names if name in err_T]
        orig_tensors = [
            original_weights[name] for name in weight_names if name in original_weights
        ]

        if not err_tensors or not orig_tensors:
            continue

        err_flat = torch.cat([t.flatten() for t in err_tensors]).to(device)
        orig_flat = torch.cat([t.flatten() for t in orig_tensors]).to(device)

        err_norm = torch.norm(err_flat).item()
        orig_norm = torch.norm(orig_flat).item()

        scores["frobenius_norm_error"][gkey] = err_norm
        if orig_norm > 1e-8:
            scores["normalized_error_ratio"][gkey] = err_norm / orig_norm

        wq_flat = orig_flat - err_flat
        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_flat, wq_flat, dim=0
        ).item()
        scores["cosine_similarity"][gkey] = 1.0 - cosine_sim


    rankings = {}
    for metric, gkey_scores in scores.items():
        sorted_gkeys = sorted(
            gkey_scores.keys(), key=lambda k: gkey_scores[k], reverse=True
        )
        rankings[metric] = sorted_gkeys


    rankings["layer_order"] = list(groups.keys())


    with open(args.output_json, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"\n✅ Importance rankings saved to {args.output_json}")
    for metric, ranked_list in rankings.items():
        print(f"  - Top 3 for '{metric}': {ranked_list[:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Calculate and rank group importances (Fixed)."
    )
    parser.add_argument("--err_path", type=str, required=True)
    parser.add_argument("--original_weights_path", type=str, required=True)
    parser.add_argument("--shared_path", type=str, required=True)
    parser.add_argument("--bmap_path", type=str, required=True)
    parser.add_argument(
        "--output_json", type=str, default="./output/importance_rankings.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (default: auto-detect CUDA, fallback to CPU)",
    )
    args = parser.parse_args()
    main(args)

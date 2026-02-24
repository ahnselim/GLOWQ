#!/usr/bin/env python3
"""
src/restoration/step3_1_calculate_importance.py
Calculates and ranks Shared-B restoration groups using configurable importance metrics.
output :
<dirname(output_json) or ./output>/
`-- <basename(output_json)>   (default: importance_rankings.json)
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Optional

import torch
from tqdm import tqdm


METRIC_SPECS = {
    "gsvd": {
        "rank_key": "gsvd_singular_value_sum",
        "aliases": {"gsvd", "gsvd_singular_value_sum"},
    },
    "norm_error": {
        "rank_key": "normalized_error_ratio",
        "aliases": {
            "norm",
            "norm_error",
            "normalized",
            "normalized_error",
            "normalized_error_ratio",
        },
    },
    "frobenius_norm_error": {
        "rank_key": "frobenius_norm_error",
        "aliases": {
            "fro",
            "frobenius",
            "frobenius_norm",
            "frobenius_norm_error",
        },
    },
    "cosine_similarity": {
        "rank_key": "cosine_similarity",
        "aliases": {"cos", "cosine", "cosine_similarity"},
    },
    "layer_order": {
        "rank_key": "layer_order",
        "aliases": {"layer", "layer_order"},
    },
}


def load_data(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg and device_arg.lower() != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _split_metrics_arg(metrics_arg) -> list[str]:
    if metrics_arg is None:
        return ["gsvd", "norm_error"]
    if isinstance(metrics_arg, (list, tuple)):
        raw_items = [str(x) for x in metrics_arg]
    else:
        text = str(metrics_arg)
        phrase_aliases = (
            ("norm error", "norm_error"),
            ("normalized error", "normalized_error"),
            ("frobenius norm error", "frobenius_norm_error"),
            ("layer order", "layer_order"),
        )
        lowered = text.lower()
        for src, dst in phrase_aliases:
            lowered = lowered.replace(src, dst)
        raw_items = re.split(r"[,\s]+", lowered)
    return [item.strip() for item in raw_items if str(item).strip()]


def _normalize_metric_names(metrics_arg) -> list[str]:
    alias_to_canonical = {}
    for canonical, spec in METRIC_SPECS.items():
        for alias in spec["aliases"]:
            alias_to_canonical[alias.lower()] = canonical

    selected = []
    unknown = []
    for token in _split_metrics_arg(metrics_arg):
        metric = alias_to_canonical.get(token.lower())
        if metric is None:
            unknown.append(token)
            continue
        if metric not in selected:
            selected.append(metric)

    if unknown:
        supported = ", ".join(sorted(alias_to_canonical.keys()))
        raise ValueError(
            f"Unsupported importance metric(s): {', '.join(unknown)}. Supported aliases: {supported}"
        )

    if not selected:
        return ["gsvd", "norm_error"]
    return selected


def _compute_group_scores(groups, err_tensors_map, original_weights, shared, device):
    scores = {
        "gsvd_singular_value_sum": {},
        "normalized_error_ratio": {},
        "frobenius_norm_error": {},
        "cosine_similarity": {},
    }

    print("Calculating importance scores for each group...")
    for gkey, weight_names in tqdm(groups.items(), desc="Processing Groups"):
        a_matrices = []
        if ("qkv" in gkey) or ("mlp" in gkey and "down_proj" not in gkey):
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
            scores["gsvd_singular_value_sum"][gkey] = sum(
                (A.norm() ** 2).item() for A in a_matrices
            )

        err_tensors = [err_tensors_map[name] for name in weight_names if name in err_tensors_map]
        orig_tensors = [original_weights[name] for name in weight_names if name in original_weights]
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
        cosine_sim = torch.nn.functional.cosine_similarity(orig_flat, wq_flat, dim=0).item()
        scores["cosine_similarity"][gkey] = 1.0 - cosine_sim

    return scores


def _bool_arg(args, name: str, default: bool) -> bool:
    return bool(getattr(args, name, default))


@torch.no_grad()
def main(args):
    device = _resolve_device(getattr(args, "device", "auto"))
    selected_metrics = _normalize_metric_names(getattr(args, "metrics", None))
    include_component_rankings = _bool_arg(args, "include_component_rankings", True)
    include_layer_order = _bool_arg(args, "include_layer_order", True)

    print(f"Using device: {device}")
    print(f"Selected importance metrics: {selected_metrics}")

    err_tensors_map = load_data(args.err_path)
    original_weights = load_data(args.original_weights_path)
    shared = load_data(args.shared_path, device=device)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)

    groups = get_all_groups(bmap)
    scores = _compute_group_scores(groups, err_tensors_map, original_weights, shared, device)

    rankings = {}
    if include_component_rankings:
        for metric_name in selected_metrics:
            rank_key = METRIC_SPECS[metric_name]["rank_key"]
            if rank_key == "layer_order":
                continue
            gkey_scores = scores.get(rank_key, {})
            rankings[rank_key] = sorted(
                gkey_scores.keys(), key=lambda k: gkey_scores[k], reverse=True
            )

    if include_layer_order and "layer_order" in selected_metrics:
        rankings["layer_order"] = list(groups.keys())

    if not rankings:
        raise ValueError(
            "No rankings generated. Check metrics/include flags. "
            "At least one component metric or layer_order must be enabled."
        )

    with open(args.output_json, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"\n✅ Importance rankings saved to {args.output_json}")
    for metric, ranked_list in rankings.items():
        print(f"  - Top 3 for '{metric}': {ranked_list[:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3_1: Calculate and rank restoration group importances."
    )
    parser.add_argument("--err_path", type=str, required=True)
    parser.add_argument("--original_weights_path", type=str, required=True)
    parser.add_argument("--shared_path", type=str, required=True)
    parser.add_argument("--bmap_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="./output/importance_rankings.json")
    parser.add_argument(
        "--metrics",
        type=str,
        default="gsvd,norm_error",
        help="Comma/space-separated metrics (e.g., 'gsvd,norm_error').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (default: auto-detect CUDA, fallback to CPU).",
    )
    parser.add_argument(
        "--no_include_component_rankings",
        action="store_true",
        help="Disable score-based rankings (gsvd/norm/fro/cosine).",
    )
    parser.add_argument(
        "--no_include_layer_order",
        action="store_true",
        help="Disable layer_order ranking output.",
    )
    parsed_args = parser.parse_args()
    parsed_args.include_component_rankings = not parsed_args.no_include_component_rankings
    parsed_args.include_layer_order = not parsed_args.no_include_layer_order
    main(parsed_args)

"""
w_type/step2_randomized_gsvd.py
Runs integrated randomized GSVD with Shared-B grouping and error statistics for weight-type comparisons.
output :
<output_path>/
|-- low_rank_shared.pt
|-- b_ref_map.json
`-- analysis/   (default when --analysis and --analysis_out_dir is not set)
    |-- error_stats_<group>.csv
    |-- fig_hist_loglog_<group>.png
    `-- fig_ccdf_loglog_<group>.png
<analysis_out_dir> (used instead of <output_path>/analysis when provided)
<cov_stats_path> (optional cache .pt)
"""

import os
import re
import json
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




logger = logging.getLogger("RandomizedGSVD_Integrated")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


def load_data(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    logger.info(f"Loaded {len(data)} tensors from {path}")
    return data


def extract_layer_index(name: str) -> str:
    m = re.search(r"layers?\.(\d+)\.", name)
    return m.group(1) if m else "unknown"


SUFFIX_ORDER = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "gate_proj": 0, "up_proj": 1}


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")


def resolve_torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as err:
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype '{name}'") from err






def build_groups(
    err_T: Dict[str, torch.Tensor], model_name: str
) -> Dict[str, List[str]]:
    layer_groups = defaultdict(list)
    layer_dimensions = defaultdict(dict)

    for name in err_T:
        parts = name.split(".")
        if len(parts) < 3:
            continue

        module_name = parts[-2]
        layer_idx = extract_layer_index(name)
        try:
            layer_dimensions[layer_idx][module_name] = tuple(err_T[name].shape)
        except Exception:
            continue

        if module_name in ("q_proj", "k_proj", "v_proj"):
            key = f"layer{layer_idx}_qkv"
            layer_groups[key].append(name)
        elif module_name in ("gate_proj", "up_proj"):
            key = f"layer{layer_idx}_mlp"
            layer_groups[key].append(name)

    for gk, names in layer_groups.items():
        names.sort(key=lambda n: SUFFIX_ORDER.get(n.split(".")[-2], 99))

    logger.info(f"Total groups created for Shared-B processing: {len(layer_groups)}")


    for layer_idx, dims in list(layer_dimensions.items())[:3]:
        if dims:
            logger.info(f"Layer {layer_idx} dimensions ({model_name} ):")
            for module, shape in sorted(dims.items()):
                logger.info(f"  {module}: {shape}")
    return layer_groups






def build_calibration_tokens(
    tokenizer,
    nsamples: int = 64,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = None,
) -> torch.Tensor:
    logger.info(
        f"Building calibration tokens (dataset={dataset_name}, nsamples={nsamples}, seqlen={seqlen})"
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
        logger.warning(
            f"Collected only {len(samples)}/{nsamples} sequences from {dataset_name}."
        )

    return (
        torch.stack(samples, dim=0)
        if samples
        else torch.empty(0, seqlen, dtype=torch.long)
    )





@torch.no_grad()
def estimate_input_covariance(
    model,
    tokenizer,
    device: torch.device,
    model_name: str,
    nsamples: int = 64,
    seqlen: int = 2048,
    alpha: float = 0.05,
    calib_dataset: str = "wikitext",
    calib_config: Optional[str] = None,
    cov_store_device: str = "cpu",
    matmul_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    logger.info(
        f"Collecting input activations for {model_name} (nsamples={nsamples}, seqlen={seqlen})"
    )

    cov_device = torch.device(cov_store_device)

    def stat_slot(dim: int):
        return {
            "xtx": torch.zeros(dim, dim, dtype=torch.float32, device=cov_device),
            "sumx": torch.zeros(dim, dtype=torch.float64, device=cov_device),
            "n": 0,
        }

    stats: Dict[str, dict] = {}
    handles = []

    name_lower = model_name.lower()
    is_opt = "opt" in name_lower
    is_llama_family = any(
        kw in name_lower for kw in ["llama", "vicuna", "opt6b", "qwen", "phi"]
    )

    if is_opt:
        module_to_group_map = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}
    elif is_llama_family:
        module_to_group_map = {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
        }
    else:
        module_to_group_map = {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
            "fc1": "mlp",
            "fc2": "mlp",
        }

    def get_hook(name: str):
        def hook(module, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            dim = x.shape[-1]
            parts = name.split(".")
            module_suffix = parts[-1]
            layer_idx = extract_layer_index(name)
            group_type = module_to_group_map.get(module_suffix)
            key = (
                f"layer{layer_idx}_{group_type}"
                if group_type in ("qkv", "mlp")
                else name
            )

            if key not in stats:
                stats[key] = stat_slot(dim)

            x_mm = x.to(matmul_dtype)
            stats[key]["xtx"].add_((x_mm.T @ x_mm).to(device=cov_device))
            stats[key]["sumx"].add_(
                x.sum(dim=0).to(dtype=torch.float64, device=cov_device)
            )
            stats[key]["n"] += x.shape[0]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(get_hook(name)))

    tokens = build_calibration_tokens(
        tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
        dataset_name=calib_dataset,
        dataset_config=calib_config,
    )

    if tokens.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(f"Failed to build calibration tokens from {calib_dataset}.")

    model.eval()
    for i in tqdm(range(tokens.shape[0]), desc="Calibration Forward Pass"):
        model(tokens[i : i + 1].to(device))

    for h in handles:
        h.remove()

    cov_matrices: Dict[str, torch.Tensor] = {}
    logger.info(f"Calculating covariance matrices with shrinkage (alpha={alpha})")

    for key, slot in stats.items():
        n = max(1, slot["n"])
        cov = slot["xtx"] / n
        dim = cov.shape[0]
        trace_val = torch.trace(cov)
        if trace_val > 0:
            identity = (trace_val / dim) * torch.eye(
                dim, device=cov.device, dtype=cov.dtype
            )
            cov = (1 - alpha) * cov + alpha * identity
        else:
            cov = cov + 1e-6 * torch.eye(dim, device=cov.device, dtype=cov.dtype)
        cov_matrices[key] = cov.cpu()

    logger.info(f"Estimated {len(cov_matrices)} unique covariance matrices.")
    return cov_matrices


def calculate_matrix_sqrt_and_inv_sqrt(
    S: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    S_gpu = S.to(device, dtype=torch.float64)
    L, Q = torch.linalg.eigh(S_gpu)
    L, Q = L.to(torch.float32), Q.to(torch.float32)
    L_sqrt = torch.sqrt(torch.clamp(L, min=1e-8)).diag()
    L_inv_sqrt = (1.0 / torch.sqrt(torch.clamp(L, min=1e-8))).diag()
    S_sqrt = Q @ L_sqrt @ Q.T
    S_inv_sqrt = Q @ L_inv_sqrt @ Q.T
    return S_sqrt.cpu(), S_inv_sqrt.cpu()






def randomized_svd_pytorch(
    M: torch.Tensor,
    rank: int,
    n_oversamples: int = 10,
    n_power_iters: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = M.shape
    max_rank = min(m, n)
    rank = min(rank, max_rank)
    q = min(max_rank, max(rank, rank + n_oversamples))
    Omega = torch.randn(n, q, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(max(0, n_power_iters)):
        Y = M @ (M.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ M
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    return U[:, :rank], S[:rank], Vh[:rank, :]


@torch.no_grad()
def process_randomized_gsvd_group(
    E_list: List[torch.Tensor],
    names: List[str],
    Sigma_sqrt: torch.Tensor,
    Sigma_inv_sqrt: torch.Tensor,
    rank: int,
    device: torch.device,
    n_oversamples: int,
    n_power_iters: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    E_cat = torch.cat(E_list, dim=0).to(device, dtype=torch.float32)
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)

    try:
        Q_e, R_e = torch.linalg.qr(E_cat)
    except torch.linalg.LinAlgError:
        logger.warning(
            "QR decomposition failed. Falling back to standard Right-Weighted SVD."
        )
        return process_weighted_svd_group(
            E_list, names, Sigma_sqrt, Sigma_inv_sqrt, rank, device
        )

    M = R_e @ W_sqrt
    U_m, S_vals, Vh = randomized_svd_pytorch(
        M, rank=rank, n_oversamples=n_oversamples, n_power_iters=n_power_iters
    )

    U = Q_e @ U_m
    S_sqrt_diag = torch.sqrt(S_vals).diag()


    cumulative_rows = 0
    A_final_list = []
    for E_tensor in E_list:
        rows = E_tensor.shape[0]
        U_slice = U[cumulative_rows : cumulative_rows + rows, :]
        A_i = (U_slice @ S_sqrt_diag).cpu()
        A_final_list.append(A_i)
        cumulative_rows += rows


    B_temp = S_sqrt_diag @ Vh
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B_shared_final = (B_temp @ W_inv_sqrt).cpu()

    return A_final_list, B_shared_final


@torch.no_grad()
def process_weighted_svd_group(
    E_list: List[torch.Tensor],
    names: List[str],
    Sigma_sqrt: torch.Tensor,
    Sigma_inv_sqrt: torch.Tensor,
    rank: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    E_tilde_list = [E.to(device, dtype=torch.float32) @ W_sqrt for E in E_list]
    E_tilde_cat = torch.cat(E_tilde_list, dim=0)
    U, S_vals, Vh = torch.linalg.svd(E_tilde_cat, full_matrices=False)

    current_rank = min(rank, len(S_vals))
    S_sqrt_diag = torch.sqrt(S_vals[:current_rank]).diag()

    cumulative_rows, A_final_list = 0, []
    for E_tensor in E_list:
        rows = E_tensor.shape[0]
        U_slice = U[cumulative_rows : cumulative_rows + rows, :current_rank]
        A_final_list.append((U_slice @ S_sqrt_diag).cpu())
        cumulative_rows += rows

    B_temp = S_sqrt_diag @ Vh[:current_rank, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B_shared_final = (B_temp @ W_inv_sqrt).cpu()
    return A_final_list, B_shared_final






def _select_names_by_group(err_T: Dict[str, torch.Tensor], group: str) -> List[str]:
    group = group.upper()
    if group == "ALL":
        return list(err_T.keys())
    sel = []
    for n in err_T:
        last = n.split(".")[-2]
        if group == "ATTN" and last in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sel.append(n)
        elif group == "MLP" and last in ("gate_proj", "up_proj", "fc1", "fc2"):
            sel.append(n)
        elif group == "DOWN" and last in ("down_proj",):
            sel.append(n)
    return sel


def _sample_abs_values_from_err(err_T: Dict[str, torch.Tensor], names: List[str],
                                per_tensor_cap: int, max_total_samples: int,
                                rng: np.random.Generator) -> np.ndarray:
    samples = []
    total = 0
    for name in names:
        t = err_T.get(name)
        if t is None:
            continue
        x = t.detach().abs().reshape(-1)
        n = x.numel()
        if n == 0:
            continue
        k = min(per_tensor_cap, n)
        if k < n:
            idx = torch.from_numpy(rng.choice(n, size=k, replace=False))
            sel = x[idx]
        else:
            sel = x
        arr = sel.cpu().numpy()
        samples.append(arr)
        total += arr.size
        if total >= max_total_samples:
            break
    if not samples:
        return np.array([], dtype=np.float64)
    out = np.concatenate(samples)
    if out.size > max_total_samples:
        out = out[:max_total_samples]
    return out


def _ccdf(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return np.array([]), np.array([])
    y = np.sort(y)
    n = y.size
    ranks = np.arange(1, n + 1)
    ccdf = 1.0 - ranks / (n + 1.0)
    return y, ccdf


def _fit_powerlaw_tail(x: np.ndarray, c: np.ndarray, tail_pct: float) -> Tuple[float, float]:

    if x.size == 0:
        return float("nan"), float("nan")
    p = max(0.0, min(100.0, tail_pct))
    thr_idx = int(np.floor(x.size * (p / 100.0)))
    thr_idx = max(1, min(thr_idx, x.size - 1))
    x_tail = x[thr_idx:]
    c_tail = c[thr_idx:]
    x_tail = x_tail[c_tail > 0]
    c_tail = c_tail[c_tail > 0]
    if x_tail.size < 10:
        return float("nan"), float("nan")
    lx = np.log10(x_tail)
    ly = np.log10(c_tail)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def _basic_stats(y: np.ndarray) -> Dict[str, float]:
    if y.size == 0:
        return {k: float("nan") for k in ["n","mean","std","median","p90","p95","p99","max"]}
    return {
        "n": int(y.size),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "median": float(np.median(y)),
        "p90": float(np.percentile(y, 90)),
        "p95": float(np.percentile(y, 95)),
        "p99": float(np.percentile(y, 99)),
        "max": float(np.max(y)),
    }


def run_error_stats_and_plots(
    label_to_err: Dict[str, Dict[str, torch.Tensor]],
    out_dir: str,
    groups: List[str],
    bins: int,
    tail_pct: float,
    per_tensor_cap: int,
    max_total_samples: int,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for group in groups:

        samples_per_label: Dict[str, np.ndarray] = {}
        for label, err_T in label_to_err.items():
            names = _select_names_by_group(err_T, group)
            arr = _sample_abs_values_from_err(err_T, names, per_tensor_cap, max_total_samples, rng)
            samples_per_label[label] = arr


        csv_path = os.path.join(out_dir, f"error_stats_{group.lower()}.csv")
        with open(csv_path, "w") as f:
            f.write("label,n,mean,std,median,p90,p95,p99,max,tail_pct,tail_slope,tail_r2\n")
            for label, arr in samples_per_label.items():
                x, c = _ccdf(arr)
                slope, r2 = _fit_powerlaw_tail(x, c, tail_pct)
                st = _basic_stats(arr)
                f.write(
                    f"{label},{st['n']},{st['mean']:.6g},{st['std']:.6g},{st['median']:.6g},{st['p90']:.6g},{st['p95']:.6g},{st['p99']:.6g},{st['max']:.6g},{tail_pct:.1f},{slope:.6g},{r2:.6g}\n"
                )


        plt.figure(figsize=(7,5))
        for label, arr in samples_per_label.items():
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size == 0:
                continue

            lo, hi = np.min(arr), np.max(arr)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0:
                continue
            edges = np.geomspace(lo, hi, num=max(5, bins))
            hist, e = np.histogram(arr, bins=edges, density=True)
            centers = np.sqrt(e[:-1]*e[1:])
            plt.plot(centers, hist + 1e-20, label=label)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("|E| (abs error)"); plt.ylabel("Density")
        plt.title(f"Error Histogram (log-log), group={group}")
        plt.legend(); plt.tight_layout()
        fig_h = os.path.join(out_dir, f"fig_hist_loglog_{group.lower()}.png")
        plt.savefig(fig_h, dpi=160); plt.close()


        plt.figure(figsize=(7,5))
        for label, arr in samples_per_label.items():
            x, c = _ccdf(arr)
            if x.size == 0:
                continue
            plt.plot(x, c + 1e-30, label=label)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("|E| (abs error)"); plt.ylabel("CCDF (1-CDF)")
        plt.title(f"Error CCDF (log-log), group={group}")
        plt.legend(); plt.tight_layout()
        fig_c = os.path.join(out_dir, f"fig_ccdf_loglog_{group.lower()}.png")
        plt.savefig(fig_c, dpi=160); plt.close()

        logger.info(f"[Analysis:{group}] saved: {fig_h}, {fig_c}, {csv_path}")






def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    err_T_main = load_data(args.err_path)


    if args.analysis:
        label_to_err: Dict[str, Dict[str, torch.Tensor]] = {}
        main_label = args.analysis_label or "MAIN"
        label_to_err[main_label] = err_T_main
        if args.compare_err_paths:
            if args.compare_labels and len(args.compare_labels) != len(args.compare_err_paths):
                raise ValueError("--compare_labels count must match --compare_err_paths count.")
            for i, pth in enumerate(args.compare_err_paths):
                lab = args.compare_labels[i] if args.compare_labels else f"ERR{i+1}"
                label_to_err[lab] = load_data(pth)
        analysis_out = args.analysis_out_dir or os.path.join(args.output_path, "analysis")
        groups = [g.strip().upper() for g in (args.analysis_groups or ["ALL"]) ]
        run_error_stats_and_plots(
            label_to_err=label_to_err,
            out_dir=analysis_out,
            groups=groups,
            bins=args.analysis_bins,
            tail_pct=args.analysis_tail_pct,
            per_tensor_cap=args.analysis_per_tensor_cap,
            max_total_samples=args.analysis_max_total_samples,
            seed=args.analysis_seed,
        )
        logger.info(f"Empirical error statistics & scaling plots saved to: {analysis_out}")


    logger.info(f"Loading model '{args.model_name}' for covariance estimation...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    matmul_dtype = resolve_torch_dtype(args.matmul_dtype)


    if (
        args.reuse_cov_stats and args.cov_stats_path and os.path.isfile(args.cov_stats_path)
    ):
        logger.info(f"Loading cached covariance statistics from {args.cov_stats_path}")
        cov_matrices = torch.load(args.cov_stats_path, map_location="cpu")
    else:
        if args.reuse_cov_stats and args.cov_stats_path:
            logger.info(
                f"Cached covariance statistics not found at {args.cov_stats_path}; recomputing."
            )
        cov_matrices = estimate_input_covariance(
            model,
            tokenizer,
            device,
            args.model_name,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            alpha=args.shrinkage_alpha,
            calib_dataset=args.calib_dataset,
            calib_config=args.calib_config,
            cov_store_device=args.cov_store_device,
            matmul_dtype=matmul_dtype,
        )
        if args.cov_stats_path:
            os.makedirs(os.path.dirname(args.cov_stats_path) or ".", exist_ok=True)
            torch.save({k: v.cpu() for k, v in cov_matrices.items()}, args.cov_stats_path)

    del model
    torch.cuda.empty_cache()


    sqrt_matrices = {}
    for name, cov in tqdm(cov_matrices.items(), desc="Calculating Matrix Square Roots"):
        sqrt_matrices[name] = calculate_matrix_sqrt_and_inv_sqrt(cov, device)


    layer_groups = build_groups(err_T_main, args.model_name)

    shared: Dict[str, torch.Tensor] = {}
    b_ref_map: Dict[str, str] = {}

    logger.info(
        f"Starting Randomized GSVD processing for rank={args.max_rank} on {args.model_name}..."
    )


    for gk, names in tqdm(layer_groups.items(), desc="Processing Shared-B Groups with Randomized GSVD"):
        if gk not in sqrt_matrices:
            logger.warning(f"Covariance matrix not found for group {gk}. Skipping.")
            continue

        Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[gk]
        E_list = [err_T_main[n] for n in names]

        A_list, B_shared = process_randomized_gsvd_group(
            E_list,
            names,
            Sigma_sqrt,
            Sigma_inv_sqrt,
            args.max_rank,
            device,
            args.oversamples,
            args.power_iters,
        )

        b_key_shared = f"{gk}.B_shared"
        shared[b_key_shared] = B_shared.to(torch.float16)

        for i, name in enumerate(names):
            module_suffix = name.split(".")[-2]
            a_key = f"{gk}.{module_suffix}.A"
            shared[a_key] = A_list[i].to(torch.float16)
            b_ref_map[name] = b_key_shared


    grouped_names = {n for names in layer_groups.values() for n in names}
    solo_names = sorted([n for n in err_T_main if n not in grouped_names and "layers" in n])

    for name in tqdm(solo_names, desc="Processing Solo Layers with Randomized GSVD"):
        module_name = name.replace(".weight", "")
        if module_name not in sqrt_matrices:
            logger.warning(
                f"Covariance matrix not found for solo layer {module_name}. Skipping."
            )
            continue

        Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[module_name]
        A_list, B = process_randomized_gsvd_group(
            [err_T_main[name]],
            [name],
            Sigma_sqrt,
            Sigma_inv_sqrt,
            args.max_rank,
            device,
            args.oversamples,
            args.power_iters,
        )

        a_key, b_key = f"{module_name}.A", f"{module_name}.B"
        shared[a_key] = A_list[0].to(torch.float16)
        shared[b_key] = B.to(torch.float16)
        b_ref_map[name] = b_key


    os.makedirs(args.output_path, exist_ok=True)
    torch.save(shared, os.path.join(args.output_path, "low_rank_shared.pt"))
    with open(os.path.join(args.output_path, "b_ref_map.json"), "w") as f:
        json.dump(b_ref_map, f, indent=2)

    logger.info(f"\nSaved artifacts to {args.output_path}")
    logger.info(f"  - low_rank_shared.pt: {len(shared)} tensors")
    logger.info(f"  - b_ref_map.json: {len(b_ref_map)} mappings")
    logger.info(f"\nRandomized GSVD processing complete for {args.model_name}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "STEP 2 (Integrated) - Randomized GSVD with Shared-B Grouping + Error Stats"
    )

    p.add_argument("--model_name", type=str, required=True, help="HF model name for covariance estimation.")
    p.add_argument("--err_path", type=str, required=True, help="Path to quantization error dictionary from step1.")
    p.add_argument("--output_path", type=str, required=True, help="Directory to save artifacts.")
    p.add_argument("--trust_remote_code", action="store_true", help="Set for models like Qwen requiring custom code.")
    p.add_argument("--max_rank", type=int, default=64, help="Maximum rank for SVD.")


    p.add_argument("--shrinkage_alpha", type=float, default=0.05, help="Alpha for covariance shrinkage.")
    p.add_argument("--nsamples", type=int, default=64, help="# calibration samples.")
    p.add_argument("--seqlen", type=int, default=2048, help="Sequence length for calibration tokens.")
    p.add_argument("--calib_dataset", type=str, default="wikitext", help="HF datasets id for calibration.")
    p.add_argument("--calib_config", type=str, default=None, help="Optional config name for dataset.")
    p.add_argument("--cov_store_device", type=str, default="cpu", help="Device for XtX accumulation (cpu|cuda).")
    p.add_argument("--oversamples", type=int, default=10, help="Oversampling for randomized SVD.")
    p.add_argument("--power_iters", type=int, default=2, help="Power iteration count for randomized SVD.")
    p.add_argument("--cov_stats_path", type=str, default=None, help="Path to cache covariance statistics (XtX).")
    p.add_argument("--reuse_cov_stats", type=str2bool, default=False, help="Reuse cached covariance statistics.")
    p.add_argument("--matmul_dtype", type=str, default="float32", help="Torch dtype for XtX accumulation.")




    p.add_argument("--analysis", action="store_true", help="Run empirical error statistics & scaling-law plots.")
    p.add_argument("--analysis_label", type=str, default=None, help="Label for --err_path (e.g., W4A16).")
    p.add_argument("--compare_err_paths", type=str, nargs="*", default=None, help="Extra err.pt files to compare.")
    p.add_argument("--compare_labels", type=str, nargs="*", default=None, help="Labels for --compare_err_paths.")
    p.add_argument("--analysis_out_dir", type=str, default=None, help="Output dir for analysis (default: output_path/analysis).")
    p.add_argument("--analysis_groups", type=str, nargs="*", default=["ALL"], help="Groups to analyze: ALL, ATTN, MLP, DOWN")
    p.add_argument("--analysis_bins", type=int, default=128, help="#bins for log-spaced histogram.")
    p.add_argument("--analysis_tail_pct", type=float, default=95.0, help="Tail percentile for log-log linear fit (power-law).")
    p.add_argument("--analysis_per_tensor_cap", type=int, default=200_000, help="Max samples per tensor to draw.")
    p.add_argument("--analysis_max_total_samples", type=int, default=2_000_000, help="Global max samples (per group×label).")
    p.add_argument("--analysis_seed", type=int, default=42, help="Random seed for sampling.")

    args = p.parse_args()
    main(args)

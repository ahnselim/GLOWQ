"""
moe_ffn/similarity_heatmap_moe.py
Analyzes shared-B representativeness for dense and MoE FFN groups and plots similarity heatmaps.
output :
<fig_out_dir or <output_path>/analysis_figs>/
|-- {group_key}_crossbasis_{mod_label}_{suffix}.png
`-- {group_key}_angles_bars_{suffix}.png
"""

import os
import re
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger("RandomizedGSVD_Integrated")
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_ch.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_ch)


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ("1", "true", "yes", "y", "on"):
        return True
    if x in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("bool expected")


def resolve_torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as e:
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype '{name}'") from e


def load_data(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    logger.info(f"Loaded {len(data)} tensors from {path}")
    return data


def extract_layer_index(name: str) -> str:
    m = re.search(r"layers?\.(\d+)\.", name)
    return m.group(1) if m else "unknown"


def extract_expert_index(name: str) -> Optional[str]:

    m = re.search(r"experts\.(\d+)\.", name)
    return m.group(1) if m else None



SUFFIX_ORDER = {

    "q_proj": 0, "k_proj": 1, "v_proj": 2,

    "gate_proj": 0, "up_proj": 1,
}


def get_group_key_and_role(name: str, model_name: str) -> Tuple[Optional[str], Optional[str]]:

    parts = name.split(".")
    if len(parts) < 2:
        return None, None

    module = parts[-2].lower()
    lid_str = extract_layer_index(name)
    if lid_str == "unknown":
        return None, None

    if module in {"q_proj", "k_proj", "v_proj"}:
        gkey = f"layer{lid_str}_qkv"
        return gkey, module

    if module in {"gate_proj", "up_proj"}:
        gkey = f"layer{lid_str}_mlp"
        return gkey, module

    return None, None


def build_groups(err_T: Dict[str, torch.Tensor], model_name: str) -> Dict[str, List[str]]:

    from collections import defaultdict

    layer_groups = defaultdict(list)
    for name in err_T:
        gkey, role = get_group_key_and_role(name, model_name)
        if gkey is None:
            continue
        layer_groups[gkey].append(name)


    for gk, names in layer_groups.items():
        def sort_key(n):
            mod = n.split(".")[-2].lower()
            return SUFFIX_ORDER.get(mod, 99)
        names.sort(key=sort_key)

    logger.info(f"Total groups created for Shared-B processing: {len(layer_groups)}")
    return dict(layer_groups)


def parse_layers(spec: str) -> Optional[set]:
    if spec is None or spec.strip() == "":
        return None
    out = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(token))
    return out


def build_calibration_tokens(tokenizer,
                             nsamples=64,
                             seqlen=2048,
                             dataset_name="wikitext",
                             dataset_config=None):
    logger.info(f"Building calibration tokens (dataset={dataset_name}, nsamples={nsamples}, seqlen={seqlen})")
    try:
        ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
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

    buf, samples = [], []
    for row in iterator:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if not ids:
            continue
        if eos_id is not None:
            ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
            if len(samples) >= nsamples:
                break
        if len(samples) >= nsamples:
            break

    if len(samples) < nsamples:
        logger.warning(f"Collected only {len(samples)}/{nsamples} sequences from {dataset_name}.")
    return torch.stack(samples, 0) if samples else torch.empty(0, seqlen, dtype=torch.long)


@torch.no_grad()
def estimate_input_covariance(model,
                              tokenizer,
                              device,
                              model_name,
                              nsamples=64,
                              seqlen=2048,
                              alpha=0.05,
                              calib_dataset="wikitext",
                              calib_config=None,
                              cov_store_device="cpu",
                              matmul_dtype=torch.float32) -> Dict[str, torch.Tensor]:

    logger.info(f"Collecting input activations for {model_name} (nsamples={nsamples}, seqlen={seqlen})")
    cov_device = torch.device(cov_store_device)

    def slot(d):
        return {
            "xtx": torch.zeros(d, d, dtype=torch.float32, device=cov_device),
            "sumx": torch.zeros(d, dtype=torch.float64, device=cov_device),
            "n": 0,
        }

    stats: Dict[str, dict] = {}
    handles = []

    name_lower = model_name.lower()
    is_opt = "opt" in name_lower

    is_llama_family = any(
        kw in name_lower for kw in ["llama", "vicuna", "qwen", "phi", "mixtral", "mistral"]
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

    def make_hook(name: str):
        def hook(module, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            d = x.shape[-1]
            parts = name.split(".")
            module_suffix = parts[-1]
            layer_idx = extract_layer_index(name)
            gtype = module_to_group_map.get(module_suffix)
            key = f"layer{layer_idx}_{gtype}" if gtype in ("qkv", "mlp") else None
            if key is None:
                return

            if key not in stats:
                stats[key] = slot(d)

            xm = x.to(matmul_dtype)
            stats[key]["xtx"].add_((xm.T @ xm).to(cov_device))
            stats[key]["sumx"].add_(x.sum(dim=0, dtype=torch.float64).to(device=cov_device))
            stats[key]["n"] += x.shape[0]
        return hook


    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    toks = build_calibration_tokens(tokenizer, nsamples, seqlen, calib_dataset, calib_config)
    if toks.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(f"Failed to build calibration tokens from {calib_dataset}.")

    model.eval()
    for i in tqdm(range(toks.shape[0]), desc="Calibration Forward Pass"):
        model(toks[i : i + 1].to(device))

    for h in handles:
        h.remove()

    covs = {}
    logger.info(f"Calculating covariance matrices with shrinkage (alpha={alpha})")
    for key, slot in stats.items():
        n = max(1, slot["n"])
        C = slot["xtx"] / n
        d = C.shape[0]
        tr = torch.trace(C)
        if tr > 0:
            I = (tr / d) * torch.eye(d, device=C.device, dtype=C.dtype)
            C = (1 - alpha) * C + alpha * I
        else:
            C = C + 1e-6 * torch.eye(d, device=C.device, dtype=C.dtype)
        covs[key] = C.cpu()

    logger.info(f"Estimated {len(covs)} unique covariance matrices.")
    return covs


def matrix_sqrt(S: torch.Tensor, device) -> torch.Tensor:
    Sg = S.to(device, dtype=torch.float64)
    L, Q = torch.linalg.eigh(Sg)
    L = torch.clamp(L, min=1e-8)
    return (Q @ torch.diag(torch.sqrt(L)) @ Q.T).to(torch.float32).cpu()


def orth_basis_from_rows(M: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(M.T)
    return Q[:, : M.shape[0]]


def U_shared_whitened(B_shared: torch.Tensor,
                      S_half: torch.Tensor,
                      device) -> torch.Tensor:
    return orth_basis_from_rows(
        B_shared.to(device=device, dtype=torch.float32)
        @ S_half.to(device=device, dtype=torch.float32)
    ).cpu()


def Ui_from_E_whitened(E_i: torch.Tensor,
                       S_half: torch.Tensor,
                       r: int,
                       device) -> torch.Tensor:
    Etil = E_i.to(device=device, dtype=torch.float32) @ S_half.to(device=device, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(Etil, full_matrices=False)
    return Vh[:r, :].T.cpu()


def principal_angles(Ua: torch.Tensor,
                     Ub: torch.Tensor):
    Ua64 = Ua.detach().cpu().to(torch.float64)
    Ub64 = Ub.detach().cpu().to(torch.float64)
    W = Ua64.T @ Ub64
    s = torch.linalg.svdvals(W).clamp(0, 1).cpu().numpy()
    theta = np.arccos(s)
    return s, theta


def hungarian_reorder(A: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
        r_ind, c_ind = linear_sum_assignment(-A)
        P = np.zeros_like(A)
        P[r_ind, c_ind] = 1.0
        return P
    except Exception:
        r = A.shape[0]
        cols = set(range(r))
        P = np.zeros_like(A)
        for i in range(r):
            j = max(cols, key=lambda c: A[i, c])
            P[i, j] = 1.0
            cols.remove(j)
        return P


def module_label_from_name(name: str) -> str:

    eid = extract_expert_index(name)
    module = name.split(".")[-2]
    if eid is not None:
        return f"expert{eid}_{module}"
    if "shared_expert" in name:
        return f"shared_{module}"
    return module


def plot_cross_basis_heatmap(group_key: str,
                             mod_label: str,
                             Ush: torch.Tensor,
                             Ui: torch.Tensor,
                             out_dir: str,
                             suffix: str,
                             axis_tick_step: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    C = (Ush.cpu().T @ Ui.cpu()).abs().numpy()
    P = hungarian_reorder(C)
    C_sorted = C @ P
    r = C_sorted.shape[0]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(C_sorted, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"|U^T V| (Hungarian) — {group_key} / {mod_label} [{suffix}]")
    ax.set_xlabel(f"{mod_label} basis index (sorted) →")
    ax.set_ylabel("shared-B basis index (sorted) →")

    step = max(1, axis_tick_step)
    ticks = list(range(0, r, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=0)
    ax.set_yticklabels([str(t) for t in ticks])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|cos| between basis vectors")

    fig.tight_layout()
    fp = os.path.join(out_dir, f"{group_key}_crossbasis_{mod_label}_{suffix}.png")
    fig.savefig(fp, dpi=220)
    plt.close(fig)


def plot_angle_bars(group_key: str,
                    mod_labels: List[str],
                    thetamax: List[float],
                    projres: List[float],
                    out_dir: str,
                    suffix: str):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(mod_labels))
    fig, ax = plt.subplots(1, 2, figsize=(max(8.0, len(mod_labels) * 0.25), 4.0))

    ax[0].bar(x, thetamax)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(mod_labels, rotation=90)
    ax[0].set_ylabel("θ_max (deg)")
    ax[0].set_title(f"Principal Angle (max) — {group_key} [{suffix}]")

    ax[1].bar(x, projres)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(mod_labels, rotation=90)
    ax[1].set_ylabel("ProjResidual = mean(sin^2 θ)")
    ax[1].set_title(f"Projector Residual — {group_key} [{suffix}]")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{group_key}_angles_bars_{suffix}.png"), dpi=220)
    plt.close(fig)


def run_representativeness_analysis(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shared_path = os.path.join(args.output_path, "low_rank_shared.pt")
    if not os.path.isfile(shared_path):
        raise FileNotFoundError(f"Shared state not found: {shared_path}")
    shared = torch.load(shared_path, map_location="cpu")

    err_T = load_data(args.err_path)
    groups_dict = build_groups(err_T, args.model_name)


    groups_all = sorted([k.replace(".B_shared", "") for k in shared if k.endswith(".B_shared")])
    logger.info(f"[Analysis] Raw target groups from shared: {groups_all}")
    rgx = re.compile(args.groups_regex)
    groups = [g for g in groups_all if rgx.fullmatch(g)]

    layer_set = parse_layers(args.layers)
    if layer_set is not None:
        groups = [g for g in groups if int(re.search(r"layer(\d+)_", g).group(1)) in layer_set]


    final_groups = []
    for g in groups:
        if g not in groups_dict:
            logger.warning(f"[Analysis] Group {g} not found in err_T; skip for Σx/analysis.")
            continue
        final_groups.append(g)

    if not final_groups:
        raise RuntimeError("No groups to analyze after filtering (shared_B ∩ err_T is empty).")
    logger.info(f"[Analysis] Final target groups (with err & modules): {final_groups}")


    need_sigma = args.whiten_for_analysis or args.compare_whitening
    Sigma_by_group = {}
    if need_sigma:
        logger.info("[Analysis] Estimating Sigma_x for whitening pipeline...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        covs = estimate_input_covariance(
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
            matmul_dtype=resolve_torch_dtype(args.matmul_dtype),
        )
        del model
        torch.cuda.empty_cache()

        for g in final_groups:
            if g in covs:
                Sigma_by_group[g] = covs[g]
            else:
                logger.warning(f"[Analysis] Sigma_x missing for {g}; using identity.")


    if args.compare_whitening:
        modes = [("nowhiten", False), ("whiten", True)]
    else:
        modes = [("whiten", True)] if args.whiten_for_analysis else [("nowhiten", False)]

    os.makedirs(args.fig_out_dir, exist_ok=True)

    for suffix, use_whiten in modes:
        logger.info(f"[Analysis] Running mode: {suffix}")
        for g in final_groups:
            bkey = f"{g}.B_shared"
            if bkey not in shared:
                logger.warning(f"[Analysis] Missing {bkey}; skip.")
                continue

            B = shared[bkey].float()
            d = B.shape[1]
            if use_whiten and g in Sigma_by_group:
                S_half = matrix_sqrt(Sigma_by_group[g], torch.device(device))
            else:
                S_half = torch.eye(d, dtype=torch.float32)

            Ush = U_shared_whitened(B, S_half, device)

            if g not in groups_dict:
                logger.warning(f"[Analysis] Group {g} not found in err_T (second pass); skip.")
                continue

            mod_keys = groups_dict[g]
            mod_labels = [module_label_from_name(mk) for mk in mod_keys]
            r = B.shape[0]
            theta_max_list = []
            proj_res_list = []

            for name, label in zip(mod_keys, mod_labels):
                Ei = err_T[name].float()
                Ui = Ui_from_E_whitened(Ei, S_half, r, device)
                s, th = principal_angles(Ush, Ui)
                theta_max_list.append(float(np.degrees(th).max()))
                proj_res_list.append(1.0 - float((s ** 2).mean()))
                plot_cross_basis_heatmap(
                    g, label, Ush, Ui, args.fig_out_dir,
                    suffix=suffix, axis_tick_step=args.axis_tick_step
                )

            plot_angle_bars(
                g, mod_labels, theta_max_list, proj_res_list,
                args.fig_out_dir, suffix=suffix
            )

    logger.info("[Analysis] Intra-group representativeness plots saved.")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _ = device


    if not args.analysis_only:
        logger.warning("Decomposition (GSVD) part is not implemented in this script. please use analysis_only=True.")

    if args.do_analysis != "none":
        os.makedirs(args.fig_out_dir, exist_ok=True)
        run_representativeness_analysis(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "STEP 2 (Analysis, Dense + MoE) - Shared-B Intra-Group Representativeness"
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--err_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")


    p.add_argument("--analysis_only", type=str2bool, default=True)
    p.add_argument("--do_analysis", type=str, choices=["none", "all"], default="all")

    p.add_argument(
        "--groups_regex",
        type=str,
        default=r"layer\d+_(qkv|mlp|expert\d+_moe)"
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="e.g., '0,10,12-15' to pick specific layers"
    )
    p.add_argument("--max_groups", type=int, default=9999)


    p.add_argument("--whiten_for_analysis", type=str2bool, default=True)
    p.add_argument(
        "--compare_whitening",
        type=str2bool,
        default=False,
        help="plot both whitened and unwhitened results"
    )
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib_dataset", type=str, default="wikitext")
    p.add_argument("--calib_config", type=str, default=None)
    p.add_argument("--shrinkage_alpha", type=float, default=0.05)
    p.add_argument("--cov_store_device", type=str, default="cpu")
    p.add_argument("--matmul_dtype", type=str, default="float32")


    p.add_argument("--fig_out_dir", type=str, default=None)
    p.add_argument("--axis_tick_step", type=int, default=8)

    args = p.parse_args()
    if args.fig_out_dir is None:
        args.fig_out_dir = os.path.join(args.output_path, "analysis_figs")
    main(args)

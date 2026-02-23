"""
error_capture_compare.py
Compares captured error matrices with input-spectrum-weighted variants and visualizes layer-level error behavior.
output :
<out_dir>/
|-- input_spectrum_linear_{qkv|mlp}_layer{layer_idx:02d}.csv
|-- input_spectrum_linear_{qkv|mlp}_layer{layer_idx:02d}.png
|-- error_energy_compare_{qkv|mlp}_layer{layer_idx:02d}.png
|-- error_energy_compare_{qkv|mlp}_layer{layer_idx:02d}.csv
|-- minrank_error_compare_{qkv|mlp}_layer{layer_idx:02d}.csv
|-- input_spectrum_linear_{qkv|mlp}_layer{layer_idx:02d}_N{N}.csv   (when --nsamples_sweep)
|-- input_spectrum_linear_{qkv|mlp}_layer{layer_idx:02d}_N{N}.png   (when --nsamples_sweep)
|-- error_energy_compare_nsamples_{qkv|mlp}_layer{layer_idx:02d}.png (when --nsamples_sweep)
|-- error_energy_compare_nsamples_{qkv|mlp}_layer{layer_idx:02d}.csv (when --nsamples_sweep)
`-- minrank_error_compare_nsamples_{qkv|mlp}_layer{layer_idx:02d}.csv (when --nsamples_sweep)
"""

import os, re, argparse, logging, warnings, copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("ErrCaptureCompare")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(sh)



def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_err_2d(path: str) -> Dict[str, torch.Tensor]:
    data = safe_torch_load(path)
    if (
        isinstance(data, dict)
        and "err_dict" in data
        and isinstance(data["err_dict"], dict)
    ):
        data = data["err_dict"]
    out = {k: v for k, v in data.items() if torch.is_tensor(v) and v.ndim == 2}
    logger.info(f"[load] {len(out)} matrices loaded from {path}")
    return out



def model_family_flags(model_name: str) -> Tuple[bool, bool]:
    name = model_name.lower()
    return (
        "opt" in name,
        any(x in name for x in ["llama", "vicuna", "mistral", "qwen"]),
    )


def collect_group_names_for_layer(
    err_T: Dict[str, torch.Tensor], layer_idx: int, is_opt: bool, is_llama: bool
):
    attn_suffixes = ("q_proj", "k_proj", "v_proj")
    mlp_suffixes = (
        ("gate_proj", "up_proj")
        if is_llama
        else (("fc1", "fc2") if is_opt else ("gate_proj", "up_proj"))
    )
    attn = [
        k
        for k in err_T
        if f".layers.{layer_idx}." in k and k.split(".")[-2] in attn_suffixes
    ]
    attn_order = ["q_proj", "k_proj", "v_proj"]
    attn.sort(
        key=lambda n: (
            attn_order.index(n.split(".")[-2]) if n.split(".")[-2] in attn_order else 99
        )
    )
    mlp = [
        k
        for k in err_T
        if f".layers.{layer_idx}." in k and k.split(".")[-2] in mlp_suffixes
    ]
    mlp_order = [mlp_suffixes[0], mlp_suffixes[1]]
    mlp.sort(
        key=lambda n: (
            mlp_order.index(n.split(".")[-2]) if n.split(".")[-2] in mlp_order else 99
        )
    )
    return attn, mlp


def build_tokens(
    tok, nsamples=64, seqlen=2048, dataset="DKYoon/SlimPajama-6B", config=None
):
    if nsamples <= 0:
        raise ValueError("nsamples must be > 0")
    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    take = ds.take(max(nsamples * 5, 1))
    eos = tok.eos_token_id or tok.pad_token_id
    if eos is None and getattr(tok, "eos_token", None):
        eos = tok.convert_tokens_to_ids(tok.eos_token)
    buf, out = [], []
    for row in take:
        txt = (row.get("text") or "").strip()
        if not txt:
            continue
        ids = (
            tok(txt, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        if not ids:
            continue
        if eos is not None:
            ids.append(eos)
        buf.extend(ids)
        while len(buf) >= seqlen and len(out) < nsamples:
            out.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
            if len(out) >= nsamples:
                break
        if len(out) >= nsamples:
            break
    if len(out) < nsamples:
        logger.warning(f"[tokens] collected {len(out)}/{nsamples}")
    return torch.stack(out, dim=0)



@torch.no_grad()
def stream_xtx_sumx_sweep(
    model,
    tokens: torch.Tensor,
    device,
    model_name,
    layer_idx: int,
    ns_list: List[int],
    matmul_dtype=torch.float32,
    store_device="cpu",
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:

    store_dev = torch.device(store_device)
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    snapshots: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    target_ns = sorted(set(int(n) for n in ns_list if n > 0))
    maxN = target_ns[-1]

    def slot(d: int):
        return {
            "xtx": torch.zeros(d, d, dtype=torch.float32, device=store_dev),
            "sumx": torch.zeros(d, dtype=torch.float64, device=store_dev),
            "n": 0,
        }

    is_opt, is_llama = model_family_flags(model_name)
    if is_llama:
        m2g = {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
        }
    elif is_opt:
        m2g = {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "fc1": "mlp",
            "fc2": "mlp",
        }
    else:
        m2g = {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
        }

    layer_pat = f".layers.{layer_idx}."

    def get_hook(name):
        def hook(mod, inp, out):
            if layer_pat not in name:
                return
            suf = name.split(".")[-1]
            g = m2g.get(suf)
            if g not in ("qkv", "mlp"):
                return
            key = f"layer{layer_idx}_{g}"
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            d = x.shape[-1]
            if key not in stats:
                stats[key] = slot(d)
            x32 = x.to(matmul_dtype)
            stats[key]["xtx"].add_((x32.T @ x32).to(store_dev))
            stats[key]["sumx"].add_(
                x.sum(dim=0).to(dtype=torch.float64, device=store_dev)
            )
            stats[key]["n"] += x.shape[0]

        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and layer_pat in name:
            handles.append(mod.register_forward_hook(get_hook(name)))

    model.eval()
    total = min(maxN, tokens.shape[0])
    next_k_idx = 0
    for i in tqdm(range(total), desc="Forward for activations (sweep)"):
        model(tokens[i : i + 1].to(device))

        while next_k_idx < len(target_ns) and (i + 1) == target_ns[next_k_idx]:
            N = target_ns[next_k_idx]

            snap: Dict[str, Dict[str, torch.Tensor]] = {}
            for k, v in stats.items():
                snap[k] = {
                    "xtx": v["xtx"].clone(),
                    "sumx": v["sumx"].clone(),
                    "n": int(v["n"]),
                }
            snapshots[N] = snap
            next_k_idx += 1

    for h in handles:
        h.remove()
    return snapshots



def eigvals_desc(S: torch.Tensor, device) -> np.ndarray:
    S = S.to(device, dtype=torch.float32)
    L, _ = torch.linalg.eigh(S)
    L = torch.clamp(L, min=1e-12)
    vals, _ = torch.sort(L, descending=True)
    return vals.detach().cpu().numpy()


def cov_from_stats(slot: Dict[str, torch.Tensor], center: bool):
    N = max(1, int(slot["n"]))
    XtX = slot["xtx"]
    sumx = slot["sumx"].to(torch.float32).view(-1, 1)
    if center:
        return (XtX - float(N) * (sumx @ sumx.T)) / float(max(1, N - 1))
    else:
        return XtX / float(N)


def sqrtm_from_cov(S: torch.Tensor, device) -> torch.Tensor:
    L, Q = torch.linalg.eigh(S.to(device, dtype=torch.float32))
    L = torch.clamp(L, min=1e-12)
    Lsqrt = torch.diag(torch.sqrt(L))
    return (Q @ Lsqrt @ Q.T).to(torch.float32)


def sv_energy_curve(
    E: torch.Tensor, topk: Optional[int] = None, device=None
) -> np.ndarray:

    E = E.to(device or E.device, dtype=torch.float32)
    s = torch.linalg.svdvals(E)
    s2 = (s * s).detach().cpu().numpy()
    s2.sort()
    s2 = s2[::-1]
    if topk is not None:
        s2 = s2[:topk]
    return np.cumsum(s2) / (np.sum(s2) + 1e-12)


def plot_energy_compare_overlay(
    ec_map_no, ec_map_cov, title, out_png, topk=None, thresholds=(0.9, 0.95)
):

    plt.figure(figsize=(7.8, 4.6))
    Ns = sorted(ec_map_cov.keys())
    for N in Ns:
        ec_cov = ec_map_cov[N]
        ec_no = ec_map_no.get(N)
        R = len(ec_cov) if ec_no is None else min(len(ec_cov), len(ec_no))
        x = np.arange(1, R + 1)
        plt.plot(x, ec_cov[:R], label=f"cov, N={N}")
        if ec_no is not None:
            plt.plot(x, ec_no[:R], linestyle="--", label=f"no-cov, N={N}")
    for thr in thresholds:
        plt.axhline(thr, linestyle=":", linewidth=1)
    plt.xlabel("rank r")
    plt.ylabel("error capture ≤ r (∥·∥_F^2)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def plot_input_spectrum_linear(vals, title, out_png, topk=None):
    y = vals[:topk] if topk else vals
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(x, y)
    plt.xlabel("eigenvalue rank r")
    plt.ylabel("eigenvalue (linear)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def min_rank(ec, thr):
    idx = np.where(ec >= thr)[0]
    return int(idx[0] + 1) if idx.size else -1



def main():
    ap = argparse.ArgumentParser(
        "Error capture: E vs E Σ^{1/2} + input spectrum (linear)"
    )
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--err_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--layer_idx", type=int, default=0)
    ap.add_argument(
        "--nsamples",
        type=int,
        default=256,
        help="Used when --nsamples_sweep is not set",
    )
    ap.add_argument(
        "--nsamples_sweep",
        type=int,
        nargs="*",
        default=None,
        help="Provide multiple Ns to sweep",
    )
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib_dataset", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--calib_config", type=str, default=None)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--energy_thresholds", type=float, nargs="*", default=[0.90, 0.95])
    ap.add_argument(
        "--center",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=False,
        help="True: covariance; False: uncentered second moment (X^T X / N)",
    )
    ap.add_argument(
        "--cov_store_device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    ap.add_argument("--use_tf32", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high" if args.use_tf32 else "highest")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    err_T = load_err_2d(args.err_path)
    is_opt, is_llama = model_family_flags(args.model_name)
    attn_names, mlp_names = collect_group_names_for_layer(
        err_T, args.layer_idx, is_opt, is_llama
    )


    logger.info("[hf] loading model/tokenizer …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )


    if args.nsamples_sweep and len(args.nsamples_sweep) > 0:
        ns_list = sorted(set([int(n) for n in args.nsamples_sweep if int(n) > 0]))
        logger.info(f"[sweep] nsamples list = {ns_list}")
        tokens = build_tokens(
            tok,
            nsamples=max(ns_list),
            seqlen=args.seqlen,
            dataset=args.calib_dataset,
            config=args.calib_config,
        )
        snaps = stream_xtx_sumx_sweep(
            model,
            tokens,
            device,
            args.model_name,
            args.layer_idx,
            ns_list,
            matmul_dtype=torch.float32,
            store_device=args.cov_store_device,
        )


        for stem, e_names in [("qkv", attn_names), ("mlp", mlp_names)]:
            if not e_names:
                logger.warning(f"[skip] no error matrices for group {stem}")
                continue
            E_cat = torch.cat([err_T[n] for n in e_names], dim=0).to(
                device, dtype=torch.float32
            )

            ec_map_cov: Dict[int, np.ndarray] = {}
            ec_map_no: Dict[int, np.ndarray] = {}


            for N in ns_list:
                slot = snaps[N].get(f"layer{args.layer_idx}_{stem}")
                if slot is None:
                    logger.warning(
                        f"[warn] missing stats for {stem} at N={N}, skipping"
                    )
                    continue
                Sig = cov_from_stats(slot, center=args.center)


                eigs = eigvals_desc(Sig, device)
                np.savetxt(
                    os.path.join(
                        args.out_dir,
                        f"input_spectrum_linear_{stem}_layer{args.layer_idx:02d}_N{N}.csv",
                    ),
                    eigs,
                    delimiter=",",
                    fmt="%.6e",
                )
                plot_input_spectrum_linear(
                    eigs,
                    f"Layer {args.layer_idx} — input eigenvalues (linear, {stem}, {'cov' if args.center else 'no-cov'}, N={N})",
                    os.path.join(
                        args.out_dir,
                        f"input_spectrum_linear_{stem}_layer{args.layer_idx:02d}_N{N}.png",
                    ),
                    topk=args.topk if args.topk > 0 else None,
                )

                Ssqrt = sqrtm_from_cov(Sig, device)
                ec_no = sv_energy_curve(E_cat, topk=args.topk, device=device)
                ec_cov = sv_energy_curve(E_cat @ Ssqrt, topk=args.topk, device=device)
                ec_map_no[N] = ec_no
                ec_map_cov[N] = ec_cov


            cmp_png = os.path.join(
                args.out_dir,
                f"error_energy_compare_nsamples_{stem}_layer{args.layer_idx:02d}.png",
            )
            plot_energy_compare_overlay(
                ec_map_no,
                ec_map_cov,
                f"Layer {args.layer_idx} — error capture vs rank (overlay by N, {stem})",
                cmp_png,
                topk=args.topk,
                thresholds=tuple(args.energy_thresholds),
            )


            cmp_csv = os.path.join(
                args.out_dir,
                f"error_energy_compare_nsamples_{stem}_layer{args.layer_idx:02d}.csv",
            )
            with open(cmp_csv, "w") as f:
                Ns = sorted(ec_map_cov.keys())
                header = (
                    ["rank"]
                    + [f"E_no_cov_N{N}" for N in Ns]
                    + [f"E_cov_align_N{N}" for N in Ns]
                )
                f.write(",".join(header) + "\n")

                R = None
                for N in Ns:
                    ln = len(ec_map_cov[N])
                    if N in ec_map_no:
                        ln = min(ln, len(ec_map_no[N]))
                    R = ln if R is None else min(R, ln)
                R = R or 0
                for r in range(1, R + 1):
                    row = [str(r)]
                    for N in Ns:
                        row.append(f"{ec_map_no[N][r-1]:.6f}")
                    for N in Ns:
                        row.append(f"{ec_map_cov[N][r-1]:.6f}")
                    f.write(",".join(row) + "\n")


            mr_csv = os.path.join(
                args.out_dir,
                f"minrank_error_compare_nsamples_{stem}_layer{args.layer_idx:02d}.csv",
            )
            with open(mr_csv, "w") as f:
                Ns = sorted(ec_map_cov.keys())
                header = (
                    ["threshold"]
                    + [f"minrank_E_no_cov_N{N}" for N in Ns]
                    + [f"minrank_E_cov_align_N{N}" for N in Ns]
                )
                f.write(",".join(header) + "\n")
                for thr in args.energy_thresholds:
                    row = [f"{thr:.2f}"]
                    for N in Ns:
                        row.append(str(min_rank(ec_map_no[N], thr)))
                    for N in Ns:
                        row.append(str(min_rank(ec_map_cov[N], thr)))
                    f.write(",".join(row) + "\n")

            logger.info(f"[save] {cmp_png}")
            logger.info(f"[save] {cmp_csv}")
            logger.info(f"[save] {mr_csv}")

    else:

        os.makedirs(args.out_dir, exist_ok=True)
        tokens = build_tokens(
            tok,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            dataset=args.calib_dataset,
            config=args.calib_config,
        )


        snaps = stream_xtx_sumx_sweep(
            model,
            tokens,
            device,
            args.model_name,
            args.layer_idx,
            [args.nsamples],
            matmul_dtype=torch.float32,
            store_device=args.cov_store_device,
        )
        stats = snaps[args.nsamples]

        for gkey in [f"layer{args.layer_idx}_qkv", f"layer{args.layer_idx}_mlp"]:
            stem = gkey.split("_")[-1]
            if stem == "qkv":
                e_names = attn_names
            else:
                e_names = mlp_names
            if not e_names:
                logger.warning(f"[skip] no error matrices for group {stem}")
                continue

            slot = stats.get(gkey)
            if slot is None:
                logger.warning(f"[skip] no stats for {gkey}")
                continue
            Sig = cov_from_stats(slot, center=args.center)


            eigs = eigvals_desc(Sig, device)
            np.savetxt(
                os.path.join(
                    args.out_dir,
                    f"input_spectrum_linear_{stem}_layer{args.layer_idx:02d}.csv",
                ),
                eigs,
                delimiter=",",
                fmt="%.6e",
            )
            plot_input_spectrum_linear(
                eigs,
                f"Layer {args.layer_idx} — input eigenvalues (linear, {stem}, {'cov' if args.center else 'no-cov'})",
                os.path.join(
                    args.out_dir,
                    f"input_spectrum_linear_{stem}_layer{args.layer_idx:02d}.png",
                ),
                topk=args.topk if args.topk > 0 else None,
            )


            Ssqrt = sqrtm_from_cov(Sig, device)


            E_cat = torch.cat([err_T[n] for n in e_names], dim=0).to(
                device, dtype=torch.float32
            )


            ec_no = sv_energy_curve(E_cat, topk=args.topk, device=device)
            ec_cov = sv_energy_curve(E_cat @ Ssqrt, topk=args.topk, device=device)


            cmp_png = os.path.join(
                args.out_dir,
                f"error_energy_compare_{stem}_layer{args.layer_idx:02d}.png",
            )
            plt.figure(figsize=(7.0, 4.4))
            R = min(len(ec_no), len(ec_cov))
            x = np.arange(1, R + 1)
            plt.plot(
                x, ec_no[:R], label="error capture — E (no cov aware)", linestyle="--"
            )
            plt.plot(x, ec_cov[:R], label="error capture — E Σ^{1/2} (cov aware)")
            for thr in args.energy_thresholds:
                plt.axhline(thr, linestyle=":", linewidth=1)
            plt.xlabel("rank r")
            plt.ylabel("error capture ≤ r (∥·∥_F^2)")
            plt.title(f"Layer {args.layer_idx} — error capture (E vs E Σ^1/2, {stem})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(cmp_png, dpi=180, bbox_inches="tight")
            plt.close()

            cmp_csv = os.path.join(
                args.out_dir,
                f"error_energy_compare_{stem}_layer{args.layer_idx:02d}.csv",
            )
            with open(cmp_csv, "w") as f:
                f.write("rank,E_no_cov,E_cov_align\n")
                for r in range(1, R + 1):
                    f.write(f"{r},{ec_no[r-1]:.6f},{ec_cov[r-1]:.6f}\n")

            mr_csv = os.path.join(
                args.out_dir,
                f"minrank_error_compare_{stem}_layer{args.layer_idx:02d}.csv",
            )
            with open(mr_csv, "w") as f:
                f.write("threshold,minrank_E_no_cov,minrank_E_cov_align\n")
                for thr in args.energy_thresholds:
                    f.write(
                        f"{thr:.2f},{min_rank(ec_no, thr)},{min_rank(ec_cov, thr)}\n"
                    )

            logger.info(f"[save] {cmp_png}")
            logger.info(f"[save] {cmp_csv}")
            logger.info(f"[save] {mr_csv}")


if __name__ == "__main__":
    main()

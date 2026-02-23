"""
svd_time.py
Benchmarks exact and randomized SVD runtime on GSVD core matrices across configurable settings.
output :
<dirname(csv_out)>/
|-- <basename(csv_out)>
`-- svd_artifacts/   (or --artifact_root)
    `-- <cfg_name>/
        |-- low_rank_shared.pt
        `-- b_ref_map.json
<cov_stats_path> (optional cache .pt)
<log_path> (optional)
"""

import os
import re
import csv
import json
import time
import torch
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




logger = logging.getLogger("SVDTimer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


def setup_file_logger(log_path: Optional[str]):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)





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


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",")]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Failed to parse int list from '{s}': {e}")


def load_data(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    logger.info(f"Loaded {len(data)} tensors from {path}")
    return data


def extract_layer_index(name: str) -> str:
    m = re.search(r"layers?\.(\d+)\.", name)
    return m.group(1) if m else "unknown"


SUFFIX_ORDER = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "gate_proj": 0, "up_proj": 1}


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
        layer_dimensions[layer_idx][module_name] = err_T[name].shape

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
            logger.info(f"Layer {layer_idx} dimensions ({model_name}):")
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
        kw in name_lower for kw in ["llama", "vicuna", "mistral", "qwen", "phi"]
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
    m, n = M.shape
    proj_dim = min(n, rank + n_oversamples)

    Omega = torch.randn(n, proj_dim, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(max(0, n_power_iters)):
        Y = M @ (M.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ M
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    return U[:, :rank], S[:rank], Vh[:rank, :]


def exact_svd_core(M: torch.Tensor, rank: int):

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r], S[:r], Vh[:r, :]


def timeit_cuda_sync(fn, *args, **kwargs) -> float:

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0





def build_core_qr(
    E_list: List[torch.Tensor],
    Sigma_sqrt: torch.Tensor,
    device: torch.device,
):

    E_cat = torch.cat(E_list, dim=0).to(device, dtype=torch.float32)
    rows_per_tensor = [E.shape[0] for E in E_list]
    Q_e, R_e = torch.linalg.qr(E_cat, mode="reduced")
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    M = R_e @ W_sqrt
    return Q_e, M, rows_per_tensor


def timed_build_core_qr(
    E_list: List[torch.Tensor],
    Sigma_sqrt: torch.Tensor,
    device: torch.device,
):

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Q_e, R_e = torch.linalg.qr(
        torch.cat(E_list, dim=0).to(device, torch.float32), mode="reduced"
    )
    M = R_e @ Sigma_sqrt.to(device, torch.float32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    rows_per_tensor = [E.shape[0] for E in E_list]
    return Q_e, M, rows_per_tensor, (t1 - t0)


def assemble_A_B_from_factors(
    U_m: torch.Tensor,
    S_vals: torch.Tensor,
    Vh: torch.Tensor,
    Q_e: torch.Tensor,
    rows_per_tensor: List[int],
    Sigma_inv_sqrt: torch.Tensor,
):

    device = U_m.device
    U = Q_e @ U_m
    r = U_m.shape[1]
    S_sqrt_diag = torch.sqrt(torch.clamp(S_vals[:r], min=1e-8)).diag().to(device)
    A_list = []
    start = 0
    for rows in rows_per_tensor:
        U_slice = U[start : start + rows, :r]
        A_i = (U_slice @ S_sqrt_diag).detach().cpu().to(torch.float16)
        A_list.append(A_i)
        start += rows
    B_temp = S_sqrt_diag @ Vh[:r, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B_shared = (B_temp @ W_inv_sqrt).detach().cpu().to(torch.float16)
    return A_list, B_shared





def main(args):
    setup_file_logger(args.log_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    err_T = load_data(args.err_path)


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
        args.reuse_cov_stats
        and args.cov_stats_path
        and os.path.isfile(args.cov_stats_path)
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
            torch.save(
                {k: v.cpu() for k, v in cov_matrices.items()}, args.cov_stats_path
            )


    del model
    torch.cuda.empty_cache()


    sqrt_matrices = {}
    for name, cov in tqdm(cov_matrices.items(), desc="Calculating Matrix Square Roots"):
        sqrt_matrices[name] = calculate_matrix_sqrt_and_inv_sqrt(cov, device)

    layer_groups = build_groups(err_T, args.model_name)


    grouped_names = {n for names in layer_groups.values() for n in names}
    solo_names = sorted([n for n in err_T if n not in grouped_names and "layers" in n])


    p_list = parse_int_list(args.p_list) if args.p_list is not None else None
    q_list = parse_int_list(args.q_list) if args.q_list is not None else None
    if p_list is None:
        p_list = [args.oversamples]
    if q_list is None:
        q_list = [args.power_iters]



    configs: List[Tuple[str, Optional[int], Optional[int], bool]] = [
        ("exact_svd", None, None, False)
    ]
    seen_pairs = set()
    for p_val in p_list:
        for q_val in q_list:
            if (p_val, q_val) not in seen_pairs:
                configs.append((f"qr_rsvd_q{q_val}_p{p_val}", p_val, q_val, False))
                seen_pairs.add((p_val, q_val))
            if (
                args.compare_swap
                and (p_val != q_val)
                and (q_val, p_val) not in seen_pairs
            ):
                configs.append((f"qr_rsvd_q{p_val}_p{q_val}", q_val, p_val, True))
                seen_pairs.add((q_val, p_val))


    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "group_key",
                "rank",
                "p",
                "q",
                "swapped",
                "rand_time_s",
                "exact_time_s",
                "m_rows",
                "m_cols",
                "qr_core_time_s",
                "rand_total_s",
                "exact_total_s",
            ]
        )


        artifact_root = args.artifact_root or os.path.join(
            os.path.dirname(args.csv_out) or ".", "svd_artifacts"
        )
        os.makedirs(artifact_root, exist_ok=True)


        per_config_shared: Dict[str, Dict[str, torch.Tensor]] = {
            name: {} for name, _, _, _ in configs
        }
        per_config_bref: Dict[str, Dict[str, str]] = {
            name: {} for name, _, _, _ in configs
        }


        processed = 0
        for gk, names in tqdm(layer_groups.items(), desc="Timing & Saving per group"):
            if args.max_groups > 0 and processed >= args.max_groups:
                break
            if gk not in sqrt_matrices:
                logger.warning(f"Covariance matrix not found for group {gk}. Skipping.")
                continue

            Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[gk]
            E_list = [err_T[n] for n in names]


            try:
                Q_e, M, rows_per, qr_core_time = timed_build_core_qr(
                    E_list, Sigma_sqrt, device
                )
            except torch.linalg.LinAlgError as e:
                logger.warning(f"QR failed for {gk}: {e}. Skipping.")
                continue


            exact_time = timeit_cuda_sync(exact_svd_core, M, args.max_rank)

            Ue, Se, Vhe = exact_svd_core(M, args.max_rank)
            A_list_exact, B_shared_exact = assemble_A_B_from_factors(
                Ue, Se, Vhe, Q_e, rows_per, Sigma_inv_sqrt
            )

            exact_shared = per_config_shared["exact_svd"]
            exact_bref = per_config_bref["exact_svd"]
            b_key_shared = f"{gk}.B_shared"
            exact_shared[b_key_shared] = B_shared_exact
            for i, name in enumerate(names):
                module_suffix = name.split(".")[-2]
                a_key = f"{gk}.{module_suffix}.A"
                exact_shared[a_key] = A_list_exact[i]
                exact_bref[name] = b_key_shared


            for cfg_name, p_val, q_val, swapped_flag in configs:
                if cfg_name == "exact_svd":
                    continue
                rand_time = timeit_cuda_sync(
                    randomized_svd_pytorch, M, args.max_rank, p_val, q_val
                )

                rand_total = qr_core_time + rand_time
                exact_total = qr_core_time + exact_time

                writer.writerow(
                    [
                        args.model_name,
                        gk,
                        args.max_rank,
                        p_val,
                        q_val,
                        swapped_flag,
                        f"{rand_time:.6f}",
                        f"{exact_time:.6f}",
                        M.shape[0],
                        M.shape[1],
                        f"{qr_core_time:.6f}",
                        f"{rand_total:.6f}",
                        f"{exact_total:.6f}",
                    ]
                )


                Ur, Sr, Vhr = randomized_svd_pytorch(M, args.max_rank, p_val, q_val)
                A_list_r, B_shared_r = assemble_A_B_from_factors(
                    Ur, Sr, Vhr, Q_e, rows_per, Sigma_inv_sqrt
                )
                cfg_shared = per_config_shared[cfg_name]
                cfg_bref = per_config_bref[cfg_name]
                b_key_shared = f"{gk}.B_shared"
                cfg_shared[b_key_shared] = B_shared_r
                for i, name in enumerate(names):
                    module_suffix = name.split(".")[-2]
                    a_key = f"{gk}.{module_suffix}.A"
                    cfg_shared[a_key] = A_list_r[i]
                    cfg_bref[name] = b_key_shared

            processed += 1



        for name in tqdm(solo_names, desc="Processing solo layers"):
            module_name = name.replace(".weight", "")
            if module_name not in sqrt_matrices:
                logger.warning(
                    f"Covariance matrix not found for solo layer {module_name}. Skipping."
                )
                continue

            Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[module_name]
            E_single = [err_T[name]]

            try:
                Q_e, M, rows_per, _ = timed_build_core_qr(E_single, Sigma_sqrt, device)
            except torch.linalg.LinAlgError as e:
                logger.warning(f"QR failed for solo {module_name}: {e}. Skipping.")
                continue


            Ue, Se, Vhe = exact_svd_core(M, args.max_rank)
            A_list_exact, B_exact = assemble_A_B_from_factors(
                Ue, Se, Vhe, Q_e, rows_per, Sigma_inv_sqrt
            )
            per_config_shared["exact_svd"][f"{module_name}.A"] = A_list_exact[0]
            per_config_shared["exact_svd"][f"{module_name}.B"] = B_exact
            per_config_bref["exact_svd"][name] = f"{module_name}.B"


            for cfg_name, p_val, q_val, _swapped in configs:
                if cfg_name == "exact_svd":
                    continue
                Ur, Sr, Vhr = randomized_svd_pytorch(M, args.max_rank, p_val, q_val)
                A_list_r, B_r = assemble_A_B_from_factors(
                    Ur, Sr, Vhr, Q_e, rows_per, Sigma_inv_sqrt
                )
                per_config_shared[cfg_name][f"{module_name}.A"] = A_list_r[0]
                per_config_shared[cfg_name][f"{module_name}.B"] = B_r
                per_config_bref[cfg_name][name] = f"{module_name}.B"


        for cfg_name, _, _, _ in configs:
            out_dir = os.path.join(artifact_root, cfg_name)
            os.makedirs(out_dir, exist_ok=True)
            lowrank_path = os.path.join(out_dir, "low_rank_shared.pt")
            bref_path = os.path.join(out_dir, "b_ref_map.json")
            torch.save(per_config_shared[cfg_name], lowrank_path)
            with open(bref_path, "w") as bf:
                json.dump(per_config_bref[cfg_name], bf, indent=2, ensure_ascii=False)
            logger.info(f"Saved artifacts for {cfg_name} -> {out_dir}")

    logger.info(f"Saved SVD timing CSV to: {args.csv_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "SVD timing (Exact vs Randomized) on GSVD core matrices + Artifact saving"
    )
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF model name for covariance estimation.",
    )
    p.add_argument(
        "--err_path",
        type=str,
        required=True,
        help="Path to quantization error dictionary from step1.",
    )
    p.add_argument("--csv_out", type=str, required=True, help="Output CSV path.")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set for models like Qwen requiring custom code.",
    )


    p.add_argument("--max_rank", type=int, default=64, help="Top-r rank to compute.")
    p.add_argument(
        "--oversamples",
        type=int,
        default=10,
        help="Default randomized SVD oversampling p (used if --p_list is not set).",
    )
    p.add_argument(
        "--power_iters",
        type=int,
        default=2,
        help="Default randomized SVD power iterations q (used if --q_list is not set).",
    )
    p.add_argument(
        "--p_list",
        type=str,
        default=None,
        help="Comma-separated list of p (oversamples), e.g., '0,4,8,16'.",
    )
    p.add_argument(
        "--q_list",
        type=str,
        default=None,
        help="Comma-separated list of q (power iters), e.g., '0,1,2'.",
    )
    p.add_argument(
        "--compare_swap",
        type=str2bool,
        default=True,
        help="Also include swapped (p<->q) pairs if not already in the sweep.",
    )


    p.add_argument(
        "--shrinkage_alpha",
        type=float,
        default=0.05,
        help="Alpha for covariance shrinkage.",
    )
    p.add_argument("--nsamples", type=int, default=64, help="Calibration samples.")
    p.add_argument(
        "--seqlen", type=int, default=2048, help="Calibration sequence length."
    )
    p.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext",
        help="HF datasets identifier (e.g., 'wikitext' or 'DKYoon/SlimPajama-6B').",
    )
    p.add_argument(
        "--calib_config", type=str, default=None, help="HF datasets config name."
    )
    p.add_argument(
        "--cov_store_device",
        type=str,
        default="cpu",
        help="Where to accumulate XtX (cpu or cuda).",
    )
    p.add_argument(
        "--matmul_dtype",
        type=str,
        default="float32",
        help="Torch dtype for XtX accumulation.",
    )


    p.add_argument(
        "--cov_stats_path",
        type=str,
        default=None,
        help="Path to cache covariance stats (XtX).",
    )
    p.add_argument(
        "--reuse_cov_stats",
        type=str2bool,
        default=False,
        help="Reuse cached covariance statistics when available.",
    )


    p.add_argument(
        "--max_groups",
        type=int,
        default=0,
        help="Limit number of groups to process (0 = all).",
    )
    p.add_argument("--log_path", type=str, default=None, help="Optional log file path.")
    p.add_argument(
        "--artifact_root",
        type=str,
        default=None,
        help="Directory under which per-config artifacts are saved (defaults to <csv_dir>/svd_artifacts).",
    )

    args = p.parse_args()
    main(args)

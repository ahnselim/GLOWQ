"""
moe_ffn/step2_randomized_gsvd_layerwise.py
Performs layer-wise randomized GSVD for MoE FFN quantization error tensors without Shared-B grouping.
output :
<output_path>/
|-- low_rank_shared.pt
`-- b_ref_map.json
<cov_stats_path> (optional cache .pt)
"""

import os
import re
import json
import math
import torch
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




logger = logging.getLogger("RandomizedGSVD_Layerwise")
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





def build_module_to_group_map(model_name: str) -> Dict[str, str]:

    name_lower = model_name.lower()
    is_opt = "opt" in name_lower
    is_llama_family = any(
        kw in name_lower for kw in ["llama", "vicuna", "qwen", "phi"]
    )

    if is_opt:
        return {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}
    elif is_llama_family:
        return {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
        }
    else:
        return {
            "q_proj": "qkv",
            "k_proj": "qkv",
            "v_proj": "qkv",
            "gate_proj": "mlp",
            "up_proj": "mlp",
            "fc1": "mlp",
            "fc2": "mlp",
        }


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

    module_to_group_map = build_module_to_group_map(model_name)

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
            E_list,
            names,
            Sigma_sqrt,
            Sigma_inv_sqrt,
            rank,
            device,
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
    E_tilde_list = []
    for E in E_list:
        E_tilde = E.to(device, dtype=torch.float32) @ W_sqrt
        E_tilde_list.append(E_tilde)

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






def get_cov_key_for_weight(weight_name: str, model_name: str) -> str:

    if not weight_name.endswith(".weight"):
        return weight_name

    parts = weight_name.split(".")
    if len(parts) < 3:
        return weight_name.replace(".weight", "")

    module_suffix = parts[-2]
    layer_idx = extract_layer_index(weight_name)
    module_to_group_map = build_module_to_group_map(model_name)
    group_type = module_to_group_map.get(module_suffix)

    if group_type in ("qkv", "mlp"):
        return f"layer{layer_idx}_{group_type}"


    return weight_name.replace(".weight", "")





def main(args):
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

    cov_matrices: Dict[str, torch.Tensor]
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

    sqrt_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, cov in tqdm(cov_matrices.items(), desc="Calculating Matrix Square Roots"):
        sqrt_matrices[name] = calculate_matrix_sqrt_and_inv_sqrt(cov, device)

    shared: Dict[str, torch.Tensor] = {}
    b_ref_map: Dict[str, str] = {}

    logger.info(
        f"Starting Layerwise Randomized GSVD processing for rank={args.max_rank} on {args.model_name}..."
    )


    for name in tqdm(sorted(err_T.keys()), desc="Processing Layerwise GSVD"):
        if "layers" not in name:

            continue

        cov_key = get_cov_key_for_weight(name, args.model_name)
        if cov_key not in sqrt_matrices:
            logger.warning(
                f"Covariance matrix not found for weight {name} (cov_key={cov_key}). Skipping."
            )
            continue

        Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[cov_key]
        E_tensor = err_T[name]

        A_list, B = process_randomized_gsvd_group(
            [E_tensor],
            [name],
            Sigma_sqrt,
            Sigma_inv_sqrt,
            args.max_rank,
            device,
            args.oversamples,
            args.power_iters,
        )

        A = A_list[0]

        base_name = name.replace(".weight", "")
        a_key, b_key = f"{base_name}.A", f"{base_name}.B"

        shared[a_key] = A.to(torch.float16)
        shared[b_key] = B.to(torch.float16)
        b_ref_map[name] = b_key


    os.makedirs(args.output_path, exist_ok=True)
    torch.save(shared, os.path.join(args.output_path, "low_rank_shared.pt"))
    with open(os.path.join(args.output_path, "b_ref_map.json"), "w") as f:
        json.dump(b_ref_map, f, indent=2)

    logger.info(f"\nSaved artifacts to {args.output_path}")
    logger.info(f"  - low_rank_shared.pt: {len(shared)} tensors")
    logger.info(f"  - b_ref_map.json: {len(b_ref_map)} mappings")
    logger.info(
        f"\nLayerwise Randomized GSVD processing complete for {args.model_name}."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "STEP 2 (Layerwise) - Randomized GSVD without Shared-B Grouping"
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
    p.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the final artifacts.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set for models like Qwen requiring custom code.",
    )
    p.add_argument("--max_rank", type=int, default=64, help="Maximum rank for SVD.")
    p.add_argument(
        "--shrinkage_alpha",
        type=float,
        default=0.05,
        help="Alpha for covariance matrix shrinkage.",
    )
    p.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples for covariance estimation.",
    )
    p.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration tokens.",
    )
    p.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset identifier (HF datasets format).",
    )
    p.add_argument(
        "--calib_config",
        type=str,
        default=None,
        help="Optional config name for the calibration dataset.",
    )
    p.add_argument(
        "--cov_store_device",
        type=str,
        default="cpu",
        help="Device used to accumulate covariance statistics (e.g., cpu or cuda).",
    )
    p.add_argument(
        "--oversamples",
        type=int,
        default=10,
        help="Oversampling parameter for randomized SVD.",
    )
    p.add_argument(
        "--power_iters",
        type=int,
        default=2,
        help="Power iteration count for randomized SVD.",
    )
    p.add_argument(
        "--cov_stats_path",
        type=str,
        default=None,
        help="Path to cache covariance statistics (XtX).",
    )
    p.add_argument(
        "--reuse_cov_stats",
        type=str2bool,
        default=False,
        help="Reuse cached covariance statistics when available.",
    )
    p.add_argument(
        "--matmul_dtype",
        type=str,
        default="float32",
        help="Torch dtype name used for XtX accumulation (e.g., float32, float16).",
    )

    args = p.parse_args()
    main(args)

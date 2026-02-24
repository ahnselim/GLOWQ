"""
moe_ffn/step2_rsvd.py
Performs integrated randomized GSVD with usage-aware Shared-B grouping for MoE FFN quantization errors.
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
        kw in name_lower for kw in ["llama", "vicuna", "opt6b", "qwen", "phi", "mixtral", "mistral"]
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
    usage_weights: Optional[List[float]] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:

    if usage_weights is None:
        usage_weights = [1.0] * len(E_list)
    assert len(usage_weights) == len(E_list), "usage_weights length mismatch."


    scaled_list = []
    for E, w in zip(E_list, usage_weights):
        w = float(w)
        if w <= 0:
            w = 1e-6
        scaled_list.append(E * math.sqrt(w))

    E_cat = torch.cat(scaled_list, dim=0).to(device, dtype=torch.float32)
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
            usage_weights=usage_weights,
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
    usage_weights: Optional[List[float]] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:

    if usage_weights is None:
        usage_weights = [1.0] * len(E_list)
    assert len(usage_weights) == len(E_list), "usage_weights length mismatch."

    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    E_tilde_list = []
    for E, w in zip(E_list, usage_weights):
        w = float(w)
        if w <= 0:
            w = 1e-6
        E_tilde = E.to(device, dtype=torch.float32) @ W_sqrt
        E_tilde_list.append(E_tilde * math.sqrt(w))

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





def load_usage_stats(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    if not os.path.isfile(path):
        logger.warning(f"Usage stats file not found at {path}; fallback to uniform.")
        return {}
    with open(path, "r") as f:
        raw = json.load(f)

    cleaned = {}
    for k, v in raw.items():
        try:
            val = float(v)
        except Exception:
            continue
        if val < 0:
            val = 0.0
        cleaned[k] = val

    if not cleaned:
        logger.warning(f"Usage stats at {path} is empty or invalid; fallback to uniform.")
        return {}

    total = sum(cleaned.values())
    if total > 0:
        cleaned = {k: v / total for k, v in cleaned.items()}

    logger.info(
        f"Loaded usage stats for {len(cleaned)} parameters from {path} (may be counts or probs)."
    )
    return cleaned


def get_usage_weight_for_group(
    names: List[str],
    usage_stats: Dict[str, float],
    usage_mode: str = "uniform",
) -> List[float]:
    if usage_mode == "uniform" or not usage_stats:
        return [1.0 for _ in names]

    weights = []
    for n in names:
        w = usage_stats.get(n, 1.0)
        if w <= 0:
            w = 1e-6
        weights.append(w)
    return weights





def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    usage_stats = load_usage_stats(args.usage_stats_path)
    if args.usage_mode not in ("uniform", "usage_weighted"):
        raise ValueError(
            f"usage_mode must be one of ['uniform', 'usage_weighted'], got {args.usage_mode}"
        )
    logger.info(
        f"Usage mode: {args.usage_mode} "
        f"({'with' if usage_stats and args.usage_mode == 'usage_weighted' else 'no'} usage stats)"
    )

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

    shared: Dict[str, torch.Tensor] = {}
    b_ref_map: Dict[str, str] = {}

    logger.info(
        f"Starting Randomized GSVD processing for rank={args.max_rank} on {args.model_name}..."
    )


    for gk, names in tqdm(
        layer_groups.items(), desc="Processing Shared-B Groups with Randomized GSVD"
    ):
        if gk not in sqrt_matrices:
            logger.warning(f"Covariance matrix not found for group {gk}. Skipping.")
            continue

        Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[gk]
        E_list = [err_T[n] for n in names]

        usage_w = get_usage_weight_for_group(
            names, usage_stats=usage_stats, usage_mode=args.usage_mode
        )

        A_list, B_shared = process_randomized_gsvd_group(
            E_list,
            names,
            Sigma_sqrt,
            Sigma_inv_sqrt,
            args.max_rank,
            device,
            args.oversamples,
            args.power_iters,
            usage_weights=usage_w,
        )

        b_key_shared = f"{gk}.B_shared"
        shared[b_key_shared] = B_shared.to(torch.float16)

        for i, name in enumerate(names):
            module_suffix = name.split(".")[-2]
            a_key = f"{gk}.{module_suffix}.A"
            shared[a_key] = A_list[i].to(torch.float16)
            b_ref_map[name] = b_key_shared


    grouped_names = {n for names in layer_groups.values() for n in names}
    solo_names = sorted([n for n in err_T if n not in grouped_names and "layers" in n])

    for name in tqdm(solo_names, desc="Processing Solo Layers with Randomized GSVD"):
        module_name = name.replace(".weight", "")
        if module_name not in sqrt_matrices:
            logger.warning(
                f"Covariance matrix not found for solo layer {module_name}. Skipping."
            )
            continue

        Sigma_sqrt, Sigma_inv_sqrt = sqrt_matrices[module_name]
        usage_w = get_usage_weight_for_group(
            [name], usage_stats=usage_stats, usage_mode=args.usage_mode
        )

        A_list, B = process_randomized_gsvd_group(
            [err_T[name]],
            [name],
            Sigma_sqrt,
            Sigma_inv_sqrt,
            args.max_rank,
            device,
            args.oversamples,
            args.power_iters,
            usage_weights=usage_w,
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
    logger.info(
        f"\nRandomized GSVD processing complete for {args.model_name} "
        f"(usage_mode={args.usage_mode})."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "STEP 2 (Integrated) - Randomized GSVD with Shared-B Grouping (usage-aware)"
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


    p.add_argument(
        "--usage_mode",
        type=str,
        default="uniform",
        choices=["uniform", "usage_weighted"],
        help=(
            "How to weight each error matrix within a group: "
            "'uniform' (Design A) or 'usage_weighted' (Design B, requires usage_stats_path)."
        ),
    )
    p.add_argument(
        "--usage_stats_path",
        type=str,
        default=None,
        help=(
            "JSON file containing usage stats per weight name, e.g., "
            '{\"model.layers.0.mlp.experts.0.w1.weight\": 1234, ...}. '
            "Values can be counts or probabilities."
        ),
    )

    args = p.parse_args()
    main(args)

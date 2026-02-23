"""
triton/step2_randomized_gsvd_integrated.py
Builds randomized GSVD Shared-B low-rank correction artifacts from step1 quantization errors and estimated covariances.
output :
<output_path>/
|-- low_rank_shared.pt    (shared low-rank tensors)
`-- b_ref_map.json        (module-to-shared-B mapping)
"""

import os
import re
import json
import torch
import argparse
import logging
from typing import Dict, List, Tuple
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


def setup_file_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


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
            logger.info(f"Layer {layer_idx} dimensions ({model_name} ):")
            for module, shape in sorted(dims.items()):
                logger.info(f"  {module}: {shape}")
    return layer_groups





@torch.no_grad()
def estimate_input_covariance(
    model, tokenizer, device, model_name, nsamples=32, seqlen=2048, alpha=0.05
) -> Dict[str, torch.Tensor]:

    logger.info(
        f"Collecting input activations for {model_name} to estimate covariance matrices ..."
    )
    inputs = defaultdict(list)
    handles = []

    module_to_group_map = {
        "q_proj": "qkv",
        "k_proj": "qkv",
        "v_proj": "qkv",
        "o_proj": "qkv_out",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp_out",
    }

    def get_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            weight_key = f"{name}.weight"
            inputs[weight_key].append(x.reshape(-1, x.shape[-1]).cpu())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(get_hook(name)))

    calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([row["text"] for row in calib_data if row["text"].strip()])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    nsamples_actual = min(nsamples, tokens.numel() // seqlen)
    calib_tokens = torch.stack(
        [tokens[i * seqlen : (i + 1) * seqlen] for i in range(nsamples_actual)]
    )

    for i in tqdm(range(calib_tokens.shape[0]), desc="Calibration Forward Pass"):
        model(calib_tokens[i : i + 1].to(device))

    for h in handles:
        h.remove()

    cov_matrices = {}
    logger.info(f"Calculating covariance matrices with shrinkage (alpha={alpha})...")

    layer_group_data = defaultdict(lambda: defaultdict(list))
    individual_module_data = {}


    for name, data_list in inputs.items():
        parts = name.split(".")
        if len(parts) < 2:
            continue

        layer_idx = extract_layer_index(name)
        module_suffix = parts[-2]
        group_type = module_to_group_map.get(module_suffix)

        if group_type in ("qkv", "mlp"):
            layer_group_data[layer_idx][group_type].extend(data_list)
        else:

            module_key = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )
            individual_module_data[module_key] = data_list


    for layer_idx, groups in layer_group_data.items():
        for group_type, data_list in groups.items():
            if not data_list:
                continue

            group_key = f"layer{layer_idx}_{group_type}"
            x_cat = torch.cat(data_list, dim=0).to(torch.float32)
            cov = torch.matmul(x_cat.T, x_cat) / x_cat.shape[0]

            d = cov.shape[0]
            cov_trace = torch.trace(cov)
            if cov_trace > 0:
                identity_term = (cov_trace / d) * torch.eye(d, device=cov.device)
                stable_cov = (1 - alpha) * cov + alpha * identity_term
            else:
                stable_cov = cov + (1e-6 * torch.eye(d, device=cov.device))
            cov_matrices[group_key] = stable_cov.cpu()


    for module_key, data_list in individual_module_data.items():
        x_cat = torch.cat(data_list, dim=0).to(torch.float32)
        cov = torch.matmul(x_cat.T, x_cat) / x_cat.shape[0]

        d = cov.shape[0]
        cov_trace = torch.trace(cov)
        if cov_trace > 0:
            identity_term = (cov_trace / d) * torch.eye(d, device=cov.device)
            stable_cov = (1 - alpha) * cov + alpha * identity_term
        else:
            stable_cov = cov + (1e-6 * torch.eye(d, device=cov.device))
        cov_matrices[module_key] = stable_cov.cpu()

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
    M: torch.Tensor, rank: int, n_oversamples: int = 10, n_power_iters: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = rank + n_oversamples
    m, n = M.shape
    Omega = torch.randn(n, q, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(n_power_iters):
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
        M, rank=rank, n_oversamples=10, n_power_iters=2
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


    cov_matrices = estimate_input_covariance(
        model, tokenizer, device, args.model_name, alpha=args.shrinkage_alpha
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

        A_list, B_shared = process_randomized_gsvd_group(
            E_list, names, Sigma_sqrt, Sigma_inv_sqrt, args.max_rank, device
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
        A_list, B = process_randomized_gsvd_group(
            [err_T[name]], [name], Sigma_sqrt, Sigma_inv_sqrt, args.max_rank, device
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
        "STEP 2 (Integrated) - Randomized GSVD with Shared-B Grouping (FIXED)"
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
        "--log_path", type=str, default="./logs/randomized_gsvd_integrated.log"
    )

    args = p.parse_args()
    main(args)

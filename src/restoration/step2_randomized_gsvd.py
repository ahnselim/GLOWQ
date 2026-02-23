"""
step2_randomized_gsvd.py
Runs streaming randomized GSVD with Shared-B grouping and exports low-rank restoration artifacts.
output :
<output_path>/
|-- low_rank_shared.pt
`-- b_ref_map.json
"""

import os, re, json, torch, argparse, logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




logger = logging.getLogger("RandomizedGSVD_SharedB_Streaming")
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


SUFFIX_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "gate_proj": 0,
    "up_proj": 1,
    "fc1": 0,
    "fc2": 1,
}


def build_groups(err_T: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    layer_groups = defaultdict(list)
    layer_dimensions = defaultdict(dict)
    for name in err_T:
        parts = name.split(".")
        if len(parts) < 3:
            continue
        module_name = parts[
            -2
        ]
        layer_idx = extract_layer_index(name)
        layer_dimensions[layer_idx][module_name] = err_T[name].shape
        if module_name in ("q_proj", "k_proj", "v_proj"):
            layer_groups[f"layer{layer_idx}_qkv"].append(name)
        elif module_name in ("gate_proj", "up_proj", "fc1", "fc2"):
            layer_groups[f"layer{layer_idx}_mlp"].append(name)
    for gk, names in layer_groups.items():
        names.sort(key=lambda n: SUFFIX_ORDER.get(n.split(".")[-2], 99))
    logger.info(f"Total groups created for Shared-B processing: {len(layer_groups)}")
    return layer_groups





def build_calibration_tokens(
    tokenizer,
    nsamples=64,
    seqlen=2048,
    dataset_name="DKYoon/SlimPajama-6B",
    dataset_config=None,
) -> torch.Tensor:
    logger.info(
        f"Building calibration tokens via streaming (dataset={dataset_name}, nsamples={nsamples}, seqlen={seqlen})..."
    )
    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    ds_subset = ds.take(nsamples * 5)
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None):
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    token_buf, samples = [], []
    for row in ds_subset:
        txt = (row.get("text") or "").strip()
        if not txt:
            continue
        ids = (
            tokenizer(txt, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        if not ids:
            continue
        if eos_id is not None:
            ids.append(eos_id)
        token_buf.extend(ids)
        while len(token_buf) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(token_buf[:seqlen], dtype=torch.long))
            token_buf = token_buf[seqlen:]
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
def estimate_input_covariance_streaming(
    model,
    tokenizer,
    device,
    nsamples=64,
    seqlen=2048,
    alpha=0.05,
    calib_dataset="DKYoon/SlimPajama-6B",
    calib_config=None,
    cov_store_device="cpu",
    center=False,
    matmul_dtype=torch.float32,
) -> Dict[str, torch.Tensor]:
    
    logger.info("Collecting input activations (Streaming XtX...")
    cov_dev = torch.device(cov_store_device)

    def stat_slot(d: int):
        return {
            "xtx": torch.zeros(d, d, dtype=torch.float32, device=cov_dev),
            "sumx": torch.zeros(d, dtype=torch.float64, device=cov_dev),
            "n": 0,
        }

    stats: Dict[str, dict] = {}
    handles = []


    module_to_group_map = {
        "q_proj": "qkv",
        "k_proj": "qkv",
        "v_proj": "qkv",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "fc1": "mlp",
        "fc2": "mlp",
    }

    def get_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])
            d = x.shape[-1]
            parts = name.split(".")
            layer_idx = extract_layer_index(name)
            module_suffix = parts[
                -1
            ]

            group_type = module_to_group_map.get(module_suffix)
            key = (
                f"layer{layer_idx}_{group_type}"
                if group_type in ("qkv", "mlp")
                else name
            )

            if key not in stats:
                stats[key] = stat_slot(d)

            x32 = x.to(matmul_dtype)
            xtx = x32.T @ x32
            stats[key]["xtx"].add_(xtx.to(device=cov_dev))
            stats[key]["sumx"].add_(
                x.sum(dim=0).to(dtype=torch.float64, device=cov_dev)
            )
            stats[key]["n"] += x.shape[0]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(get_hook(name)))

    calib_tokens = build_calibration_tokens(
        tokenizer, nsamples, seqlen, calib_dataset, calib_config
    )
    if calib_tokens.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(f"Failed to build calibration tokens from {calib_dataset}.")

    for i in tqdm(range(calib_tokens.shape[0]), desc="Calibration Forward Pass"):
        model(calib_tokens[i : i + 1].to(device))

    for h in handles:
        h.remove()


    cov_mats: Dict[str, torch.Tensor] = {}
    logger.info(
        f"Calculating covariance matrices with shrinkage (alpha={alpha}), center={center} ..."
    )
    for key, slot in stats.items():
        n = max(1, slot["n"])
        xtx = slot["xtx"]
        d = xtx.shape[0]
        if center:
            mean = (slot["sumx"] / n).to(torch.float32)
            cov = (xtx - n * (mean[:, None] @ mean[None, :])) / max(1, (n - 1))
        else:
            cov = xtx / n
        tr = torch.trace(cov)
        if tr > 0:
            cov = (1 - alpha) * cov + alpha * (tr / d) * torch.eye(
                d, device=cov.device, dtype=cov.dtype
            )
        else:
            cov = cov + 1e-6 * torch.eye(d, device=cov.device, dtype=cov.dtype)
        cov_mats[key] = cov

    logger.info(f"[Streaming] Estimated {len(cov_mats)} covariance matrices.")

    sample_keys = [k for k in cov_mats.keys() if ("_qkv" in k or "_mlp" in k)][:6]
    logger.info(f"Sample group cov keys: {sample_keys}")
    return cov_mats





def calculate_matrix_sqrt_and_inv_sqrt(
    S: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    S_gpu = S.to(device, dtype=torch.float32)
    L, Q = torch.linalg.eigh(S_gpu)
    L = torch.clamp(L, min=1e-8)
    L_sqrt = torch.diag(torch.sqrt(L))
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L))
    S_sqrt = Q @ L_sqrt @ Q.T
    S_inv_sqrt = Q @ L_inv_sqrt @ Q.T
    return S_sqrt.cpu(), S_inv_sqrt.cpu()


def randomized_svd_pytorch(
    M: torch.Tensor, rank: int, n_oversamples: int = 10, n_power_iters: int = 2
):
    q = rank + n_oversamples
    m, n = M.shape
    Omega = torch.randn(n, q, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(n_power_iters):
        Y = M @ (M.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ M
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    return U[:, :rank], S[:rank], Vh[:rank, :]


@torch.no_grad()
def process_randomized_gsvd_group(
    E_list, Sigma_sqrt, Sigma_inv_sqrt, rank, device, n_oversamples=10, n_power_iters=2
):
    E_cat = torch.cat(E_list, dim=0).to(device, dtype=torch.float32)
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    Q_e, R_e = torch.linalg.qr(E_cat, mode="reduced")
    M = R_e @ W_sqrt
    U_m, S_vals, Vh = randomized_svd_pytorch(
        M, rank=rank, n_oversamples=n_oversamples, n_power_iters=n_power_iters
    )
    U = Q_e @ U_m
    r = min(rank, S_vals.numel())
    S_sqrt_diag = torch.diag(torch.sqrt(S_vals[:r]))
    A_list = []
    cur = 0
    for E_tensor in E_list:
        rows = E_tensor.shape[0]
        U_slice = U[cur : cur + rows, :r]
        A_list.append((U_slice @ S_sqrt_diag).cpu())
        cur += rows
    B_temp = S_sqrt_diag @ Vh[:r, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B_shared = (B_temp @ W_inv_sqrt).cpu()
    del E_cat, Q_e, R_e, M, U_m, Vh, W_sqrt, W_inv_sqrt
    torch.cuda.empty_cache()
    return A_list, B_shared


@torch.no_grad()
def process_weighted_svd_group(E_list, Sigma_sqrt, Sigma_inv_sqrt, rank, device):
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    E_tilde_list = [E.to(device, dtype=torch.float32) @ W_sqrt for E in E_list]
    E_tilde_cat = torch.cat(E_tilde_list, dim=0)
    U, S_vals, Vh = torch.linalg.svd(E_tilde_cat, full_matrices=False)
    r = min(rank, S_vals.numel())
    S_sqrt_diag = torch.diag(torch.sqrt(S_vals[:r]))
    A_list = []
    cur = 0
    for E_tensor in E_list:
        rows = E_tensor.shape[0]
        U_slice = U[cur : cur + rows, :r]
        A_list.append((U_slice @ S_sqrt_diag).cpu())
        cur += rows
    B_temp = S_sqrt_diag @ Vh[:r, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B_final = (B_temp @ W_inv_sqrt).cpu()
    del E_tilde_list, E_tilde_cat, U, Vh, W_sqrt, W_inv_sqrt
    torch.cuda.empty_cache()
    return A_list, B_final





def main(args):
    if args.use_tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("TF32/fast matmul enabled (high).")
    else:
        torch.set_float32_matmul_precision("highest")
        logger.info("TF32 disabled (highest).")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    err_T = load_data(args.err_path)

    logger.info("Loading FP16 model for covariance estimation...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    cov_matrices = estimate_input_covariance_streaming(
        model,
        tokenizer,
        device,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        alpha=args.shrinkage_alpha,
        calib_dataset=args.calib_dataset,
        calib_config=args.calib_config,
        cov_store_device=args.cov_store_device,
        center=args.stream_center,
        matmul_dtype=torch.float32,
    )
    del model
    torch.cuda.empty_cache()

    layer_groups = build_groups(err_T)
    shared, b_ref_map = {}, {}
    logger.info(
        f"Starting Randomized GSVD processing per-group (rank={args.max_rank}) ..."
    )


    for gk, names in tqdm(layer_groups.items(), desc="Processing Shared-B Groups"):
        cov = cov_matrices.get(gk, None)
        if cov is None:
            logger.warning(f"[skip] No covariance for group {gk}.")
            continue
        Sigma_sqrt, Sigma_inv_sqrt = calculate_matrix_sqrt_and_inv_sqrt(cov, device)
        E_list = [err_T[n] for n in names]
        try:
            A_list, B_shared = process_randomized_gsvd_group(
                E_list,
                Sigma_sqrt,
                Sigma_inv_sqrt,
                args.max_rank,
                device,
                n_oversamples=args.oversamples,
                n_power_iters=args.power_iters,
            )
        except torch.linalg.LinAlgError:
            logger.warning(f"QR/SVD failed for {gk}, falling back to weighted SVD.")
            A_list, B_shared = process_weighted_svd_group(
                E_list, Sigma_sqrt, Sigma_inv_sqrt, args.max_rank, device
            )
        b_key_shared = f"{gk}.B_shared"
        shared[b_key_shared] = B_shared.to(torch.float16)
        for name, A_i in zip(names, A_list):
            module_suffix = name.split(".")[-2]
            a_key = f"{gk}.{module_suffix}.A"
            shared[a_key] = A_i.to(torch.float16)
            b_ref_map[name] = b_key_shared
        del cov_matrices[gk], Sigma_sqrt, Sigma_inv_sqrt, E_list
        torch.cuda.empty_cache()


    grouped_names = {n for names in layer_groups.values() for n in names}
    solo_names = [n for n in err_T if n not in grouped_names]
    for name in tqdm(solo_names, desc="Processing Solo Layers"):
        module_key = name.replace(".weight", "")
        cov = cov_matrices.get(module_key, None)
        if cov is None:
            logger.warning(f"[skip] No covariance for solo layer {module_key}.")
            continue
        Sigma_sqrt, Sigma_inv_sqrt = calculate_matrix_sqrt_and_inv_sqrt(cov, device)
        E = err_T[name]
        try:
            A_list, B = process_randomized_gsvd_group(
                [E],
                Sigma_sqrt,
                Sigma_inv_sqrt,
                args.max_rank,
                device,
                n_oversamples=args.oversamples,
                n_power_iters=args.power_iters,
            )
        except torch.linalg.LinAlgError:
            logger.warning(
                f"QR/SVD failed for {module_key}, falling back to weighted SVD."
            )
            A_list, B = process_weighted_svd_group(
                [E], Sigma_sqrt, Sigma_inv_sqrt, args.max_rank, device
            )
        key_base = name.replace(".", "_")
        shared[f"{key_base}.A"] = A_list[0].to(torch.float16)
        shared[f"{key_base}.B"] = B.to(torch.float16)
        b_ref_map[name] = f"{key_base}.B"
        del cov_matrices[module_key], Sigma_sqrt, Sigma_inv_sqrt, E
        torch.cuda.empty_cache()

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(shared, os.path.join(args.output_path, "low_rank_shared.pt"))
    with open(os.path.join(args.output_path, "b_ref_map.json"), "w") as f:
        json.dump(b_ref_map, f, indent=2)

    logger.info(f"\nSaved artifacts to {args.output_path}")
    logger.info(f"   - low_rank_shared.pt: {len(shared)} tensors")
    logger.info(f"   - b_ref_map.json: {len(b_ref_map)} mappings")
    logger.info("\nRandomized GSVD (Streaming) complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "Randomized GSVD (QR + Randomized SVD) with Shared-B — Streaming XtX & On-the-fly sqrt"
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--err_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set for models requiring custom Hugging Face code (e.g., some Qwen variants).",
    )
    p.add_argument("--max_rank", type=int, default=64)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--shrinkage_alpha", type=float, default=0.05)
    p.add_argument("--calib_dataset", type=str, default="DKYoon/SlimPajama-6B")
    p.add_argument("--calib_config", type=str, default=None)
    p.add_argument(
        "--cov_store_device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    p.add_argument(
        "--stream_center",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    p.add_argument(
        "--use_tf32",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
    )
    p.add_argument("--oversamples", type=int, default=10)
    p.add_argument("--power_iters", type=int, default=2)
    args = p.parse_args()
    main(args)

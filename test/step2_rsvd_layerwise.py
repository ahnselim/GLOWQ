"""
step2_rsvd_layerwise.py
Runs layer-wise streaming randomized SVD/GSVD on quantization error tensors using calibration covariance statistics.
output :
<output_path>/
|-- low_rank_layerwise.pt
`-- b_ref_map_layerwise.json
<cov_stats_path> (optional cache .pt)
"""

import os, re, json, torch, argparse, logging
from typing import Dict, List, Tuple, Optional

SHARED_GROUP_SUFFIX = {
    "q_proj": "qkv",
    "k_proj": "qkv",
    "v_proj": "qkv",
    "gate_proj": "mlp",
    "up_proj": "mlp",
}
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




logger = logging.getLogger("RandomizedSVD_Layerwise_Streaming")
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





def build_calibration_tokens(
    tokenizer,
    nsamples=64,
    seqlen=2048,
    dataset_name="DKYoon/SlimPajama-6B",
    dataset_config: Optional[str] = None,
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

    logger.info(
        "Collecting input activations (Streaming XtX, Layer-wise; )..."
    )
    cov_dev = torch.device(cov_store_device)

    def stat_slot(d: int):
        return {
            "xtx": torch.zeros(d, d, dtype=torch.float32, device=cov_dev),
            "sumx": torch.zeros(d, dtype=torch.float64, device=cov_dev),
            "n": 0,
        }

    stats: Dict[str, dict] = {}
    handles = []

    def get_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])
            d = x.shape[-1]
            key = name

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

    logger.info(
        f"[Streaming] Estimated {len(cov_mats)} layer-wise covariance matrices."
    )
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
def process_randomized_weighted_svd(
    E, Sigma_sqrt, Sigma_inv_sqrt, rank, device, n_oversamples=10, n_power_iters=2
):
    E_gpu = E.to(device, dtype=torch.float32)
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)




    Q_e, R_e = torch.linalg.qr(E_gpu, mode="reduced")


    M = R_e @ W_sqrt

    U_m, S_vals, Vh = randomized_svd_pytorch(
        M, rank=rank, n_oversamples=n_oversamples, n_power_iters=n_power_iters
    )


    U = Q_e @ U_m

    r = min(rank, S_vals.numel())
    S_sqrt_diag = torch.diag(torch.sqrt(S_vals[:r]))

    A = (U[:, :r] @ S_sqrt_diag).cpu()

    B_temp = S_sqrt_diag @ Vh[:r, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B = (B_temp @ W_inv_sqrt).cpu()

    del E_gpu, Q_e, R_e, M, U_m, Vh, W_sqrt, W_inv_sqrt
    torch.cuda.empty_cache()
    return A, B


@torch.no_grad()
def process_weighted_svd(E, Sigma_sqrt, Sigma_inv_sqrt, rank, device):
    W_sqrt = Sigma_sqrt.to(device, dtype=torch.float32)
    E_tilde = E.to(device, dtype=torch.float32) @ W_sqrt

    U, S_vals, Vh = torch.linalg.svd(E_tilde, full_matrices=False)

    r = min(rank, S_vals.numel())
    S_sqrt_diag = torch.diag(torch.sqrt(S_vals[:r]))

    A = (U[:, :r] @ S_sqrt_diag).cpu()

    B_temp = S_sqrt_diag @ Vh[:r, :]
    W_inv_sqrt = Sigma_inv_sqrt.to(device, dtype=torch.float32)
    B = (B_temp @ W_inv_sqrt).cpu()

    del E_tilde, U, Vh, W_sqrt, W_inv_sqrt
    torch.cuda.empty_cache()
    return A, B





def main(args):
    if args.use_tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("TF32/fast matmul enabled (high).")
    else:
        torch.set_float32_matmul_precision("highest")
        logger.info("TF32 disabled (highest).")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    err_T = load_data(args.err_path)

    logger.info("Loading FP16 model for covariance estimation ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )

    def resolve_shared_group(module_name: str) -> Optional[str]:
        match = re.search(r"layers\.(\d+)\.", module_name)
        if not match:
            return None
        suffix = module_name.split(".")[-1]
        group_suffix = SHARED_GROUP_SUFFIX.get(suffix)
        if not group_suffix:
            return None
        return f"layer{match.group(1)}_{group_suffix}"

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
        if args.cov_stats_path:
            os.makedirs(os.path.dirname(args.cov_stats_path) or ".", exist_ok=True)
            torch.save(
                {k: v.cpu() for k, v in cov_matrices.items()}, args.cov_stats_path
            )
    del model
    torch.cuda.empty_cache()

    shared_usage_expected: Dict[str, int] = {}
    for name in err_T:
        module_key = name.replace(".weight", "")
        group_key = resolve_shared_group(module_key) or module_key
        shared_usage_expected[group_key] = shared_usage_expected.get(group_key, 0) + 1

    usage_remaining = dict(shared_usage_expected)

    layerwise_results, b_ref_map = {}, {}
    logger.info(
        f"Starting Randomized SVD processing per-layer (rank={args.max_rank}) ..."
    )

    for name, E in tqdm(err_T.items(), desc="Processing Layer-wise SVD"):
        module_key = name.replace(".weight", "")
        cov_key = module_key
        cov = cov_matrices.get(cov_key, None)

        if cov is None:
            shared_key = resolve_shared_group(module_key)
            if shared_key and shared_key in cov_matrices:
                cov_key = shared_key
                cov = cov_matrices[shared_key]

        if cov is None:
            logger.warning(f"[skip] No covariance found for layer {module_key}.")
            continue

        Sigma_sqrt, Sigma_inv_sqrt = calculate_matrix_sqrt_and_inv_sqrt(cov, device)

        try:
            A, B = process_randomized_weighted_svd(
                E,
                Sigma_sqrt,
                Sigma_inv_sqrt,
                args.max_rank,
                device,
                n_oversamples=args.oversamples,
                n_power_iters=args.power_iters,
            )
        except torch.linalg.LinAlgError:
            logger.warning(
                f"Randomized SVD failed for {module_key}, falling back to standard weighted SVD."
            )
            A, B = process_weighted_svd(
                E, Sigma_sqrt, Sigma_inv_sqrt, args.max_rank, device
            )

        a_key = f"{name}.A"
        b_key = f"{name}.B"
        layerwise_results[a_key] = A.to(torch.float16)
        layerwise_results[b_key] = B.to(torch.float16)
        b_ref_map[name] = b_key

        if cov_key in usage_remaining:
            usage_remaining[cov_key] -= 1
            if usage_remaining[cov_key] <= 0:
                usage_remaining.pop(cov_key, None)
                cov_matrices.pop(cov_key, None)
        else:
            cov_matrices.pop(cov_key, None)

        del Sigma_sqrt, Sigma_inv_sqrt, E
        torch.cuda.empty_cache()

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(
        layerwise_results, os.path.join(args.output_path, "low_rank_layerwise.pt")
    )
    with open(os.path.join(args.output_path, "b_ref_map_layerwise.json"), "w") as f:
        json.dump(b_ref_map, f, indent=2)

    logger.info(f"\nSaved artifacts to {args.output_path}")
    logger.info(f"   - low_rank_layerwise.pt: {len(layerwise_results)} tensors")
    logger.info(f"   - b_ref_map_layerwise.json: {len(b_ref_map)} mappings")
    logger.info("\nRandomized SVD (Layer-wise, Streaming) complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "Randomized SVD (Layer-wise) — Streaming XtX & On-the-fly sqrt"
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--err_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
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
    p.add_argument(
        "--cov_stats_path",
        type=str,
        default=None,
        help="Optional path to cache covariance statistics (shared with other scripts).",
    )
    p.add_argument(
        "--reuse_cov_stats",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="If true and stats exist at --cov_stats_path, reuse instead of recomputing.",
    )
    args = p.parse_args()
    main(args)

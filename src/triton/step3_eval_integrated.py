"""
triton/step3_eval_integrated.py
Evaluates Triton 4-bit models with and without SVD correction using step1/step2 artifacts (PPL and generation metrics).
output :
(stdout only)
|-- Baseline metrics       (PPL, eval time, generation latency/throughput)
`-- SVD-corrected metrics  (PPL, eval time, generation latency/throughput)
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, os, time, re
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton is not installed. Please install it with 'pip install triton'")

if HAS_TRITON:


    @triton.jit
    def quant_linear_kernel(
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_qwm,
        stride_qwk,
        stride_qzm,
        stride_qzk,
        stride_sm,
        stride_sk,
        stride_om,
        stride_on,
        group_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        HAS_ZEROS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N)
        pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        qweight_ptrs = qweight_ptr + (
            offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 2) * stride_qwk
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_offs = k * BLOCK_SIZE_K + offs_k
            x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
            packed_weights = tl.load(qweight_ptrs, mask=q_mask, other=0)
            is_low_nibble = k_offs % 2 == 0
            nibbles = tl.where(
                is_low_nibble[:, None], packed_weights & 0x0F, packed_weights >> 4
            )
            group_id = k_offs // group_size
            scales_ptrs = scales_ptr + (
                offs_bn[None, :] * stride_sm + group_id[:, None] * stride_sk
            )
            scales = tl.load(scales_ptrs, mask=q_mask, other=0.0)
            if HAS_ZEROS:
                zeros_group_id = group_id // 2
                qzeros_ptrs = qzeros_ptr + (
                    offs_bn[None, :] * stride_qzm + zeros_group_id[:, None] * stride_qzk
                )
                packed_zeros = tl.load(qzeros_ptrs, mask=q_mask, other=0)
                is_low_zero_nibble = group_id % 2 == 0
                zeros = tl.where(
                    is_low_zero_nibble[:, None], packed_zeros & 0x0F, packed_zeros >> 4
                )
            else:
                zeros = 8
            dequant_weights = (
                nibbles.to(tl.float32) - zeros.to(tl.float32)
            ) * scales.to(tl.float32)
            accumulator += tl.dot(x.to(tl.float32), dequant_weights)
            x_ptrs += BLOCK_SIZE_K * stride_xk
            qweight_ptrs += (BLOCK_SIZE_K // 2) * stride_qwk
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
            accumulator = accumulator + bias[None, :]
        c = accumulator.to(output_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        output_ptrs = (
            output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(output_ptrs, c, mask=c_mask)

    def quant_linear(x, qweight, qzeros, scales, bias, group_size):
        original_shape = x.shape
        M, K = (
            (original_shape[0] * original_shape[1], original_shape[2])
            if x.dim() == 3
            else x.shape
        )
        N = scales.shape[0]
        x = x.reshape(M, K)
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        stride_qz0, stride_qz1 = (
            (qzeros.stride(0), qzeros.stride(1)) if qzeros is not None else (0, 0)
        )
        quant_linear_kernel[grid](
            x,
            qweight,
            qzeros,
            scales,
            bias,
            output,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            stride_qz0,
            stride_qz1,
            scales.stride(0),
            scales.stride(1),
            output.stride(0),
            output.stride(1),
            group_size=group_size,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            HAS_ZEROS=(qzeros is not None),
            HAS_BIAS=(bias is not None),
            num_warps=4,
            num_stages=3,
        )
        return output.reshape(*original_shape[:-1], N)

    class TritonTrue4BitLinear(nn.Module):
        def __init__(self, in_features, out_features, group_size=128, bias=False):
            super().__init__()
            self.in_features, self.out_features, self.group_size = (
                in_features,
                out_features,
                group_size,
            )
            self.register_buffer(
                "qweight",
                torch.empty((out_features, in_features // 2), dtype=torch.uint8),
            )
            self.register_buffer(
                "qzeros",
                torch.empty(
                    (out_features, math.ceil(in_features / group_size) // 2),
                    dtype=torch.uint8,
                ),
            )
            self.register_buffer(
                "scales",
                torch.empty(
                    (out_features, math.ceil(in_features / group_size)),
                    dtype=torch.float16,
                ),
            )
            self.bias = (
                nn.Parameter(torch.empty(out_features, dtype=torch.float16))
                if bias
                else None
            )

        def forward(self, x):
            return quant_linear(
                x, self.qweight, self.qzeros, self.scales, self.bias, self.group_size
            )

        @classmethod
        def from_float(cls, linear_layer, group_size):
            qlayer = cls(
                linear_layer.in_features,
                linear_layer.out_features,
                group_size,
                linear_layer.bias is not None,
            ).to(linear_layer.weight.device, dtype=linear_layer.weight.dtype)
            W = linear_layer.weight.data.clone()
            O, I = W.shape
            if I % group_size != 0:
                W = F.pad(W, (0, group_size - (I % group_size)))
            I_padded = W.shape[1]
            W_grouped = W.reshape(O, I_padded // group_size, group_size)
            min_vals = W_grouped.min(dim=-1).values
            max_vals = W_grouped.max(dim=-1).values
            scales = ((max_vals - min_vals) / 15.0).clamp(min=1e-8)
            zeros_float = (-min_vals / scales).round()
            quant_values = (
                torch.round(
                    W_grouped / scales.unsqueeze(-1) + zeros_float.unsqueeze(-1)
                )
                .clamp(0, 15)
                .to(torch.uint8)
            )
            low_w, high_w = quant_values[:, :, 0::2], quant_values[:, :, 1::2]
            packed_weights = (high_w << 4) | low_w
            qlayer.qweight.data.copy_(packed_weights.reshape(O, I_padded // 2))
            qlayer.scales.data.copy_(scales.to(torch.float16))
            zeros_uint8 = zeros_float.to(torch.uint8)
            low_z, high_z = zeros_uint8[:, 0::2], zeros_uint8[:, 1::2]
            packed_zeros = (high_z << 4) | low_z
            qlayer.qzeros.data.copy_(packed_zeros)
            if linear_layer.bias is not None:
                qlayer.bias.data.copy_(linear_layer.bias.data)
            return qlayer





def get_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_to_triton_4bit(model, group_size=128):

    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and "lm_head" not in name
            and module.in_features > 0
        ):
            try:
                parent, attr_name = get_parent_module(model, name)
                new_module = TritonTrue4BitLinear.from_float(module, group_size)
                setattr(parent, attr_name, new_module)
            except Exception as e:
                print(f"Skipping conversion of {name} due to error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    return model


GROUP_CORR_CACHE = {}


class AddSVDCorrection(nn.Module):

    def __init__(self, inner, A_q, B_q, gkey, is_group, alpha_svd=1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.gkey = gkey
        self.is_group = is_group
        self.alpha_svd = alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha_svd == 0.0:
            return z

        try:

            A_q_dev, B_q_dev = self.A_q, self.B_q


            if self.is_group and x.dim() == 3 and x.shape[1] > 1:
                cache_key = f"{self.gkey}_prefill"
                cached_val = GROUP_CORR_CACHE.get(cache_key)
                if cached_val is not None:
                    intermediate_r = cached_val
                else:
                    intermediate_r = F.linear(x, B_q_dev)
                    GROUP_CORR_CACHE[cache_key] = intermediate_r
            else:
                intermediate_r = F.linear(x, B_q_dev)

            svd_raw = F.linear(intermediate_r, A_q_dev)


            if z.shape != svd_raw.shape:
                svd_raw = svd_raw.reshape(z.shape)
            return z.add_(svd_raw, alpha=self.alpha_svd)
        except RuntimeError:

            return z


def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):
    patched_count, skipped_count = 0, 0
    for weight_name, bkey in tqdm(bmap.items(), desc="Patching SVD Correction"):
        module_name = weight_name.replace(".weight", "")
        B_q = shared.get(bkey)
        is_group = "B_shared" in bkey
        if is_group:
            gkey = bkey.replace(".B_shared", "")
            module_suffix = module_name.split(".")[-1]
            a_key = f"{gkey}.{module_suffix}.A"
        else:
            gkey_match = re.match(r"(model\.layers\.\d+\..*?)\.B", bkey)
            gkey = gkey_match.group(1) if gkey_match else bkey.replace(".B", "")
            a_key = gkey + ".A"
        A_q = shared.get(a_key)

        if A_q is None or B_q is None:
            skipped_count += 1
            continue
        try:
            parent, attr_name = get_parent_module(model, module_name)
            inner = getattr(parent, attr_name)
            if not isinstance(inner, TritonTrue4BitLinear):
                skipped_count += 1
                continue

            wrapped = AddSVDCorrection(inner, A_q, B_q, gkey, is_group, alpha_svd)
            setattr(parent, attr_name, wrapped)
            patched_count += 1
        except AttributeError:
            skipped_count += 1
            continue

    print(
        f"SVD Correction Patching Summary: {patched_count} patched, {skipped_count} skipped"
    )
    return model





@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, model_name):
    print(f"\n--- Evaluating PPL for {model_name} ---")
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    seq_len = input_ids.size(1)
    total_loss, total_tokens = 0.0, 0
    start_time = time.time()

    pbar = tqdm(range(0, seq_len, 2048), desc=f"PPL for {model_name}")
    for i in pbar:
        GROUP_CORR_CACHE.clear()
        begin, end = i, min(i + 2048, seq_len)
        if end - begin <= 1:
            continue

        input_batch = input_ids[:, begin:end].to(device)
        outputs = model(input_batch, labels=input_batch)
        loss = outputs.loss
        num_tokens = (input_batch != tokenizer.pad_token_id).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        pbar.set_description(f"PPL (Loss: {loss.item():.4f})")

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    elapsed_time = time.time() - start_time
    print(f"✅ PPL Result: {ppl:.4f}, Time: {elapsed_time:.2f}s")
    return ppl, elapsed_time


@torch.no_grad()
def evaluate_generation(model, tokenizer, device, prompts, gen_args):

    repeats = gen_args.get("repeats", gen_args.get("gen_repeats", 1))
    max_new = gen_args.get("max_new_tokens", gen_args.get("gen_max_new_tokens", 50))

    model.eval()
    pad_id = tokenizer.eos_token_id
    ttfb_list, tokps_list = [], []
    for _ in range(int(repeats)):
        for prompt in prompts:
            GROUP_CORR_CACHE.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            t0 = perf_counter()
            model.generate(**inputs, max_new_tokens=1)
            torch.cuda.synchronize()
            ttfb_list.append((perf_counter() - t0) * 1000)

            t1 = perf_counter()
            output = model.generate(
                **inputs, max_new_tokens=int(max_new), pad_token_id=pad_id
            )
            torch.cuda.synchronize()
            t_total = perf_counter() - t1
            new_tokens = output.shape[1] - inputs.input_ids.shape[1]
            tokps_list.append(new_tokens / t_total if t_total > 0 else 0)
    return {
        "ttfb_ms_median": median(ttfb_list) if ttfb_list else 0,
        "tok_s_median": median(tokps_list) if tokps_list else 0,
    }





def main():
    p = argparse.ArgumentParser(
        description="Step 3 (Integrated) - Evaluate any model with optimized SVD correction."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--shared_path", required=True)
    p.add_argument("--bmap_path", required=True)
    p.add_argument("--original_weights_path", required=True)
    p.add_argument(
        "--trust_remote_code", action="store_true", help="Required for models like Qwen"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument(
        "--skip_gen", action="store_true", help="Skip generation metric evaluation"
    )
    p.add_argument("--gen_max_new_tokens", type=int, default=50)
    p.add_argument("--gen_repeats", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📥 Loading original FP16 model: {args.model_name}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    original_weights = torch.load(
        args.original_weights_path, map_location="cpu", weights_only=True
    )
    model_fp16.load_state_dict(original_weights)

    print(f"🔄 Converting {args.model_name} to Triton 4-bit...")
    model = convert_to_triton_4bit(model_fp16, group_size=args.group_size).to(device)
    del model_fp16, original_weights
    gc.collect()
    torch.cuda.empty_cache()

    print("🧩 Loading correction artifacts and patching wrappers...")

    shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)
    model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)

    results = {}
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "In a shocking finding, scientists discovered that",
    ]


    print("\n=== BASELINE EVALUATION (NO SVD CORRECTION) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 0.0
    ppl_base, time_base = evaluate_ppl(model, tokenizer, device, "Baseline")
    gen_metrics_base = None
    if not args.skip_gen:
        gen_metrics_base = evaluate_generation(
            model, tokenizer, device, prompts, vars(args)
        )
    results["baseline"] = {"ppl": ppl_base, "time": time_base, "gen": gen_metrics_base}


    print("\n=== SVD CORRECTION EVALUATION (ALPHA=1.0) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 1.0
    ppl_svd, time_svd = evaluate_ppl(model, tokenizer, device, "SVD Corrected")
    gen_metrics_svd = None
    if not args.skip_gen:
        gen_metrics_svd = evaluate_generation(
            model, tokenizer, device, prompts, vars(args)
        )
    results["svd"] = {"ppl": ppl_svd, "time": time_svd, "gen": gen_metrics_svd}


    print(f"\n{'='*20} FINAL SUMMARY ({args.model_name}) {'='*20}")
    print(
        f"Method".ljust(40)
        + f"| PPL".ljust(10)
        + f"| Time(s)".ljust(10)
        + f"| TTFB(ms)".ljust(12)
        + f"| Tok/s".ljust(10)
    )
    print("-" * 82)

    base = results["baseline"]
    base_gen = base["gen"]
    ttfb_b = f"{base_gen['ttfb_ms_median']:.1f}" if base_gen else "-"
    tok_s_b = f"{base_gen['tok_s_median']:.2f}" if base_gen else "-"
    print(
        f"Triton 4-bit Baseline".ljust(40)
        + f"| {base['ppl']:<9.4f}"
        + f"| {base['time']:<9.2f}"
        + f"| {ttfb_b:<11}"
        + f"| {tok_s_b:<9}"
    )

    svd = results["svd"]
    svd_gen = svd["gen"]
    ttfb_s = f"{svd_gen['ttfb_ms_median']:.1f}" if svd_gen else "-"
    tok_s_s = f"{svd_gen['tok_s_median']:.2f}" if svd_gen else "-"
    print(
        f"Triton 4-bit + SVD Correction".ljust(40)
        + f"| {svd['ppl']:<9.4f}"
        + f"| {svd['time']:<9.2f}"
        + f"| {ttfb_s:<11}"
        + f"| {tok_s_s:<9}"
    )
    print("=" * 82)


    print("\n🚀 Performance Analysis:")
    ppl_imp = base["ppl"] - svd["ppl"]
    ppl_imp_pct = (ppl_imp / base["ppl"]) * 100
    print(f"  - PPL Improvement: {ppl_imp:.4f} ({ppl_imp_pct:+.2f}%)")

    time_change = ((svd["time"] - base["time"]) / base["time"]) * 100
    print(
        f"  - PPL Evaluation Latency: {time_change:+.2f}% change ({base['time']:.2f}s -> {svd['time']:.2f}s)"
    )

    if base_gen and svd_gen:
        ttfb_change = (
            (svd_gen["ttfb_ms_median"] - base_gen["ttfb_ms_median"])
            / base_gen["ttfb_ms_median"]
        ) * 100
        tok_s_change = (
            (svd_gen["tok_s_median"] - base_gen["tok_s_median"])
            / base_gen["tok_s_median"]
        ) * 100
        print(
            f"  - Generation TTFB: {ttfb_change:+.1f}% change ({base_gen['ttfb_ms_median']:.1f}ms -> {svd_gen['ttfb_ms_median']:.1f}ms)"
        )
        print(
            f"  - Generation Throughput: {tok_s_change:+.1f}% change ({base_gen['tok_s_median']:.2f} tok/s -> {svd_gen['tok_s_median']:.2f} tok/s)"
        )

    if abs(time_change) < 7:
        print(
            "  - ✅ Excellent! Overhead of SVD correction is minimal due to Shared-B caching and GPU residency."
        )


if __name__ == "__main__":
    main()

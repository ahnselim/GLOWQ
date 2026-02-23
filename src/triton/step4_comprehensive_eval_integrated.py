"""
triton/step4_comprehensive_eval_integrated.py
Runs comprehensive benchmarking across FP16/BNB/AWQ/GPTQ/Triton variants with optional LM Harness evaluation.
output :
(stdout only)
|-- Per-method summary      (PPL, timing, generation metrics)
`-- Comprehensive table     (including LM Harness task scores when available)
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, os, time, re
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset


try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM as HuggingFaceAutoLM

    HAS_LM_HARNESS = True
except ImportError:
    HAS_LM_HARNESS = False
    print(
        "Warning: lm-evaluation-harness is not installed. Skipping harness evaluation. `pip install lm-eval`"
    )



try:
    import triton, triton.language as tl

    HAS_TRITON = True


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

except (ImportError, NameError):
    HAS_TRITON = False



def get_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_to_triton_4bit(model, group_size=128):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            try:
                parent, attr_name = get_parent_module(model, name)
                setattr(
                    parent,
                    attr_name,
                    TritonTrue4BitLinear.from_float(module, group_size),
                )
            except Exception:
                pass
    cleanup_memory()
    return model


GROUP_CORR_CACHE = {}


class AddSVDCorrection(nn.Module):
    def __init__(self, inner, A_q, B_q, gkey, is_group, alpha_svd=1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.gkey, self.is_group, self.alpha_svd = gkey, is_group, alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha_svd == 0.0:
            return z
        try:
            is_prefill = x.dim() == 3 and x.shape[1] > 1
            if self.is_group and is_prefill:
                key = f"{self.gkey}_prefill"
                cached_val = GROUP_CORR_CACHE.get(key)
                if cached_val is not None:
                    intermediate_r = cached_val
                else:
                    intermediate_r = F.linear(x, self.B_q)
                    GROUP_CORR_CACHE[key] = intermediate_r
            else:
                intermediate_r = F.linear(x, self.B_q)
            svd_raw = F.linear(intermediate_r, self.A_q)
            if z.shape != svd_raw.shape:
                svd_raw = svd_raw.reshape(z.shape)
            return z.add_(svd_raw, alpha=self.alpha_svd)
        except RuntimeError:
            return z


def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):
    patched, skipped = 0, 0
    for name, bkey in tqdm(bmap.items(), desc="Patching SVD Wrappers"):
        module_name = name.replace(".weight", "")
        B_q = shared.get(bkey)
        is_group = "B_shared" in bkey
        if is_group:
            gkey = bkey.replace(".B_shared", "")
            a_key = f"{gkey}.{module_name.split('.')[-1]}.A"
        else:
            gkey_match = re.match(r"(model\.layers\.\d+\..*?)\.B", bkey)
            gkey = gkey_match.group(1) if gkey_match else bkey.replace(".B", "")
            a_key = gkey + ".A"
        A_q = shared.get(a_key)
        if A_q is None or B_q is None:
            skipped += 1
            continue
        try:
            parent, attr = get_parent_module(model, module_name)
            inner = getattr(parent, attr)
            if not isinstance(inner, (TritonTrue4BitLinear, nn.Linear)):
                skipped += 1
                continue
            setattr(
                parent,
                attr,
                AddSVDCorrection(inner, A_q, B_q, gkey, is_group, alpha_svd),
            )
            patched += 1
        except AttributeError:
            skipped += 1
    print(f"SVD Patching: {patched} patched, {skipped} skipped")
    return model


def cleanup_memory(*args):
    for arg in args:
        if arg is not None:
            del arg
    gc.collect()
    torch.cuda.empty_cache()



@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, model_name_str):
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    seq_len = input_ids.size(1)
    total_loss, total_tokens = 0.0, 0
    start_time = time.time()
    pbar = tqdm(range(0, seq_len, 2048), desc=f"PPL for {model_name_str}")
    for i in pbar:
        GROUP_CORR_CACHE.clear()
        begin_loc, end_loc = i, min(i + 2048, seq_len)
        if end_loc - begin_loc <= 1:
            continue
        input_batch = input_ids[:, begin_loc:end_loc].to(device)
        outputs = model(input_batch, labels=input_batch)
        total_loss += outputs.loss.item() * (end_loc - begin_loc)
        total_tokens += end_loc - begin_loc
        pbar.set_description(f"PPL (Loss: {outputs.loss.item():.4f})")
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    elapsed_time = time.time() - start_time
    print(f"✅ PPL Result: {ppl:.4f}, Time: {elapsed_time:.2f}s")
    return ppl, elapsed_time


@torch.no_grad()
def measure_generation_metrics(model, tokenizer, device, args):
    model.eval()
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "In a shocking finding, scientists discovered that",
    ]
    pad_id = tokenizer.eos_token_id
    ttfb_list_ms, tokps_list = [], []
    for _ in range(args.gen_repeats):
        for prompt in prompts:
            GROUP_CORR_CACHE.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            try:
                t0 = perf_counter()
                model.generate(**inputs, max_new_tokens=1, pad_token_id=pad_id)
                torch.cuda.synchronize()
                ttfb_list_ms.append((perf_counter() - t0) * 1000)
                t1 = perf_counter()
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.gen_max_new_tokens,
                    pad_token_id=pad_id,
                )
                torch.cuda.synchronize()
                t_total = perf_counter() - t1
                gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                tokps_list.append(gen_tokens / t_total if t_total > 0 else 0.0)
            except Exception:
                continue
    return {
        "ttfb_ms_median": median(ttfb_list_ms) if ttfb_list_ms else 0,
        "tok_s_median": median(tokps_list) if tokps_list else 0,
    }


@torch.no_grad()
def evaluate_lm_harness(model, tokenizer, device, batch_size=1):
    if not HAS_LM_HARNESS:
        print("Skipping LM Harness.")
        return {}
    print(f"\n--- Evaluating with LM Harness ---")
    GROUP_CORR_CACHE.clear()
    torch.cuda.empty_cache()
    hf_model = HuggingFaceAutoLM(
        pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    tasks = ["arc_easy", "hellaswag", "winogrande"]
    results = evaluator.simple_evaluate(
        model=hf_model, tasks=tasks, num_fewshot=0, batch_size=batch_size, limit=100
    )
    harness_results = {}
    for task, res in results["results"].items():
        acc = res.get("acc,none", res.get("acc_norm", res.get("acc", 0.0)))
        harness_results[task] = acc * 100
        print(f"  • {task} (acc): {harness_results[task]:.2f}%")
    return harness_results



def load_model_and_tokenizer(model_name, quant_type, device, args):
    print(f"\n{'='*20} Loading Model: {model_name} ({quant_type}) {'='*20}")
    model_id = model_name
    kwargs = {"torch_dtype": torch.float16, "trust_remote_code": args.trust_remote_code}

    if quant_type == "fp16":
        kwargs["device_map"] = device
    elif quant_type in ["bnb_nf4", "bnb_nf4_dq"]:
        use_dq = quant_type == "bnb_nf4_dq"
        print(f"BNB NF4 Config: Double Quant = {use_dq}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=use_dq,
        )
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = device
    elif quant_type == "awq":
        model_id = args.awq_model_id
        print(f"Loading custom AWQ model: {model_id}")
        kwargs["device_map"] = device
    elif quant_type == "gptq":
        model_id = args.gptq_model_id
        print(f"Loading custom GPTQ model: {model_id}")
        kwargs["device_map"] = device
    else:
        kwargs["device_map"] = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if "triton" in quant_type:
        if not HAS_TRITON:
            raise ImportError("Triton not found.")
        print(f"Loading original weights from: {args.original_weights_path}")
        model.load_state_dict(
            torch.load(
                args.original_weights_path, map_location="cpu", weights_only=True
            )
        )
        print("Converting to Triton 4-bit...")
        model = convert_to_triton_4bit(model, group_size=args.group_size).to(device)
        if quant_type == "triton_svd":
            print("Loading and patching SVD corrections...")
            shared = torch.load(
                args.shared_path, map_location=device, weights_only=True
            )
            with open(args.bmap_path, "r") as f:
                bmap = json.load(f)
            model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)
            cleanup_memory(shared, bmap)
    return model, tokenizer



def main():
    p = argparse.ArgumentParser(
        description="Integrated comprehensive benchmark for language models."
    )

    p.add_argument("--model_name", required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--original_weights_path",
        type=str,
        help="Path to original weights from step1 for Triton models",
    )
    p.add_argument(
        "--shared_path",
        type=str,
        help="Path to low_rank_shared.pt from step2 for SVD correction",
    )
    p.add_argument(
        "--bmap_path",
        type=str,
        help="Path to b_ref_map.json from step2 for SVD correction",
    )

    p.add_argument("--eval_fp16", action="store_true", help="Evaluate FP16 baseline")
    p.add_argument(
        "--eval_bnb_nf4", action="store_true", help="Evaluate BitsAndBytes NF4"
    )
    p.add_argument(
        "--eval_bnb_nf4_dq",
        action="store_true",
        help="Evaluate BitsAndBytes NF4 with DoubleQuant",
    )
    p.add_argument("--eval_awq", action="store_true", help="Evaluate AWQ")
    p.add_argument("--eval_gptq", action="store_true", help="Evaluate GPTQ")
    p.add_argument(
        "--eval_triton_base", action="store_true", help="Evaluate custom Triton 4-bit"
    )
    p.add_argument(
        "--eval_triton_svd",
        action="store_true",
        help="Evaluate custom Triton 4-bit + SVD correction",
    )

    p.add_argument(
        "--awq_model_id", type=str, help="Custom model ID/path for AWQ model"
    )
    p.add_argument(
        "--gptq_model_id", type=str, help="Custom model ID/path for GPTQ model"
    )

    p.add_argument("--skip_gen", action="store_true")
    p.add_argument("--gen_max_new_tokens", type=int, default=50)
    p.add_argument("--gen_repeats", type=int, default=3)
    args = p.parse_args()


    EVAL_CONFIGS = {}
    if args.eval_fp16:
        EVAL_CONFIGS["FP16 Baseline"] = {"quant_type": "fp16"}
    if args.eval_bnb_nf4:
        EVAL_CONFIGS["BNB NF4 (No DQ)"] = {"quant_type": "bnb_nf4"}
    if args.eval_bnb_nf4_dq:
        EVAL_CONFIGS["BNB NF4 (DQ)"] = {"quant_type": "bnb_nf4_dq"}
    if args.eval_awq:
        if args.awq_model_id:
            EVAL_CONFIGS["AWQ 4-bit"] = {"quant_type": "awq"}
        else:
            print("⚠️ Warning: --eval_awq specified without --awq_model_id. Skipping.")
    if args.eval_gptq:
        if args.gptq_model_id:
            EVAL_CONFIGS["GPTQ 4-bit"] = {"quant_type": "gptq"}
        else:
            print("⚠️ Warning: --eval_gptq specified without --gptq_model_id. Skipping.")
    if args.eval_triton_base:
        if args.original_weights_path:
            EVAL_CONFIGS["Triton 4-bit (Base)"] = {"quant_type": "triton_base"}
        else:
            print(
                "⚠️ Warning: --eval_triton_base requires --original_weights_path. Skipping."
            )
    if args.eval_triton_svd:
        if all([args.original_weights_path, args.shared_path, args.bmap_path]):
            EVAL_CONFIGS["Triton 4-bit + SVD"] = {"quant_type": "triton_svd"}
        else:
            print(
                "⚠️ Warning: --eval_triton_svd requires --original_weights_path, --shared_path, and --bmap_path. Skipping."
            )

    all_results = {}
    for name, config in EVAL_CONFIGS.items():
        model, tokenizer = None, None
        try:
            model, tokenizer = load_model_and_tokenizer(
                args.model_name, config["quant_type"], args.device, args
            )
            ppl, ppl_time = evaluate_ppl(model, tokenizer, args.device, name)
            gen_metrics = (
                {}
                if args.skip_gen
                else measure_generation_metrics(model, tokenizer, args.device, args)
            )
            harness_results = evaluate_lm_harness(model, tokenizer, args.device)
            all_results[name] = {
                "ppl": ppl,
                "ppl_time": ppl_time,
                "gen": gen_metrics,
                "harness": harness_results,
            }
        except Exception as e:
            print(f"\n❌ FAILED to evaluate '{name}'. Reason: {type(e).__name__} - {e}")
            all_results[name] = {"error": str(e)}
        finally:
            print(f"--- Cleaning up memory for {name} ---")
            cleanup_memory(model, tokenizer)


    print(f"\n\n{'='*45} FINAL COMPREHENSIVE SUMMARY {'='*45}")
    print(f"Base Model: {args.model_name}\n")
    header = f"{'Method':<30} | {'Perplexity':<12} | {'PPL Time(s)':<13} | {'TTFB(ms)':<10} | {'tok/s':<10} | {'ARC-E':<7} | {'HellaSwag':<10} | {'WinoGrande':<10}"
    print(header)
    print("-" * len(header))
    for name, data in all_results.items():
        if "error" in data:
            print(
                f"{name:<30} | {'ERROR':<12} | {'-':<13} | {'-':<10} | {'-':<10} | {'-':<7} | {'-':<10} | {'-':<10}"
            )
            continue
        ppl, time = f"{data['ppl']:.4f}", f"{data['ppl_time']:.2f}"
        gen = data.get("gen", {})
        harness = data.get("harness", {})
        ttfb, tok_s = (
            f"{gen.get('ttfb_ms_median', 0):.1f}",
            f"{gen.get('tok_s_median', 0):.2f}",
        )
        arc, hs, wg = (
            f"{harness.get('arc_easy', 0):.2f}",
            f"{harness.get('hellaswag', 0):.2f}",
            f"{harness.get('winogrande', 0):.2f}",
        )
        print(
            f"{name:<30} | {ppl:<12} | {time:<13} | {ttfb:<10} | {tok_s:<10} | {arc:<7} | {hs:<10} | {wg:<10}"
        )
    print("=" * len(header))
    print("\nEvaluation Complete! 🚀")


if __name__ == "__main__":
    main()

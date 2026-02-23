"""
step3_1_layerwise.py
Evaluates a layer-wise SVD-corrected quantized model without correction caching and measures restoration quality.
output :
(no output files)
`-- stdout/stderr metrics and logs only
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, os, time, re, sys
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict







_argv_disable = any(arg == "--use_cuda_w4a16" for arg in sys.argv)
_DISABLE_TRITON = (
    _argv_disable
    or os.environ.get("DISABLE_TRITON", "").lower() in ("1", "true", "yes")
    or os.environ.get("USE_CUDA_W4A16", "").lower() in ("1", "true", "yes")
)
try:
    if not _DISABLE_TRITON:
        import triton
        import triton.language as tl

        HAS_TRITON = True
    else:
        HAS_TRITON = False
except Exception:
    HAS_TRITON = False
    print(
        "Triton is not available or disabled. If needed, install with 'pip install triton'."
    )

def get_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

if HAS_TRITON:
    @triton.jit
    def quant_linear_kernel(
        x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, bias_ptr, output_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_qwm, stride_qwk, stride_qzm, stride_qzk,
        stride_sm, stride_sk, stride_om, stride_on,
        group_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0); num_pid_m = tl.cdiv(M, BLOCK_SIZE_M); num_pid_n = tl.cdiv(N, BLOCK_SIZE_N); pid_m = pid // num_pid_n; pid_n = pid % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M); offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N); offs_k = tl.arange(0, BLOCK_SIZE_K); x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk); qweight_ptrs = qweight_ptr + (offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 2) * stride_qwk)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_start = k * BLOCK_SIZE_K; k_offs = k_start + offs_k; x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K); x = tl.load(x_ptrs, mask=x_mask, other=0.0); q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N); packed_weights = tl.load(qweight_ptrs, mask=q_mask, other=0); is_low_nibble = k_offs % 2 == 0; nibbles = tl.where(is_low_nibble[:, None], packed_weights & 0x0F, packed_weights >> 4)
            group_id = k_offs // group_size; scales_ptrs = scales_ptr + (offs_bn[None, :] * stride_sm + group_id[:, None] * stride_sk); scales = tl.load(scales_ptrs, mask=q_mask, other=0.0)
            if HAS_ZEROS:
                zeros_group_id = group_id // 2; qzeros_ptrs = qzeros_ptr + (offs_bn[None, :] * stride_qzm + zeros_group_id[:, None] * stride_qzk); packed_zeros = tl.load(qzeros_ptrs, mask=q_mask, other=0); is_low_zero_nibble = group_id % 2 == 0; zeros = tl.where(is_low_zero_nibble[:, None], packed_zeros & 0x0F, packed_zeros >> 4)
            else: zeros = 8
            dequant_weights = (nibbles.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32); accumulator += tl.dot(x.to(tl.float32), dequant_weights); x_ptrs += BLOCK_SIZE_K * stride_xk; qweight_ptrs += (BLOCK_SIZE_K // 2) * stride_qwk
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0); accumulator = accumulator + bias[None, :]
        c = accumulator.to(output_ptr.dtype.element_ty); offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M); offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N); output_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]; c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N); tl.store(output_ptrs, c, mask=c_mask)

    def quant_linear(x, qweight, qzeros, scales, bias, group_size):
        original_shape = x.shape
        if x.dim() == 3: M, K = original_shape[0] * original_shape[1], original_shape[2]
        else: M, K = x.shape
        N = scales.shape[0]; x = x.reshape(M, K); output = torch.empty((M, N), device=x.device, dtype=x.dtype); grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        stride_qz0, stride_qz1 = (qzeros.stride(0), qzeros.stride(1)) if qzeros is not None else (0, 0)
        quant_linear_kernel[grid](x, qweight, qzeros, scales, bias, output, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), stride_qz0, stride_qz1, scales.stride(0), scales.stride(1), output.stride(0), output.stride(1), group_size=group_size, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, HAS_ZEROS=(qzeros is not None), HAS_BIAS=(bias is not None), num_warps=4, num_stages=3)
        return output.reshape(*original_shape[:-1], N)

    class TritonTrue4BitLinear(nn.Module):
        def __init__(self, in_features, out_features, group_size=128, bias=False):
            super().__init__(); self.in_features = in_features; self.out_features = out_features; self.group_size = group_size
            self.register_buffer("qweight", torch.empty((out_features, in_features // 2), dtype=torch.uint8)); self.register_buffer("qzeros", torch.empty((out_features, math.ceil(in_features / group_size) // 2), dtype=torch.uint8)); self.register_buffer("scales", torch.empty((out_features, math.ceil(in_features / group_size)), dtype=torch.float16))
            if bias: self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
            else: self.bias = None
        def forward(self, x): return quant_linear(x, self.qweight, self.qzeros, self.scales, self.bias, self.group_size)
        @classmethod
        def from_float(cls, linear_layer, group_size):
            qlayer = cls(linear_layer.in_features, linear_layer.out_features, group_size, linear_layer.bias is not None).to(linear_layer.weight.device, dtype=linear_layer.weight.dtype); W = linear_layer.weight.data.clone(); O, I = W.shape
            if I % group_size != 0: W = torch.nn.functional.pad(W, (0, group_size - (I % group_size)))
            I_padded = W.shape[1]; W_grouped = W.reshape(O, I_padded // group_size, group_size); min_vals = W_grouped.min(dim=-1).values; max_vals = W_grouped.max(dim=-1).values; scales = ((max_vals - min_vals) / 15.0).clamp(min=1e-8); zeros_float = (-min_vals / scales).round(); quant_values = torch.round(W_grouped / scales.unsqueeze(-1) + zeros_float.unsqueeze(-1)).clamp(0, 15).to(torch.uint8); low_w = quant_values[:, :, 0::2]; high_w = quant_values[:, :, 1::2]; packed_weights = (high_w << 4) | low_w; qlayer.qweight.data.copy_(packed_weights.reshape(O, I_padded // 2)); qlayer.scales.data.copy_(scales.to(torch.float16)); zeros_uint8 = zeros_float.to(torch.uint8); low_z = zeros_uint8[:, 0::2]; high_z = zeros_uint8[:, 1::2]; packed_zeros = (high_z << 4) | low_z; qlayer.qzeros.data.copy_(packed_zeros)
            if linear_layer.bias is not None: qlayer.bias.data.copy_(linear_layer.bias.data)
            return qlayer

    def convert_to_triton_4bit(model, group_size=128):
        TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(name.endswith(target) for target in TARGET_LAYERS):
                parent, attr_name = get_parent_module(model, name); new_module = TritonTrue4BitLinear.from_float(module, group_size); setattr(parent, attr_name, new_module)
        gc.collect(); torch.cuda.empty_cache(); return model




class AddSVDCorrection(nn.Module):
    def __init__(self, inner, A_q, B_q, module_name, alpha_svd=1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.module_name = module_name
        self.alpha_svd = alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha_svd == 0.0: return z
        A_q_dev = self.A_q; B_q_dev = self.B_q
        try:
            intermediate_r = F.linear(x, B_q_dev)
            svd_raw = F.linear(intermediate_r, A_q_dev)
            if z.shape != svd_raw.shape: svd_raw = svd_raw.reshape(z.shape)
            return z.add_(svd_raw, alpha=self.alpha_svd)
        except RuntimeError as e:
            if "size of tensor" in str(e) or "invalid for input of size" in str(e): return z
            else: raise e

def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):
    patched_count, skipped_count = 0, 0
    for weight_name, bkey in tqdm(bmap.items(), desc="Patching SVD Correction"):
        module_name = weight_name.replace(".weight", ""); B_q = shared.get(bkey)
        a_key = bkey.replace(".B", ".A"); A_q = shared.get(a_key)
        if A_q is None or B_q is None: skipped_count += 1; continue
        try:
            parent, attr_name = get_parent_module(model, module_name); inner = getattr(parent, attr_name)
            types_list = []
            try: from cuda_w4a16.linear import CudaW4A16Linear; types_list.append(CudaW4A16Linear)
            except Exception: pass
            try: types_list.append(TritonTrue4BitLinear)
            except Exception: pass
            valid_types = tuple(types_list) if types_list else (nn.Module,)
            if not isinstance(inner, valid_types): skipped_count += 1; continue
            wrapped = AddSVDCorrection(inner, A_q, B_q, module_name, alpha_svd); setattr(parent, attr_name, wrapped); patched_count += 1
        except AttributeError as e: print(f"AttributeError for {module_name}: {e}"); skipped_count += 1; continue
    print(f"SVD Correction Patching Summary: {patched_count} patched, {skipped_count} skipped"); return model




def _cuda_sync(device):
    if isinstance(device, torch.device) and device.type == "cuda": torch.cuda.synchronize(device)


@torch.no_grad()
def measure_generation_with_manual_loop(model, tokenizer, device, prompts, max_new_tokens=50, repeats=1):
    model.eval()
    ttfb_list_ms, tokps_list = [], []
    for _ in range(repeats):
        for prompt in tqdm(prompts, desc="Manual Generation (No Cache)", leave=False):
            try:

                input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
                _cuda_sync(device); t0 = perf_counter()
                outputs = model(input_ids=input_ids, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]; next_token = torch.argmax(next_token_logits, dim=-1); past_key_values = outputs.past_key_values
                _cuda_sync(device); ttfb_list_ms.append((perf_counter() - t0) * 1000.0)


                generated_token_ids = [next_token.item()]
                _cuda_sync(device); t1 = perf_counter()
                for _ in range(max_new_tokens - 1):

                    outputs = model(input_ids=next_token.unsqueeze(-1), past_key_values=past_key_values, use_cache=True)
                    next_token_logits = outputs.logits[:, -1, :]; next_token = torch.argmax(next_token_logits, dim=-1); past_key_values = outputs.past_key_values
                    generated_token_ids.append(next_token.item())
                    if next_token.item() == tokenizer.eos_token_id: break
                _cuda_sync(device); t_total_decode = perf_counter() - t1

                num_decoded_tokens = len(generated_token_ids)
                if t_total_decode > 0: tokps_list.append(num_decoded_tokens / t_total_decode)
            except Exception as e:
                print(f"Manual generation failed for prompt '{prompt[:30]}...': {e}"); continue
    return {"ttfb_ms_median": median(ttfb_list_ms) if ttfb_list_ms else 0, "tok_s_median": median(tokps_list) if tokps_list else 0}

@torch.no_grad()
def evaluate(model, tokenizer, device, model_name, batch_size=1, seq_len=2048):
    print(f"\n--- Evaluating {model_name} (batch_size={batch_size}, seq_len={seq_len}) ---"); model.eval(); ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test"); text = "\n\n".join(ds["text"]); enc = tokenizer(text, return_tensors="pt"); input_ids = enc.input_ids; total_seq_len = input_ids.size(1); total_loss, total_tokens = 0.0, 0; start_time = time.time()
    if batch_size == 1:
        pbar = tqdm(range(0, total_seq_len, seq_len), desc=f"PPL for {model_name}")
        for i in pbar:
            begin_loc, end_loc = i, min(i + seq_len, total_seq_len)
            if end_loc - begin_loc <= 1: continue
            input_batch = input_ids[:, begin_loc:end_loc].to(device); labels = input_batch; outputs = model(input_batch); logits = outputs.logits; shift_logits = logits[..., :-1, :].contiguous(); shift_labels = labels[..., 1:].contiguous(); loss_fct = nn.CrossEntropyLoss(); loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)); num_tokens = shift_labels.numel(); total_loss += loss.item() * num_tokens; total_tokens += num_tokens; pbar.set_description(f"PPL for {model_name} (Loss: {loss.item():.4f})")
    else:
        print(f"Creating {batch_size} sequences of length {seq_len} each..."); batches = []
        for i in range(0, min(total_seq_len, seq_len * batch_size * 10), seq_len):
            begin_loc, end_loc = i, min(i + seq_len, total_seq_len)
            if end_loc - begin_loc < seq_len:
                seq = input_ids[:, begin_loc:end_loc]; padding_needed = seq_len - (end_loc - begin_loc); seq = torch.nn.functional.pad(seq, (0, padding_needed), value=tokenizer.pad_token_id or tokenizer.eos_token_id)
            else: seq = input_ids[:, begin_loc:end_loc]
            batches.append(seq)
            if len(batches) >= batch_size * 10: break
        pbar = tqdm(range(0, len(batches), batch_size), desc=f"PPL for {model_name} (batch={batch_size})")
        for i in pbar:
            batch_seqs = batches[i:i+batch_size]
            if len(batch_seqs) < batch_size:
                while len(batch_seqs) < batch_size: batch_seqs.append(batch_seqs[-1])
            input_batch = torch.cat(batch_seqs, dim=0).to(device); labels = input_batch; outputs = model(input_batch); logits = outputs.logits; shift_logits = logits[..., :-1, :].contiguous(); shift_labels = labels[..., 1:].contiguous(); loss_fct = nn.CrossEntropyLoss(reduction='none'); losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(input_batch.size(0), -1); loss = losses.mean(); num_tokens = shift_labels.numel(); total_loss += loss.item() * num_tokens; total_tokens += num_tokens; pbar.set_description(f"PPL for {model_name} (Loss: {loss.item():.4f})")
    end_time = time.time(); ppl = math.exp(total_loss / total_tokens); elapsed_time = end_time - start_time; print(f"✅ Result for {model_name}: PPL = {ppl:.4f}, Time = {elapsed_time:.2f}s"); return ppl, elapsed_time

def main():
    p = argparse.ArgumentParser(description="Evaluate Layer-wise model with NO CACHING - All layers perform A@B@X matmul."); p.add_argument("--model_name", required=True); p.add_argument("--shared_path", required=True); p.add_argument("--bmap_path", required=True); p.add_argument("--original_weights_path", required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--trust_remote_code", action="store_true"); p.add_argument("--group_size", type=int, default=128); p.add_argument("--skip_gen", action="store_true"); p.add_argument("--gen_max_new_tokens", type=int, default=50); p.add_argument("--gen_repeats", type=int, default=1); p.add_argument("--use_cuda_w4a16", action="store_true"); p.add_argument("--batch_size", type=int, default=1); p.add_argument("--eval_seq_len", type=int, default=2048); args = p.parse_args()
    device = torch.device(args.device); tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    default_prompts = ["Hello, my name is", "The quick brown fox", "In a shocking finding, scientists discovered that"]
    print(f"📥 Loading original FP16 model: {args.model_name}"); model_fp16 = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=args.trust_remote_code)
    print(f"📥 Loading original weights from: {args.original_weights_path}"); original_weights = torch.load(args.original_weights_path, map_location="cpu", weights_only=True); model_fp16.load_state_dict(original_weights)
    if args.use_cuda_w4a16:
        print("🔄 Converting model to CUDA W4A16..."); from cuda_w4a16.linear import convert_to_cuda_w4a16; model = convert_to_cuda_w4a16(model_fp16, group_size=args.group_size).to(device); method_label = "CUDA W4A16"
    else:
        print(f"🔄 Converting to Triton 4-bit..."); model = convert_to_triton_4bit(model_fp16, group_size=args.group_size).to(device); method_label = "Triton 4-bit"
    del model_fp16, original_weights; gc.collect(); torch.cuda.empty_cache()
    print(f"🧩 Loading layer-wise correction artifacts..."); shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f: bmap = json.load(f)
    model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0); results = {}


    print("\n=== BASELINE EVALUATION (NO SVD CORRECTION) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection): module.alpha_svd = 0.0
    ppl_base, time_base = evaluate(model, tokenizer, device, f"{method_label} Baseline", batch_size=args.batch_size, seq_len=args.eval_seq_len)
    gen_metrics_base = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for baseline (Manual Loop)...")

        gen_metrics_base = measure_generation_with_manual_loop(model, tokenizer, device, prompts=default_prompts, max_new_tokens=args.gen_max_new_tokens, repeats=args.gen_repeats)
        print(f"   • Baseline TTFB: {gen_metrics_base['ttfb_ms_median']:.1f}ms (median)"); print(f"   • Baseline Throughput: {gen_metrics_base['tok_s_median']:.2f} tok/s (median)")
    results["baseline"] = {"ppl": ppl_base, "time": time_base, "generation_metrics": gen_metrics_base}


    print("\n=== SVD CORRECTION EVALUATION (ALPHA=1.0, NO CACHING) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection): module.alpha_svd = 1.0
    ppl, time_val = evaluate(model, tokenizer, device, f"{method_label} + SVD (NO CACHE)", batch_size=args.batch_size, seq_len=args.eval_seq_len)
    gen_metrics_svd = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for SVD correction (Manual Loop, NO CACHING)...")

        gen_metrics_svd = measure_generation_with_manual_loop(model, tokenizer, device, prompts=default_prompts, max_new_tokens=args.gen_max_new_tokens, repeats=args.gen_repeats)
        print(f"   • SVD TTFB: {gen_metrics_svd['ttfb_ms_median']:.1f}ms (median)"); print(f"   • SVD Throughput: {gen_metrics_svd['tok_s_median']:.2f} tok/s (median)")
    results["svd"] = {"ppl": ppl, "time": time_val, "generation_metrics": gen_metrics_svd}


    print(f"\n{'='*15} FINAL SUMMARY ({args.model_name} + Layer-wise + NO CACHING) {'='*15}"); print(f"Model: {args.model_name} (, Layer-wise, NO CACHING, Batch: {args.batch_size})")
    print("-" * 120); print(f"{'Method':<50} | {'Perplexity':<10} | {'Time (s)':<8} | {'TTFB(ms)':<10} | {'tok/s':<10}"); print("-" * 120)
    base_data = results["baseline"]; base_gen = base_data.get("generation_metrics"); ttfb_base_str = f"{base_gen['ttfb_ms_median']:.1f}" if base_gen else "-"; tok_s_base_str = f"{base_gen['tok_s_median']:.2f}" if base_gen else "-"; print(f"{method_label + ' Baseline (no SVD)':<50} | {base_data['ppl']:<10.4f} | {base_data['time']:<8.2f} | {ttfb_base_str:<10} | {tok_s_base_str:<10}")
    svd_data = results["svd"]; svd_gen = svd_data.get("generation_metrics"); ttfb_svd_str = f"{svd_gen['ttfb_ms_median']:.1f}" if svd_gen else "-"; tok_s_svd_str = f"{svd_gen['tok_s_median']:.2f}" if svd_gen else "-"; print(f"{method_label + ' + SVD (α=1.0, NO CACHE)':<50} | {svd_data['ppl']:<10.4f} | {svd_data['time']:<8.2f} | {ttfb_svd_str:<10} | {tok_s_svd_str:<10}")
    print("=" * 120)

if __name__ == "__main__":
    main()

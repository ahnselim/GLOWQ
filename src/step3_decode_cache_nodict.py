"""
src/step3_decode_cache_nodict.py
Evaluates dict-free decode-cache SVD correction with baseline/SVD PPL and generation performance comparisons.
output :
(stdout only)
|-- Baseline metrics         (PPL, timing, generation metrics)
`-- SVD-corrected metrics    (PPL, timing, generation metrics)
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, os, time, re, sys
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset




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


def _configure_cuda_w4a16_env(args) -> None:
    if not getattr(args, "use_cuda_w4a16", False):
        return

    # Keep old step3 behavior (fp16 activations) while using the new QM-style kernel flow.
    os.environ.setdefault("W4A16_KERNEL_OUT_DTYPE", "fp16")
    os.environ.setdefault("W4A16_GEMM_CUDA", "1")
    os.environ.setdefault("W4A16_DEQUANT_CACHE", "0")

    if getattr(args, "cuda_w4a16_kernel_out_dtype", None):
        os.environ["W4A16_KERNEL_OUT_DTYPE"] = args.cuda_w4a16_kernel_out_dtype
    if getattr(args, "cuda_w4a16_gemv_m_max", None) is not None:
        os.environ["W4A16_GEMV_M_MAX"] = str(int(args.cuda_w4a16_gemv_m_max))
    if getattr(args, "cuda_w4a16_dequant_chunk", None) is not None:
        os.environ["W4A16_DEQUANT_CHUNK"] = str(int(args.cuda_w4a16_dequant_chunk))
    if getattr(args, "cuda_w4a16_dequant_cache", False):
        os.environ["W4A16_DEQUANT_CACHE"] = "1"
    if getattr(args, "cuda_w4a16_force_gemm", False):
        os.environ["W4A16_FORCE_GEMM"] = "1"
        os.environ.pop("W4A16_FORCE_GEMV", None)
    if getattr(args, "cuda_w4a16_force_gemv", False):
        os.environ["W4A16_FORCE_GEMV"] = "1"
        os.environ.pop("W4A16_FORCE_GEMM", None)

    print(
        "[CUDA W4A16] env:"
        f" out_dtype={os.environ.get('W4A16_KERNEL_OUT_DTYPE')}"
        f", gemv_m_max={os.environ.get('W4A16_GEMV_M_MAX', '(default)')}"
        f", force_gemm={os.environ.get('W4A16_FORCE_GEMM', '0')}"
        f", force_gemv={os.environ.get('W4A16_FORCE_GEMV', '0')}"
        f", dequant_chunk={os.environ.get('W4A16_DEQUANT_CHUNK', '(default)')}"
        f", dequant_cache={os.environ.get('W4A16_DEQUANT_CACHE', '0')}"
    )

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
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        qweight_ptrs = qweight_ptr + (
            offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 2) * stride_qwk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_start = k * BLOCK_SIZE_K
            k_offs = k_start + offs_k

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
        if x.dim() == 3:
            M, K = original_shape[0] * original_shape[1], original_shape[2]
        else:
            M, K = x.shape

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
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size

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

            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
            else:
                self.bias = None

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
                W = torch.nn.functional.pad(W, (0, group_size - (I % group_size)))

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

            low_w = quant_values[:, :, 0::2]
            high_w = quant_values[:, :, 1::2]
            packed_weights = (high_w << 4) | low_w
            qlayer.qweight.data.copy_(packed_weights.reshape(O, I_padded // 2))

            qlayer.scales.data.copy_(scales.to(torch.float16))

            zeros_uint8 = zeros_float.to(torch.uint8)
            low_z = zeros_uint8[:, 0::2]
            high_z = zeros_uint8[:, 1::2]
            packed_zeros = (high_z << 4) | low_z
            qlayer.qzeros.data.copy_(packed_zeros)

            if linear_layer.bias is not None:
                qlayer.bias.data.copy_(linear_layer.bias.data)

            return qlayer

    def convert_to_triton_4bit(model, group_size=128):
        TARGET_LAYERS = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        ]
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(
                name.endswith(target) for target in TARGET_LAYERS
            ):
                parent, attr_name = get_parent_module(model, name)
                new_module = TritonTrue4BitLinear.from_float(module, group_size)
                setattr(parent, attr_name, new_module)
        gc.collect()
        torch.cuda.empty_cache()
        return model







def _cuda_sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def temp_generation_overrides(model, **overrides):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        yield
        return
    old_vals = {k: getattr(gen_cfg, k, None) for k in overrides}
    for k, v in overrides.items():
        try:
            setattr(gen_cfg, k, v)
        except:
            pass
    try:
        yield
    finally:
        for k, v in old_vals.items():
            try:
                setattr(gen_cfg, k, v)
            except:
                pass


def _get_sequences_from_generate(output):
    return output.sequences if hasattr(output, "sequences") else output







def get_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]





def clear_group_cache():
    
    pass





class MiniGroupCache:
    __slots__ = ("r", "valid", "uses_left")

    def __init__(self):
        self.r: Optional[torch.Tensor] = None
        self.valid: bool = False
        self.uses_left: int = 0

    def set(self, r: torch.Tensor, uses: int):
        self.r = r
        self.valid = True
        self.uses_left = uses

    def consume(self):
        if self.valid and self.uses_left > 0:
            self.uses_left -= 1
            out = self.r
            if self.uses_left == 0:
                
                self.valid = False
                self.r = None
            return out, True
        return None, False

    def clear(self):
        self.r = None
        self.valid = False
        self.uses_left = 0





class AddSVDCorrection(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        role: str,  
        is_group: bool,
        group_cache: Optional[MiniGroupCache],
        alpha_svd: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.role = role
        self.is_group = is_group
        self.group_cache = group_cache
        self.alpha_svd = alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)

        if self.alpha_svd == 0.0:
            return z

        A_q_dev = self.A_q
        B_q_dev = self.B_q

        
        try:
            if self.is_group:
                if self.role in ("q", "gate"):
                    
                    r = F.linear(x, B_q_dev)
                    uses = 2 if self.role == "q" else 1  
                    if self.group_cache is not None:
                        self.group_cache.set(r, uses)
                    intermediate_r = r
                elif self.role in ("k", "v", "up"):
                    
                    r, ok = (
                        self.group_cache.consume()
                        if self.group_cache is not None
                        else (None, False)
                    )
                    if not ok or r is None:
                        r = F.linear(x, B_q_dev)
                    intermediate_r = r
                else:
                    
                    intermediate_r = F.linear(x, B_q_dev)
            else:
                
                intermediate_r = F.linear(x, B_q_dev)

            svd_raw = F.linear(intermediate_r, A_q_dev)

            if z.shape != svd_raw.shape:
                svd_raw = svd_raw.reshape(z.shape)

            return z.add_(svd_raw, alpha=self.alpha_svd)
        except RuntimeError as e:
            if "size of tensor" in str(e) or "invalid for input of size" in str(e):
                return z
            else:
                raise e


def _role_from_suffix(sfx: str) -> str:
    
    if sfx.endswith("q_proj"):
        return "q"
    if sfx.endswith("k_proj"):
        return "k"
    if sfx.endswith("v_proj"):
        return "v"
    if sfx.endswith("gate_proj"):
        return "gate"
    if sfx.endswith("up_proj"):
        return "up"
    
    return "solo"


def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):
    patched_count, skipped_count = 0, 0
    
    gkey2cache = {}

    for weight_name, bkey in tqdm(bmap.items(), desc="Patching SVD Correction"):
        module_name = weight_name.replace(".weight", "")
        B_q = shared.get(bkey)
        is_group = "B_shared" in bkey

        if is_group:
            gkey = bkey.replace(".B_shared", "")
            module_suffix = module_name.split(".")[-1]
            a_key = f"{gkey}.{module_suffix}.A"
            role = _role_from_suffix(module_suffix)
            cache = gkey2cache.setdefault(gkey, MiniGroupCache())
        else:
            
            if m := re.match(r"(model\.layers\.\d+\..*?)\.B", bkey):
                gkey = m.group(1)
            else:
                gkey = bkey.replace(".B", "")
            a_key = gkey + ".A"
            role = "solo"
            cache = None

        A_q = shared.get(a_key)
        if A_q is None or B_q is None:
            skipped_count += 1
            continue
        try:
            parent, attr_name = get_parent_module(model, module_name)
            inner = getattr(parent, attr_name)

            types_list = []
            try:
                from cuda_w4a16.linear import CudaW4A16Linear

                types_list.append(CudaW4A16Linear)
            except Exception:
                pass
            try:
                types_list.append(TritonTrue4BitLinear)
            except Exception:
                pass
            valid_types = tuple(types_list) if types_list else (nn.Module,)
            if not isinstance(inner, valid_types):
                skipped_count += 1
                continue

            wrapped = AddSVDCorrection(
                inner,
                A_q,
                B_q,
                role=role,
                is_group=is_group,
                group_cache=cache,
                alpha_svd=alpha_svd,
            )
            setattr(parent, attr_name, wrapped)
            patched_count += 1
        except AttributeError as e:
            print(f"AttributeError for {module_name}: {e}")
            skipped_count += 1
            continue
    print(
        f"SVD Correction Patching Summary: {patched_count} patched, {skipped_count} skipped"
    )
    return model


@torch.no_grad()
def measure_generation_metrics(
    model,
    tokenizer,
    device,
    prompts,
    max_new_tokens=50,
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_p=1.0,
    repeats=1,
):
    model.eval()
    pad_id = tokenizer.eos_token_id
    override_kwargs = {"temperature": 1.0, "top_p": 1.0} if not do_sample else {}
    with temp_generation_overrides(model, **override_kwargs):
        try:
            warmup_inputs = tokenizer(
                prompts[0], return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            if warmup_inputs["input_ids"].dim() == 1:
                warmup_inputs["input_ids"] = warmup_inputs["input_ids"].unsqueeze(0)
            if (
                "attention_mask" in warmup_inputs
                and warmup_inputs["attention_mask"].dim() == 1
            ):
                warmup_inputs["attention_mask"] = warmup_inputs[
                    "attention_mask"
                ].unsqueeze(0)
            _ = model.generate(**warmup_inputs, max_new_tokens=1, use_cache=True)
            _cuda_sync(device)
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")
        ttfb_list_ms, tokps_list, total_times, total_new_tokens = [], [], [], []
        gen_kwargs = dict(
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=pad_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        for _ in range(repeats):
            for prompt in prompts:
                clear_group_cache()
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                try:
                    if inputs["input_ids"].dim() == 1:
                        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
                    if (
                        "attention_mask" in inputs
                        and inputs["attention_mask"].dim() == 1
                    ):
                        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)

                    _cuda_sync(device)
                    t0 = perf_counter()
                    model.generate(**inputs, max_new_tokens=1, **gen_kwargs)
                    _cuda_sync(device)
                    ttfb_list_ms.append((perf_counter() - t0) * 1000.0)

                    clear_group_cache()

                    _cuda_sync(device)
                    t1 = perf_counter()
                    outN = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, **gen_kwargs
                    )
                    _cuda_sync(device)
                    t_total = perf_counter() - t1
                    gen_tokens = (
                        _get_sequences_from_generate(outN).shape[1]
                        - inputs["input_ids"].shape[1]
                    )
                    tokps_list.append((gen_tokens / t_total) if t_total > 0 else 0.0)
                    total_times.append(t_total)
                    total_new_tokens.append(gen_tokens)
                except Exception as e:
                    print(
                        f"Warning: Generation measurement failed for prompt '{prompt[:30]}...': {e}"
                    )
                    continue
    return {
        "ttfb_ms_mean": mean(ttfb_list_ms) if ttfb_list_ms else 0,
        "ttfb_ms_median": median(ttfb_list_ms) if ttfb_list_ms else 0,
        "tok_s_mean": mean(tokps_list) if tokps_list else 0,
        "tok_s_median": median(tokps_list) if tokps_list else 0,
        "avg_total_time_s": mean(total_times) if total_times else 0,
        "avg_new_tokens": mean(total_new_tokens) if total_new_tokens else 0,
    }


@torch.no_grad()
def evaluate(model, tokenizer, device, model_name):
    print(f"\n--- Evaluating {model_name} ---")
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
        clear_group_cache()
        begin_loc, end_loc = i, min(i + 2048, seq_len)
        if end_loc - begin_loc <= 1:
            continue
        input_batch = input_ids[:, begin_loc:end_loc].to(device)
        labels = input_batch
        outputs = model(input_batch)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        num_tokens = shift_labels.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        pbar.set_description(f"PPL for {model_name} (Loss: {loss.item():.4f})")
    end_time = time.time()
    ppl = math.exp(total_loss / total_tokens)
    elapsed_time = end_time - start_time
    print(f"✅ Result for {model_name}: PPL = {ppl:.4f}, Time = {elapsed_time:.2f}s")
    return ppl, elapsed_time







def main():
    p = argparse.ArgumentParser(
        description="Evaluate quantized model with Shared-B correction and dict-free fixed reuse."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--shared_path", required=True)
    p.add_argument("--bmap_path", required=True)
    p.add_argument(
        "--original_weights_path",
        required=True,
        help="Path to original weights (from step1)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Required for some model families",
    )
    p.add_argument(
        "--group_size", type=int, default=128, help="Group size for Triton quantization"
    )
    p.add_argument(
        "--skip_gen",
        action="store_true",
        help="Skip generation latency/throughput measurement",
    )
    p.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=50,
        help="New tokens for throughput measurement",
    )
    p.add_argument(
        "--gen_repeats",
        type=int,
        default=1,
        help="Repeat generation measurement this many times",
    )
    p.add_argument(
        "--gen_do_sample", action="store_true", help="Use sampling for generation"
    )
    p.add_argument(
        "--gen_num_beams", type=int, default=1, help="Beam size for generation"
    )
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--gen_top_p", type=float, default=1.0)
    p.add_argument(
        "--use_cuda_w4a16",
        action="store_true",
        help="Use custom CUDA W4A16 kernels instead of Triton",
    )
    p.add_argument(
        "--cuda_w4a16_kernel_out_dtype",
        type=str,
        default=None,
        choices=["fp16", "fp32"],
        help="Override W4A16 kernel output dtype (default step3 sets fp16)",
    )
    p.add_argument(
        "--cuda_w4a16_gemv_m_max",
        type=int,
        default=None,
        help="Prefer GEMV when M<=this threshold (W4A16_GEMV_M_MAX)",
    )
    p.add_argument(
        "--cuda_w4a16_dequant_chunk",
        type=int,
        default=None,
        help="Chunk size for W4A16 dequant+matmul fallback/GEMM path",
    )
    p.add_argument(
        "--cuda_w4a16_dequant_cache",
        action="store_true",
        help="Enable fp16 dequant weight cache in W4A16 path",
    )
    p.add_argument(
        "--cuda_w4a16_force_gemm",
        action="store_true",
        help="Force W4A16 GEMM/dequant path",
    )
    p.add_argument(
        "--cuda_w4a16_force_gemv",
        action="store_true",
        help="Force W4A16 GEMV path",
    )
    p.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use fast tokenizer (default False to match eval_fp16_ppl.py)",
    )

    args = p.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    default_prompts = [
        "Hello, my name is",
        "The quick brown fox",
        "In a shocking finding, scientists discovered that",
    ]
    print(f"📥 Loading original FP16 model for baseline comparison: {args.model_name}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    print(f"📥 Loading original weights from: {args.original_weights_path}")
    original_weights = torch.load(
        args.original_weights_path, map_location="cpu", weights_only=True
    )
    model_fp16.load_state_dict(original_weights)
    if args.use_cuda_w4a16:
        _configure_cuda_w4a16_env(args)
        print("🔄 Converting model to CUDA W4A16...")
        try:
            from cuda_w4a16.linear import convert_to_cuda_w4a16
        except Exception as e:
            raise RuntimeError(f"Failed to import CUDA W4A16 module: {e}")
        model = convert_to_cuda_w4a16(model_fp16, group_size=args.group_size).to(device)
        method_label = "CUDA W4A16"
    else:
        print(f"🔄 Converting original {args.model_name} model to Triton 4-bit...")
        model = convert_to_triton_4bit(model_fp16, group_size=args.group_size).to(
            device
        )
        method_label = "Triton 4-bit"
    del model_fp16, original_weights
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"🧩 Loading correction artifacts and patching wrappers for {args.model_name}..."
    )
    shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)
    model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)
    results = {}

    
    print("\n=== BASELINE EVALUATION (NO SVD CORRECTION) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 0.0
    ppl_base, time_base = evaluate(
        model,
        tokenizer,
        device,
        f"{method_label} Original Weights ONLY ({args.model_name})",
    )
    gen_metrics_base = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for baseline...")
        try:
            gen_metrics_base = measure_generation_metrics(
                model,
                tokenizer,
                device,
                prompts=default_prompts,
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=args.gen_do_sample,
                num_beams=args.gen_num_beams,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repeats=args.gen_repeats,
            )
            print(
                f"   • Baseline TTFB: {gen_metrics_base['ttfb_ms_median']:.1f}ms (median)"
            )
            print(
                f"   • Baseline Throughput: {gen_metrics_base['tok_s_median']:.2f} tok/s (median)"
            )
        except Exception as e:
            print(f"Generation measurement failed for baseline: {e}")
            gen_metrics_base = None
    results["baseline"] = {
        "ppl": ppl_base,
        "time": time_base,
        "generation_metrics": gen_metrics_base,
    }

    
    print("\n=== SVD CORRECTION EVALUATION (ALPHA=1.0) ===")
    for module in model.modules():
        if isinstance(module, AddSVDCorrection):
            module.alpha_svd = 1.0
    ppl, time_val = evaluate(
        model,
        tokenizer,
        device,
        f"{method_label} Original Weights + SVD Correction (SVD α=1.0, {args.model_name})",
    )
    gen_metrics_svd = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for SVD correction...")
        try:
            gen_metrics_svd = measure_generation_metrics(
                model,
                tokenizer,
                device,
                prompts=default_prompts,
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=args.gen_do_sample,
                num_beams=args.gen_num_beams,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repeats=args.gen_repeats,
            )
            print(f"   • SVD TTFB: {gen_metrics_svd['ttfb_ms_median']:.1f}ms (median)")
            print(
                f"   • SVD Throughput: {gen_metrics_svd['tok_s_median']:.2f} tok/s (median)"
            )
        except Exception as e:
            print(f"Generation measurement failed for SVD: {e}")
            gen_metrics_svd = None
    results["svd"] = {
        "ppl": ppl,
        "time": time_val,
        "generation_metrics": gen_metrics_svd,
    }

    
    print(f"\n{'='*15} FINAL SUMMARY ({args.model_name} + SVD Correction) {'='*15}")
    print(f"Model: {args.model_name}")
    print("-" * 80)
    print(
        f"{'Method':<50} | {'Perplexity':<10} | {'Time (s)':<8} | {'TTFB(ms)':<10} | {'tok/s':<10}"
    )
    print("-" * 80)
    svd_data = results["svd"]
    svd_gen = svd_data.get("generation_metrics")
    ttfb_svd_str = f"{svd_gen['ttfb_ms_median']:.1f}" if svd_gen else "-"
    tok_s_svd_str = f"{svd_gen['tok_s_median']:.2f}" if svd_gen else "-"
    print(
        f"{method_label + ' + SVD Correction (α=1.0)':<50} | {svd_data['ppl']:<10.4f} | {svd_data['time']:<8.2f} | {ttfb_svd_str:<10} | {tok_s_svd_str:<10}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

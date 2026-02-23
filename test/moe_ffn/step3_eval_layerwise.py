"""
moe_ffn/step3_eval_layerwise.py
Evaluates MoE fake-quant models with layer-wise SVD correction (without Shared-B sharing).
output :
(no output files)
`-- stdout/stderr metrics and logs only
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, time
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Optional


TARGET_SUFFIXES = {
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
}




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
        except Exception:
            pass
    try:
        yield
    finally:
        for k, v in old_vals.items():
            try:
                setattr(gen_cfg, k, v)
            except Exception:
                pass


def _get_sequences_from_generate(output):
    return output.sequences if hasattr(output, "sequences") else output


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
                        inputs["attention_mask"] = inputs[
                            "attention_mask"
                        ].unsqueeze(0)
                    _cuda_sync(device)

                    t0 = perf_counter()
                    model.generate(**inputs, max_new_tokens=1, **gen_kwargs)
                    _cuda_sync(device)
                    ttfb_list_ms.append((perf_counter() - t0) * 1000.0)


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
                    tokps = (gen_tokens / t_total) if t_total > 0 else 0.0
                    tokps_list.append(tokps)
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





def get_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _is_target_module_name(name: str) -> bool:
    if not name:
        return False
    if not (
        "layers" in name or "encoder.layers" in name or "model.layers" in name
    ):
        return False
    return name.split(".")[-1] in TARGET_SUFFIXES


@torch.no_grad()
def fake_quantize_activation(
    x: torch.Tensor, group_size: int = 128, num_bits: int = 8
) -> torch.Tensor:
    if num_bits <= 0:
        return x
    if x.dim() == 0:
        return x
    original_shape = x.shape
    last_dim = original_shape[-1]
    if last_dim == 0:
        return x
    x_2d = x.reshape(-1, last_dim)
    pad = (group_size - (last_dim % group_size)) % group_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad), mode="constant", value=0.0)
    x_2d = x_2d.to(torch.float32)
    grouped = x_2d.view(-1, group_size)

    min_vals = grouped.min(dim=-1, keepdim=True).values
    max_vals = grouped.max(dim=-1, keepdim=True).values
    qmax = (1 << num_bits) - 1
    ranges = max_vals - min_vals
    scales = torch.clamp(ranges / max(qmax, 1), min=1e-8)
    zeros = torch.round(-min_vals / scales).clamp(0, qmax)
    quant = torch.round(grouped / scales + zeros).clamp(0, qmax)
    dequant = (quant - zeros) * scales

    tiny_mask = (ranges < 1e-8).expand_as(grouped)
    if tiny_mask.any():
        dequant = torch.where(tiny_mask, grouped, dequant)

    dequant = dequant.view(-1, group_size)
    if pad:
        dequant = dequant[:, :last_dim]
    return dequant.reshape(*original_shape).to(x.dtype)


class ActivationFakeQuantWrapper(nn.Module):
    def __init__(self, inner: nn.Module, act_bits: int = 8, group_size: int = 128):
        super().__init__()
        self.inner = inner
        self.act_bits = act_bits
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = fake_quantize_activation(
            x, group_size=self.group_size, num_bits=self.act_bits
        )
        return self.inner(qx)


def _unwrap_base_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, (AddSVDCorrection, ActivationFakeQuantWrapper)):
        module = module.inner
    return module


@torch.no_grad()
def apply_quantized_weights(model, qweights):
    injected, missing, mismatch = 0, 0, 0
    for wkey, Wq in qweights.items():
        if not (
            isinstance(wkey, str)
            and wkey.endswith(".weight")
            and getattr(Wq, "ndim", 0) == 2
        ):
            continue
        module_name = wkey[:-7]
        try:
            parent, attr_name = get_parent_module(model, module_name)
        except AttributeError:
            missing += 1
            continue
        target = getattr(parent, attr_name, None)
        if target is None:
            missing += 1
            continue
        inner = _unwrap_base_linear(target)
        if not hasattr(inner, "weight"):
            missing += 1
            continue
        if inner.weight.shape != Wq.shape:
            mismatch += 1
            continue
        inner.weight.data.copy_(
            Wq.to(device=inner.weight.device, dtype=inner.weight.dtype)
        )
        injected += 1
    print(
        f"[FakeQuant] injected={injected}, missing={missing}, shape_mismatch={mismatch}"
    )


@torch.no_grad()
def apply_activation_fake_quant(
    model: nn.Module,
    module_names,
    act_bits: int = 8,
    group_size: int = 128,
):
    if act_bits <= 0:
        return model
    unique_modules = sorted(set(module_names))
    wrapped = 0
    for module_name in unique_modules:
        if not _is_target_module_name(module_name):
            continue
        try:
            parent, attr_name = get_parent_module(model, module_name)
        except AttributeError:
            continue
        current = getattr(parent, attr_name, None)
        if current is None:
            continue
        if isinstance(current, ActivationFakeQuantWrapper):
            current.act_bits = act_bits
            current.group_size = group_size
            continue
        setattr(
            parent,
            attr_name,
            ActivationFakeQuantWrapper(
                current, act_bits=act_bits, group_size=group_size
            ),
        )
        wrapped += 1
    print(
        f"[Activation Quant] Wrapped {wrapped} modules with fake int{act_bits} activation quantization."
    )
    return model





class AddSVDCorrection(nn.Module):


    def __init__(
        self,
        inner: nn.Module,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        alpha_svd: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.alpha_svd = alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha_svd == 0.0:
            return z

        A_q_dev = self.A_q
        B_q_dev = self.B_q
        try:

            intermediate_r = F.linear(x, B_q_dev)
            svd_raw = F.linear(intermediate_r, A_q_dev)


            if z.shape != svd_raw.shape:
                if len(z.shape) == len(svd_raw.shape):
                    if z.shape[:-1] == svd_raw.shape[:-1]:
                        min_last_dim = min(z.shape[-1], svd_raw.shape[-1])
                        svd_raw = svd_raw[..., :min_last_dim]
                        if z.shape[-1] > min_last_dim:
                            pad_size = z.shape[-1] - min_last_dim
                            svd_raw = F.pad(svd_raw, (0, pad_size))
                    else:
                        if svd_raw.numel() == z.numel():
                            svd_raw = svd_raw.reshape(z.shape)
                        else:
                            return z
                else:
                    if svd_raw.numel() == z.numel():
                        svd_raw = svd_raw.reshape(z.shape)
                    else:
                        return z

            return z.add_(svd_raw, alpha=self.alpha_svd)
        except RuntimeError:
            return z
        except Exception:
            return z


def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):

    patched_count = 0
    skipped_count = 0
    missing_B = 0
    missing_A = 0
    missing_mod = 0
    missing_weight_attr = 0


    print(f"[LayerwisePatch] shared tensors: {len(shared)}")
    print(f"[LayerwisePatch] bmap entries: {len(bmap)}")
    sample_shared_keys = list(shared.keys())[:5]
    sample_bmap = list(bmap.items())[:5]
    print("[LayerwisePatch] sample shared keys:", sample_shared_keys)
    print("[LayerwisePatch] sample bmap items:", sample_bmap)

    for weight_name, bkey in tqdm(bmap.items(), desc="Patching SVD Correction (layerwise)"):


        module_name = weight_name.replace(".weight", "")
        base = module_name




        candidate_B_keys = []
        if isinstance(bkey, str):
            candidate_B_keys.append(bkey)
        candidate_B_keys.append(f"{base}.B")
        candidate_B_keys.append(f"{base}.B_shared")

        B_key_used = None
        for cand in candidate_B_keys:
            if cand in shared:
                B_key_used = cand
                break

        if B_key_used is None:
            missing_B += 1
            if missing_B <= 5:
                print(f"[LayerwisePatch][WARN] No B found for {weight_name}. tried={candidate_B_keys}")
            skipped_count += 1
            continue

        B_q = shared[B_key_used]




        candidate_A_keys = []

        candidate_A_keys.append(
            B_key_used.replace(".B_shared", ".A_shared").replace(".B", ".A")
        )

        candidate_A_keys.append(f"{base}.A")

        A_key_used = None
        for cand in candidate_A_keys:
            if cand in shared:
                A_key_used = cand
                break

        if A_key_used is None:
            missing_A += 1
            if missing_A <= 5:
                print(f"[LayerwisePatch][WARN] No A found for {weight_name}. tried={candidate_A_keys}")
            skipped_count += 1
            continue

        A_q = shared[A_key_used]




        try:
            parent, attr_name = get_parent_module(model, module_name)
            current = getattr(parent, attr_name)
        except AttributeError as e:
            missing_mod += 1
            if missing_mod <= 5:
                print(f"[LayerwisePatch][WARN] AttributeError for {module_name}: {e}")
            skipped_count += 1
            continue

        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            missing_weight_attr += 1
            if missing_weight_attr <= 5:
                print(
                    f"[LayerwisePatch][WARN] Target module has no 'weight': "
                    f"{module_name} / {type(inner)}"
                )
            skipped_count += 1
            continue

        wrapped = AddSVDCorrection(inner, A_q, B_q, alpha_svd=alpha_svd)
        if isinstance(current, ActivationFakeQuantWrapper):
            current.inner = wrapped
            setattr(parent, attr_name, current)
        else:
            setattr(parent, attr_name, wrapped)
        patched_count += 1

    print(
        f"SVD Correction Patching Summary (layerwise): "
        f"patched={patched_count}, skipped={skipped_count}, "
        f"missing_B={missing_B}, missing_A={missing_A}, "
        f"missing_mod={missing_mod}, no_weight_attr={missing_weight_attr}"
    )
    return model






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





def safe_percentage_change(new_val, old_val):
    if old_val == 0 or old_val is None:
        if new_val == 0 or new_val is None:
            return 0.0
        else:
            return float("inf") if new_val > 0 else float("-inf")
    return ((new_val - old_val) / old_val) * 100.0





def main():
    p = argparse.ArgumentParser(
        description="Evaluate FakeQuant model with Layerwise SVD correction (no Shared-B)."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--shared_path", required=True)
    p.add_argument("--bmap_path", required=True)
    p.add_argument(
        "--original_weights_path",
        required=True,
        help="Path to original weights (from step1, FP16 state_dict).",
    )
    p.add_argument(
        "--quantized_weights_path",
        required=True,
        help="Path to fake-quant (dequantized) weights saved in step1 (subset dict).",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--trust_remote_code", action="store_true", help="Required for some model families"
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
        "--activation_bits",
        type=int,
        default=8,
        help="Bit-width for fake activation quantization (set <=0 to disable).",
    )
    p.add_argument(
        "--activation_group_size",
        type=int,
        default=128,
        help="Group size for activation fake quantization.",
    )

    args = p.parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    default_prompts = [
        "Hello, my name is",
        "The quick brown fox",
        "In a shocking finding, scientists discovered that",
    ]


    print(f"📥 Loading FP16 model for baseline comparison: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    print(f"📥 Loading original weights from: {args.original_weights_path}")
    original_weights = torch.load(
        args.original_weights_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(original_weights)
    del original_weights
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device)

    results = {}

    print("\n=== FP16 BASELINE EVALUATION (ORIGINAL MODEL) ===")
    ppl_fp16, time_fp16 = evaluate(
        model, tokenizer, device, f"FP16 (Original, {args.model_name})"
    )
    gen_metrics_fp16 = None
    if not args.skip_gen:
        print("Measuring generation metrics for FP16 baseline...")
        try:
            gen_metrics_fp16 = measure_generation_metrics(
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
                f"   • FP16 TTFB: {gen_metrics_fp16['ttfb_ms_median']:.1f}ms (median)"
            )
            print(
                f"   • FP16 Throughput: {gen_metrics_fp16['tok_s_median']:.2f} tok/s (median)"
            )
        except Exception as e:
            print(f"Generation measurement failed for FP16 baseline: {e}")
            gen_metrics_fp16 = None

    results["fp16"] = {
        "ppl": ppl_fp16,
        "time": time_fp16,
        "generation_metrics": gen_metrics_fp16,
    }


    method_label = "Fake-Quant W4A? GEMM (Layerwise A/B)"

    print(f"\n📦 Loading fake-quant weights from: {args.quantized_weights_path}")
    fake_quant_weights = torch.load(
        args.quantized_weights_path, map_location="cpu", weights_only=True
    )
    apply_quantized_weights(model, fake_quant_weights)
    del fake_quant_weights
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧩 Loading layerwise correction artifacts for {args.model_name}...")
    shared = torch.load(args.shared_path, map_location=device, weights_only=True)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)

    module_names = [name.replace(".weight", "") for name in bmap.keys()]
    if args.activation_bits > 0:
        apply_activation_fake_quant(
            model,
            module_names,
            act_bits=args.activation_bits,
            group_size=args.activation_group_size,
        )
    model = patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)


    print("\n=== BASELINE EVALUATION (FAKE-QUANT, NO SVD) ===")
    for m in model.modules():
        if isinstance(m, AddSVDCorrection):
            m.alpha_svd = 0.0
    ppl_base, time_base = evaluate(
        model, tokenizer, device, f"{method_label} (No SVD, {args.model_name})"
    )
    gen_metrics_base = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for fake-quant baseline...")
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
            print(f"Generation measurement failed for fake-quant baseline: {e}")
            gen_metrics_base = None
    results["baseline"] = {
        "ppl": ppl_base,
        "time": time_base,
        "generation_metrics": gen_metrics_base,
    }


    print("\n=== SVD CORRECTION EVALUATION (ALPHA=1.0, LAYERWISE) ===")
    for m in model.modules():
        if isinstance(m, AddSVDCorrection):
            m.alpha_svd = 1.0
    ppl_svd, time_svd = evaluate(
        model,
        tokenizer,
        device,
        f"{method_label} + SVD Correction (α=1.0, {args.model_name})",
    )
    gen_metrics_svd = None
    if not args.skip_gen:
        print(f"Measuring generation metrics for SVD correction (layerwise)...")
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
        "ppl": ppl_svd,
        "time": time_svd,
        "generation_metrics": gen_metrics_svd,
    }


    print(
        f"\n{'='*15} FINAL SUMMARY ({args.model_name} + Fake-Quant + Layerwise A/B) {'='*15}"
    )
    print(f"Model: {args.model_name} (FP16 vs Fake-Quant vs Fake-Quant+SVD, Layerwise)")
    print("-" * 120)
    print(
        f"{'Method':<50} | {'Perplexity':<10} | {'Time (s)':<8} | {'TTFB(ms)':<10} | {'tok/s':<10}"
    )
    print("-" * 120)


    fp16_data = results["fp16"]
    fp16_gen = fp16_data.get("generation_metrics")
    ttfb_fp16_str = f"{fp16_gen['ttfb_ms_median']:.1f}" if fp16_gen else "-"
    tok_s_fp16_str = f"{fp16_gen['tok_s_median']:.2f}" if fp16_gen else "-"
    print(
        f"{'FP16 (original)':<50} | {fp16_data['ppl']:<10.4f} | {fp16_data['time']:<8.2f} | {ttfb_fp16_str:<10} | {tok_s_fp16_str:<10}"
    )


    base_data = results["baseline"]
    base_gen = base_data.get("generation_metrics")
    ttfb_base_str = f"{base_gen['ttfb_ms_median']:.1f}" if base_gen else "-"
    tok_s_base_str = f"{base_gen['tok_s_median']:.2f}" if base_gen else "-"
    print(
        f"{method_label + ' Baseline (no SVD)':<50} | {base_data['ppl']:<10.4f} | {base_data['time']:<8.2f} | {ttfb_base_str:<10} | {tok_s_base_str:<10}"
    )


    svd_data = results["svd"]
    svd_gen = svd_data.get("generation_metrics")
    ttfb_svd_str = f"{svd_gen['ttfb_ms_median']:.1f}" if svd_gen else "-"
    tok_s_svd_str = f"{svd_gen['tok_s_median']:.2f}" if svd_gen else "-"
    print(
        f"{method_label + ' + SVD Correction (α=1.0)':<50} | {svd_data['ppl']:<10.4f} | {svd_data['time']:<8.2f} | {ttfb_svd_str:<10} | {tok_s_svd_str:<10}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()

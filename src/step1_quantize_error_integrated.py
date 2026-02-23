"""
src/step1_quantize_error_integrated.py
Computes integrated Triton 4-bit quantization error tensors from original weights and saves artifacts for downstream correction.
output :
(user-specified output paths)
|-- <out_quant_err>           (.pt)
`-- <out_original_weights>    (.pt)
"""

import os, gc, re, torch, argparse, math
import torch.nn as nn
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
            ) * scales.to(x.dtype)
            accumulator += tl.dot(x, dequant_weights)

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
            low_z, high_z = zeros_uint8[:, 0::2], zeros_uint8[:, 1::2]
            packed_zeros = (high_z << 4) | low_z
            qlayer.qzeros.data.copy_(packed_zeros)

            if linear_layer.bias is not None:
                qlayer.bias.data.copy_(linear_layer.bias.data)
            return qlayer





@torch.no_grad()
def dequantize_from_triton_layer(triton_layer: "TritonTrue4BitLinear") -> torch.Tensor:
    qweight, scales, qzeros = (
        triton_layer.qweight,
        triton_layer.scales,
        triton_layer.qzeros,
    )
    O, I_packed = qweight.shape
    I_padded = I_packed * 2

    low_nibble_w = qweight & 0x0F
    high_nibble_w = qweight >> 4
    unpacked_w = torch.stack((low_nibble_w, high_nibble_w), dim=-1).view(O, I_padded)

    low_nibble_z = qzeros & 0x0F
    high_nibble_z = qzeros >> 4
    unpacked_z = torch.stack((low_nibble_z, high_nibble_z), dim=-1).view(O, -1)

    W_grouped = unpacked_w.reshape(
        O, I_padded // triton_layer.group_size, triton_layer.group_size
    )
    dequant_grouped = (
        W_grouped.to(scales.dtype) - unpacked_z.unsqueeze(-1)
    ) * scales.unsqueeze(-1)

    Wq_padded = dequant_grouped.reshape(O, I_padded)
    Wq = Wq_padded[:, : triton_layer.in_features].contiguous()
    return Wq.to(scales.dtype)


@torch.no_grad()
def get_triton_dequantized_weight(
    Ws: torch.Tensor, device: torch.device, group_size: int
) -> torch.Tensor:
    in_features, out_features = Ws.shape[1], Ws.shape[0]
    temp_linear_fp16 = nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=torch.float16
    )
    temp_linear_fp16.weight.data = Ws
    temp_linear_triton = TritonTrue4BitLinear.from_float(
        temp_linear_fp16, group_size=group_size
    )
    Wq = dequantize_from_triton_layer(temp_linear_triton)
    del temp_linear_fp16, temp_linear_triton
    gc.collect()
    torch.cuda.empty_cache()
    return Wq





def main():
    ap = argparse.ArgumentParser(
        description="Step 1 (Integrated Version) - Calculate quantization error for various models using Triton."
    )
    ap.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g., 'meta-llama/Llama-3.1-8B')",
    )
    ap.add_argument(
        "--out_quant_err",
        required=True,
        help="Output path for quantization error dictionary (.pt)",
    )
    ap.add_argument(
        "--out_original_weights",
        required=True,
        help="Output path for original weights dictionary (.pt)",
    )
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set this flag for models like Qwen that require custom code",
    )
    ap.add_argument("--device", default="cuda", help="Device to use for computation")
    ap.add_argument(
        "--group_size", type=int, default=128, help="Group size for Triton quantization"
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = ap.parse_args()

    if not HAS_TRITON or not torch.cuda.is_available():
        print("This script requires Triton and a CUDA-enabled GPU.")
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    print(f"Loading original FP16 model: {args.model_name}...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"\nUsing ORIGINAL weights for {args.model_name} ...")
    print(
        "Note: This script automatically detects layer dimensions, supporting heterogeneous attention."
    )

    
    original_state_dict = {
        k: v.cpu().clone() for k, v in model_fp16.state_dict().items()
    }
    quant_err_dict = {}

    print(
        f"\nCalculating quantization errors from ORIGINAL weights using Asymmetric Triton..."
    )
    print(f"Formula: Eq = W_original - Wq_original")

    processed_layers = 0
    layer_dimensions = {}

    for name, W_original_cpu in tqdm(
        original_state_dict.items(), desc="Processing layers with Triton"
    ):
        if not (
            name.endswith(".weight") and W_original_cpu.ndim == 2 and "layers" in name
        ):
            continue

        
        if not any(
            kw in name
            for kw in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ):
            continue

        
        layer_idx_match = re.search(r"layers\.(\d+)\.", name)
        if layer_idx_match:
            layer_num = layer_idx_match.group(1)
            module_type = name.split(".")[-2]
            layer_dimensions.setdefault(layer_num, {})[
                module_type
            ] = W_original_cpu.shape

        W_original = W_original_cpu.to(device)
        Wq = get_triton_dequantized_weight(W_original, device, args.group_size)

        Eq = W_original.cpu() - Wq.cpu()
        quant_err_dict[name] = Eq.to(torch.float32)  

        processed_layers += 1

        
        original_norm = W_original.norm().item()
        quantized_norm = Wq.norm().item()
        error_norm = Eq.norm().item()
        error_ratio = error_norm / max(original_norm, 1e-12)

        print(f"  Layer: {name}")
        print(
            f"    Shape: {W_original_cpu.shape}, Original norm: {original_norm:.4f}, Quantized norm: {quantized_norm:.4f}"
        )
        print(f"    Error norm: {error_norm:.4f}, Error ratio: {error_ratio:.4f}")

    
    print(f"\n🔍 Discovered Layer Dimensions for {args.model_name} (First 3 layers):")
    for layer_num in sorted(layer_dimensions.keys())[:3]:
        print(f"  Layer {layer_num}:")
        for module_type, shape in sorted(layer_dimensions[layer_num].items()):
            print(f"    {module_type}: {shape}")

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    
    os.makedirs(os.path.dirname(args.out_quant_err) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_original_weights) or ".", exist_ok=True)
    torch.save(quant_err_dict, args.out_quant_err)
    torch.save(original_state_dict, args.out_original_weights)

    print(f"\nCOMPLETED: Files saved!")
    print(f"  Quantization errors: {args.out_quant_err}")
    print(f"  Original weights: {args.out_original_weights}")
    print(f"  Processed layers: {processed_layers}")
    print(
        f"  Configuration: Model='{args.model_name}', group_size={args.group_size}, seed={args.seed}"
    )

    
    if quant_err_dict:
        total_error_elements = sum(tensor.numel() for tensor in quant_err_dict.values())
        avg_error_magnitude = (
            torch.cat([tensor.flatten() for tensor in quant_err_dict.values()])
            .abs()
            .mean()
            .item()
        )
        print(f"\nError Statistics:")
        print(f"  • Total error elements: {total_error_elements:,}")
        print(f"  • Average error magnitude: {avg_error_magnitude:.6f}")

    print(f"  • Ready for direct error correction in step3!")


if __name__ == "__main__":
    main()

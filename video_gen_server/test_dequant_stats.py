import torch
import safetensors.torch as st
import os
from torchao.prototype.mx_formats.mx_tensor import to_dtype

model_path = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
sub_path = os.path.join(model_path, "transformer_ref")
shard1 = os.path.join(sub_path, "diffusion_pytorch_model-00001-of-00013.safetensors")

with st.safe_open(shard1, framework="pt", device="cpu") as f:
    w_fp8 = f.get_tensor("transformer_blocks.0.attn.to_q.weight")
    mx_scale = f.get_tensor("transformer_blocks.0.attn.to_q.weight.mx_scale")

w_bf16 = to_dtype(w_fp8, mx_scale, torch.float8_e4m3fn, 32, torch.bfloat16)
print("w_bf16 from to_dtype:")
print(f"min={w_bf16.float().min().item():.6f}, max={w_bf16.float().max().item():.6f}, mean={w_bf16.float().mean().item():.6f}, abs_mean={w_bf16.float().abs().mean().item():.6f}")

# Check adaln_proj
with st.safe_open(shard1, framework="pt", device="cpu") as f:
    w_fp8 = f.get_tensor("transformer_blocks.0.adaln_proj.linear.weight")
    fp8_scale = f.get_tensor("transformer_blocks.0.adaln_proj.linear.weight.fp8_scale")

block_size = w_fp8.shape[1] // fp8_scale.shape[1]
w_adaln = (w_fp8.float() / fp8_scale.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)
print("\nw_adaln from dequantizer:")
print(f"min={w_adaln.float().min().item():.6f}, max={w_adaln.float().max().item():.6f}, mean={w_adaln.float().mean().item():.6f}, abs_mean={w_adaln.float().abs().mean().item():.6f}")

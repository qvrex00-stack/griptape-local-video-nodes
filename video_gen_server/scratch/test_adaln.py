import torch
import safetensors.torch as st
import json, os

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8/transformer_ref'
if not os.path.exists(path):
    path = 'D:/MiniMax-H3-FP8/transformer_ref'

idx_file = os.path.join(path, 'diffusion_pytorch_model.safetensors.index.json')
with open(idx_file) as f:
    wm = json.load(f)['weight_map']

shard_name = wm['transformer_blocks.0.adaln_proj.linear.weight']
data = st.load_file(os.path.join(path, shard_name))

w_fp8 = data['transformer_blocks.0.adaln_proj.linear.weight']
scale = data['transformer_blocks.0.adaln_proj.linear.weight.fp8_scale']
bias = data['transformer_blocks.0.adaln_proj.linear.bias']

print('w_fp8:', w_fp8.shape, w_fp8.dtype)
print('scale:', scale.shape, scale.dtype, scale.min().item(), scale.max().item())
print('bias:', bias.shape, bias.dtype, bias.min().item(), bias.max().item())

# Current handler dequant formula: (w_fp8.float() / scale.repeat_interleave(...))
block_size = w_fp8.shape[1] // scale.shape[1] # 2688 // 21 = 128
w_dequant_div = (w_fp8.float() / scale.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)

# Alternative formula 1: w_fp8.float() * scale
w_dequant_mul = (w_fp8.float() * scale.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)

# Alternative formula 2: what if scale is reciprocal or something else?
print('w_dequant_div stats: min=', w_dequant_div.min().item(), 'max=', w_dequant_div.max().item(), 'std=', w_dequant_div.std().item())
print('w_dequant_mul stats: min=', w_dequant_mul.min().item(), 'max=', w_dequant_mul.max().item(), 'std=', w_dequant_mul.std().item())

# Test forward with dummy temb (timestep embedding usually std ~ 1)
temb = torch.randn(1, 2688, dtype=torch.bfloat16)
temb_act = torch.nn.functional.silu(temb)

out_div = torch.nn.functional.linear(temb_act, w_dequant_div, bias.to(torch.bfloat16))
print('out_div stats: min=', out_div.min().item(), 'max=', out_div.max().item(), 'std=', out_div.std().item())
chunks_div = out_div.view(-1, 6 * 5376).chunk(6, dim=-1)
print('gate_msa_div: mean=', chunks_div[2].mean().item(), 'std=', chunks_div[2].std().item(), 'min=', chunks_div[2].min().item(), 'max=', chunks_div[2].max().item())

out_bias_only = bias.to(torch.bfloat16)
chunks_b = out_bias_only.view(-1, 6 * 5376).chunk(6, dim=-1)
print('bias only gate_msa: mean=', chunks_b[2].mean().item(), 'std=', chunks_b[2].std().item(), 'min=', chunks_b[2].min().item(), 'max=', chunks_b[2].max().item())

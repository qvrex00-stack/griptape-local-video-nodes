import torch
import safetensors.torch as st
import json, os
from torchao.prototype.mx_formats.mx_tensor import to_dtype
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8/transformer_ref'
shard = os.path.join(path, 'diffusion_pytorch_model-00001-of-00013.safetensors')
data = st.load_file(shard)

# Load ff weights
w_proj_fp8 = data['transformer_blocks.0.ff.net.0.proj.weight']
s_proj = data['transformer_blocks.0.ff.net.0.proj.weight.mx_scale']
w_proj = to_dtype(w_proj_fp8, s_proj, torch.float8_e4m3fn, 32, torch.bfloat16).cuda()

w_out_fp8 = data['transformer_blocks.0.ff.net.2.weight']
s_out = data['transformer_blocks.0.ff.net.2.weight.mx_scale']
w_out = to_dtype(w_out_fp8, s_out, torch.float8_e4m3fn, 32, torch.bfloat16).cuda()

print('w_proj stats: min=', w_proj.min().item(), 'max=', w_proj.max().item(), 'std=', w_proj.std().item())
print('w_out stats: min=', w_out.min().item(), 'max=', w_out.max().item(), 'std=', w_out.std().item())

# Test SwiGLU with normalized input (std ~ 1.0)
x = torch.randn(1, 1000, 5376, device='cuda', dtype=torch.bfloat16)
proj_out = torch.nn.functional.linear(x, w_proj)
print('proj_out stats: min=', proj_out.min().item(), 'max=', proj_out.max().item(), 'std=', proj_out.std().item())

x1, x2 = proj_out.chunk(2, dim=-1)
swiglu_act = torch.nn.functional.silu(x1) * x2
print('swiglu_act stats: min=', swiglu_act.min().item(), 'max=', swiglu_act.max().item(), 'std=', swiglu_act.std().item())

ff_out = torch.nn.functional.linear(swiglu_act, w_out)
print('ff_out stats: min=', ff_out.min().item(), 'max=', ff_out.max().item(), 'std=', ff_out.std().item())

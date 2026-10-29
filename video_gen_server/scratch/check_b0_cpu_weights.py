import torch, json, os
import safetensors.torch as st
from diffusers import ModularPipeline, ComponentsManager
from torchao.prototype.mx_formats.mx_tensor import to_dtype

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8/transformer_ref'
shard = os.path.join(path, 'diffusion_pytorch_model-00001-of-00013.safetensors')
data = st.load_file(shard)

# Inspect all keys in shard 1
print('Total keys in shard 1:', len(data))

# Check block 0
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
b0 = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)

# Simulate _dequantize on block 0
for k in data.keys():
    if not k.startswith('transformer_blocks.0.'):
        continue
    if k.endswith('.mx_scale') or k.endswith('.fp8_scale'):
        continue
    parts = k.split('.')
    # parts: ['transformer_blocks', '0', 'attn', 'to_q', 'weight']
    curr = b0
    for p in parts[2:-1]:
        if p.isdigit():
            curr = curr[int(p)]
        else:
            curr = getattr(curr, p, None)
    attr = parts[-1]
    
    mx_scale_k = f"{k}.mx_scale"
    fp8_scale_k = f"{k}.fp8_scale"
    if mx_scale_k in data:
        w = to_dtype(data[k], data[mx_scale_k], torch.float8_e4m3fn, 32, torch.bfloat16)
    elif fp8_scale_k in data:
        s = data[fp8_scale_k]
        w_fp8 = data[k]
        block_size = w_fp8.shape[1] // s.shape[1]
        w = (w_fp8.float() / s.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)
    else:
        w = data[k].to(torch.bfloat16)
        
    setattr(curr, f"_cpu_{attr}", w.to('cpu'))
    print(f"Bound _cpu_{attr} to {'.'.join(parts[2:-1])}: shape={w.shape}, std={w.std().item():.4f}")

# Now check all linears in b0
print("\nChecking all linears in b0:")
for name, mod in b0.named_modules():
    if isinstance(mod, torch.nn.Linear):
        has_w = hasattr(mod, '_cpu_weight')
        has_b = hasattr(mod, '_cpu_bias') or mod.bias is None
        print(f"Linear {name}: has_cpu_weight={has_w}, has_cpu_bias={has_b}")

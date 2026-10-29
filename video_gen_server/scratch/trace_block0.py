import torch
import safetensors.torch as st
import json, os
from torchao.prototype.mx_formats.mx_tensor import to_dtype
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8/transformer_ref'
if not os.path.exists(path):
    path = 'D:/MiniMax-H3-FP8/transformer_ref'

idx_file = os.path.join(path, 'diffusion_pytorch_model.safetensors.index.json')
with open(idx_file) as f:
    wm = json.load(f)['weight_map']

# Create block 0 with transformer_ref config
block0 = MiniMaxH3TransformerBlock(
    hidden_size=5376,
    num_attention_heads=56,
    attention_head_dim=128,
    ffn_dim=14336,
    time_embed_dim=2688,
    norm_eps=1e-5,
    qk_norm_eps=1e-5
)

# Find all keys for block 0
b0_keys = [k for k in wm.keys() if k.startswith('transformer_blocks.0.')]
print(f'Total keys for block 0: {len(b0_keys)}')

# Group by shard
shards_needed = set(wm[k] for k in b0_keys)
print('Shards needed:', shards_needed)

state_dict = {}
for s_name in shards_needed:
    data = st.load_file(os.path.join(path, s_name))
    for k in b0_keys:
        if wm[k] != s_name:
            continue
        if k.endswith('.mx_scale') or k.endswith('.fp8_scale'):
            continue
        
        mx_scale_k = f"{k}.mx_scale"
        fp8_scale_k = f"{k}.fp8_scale"
        
        if mx_scale_k in data:
            w_fp8 = data[k]
            scale = data[mx_scale_k]
            w_bf16 = to_dtype(w_fp8, scale, torch.float8_e4m3fn, 32, torch.bfloat16)
        elif fp8_scale_k in data:
            w_fp8 = data[k]
            scale = data[fp8_scale_k]
            block_size = w_fp8.shape[1] // scale.shape[1]
            w_bf16 = (w_fp8.float() / scale.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)
        elif k in data:
            w_bf16 = data[k].to(torch.bfloat16)
        
        mod_key = k.replace('transformer_blocks.0.', '')
        state_dict[mod_key] = w_bf16

# Check weights
missing, unexpected = block0.load_state_dict(state_dict, strict=False)
print('Loaded state dict. Missing:', missing, 'Unexpected:', unexpected)
for name, p in block0.named_parameters():
    print(f"Param {name}: shape={p.shape}, std={p.std().item():.4f}, min={p.min().item():.4f}, max={p.max().item():.4f}")

# Now let's trace forward step-by-step!
block0 = block0.to(device='cuda', dtype=torch.bfloat16)

# Realistic inputs from user log:
# latents shape [24676, 96] -> patchify & proj to [24676, 5376]
# in hidden_states: mean=-0.00208, std=2.73438
seq_len = 1000 # test with 1000 tokens for speed
hidden_states = torch.randn(1, seq_len, 5376, device='cuda', dtype=torch.bfloat16) * 2.7
temb = torch.randn(1, 2688, device='cuda', dtype=torch.float32)
adaln_indices = torch.zeros(seq_len, dtype=torch.long, device='cuda')

# RoPE dummy: cos, sin shape (seq_len, rotary_dim) where rotary_dim = 2 * 3 * rope_freq_dim = 96
rotary_dim = 96
cos = torch.randn(seq_len, rotary_dim, device='cuda', dtype=torch.bfloat16)
sin = torch.randn(seq_len, rotary_dim, device='cuda', dtype=torch.bfloat16)
rotary_emb = (cos, sin)

print("--- Tracing block 0 forward ---")
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block0.adaln_proj(temb)
print(f"scale_msa: std={scale_msa.std().item():.4f}, min={scale_msa.min().item():.4f}, max={scale_msa.max().item():.4f}")
print(f"gate_msa: std={gate_msa.std().item():.4f}, min={gate_msa.min().item():.4f}, max={gate_msa.max().item():.4f}")

norm1_out = block0.norm1(hidden_states)
print(f"norm1_out: std={norm1_out.std().item():.4f}, min={norm1_out.min().item():.4f}, max={norm1_out.max().item():.4f}")

norm1_mod = norm1_out * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
print(f"norm1_mod: std={norm1_mod.std().item():.4f}, min={norm1_mod.min().item():.4f}, max={norm1_mod.max().item():.4f}")

attn_out = block0.attn(norm1_mod, rotary_emb, None)
print(f"attn_out: std={attn_out.std().item():.4f}, min={attn_out.min().item():.4f}, max={attn_out.max().item():.4f}")

hs1 = hidden_states + gate_msa.index_select(0, adaln_indices) * attn_out
print(f"after attn + res: std={hs1.std().item():.4f}, min={hs1.min().item():.4f}, max={hs1.max().item():.4f}")

norm2_out = block0.norm2(hs1)
norm2_mod = norm2_out * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
ff_out = block0.ff(norm2_mod)
print(f"ff_out: std={ff_out.std().item():.4f}, min={ff_out.min().item():.4f}, max={ff_out.max().item():.4f}")

hs2 = hs1 + gate_mlp.index_select(0, adaln_indices) * ff_out
print(f"block0 final out: std={hs2.std().item():.4f}, min={hs2.min().item():.4f}, max={hs2.max().item():.4f}")

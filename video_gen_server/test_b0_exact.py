import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
import safetensors.torch as st

d_ref = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors")
d_fl = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors")

b0_ref = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
b0_fl = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)

dict_ref = {k.replace("transformer_blocks.0.", ""): v for k, v in d_ref.items() if k.startswith("transformer_blocks.0.")}
dict_fl = {k.replace("transformer_blocks.0.", ""): v.to(torch.bfloat16) for k, v in d_fl.items() if k.startswith("transformer_blocks.0.")}

b0_ref.load_state_dict(dict_ref)
b0_fl.load_state_dict(dict_fl)

b0_ref = b0_ref.to(device="cuda", dtype=torch.bfloat16)
b0_fl = b0_fl.to(device="cuda", dtype=torch.bfloat16)

# Test same input:
torch.manual_seed(42)
x = torch.randn(1, 1024, 5376, device="cuda", dtype=torch.bfloat16)

# Test FF
ff_ref = b0_ref.ff(x)
ff_fl = b0_fl.ff(x)
print("FF REF:", ff_ref.mean().item(), ff_ref.std().item(), ff_ref.min().item(), ff_ref.max().item())
print("FF FL :", ff_fl.mean().item(), ff_fl.std().item(), ff_fl.min().item(), ff_fl.max().item())

# Test Attn
rotary_emb = (torch.ones(1024, 48, device="cuda", dtype=torch.bfloat16), torch.zeros(1024, 48, device="cuda", dtype=torch.bfloat16))
attn_ref = b0_ref.attn(x, rotary_emb=rotary_emb)
attn_fl = b0_fl.attn(x, rotary_emb=rotary_emb)
print("Attn REF:", attn_ref.mean().item(), attn_ref.std().item(), attn_ref.min().item(), attn_ref.max().item())
print("Attn FL :", attn_fl.mean().item(), attn_fl.std().item(), attn_fl.min().item(), attn_fl.max().item())

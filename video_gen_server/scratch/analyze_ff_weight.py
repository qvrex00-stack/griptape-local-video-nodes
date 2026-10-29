import torch
import safetensors.torch as st
from torchao.prototype.mx_formats.mx_tensor import from_blocked, get_fp_scale

p_d = r"D:\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors"
p_c = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors"
p_fl = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors"

d_raw = st.load_file(p_d)
d_c = st.load_file(p_c)
d_fl = st.load_file(p_fl)

k = "transformer_blocks.0.ff.net.0.proj.weight"
w_raw = d_raw[k] # float8_e4m3fn
scale_raw = d_raw[f"{k}.mx_scale"]

rows, cols = w_raw.shape
scale_unblocked = from_blocked(scale_raw, rows, cols // 32)
s_fp = get_fp_scale(scale_unblocked).repeat_interleave(32, dim=1)
w_dequant = (w_raw.to(torch.bfloat16).float() * s_fp).to(torch.bfloat16)

w_in_c = d_c[k]
w_fl_val = d_fl[k]

print("w_dequant vs w_in_c diff:", (w_dequant.float() - w_in_c.float()).abs().max().item())
print(f"w_dequant: std={w_dequant.float().std():.4f}, mean={w_dequant.float().mean():.6f}")
print(f"w_fl_val:  std={w_fl_val.float().std():.4f}, mean={w_fl_val.float().mean():.6f}")
print("w_dequant vs w_fl_val cosine:", torch.nn.functional.cosine_similarity(w_dequant.float().flatten(), w_fl_val.float().flatten(), dim=0).item())

# Wait, why was cosine ~0 with w_fl_val? Is SwiGLU proj weight in fl2va vs ref2va structured differently or completely different?
# Let's check other blocks! Are all ff.net.0.proj.weight having cosine ~0 between FL and REF?
print("\n--- Checking multiple blocks between FL and REF ---")
# Let's inspect block 0 attention weights vs ff weights
for b_k in ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight"]:
    key = f"transformer_blocks.0.{b_k}"
    c_ref = d_c[key].float()
    c_fl = d_fl[key].float()
    cos = torch.nn.functional.cosine_similarity(c_ref.flatten(), c_fl.flatten(), dim=0).item()
    print(f"{key:45s}: shape={list(c_ref.shape)}, cos={cos:.4f}")

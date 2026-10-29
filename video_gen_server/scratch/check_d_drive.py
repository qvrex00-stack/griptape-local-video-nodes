import safetensors.torch as st
import os

p_c = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors"
p_d = r"D:\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors"

print("C size:", os.path.getsize(p_c))
print("D size:", os.path.getsize(p_d))

with st.safe_open(p_d, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print("D keys count:", len(keys))
    has_scale = any("scale" in k for k in keys)
    print("D has scale keys?", has_scale)
    if has_scale:
        print("Sample scale keys in D:", [k for k in keys if "scale" in k][:5])
    k_test = "transformer_blocks.0.ff.net.0.proj.weight"
    t_d = f.get_tensor(k_test)
    print(f"D tensor {k_test}: dtype={t_d.dtype}, shape={t_d.shape}")

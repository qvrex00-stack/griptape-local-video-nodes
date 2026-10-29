import safetensors.torch as st
import os

model_path = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
sub_path = os.path.join(model_path, "transformer_ref")
shard1 = os.path.join(sub_path, "diffusion_pytorch_model-00001-of-00013.safetensors")

with st.safe_open(shard1, framework="pt", device="cpu") as f:
    for k in f.keys():
        if "adaln_proj" in k or "to_q" in k:
            t = f.get_tensor(k).float()
            print(f"{k}: shape={t.shape}, min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}, abs_mean={t.abs().mean().item():.6f}")

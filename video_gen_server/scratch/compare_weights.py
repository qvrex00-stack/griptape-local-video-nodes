import json, os, sys
import torch
import safetensors.torch as st
from safetensors import safe_open
base = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
maps = {}
for sub in ("transformer", "transformer_ref"):
    p = os.path.join(base, sub)
    maps[sub] = json.load(open(os.path.join(p, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]

def get(sub, k):
    p = os.path.join(base, sub, maps[sub][k])
    with safe_open(p, framework="pt", device="cpu") as f:
        return f.get_tensor(k)

keys = [
 "transformer_blocks.0.attn.to_q.weight",
 "transformer_blocks.0.attn.to_out.weight",
 "transformer_blocks.0.ff.net.proj.weight",
 "transformer_blocks.0.adaln_proj.linear.weight",
 "transformer_blocks.0.adaln_proj.linear.bias",
 "transformer_blocks.25.attn.to_q.weight",
 "transformer_blocks.25.ff.net.weight",
 "transformer_blocks.25.adaln_proj.linear.weight",
 "transformer_blocks.49.attn.to_v.weight",
 "transformer_blocks.49.adaln_proj.linear.weight",
 "context_embedder.weight", "proj_in.weight", "time_embedder.linear_1.weight",
 "norm_out.linear.weight", "proj_out.weight",
 "token_refiner.refiner_blocks.0.attn.to_q.weight",
]
for k in keys:
    if k not in maps["transformer"] or k not in maps["transformer_ref"]:
        print("MISSING", k, k in maps["transformer"], k in maps["transformer_ref"]); continue
    a = get("transformer", k).float()
    b = get("transformer_ref", k).float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    ratio = (b.abs().mean() / a.abs().mean().clamp_min(1e-12)).item()
    print(f"{k}: base_absmean={a.abs().mean():.5f} ref_absmean={b.abs().mean():.5f} ratio={ratio:.4f} cos={cos:.4f} nan_ref={torch.isnan(b).any().item()}")

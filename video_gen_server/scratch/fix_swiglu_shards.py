"""
Permanently fix the SwiGLU Gate/Up chunk reversal in transformer_ref safetensors files.
Swaps chunk(0) and chunk(1) along dim=0 for all ff.net.0.proj.weight keys.
"""
import os
import json
import time
import torch
import safetensors.torch as st

model_dir = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref"
idx_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")

with open(idx_path, "r", encoding="utf-8") as f:
    idx_data = json.load(f)

weight_map = idx_data["weight_map"]

# Find which shards contain ff.net.0.proj.weight
target_keys = [k for k in weight_map.keys() if k.endswith("ff.net.0.proj.weight")]
print(f"Found {len(target_keys)} target SwiGLU keys to fix across transformer_ref.")

shard_to_keys = {}
for k in target_keys:
    s_name = weight_map[k]
    shard_to_keys.setdefault(s_name, []).append(k)

total_shards = len(shard_to_keys)
print(f"These keys are distributed across {total_shards} shards.")

t0 = time.time()
fixed_count = 0

for s_idx, (shard_name, keys) in enumerate(sorted(shard_to_keys.items())):
    shard_path = os.path.join(model_dir, shard_name)
    temp_path = os.path.join(model_dir, f"temp_{shard_name}")
    
    print(f"[{s_idx+1}/{total_shards}] Processing {shard_name} ({len(keys)} SwiGLU keys)...")
    data = st.load_file(shard_path)
    
    for k in keys:
        w = data[k] # [28672, 5376]
        # Verify it has even rows
        assert w.shape[0] % 2 == 0, f"Odd rows in {k}: {w.shape}"
        ra, rb = w.chunk(2, dim=0)
        # Swap chunks so that Up is first and Gate is second (matching diffusers SwiGLU spec)
        w_fixed = torch.cat([rb, ra], dim=0).contiguous()
        data[k] = w_fixed
        fixed_count += 1
    
    # Save back to temp file
    st.save_file(data, temp_path)
    del data
    
    # Replace old shard
    os.remove(shard_path)
    os.rename(temp_path, shard_path)
    print(f"[{s_idx+1}/{total_shards}] {shard_name} updated successfully.")

print(f"\n=== SUCCESS! Corrected {fixed_count} SwiGLU layers in {round(time.time() - t0, 1)}s ===")

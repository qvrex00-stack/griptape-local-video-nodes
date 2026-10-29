import os
import gc
import json
import time
import torch
import safetensors.torch as st
from torchao.prototype.mx_formats.mx_tensor import from_blocked, get_fp_scale

def convert_transformer_ref(model_dir: str):
    sub_path = os.path.join(model_dir, "transformer_ref")
    idx_path = os.path.join(sub_path, "diffusion_pytorch_model.safetensors.index.json")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Index not found: {idx_path}")

    with open(idx_path, "r", encoding="utf-8") as f:
        idx_data = json.load(f)

    weight_map = idx_data.get("weight_map", {})
    shards = sorted(list(set(weight_map.values())))
    total_shards = len(shards)

    print(f"=== Converting {total_shards} shards of transformer_ref to native unswizzled bfloat16 ===")
    t0_total = time.time()
    converted_keys_set = set()

    # Modules that must stay float32 according to diffusers MiniMax-H3 spec:
    fp32_prefixes = (
        "time_embedder",
        "norm_out",
        "proj_in",
        "proj_out",
        "audio_proj_in",
        "audio_proj_out",
    )

    for shard_idx, shard_name in enumerate(shards):
        shard_path = os.path.join(sub_path, shard_name)
        temp_shard_path = os.path.join(sub_path, f"temp_{shard_name}")

        t0_shard = time.time()
        print(f"[{shard_idx + 1}/{total_shards}] Loading {shard_name}...")
        shard_data = st.load_file(shard_path)

        out_data = {}
        # Identify base keys
        base_keys = [k for k in shard_data.keys() if not (k.endswith(".mx_scale") or k.endswith(".fp8_scale"))]

        for k in base_keys:
            val = shard_data[k]
            mx_scale_k = f"{k}.mx_scale"
            fp8_scale_k = f"{k}.fp8_scale"

            # Check if this module should be float32
            is_fp32 = any(fn in k for fn in fp32_prefixes)

            if mx_scale_k in shard_data:
                # MXFP8 with cuBLASLt 128x4 swizzled scale
                scale_blocked = shard_data[mx_scale_k]
                rows, cols = val.shape
                scale_unblocked = from_blocked(scale_blocked, rows, cols // 32)
                s_fp = get_fp_scale(scale_unblocked).repeat_interleave(32, dim=1)
                w_bf16 = (val.to(torch.bfloat16).float() * s_fp).to(torch.bfloat16)
                out_data[k] = w_bf16
                del scale_blocked, scale_unblocked, s_fp, w_bf16
            elif fp8_scale_k in shard_data:
                # Block-scaled float32 adaln_proj
                scale = shard_data[fp8_scale_k]
                if scale.ndim == 2 and val.ndim == 2:
                    block_size = val.shape[1] // scale.shape[1]
                    scale_expanded = scale.repeat_interleave(block_size, dim=1)
                    w_bf16 = (val.float() / scale_expanded).to(torch.bfloat16)
                elif scale.ndim == 1:
                    w_bf16 = (val.float() / scale.unsqueeze(0)).to(torch.bfloat16)
                else:
                    w_bf16 = (val.float() / scale).to(torch.bfloat16)
                out_data[k] = w_bf16
                del scale, w_bf16
            else:
                # Unscaled weights (biases, norms, embeddings)
                if is_fp32:
                    out_data[k] = val.to(torch.float32)
                else:
                    out_data[k] = val.to(torch.bfloat16)

            converted_keys_set.add(k)

        del shard_data
        gc.collect()

        print(f"[{shard_idx + 1}/{total_shards}] Saving {len(out_data)} converted layers to {temp_shard_path}...")
        st.save_file(out_data, temp_shard_path)
        del out_data
        gc.collect()

        # Verify saved file
        test_read = st.load_file(temp_shard_path)
        assert len(test_read) == len(base_keys), f"Key count mismatch in {shard_name}"
        del test_read
        gc.collect()

        # Atomically replace old shard
        os.remove(shard_path)
        os.rename(temp_shard_path, shard_path)
        new_size_gb = os.path.getsize(shard_path) / 1024**3
        dt = time.time() - t0_shard
        print(f"[{shard_idx + 1}/{total_shards}] {shard_name} successfully converted and replaced ({new_size_gb:.2f} GB) in {dt:.1f}s")

    # Update index.json: remove all .mx_scale and .fp8_scale keys
    print("Updating diffusion_pytorch_model.safetensors.index.json...")
    new_weight_map = {k: v for k, v in weight_map.items() if k in converted_keys_set}
    idx_data["weight_map"] = new_weight_map
    if "metadata" in idx_data and "total_size" in idx_data["metadata"]:
        total_bytes = sum(os.path.getsize(os.path.join(sub_path, s)) for s in shards)
        idx_data["metadata"]["total_size"] = total_bytes

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx_data, f, indent=2)

    total_time = time.time() - t0_total
    print(f"=== ALL {total_shards} SHARDS CONVERTED SUCCESSFULLY in {total_time:.1f}s! ===")

if __name__ == "__main__":
    convert_transformer_ref(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8")

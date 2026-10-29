import os
import torch
import safetensors.torch as st
from diffusers import MiniMaxH3Transformer3DModel
import json

def test_lora_fusion_logic():
    print("Testing LoRA fusion logic...")
    with open(r'C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\config.json') as f:
        cfg = json.load(f)
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel(**cfg)

    # Fake weights for 1 block
    b0_q = model.transformer_blocks[0].attn.to_q
    b0_q._cpu_weight = torch.randn(3072, 3072, dtype=torch.bfloat16)

    # Create dummy LoRA weights (Diffusers and ComfyUI format)
    rank = 16
    lora_a = torch.randn(rank, 3072, dtype=torch.float32)
    lora_b = torch.randn(3072, rank, dtype=torch.float32)

    diffusers_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": lora_a,
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": lora_b,
        "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(16.0),
    }
    comfy_dict = {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": lora_a,
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": lora_b,
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": torch.tensor(32.0),
    }

    test_lora_path = r"C:\Foundry\Griptape\video_gen_server\scratch\test_dummy_lora.safetensors"
    os.makedirs(os.path.dirname(test_lora_path), exist_ok=True)
    st.save_file(diffusers_dict, test_lora_path)

    # Fusion function implementation test
    def fuse(tf_model, lora_path, lora_scale=1.0):
        named_mods = dict(tf_model.named_modules())
        mod_underscore_map = {k.replace(".", "_"): k for k in named_mods.keys()}
        
        lora_data = st.load_file(lora_path)
        pairs = {}
        for k in lora_data.keys():
            down_suffix = None
            up_suffix = None
            for s in [".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".down.weight"]:
                if k.endswith(s):
                    down_suffix = s
                    break
            for s in [".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".up.weight"]:
                if k.endswith(s):
                    up_suffix = s
                    break

            if down_suffix:
                prefix = k[:-len(down_suffix)]
                pairs.setdefault(prefix, {})["down"] = lora_data[k]
            elif up_suffix:
                prefix = k[:-len(up_suffix)]
                pairs.setdefault(prefix, {})["up"] = lora_data[k]
            elif k.endswith(".alpha"):
                prefix = k[:-len(".alpha")]
                pairs.setdefault(prefix, {})["alpha"] = float(lora_data[k].item())

        print(f"Detected {len(pairs)} LoRA module pairs")
        fused_count = 0
        for prefix, pdata in pairs.items():
            if "down" not in pdata or "up" not in pdata:
                continue
            down = pdata["down"].float()
            up = pdata["up"].float()
            alpha = pdata.get("alpha", None)

            # Strip known prefixes
            clean_name = prefix
            for pfx in ["diffusion_model.", "transformer.", "lora_unet_"]:
                if clean_name.startswith(pfx):
                    clean_name = clean_name[len(pfx):]

            # Try exact match or underscore match
            target_mod_name = None
            if clean_name in named_mods:
                target_mod_name = clean_name
            elif clean_name in mod_underscore_map:
                target_mod_name = mod_underscore_map[clean_name]

            if not target_mod_name:
                print(f"Could not map {prefix} -> {clean_name}")
                continue

            mod = named_mods[target_mod_name]
            if not hasattr(mod, "_cpu_weight"):
                continue

            r = down.shape[0] if down.shape[0] < down.shape[1] else down.shape[1]
            eff_alpha = alpha if alpha is not None else float(r)
            scale = lora_scale * (eff_alpha / float(r))

            if up.shape[1] == down.shape[0]:
                delta = up @ down
            elif up.shape[0] == down.shape[1]:
                delta = (down @ up).t()
            else:
                print(f"Shape mismatch: up {up.shape}, down {down.shape}")
                continue

            delta = delta * scale
            if not hasattr(mod, "_orig_cpu_weight"):
                mod._orig_cpu_weight = mod._cpu_weight.clone()

            mod._cpu_weight = (mod._orig_cpu_weight.float() + delta).to(dtype=torch.bfloat16)
            fused_count += 1
            print(f"Successfully fused {target_mod_name} (rank={r}, scale={scale:.2f})")

        return fused_count

    count1 = fuse(model, test_lora_path, 1.0)
    print("Diffusers LoRA fused:", count1)
    assert count1 == 1

    # Now test ComfyUI format
    st.save_file(comfy_dict, test_lora_path)
    count2 = fuse(model, test_lora_path, 1.0)
    print("ComfyUI LoRA fused:", count2)
    assert count2 == 1
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test_lora_fusion_logic()

import torch
import safetensors.torch as st
import json, os

base = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
idx_ref = json.load(open(os.path.join(base, "transformer_ref", "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
idx_fl = json.load(open(os.path.join(base, "transformer", "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]

# Check all 50 blocks
flipped_count = 0
total_checked = 0

for b in range(50):
    k = f"transformer_blocks.{b}.ff.net.0.proj.weight"
    if k not in idx_ref or k not in idx_fl:
        continue
    
    file_ref = os.path.join(base, "transformer_ref", idx_ref[k])
    file_fl = os.path.join(base, "transformer", idx_fl[k])
    
    with st.safe_open(file_ref, framework="pt", device="cpu") as fr, st.safe_open(file_fl, framework="pt", device="cpu") as ff:
        w_ref = fr.get_tensor(k).float()
        w_fl = ff.get_tensor(k).float()
        
        ra, rb = w_ref.chunk(2, dim=0)
        fa, fb = w_fl.chunk(2, dim=0)
        
        c_direct = torch.nn.functional.cosine_similarity(ra.flatten(), fa.flatten(), dim=0).item()
        c_flipped = torch.nn.functional.cosine_similarity(ra.flatten(), fb.flatten(), dim=0).item()
        
        total_checked += 1
        if c_flipped > 0.9 and c_direct < 0.1:
            flipped_count += 1
        else:
            print(f"Block {b}: direct={c_direct:.4f}, flipped={c_flipped:.4f}")

print(f"Total checked: {total_checked}, FLIPPED (ref_a == fl_b, ref_b == fl_a): {flipped_count}")

# Also check token_refiner blocks!
for r in [0, 1]:
    k = f"token_refiner.refiner_blocks.{r}.ff.net.0.proj.weight"
    if k in idx_ref and k in idx_fl:
        file_ref = os.path.join(base, "transformer_ref", idx_ref[k])
        file_fl = os.path.join(base, "transformer", idx_fl[k])
        with st.safe_open(file_ref, framework="pt", device="cpu") as fr, st.safe_open(file_fl, framework="pt", device="cpu") as ff:
            w_ref = fr.get_tensor(k).float()
            w_fl = ff.get_tensor(k).float()
            ra, rb = w_ref.chunk(2, dim=0)
            fa, fb = w_fl.chunk(2, dim=0)
            c_direct = torch.nn.functional.cosine_similarity(ra.flatten(), fa.flatten(), dim=0).item()
            c_flipped = torch.nn.functional.cosine_similarity(ra.flatten(), fb.flatten(), dim=0).item()
            print(f"token_refiner {r}: direct={c_direct:.4f}, flipped={c_flipped:.4f}")

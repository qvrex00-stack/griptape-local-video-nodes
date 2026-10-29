import torch
import safetensors.torch as st

p_c = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors"
p_fl = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors"

d_c = st.load_file(p_c)
d_fl = st.load_file(p_fl)

k = "transformer_blocks.0.ff.net.0.proj.weight"
w_ref = d_c[k].float() # [28672, 5376]
w_fl = d_fl[k].float()  # [28672, 5376]

ref_a, ref_b = w_ref.chunk(2, dim=0) # [14336, 5376]
fl_a, fl_b = w_fl.chunk(2, dim=0)   # [14336, 5376]

def sim(x, y):
    return torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()

print("ref_a vs fl_a:", sim(ref_a, fl_a))
print("ref_a vs fl_b:", sim(ref_a, fl_b))
print("ref_b vs fl_a:", sim(ref_b, fl_a))
print("ref_b vs fl_b:", sim(ref_b, fl_b))

# What if it's interleaved along dim 0? (i.e. gate, up, gate, up...)
ref_even = w_ref[0::2]
ref_odd = w_ref[1::2]
print("ref_even vs fl_a:", sim(ref_even, fl_a))
print("ref_even vs fl_b:", sim(ref_even, fl_b))
print("ref_odd vs fl_a:", sim(ref_odd, fl_a))
print("ref_odd vs fl_b:", sim(ref_odd, fl_b))

# What if it's transposed?
print("ref.T vs fl:", sim(w_ref.T, w_fl[:5376, :5376]))

# What about other blocks? Check block 1, 2, 3!
for b in [1, 2, 5, 10]:
    kb = f"transformer_blocks.{b}.ff.net.0.proj.weight"
    if kb in d_c and kb in d_fl:
        ra, rb = d_c[kb].float().chunk(2, dim=0)
        fa, fb = d_fl[kb].float().chunk(2, dim=0)
        print(f"B{b} ra vs fa: {sim(ra, fa):.4f}, ra vs fb: {sim(ra, fb):.4f}, rb vs fa: {sim(rb, fa):.4f}, rb vs fb: {sim(rb, fb):.4f}")

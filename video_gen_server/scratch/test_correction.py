import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
import safetensors.torch as st

time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
time_embedder = TimestepEmbedding(
    in_channels=256,
    time_embed_dim=5376,
    act_fn="silu",
    out_dim=2688,
    post_act_fn="silu",
)

d_ref = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors")
time_embedder_dict = {k.replace("time_embedder.", ""): v for k, v in d_ref.items() if k.startswith("time_embedder.")}
time_embedder.load_state_dict(time_embedder_dict)
time_embedder = time_embedder.to("cuda")

# Block 0 for REF - uncorrected
b0_uncorrected = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
dict_ref = {k.replace("transformer_blocks.0.", ""): v.to(torch.bfloat16) for k, v in d_ref.items() if k.startswith("transformer_blocks.0.")}
b0_uncorrected.load_state_dict(dict_ref)
b0_uncorrected = b0_uncorrected.to(device="cuda", dtype=torch.bfloat16)

# Block 0 for REF - CORRECTED (flip SwiGLU chunks!)
b0_corrected = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
dict_corrected = dict_ref.copy()
# Swap ra and rb:
w_ff = dict_ref["ff.net.0.proj.weight"]
ra, rb = w_ff.chunk(2, dim=0)
dict_corrected["ff.net.0.proj.weight"] = torch.cat([rb, ra], dim=0)
b0_corrected.load_state_dict(dict_corrected)
b0_corrected = b0_corrected.to(device="cuda", dtype=torch.bfloat16)

t_val = 0.0588
t_tensor = torch.tensor([t_val], device="cuda", dtype=torch.float32)
temb = time_proj(t_tensor)
temb = time_embedder(temb.to(time_embedder.linear_1.weight.dtype))

seq_len = 1024
torch.manual_seed(42)
hidden_states = (torch.randn(1, seq_len, 5376, device="cuda", dtype=torch.bfloat16) * 2.734 - 0.006)
adaln_indices = torch.zeros(seq_len, dtype=torch.long, device="cuda")
rotary_emb = (torch.ones(seq_len, 48, device="cuda", dtype=torch.bfloat16), torch.zeros(seq_len, 48, device="cuda", dtype=torch.bfloat16))

out_uncorr = b0_uncorrected(hidden_states, temb, adaln_indices, rotary_emb)
out_corr = b0_corrected(hidden_states, temb, adaln_indices, rotary_emb)

print(f"UNCORRECTED B0: mean={out_uncorr.mean().item():.5f}, std={out_uncorr.std().item():.5f}, min={out_uncorr.min().item():.5f}, max={out_uncorr.max().item():.5f}")
print(f"CORRECTED   B0: mean={out_corr.mean().item():.5f}, std={out_corr.std().item():.5f}, min={out_corr.min().item():.5f}, max={out_corr.max().item():.5f}")

# And compare with FL2VA (T2V) output:
d_fl = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors")
b0_fl = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
dict_fl = {k.replace("transformer_blocks.0.", ""): v.to(torch.bfloat16) for k, v in d_fl.items() if k.startswith("transformer_blocks.0.")}
b0_fl.load_state_dict(dict_fl)
b0_fl = b0_fl.to(device="cuda", dtype=torch.bfloat16)
out_fl = b0_fl(hidden_states, temb, adaln_indices, rotary_emb)
print(f"FL2VA (T2V) B0: mean={out_fl.mean().item():.5f}, std={out_fl.std().item():.5f}, min={out_fl.min().item():.5f}, max={out_fl.max().item():.5f}")
print("CORRECTED vs FL2VA cosine similarity:", torch.nn.functional.cosine_similarity(out_corr.flatten(), out_fl.flatten(), dim=0).item())

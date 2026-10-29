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

d = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors")
time_embedder_dict = {k.replace("time_embedder.", ""): v for k, v in d.items() if k.startswith("time_embedder.")}
time_embedder.load_state_dict(time_embedder_dict)
time_embedder = time_embedder.to("cuda")

b0_fl = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
dict_fl = {k.replace("transformer_blocks.0.", ""): v.to(torch.bfloat16) for k, v in d.items() if k.startswith("transformer_blocks.0.")}
b0_fl.load_state_dict(dict_fl)
b0_fl = b0_fl.to(device="cuda", dtype=torch.bfloat16)

t_val = 0.0588
t_tensor = torch.tensor([t_val], device="cuda", dtype=torch.float32)
temb = time_proj(t_tensor)
temb = time_embedder(temb.to(time_embedder.linear_1.weight.dtype))

seq_len = 1024
torch.manual_seed(42)
hidden_states = (torch.randn(1, seq_len, 5376, device="cuda", dtype=torch.bfloat16) * 2.734 - 0.006)
adaln_indices = torch.zeros(seq_len, dtype=torch.long, device="cuda")
rotary_emb = (torch.ones(seq_len, 48, device="cuda", dtype=torch.bfloat16), torch.zeros(seq_len, 48, device="cuda", dtype=torch.bfloat16))

out_fl = b0_fl(hidden_states, temb, adaln_indices, rotary_emb)
print(f"FL BLOCK 0 OUT: mean={out_fl.mean().item():.5f}, std={out_fl.std().item():.5f}, min={out_fl.min().item():.5f}, max={out_fl.max().item():.5f}")

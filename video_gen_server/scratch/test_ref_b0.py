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

# Load time_embedder from transformer_ref
d_ref = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer_ref\diffusion_pytorch_model-00001-of-00013.safetensors")
time_embedder_dict = {k.replace("time_embedder.", ""): v for k, v in d_ref.items() if k.startswith("time_embedder.")}
time_embedder.load_state_dict(time_embedder_dict)
time_embedder = time_embedder.to("cuda")

# Block 0 for REF
b0_ref = MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5)
dict_ref = {k.replace("transformer_blocks.0.", ""): v.to(torch.bfloat16) for k, v in d_ref.items() if k.startswith("transformer_blocks.0.")}
b0_ref.load_state_dict(dict_ref)
b0_ref = b0_ref.to(device="cuda", dtype=torch.bfloat16)

t_val = 0.0588
t_tensor = torch.tensor([t_val], device="cuda", dtype=torch.float32)
temb = time_proj(t_tensor)
temb = time_embedder(temb.to(time_embedder.linear_1.weight.dtype))

seq_len = 1024
torch.manual_seed(42)
hidden_states = (torch.randn(1, seq_len, 5376, device="cuda", dtype=torch.bfloat16) * 2.734 - 0.006)
adaln_indices = torch.zeros(seq_len, dtype=torch.long, device="cuda")
rotary_emb = (torch.ones(seq_len, 48, device="cuda", dtype=torch.bfloat16), torch.zeros(seq_len, 48, device="cuda", dtype=torch.bfloat16))

out_ref = b0_ref(hidden_states, temb, adaln_indices, rotary_emb)
print(f"REF BLOCK 0 OUT: mean={out_ref.mean().item():.5f}, std={out_ref.std().item():.5f}, min={out_ref.min().item():.5f}, max={out_ref.max().item():.5f}")

# Also compare the weights of block 0 between FL and REF!
d_fl = st.load_file(r"C:\Foundry\Griptape\models\MiniMax-H3-FP8\transformer\diffusion_pytorch_model-00001-of-00014.safetensors")
for k in sorted(dict_ref.keys()):
    fl_k = f"transformer_blocks.0.{k}"
    if fl_k in d_fl:
        w_fl = d_fl[fl_k].float()
        w_ref = dict_ref[k].float()
        cos = torch.nn.functional.cosine_similarity(w_fl.flatten(), w_ref.flatten(), dim=0).item()
        diff = (w_ref - w_fl).abs().mean().item()
        print(f"B0 Param {k:35s}: FL std={w_fl.std().item():.4f}, REF std={w_ref.std().item():.4f}, diff={diff:.6f}, cos={cos:.4f}")
    else:
        print(f"B0 Param {k:35s}: in REF but NOT in FL shard 1")

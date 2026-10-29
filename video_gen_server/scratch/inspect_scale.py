import safetensors.torch as st
import json, os

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8/transformer_ref'
if not os.path.exists(path):
    path = 'D:/MiniMax-H3-FP8/transformer_ref'

idx_file = os.path.join(path, 'diffusion_pytorch_model.safetensors.index.json')
with open(idx_file) as f:
    wm = json.load(f)['weight_map']

fp8_scale_keys = [k for k in wm.keys() if 'fp8_scale' in k]
print(f'Total fp8_scale keys: {len(fp8_scale_keys)}')
print('Sample fp8_scale keys:', fp8_scale_keys[:5])

# Load first shard containing fp8_scale
shard = wm[fp8_scale_keys[0]]
data = st.load_file(os.path.join(path, shard))
k_scale = fp8_scale_keys[0]
k_w = k_scale.replace('.fp8_scale', '')
w = data[k_w]
s = data[k_scale]
print(f'Weight {k_w}: shape={w.shape}, dtype={w.dtype}')
print(f'Scale {k_scale}: shape={s.shape}, dtype={s.dtype}, min={s.min().item()}, max={s.max().item()}, mean={s.mean().item()}')

# Also check how diffusers PR 14355 loads weights or how abenliao/minimax-h3-ref2va-transformer-mxfp8 defines it

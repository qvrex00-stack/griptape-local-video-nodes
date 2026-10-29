import torch, json, os, glob
from diffusers import ModularPipeline, ComponentsManager

model_path = r'C:\Foundry\Griptape\models\MiniMax-H3-FP8'
cm = ComponentsManager()
pipe = ModularPipeline.from_pretrained(
    model_path,
    workflow="ref2va",
    components_manager=cm,
    local_files_only=True
)
pipe.load_components(names=["transformer_ref"], dtype=torch.bfloat16, low_cpu_mem_usage=True)
tf = pipe.transformer_ref

# Check how many parameters exist in tf
total_params = dict(tf.named_parameters())
print(f"Total named parameters in transformer_ref: {len(total_params)}")

# Check their values before dequantization
zero_or_nan = 0
for name, p in total_params.items():
    if torch.isnan(p).any() or torch.isinf(p).any():
        print(f"NaN/Inf in {name}")
        zero_or_nan += 1
print(f"NaN/Inf params: {zero_or_nan}")

# Now run _dequantize_transformer_weights from minimax_handler
from video_gen_server.models.minimax_handler import _dequantize_transformer_weights
_dequantize_transformer_weights(tf, model_path, "transformer_ref")

# Check after dequantization:
# 1. Does every linear have _cpu_weight?
# 2. Are any weights all zeros or crazy?
missing_cpu_weights = []
suspicious_weights = []
for name, mod in tf.named_modules():
    if isinstance(mod, torch.nn.Linear):
        if not hasattr(mod, "_cpu_weight") or mod._cpu_weight is None or mod._cpu_weight.numel() == 0:
            missing_cpu_weights.append(name)
        else:
            w = mod._cpu_weight
            if w.std() < 1e-6 or w.std() > 100 or torch.isnan(w).any():
                suspicious_weights.append((name, w.std().item(), w.min().item(), w.max().item()))

print(f"Linear modules missing _cpu_weight: {len(missing_cpu_weights)}")
if missing_cpu_weights:
    print("Sample missing:", missing_cpu_weights[:5])
print(f"Suspicious weights: {len(suspicious_weights)}")
if suspicious_weights:
    print("Sample suspicious:", suspicious_weights[:5])

# Print sample of good weights
for name in ['token_refiner.refiner_blocks.0.attn.to_q', 'transformer_blocks.0.attn.to_q', 'transformer_blocks.49.ff.net.0.proj']:
    parts = name.split('.')
    m = tf
    for p in parts:
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    w = getattr(m, '_cpu_weight', m.weight)
    print(f"{name}: shape={w.shape}, std={w.std().item():.4f}, min={w.min().item():.4f}, max={w.max().item():.4f}")

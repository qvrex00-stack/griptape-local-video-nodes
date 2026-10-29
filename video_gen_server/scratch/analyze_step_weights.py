import torch, json, os
from diffusers import ModularPipeline, ComponentsManager

model_path = r'C:\Foundry\Griptape\models\MiniMax-H3-FP8'

# Let's inspect minimax_handler.py's _pipeline call in R2V
# What parameters were actually passed in user's run?
# User log:
# INFO:video_gen_server:[MiniMax-H3] Adding video reference: C:\Foundry\Griptape\inputs\videos\runAnim_shot_1.mov
# INFO:video_gen_server:[MiniMax-H3] R2V: 640x544, 124f (~5.2s), 15steps, seed=1973797832
# [DIAG STEP 0] latents: shape=torch.Size([24676, 96]), mean=-0.00783, std=1.03930
# [DIAG STEP 0] prompt_embeds: shape=torch.Size([1, 2708, 5120]), mean=0.01001, std=10.75000
# [DIAG STEP 0] noise_pred: shape=torch.Size([1, 24676, 96]), mean=0.05884, std=0.46484

# Let's check:
# 1. In denoise.py:
#    gen_noise_pred = block_state.noise_pred[0, num_condition_video_rows:].float()
#    gen_latents = block_state.latents[num_condition_video_rows:]
#    stepped = components.scheduler.step(gen_noise_pred, t, gen_latents, return_dict=False)[0]
#    block_state.latents[num_condition_video_rows:] = stepped

# Let's check scheduler.step with actual numbers from log:
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
s = MiniMaxH3Scheduler(shift=12.0)
s.set_timesteps(15)

# Simulate step 0
t0 = s.timesteps[0] # 0.0000
sigma0 = s.sigmas[0] # 1.0000
sigma1 = s.sigmas[1] # 0.9936
ratio0 = (sigma1 / sigma0).item()
print(f"Step 0: t={t0.item():.4f}, sigma={sigma0.item():.4f}, sigma_next={sigma1.item():.4f}, ratio={ratio0:.4f}")

# At step 0:
# sample is gen_latents (std = 0.9977)
# noise_pred has std = 0.4648
# denoised = sample + (1 - t0) * noise_pred = sample + 1.0 * noise_pred
# stepped = ratio * sample + (1 - ratio) * denoised
#         = 0.9936 * sample + 0.0064 * (sample + noise_pred)
#         = sample + 0.0064 * noise_pred !
# Notice that at step 0, stepped only moves by 0.0064 * noise_pred !!
print(f"Update weight at step 0: (1 - ratio0) = {1.0 - ratio0:.6f} !!")

# What about all steps?
for i, t in enumerate(s.timesteps):
    sig = s.sigmas[i]
    signext = s.sigmas[i+1]
    r = (signext / sig).item()
    eff_weight = (1.0 - r) * (1.0 - t.item())
    print(f"Step {i:2d}: t={t.item():.4f}, sigma={sig.item():.4f} -> {signext.item():.4f}, ratio={r:.4f}, (1-ratio)={1.0-r:.4f}, effective velocity weight={eff_weight:.4f}")

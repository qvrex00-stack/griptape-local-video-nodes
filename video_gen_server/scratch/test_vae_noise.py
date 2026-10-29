import torch
from diffusers import AutoencoderKLMiniMaxH3
import cv2, numpy as np, os

path = 'C:/Foundry/Griptape/models/MiniMax-H3-FP8'
vae = AutoencoderKLMiniMaxH3.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32).cuda()

# Latent shape: (1, 24, num_frames=13, h=68, w=80) for 640x544, 124 frames (124/8 = 15.5 -> 13 latent frames)
# Let's test with 5 frames (minimum temporal group for VAE): (1, 24, 5, 68, 80)
# Random noise with std = 0.7332 (same as Step 13 output)
noise = torch.randn(1, 24, 5, 68, 80, device='cuda', dtype=torch.float32) * 0.7332

# VAE denormalization and decode
mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)

denorm_latents = noise * std + mean

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.float16):
        video = vae.decode(denorm_latents).sample

# Convert to uint8 RGB
video = (video[0, :, 0].permute(1, 2, 0).cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()
video = (video * 255).astype(np.uint8)
video_bgr = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

cv2.imwrite('C:/Foundry/Griptape/video_gen_server/scratch/vae_noise_output.png', video_bgr)
print('Saved vae_noise_output.png successfully!')

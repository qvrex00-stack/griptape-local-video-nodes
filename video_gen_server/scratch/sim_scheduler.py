import torch
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

def simulate_denoise(num_steps, shift=12.0):
    s = MiniMaxH3Scheduler(shift=shift)
    s.set_timesteps(num_steps)
    
    # Ground truth data x0 (std ~ 0.8) and pure noise eps (std = 1.0)
    torch.manual_seed(42)
    x0 = torch.randn(1, 1000) * 0.8
    eps = torch.randn(1, 1000)
    
    # Initial sample at t=0 (sigma=1.0) is pure noise
    sample = eps.clone()
    
    print(f"\n--- Simulation with {num_steps} steps (shift={shift}) ---")
    for i, t in enumerate(s.timesteps):
        sigma = 1.0 - t
        # Perfect model prediction: v = x0 - eps
        # In actual diffusion at time t: x_t = (1-sigma)*x0 + sigma*eps
        # v_ideal = x0 - eps
        v_ideal = x0 - eps
        
        # Scheduler step
        sample = s.step(v_ideal, t, sample, return_dict=False)[0]
        # Measure error to x0
        err = (sample - x0).abs().mean().item()
        std = sample.std().item()
        if i in (0, 1, num_steps // 2, len(s.timesteps) - 2, len(s.timesteps) - 1):
            print(f"Step {i:2d} (t={t.item():.4f}, sigma={sigma.item():.4f}): sample std={std:.4f}, MAE to x0={err:.4f}")

simulate_denoise(15, shift=12.0)
simulate_denoise(30, shift=12.0)
simulate_denoise(50, shift=12.0)
simulate_denoise(15, shift=3.0)
simulate_denoise(30, shift=3.0)

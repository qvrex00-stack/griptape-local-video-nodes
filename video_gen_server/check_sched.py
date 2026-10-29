import torch
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

for steps in [15, 30]:
    for shift in [3.5, 12.0]:
        s = MiniMaxH3Scheduler(shift=shift)
        s.set_timesteps(steps)
        print(f"=== Steps={steps}, Shift={shift} ===")
        print("Sigmas (len", len(s.sigmas), "):", [round(x.item(), 4) for x in s.sigmas])
        print("Timesteps (len", len(s.timesteps), "):", [round(x.item(), 4) for x in s.timesteps])
        print("Terminal sigma:", s.sigmas[-1].item())

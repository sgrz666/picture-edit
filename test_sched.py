import torch
from diffusers import FlowMatchEulerDiscreteScheduler

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/scheduler"
)
scheduler.set_timesteps(25)
print("timesteps:", scheduler.timesteps)
print("sigmas:", scheduler.sigmas)

x = torch.randn(1, 16, 64, 64)
v = torch.randn(1, 16, 64, 64)
t = scheduler.timesteps[0]
out = scheduler.step(v, t, x)
print("out prev_sample min/max/std:", out.prev_sample.min().item(), out.prev_sample.max().item(), out.prev_sample.std().item())
print("difference between x and out:", (out.prev_sample - x).abs().mean().item())
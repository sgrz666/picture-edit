import sys
sys.path.insert(0, "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
import torch
from diffusers import DiffusionPipeline

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Let's inspect step by step
def step_callback(pipe_self, step_idx, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    print(f"Step {step_idx}: t={timestep.item() if hasattr(timestep, 'item') else timestep}, latents mean={latents.mean().item():.3f}, std={latents.std().item():.3f}, min={latents.min().item():.3f}, max={latents.max().item():.3f}", flush=True)
    return callback_kwargs

res = pipe(
    "a photo of a cute red apple on a wooden table",
    height=512, width=512,
    num_inference_steps=10,
    guidance_scale=4.5,
    seed=42,
    callback_on_step_end=step_callback,
    callback_on_step_end_tensor_inputs=["latents"]
)
out_img = res.images[0]
out_img.save("/home/shangguanrz/project/pic-edit/demooutput/test_step_apple.png")
print("Saved /home/shangguanrz/project/pic-edit/demooutput/test_step_apple.png", flush=True)
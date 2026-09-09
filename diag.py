import sys
sys.path.insert(0, "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import numpy as np

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
print("Loading pipe...", flush=True)
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
pipe.to("cuda")

# 1. Test VAE reconstruction
src_img = Image.open("/home/shangguanrz/project/pic-edit/inputs/person1.jpg").convert("RGB").resize((512, 512))
arr = (np.array(src_img).astype(np.float32) / 127.5) - 1.0 # [-1, 1]
t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    z = pipe.pixels_to_latents(t_img)
    print(f"VAE encoded latents: shape={z.shape}, min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}, std={z.std():.3f}", flush=True)
    rec = pipe.latents_to_pixels(z)
    print(f"VAE decoded pixels: shape={rec.shape}, min={rec.min():.3f}, max={rec.max():.3f}, mean={rec.mean():.3f}, std={rec.std():.3f}", flush=True)
    rec_img = torch.clamp(127.5 * rec[0].permute(1, 2, 0) + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    Image.fromarray(rec_img).save("/home/shangguanrz/project/pic-edit/demooutput/diag_vae_rec.png")
    print("Saved VAE reconstruction to diag_vae_rec.png", flush=True)
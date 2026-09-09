import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL

vae_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/vae"
print(f"Loading VAE from {vae_path} on CUDA...")
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.bfloat16).to(
    "cuda"
)
print("✓ VAE loaded successfully!")

img_path = (
    "/home/shangguanrz/project/pic-edit/data/debug_samples/images/sample_000_src.png"
)
img = Image.open(img_path).convert("RGB")
img_tensor = (
    torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    / 127.5
    - 1.0
)
img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

with torch.no_grad():
  latents = vae.encode(img_tensor).latent_dist.sample()
  print(f"✓ Latent shape: {latents.shape}")
  recon = vae.decode(latents).sample
  print(f"✓ Reconstruction shape: {recon.shape}")

print("\n*** VAE ENCODE / DECODE TEST PASSED! ***")

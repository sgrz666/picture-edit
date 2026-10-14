import argparse
import json
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
from src.adapter.pose_adapter import PoseConditionAdapter
from diffusers import DiffusionPipeline


class PairedPoseDataset(Dataset):
  """Dataset for paired image editing with 8-channel SMPL-X control maps."""

  def __init__(self, data_dir: str, resolution: int = 512):
    self.data_dir = data_dir
    self.resolution = resolution
    manifest_path = os.path.join(data_dir, "pairs.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
      self.manifest = json.load(f)

  def __len__(self):
    return len(self.manifest)

  def __getitem__(self, idx):
    item = self.manifest[idx]
    images_dir = os.path.join(self.data_dir, "images")
    controls_dir = os.path.join(self.data_dir, "controls")

    # Load source & target images
    src_img = (
        Image.open(os.path.join(images_dir, item["src_image"]))
        .convert("RGB")
        .resize((self.resolution, self.resolution))
    )
    tgt_img = (
        Image.open(os.path.join(images_dir, item["tgt_image"]))
        .convert("RGB")
        .resize((self.resolution, self.resolution))
    )

    # Normalize to [-1, 1]
    src_tensor = (
        torch.from_numpy(np.array(src_img)).float().permute(2, 0, 1) / 127.5
        - 1.0
    )
    tgt_tensor = (
        torch.from_numpy(np.array(tgt_img)).float().permute(2, 0, 1) / 127.5
        - 1.0
    )

    # Load precomputed 8ch control map: (H, W, 8) -> (8, H, W)
    control_path = os.path.join(controls_dir, item["control_tensor"])
    control_np = np.load(control_path).astype(np.float32)
    control_tensor = torch.from_numpy(control_np).permute(2, 0, 1)

    return {
        "src_image": src_tensor,
        "tgt_image": tgt_tensor,
        "control_map": control_tensor,
        "prompt": item["prompt"],
    }


def train_adapter(args):
  print("=== STARTING POSE ADAPTER TRAINING PIPELINE ===", flush=True)
  device = torch.device(args.device)
  dtype = torch.bfloat16 if args.bf16 else torch.float32
  os.makedirs(args.output_dir, exist_ok=True)

  # 1. Load pipeline and freeze all components
  print(f"Loading base DeepGen model from {args.model_path}...", flush=True)
  pipe = DiffusionPipeline.from_pretrained(
      args.model_path,
      torch_dtype=dtype,
      trust_remote_code=True,
  )
  pipe.to(device)

  # Freeze entire base pipeline
  pipe.vae.eval().requires_grad_(False)
  pipe.transformer.eval().requires_grad_(False)
  if hasattr(pipe, "vlm") and pipe.vlm is not None:
    pipe.vlm.eval().requires_grad_(False)
  if hasattr(pipe, "connector") and pipe.connector is not None:
    pipe.connector.eval().requires_grad_(False)

  transformer = pipe.transformer
  vae = pipe.vae

  if args.gradient_checkpointing and hasattr(
      transformer, "enable_gradient_checkpointing"
  ):
    transformer.enable_gradient_checkpointing()
    print("✓ Gradient checkpointing enabled on transformer", flush=True)

  # 2. Instantiate Trainable Pose Adapter
  hidden_size = transformer.config.hidden_size
  num_blocks = len(transformer.transformer_blocks)
  adapter = PoseConditionAdapter(
      in_channels=8,
      hidden_dim=hidden_size,
      num_injection_layers=num_blocks,
  ).to(device, dtype=dtype)
  adapter.train()

  trainable_params = adapter.count_parameters()
  print(
      f"✓ Adapter trainable parameters: {trainable_params / 1e6:.2f} M",
      flush=True,
  )

  # 3. Setup Optimizer
  optimizer = torch.optim.AdamW(
      adapter.parameters(),
      lr=args.learning_rate,
      betas=(0.9, 0.999),
      weight_decay=1e-2,
  )

  # 4. DataLoader
  dataset = PairedPoseDataset(args.data_dir, resolution=args.resolution)
  dataloader = DataLoader(
      dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
  )
  print(f"✓ Dataset loaded with {len(dataset)} paired samples", flush=True)

  # 5. Training Loop
  global_step = 0
  step_times = []
  t_train_start = time.time()

  print(
      f"\nStarting training for {args.max_train_steps} steps (accumulation"
      f"={args.gradient_accumulation_steps})...\n",
      flush=True,
  )

  optimizer.zero_grad()
  while global_step < args.max_train_steps:
    for batch in dataloader:
      if global_step >= args.max_train_steps:
        break

      t_step_0 = time.time()
      tgt_images = batch["tgt_image"].to(device, dtype=dtype)
      control_maps = batch["control_map"].to(device, dtype=dtype)

      # VAE Encode target image to latent space
      with torch.no_grad():
        latents = vae.encode(tgt_images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

      # Sample Flow Matching timestep: t in [0, 1000]
      B = latents.shape[0]
      timesteps = torch.rand(B, device=device) * 1000.0
      sigma = (timesteps / 1000.0).view(B, 1, 1, 1).to(dtype)

      # Noise and velocity target: x_t = (1 - sigma)*x_0 + sigma*x_1
      noise = torch.randn_like(latents)
      noisy_latents = (1.0 - sigma) * noise + sigma * latents
      target_velocity = latents - noise

      # Dummy or cached text condition (joint attention)
      dummy_encoder_hidden_states = torch.zeros(
          B, 64, transformer.config.joint_attention_dim, device=device, dtype=dtype
      )

      # Adapter forward
      residuals = adapter(control_maps)

      # Transformer forward with adapter residuals
      model_pred = transformer(
          hidden_states=noisy_latents,
          timestep=timesteps,
          encoder_hidden_states=dummy_encoder_hidden_states,
          block_controlnet_hidden_states=residuals,
          return_dict=False,
      )[0]

      # Flow matching loss
      loss = F.mse_loss(model_pred.float(), target_velocity.float(), reduction="mean")
      loss = loss / args.gradient_accumulation_steps
      loss.backward()

      if (global_step + 1) % args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

      torch.cuda.synchronize()
      step_duration = time.time() - t_step_0
      step_times.append(step_duration)
      global_step += 1

      if global_step % args.log_steps == 0 or global_step == 1:
        alloc = torch.cuda.max_memory_allocated() / (1024**3)
        resv = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"Step {global_step:4d}/{args.max_train_steps} | Loss:"
            f" {loss.item() * args.gradient_accumulation_steps:.4f} | Speed:"
            f" {step_duration*1000:.1f}ms/step | VRAM Alloc: {alloc:.2f} GiB,"
            f" Resv: {resv:.2f} GiB",
            flush=True,
        )

      # Checkpoint saving
      if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
        ckpt_path = os.path.join(
            args.output_dir, f"adapter_step_{global_step}.pt"
        )
        torch.save(
            {
                "step": global_step,
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"✓ Checkpoint saved to {ckpt_path}", flush=True)

  total_time = time.time() - t_train_start
  print(
      f"\n=== TRAINING COMPLETE: {global_step} steps in {total_time:.2f}s ===",
      flush=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_path",
      type=str,
      default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/home/shangguanrz/project/pic-edit/data/debug_samples",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/home/shangguanrz/project/pic-edit/experiments/adapter_v1",
  )
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--max_train_steps", type=int, default=100)
  parser.add_argument("--log_steps", type=int, default=10)
  parser.add_argument("--save_steps", type=int, default=50)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--bf16", action="store_true", default=True)
  parser.add_argument(
      "--gradient_checkpointing", action="store_true", default=True
  )
  args = parser.parse_args()

  train_adapter(args)

import os
import sys
import time
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
from src.adapter.pose_adapter import PoseConditionAdapter
from diffusers import DiffusionPipeline


def run_backward_pass_and_benchmark():
  print("=== TESTING FROZEN BACKBONE + ADAPTER BACKWARD PASS ===", flush=True)
  device = "cuda"
  dtype = torch.bfloat16
  model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
  if model_path not in sys.path:
    sys.path.insert(0, model_path)
  from deepgen_pipeline import DeepGenPipeline

  print(f"Loading transformer model from {model_path}...", flush=True)
  pipe = DeepGenPipeline.from_pretrained(
      model_path,
      torch_dtype=dtype,
  )
  transformer = pipe.transformer.to(device)

  # 1. Freeze all backbone parameters
  transformer.eval()
  for name, param in transformer.named_parameters():
    param.requires_grad = False

  # Verify backbone is 100% frozen
  trainable_backbone = sum(
      p.numel() for p in transformer.parameters() if p.requires_grad
  )
  print(
      f"✓ Backbone parameters frozen: {trainable_backbone} trainable parameters"
      " (expected 0)",
      flush=True,
  )
  assert trainable_backbone == 0, "Error: Backbone has trainable parameters!"

  # 2. Instantiate Pose Adapter
  # DeepGen DiT has hidden_size = num_attention_heads * attention_head_dim (1536)
  hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim  # 1536
  num_blocks = len(transformer.transformer_blocks)  # 24
  num_injection_layers = (
      num_blocks  # Provide residuals to all blocks or subset
  )

  adapter = PoseConditionAdapter(
      in_channels=8,
      hidden_dim=hidden_size,
      num_injection_layers=num_injection_layers,
  ).to(device, dtype=dtype)
  adapter.train()

  adapter_params = adapter.count_parameters()
  print(
      f"✓ Adapter trainable parameters: {adapter_params / 1e6:.2f} M (budget"
      " target: 100M - 300M)",
      flush=True,
  )
  assert (
      100e6 <= adapter_params <= 300e6
  ), f"Adapter parameters {adapter_params} out of budget!"

  optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

  # 3. Test Zero-Init Behavior (Residuals are strictly zero)
  B = 1
  control_map = torch.randn(B, 8, 512, 512, device=device, dtype=dtype)
  residuals_0 = adapter(control_map)
  max_init_val = max(r.abs().max().item() for r in residuals_0)
  print(
      f"✓ Adapter zero-conv initialization check: max abs residual ="
      f" {max_init_val:.8f}",
      flush=True,
  )
  assert (
      max_init_val == 0.0
  ), "Adapter must be strictly zero-initialized for baseline equivalence!"

  # 4. Prepare Dummy Forward Inputs for DiT
  # Latent tokens: 512x512 / 8 VAE downsample = 64x64.
  in_channels = transformer.config.in_channels
  dummy_latents = torch.randn(
      B, in_channels, 64, 64, device=device, dtype=dtype
  )
  dummy_timestep = torch.tensor([500.0], device=device, dtype=dtype)
  dummy_encoder_hidden_states = torch.randn(
      B, 64, transformer.config.joint_attention_dim, device=device, dtype=dtype
  )
  dummy_pooled_projections = torch.zeros(
      B, transformer.config.pooled_projection_dim, device=device, dtype=dtype
  )

  # 5. One Forward + Backward Pass
  print("\nExecuting forward pass with adapter residuals injection...")
  residuals = adapter(control_map)

  # Forward DiT with block_controlnet_hidden_states
  output = transformer(
      hidden_states=dummy_latents,
      timestep=dummy_timestep,
      encoder_hidden_states=dummy_encoder_hidden_states,
      pooled_projections=dummy_pooled_projections,
      block_controlnet_hidden_states=residuals,
      return_dict=False,
  )[0]

  target_velocity = torch.randn_like(output)
  loss = F.mse_loss(output, target_velocity)
  print(f"✓ Loss computed: {loss.item():.4f}")

  print("Executing backward pass...")
  optimizer.zero_grad()
  loss.backward()

  # 6. Verify Gradients
  adapter_has_grad = False
  for name, param in adapter.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
      adapter_has_grad = True
      break

  backbone_has_grad = any(
      param.grad is not None for param in transformer.parameters()
  )

  print(f"✓ Adapter has non-zero gradients: {adapter_has_grad}")
  print(
      f"✓ Backbone has NO gradients (strictly frozen):"
      f" {not backbone_has_grad}"
  )

  assert adapter_has_grad, "Error: Adapter received no gradients!"
  assert (
      not backbone_has_grad
  ), "Error: Backbone parameters received gradients despite being frozen!"

  optimizer.step()
  print("✓ Optimizer step successful! Only adapter parameters updated.")

  # 7. Steady-state 100 Updates Benchmark and VRAM Profiling
  print("\n--- Running 100 Steady-State Update Steps Profiling ---")
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  t_bench_start = time.time()

  step_times = []
  for step in range(1, 101):
    t_step_0 = time.time()
    optimizer.zero_grad()

    # Online random noise and timesteps
    c_map = torch.randn(B, 8, 512, 512, device=device, dtype=dtype)
    res = adapter(c_map)

    out = transformer(
        hidden_states=dummy_latents,
        timestep=dummy_timestep,
        encoder_hidden_states=dummy_encoder_hidden_states,
        pooled_projections=dummy_pooled_projections,
        block_controlnet_hidden_states=res,
        return_dict=False,
    )[0]

    loss = F.mse_loss(out, target_velocity)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    step_duration = time.time() - t_step_0
    step_times.append(step_duration)

    if step in (1, 10, 50, 100):
      alloc = torch.cuda.max_memory_allocated() / (1024**3)
      resv = torch.cuda.max_memory_reserved() / (1024**3)
      print(
          f"Step {step:3d}/100 | Time: {step_duration*1000:.1f} ms | VRAM"
          f" Allocated: {alloc:.2f} GiB | VRAM Reserved: {resv:.2f} GiB"
      )

  total_bench_time = time.time() - t_bench_start
  avg_step_ms = (sum(step_times) / len(step_times)) * 1000
  peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
  peak_resv = torch.cuda.max_memory_reserved() / (1024**3)

  print(f"\nBenchmark completed 100 steps in {total_bench_time:.2f}s")
  print(f"Average step time: {avg_step_ms:.2f} ms (~{1000/avg_step_ms:.1f} it/s)")
  print(f"Peak VRAM Allocated: {peak_alloc:.2f} GiB")
  print(f"Peak VRAM Reserved:  {peak_resv:.2f} GiB")
  print(
      f"Total GPU VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB"
  )
  print(f"GPU VRAM Headroom Remaining: {83.05 - peak_resv:.2f} GiB")
  print("\n*** ALL BACKWARD PASS AND BENCHMARK CHECKS PASSED! ***")


if __name__ == "__main__":
  run_backward_pass_and_benchmark()

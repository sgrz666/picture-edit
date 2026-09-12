"""Overfit PoseConditionAdapter on one aligned CHAMP source/target pair."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("DEEPGEN_PROJECT", str(LOCAL_PROJECT_ROOT))
).resolve()
DEFAULT_MODEL_PATH = DEFAULT_PROJECT_ROOT / "models" / "DeepGen-1.0-diffusers"

sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))
sys.path.insert(0, str(DEFAULT_MODEL_PATH))

from src.adapter.pose_adapter import PoseConditionAdapter
from src.integration.champ_deepgen import (
    align_control_residuals,
    image_diagnostics,
    load_champ_sample,
)


NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, broken content, "
    "missing parts, damaged details, artifacts, glitch, noise, extra fingers, "
    "missing fingers, mutated hands, bad composition, wrong proportion, unfinished"
)

DEFAULT_PROMPT = (
    "Change only the woman's pose so that both hands form a heart gesture in front "
    "of her chest. Preserve the exact same identity, face, long dark hair, white "
    "clothing, body proportions, lighting, and grey curtain background."
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="CHAMP controls + PoseAdapter + DeepGen single-sample overfit"
  )
  parser.add_argument(
      "--data_dir",
      default=str(DEFAULT_PROJECT_ROOT / "inputs" / "champ_sample"),
  )
  parser.add_argument(
      "--output_dir",
      default=str(DEFAULT_PROJECT_ROOT / "experiments" / "champ_deepgen_formal_v1"),
  )
  parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--max_steps", type=int, default=120)
  parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5)
  parser.add_argument("--save_every", type=int, default=30)
  parser.add_argument("--inference_steps", type=int, default=30)
  parser.add_argument("--guidance_scale", type=float, default=4.5)
  parser.add_argument("--control_scale", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--prompt", default=DEFAULT_PROMPT)
  parser.add_argument("--validate_only", action="store_true")
  return parser.parse_args()


def _save_json(path: Path, value: object) -> None:
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2),
      encoding="utf-8",
  )


def _control_image(control: torch.Tensor, channels: slice | int) -> Image.Image:
  array = control[0, channels].detach().cpu().float().numpy()
  if array.ndim == 2:
    array = np.repeat(array[None], 3, axis=0)
  array = np.transpose(array, (1, 2, 0))
  return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _banner(image: Image.Image, title: str, color: tuple[int, int, int]) -> Image.Image:
  image = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
  canvas = Image.new("RGB", (512, 556), color=color)
  canvas.paste(image, (0, 44))
  ImageDraw.Draw(canvas).text((12, 12), title, fill=(255, 255, 255))
  return canvas


def _save_comparison(
    path: Path,
    source: Image.Image,
    target: Image.Image,
    control: torch.Tensor,
    outputs: dict[int, Image.Image],
) -> None:
  columns = [
      _banner(source, "[1] Source image A", (20, 60, 120)),
      _banner(_control_image(control, slice(4, 7)), "[2] Target pose", (20, 80, 80)),
      _banner(_control_image(control, slice(0, 3)), "[3] Target normal", (40, 80, 40)),
      _banner(_control_image(control, 3), "[4] Target depth", (50, 70, 90)),
      _banner(_control_image(control, 7), "[5] Target semantic", (70, 70, 70)),
      _banner(target, "[6] Ground truth target", (0, 120, 0)),
  ]
  palette = [(120, 60, 20), (140, 100, 20), (120, 80, 20), (100, 60, 120)]
  for index, step in enumerate(sorted(outputs)):
    suffix = "untrained" if step == 0 else "trained"
    columns.append(
        _banner(
            outputs[step],
            f"[{7 + index}] Model step {step} ({suffix})",
            palette[index % len(palette)],
        )
    )
  grid = Image.new("RGB", (512 * len(columns), 556), color=(20, 20, 20))
  for index, column in enumerate(columns):
    grid.paste(column, (index * 512, 0))
  grid.save(path)


def _residual_diagnostics(residuals: list[torch.Tensor]) -> dict[str, float]:
  squared_sum = torch.zeros((), dtype=torch.float64)
  count = 0
  maximum = 0.0
  for residual in residuals:
    detached = residual.detach().double()
    squared_sum += detached.square().sum().cpu()
    count += detached.numel()
    maximum = max(maximum, float(detached.abs().max().cpu()))
  residual_rms = float(torch.sqrt(squared_sum / max(count, 1)))
  return {"residual_rms": residual_rms, "residual_max_abs": maximum}


def _validate_configuration(args: argparse.Namespace) -> None:
  if args.max_steps <= 0:
    raise ValueError("max_steps must be positive")
  if args.save_every <= 0:
    raise ValueError("save_every must be positive")
  if args.learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
  if args.inference_steps <= 0:
    raise ValueError("inference_steps must be positive")


def main() -> None:
  args = parse_args()
  _validate_configuration(args)
  sample = load_champ_sample(args.data_dir, resolution=args.resolution)
  validation = {
      **sample.metadata,
      "control_shape": list(sample.control.shape),
      "control_min": float(sample.control.min()),
      "control_max": float(sample.control.max()),
      "target_diagnostics": image_diagnostics(sample.target),
  }
  print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
  if args.validate_only:
    print("CHAMP sample validation: OK", flush=True)
    return

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  model_path = Path(args.model_path)
  if not model_path.is_dir():
    raise FileNotFoundError(f"DeepGen model directory is missing: {model_path}")

  from diffusers import DiffusionPipeline

  device = torch.device("cuda")
  dtype = torch.bfloat16
  torch.manual_seed(args.seed)

  print(f"Loading DeepGen pipeline from {model_path}", flush=True)
  pipe = DiffusionPipeline.from_pretrained(
      str(model_path),
      torch_dtype=dtype,
  ).to(device)
  pipe.vae.to(device, dtype=dtype)
  pipe.transformer.to(device, dtype=dtype)
  pipe._load_extras(attn_implementation="sdpa")

  pipe.transformer.eval().requires_grad_(False)
  pipe.vae.eval().requires_grad_(False)
  if getattr(pipe, "llm", None) is not None:
    pipe.llm.eval().requires_grad_(False)
  if getattr(pipe, "connector_module", None) is not None:
    pipe.connector_module.eval().requires_grad_(False)

  transformer = pipe.transformer
  source = sample.source
  target = sample.target
  control_tensor = sample.control.to(device=device, dtype=torch.float32)

  with torch.no_grad():
    source_pixels = (
        torch.from_numpy(np.asarray(source).copy()).float() / 127.5 - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    target_pixels = (
        torch.from_numpy(np.asarray(target).copy()).float() / 127.5 - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

    image_embeds, image_grid = pipe.get_semantic_features_dynamic([source_pixels[0]])
    text_inputs = pipe.prepare_image2image_prompts(
        [args.prompt],
        num_refs=[1],
        ref_lens=[len(image_embeds[0])],
    )
    text_inputs.update(
        image_embeds=torch.cat(image_embeds),
        image_grid_thw=image_grid,
    )
    queries = pipe.connector_module.meta_queries[None]
    forward_inputs = pipe.prepare_forward_input(query_embeds=queries, **text_inputs)
    llm_output = pipe.llm(
        **forward_inputs,
        return_dict=True,
        output_hidden_states=True,
    )
    hidden_states = llm_output.hidden_states
    merged = torch.cat(
        [hidden_states[index] for index in range(len(hidden_states) - 2, 0, -6)],
        dim=-1,
    )
    pooled, sequence = pipe.connector_module.llm2dit(merged)
    reference_latent = pipe.pixels_to_latents(source_pixels)
    target_latent = pipe.pixels_to_latents(target_pixels)

  hidden_size = getattr(transformer.config, "hidden_size", None)
  if hidden_size is None:
    hidden_size = (
        transformer.config.num_attention_heads
        * transformer.config.attention_head_dim
    )
  adapter = PoseConditionAdapter(
      in_channels=8,
      hidden_dim=hidden_size,
      num_blocks=4,
      num_injection_layers=6,
  ).to(device=device, dtype=torch.float32)
  optimizer = torch.optim.AdamW(
      adapter.parameters(),
      lr=args.learning_rate,
      betas=(0.9, 0.999),
      weight_decay=1e-2,
  )

  target_tokens = (args.resolution // 16) ** 2
  reference_tokens = (
      reference_latent.shape[-2] // transformer.config.patch_size
  ) * (
      reference_latent.shape[-1] // transformer.config.patch_size
  )

  def make_aligned_residuals() -> list[torch.Tensor]:
    raw_residuals = adapter(control_tensor)
    return align_control_residuals(
        raw_residuals,
        target_tokens=target_tokens,
        reference_tokens=reference_tokens,
        batch_size=1,
    )

  def run_controlled_inference(step: int) -> tuple[Image.Image, dict[str, object]]:
    adapter.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
      aligned_residuals = make_aligned_residuals()
      output = pipe(
          prompt=args.prompt,
          image=source,
          negative_prompt=NEGATIVE_PROMPT,
          height=args.resolution,
          width=args.resolution,
          num_inference_steps=args.inference_steps,
          guidance_scale=args.guidance_scale,
          seed=args.seed,
          block_controlnet_hidden_states=aligned_residuals,
          control_scale=args.control_scale,
      ).images[0]
    adapter.train()
    diagnostics = {
        "step": step,
        **_residual_diagnostics(aligned_residuals),
        **image_diagnostics(output),
    }
    return output, diagnostics

  sample.source.save(output_dir / "01_INPUT_SOURCE_IMAGE.png")
  sample.target.save(output_dir / "02_GROUND_TRUTH_TARGET_IMAGE.png")
  _control_image(sample.control, slice(0, 3)).save(
      output_dir / "03_INPUT_CONTROL_NORMAL.png"
  )
  _control_image(sample.control, 3).save(output_dir / "03_INPUT_CONTROL_DEPTH.png")
  _control_image(sample.control, slice(4, 7)).save(
      output_dir / "03_INPUT_CONTROL_POSE.png"
  )
  _control_image(sample.control, 7).save(
      output_dir / "03_INPUT_CONTROL_SEMANTIC.png"
  )

  config = {
      "data_dir": str(Path(args.data_dir).resolve()),
      "model_path": str(model_path.resolve()),
      "prompt": args.prompt,
      "negative_prompt": NEGATIVE_PROMPT,
      "resolution": args.resolution,
      "max_steps": args.max_steps,
      "learning_rate": args.learning_rate,
      "save_every": args.save_every,
      "inference_steps": args.inference_steps,
      "guidance_scale": args.guidance_scale,
      "control_scale": args.control_scale,
      "seed": args.seed,
      "control_channels": list(sample.metadata["control_channels"]),
  }
  _save_json(output_dir / "config.json", config)
  _save_json(
      output_dir / "environment.json",
      {
          "python": platform.python_version(),
          "torch": torch.__version__,
          "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(device),
      },
  )

  outputs: dict[int, Image.Image] = {}
  diagnostic_records: list[dict[str, object]] = []
  loss_history: list[dict[str, float | int]] = []

  print("Running Step 0 formal-interface baseline", flush=True)
  output, diagnostic = run_controlled_inference(0)
  outputs[0] = output
  diagnostic_records.append(diagnostic)
  output.save(output_dir / "04_MODEL_OUTPUT_STEP_000_UNTRAINED.png")

  generator = torch.Generator(device=device).manual_seed(args.seed)
  started = time.time()
  for step in range(1, args.max_steps + 1):
    optimizer.zero_grad(set_to_none=True)
    noise = torch.randn(
        target_latent.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    timestep = torch.rand((1,), device=device, generator=generator) * 1000.0
    sigma = (timestep / 1000.0).reshape(1, 1, 1, 1).to(dtype)
    noisy_latent = (1.0 - sigma) * target_latent + sigma * noise

    with torch.autocast("cuda", dtype=dtype):
      training_residuals = make_aligned_residuals()
      prediction = transformer(
          hidden_states=noisy_latent,
          encoder_hidden_states=sequence,
          pooled_projections=pooled,
          cond_hidden_states=[[reference_latent[0]]],
          timestep=timestep,
          block_controlnet_hidden_states=[
              value.to(dtype=dtype) for value in training_residuals
          ],
          return_dict=False,
      )[0]
      target_velocity = noise - target_latent
      loss = F.mse_loss(prediction.float(), target_velocity.float())

    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
    optimizer.step()
    residual_stats = _residual_diagnostics(training_residuals)
    record = {
        "step": step,
        "loss": float(loss.item()),
        "gradient_norm": gradient_norm,
        **residual_stats,
    }
    loss_history.append(record)

    if step == 1 or step % 10 == 0:
      print(
          f"step={step:04d}/{args.max_steps} loss={record['loss']:.6f} "
          f"grad={gradient_norm:.4f} residual_rms={record['residual_rms']:.6f}",
          flush=True,
      )

    if step % args.save_every == 0 or step == args.max_steps:
      checkpoint = output_dir / f"adapter_step_{step:03d}.pt"
      torch.save(
          {
              "step": step,
              "adapter_state_dict": adapter.state_dict(),
              "config": config,
          },
          checkpoint,
      )
      generated, output_diagnostic = run_controlled_inference(step)
      outputs[step] = generated
      diagnostic_records.append(output_diagnostic)
      generated.save(output_dir / f"05_MODEL_OUTPUT_STEP_{step:03d}.png")
      _save_json(output_dir / "loss_history.json", loss_history)
      _save_json(output_dir / "diagnostics.json", diagnostic_records)

  duration = time.time() - started
  valid_steps = [
      int(record["step"])
      for record in diagnostic_records
      if bool(record["valid"]) and int(record["step"]) > 0
  ]
  summary = {
      "duration_seconds": duration,
      "completed_steps": args.max_steps,
      "valid_output_steps": valid_steps,
      "all_outputs_valid": len(valid_steps) == len(outputs) - 1,
      "first_loss": loss_history[0]["loss"],
      "last_loss": loss_history[-1]["loss"],
  }
  _save_json(output_dir / "summary.json", summary)
  _save_comparison(
      output_dir / "06_EXPERIMENT_COMPARISON_GRID.png",
      sample.source,
      sample.target,
      sample.control,
      outputs,
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
  main()

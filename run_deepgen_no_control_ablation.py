"""Run a pure-DeepGen ablation with no Adapter or spatial-control injection."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("DEEPGEN_PROJECT", str(PROJECT_ROOT))
).resolve()
DEFAULT_MODEL_PATH = DEFAULT_PROJECT_ROOT / "models" / "DeepGen-1.0-diffusers"
DEFAULT_CONTROLLED_DIR = (
    DEFAULT_PROJECT_ROOT / "experiments" / "champ_deepgen_formal_v1"
)

sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))
sys.path.insert(0, str(DEFAULT_MODEL_PATH))

from src.integration.champ_deepgen import (
    call_deepgen_without_control,
    image_diagnostics,
    image_error_metrics,
    load_champ_sample,
)


PROMPT = (
    "Change only the woman's pose so that both hands form a heart gesture in front "
    "of her chest. Preserve the exact same identity, face, long dark hair, white "
    "clothing, body proportions, lighting, and grey curtain background."
)
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, broken content, "
    "missing parts, damaged details, artifacts, glitch, noise, extra fingers, "
    "missing fingers, mutated hands, bad composition, wrong proportion, unfinished"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Pure DeepGen no-control ablation")
  parser.add_argument(
      "--data_dir",
      default=str(DEFAULT_PROJECT_ROOT / "inputs" / "champ_sample"),
  )
  parser.add_argument("--controlled_dir", default=str(DEFAULT_CONTROLLED_DIR))
  parser.add_argument(
      "--output_dir",
      default=str(
          DEFAULT_PROJECT_ROOT / "experiments" / "deepgen_no_control_ablation_v1"
      ),
  )
  parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--inference_steps", type=int, default=30)
  parser.add_argument("--guidance_scale", type=float, default=4.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--prompt", default=PROMPT)
  return parser.parse_args()


def _save_json(path: Path, value: object) -> None:
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2),
      encoding="utf-8",
  )


def _banner(
    image: Image.Image,
    title: str,
    color: tuple[int, int, int],
) -> Image.Image:
  image = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
  canvas = Image.new("RGB", (512, 556), color=color)
  canvas.paste(image, (0, 44))
  ImageDraw.Draw(canvas).text((12, 12), title, fill=(255, 255, 255))
  return canvas


def main() -> None:
  args = parse_args()
  if args.inference_steps <= 0:
    raise ValueError("inference_steps must be positive")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the DeepGen ablation")

  sample = load_champ_sample(args.data_dir, resolution=args.resolution)
  controlled_dir = Path(args.controlled_dir)
  controlled_paths = {
      "controlled_step_090": controlled_dir / "05_MODEL_OUTPUT_STEP_090.png",
      "controlled_step_120": controlled_dir / "05_MODEL_OUTPUT_STEP_120.png",
  }
  for path in controlled_paths.values():
    if not path.is_file():
      raise FileNotFoundError(f"controlled comparison image is missing: {path}")
  controlled = {
      name: Image.open(path).convert("RGB").resize(
          sample.target.size,
          Image.Resampling.LANCZOS,
      )
      for name, path in controlled_paths.items()
  }

  model_path = Path(args.model_path)
  if not model_path.is_dir():
    raise FileNotFoundError(f"DeepGen model directory is missing: {model_path}")
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  from diffusers import DiffusionPipeline

  dtype = torch.bfloat16
  torch.manual_seed(args.seed)
  pipe = DiffusionPipeline.from_pretrained(
      str(model_path),
      torch_dtype=dtype,
  ).to("cuda")
  pipe.vae.to("cuda", dtype=dtype)
  pipe.transformer.to("cuda", dtype=dtype)
  pipe._load_extras(attn_implementation="sdpa")
  pipe.transformer.eval().requires_grad_(False)
  pipe.vae.eval().requires_grad_(False)
  if getattr(pipe, "llm", None) is not None:
    pipe.llm.eval().requires_grad_(False)
  if getattr(pipe, "connector_module", None) is not None:
    pipe.connector_module.eval().requires_grad_(False)

  started = time.time()
  with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
    generated = call_deepgen_without_control(
        pipe,
        prompt=args.prompt,
        image=sample.source,
        negative_prompt=NEGATIVE_PROMPT,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    ).images[0]
  duration = time.time() - started

  sample.source.save(output_dir / "01_INPUT_SOURCE_IMAGE.png")
  sample.target.save(output_dir / "02_GROUND_TRUTH_TARGET_IMAGE.png")
  generated.save(output_dir / "03_DEEPGEN_NO_CONTROL.png")

  images = {"deepgen_no_control": generated, **controlled}
  diagnostics = {
      name: {
          **image_diagnostics(image),
          **image_error_metrics(image, sample.target),
      }
      for name, image in images.items()
  }
  config = {
      "data_dir": str(Path(args.data_dir).resolve()),
      "controlled_dir": str(controlled_dir.resolve()),
      "model_path": str(model_path.resolve()),
      "prompt": args.prompt,
      "negative_prompt": NEGATIVE_PROMPT,
      "resolution": args.resolution,
      "inference_steps": args.inference_steps,
      "guidance_scale": args.guidance_scale,
      "seed": args.seed,
      "adapter_created": False,
      "control_arguments_passed": False,
  }
  summary = {
      "duration_seconds": duration,
      "all_outputs_valid": all(
          bool(value["valid"]) for value in diagnostics.values()
      ),
      "best_psnr": max(
          diagnostics,
          key=lambda name: diagnostics[name]["psnr"],
      ),
  }
  _save_json(output_dir / "config.json", config)
  _save_json(output_dir / "diagnostics.json", diagnostics)
  _save_json(output_dir / "summary.json", summary)

  columns = [
      _banner(sample.source, "[1] Source image A", (20, 60, 120)),
      _banner(sample.target, "[2] Ground truth target", (0, 120, 0)),
      _banner(generated, "[3] DeepGen no control", (120, 60, 20)),
      _banner(
          controlled["controlled_step_090"],
          "[4] Controlled step 90",
          (100, 70, 20),
      ),
      _banner(
          controlled["controlled_step_120"],
          "[5] Controlled step 120",
          (100, 60, 120),
      ),
  ]
  grid = Image.new("RGB", (512 * len(columns), 556), color=(20, 20, 20))
  for index, column in enumerate(columns):
    grid.paste(column, (index * 512, 0))
  grid.save(output_dir / "04_ABLATION_COMPARISON_GRID.png")
  print(
      json.dumps(
          {"config": config, "diagnostics": diagnostics, "summary": summary},
          ensure_ascii=False,
          indent=2,
      ),
      flush=True,
  )


if __name__ == "__main__":
  main()

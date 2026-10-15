# DeepGen No-Control Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible pure-DeepGen single-sample output and compare it with the existing controlled Step 90 and Step 120 outputs while changing only control-residual injection.

**Architecture:** Add two small, model-independent helpers to the existing CHAMP/DeepGen integration module: one forbids control kwargs before calling DeepGen, and one computes aligned image-error metrics. A separate GPU runner performs one control-free DeepGen inference, reads the existing controlled images, writes diagnostics, and builds a fixed-order comparison grid.

**Tech Stack:** Python 3, PyTorch, Diffusers/DeepGen, Pillow, NumPy, `unittest`, SSH/SCP.

---

### Task 1: Add Auditable No-Control Invocation and Metrics

**Files:**
- Modify: `src/integration/champ_deepgen.py`
- Modify: `src/integration/__init__.py`
- Test: `tests/test_champ_deepgen_integration.py`

- [ ] **Step 1: Write failing tests for the no-control contract and metrics**

Add the imports and tests below:

```python
import src.integration.champ_deepgen as integration


class NoControlInvocationTests(unittest.TestCase):

  def test_calls_pipeline_without_control_arguments(self):
    received = {}

    def pipeline(**kwargs):
      received.update(kwargs)
      return "result"

    result = integration.call_deepgen_without_control(
        pipeline, prompt="heart pose", image="source", seed=42
    )

    self.assertEqual(result, "result")
    self.assertEqual(received, {
        "prompt": "heart pose", "image": "source", "seed": 42
    })

  def test_rejects_control_arguments(self):
    for name in ("block_controlnet_hidden_states", "control_scale"):
      with self.subTest(name=name), self.assertRaisesRegex(
          ValueError, "pure DeepGen ablation"
      ):
        integration.call_deepgen_without_control(lambda **_: None, **{name: 0})


class ImageErrorMetricTests(unittest.TestCase):

  def test_reports_exact_error_for_black_and_white_images(self):
    black = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))

    result = integration.image_error_metrics(black, white)

    self.assertEqual(result["mae"], 255.0)
    self.assertEqual(result["mse"], 65025.0)
    self.assertEqual(result["psnr"], 0.0)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest -v tests.test_champ_deepgen_integration.NoControlInvocationTests tests.test_champ_deepgen_integration.ImageErrorMetricTests
```

Expected: FAIL because `call_deepgen_without_control` and `image_error_metrics` do not exist.

- [ ] **Step 3: Implement the minimal helpers**

Add to `src/integration/champ_deepgen.py`:

```python
CONTROL_PIPELINE_ARGUMENTS = frozenset({
    "block_controlnet_hidden_states",
    "control_scale",
})


def call_deepgen_without_control(pipeline, **kwargs):
  """Call DeepGen while making control injection structurally impossible."""
  forbidden = sorted(CONTROL_PIPELINE_ARGUMENTS.intersection(kwargs))
  if forbidden:
    raise ValueError(
        "pure DeepGen ablation forbids control arguments: " + ", ".join(forbidden)
    )
  return pipeline(**kwargs)


def image_error_metrics(image: Image.Image, target: Image.Image) -> dict[str, float]:
  """Return full-image MAE, MSE, and PSNR against an aligned RGB target."""
  actual = np.asarray(image.convert("RGB"), dtype=np.float32)
  expected = np.asarray(target.convert("RGB"), dtype=np.float32)
  if actual.shape != expected.shape:
    raise ValueError(
        f"image and target shapes must match, got {actual.shape} and {expected.shape}"
    )
  difference = actual - expected
  mae = float(np.mean(np.abs(difference)))
  mse = float(np.mean(np.square(difference)))
  psnr = float("inf") if mse == 0.0 else float(
      -10.0 * np.log10(mse / (255.0 ** 2))
  )
  return {"mae": mae, "mse": mse, "psnr": psnr}
```

Export both helpers from `src/integration/__init__.py`.

- [ ] **Step 4: Run focused and complete tests and verify GREEN**

Run:

```powershell
python -m unittest -v tests.test_champ_deepgen_integration.NoControlInvocationTests tests.test_champ_deepgen_integration.ImageErrorMetricTests
python -m unittest discover -s tests -v
```

Expected: focused tests pass and the complete suite reports no failures.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/integration/champ_deepgen.py src/integration/__init__.py tests/test_champ_deepgen_integration.py
git commit -m "feat: add auditable DeepGen no-control call"
```

### Task 2: Add the Pure DeepGen Ablation Runner

**Files:**
- Create: `run_deepgen_no_control_ablation.py`
- Test: `tests/test_deepgen_no_control_ablation.py`

- [ ] **Step 1: Write the failing runner-contract tests**

Create `tests/test_deepgen_no_control_ablation.py`:

```python
import unittest
from pathlib import Path


RUNNER = Path("run_deepgen_no_control_ablation.py")


class NoControlRunnerContractTests(unittest.TestCase):

  def test_runner_exists_and_uses_audited_no_control_call(self):
    self.assertTrue(RUNNER.is_file())
    source = RUNNER.read_text(encoding="utf-8")
    self.assertIn("call_deepgen_without_control", source)
    self.assertNotIn("PoseConditionAdapter", source)
    self.assertNotIn("block_controlnet_hidden_states=", source)
    self.assertNotIn("control_scale=", source)

  def test_runner_writes_required_ablation_artifacts(self):
    source = RUNNER.read_text(encoding="utf-8")
    for filename in (
        "03_DEEPGEN_NO_CONTROL.png",
        "04_ABLATION_COMPARISON_GRID.png",
        "config.json",
        "diagnostics.json",
        "summary.json",
    ):
      self.assertIn(filename, source)


if __name__ == "__main__":
  unittest.main()
```

- [ ] **Step 2: Run the runner tests and verify RED**

Run:

```powershell
python -m unittest -v tests.test_deepgen_no_control_ablation
```

Expected: FAIL because `run_deepgen_no_control_ablation.py` is missing.

- [ ] **Step 3: Implement the standalone runner**

Create `run_deepgen_no_control_ablation.py` with these behaviors:

```python
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


def parse_args():
  parser = argparse.ArgumentParser(description="Pure DeepGen no-control ablation")
  parser.add_argument("--data_dir", default=str(DEFAULT_PROJECT_ROOT / "inputs" / "champ_sample"))
  parser.add_argument("--controlled_dir", default=str(DEFAULT_CONTROLLED_DIR))
  parser.add_argument("--output_dir", default=str(DEFAULT_PROJECT_ROOT / "experiments" / "deepgen_no_control_ablation_v1"))
  parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--inference_steps", type=int, default=30)
  parser.add_argument("--guidance_scale", type=float, default=4.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--prompt", default=PROMPT)
  return parser.parse_args()


def save_json(path, value):
  Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def banner(image, title, color):
  image = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
  canvas = Image.new("RGB", (512, 556), color=color)
  canvas.paste(image, (0, 44))
  ImageDraw.Draw(canvas).text((12, 12), title, fill=(255, 255, 255))
  return canvas


def main():
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
      name: Image.open(path).convert("RGB").resize(sample.target.size, Image.Resampling.LANCZOS)
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
  pipe = DiffusionPipeline.from_pretrained(str(model_path), torch_dtype=dtype).to("cuda")
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
      "all_outputs_valid": all(bool(value["valid"]) for value in diagnostics.values()),
      "best_psnr": max(diagnostics, key=lambda name: diagnostics[name]["psnr"]),
  }
  save_json(output_dir / "config.json", config)
  save_json(output_dir / "diagnostics.json", diagnostics)
  save_json(output_dir / "summary.json", summary)

  columns = [
      banner(sample.source, "[1] Source image A", (20, 60, 120)),
      banner(sample.target, "[2] Ground truth target", (0, 120, 0)),
      banner(generated, "[3] DeepGen no control", (120, 60, 20)),
      banner(controlled["controlled_step_090"], "[4] Controlled step 90", (100, 70, 20)),
      banner(controlled["controlled_step_120"], "[5] Controlled step 120", (100, 60, 120)),
  ]
  grid = Image.new("RGB", (512 * len(columns), 556), color=(20, 20, 20))
  for index, column in enumerate(columns):
    grid.paste(column, (index * 512, 0))
  grid.save(output_dir / "04_ABLATION_COMPARISON_GRID.png")
  print(json.dumps({"config": config, "diagnostics": diagnostics, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
```

- [ ] **Step 4: Run the focused and complete tests and verify GREEN**

Run:

```powershell
python -m unittest -v tests.test_deepgen_no_control_ablation
python -m unittest discover -s tests -v
```

Expected: all tests pass without importing Diffusers during test discovery.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- run_deepgen_no_control_ablation.py tests/test_deepgen_no_control_ablation.py
git commit -m "feat: add pure DeepGen ablation runner"
```

### Task 3: Run, Compare, Document, and Verify

**Files:**
- Modify: `docs/champ_deepgen_single_sample_runbook.md`
- Remote output: `/home/shangguanrz/project/pic-edit/experiments/deepgen_no_control_ablation_v1/`
- Local output: `results/deepgen_no_control_ablation_v1/`

- [ ] **Step 1: Sync code and run remote tests**

```powershell
scp src/integration/champ_deepgen.py src/integration/__init__.py sg:/home/shangguanrz/project/pic-edit/src/integration/
scp run_deepgen_no_control_ablation.py sg:/home/shangguanrz/project/pic-edit/
scp tests/test_champ_deepgen_integration.py tests/test_deepgen_no_control_ablation.py sg:/home/shangguanrz/project/pic-edit/tests/
ssh sg "cd /home/shangguanrz/project/pic-edit && /home/shangguanrz/miniconda3/envs/deepgen/bin/python -m unittest -v tests.test_champ_deepgen_integration tests.test_deepgen_no_control_ablation"
```

Expected: all focused remote tests pass.

- [ ] **Step 2: Run the fixed-seed no-control inference**

```bash
cd /home/shangguanrz/project/pic-edit
export DEEPGEN_PROJECT=/home/shangguanrz/project/pic-edit
/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  run_deepgen_no_control_ablation.py \
  --data_dir inputs/champ_sample \
  --controlled_dir experiments/champ_deepgen_formal_v1 \
  --output_dir experiments/deepgen_no_control_ablation_v1 \
  --inference_steps 30 \
  --guidance_scale 4.5 \
  --seed 42
```

Expected: exit 0 with `adapter_created: false`, `control_arguments_passed: false`, and all required PNG/JSON files.

- [ ] **Step 3: Copy review artifacts back and inspect them**

```powershell
$artifactDir = 'D:\codeplus\pictureedit\results\deepgen_no_control_ablation_v1'
New-Item -ItemType Directory -Force -Path $artifactDir
scp 'sg:/home/shangguanrz/project/pic-edit/experiments/deepgen_no_control_ablation_v1/*.png' $artifactDir
scp 'sg:/home/shangguanrz/project/pic-edit/experiments/deepgen_no_control_ablation_v1/*.json' $artifactDir
```

Expected: the comparison grid has five columns in the specified order and the JSON values are finite.

- [ ] **Step 4: Add the reproducible ablation command to the runbook**

Append a `DeepGen 无控制消融` section containing the exact command from Step 2, the fixed variables, and the output file meanings.

- [ ] **Step 5: Run final verification**

```powershell
python -m unittest discover -s tests -v
python -m vram_lab.acceptance --root results\review_20260908_v2
git diff --check
```

Expected: complete tests pass, historical acceptance reports `111/111 PASS`, and `git diff --check` is clean.

- [ ] **Step 6: Commit documentation**

```powershell
git add -- docs/champ_deepgen_single_sample_runbook.md
git commit -m "docs: add DeepGen no-control ablation workflow"
```

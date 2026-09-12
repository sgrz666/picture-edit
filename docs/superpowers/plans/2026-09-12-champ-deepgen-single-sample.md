# CHAMP–DeepGen Single-Sample Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CHAMP single-sample experiment's runtime `transformer.forward` monkeypatch with a tested DeepGen Pipeline control interface and produce a reproducible, non-degenerate single-sample output from image A, text, Normal, Pose, Depth, and Semantic controls.

**Architecture:** A tracked CHAMP integration module validates and assembles the 8-channel control tensor, then aligns PoseConditionAdapter residuals to DeepGen target/reference tokens and CFG batches. A reproducible patch exposes those residuals through the outer DeepGen Pipeline and inner `_SD3Pipeline`; the experiment runner uses the same alignment code for training and inference.

**Tech Stack:** Python 3.12, PyTorch, Pillow, OpenCV, NumPy, Diffusers/DeepGen, `unittest`, RTX 6000D remote execution.

---

## File Map

- Create `src/integration/__init__.py`: public integration exports.
- Create `src/integration/champ_deepgen.py`: CHAMP sample loading, validation, residual alignment, and image diagnostics.
- Create `tests/test_champ_deepgen_integration.py`: unit tests for sample and tensor contracts.
- Create `patches/deepgen_pose_control.patch`: reproducible patch for the untracked DeepGen vendor source.
- Create `tests/test_deepgen_pose_patch.py`: text-level patch contract and idempotency checks.
- Modify `run_champ_single_overfit.py`: use the shared integration module and formal Pipeline interface; remove the forward monkeypatch.
- Create `docs/champ_deepgen_single_sample_runbook.md`: exact remote commands and artifact interpretation.

### Task 1: CHAMP Sample Contract

**Files:**
- Create: `tests/test_champ_deepgen_integration.py`
- Create: `src/integration/__init__.py`
- Create: `src/integration/champ_deepgen.py`

- [ ] **Step 1: Write failing tests for a valid RGB target and fixed channel order**

```python
class ChampSampleTests(unittest.TestCase):
    def test_loads_eight_channels_in_normal_depth_pose_semantic_order(self):
        sample = load_champ_sample(self.root, resolution=16)
        self.assertEqual(tuple(sample.control.shape), (1, 8, 16, 16))
        self.assertAlmostEqual(float(sample.control[0, 0].mean()), 10 / 255)
        self.assertAlmostEqual(float(sample.control[0, 3].mean()), 40 / 255)
        self.assertAlmostEqual(float(sample.control[0, 4].mean()), 50 / 255)
        self.assertAlmostEqual(float(sample.control[0, 7].mean()), 80 / 255)

    def test_rejects_binary_mask_as_target_rgb(self):
        write_binary_rgb(self.root / "real_tgt_frame_60.png")
        with self.assertRaisesRegex(ValueError, "target RGB looks like a mask"):
            load_champ_sample(self.root, resolution=16)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m unittest tests.test_champ_deepgen_integration.ChampSampleTests -v`

Expected: import failure because `src.integration.champ_deepgen` does not exist.

- [ ] **Step 3: Implement the minimal loader and validation**

```python
@dataclass(frozen=True)
class ChampSample:
    source: Image.Image
    target: Image.Image
    control: torch.Tensor
    paths: dict[str, Path]

def load_champ_sample(root: Path | str, resolution: int) -> ChampSample:
    paths = resolve_champ_paths(Path(root))
    source = Image.open(paths["source"]).convert("RGB")
    target = Image.open(paths["target"]).convert("RGB")
    validate_target_rgb(target)
    normal = load_rgb(paths["normal"], resolution)
    depth = load_gray(paths["depth"], resolution)
    pose = load_rgb(paths["pose"], resolution)
    semantic = load_gray(paths["semantic"], resolution)
    stacked = np.concatenate([normal, depth, pose, semantic], axis=-1)
    control = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
    return ChampSample(resize_rgb(source, resolution), resize_rgb(target, resolution), control, paths)
```

`validate_target_rgb` must require three non-identical color channels, at least 32 unique luminance values, and a non-binary luminance distribution. Missing files must raise `FileNotFoundError` with the exact path.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python -m unittest tests.test_champ_deepgen_integration.ChampSampleTests -v`

Expected: all `ChampSampleTests` pass.

- [ ] **Step 5: Commit the loader**

```bash
git add src/integration tests/test_champ_deepgen_integration.py
git commit -m "feat: validate CHAMP single-sample controls"
```

### Task 2: Adapter Residual Token Alignment

**Files:**
- Modify: `tests/test_champ_deepgen_integration.py`
- Modify: `src/integration/champ_deepgen.py`

- [ ] **Step 1: Write failing tests for reference-token padding and strict batch validation**

```python
class ResidualAlignmentTests(unittest.TestCase):
    def test_pads_only_reference_tokens(self):
        residuals = [torch.ones(1, 4, 3)] * 2
        aligned = align_control_residuals(
            residuals, target_tokens=4, reference_tokens=4, batch_size=1,
        )
        self.assertEqual(tuple(aligned[0].shape), (1, 8, 3))
        torch.testing.assert_close(aligned[0][:, :4], torch.ones(1, 4, 3))
        torch.testing.assert_close(aligned[0][:, 4:], torch.zeros(1, 4, 3))

    def test_rejects_wrong_batch(self):
        with self.assertRaisesRegex(ValueError, "expected batch 1"):
            align_control_residuals([torch.ones(2, 4, 3)], 4, 4, 1)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m unittest tests.test_champ_deepgen_integration.ResidualAlignmentTests -v`

Expected: failure because `align_control_residuals` is missing.

- [ ] **Step 3: Implement strict residual alignment**

```python
def align_control_residuals(
    residuals, target_tokens, reference_tokens, batch_size,
):
    aligned = []
    for index, residual in enumerate(residuals):
        if residual.ndim != 3 or residual.shape[1] != target_tokens:
            raise ValueError(f"control layer {index}: expected [B,{target_tokens},D], got {tuple(residual.shape)}")
        if residual.shape[0] != batch_size:
            raise ValueError(f"control layer {index}: expected batch {batch_size}, got {residual.shape[0]}")
        value = F.pad(residual, (0, 0, 0, reference_tokens))
        aligned.append(value)
    return aligned
```

- [ ] **Step 4: Run all integration tests and confirm GREEN**

Run: `python -m unittest tests.test_champ_deepgen_integration -v`

Expected: loader and residual tests pass.

- [ ] **Step 5: Commit residual alignment**

```bash
git add src/integration/champ_deepgen.py tests/test_champ_deepgen_integration.py
git commit -m "feat: align pose residuals for DeepGen CFG"
```

### Task 3: Reproducible DeepGen Pipeline Patch

**Files:**
- Create: `patches/deepgen_pose_control.patch`
- Create: `tests/test_deepgen_pose_patch.py`

- [ ] **Step 1: Write a failing patch-contract test**

```python
class DeepGenPatchTests(unittest.TestCase):
    def test_patch_exposes_and_forwards_control_arguments(self):
        patch = Path("patches/deepgen_pose_control.patch").read_text(encoding="utf-8")
        self.assertGreaterEqual(patch.count("block_controlnet_hidden_states"), 5)
        self.assertIn("control_scale", patch)
        self.assertIn("block_controlnet_hidden_states=control_states", patch)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python -m unittest tests.test_deepgen_pose_patch -v`

Expected: `FileNotFoundError` for the patch file.

- [ ] **Step 3: Create the vendor patch**

The patch must make these exact behavioral changes in `models/DeepGen-1.0-diffusers/deepgen_pipeline.py`:

```python
# _SD3Pipeline.__call__ additions
block_controlnet_hidden_states=None,
control_scale: float = 1.0,

control_states = None
if block_controlnet_hidden_states is not None:
    control_states = []
    for index, state in enumerate(block_controlnet_hidden_states):
        state = state.to(device=device, dtype=prompt_embeds.dtype) * float(control_scale)
        if state.shape[0] == batch_size and self.do_classifier_free_guidance:
            state = torch.cat([state, state], dim=0)
        if state.shape[0] != latent_model_input.shape[0]:
            raise ValueError(
                f"control layer {index}: batch {state.shape[0]} does not match denoising batch {latent_model_input.shape[0]}"
            )
        control_states.append(state)

noise_pred = self.transformer(
    hidden_states=latent_model_input,
    cond_hidden_states=cond_latents,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    joint_attention_kwargs=self.joint_attention_kwargs,
    block_controlnet_hidden_states=control_states,
    return_dict=False,
)[0]
```

The outer DeepGen `__call__` must expose both parameters and pass them into `_SD3Pipeline(...)` without reading files or invoking the Adapter.

- [ ] **Step 4: Run patch-contract and full local tests**

Run: `python -m unittest tests.test_deepgen_pose_patch -v`

Expected: patch contract passes.

Run: `python -m unittest discover -s tests -v`

Expected: all prior 17 tests plus new tests pass.

- [ ] **Step 5: Commit the patch artifact**

```bash
git add patches/deepgen_pose_control.patch tests/test_deepgen_pose_patch.py
git commit -m "feat: expose DeepGen pose-control residuals"
```

### Task 4: Refactor the Single-Sample Runner

**Files:**
- Modify: `tests/test_champ_deepgen_integration.py`
- Modify: `run_champ_single_overfit.py`

- [ ] **Step 1: Write a failing source-contract test that forbids monkeypatching**

```python
def test_runner_uses_formal_pipeline_control_interface(self):
    source = Path("run_champ_single_overfit.py").read_text(encoding="utf-8")
    self.assertNotIn("transformer.forward =", source)
    self.assertIn("block_controlnet_hidden_states=", source)
    self.assertIn("load_champ_sample", source)
    self.assertIn("align_control_residuals", source)
```

- [ ] **Step 2: Run the contract test and confirm RED**

Run: `python -m unittest tests.test_champ_deepgen_integration.RunnerContractTests -v`

Expected: failure because the runner still assigns `transformer.forward`.

- [ ] **Step 3: Replace ad-hoc loading and monkeypatch inference**

The runner must load the sample through `load_champ_sample`, compute Adapter residuals, align only target/reference tokens, and call:

```python
out_img = pipe(
    prompt=args.prompt,
    image=sample.source,
    negative_prompt=NEGATIVE,
    height=args.resolution,
    width=args.resolution,
    num_inference_steps=args.inference_steps,
    guidance_scale=args.guidance_scale,
    seed=args.seed,
    block_controlnet_hidden_states=aligned_residuals,
    control_scale=args.control_scale,
).images[0]
```

Add CLI arguments with fixed defaults:

```text
--max_steps 120
--learning_rate 5e-5
--save_every 30
--inference_steps 30
--guidance_scale 4.5
--control_scale 1.0
```

Save `diagnostics.json` with step, loss, gradient norm, residual RMS/max, image mean/std, near-white ratio, and validity flag. Do not label a run successful when the output validity flag is false.

- [ ] **Step 4: Run contract and full tests**

Run: `python -m unittest tests.test_champ_deepgen_integration.RunnerContractTests -v`

Expected: contract passes.

Run: `python -m unittest discover -s tests -v`

Expected: entire suite passes.

- [ ] **Step 5: Commit the runner**

```bash
git add run_champ_single_overfit.py src/integration tests/test_champ_deepgen_integration.py
git commit -m "feat: run CHAMP overfit through DeepGen control API"
```

### Task 5: Local Regression and Artifact Checks

**Files:**
- Create: `docs/champ_deepgen_single_sample_runbook.md`

- [ ] **Step 1: Run the complete local suite**

Run: `python -m unittest discover -s tests -v`

Expected: zero failures and zero errors.

- [ ] **Step 2: Run historical experiment acceptance**

Run: `python -m vram_lab.acceptance --root results/review_20260908_v2`

Expected: `passed: true`, `checked_primary_runs: 111`, `PASS: 111`.

- [ ] **Step 3: Validate the real CHAMP sample without loading the model**

Run:

```bash
python run_champ_single_overfit.py \
  --data_dir inputs/champ_sample \
  --validate_only
```

Expected: source/target are RGB, all controls exist, control shape is `[1,8,512,512]`, and target is not mask-like.

- [ ] **Step 4: Write the exact runbook**

Document the local validation command, remote patch command, remote training command, output files, diagnostic thresholds, and restore procedure for the backed-up vendor Pipeline.

- [ ] **Step 5: Commit the runbook**

```bash
git add docs/champ_deepgen_single_sample_runbook.md
git commit -m "docs: add CHAMP DeepGen single-sample runbook"
```

### Task 6: Apply and Verify the Pipeline on `sg`

**Files:**
- Remote modify: `/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/deepgen_pipeline.py`
- Remote create: a timestamped backup beside the vendor file.

- [ ] **Step 1: Verify exact remote target and available GPU memory**

Run:

```bash
ssh sg 'sha256sum /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/deepgen_pipeline.py'
ssh sg 'nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader'
```

Expected: the known DeepGen source is present; defer the heavy run if other users leave insufficient free VRAM.

- [ ] **Step 2: Back up and apply the tracked patch once**

Run from the local workspace:

```powershell
scp patches/deepgen_pose_control.patch sg:/home/shangguanrz/project/pic-edit/patches/deepgen_pose_control.patch
ssh sg "cd /home/shangguanrz/project/pic-edit && test ! -e models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control && cp -p models/DeepGen-1.0-diffusers/deepgen_pipeline.py models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control"
ssh sg "cd /home/shangguanrz/project/pic-edit && patch --dry-run -p1 < patches/deepgen_pose_control.patch"
ssh sg "cd /home/shangguanrz/project/pic-edit && patch -p1 < patches/deepgen_pose_control.patch"
```

Expected: backup creation succeeds, dry run reports every hunk applicable, and the real patch applies once. If the backup already exists or a hunk fails, stop and inspect the remote source instead of overwriting it.

- [ ] **Step 3: Verify the new signatures without loading model weights**

Run on `sg`:

```bash
cd /home/shangguanrz/project/pic-edit
PYTHONPATH=models/DeepGen-1.0-diffusers \
  /home/shangguanrz/miniconda3/envs/deepgen/bin/python -B - <<'PY'
import inspect
import deepgen_pipeline

inner = inspect.signature(deepgen_pipeline._SD3Pipeline.__call__).parameters
outer = inspect.signature(deepgen_pipeline.DeepGenPipeline.__call__).parameters
for name, params in (("inner", inner), ("outer", outer)):
    assert "block_controlnet_hidden_states" in params, name
    assert "control_scale" in params, name
print("DeepGen pose-control signatures: OK")
PY
```

Expected: both assertions pass; no GPU allocation is created.

- [ ] **Step 4: Sync tracked integration code and validate the remote sample**

Run from the local workspace:

```powershell
scp -r src/integration sg:/home/shangguanrz/project/pic-edit/src/
scp run_champ_single_overfit.py sg:/home/shangguanrz/project/pic-edit/run_champ_single_overfit.py
scp tests/test_champ_deepgen_integration.py tests/test_deepgen_pose_patch.py sg:/home/shangguanrz/project/pic-edit/tests/
scp docs/champ_deepgen_single_sample_runbook.md sg:/home/shangguanrz/project/pic-edit/docs/champ_deepgen_single_sample_runbook.md
ssh sg "cd /home/shangguanrz/project/pic-edit && /home/shangguanrz/miniconda3/envs/deepgen/bin/python run_champ_single_overfit.py --data_dir inputs/champ_sample --validate_only"
```

Expected: remote validation passes before model loading.

### Task 7: Run and Review the Real Single-Sample Experiment

**Files:**
- Remote output: `/home/shangguanrz/project/pic-edit/experiments/champ_deepgen_formal_v1/`
- Local review copy: `results/champ_deepgen_formal_v1/`

- [ ] **Step 1: Start the fixed-seed remote run**

Run:

```bash
ssh sg 'cd /home/shangguanrz/project/pic-edit && \
  /home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  run_champ_single_overfit.py \
  --data_dir inputs/champ_sample \
  --output_dir experiments/champ_deepgen_formal_v1 \
  --max_steps 120 \
  --learning_rate 5e-5 \
  --save_every 30 \
  --seed 42'
```

Expected: checkpoints and diagnostics at steps 0, 30, 60, 90, and 120; no OOM or NaN.

- [ ] **Step 2: Check numerical validity before visual review**

Reject outputs with NaN/Inf, image standard deviation below `5.0`, or near-white/near-black ratio above `0.95`. Confirm the last-20-step median loss is lower than the first-20-step median loss.

- [ ] **Step 3: Copy only review artifacts locally**

Copy comparison PNGs, prompt/config, loss history, diagnostics, and environment metadata. Do not copy the large checkpoint unless requested.

- [ ] **Step 4: Visually inspect source identity, clothing/background, and target pose**

Compare Step 0, every valid checkpoint, and Ground Truth. Select the best valid checkpoint and record why it was selected; do not assume the final step is best.

- [ ] **Step 5: Run final verification**

Run:

```bash
python -m unittest discover -s tests -v
python -m vram_lab.acceptance --root results/review_20260908_v2
git status --short --branch
```

Expected: tests and historical acceptance pass; Git status contains only intended commits plus pre-existing user files.

- [ ] **Step 6: Report completion and usage**

Report the Pipeline interface changes, remote command, numerical diagnostics, best output, known limitations, and clickable local artifact paths. Explain how to rerun with another CHAMP-aligned sample.

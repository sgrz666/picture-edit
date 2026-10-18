# Qwen 双人逐样本动作提示词训练实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Build four real Hi4D training pairs with per-sample action instructions, train matched short/structured adapter variants, and evaluate one untouched test-split pair with no backpropagation.

**Architecture:** Keep dual_v8 and native Qwen unchanged. New prompt/data preparation builds strict generation and supervision manifests; a new multi-sample trainer reuses the existing flow objective; a separate frozen-test runner enforces inference-only execution and records weight integrity.

**Tech Stack:** Existing remote Qwen PyTorch/Diffusers environment, SMPL-X and nvdiffrast, Hi4D/DWPose, pytest.

## Task 1: Assets, split and source-only instructions

- [x] Inspect authoritative split.json and source/target assets; select four train rows, one val row and one test row with disjoint pair identities.
- [x] Add `src/data/hi4d_dual_prompt_pilot.py` and `tests/test_hi4d_dual_prompt_pilot.py`: validate split membership and duplicate identity/frame leakage, build deterministic short/structured prompts from source anchors/action/geometry only.
- [x] Tests must reject test rows passed into training, swapped UID attribution, GT RGB/mask fields in prompt inputs and low-confidence invented hand actions.
- [x] Add `scripts/precompute/prepare_hi4d_dual_prompt_pilot.py`; generalize existing fit/control/export logic into new helpers without changing the historical two-frame implementation. Fit selected SMPL-X frames, preserve source identity beta, render geometry and encode source-only appearance. Record fit quality and all source/target provenance.
- [x] Materialize real train/val/test generation manifests and separate supervision manifests, both prompt variants and inspection boards.

## Task 2: Matched multi-sample training

- [x] Add `src/pose_control/qwen21_full/dual/prompt_pilot.py`, `scripts/train/train_qwen21_dual_prompt_pilot.py`, and `tests/test_qwen21_dual_prompt_pilot.py`.
- [x] Tests cover deterministic identical sample/noise schedules across variants; strict train/val pair isolation; validation cannot call backward or optimizer; fresh warm start; invalid stale completion/checkpoint rejection.
- [x] Cache per-source-and-prompt text/image embeddings and target latents separately. Train independent short/structured variants with four varied microbatches each optimizer step, 300 steps, identical initialization/noise/order/loss. Freeze native components and imported geometry, record hashes, finite gradients and metrics.
- [x] Every50 steps compute no-grad validation losses on the true validation row and save last/best; record checkpoint selection and completion. Never read or optimize test rows here.

## Task 3: Frozen true test sample

- [x] Add `scripts/eval/eval_qwen21_dual_prompt_pilot.py` and `tests/test_qwen21_dual_prompt_pilot_eval.py`.
- [x] Tests deliberately attempt Tensor.backward, autograd.backward/grad and optimizer.step inside the inference guard and require rejection; verify state restoration, disabled gradients and unchanged real tensor SHA.
- [x] Only accept selected best checkpoints with complete training records and matching training source-code closure; select the locked pair00 taichi00/41→hug00/108 camera4 test row absent from training/validation pair IDs. Generate native short/structured and trained adapter short/structured, 40steps, seeds42/43/44. Label explicit_target_geometry_control separately from geometry-derived text descriptions.
- [x] Snapshot native/adapter weights and gradients before/after all test generation, record no-backprop guard and exact prompts/data/checkpoint hashes. Load target RGB only after generating; preserve output/image provenance.
- [x] Reuse existing generated-RGB metrics where supported, record missing actors/contact reconstruction rather than replace them with control meshes. Save all12 images and source/GT comparison board.

## Task 4: Reviews, actual execution and delivery

- [x] Apply spec review then quality review to owned new files, fix concrete findings, run focused regressions in remote environment.
- [x] Create isolated remote runtime from previous actual mirror; copy only reviewed new files. Execute data preparation, both300-step runs and final inference-only test.
- [x] Inspect actual completion records, manifests, checkpoint hashes, training metrics and test guard evidence against user requirements.
- [x] Write `reports/Qwen21_dual_prompt_pilot_20261013.md` and `docs/experiments/2026-10-13-qwen21-dual-prompt-pilot.md` with tested commands, output locations and honest quality conclusions. Preserve historical outputs and unrelated working changes.

## Execution evidence — 2026-10-14

Both final branches completed 300 optimizer steps; true validation selected step50 in each. One locked test pair was evaluated once in four groups × seeds42/43/44 (12 images), with zero backpropagation/optimizer attempts and unchanged native/adapter parameter hashes. Original SSH interruption and failed exact continuation remain archived; structured was freshly retrained from its initial state, and original short300 retained. Original interrupted-process final native memory hash was not observed. Evaluation excludes letterbox padding from background scores. Results do not establish a consistent structured-prompt adapter improvement; GT DWPose calibration detects one actor under heavy occlusion. Report and manual include artifact hashes, denominators, reproduction and read-only inspection commands.

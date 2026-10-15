# TikTok 00001 Condition Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the TikTok `0014 -> 0074` pose and part conditions, compare fixed-seed inference, then run a 1500-step single-pair face overfit.

**Architecture:** A tested SMPLest-X projection helper produces OpenPose-25 points. A single-frame override preserves original caches. Training and inference consume the same override, while part-map rendering remains a separate ablation.

**Tech Stack:** PyTorch, NumPy, OpenCV/Pillow, existing DeepGen pipeline, optional nvdiffrast.

---

### Task 1: Correct body-keypoint projection

**Files:** `scripts/build_tiktok_v6_conditions.py`, `tests/test_tiktok_data_contract.py`.

- [ ] Add a synthetic test with 137 distinct projected points and known camera metadata. Assert OpenPose nose=SMPLest index 24, pelvis=0, wrists=13/12, and out-of-frame confidence=0.
- [ ] Run `pytest -q tests/test_tiktok_data_contract.py -k body25_projection` and confirm the new test fails because the helper is absent.
- [ ] Implement `project_smplestx_body25_to_openpose(params, frame_index)` using the saved 12x16 crop transform and explicit mapping `[24,7,9,11,13,8,10,12,0,2,4,6,1,3,5,23,22,21,20,14,15,16,17,18,19]`.
- [ ] Run the targeted test and the full TikTok data-contract test file.

### Task 2: Preserve original assets and build per-frame overrides

**Files:** `scripts/build_tiktok_v6_conditions.py`, `src/data/tiktok_dataset.py`, `train_tiktok_v65_face.py`, `scripts/run_tiktok_sample_00001_inference.py`, `tests/test_tiktok_data_contract.py`.

- [ ] Add a failing test that loads an override for target `0074`, changes only its pose heatmap/optional part one-hot, and rejects a mismatched sequence or target stem.
- [ ] Implement an isolated override writer and loader, with explicit CLI paths and no writes to `v6_conditions.pt`.
- [ ] Render 25-point overlay and check that visible shoulders, wrists, pelvis and legs align with RGB. Record old/new point ranges and valid counts.
- [ ] Run the targeted and full tests again.

### Task 3: A/B inference and single-pair training

**Files:** new files only under `experiments/tiktok_00001_condition_repair/` and a one-frame part PNG if rendering succeeds.

- [ ] Run the existing baseline and corrected-pose Face OFF/ON with seed 42, CFG 4, 28 steps; keep independent outputs.
- [ ] If nvdiffrast is available, render one part map and repeat with corrected pose plus part. Verify nonzero classes and overlay against target mask.
- [ ] Train only the corrected pair for 1500 steps, warmup 1000; write a new checkpoint directory.
- [ ] Inspect full inference from the trained checkpoint against the same baseline. Report pose, face, and hand observations without interpreting a low-noise one-step reconstruction as full inference.

No commit is planned because the TikTok files are currently untracked user work in both local and server repositories. Only the specified files will be copied to the server after source-hash verification.

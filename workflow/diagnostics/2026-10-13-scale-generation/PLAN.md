# Source-anchor generation experiment plan

> Execute sequentially with superpowers:executing-plans. The user approved generation following the preceding source-scale/anchor diagnostic. No edits to the production workflow or previous runs, and no commits in this shared checkout.

**Goal:** Generate the four bad cases and medium_02 as a control using A-measured scale and pelvis anchoring, then compare against the saved baseline.

**Architecture:** A diagnostic-only helper rescales the saved B guide using matching shoulder-midpoint to hip-midpoint definitions in A and B. A and its original preprocessing are copied unchanged, and the baseline generation prompt is frozen byte-for-byte. The existing native generate_worker.py executes the recorded baseline settings in isolated run directories. This is a 2D geometry ablation, not a complete 3D body-shape retargeter.

**Output:** workflow/runs/scale_anchor_20261013/<case>/; experiment.json distinguishes the ablation from a normal end-to-end PE workflow.

- [x] Write failing tests for matching torso semantics, preserving relative pose and NaNs, allowing off-canvas joints, rejecting unreliable anchors, and raster-clipping crossing bone segments.
- [x] Implement geometry_ablation.py without modifying pose_transfer/geometry.py, and verify tests on sg.
- [x] Implement run_experiment.py: freeze original inputs/prompt/settings, validate baseline hashes and model identity, record operation and code hashes, prepare new guide plus actual native generation requests. Use the workflow GPU lock, stage hash validation, and per-case failure records.
- [x] Run hard_02 first; inspect the generated image. Then generate hard_01, hard_04, hard_05 and medium_02 using the same settings and correction.
- [x] Measure source/baseline/new output with the same CPU DWPose model; compare image scale, visible action, cropping and duplicate-person artifacts. Record confidence and estimation limitations.
- [x] Save comparison figures, experiment metrics and actual findings in README.md. Verify completed generation artifact hashes and exact prompt/setting retention. Provide commands and result paths.


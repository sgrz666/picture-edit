# BFS Body/Face joint pilot execution

> Historical fixed-512 plan, stopped by the user. The current original-pixel-dimension requirement and implementation handoff are in `../../experiments/2026-10-08-bfs-body-face-joint-handoff.md`.

Approved design: `../specs/2026-10-08-bfs-body-face-joint-pilot-design.md`.

1. Lock qualified 512 pairs and held-out 24 TikTok / 12 iPER cases. Completed; `data/lock.json` records hashes and monitor indices.
2. Implement joint loss, strict frozen BFS conversion, inference exports and portable report. Review code; run meaningful CPU tests.
3. Run four low-noise real-model gradient probes and one high-noise Body probe per base. Freeze Qwen, LoRA and unrelated Adapter branches. Require matching passed preflight before training.
4. Enable deterministic CUDA reductions where supported. Check exact Adapter/optimizer/RNG restoration, then compare a four-step direct trajectory against continuation from its exact step-two checkpoint. Two extra continuation repeats measure remaining numerical variation. The successful check uses ten private updates; retain the earlier eight-update diagnostic attempt separately. Private updates do not belong to the formal 1,024-update trajectory.
5. Save initial 36 outputs. Train J to 128; save 12 monitor outputs and inspect for new severe failures. Continue to 256 only after quality review; save 36 outputs plus four Body-only and four Face-only diagnostics.
6. From the same J256 weights, separately run Qwen / BFS Body V1.0 / BFS Head V1.1 with fresh matching optimizers. Each arm: preflight, 12 hotload outputs, 128 updates, 12 monitor outputs and review, then 256 and 36 final outputs.
7. Score unique-face identity to A and generated pose to original transformed Body25 B. Keep failed samples. Compare paired outcomes by video / actor group; do not infer independent identities from adjacent frames.
8. Copy all raw images, crops, panels, records, locks, metrics and reports to `D:/codeplus/pictureedit/experiments/bfs_body_face_joint_pilot_20261008`. Inspect local links/images. Document actual completed budget, results and limitations.

Remote entry point: `scripts/run/run_qwen21_joint_pilot.py --phase PHASE`. Phases are explicit, sequential, and stop on errors. Mid-stage continuation requires a written image review of all 12 fixed monitor cases. No old run is overwritten.

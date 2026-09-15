# SMPL-X Condition Injection V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` and `superpowers:test-driven-development` to
> implement this plan task-by-task.

**Goal:** Separate prepared SMPL-X condition encoding into a stable
`ConditionBundle` consumed by the existing DeepGen V6 Adapter.

**Architecture:** Independent spatial stems encode Normal, 25-channel Pose and
14-channel Part maps, while small MLPs encode global and relative parameters.
Typed contact inputs and task IDs are encoded but not deeply reasoned over in
this layer. A projection bridge preserves the existing V6 experts and control
core.

**Tech Stack:** Python 3.12, PyTorch 2.8, Diffusers 0.35.2, pytest.

---

- [x] Specify `ConditionBundle`, contact and identity contracts with failing tests.
- [x] Implement native stems, role FiLM, global/relative/contact/task encoders.
- [x] Add `ConditionBundleBridge` and move raw-condition access out of experts.
- [x] Preserve source appearance, deterministic routing and zero residual heads.
- [x] Add the fixed-commit, MIT-licensed CHAMP single-frame optional backend.
- [ ] Update smoke commands and architecture documentation.
- [ ] Run unit, full-suite and real DeepGen smoke verification.
- [ ] Create final tag and atomically push `main` plus rollback tags.

Rollback baseline: `rollback/adapter-v6-condition-injection-v1-pre` at merged
main commit `9804541b107a02f3b30be38a57db5123ebd5ff92`.

# DeepGen V6.3 Dynamic Control Interface V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use TDD and atomic commits.

**Goal:** Preserve the context-aware recurrent DeepGen control core while splitting geometry and interaction residuals, adding per-branch strength schedules, and injecting controls at every denoising step without modifying the vendored DeepGen runtime.

**Architecture:** Condition injection, SMPL-X reasoning, appearance encoding, and bridge projection are static per image. Each denoising step runs geometry and, for dual-person rows only, interaction control branches against the current noisy latent, text, pooled text, and timestep. Six independently zero-initialized residual groups are scaled and passed through DeepGen's native `block_controlnet_hidden_states` interface.

**Tech Stack:** PyTorch, Diffusers 0.35.2, DeepGen SD3 DiT, pytest, CUDA/bfloat16.

---

## Tasks

1. Specify `PreparedControlConditioning`, branch residual, output, schedule, dynamic shape, and zero-initialization contracts in failing tests.
2. Split the reasoner bridge into independent geometry and interaction scene projections while preserving both high-resolution gradient paths.
3. Upgrade `SharedRecurrentControlCore` to shared-block, branch-specific dynamic states with twelve independent zero heads.
4. Add `DeepGenControlInterface`, strength scheduling, diagnostics, and target/source/padding token alignment.
5. Route `UnifiedSMPLXAdapterV6` through static preparation plus per-step dynamic control and preserve appearance/role binding.
6. Add a hook-based `ControlledDeepGenPipeline` wrapper that restores hooks on all exits and rejects nested or duplicate injection.
7. Upgrade config/docs/smoke tests, run full CPU/GPU verification, tag, and atomically publish.

## Locked behavior

- Geometry strength defaults to 1.0; interaction strength defaults to 0.8.
- Constant scheduling is the default; linear and cosine windows are supported.
- Static conditions may be cached for inference. Dynamic residuals are never cached across timesteps.
- DeepGen, VAE, VLM, and connector stay frozen, but gradients through the frozen transformer to Adapter residuals remain valid.
- Native grouped injection remains six groups. Active block ranges are 0-3, 4-7, 8-11, 12-15, 16-19, and 20-22; block 23 is `context_pre_only` and is not injected.
- No optimizer, training loop, checkpoint writer, external weights, exact-block injection, reset, or force push is introduced.

## Verification

Run the full pytest suite, dynamic interface smoke for single/dual/mixed at 512, and real DeepGen zero-equivalence checks. Initial controls must be exactly zero and DeepGen output must match the no-Adapter path within `1e-6`.

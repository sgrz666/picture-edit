# DeepGen Unified SMPL-X Adapter V6.3

V6.3 keeps the context-aware recurrent control block introduced by the V6
architecture and makes the DeepGen boundary explicit:

```text
SMPL-X maps/parameters
    -> SMPLXConditionInjector
    -> ConditionBundle
    -> SMPLXAdapterReasoner
    -> InternalControlState
    -> independent Geometry / Interaction bridge
    -> PreparedControlConditioning (static per image)
    -> dynamic dual-branch SharedRecurrentControlCore (per denoising step)
    -> ControlStrengthController
    -> six native grouped DeepGen residuals
```

No body estimator, renderer, dataset preprocessing, optimizer, training loop,
checkpoint writer, external weight download, QKV injection, or exact-block
Transformer modification is part of this phase.

## Static conditioning

`prepare_conditioning()` runs condition injection, SMPL-X reasoning, the
high/low-resolution bridge, and identity binding once per image. Geometry and
interaction stay separate:

```python
prepared.geometry_condition      # [B, 16 + 128, H_latent, W_latent]
prepared.interaction_condition   # [B, 16 + 128, H_latent, W_latent]
prepared.adapter_tokens          # bound geometry + appearance tokens
prepared.adapter_token_mask      # valid compact context tokens
```

Geometry concatenates the source-scene latent with the fused geometry feature.
Interaction concatenates a zero 16-channel prefix with the interaction feature,
so source-scene information is not counted twice. Single-person interaction
conditions are exactly zero. Both high-resolution Reasoner paths are learned-
downsampled and remain connected to the final loss graph.

Each person's masked spatial tokens are pooled to four geometry summaries and
bound to eight appearance tokens through slot and source-index embeddings.
Appearance remains outside `ConditionBundle`; the Reasoner still receives only
geometric and task conditions.

## Dynamic control core

Geometry and Interaction have independent condition patch embeddings, stage
embeddings, rank-64 stage adapters, CrossNorm modules, and six zero-initialized
heads. They share one full-width joint Transformer block initialized from
DeepGen block 0, along with target patch, timestep/text, and context projections.

Every denoising step therefore sees the current noisy target, timestep, text,
pooled text, and bound identity tokens. Geometry runs for every row. Interaction
gathers only dual-person rows and scatters its outputs back; single-person rows
remain exactly zero. Mixed single/dual batches are compacted by token-mask
pattern so invalid padding tokens never enter joint attention.

## Strength and DeepGen injection

For group `l`, V6.3 applies:

```text
delta_l =
    geometry_strength * exp(g_geo_l) * w_geo(progress) * R_geo_l
  + interaction_strength * exp(g_int_l) * w_int(progress) * R_int_l
```

Defaults are Geometry `1.0`, Interaction `0.8`, and a constant full-denoise
window. Linear and cosine windows are also available. The six positive group
gates initialize to one.

Residuals are produced for target tokens only, then aligned to DeepGen's actual
`[target | source reference | padding]` sequence. Source and padding residuals
are always zero. Token height and width are derived from the current latent and
DeepGen patch size; 32x32 is not hard-coded.

DeepGen has 24 blocks and six native residual groups. Its existing interval
logic maps the groups to blocks 0-3, 4-7, 8-11, 12-15, 16-19, and 20-22. Block
23 is `context_pre_only` and does not receive a residual. The official
Transformer source is not modified.

## Controlled pipeline

`ControlledDeepGenPipeline` wraps a loaded DeepGen pipeline. It registers a
temporary Transformer forward pre-hook, computes dynamic controls once per
denoising step, and removes the hook in `finally`. Nested/concurrent calls and
simultaneous external `block_controlnet_hidden_states` are rejected. The base
pipeline always receives `control_scale=1.0`; branch strengths are applied only
by `ControlStrengthController`.

```python
controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)
result = controlled(
    prompt=prompt,
    image=image,
    condition_bundle=bundle,
    identity_condition=identity,
    source_scene_latents=scene_latent,
    geometry_strength=1.0,
    interaction_strength=0.8,
    num_inference_steps=28,
)
```

The static `PreparedControlConditioning` may be supplied directly for reuse.
Dynamic residuals are never cached across timesteps.

## Freezing and gradients

DeepGen Transformer, VAE, VLM, and connector parameters are frozen with
`requires_grad=False`. Training code must still call the frozen Transformer
without `torch.no_grad()` so loss gradients can pass through the added residuals
to the Adapter. The controlled pipeline is an inference wrapper; training uses
the Adapter and Transformer APIs directly.

At initialization all twelve residual heads are exactly zero. Consequently,
DeepGen with the Adapter must match the no-Adapter result within `1e-6`. This is
an interface invariant, not evidence of pose-control quality before training.

## Verification

```bash
cd /home/shangguanrz/project/pic-edit-main-v6
/home/shangguanrz/miniconda3/envs/deepgen/bin/python -m pytest -q

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_deepgen_control_interface.py \
  --mode single --image-size 512 --steps 2 --device cuda

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_deepgen_control_interface.py \
  --mode dual --image-size 512 --steps 2 --device cuda

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_deepgen_control_interface.py \
  --mode mixed --image-size 512 --steps 2 --device cuda
```

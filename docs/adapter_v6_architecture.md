# DeepGen Unified SMPL-X Adapter V6.1

V6.1 separates prepared SMPL-X condition injection from person-interaction
reasoning and DeepGen residual generation. It contains no renderer, SMPL-X
estimator, training loop, optimizer, dataset, or third-party checkpoint.

## Condition boundary

`SMPLXConditionInjector` receives explicit per-person conditions:

- world normal `[B,3,H,W]`;
- 25-channel joint heatmap `[B,25,H,W]`;
- 14-channel one-hot body-part map `[B,14,H,W]`;
- global vector `[betas(10), root rotation 6D(6), translation(3), camera(7)]`;
- optional human mask and scene-normalized depth;
- optional dual-person relative geometry, contact raster and typed relations.

Independent Normal/Pose/Part stems downsample to `H/8 × W/8`. Mid-level
fusion produces 256 spatial channels, and role-conditioned FiLM binds slots A
and B. Global, relative, contact and task data are encoded into 256-dimensional
tokens. The result is a `ConditionBundle`; the reasoning layer does not reopen
raw geometry maps.

Source-person latents remain in a separate `AdapterIdentityCondition`. The
`ConditionBundleBridge` projects public 256-dimensional conditions to the
current DeepGen Adapter widths before the existing single/dual experts.

## Backends and provenance

The default `native` backend preserves explicit heatmap and part channels. An
opt-in `champ` backend is available only for Normal and Depth. It is a
single-frame 2D adaptation of CHAMP's `GuidanceEncoder` at commit
`4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140`; exact changes and the MIT license
are recorded in `THIRD_PARTY_NOTICES.md` and `third_party/champ_guidance/`.
No CHAMP data or weights are downloaded.

## Minimal API

```python
adapter = UnifiedSMPLXAdapterV6.from_deepgen(transformer)

bundle = adapter.condition_injector(
    normal_a=normal,
    pose_heatmap_a=pose_heatmaps,
    part_onehot_a=part_onehot,
    smplx_global_a=global_vector,
    task_id=task_id,
)
identity = AdapterIdentityCondition(source_person_latents, source_indices)

residuals = adapter(
    target_latents=target_latents,
    condition_bundle=bundle,
    identity_condition=identity,
    source_scene_latents=source_scene_latents,
    cond_hidden_states=reference_latents,
    encoder_hidden_states=text_tokens,
    pooled_projections=pooled_text,
    timestep=timestep,
)
```

## Structural verification

```bash
cd /home/shangguanrz/project/pic-edit-main-v6

/home/shangguanrz/miniconda3/envs/deepgen/bin/python -m pytest -q \
  tests/test_condition_injector_v1.py \
  tests/test_condition_bridge_contract.py \
  tests/test_adapter_v6_architecture.py

/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/smoke_adapter_v6.py \
  --mode single --condition-backend native \
  --model-path /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers

/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/smoke_adapter_v6.py \
  --mode dual --condition-backend native \
  --model-path /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers

/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/smoke_adapter_v6.py \
  --mode dual --condition-backend champ --use-depth \
  --skip-deepgen-equivalence \
  --model-path /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers
```

The smoke report prints total and per-module parameter counts without enforcing
a size limit. Zero-initialized residual heads intentionally make an untrained
Adapter match the frozen DeepGen output; these checks do not measure pose-edit
quality.

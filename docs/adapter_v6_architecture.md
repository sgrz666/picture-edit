# DeepGen Unified SMPL-X Adapter V6.2

V6.2 adds a standalone Adapter-internal reasoning boundary after the V6.1
condition injector:

```text
prepared SMPL-X maps/parameters
        -> SMPLXConditionInjector
        -> ConditionBundle
        -> SMPLXAdapterReasoner
        -> InternalControlState
        -> ReasonerControlBridge
        -> SharedRecurrentControlCore
        -> six zero-initialized DeepGen residual groups
```

This phase contains no body estimator, renderer, data preprocessing, training
loop, optimizer, loss, checkpoint writer, external model weight, or schedule
change.

## Condition and identity boundaries

`SMPLXConditionInjector` retains the V6.1 public contract: per-person world
normal, 25-channel pose heatmaps, 14-channel part maps, masks, optional depth,
26 global SMPL-X/camera values, and optional dual-person relative/contact
conditions. It outputs 256-channel spatial conditions plus 256-dimensional
tokens in `ConditionBundle`.

Source appearance remains deliberately separate in `AdapterIdentityCondition`.
The Reasoner accepts only `ConditionBundle`; appearance enters later through
the DeepGen compatibility bridge. This keeps “how/where the body moves” apart
from “which referenced person moves.”

## Internal reasoning

`PersonGeometryReasoner` is shared by A and B. It projects each 256-channel
condition to 512 channels, applies two masked residual blocks, downsamples by
two, and fuses four global tokens plus the task token through pre-norm
cross-attention. A/B role embeddings are applied after this shared path.

Only rows with two valid people enter the interaction branch. Two
bidirectional cross-person layers update A and B simultaneously from the
previous layer state and include relative-geometry tokens in each direction.
Contact rasters are projected at both reasoning resolutions and enter through
two gates initialized to `sigmoid(-4)`. Typed contact relations use their true
padding mask; rows without a valid relation bypass attention, avoiding
all-masked softmax NaNs.

The Reasoner produces independent low/high geometry and interaction features:

```python
state.geometry_feature       # [B,512,Hc/2,Wc/2]
state.geometry_highres       # [B,512,Hc,Wc]
state.interaction_feature    # [B,512,Hc/2,Wc/2]
state.interaction_highres    # [B,512,Hc,Wc]
state.person_tokens          # [B,2,(Hc/2)*(Wc/2),512]
```

Single-person rows keep Person-B tokens and both interaction features exactly
zero. Mixed single/dual batches are supported by gathering dual rows, running
interaction reasoning only on those rows, then scattering them back.

## DeepGen compatibility

`UnifiedSMPLXAdapterV6.reason_conditions(bundle)` exposes the standalone
reasoning output. The unchanged `forward(...)` path then uses
`ReasonerControlBridge`:

- high-resolution geometry/interaction features are learned-downsampled and
  added to their low-resolution counterparts, then independently projected to
  the existing control-condition width; this keeps both declared scales on the
  end-to-end DeepGen gradient path;
- each person's masked spatial tokens are pooled into four summary tokens;
- those summaries are projected to the DeepGen width and bound to the existing
  eight appearance tokens using slot and source-index embeddings;
- raw task, relative, and contact tokens are not appended again because they
  have already participated in reasoning.

`SharedRecurrentControlCore`, six Zero Heads, the 24-layer DiT mapping,
timestep handling, and target/source/padding residual layout are unchanged.
Consequently an untrained V6.2 Adapter still emits exact-zero residuals and
must match no-Adapter DeepGen within `1e-6`.

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

state = adapter.reason_conditions(bundle)
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

## Verification

```bash
cd /home/shangguanrz/project/pic-edit-main-v6

/home/shangguanrz/miniconda3/envs/deepgen/bin/python -m pytest -q

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_adapter_reasoner.py --mode single --image-size 512 --device cuda
/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_adapter_reasoner.py --mode dual --image-size 512 --device cuda
/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_adapter_reasoner.py --mode mixed --image-size 512 --device cuda
```

Then rerun the existing Native single, Native dual, and CHAMP+Depth dual
DeepGen smokes from `scripts/smoke_adapter_v6.py`. Both smoke scripts report
parameters by module without enforcing a parameter ceiling.

## Design references

No external source or weight is copied in V6.2. The design records these
architectural references only:

- [MultiAnimate](https://github.com/hyc001/MultiAnimate)
- [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
- [Multi-HumanVid](https://github.com/zhenzhiwang/Multi-HumanVid)
- [HumanInteraction](https://github.com/boycehbz/HumanInteraction)
- [BUDDI](https://github.com/muelea/buddi)

The CHAMP guidance source already vendored for the optional V6.1 Normal/Depth
backend remains covered by `THIRD_PARTY_NOTICES.md`; V6.2 adds no new vendored
third-party code.

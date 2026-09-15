# DeepGen Unified SMPL-X Adapter V6

V6 is an architecture-only integration of single/dual-person SMPL-X control
with the frozen 24-layer DeepGen DiT. It does not contain a training loop,
optimizer, checkpoint, or data preprocessing pipeline.

## What is implemented

- A fixed two-slot `UnifiedAdapterCondition`; single-person samples explicitly
  mask slot B and missing modalities.
- Shared Normal/Depth/Skeleton/Silhouette/Part stems and 512 -> 256 -> 128 -> 64
  geometry features. The sparse skeleton stem uses MimicMotion-style He init.
- Eight SMPL-X geometry tokens and eight masked-source appearance tokens per
  person. Slot, source index, task and interaction embeddings bind identity.
- Deterministic one/two-person routing. Prompts label interactions but cannot
  override `person_valid`.
- A single-person expert that never evaluates contact/person-B branches.
- A dual-person expert with shared bidirectional attention at half the 64-level
  resolution (32x32 for 512 inputs), depth-aware fusion, directional contact
  maps and part-pair tokens.
- One full-width DeepGen-compatible joint block recurrently unrolled for six
  stages, with rank-64 stage adapters, CrossNorm, and six independent zero
  residual heads.
- DeepGen token alignment as `[target residual | source zeros | padding zeros]`.
  Six outputs map to blocks 0-3, 4-7, 8-11, 12-15, 16-19 and 20-22; block 23
  remains untouched.

The parameter count is reported, not capped. On the current DeepGen config the
V6 adapter has 150,457,376 trainable parameters; the DeepGen/VAE/VLM/connector
remain outside the adapter and frozen.

## Structural verification

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit-main-v6

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  -m pytest -q tests/test_adapter_v6_architecture.py

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_adapter_v6.py \
  --mode single \
  --model-path /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers

/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  scripts/smoke_adapter_v6.py \
  --mode dual \
  --model-path /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers
```

Each smoke run writes a JSON report under `outputs/`. Because all six output
heads are zero initialized, an untrained adapter intentionally produces no
visual pose edit and must match DeepGen without Adapter to within `1e-6`.

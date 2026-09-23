# Third-party design provenance

The V6 adapter does not bundle third-party datasets or model weights. Its
module boundaries and algorithms were informed by these Apache-2.0 projects:

- Hugging Face Diffusers `SD3ControlNetModel`: SD3 joint-transformer control
  initialization and zero residual heads.
- Tencent MimicMotion: He-initialized sparse pose convolutional stem.
- OminiControl: adapter-internal condition tokens for a diffusion transformer.

The architecture also follows published ideas (without source-code migration)
from CHAMP/ReImagine, T2I-Adapter, IP-Adapter, ControlNeXt, MultiAnimate,
PeopleComposer, BUDDI/Hi4D and closely-interactive human generation work.

## Bundled source adaptation

`third_party/champ_guidance/` is a single-frame adaptation of CHAMP's
`models/guidance_encoder.py` from commit
`4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140`:

- Upstream: https://github.com/fudan-generative-vision/champ
- License: MIT; the upstream license is reproduced in that directory.
- Changes: 3D inflated convolutions were converted to 2D for `F=1`; the video
  Transformer dependency was replaced by a residual spatial mixer; the final
  projection is not zero-initialized because CHAMP checkpoints are not loaded.
- No CHAMP dataset, video, checkpoint, or pretrained weight is included.

## V6.5 face-conditioning source adaptations

The narrow source migrations in `third_party/v65_face/` are called directly by
the V6.5 runtime. Exact commits, original paths, local interface differences,
and license status are recorded in
`third_party/v65_face/UPSTREAM_SOURCES.md`. No upstream checkpoint, model
weight, or dataset is included.

- `third_party/v65_face/stableanimator.py` migrates the identity projection,
  four-layer `FacePerceiver`, `FusionFaceId`, and offline ArcFace wrapper from
  StableAnimator `animation/modules/id_encoder.py` at commit `020e7f2`:
  https://github.com/Francis-Rings/StableAnimator/blob/020e7f2/animation/modules/id_encoder.py
- `third_party/v65_face/visual_persona.py` migrates the `FeedForward`,
  `PerceiverAttention`, and `Resampler` structure from Visual Persona
  `model/visual_persona/resampler.py` at commit `d393d09`, with DINOv2-G input
  dimensions, arbitrary leading dimensions, and explicit context masking:
  https://github.com/cvlab-kaist/Visual-Persona/blob/d393d09/model/visual_persona/resampler.py
- `third_party/v65_face/mcld.py` migrates the PoseGuider convolutional
  channel schedule from MCLD `src/models/pose_guider.py` at commit `6279ff7`;
  its inflated 3D convolutions are converted to 2D, the input is expanded to 73
  face channels, and the final projection uses ordinary initialization. It
  also migrates expanded-region and safe-padding logic from
  `preprocess_data/face_interface.py`:
  https://github.com/jqliu09/mcld/blob/6279ff7/src/models/pose_guider.py
- `third_party/v65_face/visual_persona.py` also migrates the independent image-prompt
  key/value projection and PyTorch 2.0 scaled-dot-product attention pattern
  from Visual Persona `model/visual_persona/attention_processor.py` at commit
  `d393d09`. It is rewritten as a recurrent ROI-local DiT adapter with six
  low-rank stages, timestep FiLM, zero residual heads, and overlap-normalized
  scatter for DeepGen target tokens:
  https://github.com/cvlab-kaist/Visual-Persona/blob/d393d09/model/visual_persona/attention_processor.py
- `third_party/v65_face/xdyna.py` migrates the classifier-free-guidance batch
  ordering and optional same-level face residual addition from X-Dyna
  `animatediff/pipelines/pipeline_xdyna.py` at commit `9a54f8e`; its SD1.5
  UNet and ControlNet implementation are not copied:
  https://github.com/bytedance/X-Dyna/blob/9a54f8e/animatediff/pipelines/pipeline_xdyna.py

StableAnimator's MIT license and X-Dyna's Apache-2.0 license are reproduced in
`third_party/v65_face/licenses/`. Visual Persona and MCLD had no repository-level
license file at the pinned commits; that status is explicitly recorded rather
than inferred.

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

## V6.5 face-conditioning design adaptations

The V6.5 face modules reimplement and adapt the following upstream designs;
no upstream checkpoint, model weight, dataset, or vendored source file is
included:

- `src/pose_control/v6/face/content_encoder.py` adapts the identity projection,
  four-layer `FacePerceiver`, and zero-initialized delta projection design from
  StableAnimator `animation/modules/id_encoder.py` at commit `020e7f2`:
  https://github.com/Francis-Rings/StableAnimator/blob/020e7f2/animation/modules/id_encoder.py
- `src/pose_control/v6/face/resampler.py` adapts the `FeedForward`,
  `PerceiverAttention`, and `Resampler` structure from Visual Persona
  `model/visual_persona/resampler.py` at commit `d393d09`, with DINOv2-G input
  dimensions, arbitrary leading dimensions, and explicit context masking:
  https://github.com/cvlab-kaist/Visual-Persona/blob/d393d09/model/visual_persona/resampler.py
- `src/pose_control/v6/face/spatial.py` adapts the PoseGuider convolutional
  channel schedule from MCLD `src/models/pose_guider.py` at commit `6279ff7`;
  its inflated 3D convolutions are converted to 2D, the input is expanded to 73
  face channels, and the final projection uses ordinary initialization:
  https://github.com/jqliu09/mcld/blob/6279ff7/src/models/pose_guider.py

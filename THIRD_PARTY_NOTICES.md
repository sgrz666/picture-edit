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

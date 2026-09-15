# Third-party design provenance

The V6 adapter implementation is original project code. It does not bundle
third-party weights or copy third-party source files verbatim. Its module
boundaries and algorithms were informed by these Apache-2.0 projects:

- Hugging Face Diffusers `SD3ControlNetModel`: SD3 joint-transformer control
  initialization and zero residual heads.
- Tencent MimicMotion: He-initialized sparse pose convolutional stem.
- OminiControl: adapter-internal condition tokens for a diffusion transformer.

The architecture also follows published ideas (without source-code migration)
from CHAMP/ReImagine, T2I-Adapter, IP-Adapter, ControlNeXt, MultiAnimate,
PeopleComposer, BUDDI/Hi4D and closely-interactive human generation work.

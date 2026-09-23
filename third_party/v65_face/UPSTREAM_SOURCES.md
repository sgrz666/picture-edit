# V6.5 face upstream source record

This directory contains narrow, interface-adapted source migrations for the
V6.5 independent Face Adapter. No dataset, checkpoint, or model weight is
bundled. The pinned inputs and local changes are:

| Local file | Upstream repository and commit | Original file | Local changes | License status at pin |
| --- | --- | --- | --- | --- |
| `stableanimator.py` | `Francis-Rings/StableAnimator@020e7f23a768410e0183ce943397d96a609f4ec1` | `animation/modules/id_encoder.py`, `animation/modules/face_model.py` | 512-D/4-query configuration, arbitrary leading dimensions, optional masks, delayed InsightFace import, offline extraction API | MIT; reproduced in `licenses/STABLEANIMATOR-MIT.txt` |
| `visual_persona.py` | `cvlab-kaist/Visual-Persona@d393d09284cac20570466ae3c7ad137b10648bdc` | `model/visual_persona/resampler.py`, `model/visual_persona/attention_processor.py` | DINOv2-G dimensions, context mask, ROI-local PyTorch 2.0 face K/V interface | no repository-level license file was present at the pinned commit; retained for the user-authorized open research implementation with source attribution |
| `mcld.py` | `jqliu09/MCLD@6279ff7c601c11dff5e075410ed34df2973da088` | `src/models/pose_guider.py`, `preprocess_data/face_interface.py` | inflated 3D convolution converted to 2D, 73 input channels, ordinary output initialization, normalized-coordinate safe crop | no repository-level license file was present at the pinned commit; retained for the user-authorized open research implementation with source attribution |
| `xdyna.py` | `bytedance/X-Dyna@9a54f8e9b90c195eb1f21641c791896bcefe4ce0` | `animatediff/pipelines/pipeline_xdyna.py` | extracted CFG batch repeat and optional same-level face residual addition | Apache-2.0; reproduced in `licenses/X-DYNA-APACHE-2.0.txt` |

Upstream URLs:

- https://github.com/Francis-Rings/StableAnimator/tree/020e7f23a768410e0183ce943397d96a609f4ec1
- https://github.com/cvlab-kaist/Visual-Persona/tree/d393d09284cac20570466ae3c7ad137b10648bdc
- https://github.com/jqliu09/MCLD/tree/6279ff7c601c11dff5e075410ed34df2973da088
- https://github.com/bytedance/X-Dyna/tree/9a54f8e9b90c195eb1f21641c791896bcefe4ce0

The two repositories without a repository-level license are explicitly marked
instead of being mislabeled as permissively licensed. Redistribution should be
revisited before any use beyond the user-authorized research release.

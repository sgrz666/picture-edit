---
license: other
license_link: LICENSE
license_name: qwen-research
library_name: videox_fun
tags:
- controlnet
- controlnet-union
- text-to-image
- image-to-image
- image-inpainting
tasks:
- text-to-image-synthesis
---

# Qwen-Image-2.1-Fun-Controlnet-Union

[![Github](https://img.shields.io/badge/🎨%20Code-VideoX_Fun-blue)](https://github.com/aigc-apps/VideoX-Fun)

## Overview

Qwen-Image-2.1-Fun-Controlnet-Union is a **ControlNet-Union branch for [Qwen-Image 2.1](https://github.com/aigc-apps/VideoX-Fun)** (the flow-matching text-to-image DiT). A single checkpoint drives **8 structural control conditions** (Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose, Scribble) *and* image inpainting, without per-condition weights. The checkpoint holds **only the control branch** (`control_img_in` plus 16 `control_blocks`, about 7.0 GB) and is loaded on top of the base Qwen-Image 2.1 transformer.

## Model Card

| Name | Description |
|--|--|
| Qwen-Image-2.1-Fun-Controlnet-Union.safetensors | ControlNet-Union branch weights for Qwen-Image 2.1. Contains only the control branch (`control_img_in` + 16 `control_blocks`, about 7.0 GB); loaded with `strict=False` on top of the base Qwen-Image 2.1 transformer. One checkpoint covers 8 control conditions and image inpainting. |

## Model Features
- **Union control over 8 conditions**: one checkpoint handles Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose and Scribble control images for text-to-image generation — no per-condition checkpoint switching.
- **Dense control injection**: the control branch attaches a skip to every 2nd of the 32 transformer blocks (`control_layers = [0, 2, 4, …, 30]`, 16 injection points). Each control skip is added back to the main branch through zero-gated `before_proj` / `after_proj` projections, giving tight structural adherence while the base model stays frozen.
- **Control and inpainting share one branch**: the control input is widened to `control_in_dim = 129` — `control latents (64) | mask (1) | masked-image latents (64)`. For pure control the mask / masked-image channels are zero-padded; for inpainting the same branch re-draws the masked region from the prompt. The two can also be **combined** — a control image and a mask are fed together, so the re-drawn region follows both the prompt and the given structure.
- **CFG-distilled fast sampling**: the standalone example scripts run with `guidance_scale = 1.0` (single forward pass per step, no classifier-free guidance needed).
- `control_context_scale` scales every control skip before it is added to the main branch: `1.0` is the strongest control (used for all results below), lower values weaken the guidance, `0.0` switches the control branch off.
- **Prompt-friendly**: write a prompt that describes the **whole target image**; the masked region is conveyed by the mask channel, not by the text. Detailed prompts give better stability.
- Qwen-Image 2.1 encodes the prompt (and any condition image) with a **Qwen3-VL** text encoder + processor, and its VAE decodes to **RGBA**, so every preview is saved as PNG.

## Supported control conditions

| Condition | Control signal |
|--|--|
| Canny | Canny edge map |
| Depth | Monocular depth map |
| Grayscale | Grayscale (luminance) image |
| HED | HED edge detection map |
| Lineart | Line-art extraction |
| MLSD | Line-segment detection map |
| Pose | DWPose skeleton |
| Scribble | Free-hand / sketch lines |

Any ordinary RGB control image at the target canvas works; the model tolerates different line thickness, thresholds and crops.

## Results

All samples below are generated with `num_inference_steps = 40`, `control_context_scale = 1.0`, seed 43. In each column the top row is the control image, the bottom row is the output.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr><td>Canny</td><td>Depth</td><td>Grayscale</td><td>HED</td><td>Lineart</td><td>MLSD</td><td>Pose</td><td>Scribble</td></tr>
  <tr>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_2_00025069.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_5_00005389.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_1_00014291.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_6_00009388.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_4_00003435.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_4_00027496.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_4_00000931.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/control_13_00000494.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_2_00025069.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_5_00005389.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_1_00014291.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_6_00009388.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_4_00003435.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_4_00027496.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_4_00000931.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/control_13_00000494.png" width="100%"></td>
  </tr>
</table>

### Inpainting (+ control)

A masked region of the source image is re-drawn from the prompt while the rest of the frame is preserved. The mask image is **white where the content should be re-generated** and **black where it should be kept**. Because control and inpainting share the same branch, a control image (here a DWPose skeleton) is fed together with the mask, so the re-drawn region also follows the given pose.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr><td>Source image</td><td>Mask</td><td>Pose control</td><td>Inpaint output</td></tr>
  <tr>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/inpaint_source.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/inpaint_mask.png" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/asset/inpaint_control.jpg" width="100%"></td>
    <td><img src="https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/resolve/main/results/inpaint.png" width="100%"></td>
  </tr>
</table>

## Inference
Go to the VideoX-Fun repository for more details.

Please clone the VideoX-Fun repository and create the required directories:

```sh
# Clone the code
git clone https://github.com/aigc-apps/VideoX-Fun.git

# Enter VideoX-Fun's directory
cd VideoX-Fun

# Create model directories
mkdir -p models/Diffusion_Transformer
```

Then download the base Qwen-Image 2.1 model and this checkpoint into `models/Diffusion_Transformer`.

```
📦 models/
├──  Diffusion_Transformer/
│   ├── 📂 Qwen-Image-2.1/
│   └── 📂 Qwen-Image-2.1-Fun-Controlnet-Union/
│       └──  Qwen-Image-2.1-Fun-Controlnet-Union.safetensors
```

Then edit the settings at the top of `examples/qwenimage21_fun/predict_t2i_control.py` (or `predict_i2i_inpaint.py` for inpainting) and run it.

```python
model_name          = "models/Diffusion_Transformer/Qwen-Image-2.1"
config_path         = "config/qwenimage21/qwenimage21_control.yaml"
transformer_path    = "models/Diffusion_Transformer/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
control_image       = "asset/pose.jpg"
# inpaint only:
inpaint_image       = "asset/8.png"
mask_image          = "asset/mask.png"
prompt              = "your prompt describing the whole target image"
```

```sh
python examples/qwenimage21_fun/predict_t2i_control.py
```

Notes:
- `config_path` **must** be `config/qwenimage21/qwenimage21_control.yaml`. It builds the control branch exactly as the checkpoint expects (`control_layers: [0, 2, 4, …, 30]`, `control_in_dim: 129`); a mismatched config silently drops or misplaces control weights and produces wrong outputs.
- For pure control (no inpaint input) the pipeline zero-pads the mask / masked-image channels, so this inpaint-capable checkpoint still runs plain Canny/Depth/… control correctly.
- The control checkpoint carries only the control branch; the base Qwen-Image 2.1 weights must be present in `model_name`.
- `control_context_scale = 1.0` is the value the adapter expects; use lower values to loosen the structural constraint.
- `sample_size` sets the output canvas (e.g. `[1728, 992]`); keep both sides as multiples of 16 so the control map is not distorted.
- `use_kv_cache = True` caches the text / condition-image keys after the first denoising step for a speedup at fixed resolution.
- Memory: the transformer plus the Qwen3-VL text encoder do not fit a single consumer GPU fully loaded; use `model_group_offload` (fastest) or `model_cpu_offload_and_qfloat8` on a single high-memory GPU.

## License

This model is a derivative of Qwen-Image 2.1 and is released under the [Qwen Research License](https://modelscope.cn/models/Qwen/Qwen-Image-2.1/file/view/master/LICENSE). Please read the license carefully before use.

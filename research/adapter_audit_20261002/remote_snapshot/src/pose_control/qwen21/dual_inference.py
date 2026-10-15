"""Target-RGB-free generation boundary for Hi4D Qwen21 dual editing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from PIL import Image

from src.data.hi4d_qwen21_dataset import collate_hi4d
from .body_adapter import Qwen21ControlContext
from .runtime import Qwen21IPERCache, dataset_image_to_pil


# These fields are derived from the source or supplied target geometry. In
# particular, tgt_image, target_rgb and gt_mask_A/B never cross this boundary.
CONTROL_SAMPLE_KEYS = (
    "normal_a", "normal_b", "pose_heatmap_a", "pose_heatmap_b",
    "part_onehot_a", "part_onehot_b", "smplx_global_a", "smplx_global_b",
    "human_mask_a", "human_mask_b", "depth_a", "depth_b",
    "relative_geometry", "contact_raster", "task_id", "person_b_valid",
    "contact_src_person", "contact_src_part", "contact_dst_person",
    "contact_dst_part", "contact_contact_type", "contact_distance",
    "contact_valid_mask", "source_mask_a", "source_mask_b",
)
CONTROL_BATCH_KEYS = (set(CONTROL_SAMPLE_KEYS) - {
    "contact_src_person", "contact_src_part", "contact_dst_person",
    "contact_dst_part", "contact_contact_type", "contact_distance",
    "contact_valid_mask",
}) | {"contact_relations"}


@dataclass(frozen=True)
class NativeGenerationInput:
    source_image: Image.Image
    prompt: str


@dataclass(frozen=True)
class DualGenerationInput(NativeGenerationInput):
    controls: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self.controls) != CONTROL_BATCH_KEYS:
            missing = CONTROL_BATCH_KEYS - set(self.controls)
            extra = set(self.controls) - CONTROL_BATCH_KEYS
            raise ValueError(f"dual generation control keys invalid: missing={missing}, extra={extra}")


def generation_inputs_from_sample(
    sample: Mapping[str, Any], native_prompt: str, adapter_prompt: str,
) -> tuple[NativeGenerationInput, DualGenerationInput]:
    """Whitelist source and control tensors from a dataset row that may contain GT."""
    if not native_prompt.strip() or not adapter_prompt.strip():
        raise ValueError("both generation prompts must be nonempty")
    source = dataset_image_to_pil(sample["src_image"])
    selected = {name: sample[name] for name in CONTROL_SAMPLE_KEYS}
    controls = collate_hi4d([selected])
    return (NativeGenerationInput(source, native_prompt),
            DualGenerationInput(source, adapter_prompt, controls))


def _pipeline_kwargs(source_image: Image.Image, prompt: str, steps: int, seed: int) -> dict:
    if steps <= 0:
        raise ValueError("sampling steps must be positive")
    return dict(prompt=prompt, image=source_image, width=512, height=512,
                output_resolution=512, num_inference_steps=steps,
                true_cfg_scale=1.0, use_kv_cache=True,
                generator=torch.Generator(device="cuda").manual_seed(seed))


@torch.inference_mode()
def generate_native(pipe, inputs: NativeGenerationInput, *, steps: int, seed: int) -> Image.Image:
    """The original model receives only the source photo and detailed text."""
    if type(inputs) is not NativeGenerationInput:
        raise TypeError("native generation requires NativeGenerationInput without controls")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pipe(**_pipeline_kwargs(inputs.source_image, inputs.prompt, steps, seed)).images[0].convert("RGB")


@torch.inference_mode()
def generate_controlled(pipe, adapter, cache: Qwen21IPERCache,
                        inputs: DualGenerationInput, *, mode: str, steps: int, seed: int) -> Image.Image:
    """The adapter receives source photo, text and supplied target geometry only."""
    if mode not in ("geometry", "full"):
        raise ValueError("controlled generation mode must be geometry or full")
    reference = cache.source(pipe, inputs.source_image, prompt=inputs.prompt)["reference_latent"].to("cuda")
    state = adapter.encode_dual(inputs.controls, reference,
                                include_contact=(mode == "full"))
    if mode == "geometry":
        state = state.geometry_only()
    with Qwen21ControlContext(pipe.transformer, adapter, state) as control:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image = pipe(**_pipeline_kwargs(inputs.source_image, inputs.prompt, steps, seed)).images[0].convert("RGB")
    if any(count == 0 for count in control.call_counts.values()):
        raise RuntimeError("a Qwen control layer was not used during inference")
    return image

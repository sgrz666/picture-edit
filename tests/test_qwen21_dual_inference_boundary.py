"""Target pixels and evaluation-only masks must not enter generation inputs."""

import copy

import pytest
import torch

from src.pose_control.qwen21.dual_inference import (
    CONTROL_SAMPLE_KEYS, DualGenerationInput, NativeGenerationInput,
    generation_inputs_from_sample, generate_native,
)


def _sample():
    sample = {name: torch.zeros(1) for name in CONTROL_SAMPLE_KEYS}
    for name in ("contact_src_person", "contact_src_part", "contact_dst_person",
                 "contact_dst_part", "contact_contact_type", "contact_valid_mask"):
        sample[name] = torch.zeros(8, dtype=torch.long)
    sample["contact_distance"] = torch.zeros(8, 1)
    sample["src_image"] = torch.zeros(3, 8, 8)
    sample["tgt_image"] = torch.ones(3, 8, 8)
    sample["target_person_mask_A"] = torch.ones(1, 8, 8)
    sample["target_person_mask_B"] = torch.ones(1, 8, 8)
    sample["row"] = {"target_rgb": "/a/secret-target.jpg"}
    return sample


def test_generation_input_is_independent_of_target_pixels_and_gt_masks():
    first = _sample()
    second = copy.deepcopy(first)
    second["tgt_image"] = torch.randn(3, 8, 8)
    second["target_person_mask_A"] *= 0
    second["target_person_mask_B"] *= 0
    second["row"]["target_rgb"] = "/different-target.jpg"
    native_a, dual_a = generation_inputs_from_sample(first, "detailed", "short")
    native_b, dual_b = generation_inputs_from_sample(second, "detailed", "short")
    assert native_a.source_image.tobytes() == native_b.source_image.tobytes()
    assert dual_a.source_image.tobytes() == dual_b.source_image.tobytes()
    assert set(dual_a.controls) == set(dual_b.controls)
    for name in dual_a.controls:
        if name == "contact_relations":
            for field in vars(dual_a.controls[name]):
                assert torch.equal(getattr(dual_a.controls[name], field),
                                   getattr(dual_b.controls[name], field))
        else:
            assert torch.equal(dual_a.controls[name], dual_b.controls[name])
    assert "tgt_image" not in dual_a.controls
    assert "target_rgb" not in dual_a.controls
    assert "target_person_mask_A" not in dual_a.controls


def test_generation_boundary_rejects_target_fields_and_native_controls():
    native, dual = generation_inputs_from_sample(_sample(), "detailed", "short")
    with pytest.raises(ValueError, match="extra"):
        DualGenerationInput(dual.source_image, dual.prompt,
                            {**dual.controls, "tgt_image": torch.ones(3, 8, 8)})
    with pytest.raises(TypeError, match="without controls"):
        generate_native(None, dual, steps=1, seed=1)
    assert type(native) is NativeGenerationInput

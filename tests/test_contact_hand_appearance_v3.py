from dataclasses import replace
import io

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.canvas_contract import native_pad
from src.pose_control.contact_hand_adapter.appearance import (
    AppearanceEncoder,
    attach_dino_appearance,
    masked_rgb_stats,
)
from src.pose_control.contact_hand_adapter.augmentation import swap_condition_people
from src.pose_control.contact_hand_adapter.conditions import HandCondition, verify_appearance_sources
from src.pose_control.contact_hand_adapter.interhand import Frame, source_appearance


def test_orderless_fallback_ignores_masked_pixels_and_pixel_order():
    generator = torch.Generator().manual_seed(12)
    crop = torch.rand(1, 3, 24, 24, generator=generator)
    mask = torch.zeros(1, 24, 24, dtype=torch.bool)
    mask[:, :12] = True
    changed = torch.where(mask[:, None], crop, 1 - crop)
    shuffled = crop.flatten(2).flip(-1).view_as(crop)
    assert torch.allclose(masked_rgb_stats(crop, mask), masked_rgb_stats(changed, mask))
    assert torch.allclose(masked_rgb_stats(crop, mask), masked_rgb_stats(shuffled, mask.flip((-1, -2))))
    out = AppearanceEncoder(width=24)(crop, mask=mask)
    assert out.tokens.shape == (1, 16, 24)
    assert not out.part_valid.any()
    assert torch.equal(out.tokens[:, :1].expand_as(out.tokens), out.tokens)


def test_local_tokens_follow_source_anatomical_id_with_no_position_routing():
    crop = torch.zeros(1, 3, 32, 32)
    crop[:, 0, :16] = 1
    crop[:, 1, 16:] = 1
    parts = torch.full((1, 32, 32), 2, dtype=torch.long)
    parts[:, 16:] = 7
    swapped = torch.where(parts == 2, 7, 2)
    encoder = AppearanceEncoder(width=16)
    a, b = encoder(crop, parts), encoder(crop, swapped)
    assert a.part_valid.nonzero().tolist() == [[0, 2], [0, 7]]
    assert not torch.allclose(a.tokens[0, 2], a.tokens[0, 7])
    assert torch.allclose(a.tokens[:, 2], b.tokens[:, 7])
    assert torch.allclose(a.tokens[:, 7], b.tokens[:, 2])


def test_cache_content_and_source_binding_survive_whole_person_swap():
    left, right = "a" * 64, "b" * 64
    appearance = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    mask = torch.ones(4, 16, 16, dtype=torch.bool)
    parts = torch.zeros(4, 16, 16, dtype=torch.long)
    condition = HandCondition(native_pad((32, 32)), ("left", "right"), torch.zeros(4, 21, 3),
                              torch.zeros(4, 21, dtype=torch.bool), appearance=appearance,
                              appearance_valid=torch.ones(4, dtype=torch.bool),
                              appearance_sources=(left, left, right, right),
                              appearance_parts=parts, appearance_mask=mask)

    class Extractor:
        model_sha = "f" * 64

        def __call__(self, crops, mask):
            return crops.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 2, 256).clone()

    cached = attach_dino_appearance(condition, Extractor(), (left, right))
    assert cached.appearance_dino_valid.all()
    verify_appearance_sources(cached, (left, right))
    swapped = swap_condition_people(cached, shared_appearance_source=False)
    verify_appearance_sources(swapped, (right, left))
    assert torch.equal(swapped.appearance_dino, cached.appearance_dino[[2, 3, 0, 1]])
    with pytest.raises(ValueError, match="crop"):
        verify_appearance_sources(replace(cached, appearance=1 - appearance), (left, right))


def test_interhand_256_crop_keeps_conservative_source_semantics_and_rgb():
    canvas = native_pad((64, 64))
    joints = torch.zeros(4, 21, 3)
    joints[0, :5] = torch.tensor(((24, 32, 1), (30, 29, 1), (35, 26, 1), (40, 23, 1), (48, 22, 1)))
    joints[0, 5:9] = torch.tensor(((28, 28, 1), (28, 24, 1), (26, 18, 1), (24, 11, 1)))
    projected = joints[..., 2] > 0
    source_condition = HandCondition(canvas, ("A", None), joints, projected)
    source_frame = Frame("train", "0", "A", "seq", "123", 0, "source.jpg", (64, 64),
                         "left", np.ones(42, dtype=bool))
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (102, 51, 0)).save(stream, format="PNG")
    payload = source_appearance(source_condition, source_frame, stream.getvalue(), "a" * 64)
    condition = replace(source_condition, **payload).validate()
    assert condition.appearance.shape == (4, 3, 256, 256)
    assert condition.appearance[0, :, 128, 128].tolist() == pytest.approx([.4, .2, 0.])
    assert condition.appearance_valid.tolist() == [True, False, False, False]
    parts, foreground = condition.appearance_parts[0], condition.appearance_mask[0]
    assert ((parts == 3) | (parts == 6)).any()
    assert (foreground & (parts == -1)).any()
    assert not ((parts >= 0) & ~foreground).any()

"""Training transforms for whole-person identity and local source context."""

from dataclasses import replace

import torch
from PIL import Image, ImageFilter

from .conditions import invert_relative_pose


PERSON_SWAP = (2, 3, 0, 1)
SLOT_FIELDS = (
    "joints", "visible", "support", "geometry_maps", "map_valid", "appearance", "appearance_valid",
    "appearance_sources", "joints_3d", "joint_3d_valid", "joint_2d_valid", "joint_occluded", "hand_present",
    "joint_rotation_6d", "joint_rotation_valid", "mano_shape", "mano_wrist_rotation_6d", "mano_palm_scale", "mano_valid",
    "appearance_parts", "appearance_mask", "appearance_dino", "appearance_dino_valid", "appearance_dino_crop_sha",
)


def swap_hand_slots(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(value[slot] for slot in PERSON_SWAP)
    return value[list(PERSON_SWAP)]


def _transpose(value):
    return None if value is None else value.T


def _swap_pair(pair):
    first, second = (PERSON_SWAP[slot] for slot in pair.slots)
    if first < second:
        return replace(pair, slots=(first, second))
    return replace(pair, slots=(second, first), part_state=pair.part_state.T,
                   part_confidence=_transpose(pair.part_confidence),
                   occlusion=None if pair.occlusion is None else -pair.occlusion.T,
                   relative_pose=None if pair.relative_pose is None else invert_relative_pose(pair.relative_pose),
                   surface_indices=None if pair.surface_indices is None else pair.surface_indices.flip(1))


def swap_condition_people(condition, *, shared_appearance_source):
    """Exchange A/B without changing anatomical side, canvas coordinates or actor pixels."""
    mesh = condition.mesh
    if mesh is not None:
        mesh = replace(mesh, **{name: swap_hand_slots(getattr(mesh, name))
                                for name in ("vertices", "anchor_xy", "valid", "camera_ids")})
    return replace(condition, person_ids=condition.person_ids[::-1], mesh=mesh,
                   pairs=tuple(_swap_pair(pair) for pair in condition.pairs),
                   shared_appearance_source=shared_appearance_source,
                   **{name: swap_hand_slots(getattr(condition, name)) for name in SLOT_FIELDS}).validate()


def distort_source_context(image, plan, current_support, rng):
    """Perturb the resized source RGB inside the selected current-hand support only."""
    x0, y0, x1, y1 = plan.box
    support = current_support[list(plan.slots), y0:y1, x0:x1].amax(0, keepdim=True)
    local_support = plan.condition.canvas.stack(support.float(), nearest=True)[0] > 0
    mask = Image.fromarray((local_support.byte() * 255).numpy())
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Synthetic local context requires current support for a selected hand")
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    variant = int(torch.randint(3, (), generator=rng))
    if variant == 0:
        degraded = image.filter(ImageFilter.GaussianBlur(max(1., min(width, height) * .10)))
    else:
        cx, cy = (left + right) / 2, (top + bottom) / 2
        sx, sy = (.75, 1.25) if variant == 1 else (1.25, .75)
        affine = (sx, 0., (1 - sx) * cx, 0., sy, (1 - sy) * cy)
        degraded = image.transform(image.size, Image.Transform.AFFINE, affine, resample=Image.Resampling.BICUBIC)
    return Image.composite(degraded, image, mask)

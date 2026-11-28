"""Render only trusted InterHand joints on the shared RGB canvas."""

import io
import math
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

from src.data.canvas_contract import letterbox
from src.pose_control.contact_hand_adapter.conditions import JOINT_PARENTS
from src.pose_control.contact_hand_adapter.runtime import load_condition


PALETTES = (
    ("#FF5D62", "#FF9F43", "#F7E561", "#B5E854", "#EF85FF"),
    ("#00C7E8", "#408AFF", "#886DFF", "#59E8B5", "#E2B4FF"),
)
HAND_SLOTS = {"left": (0,), "right": (1,), "interacting": (0, 1)}
POSE_CAPTIONS = {
    "left": "Edit the left hand gesture in image 1 to match the left hand pose shown in image 2.",
    "right": "Edit the right hand gesture in image 1 to match the right hand pose shown in image 2.",
    "interacting": "Edit both hand gestures in image 1 to match the corresponding left and right hand poses in image 2.",
}
PRESERVE = " Preserve the hand identity, skin appearance, lighting, and background of image 1."


class UnrenderablePose(ValueError):
    """An expected hand has too little trusted geometry to supervise the edit."""


def dashed_bone(draw, a, b, color, width):
    length = math.dist(a, b)
    if not length:
        return
    color = tuple(round(channel * .42) for channel in ImageColor.getrgb(color))
    for start in range(0, math.ceil(length), width * 4):
        start_t, end_t = start / length, min(start + width * 2, length) / length
        endpoints = [tuple(x + (y - x) * t for x, y in zip(a, b)) for t in (start_t, end_t)]
        draw.line(endpoints, fill=color, width=max(1, width - 1))


def render_pose(condition, view_valid, output_canvas, hand_type, *, pose_mode="visible"):
    import torch

    if (view_valid.shape != (2, 21) or view_valid.dtype != torch.bool
            or condition.canvas.original_hw != output_canvas.original_hw):
        raise ValueError("Pose visibility and target RGB must share the official InterHand frame")
    if pose_mode not in ("visible", "occlusion-dashed"):
        raise ValueError(f"Unknown pose mode: {pose_mode}")
    projected = condition.projected_joints()[:2]
    valid = projected & view_valid
    hidden = torch.zeros_like(valid)
    if condition.joint_occluded is not None:
        valid &= ~condition.joint_occluded[:2]
        if pose_mode == "occlusion-dashed" and condition.joint_3d_valid is not None:
            depth = condition.joints_3d[:2, :, 2]
            hidden = (projected & condition.joint_3d_valid[:2] & condition.joint_occluded[:2]
                      & torch.isfinite(depth) & (depth > 0))
    joints = condition.joints[:2, :, :2]
    original = (joints - joints.new_tensor(condition.canvas.offset_xy)) / joints.new_tensor(condition.canvas.scale_xy)
    points = output_canvas.pixel_points(original)
    h, w = output_canvas.canvas_hw
    image = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(image)
    keypoints = [[None] * 21 for _ in PALETTES]
    solid = []
    widths = {}
    for side in HAND_SLOTS[hand_type]:
        eligible = valid[side].tolist()
        coordinates = points[side].tolist()
        trusted = (valid[side] | hidden[side]).tolist()
        bones = [(parent, child) for child, parent in enumerate(JOINT_PARENTS)
                 if child and trusted[parent] and trusted[child]]
        if not trusted[0] or not bones or not any(eligible):
            raise UnrenderablePose(f"InterHand target {hand_type} lacks a trusted wrist and bone in slot {side}")
        trusted_points = points[side, valid[side] | hidden[side]]
        span = float((trusted_points.amax(0) - trusted_points.amin(0)).max())
        width = max(2, min(6, round(span / 64)))
        widths[side] = width
        for parent, child in bones:
            color = PALETTES[side][(child - 1) // 4]
            a, b = coordinates[parent], coordinates[child]
            if eligible[parent] and eligible[child]:
                solid.append(((tuple(a), tuple(b)), color, width))
            else:
                dashed_bone(draw, a, b, color, width)
        for idx, good in enumerate(eligible):
            if good:
                keypoints[side][idx] = coordinates[idx]
    for endpoints, color, width in solid:
        draw.line(endpoints, fill=color, width=width)
    for side in HAND_SLOTS[hand_type]:
        width = widths[side]
        for idx, xy in enumerate(keypoints[side]):
            if xy is not None:
                x, y = xy
                color = ("#FFFFFF", "#83BAFF")[side] if idx == 0 else PALETTES[side][(idx - 1) // 4]
                draw.ellipse((x - width, y - width, x + width, y + width), fill=color)
    return image, keypoints


def png_asset(image, name):
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return {"path": f"{name}.png", "bytes": handle.getvalue()}


def aligned_triplet(root, row, source, target, size, digest_file, *, pose_mode="visible"):
    import torch

    condition_asset = row["condition"]
    path = root / condition_asset["path"]
    if digest_file(path) != condition_asset["sha256"]:
        raise ValueError(f"Target condition SHA256 mismatch: {path}")
    condition = load_condition(path)
    view_valid = row["interhand"].get("target_view_joint_valid")
    if view_valid is None:
        raise ValueError(f"Missing per-view joint validity for {row['id']}; re-export the InterHand manifest")
    if condition.person_ids != (row["group"], None):
        raise ValueError(f"Target projection subject differs from RGB for {row['id']}")
    output = letterbox((size[1], size[0]), 512)
    kind = row["interhand"]["hand_type"]
    pose, keypoints = render_pose(condition, torch.as_tensor(view_valid), output, kind, pose_mode=pose_mode)
    with Image.open(io.BytesIO(source["bytes"])) as a, Image.open(io.BytesIO(target["bytes"])) as b:
        return (
            png_asset(output.image(a.convert("RGB")), "source"),
            png_asset(output.image(b.convert("RGB")), "target"),
            png_asset(pose, "pose"), keypoints,
        )

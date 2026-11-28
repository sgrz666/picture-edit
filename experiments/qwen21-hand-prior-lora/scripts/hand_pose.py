"""Render only trusted InterHand joints on the shared RGB canvas."""

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

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


def render_pose(condition, view_valid, output_canvas, hand_type):
    import torch

    if (view_valid.shape != (2, 21) or view_valid.dtype != torch.bool
            or condition.canvas.original_hw != output_canvas.original_hw):
        raise ValueError("Pose visibility and target RGB must share the official InterHand frame")
    valid = condition.projected_joints()[:2] & view_valid
    if condition.joint_occluded is not None:
        valid &= ~condition.joint_occluded[:2]
    joints = condition.joints[:2, :, :2]
    original = (joints - joints.new_tensor(condition.canvas.offset_xy)) / joints.new_tensor(condition.canvas.scale_xy)
    points = output_canvas.pixel_points(original)
    h, w = output_canvas.canvas_hw
    image = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(image)
    keypoints = [[None] * 21 for _ in PALETTES]
    for side in HAND_SLOTS[hand_type]:
        eligible = valid[side].tolist()
        coordinates = points[side].tolist()
        bones = [(parent, child) for child, parent in enumerate(JOINT_PARENTS)
                 if child and eligible[parent] and eligible[child]]
        if not eligible[0] or not bones:
            raise UnrenderablePose(f"InterHand target {hand_type} lacks a trusted wrist and bone in slot {side}")
        trusted_points = points[side, valid[side]]
        span = float((trusted_points.amax(0) - trusted_points.amin(0)).max())
        width = max(2, min(6, round(span / 64)))
        for parent, child in bones:
            color = PALETTES[side][(child - 1) // 4]
            draw.line((tuple(coordinates[parent]), tuple(coordinates[child])), fill=color, width=width)
        for idx, (xy, good) in enumerate(zip(coordinates, eligible)):
            if good:
                x, y = xy
                keypoints[side][idx] = xy
                color = ("#FFFFFF", "#83BAFF")[side] if idx == 0 else PALETTES[side][(idx - 1) // 4]
                draw.ellipse((x - width, y - width, x + width, y + width), fill=color)
    return image, keypoints


def png_asset(image, name):
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return {"path": f"{name}.png", "bytes": handle.getvalue()}


def aligned_triplet(root, row, source, target, size, digest_file):
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
    pose, keypoints = render_pose(condition, torch.as_tensor(view_valid), output, kind)
    with Image.open(io.BytesIO(source["bytes"])) as a, Image.open(io.BytesIO(target["bytes"])) as b:
        return (
            png_asset(output.image(a.convert("RGB")), "source"),
            png_asset(output.image(b.convert("RGB")), "target"),
            png_asset(pose, "pose"), keypoints,
        )

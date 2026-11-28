import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "experiments" / "qwen21-hand-prior-lora" / "scripts"))
from prepare_dataset import pair_rows


def write_asset(root, name, color, size=(64, 64)):
    path = root / name
    Image.new("RGB", size, color).save(path)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def pair(tmp_path):
    return {
        "id": "pair-1",
        "group": "InterHand2.6M:0",
        "split": "train",
        "source_refs": [write_asset(tmp_path, "first.png", "red")],
        "target": write_asset(tmp_path, "second.png", "blue"),
        "condition": {"path": "unused.pt"},
        "prompt": "Match target geometry.",
        "interhand": {"hand_type": "right"},
    }


def manifest(root, *rows):
    path = root / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pair_order_and_rgb_only_prompt(tmp_path, pair):
    val = dict(pair, id="val-1", split="val", group="InterHand2.6M:1",
               source_refs=[write_asset(tmp_path, "val-source.png", "yellow")],
               target=write_asset(tmp_path, "val-target.png", "green"))
    path = manifest(tmp_path, pair, val)
    row, = pair_rows(path, "train")
    assert row["id"] == "pair-1"
    assert row["source"]["bytes"] == (tmp_path / "first.png").read_bytes()
    assert row["target"]["bytes"] == (tmp_path / "second.png").read_bytes()
    assert "right hand" in row["caption"]
    assert "geometry" not in row["caption"]
    assert "condition" not in row
    assert row["source_sha256"] == pair["source_refs"][0]["sha256"]


def test_accepts_reviewed_target_instruction(tmp_path, pair):
    caption = "Extend the right index finger while keeping the remaining fingers curled."
    row, = pair_rows(manifest(tmp_path, dict(pair, caption=caption)), "train")
    assert row["caption"] == caption


def test_rejects_corrupt_rgb_and_different_canvas(tmp_path, pair):
    path = manifest(tmp_path, pair)
    Image.new("RGB", (64, 64), "green").save(tmp_path / "second.png")
    with pytest.raises(ValueError, match="SHA256"):
        list(pair_rows(path, "train"))
    pair["target"] = write_asset(tmp_path, "second.png", "blue", (32, 64))
    with pytest.raises(ValueError, match="dimensions"):
        list(pair_rows(manifest(tmp_path, pair), "train"))


def test_rejects_subject_leakage(tmp_path, pair):
    path = manifest(tmp_path, pair, dict(pair, id="val-1", split="val"))
    with pytest.raises(ValueError, match="subject"):
        list(pair_rows(path, "train"))


def test_two_image_latents_follow_reference_order_and_check_each_slot_boundary():
    import torch
    from qwen21_multiimg import pack_inputs

    target = torch.full((1, 64, 1, 4, 4), 3.)
    rgb = torch.full_like(target, 1.)
    pose = torch.full_like(target, 2.)

    def pack(latents, **kwargs):
        return latents.flatten(2).transpose(1, 2)

    image_mask = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)
    packed, shapes, mask, target_count = pack_inputs(pack, (rgb, pose), target, image_mask)
    assert packed[0, :, 0].tolist() == [1.] * 16 + [2.] * 16 + [3.] * 16
    assert shapes == [[(1, 4, 4)] * 3]
    assert target_count == 16 and mask[0, -4:].all() and mask.sum().item() == 12
    bad = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="per-image"):
        pack_inputs(pack, (rgb, pose), target, bad)


def test_render_pose_uses_same_letterbox_as_rgb_and_filters_unreliable_points():
    import torch
    from hand_pose import render_pose
    from src.data.canvas_contract import letterbox, native_pad
    from src.pose_control.contact_hand_adapter.conditions import HandCondition

    canvas = native_pad((128, 256))
    output = letterbox(canvas.original_hw, 512)
    joints = torch.zeros((4, 21, 3))
    joints[0, :3] = torch.tensor([[100., 30., 1.], [110., 32., 1.], [120., 50., 1.]])
    projected = joints[..., 2].bool()
    trusted = projected[:2].clone()
    trusted[0, 2] = False
    condition = HandCondition(canvas, ("subject", None), joints, torch.zeros_like(projected),
                              joint_2d_valid=projected)
    pose, keypoints = render_pose(condition, trusted, output, "left")
    assert pose.mode == "RGB" and pose.size == (512, 512)
    assert pose.getpixel((200, 188)) != (0, 0, 0)
    assert pose.getpixel((240, 228)) == (0, 0, 0)
    assert keypoints[0][:3] == [[200.0, 188.0], [220.0, 192.0], None]


def test_swap_selects_distinct_pose_for_the_exact_same_source_and_prompt():
    from verify_smoke import select_pose_swap

    row = {"source_sha256": "a", "caption": "edit", "group": "one", "hand_type": "left",
           "pose_sha256": "p0", "pose_joints": json.dumps([[[0, 0]] * 21, [None] * 21])}
    direction = json.dumps([[[3, 4]] * 21, [None] * 21])
    compatible = dict(row, pose_sha256="p1", pose_joints=direction)
    rows = [row, dict(compatible, source_sha256="other"), compatible]
    assert select_pose_swap(rows, 0) == (2, 5., 21)
    with pytest.raises(ValueError, match="same source"):
        select_pose_swap(rows, 0, 1)


def test_pose_dataset_keeps_source_supervision_and_provenance(tmp_path, pair):
    import io
    import torch
    from src.data.canvas_contract import native_pad
    from src.pose_control.contact_hand_adapter.conditions import HandCondition
    from src.pose_control.contact_hand_adapter.runtime import condition_record

    joints = torch.zeros((4, 21, 3))
    joints[1, :2] = torch.tensor([[10., 15., 1.], [18., 18., 1.]])
    valid = joints[..., 2].bool()
    condition = HandCondition(native_pad((64, 64)), (pair["group"], None), joints, torch.zeros_like(valid),
                              joint_2d_valid=valid)
    condition_path = tmp_path / "hand.pt"
    torch.save(condition_record(condition), condition_path)
    pair["condition"] = {"path": condition_path.name, "sha256": hashlib.sha256(condition_path.read_bytes()).hexdigest()}
    pair["interhand"]["target_view_joint_valid"] = valid[:2].tolist()
    row, = pair_rows(manifest(tmp_path, pair), "train", with_pose=True)
    assert "image 1" in row["caption"] and "image 2" in row["caption"]
    assert row["pose_sha256"] == hashlib.sha256(row["pose"]["bytes"]).hexdigest()
    assert row["source_sha256"] == pair["source_refs"][0]["sha256"]
    for column, color in (("source", (255, 0, 0)), ("target", (0, 0, 255)), ("pose", (0, 0, 0))):
        with Image.open(io.BytesIO(row[column]["bytes"])) as image:
            assert image.size == (512, 512) and image.getpixel((0, 0)) == color
    pair["interhand"]["target_view_joint_valid"] = [[False] * 21] * 2
    rejected = []
    assert not list(pair_rows(manifest(tmp_path, pair), "train", with_pose=True, rejected=rejected))
    assert rejected[0]["id"] == pair["id"] and "trusted wrist" in rejected[0]["reason"]
    pair["condition"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="condition SHA256"):
        list(pair_rows(manifest(tmp_path, pair), "train", with_pose=True))

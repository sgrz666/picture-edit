import io
import hashlib
import json
import pickle
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.data.prepare_interhand_contact_hand_adapter import main
from src.data.canvas_contract import native_pad
from src.pose_control.contact_hand_adapter.interhand import (
    Camera,
    Frame,
    TarImages,
    archive_parts,
    frame_condition,
    iter_coco,
    load_subject_splits,
    select_pairs,
)
from src.pose_control.contact_hand_adapter.interhand_geometry import (
    _SelectedShapeDirections,
    _surface_candidates,
    rasterize_hand,
)
from src.pose_control.contact_hand_adapter.training import load_manifest, load_training_pair
from src.pose_control.qwen21_pro.geometry import LiteralChumpyArray


def frame(index, *, subject="1", camera="101", split="train"):
    return Frame(split, "2", subject, "motion", camera, index,
                 f"{split}/Capture2/motion/cam{camera}/image{index}.jpg",
                 (64, 64), "interacting", np.ones(42, dtype=bool))


def test_projection_reorders_official_wrist_and_fingertips_and_keeps_unknown_visibility():
    camera = Camera(np.eye(3), np.array([100, 0, 0]), np.array([100, 100]), np.array([32, 32]))
    world_mm = np.stack((np.arange(42) + 100, np.zeros(42), np.full(42, 1000)), axis=-1)
    condition = frame_condition(frame(0), camera, world_mm, np.ones(42), native_pad((64, 64)))
    assert condition.person_ids == ("InterHand2.6M:1", None)
    assert condition.hand_present.tolist() == [True, True, False, False]
    assert condition.joints[0, 0].tolist() == pytest.approx([36.1, 32, 1])
    assert condition.joints[0, 4].tolist() == pytest.approx([34.1, 32, 1])
    assert condition.joints[1, 0].tolist() == pytest.approx([34, 32, 1])
    assert condition.joints_3d[0, 4].tolist() == pytest.approx([.021, 0, 1])
    assert condition.joint_3d_valid[:2].all()
    assert condition.joint_2d_valid[:2].all()
    assert not condition.visible.any()
    assert not condition.joint_occluded.any()


def test_pairs_never_cross_identity_or_view_and_use_different_frames(tmp_path):
    subject = tmp_path / "subject.txt"
    subject.write_text("# header\n1 train/Capture2 test/Capture5 male\n2 val/Capture2 female\n")
    owners, excluded = load_subject_splits(subject)
    assert owners == {"1": "train", "2": "val"}
    assert excluded == {"test/Capture5": "1"}
    pairs = list(select_pairs(
        [frame(0), frame(6), frame(12), frame(6, camera="102"), frame(6, subject="2")],
        min_gap=6, max_gap=7,
    ))
    assert [(a.index, b.index) for a, b in pairs] == [(0, 6), (6, 12)]


def test_intervention_can_pair_one_source_to_two_real_targets():
    pairs = select_pairs([frame(i) for i in (0, 6, 12, 18)], min_gap=6, max_gap=12,
                         targets_per_source=2)
    assert [(a.index, b.index) for a, b in pairs] == [(0, 6), (0, 12), (6, 12), (6, 18), (12, 18)]


def test_streaming_annotation_and_split_tar_member_reads(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"info": {"description": "x"}, "images": [{"id": 0}, {"id": 1}],
                                "annotations": [{"id": 2, "image_id": 1}]}))
    assert list(iter_coco(data, chunk_size=11)) == [
        ("images", {"id": 0}), ("images", {"id": 1}),
        ("annotations", {"id": 2, "image_id": 1}),
    ]
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(image, format="JPEG")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo(f"InterHand2.6M_5fps_batch1/images/{frame(6).path}")
        entry.size = len(image.getvalue())
        archive.addfile(entry, io.BytesIO(image.getvalue()))
    data = payload.getvalue()
    first, second = tmp_path / "frames.tar.partaa", tmp_path / "frames.tar.partab"
    first.write_bytes(data[:600])
    second.write_bytes(data[600:])
    with TarImages([first, second]) as archive:
        assert list(archive.members("train")) == [frame(6).path]
        assert archive.read(frame(6).path) == image.getvalue()


def test_archive_prefix_stops_at_an_incomplete_part_before_later_parts(tmp_path):
    data = (b"a" * 1024, b"b" * 1024, b"c" * 1024, b"d")
    names = [f"images.tar.parta{letter}" for letter in "abcd"]
    checksum = "\n".join(f"{hashlib.md5(chunk).hexdigest()}  {name}" for name, chunk in zip(names, data))
    (tmp_path / "images.tar.CHECKSUM").write_text(checksum)
    for name, chunk in zip(names[:3], (data[0], data[1][:600], data[2])):
        (tmp_path / name).write_bytes(chunk)
    parts, missing, incomplete = archive_parts(tmp_path, allow_partial=True)
    assert parts == [tmp_path / names[0]]
    assert missing == [names[-1]]
    assert incomplete == names[1]


def test_rasterization_uses_perspective_depth_and_part_zero_is_valid():
    canvas = native_pad((32, 32))
    pixels = np.array([[8, 8], [24, 8], [8, 24]], dtype=np.float64)
    vertices = np.array([[0, 0, 1], [2, 0, 2], [0, 2, 2]], dtype=np.float64)
    depths, normals, parts = rasterize_hand(canvas, vertices, pixels, np.array([[0, 1, 2]]),
                                           np.array([0, 1, 1]))
    assert depths[12, 12] == pytest.approx(4 / 3)
    assert np.linalg.norm(normals[:, 12, 12]) == pytest.approx(1)
    assert parts[8, 8] == 0
    assert np.isinf(depths[31, 31])


def test_mano_shape_directions_decode_only_literal_indices():
    data = np.arange(778 * 3 * 20, dtype=np.float64).reshape(778, 3, 20)
    literal = LiteralChumpyArray()
    literal.__setstate__({"x": data})
    inds = np.arange(data.size).reshape(data.shape)[:, :, :10].ravel()
    state = {"a": literal, "preferred_shape": (778, 3, 10), "idxs": inds}
    selected = _SelectedShapeDirections()
    selected.__setstate__(state)
    assert np.array_equal(selected.array(), data[:, :, :10])
    with pytest.raises(pickle.UnpicklingError, match="shape direction"):
        selected.__setstate__({**state, "a": object()})


def test_near_anchor_candidates_are_unknown_and_signed_penetration_reduces_quality():
    vertices = (np.array([[0., 0., 1.], [.2, 0., 1.]]),
                np.array([[0., 0., 1.004], [.2, 0., .993]]))
    normals = (np.array([[0., 0., 1.], [0., 0., 1.]]),
               np.array([[0., 0., -1.], [0., 0., -1.]]))
    payload = _surface_candidates(vertices, normals, (5., 8.))
    assert payload["surface_indices"].tolist() == [[0, 0], [1, 1]]
    assert payload["surface_features"][:, 0].tolist() == pytest.approx([.004, .007])
    assert payload["surface_features"][:, 2].tolist() == [-1, -1]
    assert payload["surface_quality"][:, 1].tolist() == [1., 1.]
    assert payload["surface_quality"][:, 2].tolist() == pytest.approx([.35, .35])
    assert payload["surface_quality"][0, 3] == 1 and payload["surface_quality"][1, 3] < .5


def test_cli_frame_pair_roundtrips_as_trainable_and_appearance_uses_source_rgb(tmp_path):
    annotations = tmp_path / "annotations" / "train"
    annotations.mkdir(parents=True)
    (annotations.parent / "subject.txt").write_text("1 train/Capture2 male\n")
    images, records, world = [], [], {}
    for idx, color in ((0, "red"), (6, "blue")):
        source = frame(idx)
        image = tmp_path / "images" / source.path
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(image)
        images.append({"id": idx, "file_name": source.path.split("/", 1)[1],
                       "width": 64, "height": 64, "capture": 2, "subject": 1,
                       "seq_name": source.sequence, "camera": source.camera, "frame_idx": idx})
        records.append({"id": idx, "image_id": idx, "hand_type": "interacting", "hand_type_valid": 1,
                        "joint_valid": [[1]] * 42})
        world[str(idx)] = {"world_coord": [[-180 + (j % 21) * 18 + idx * 12, -90 + (j % 4) * 50, 1000]
                                          for j in range(42)],
                           "joint_valid": [[1]] * 42}
    camera = {"camrot": {"101": np.eye(3).tolist()}, "campos": {"101": [0, 0, 0]},
              "focal": {"101": [100, 100]}, "princpt": {"101": [32, 32]}}
    for kind, data in (("data", {"images": images, "annotations": records}),
                       ("joint_3d", {"2": world}), ("camera", {"2": camera})):
        (annotations / f"InterHand2.6M_train_{kind}.json").write_text(json.dumps(data))
    out = tmp_path / "export"
    previous_threads = torch.get_num_threads()
    try:
        main(["--dataset-root", str(tmp_path), "--out", str(out), "--split", "train",
              "--min-gap", "6", "--min-motion", "5", "--max-pairs", "1"])
    finally:
        torch.set_num_threads(previous_threads)
    row, = load_manifest(out / "manifest.jsonl")
    pair = load_training_pair(row)
    assert row["interhand"]["source_frame"] == 0 and row["interhand"]["target_frame"] == 6
    assert pair.condition.appearance_valid.tolist() == [True, True, False, False]
    assert pair.condition.appearance_sources[:2] == pair.source_hashes * 2
    assert pair.condition.appearance[:2, 0].mean() > .99
    assert pair.condition.appearance[:2, 2].max() == 0
    assert pair.target.getpixel((32, 32))[2] > 250
    assert pair.regions.hands[:2].any()

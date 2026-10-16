"""Contracts that prevent split leakage and coordinate/identity corruption."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def api():
    from src.data import qwen21_union_dataset as data
    return data


def image(path, color=10, size=(19, 31)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (color, color // 2, 0)).save(path)
    return path


def record(tmp_path, index, dataset="iper", group="g", split="train", frame=None):
    source = image(tmp_path / f"source{index}.png", index + 1)
    target = image(tmp_path / f"target{index}.png", index + 50)
    control = image(tmp_path / f"pose{index}.png", 0)
    return {"sample_id": f"{dataset}_{split}_{index}", "dataset": dataset,
            "split": split, "group_id": group, "sequence_id": group,
            "source_rgb": source, "target_rgb": target, "control_image": control,
            "target_frame": index * 8 if frame is None else frame,
            "source_frame": -100, "source_kind": "standard" if index % 2 else "motion",
            "target_keypoints": (), "target_masks": (), "source_masks": (),
            "person_order": (0,), "provenance": "fixture", "metadata": {}}


def test_sample_id_uses_paths_camera_and_person_not_numeric_pair_id(tmp_path):
    data = api()
    args = ("3dpw", "test", Path("a/source.jpg"), Path("a/target.jpg"))
    first = data.make_sample_id(*args, camera_id="1", person_order=(0,))
    assert first == data.make_sample_id(*args, camera_id="1", person_order=(0,))
    assert first != data.make_sample_id(*args, camera_id="2", person_order=(0,))
    assert first != data.make_sample_id(*args, camera_id="1", person_order=(1,))


def test_provider_mapping_uses_strict_manifests_and_no_trainval_mix(tmp_path):
    data = api()
    iper = tmp_path / "datasets/iPER/iper_sampled_6src64tgt"
    source = {"rgb_1024": "rgb/source.png", "source_type": "standard_view_diverse", "frame_index0": 0}
    target = {"rgb_1024": "rgb/target.png", "dwpose_512": "pose/t.png", "keypoints": "kp/t.npz", "frame_index0": 16}
    path = iper / "splits/dev_train_pairs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"appearance": "001_1", "source": source, "target": target}) + "\n")
    rows = data.read_records(tmp_path, datasets=("iper",), splits=("train",))
    assert len(rows) == 1
    row = rows[0]
    assert row["group_id"] == "001_1"
    assert row["target_rgb"] == iper / "001_1/rgb/target.png"
    assert row["source_kind"] == "standard"
    assert row["manifest"] == path


def test_provider_three_schemas_resolve_paths_and_real_contact_counts(tmp_path):
    data = api()
    paths = {
        "tiktok": "datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/train_pairs.jsonl",
        "hi4d": "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_train.jsonl",
    }
    fixtures = {
        "tiktok": {"sequence_id": "00001", "source_video_group": "video1", "source_rgb": "s.png", "target_rgb": "t.png", "target_dwpose": "p.png", "source_dwpose_json": "source.json", "source_person_mask": "source_mask.png", "target_dwpose_json": "p.json", "source_stem": "0001", "target_stem": "0017"},
        "hi4d": {"pair": "pair01", "camera_id": 4, "source_rgb": "s.png", "target_rgb": "t.png", "source_action": "hug01", "target_action": "hug02", "source_frame_id": 1, "target_frame_id": 17, "target_contact": True, "target_contact_vertex_count_total": 0, "source_person_mask_A": "a.png", "source_person_mask_B": "b.png", "target_dwpose_A": "a.npz", "target_dwpose_B": "b.npz"},
    }
    for name, path in paths.items():
        file = tmp_path / path
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps(fixtures[name]) + "\n")
    rows = data.read_records(tmp_path, datasets=("tiktok", "hi4d"), splits=("train",))
    assert rows[0]["group_id"] == "video1"
    assert rows[0]["source_rgb"] == tmp_path / "s.png"
    assert rows[0]["source_keypoints"] == (tmp_path / "source.json",)
    assert rows[0]["source_masks"] == (tmp_path / "source_mask.png",)
    assert rows[1]["contact_count"] == 0
    assert rows[1]["same_action"] is False
    root3 = tmp_path / "datasets/3dpw_single_final_v2"
    root3.mkdir(parents=True)
    (root3 / "test_pairs.csv").write_text("pair_id,sequence,pose_bin,source_rgb,target_rgb,target_dwpose_render,source_frame_index,target_frame_index\n0,seq,large,s.png,t.png,p.png,1,17\n")
    row = data.read_records(tmp_path, datasets=("3dpw",), splits=("test",))[0]
    assert row["source_rgb"] == root3 / "s.png"
    assert row["group_id"] == "seq" and row["pose_bin"] == "large"


def test_group_overlap_rejects_tiktok_sequence_aliases(tmp_path):
    data = api()
    rows = [record(tmp_path, 1, "tiktok", "same_video", "train"), record(tmp_path, 2, "tiktok", "same_video", "val")]
    with pytest.raises(ValueError, match="group.*overlap"):
        data.audit_records(rows)


def test_selection_deduplicates_target_path_content_and_temporal_neighbors(tmp_path):
    data = api()
    rows = [record(tmp_path, i, frame=i * 4) for i in range(7)]
    rows[1]["target_rgb"] = rows[0]["target_rgb"]
    rows[2]["target_rgb"].write_bytes(rows[0]["target_rgb"].read_bytes())
    chosen, report = data.select_pilot(rows, quotas={"iper": {"train": 3}})
    assert len(chosen) == 3
    assert len({data.sha256_file(row["target_rgb"]) for row in chosen}) == 3
    frames = sorted(row["target_frame"] for row in chosen)
    assert all(b - a >= 8 for a, b in zip(frames, frames[1:]))
    assert report["excluded"]["duplicate_target_path"] >= 1
    assert report["excluded"]["duplicate_target_content"] >= 1
    reverse, _ = data.select_pilot(list(reversed(rows)), quotas={"iper": {"train": 3}})
    assert [row["sample_id"] for row in chosen] == [row["sample_id"] for row in reverse]


def test_selection_reports_real_deficit_without_relaxing_spacing(tmp_path):
    data = api()
    rows = [record(tmp_path, i, frame=i) for i in range(4)]
    with pytest.raises(ValueError, match="insufficient.*iper.*train"):
        data.select_pilot(rows, quotas={"iper": {"train": 2}})


def test_group_round_robin_and_standard_motion_are_covered(tmp_path):
    data = api()
    rows = [record(tmp_path, i, group="a" if i < 4 else "b") for i in range(8)]
    chosen, _ = data.select_pilot(rows, quotas={"iper": {"train": 4}})
    assert {row["group_id"] for row in chosen} == {"a", "b"}
    assert {row["source_kind"] for row in chosen} == {"standard", "motion"}


def test_canvas_coordinates_use_rounded_resize_and_keep_invalid_requests(tmp_path):
    data = api()
    from src.pose_control.qwen21_union.contracts import CanvasTransform
    transform = CanvasTransform.from_size((19, 31))
    assert transform.resized_wh == (314, 512)
    assert transform.offset_xy == (99, 0)
    points = np.array([[18., 30.], [-1., 4.], [9., 15.]])
    result = data.transform_keypoints(points, np.array([.9, .9, .1]), transform)
    assert result["xy_canvas"][0] == pytest.approx([18 * 314 / 19 + 99, 30 * 512 / 31])
    assert result["requested"] == [True, True, False]
    assert result["in_frame"] == [True, False, True]
    assert result["valid"] == [True, False, False]
    assert result["xy_canvas"][1][0] < 99


def test_prepare_iper_separates_target_and_does_not_insert_second_neck(tmp_path):
    data = api()
    row = record(tmp_path, 1)
    kp = tmp_path / "target.npz"
    points = np.full((134, 2), 9., np.float32)
    np.savez(kp, keypoints_134_px=points, scores_134=np.ones(134), image_width=19, image_height=31)
    row["target_keypoints"] = (kp,)
    condition, supervision = data.prepare_record(row, tmp_path / "work")
    assert Image.open(condition.source_rgb).size == (512, 512)
    assert Image.open(condition.control_image).size == (512, 512)
    assert len(condition.reference_pose_images) == 1
    assert supervision.target_rgb == row["target_rgb"]
    payload = condition.to_dict()
    assert "target_rgb" not in payload and "target_latent" not in payload
    points = json.loads(supervision.requested_keypoints[0].read_text())
    assert len(points["xy_canvas"]) == 134
    assert points["score_threshold"] == .25
    with pytest.raises(FileExistsError):
        data.prepare_record(row, tmp_path / "work")


def test_tiktok_normalized_points_use_image_metadata(tmp_path):
    data = api()
    row = record(tmp_path, 1, dataset="tiktok")
    kp = tmp_path / "pose.json"
    kp.write_text(json.dumps({"keypoints_norm": [[.5, .5]] * 134, "scores": [1.] * 134, "image_width": 19, "image_height": 31}))
    row["target_keypoints"] = (kp,)
    _, supervision = data.prepare_record(row, tmp_path / "work")
    points = json.loads(supervision.requested_keypoints[0].read_text())
    assert points["xy_native"][0] == [9.5, 15.5]
    assert points["xy_canvas"][0] == pytest.approx([256, 256])


def test_hi4d_order_is_source_mask_order_and_filters_origin_artifacts(tmp_path):
    data = api()
    row = record(tmp_path, 1, dataset="hi4d")
    masks = []
    kps = []
    for index, x in enumerate((15, 3)):
        mask = Image.new("L", (19, 31), 0)
        mask.putpixel((x, 8), 255)
        path = tmp_path / f"mask{index}.png"
        mask.save(path)
        masks.append(path)
        kp = tmp_path / f"person{index}.npz"
        points = np.full((134, 2), [x, 20.], dtype=np.float32)
        scores = np.ones(134)
        points[0] = [0, 0]
        scores[0] = .1
        np.savez(kp, xy_native=points, scores=scores)
        kps.append(kp)
    row.update(source_masks=tuple(masks), target_keypoints=tuple(kps), person_order=(0, 1))
    condition, supervision = data.prepare_record(row, tmp_path / "work")
    assert condition.person_order == (1, 0)
    assert supervision.person_order == (1, 0)
    assert len(condition.reference_pose_images) == 2
    assert np.asarray(Image.open(condition.control_image))[0, 0].sum() == 0
    first = json.loads(supervision.requested_keypoints[0].read_text())
    assert first["person_index"] == 1 and first["valid"][0] is False


def test_inference_rejects_supervision_and_training_joins_by_id(tmp_path):
    data = api()
    records = [data.prepare_record(record(tmp_path, i), tmp_path / "work") for i in (1, 2)]
    conditions = tmp_path / "conditions.jsonl"
    supervision = tmp_path / "supervision.jsonl"
    conditions.write_text("\n".join(json.dumps(c.to_dict()) for c, _ in records))
    supervision.write_text("\n".join(json.dumps(s.to_dict()) for _, s in reversed(records)))
    inference = data.UnionInferenceDataset(conditions)
    assert inference[0].sample_id == records[0][0].sample_id
    training = data.UnionTrainingDataset(conditions, supervision)
    assert training[0]["condition"].sample_id == training[0]["supervision"].sample_id
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**records[0][0].to_dict(), "target_rgb": "secret.jpg"}))
    with pytest.raises(ValueError, match="supervision"):
        data.UnionInferenceDataset(bad)


def test_training_refuses_val_supervision(tmp_path):
    data = api()
    condition, supervision = data.prepare_record(record(tmp_path, 1, split="val"), tmp_path / "work")
    with pytest.raises(ValueError, match="train"):
        data.UnionTrainingDataset([condition], [supervision])


def test_content_dedup_ignores_png_metadata(tmp_path):
    data = api()
    first, second = record(tmp_path, 1), record(tmp_path, 2)
    info = PngInfo()
    info.add_text("description", "same pixels, different bytes")
    with Image.open(first["target_rgb"]) as original:
        original.save(second["target_rgb"], pnginfo=info)
    assert data.sha256_file(first["target_rgb"]) != data.sha256_file(second["target_rgb"])
    with pytest.raises(ValueError, match="insufficient"):
        data.select_pilot([first, second], quotas={"iper": {"train": 2}})


def test_condition_preparation_refuses_source_dataset_workspace(tmp_path):
    data = api()
    row = record(tmp_path, 1)
    with pytest.raises(ValueError, match="protected"):
        data.prepare_record(row, tmp_path / "datasets/new_experiment")


def test_inference_does_not_open_any_image(tmp_path, monkeypatch):
    data = api()
    condition, _ = data.prepare_record(record(tmp_path, 1), tmp_path / "work")
    def forbidden(*args, **kwargs):
        raise AssertionError("inference manifest reader opened an image")
    monkeypatch.setattr(Image, "open", forbidden)
    dataset = data.UnionInferenceDataset([condition.to_dict()])
    assert dataset[0].sample_id == condition.sample_id


def test_build_writes_separate_locked_manifests_and_overfit_ids(tmp_path, monkeypatch):
    data = api()
    rows = []
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for index in range(count):
            row = record(tmp_path, len(rows) + 1, split=split, group=split)
            row["manifest"] = tmp_path / f"{split}.jsonl"
            row["manifest"].write_text("{}\n")
            rows.append(row)
    monkeypatch.setattr(data, "read_records", lambda root: rows)
    monkeypatch.setattr(data, "QUOTAS", {"iper": {"train": 2, "val": 1, "test": 1}})
    work = tmp_path / "work"
    lock = data.build_pilot(tmp_path, work)
    assert lock["counts"] == {"train": 2, "val": 1, "test": 1}
    assert len(lock["overfit6_sample_ids"]) == 2
    assert set(lock["selected_manifest_sha256"]) == {f"{split}_{kind}.jsonl" for split in ("train", "val", "test", "overfit6") for kind in ("conditions", "supervision")}
    registry = json.loads((work / "data/prepared_assets_sha256.json").read_text())
    assert registry["lock_sha256"] == data.sha256_file(work / "data/lock.json")
    assert registry["condition_count"] == 4
    assert len(registry["asset_sha256"]) == 8
    for line in (work / "data/train_conditions.jsonl").read_text().splitlines():
        assert "target_rgb" not in json.loads(line)
    with pytest.raises(FileExistsError):
        data.build_pilot(tmp_path, work)


def test_provider_refuses_embedded_split_mismatch(tmp_path):
    data = api()
    path = tmp_path / "datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/train_pairs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"split": "test", "sequence_id": "01", "source_video_group": "video", "source_rgb": "s", "target_rgb": "t", "target_dwpose": "p", "target_dwpose_json": "k", "source_stem": "1", "target_stem": "17"}))
    with pytest.raises(ValueError, match="split mismatch"):
        data.read_records(tmp_path, datasets=("tiktok",), splits=("train",))


def test_preparation_rejects_sample_id_path_escape(tmp_path):
    data = api()
    row = record(tmp_path, 1)
    row["sample_id"] = "../../outside"
    with pytest.raises(ValueError, match="sample_id"):
        data.prepare_record(row, tmp_path / "work")

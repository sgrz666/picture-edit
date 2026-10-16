"""Locked input integrity remains independent of re-encoded caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n")


def workspace(tmp_path):
    data = tmp_path / "data"
    assets = data / "assets/one"
    originals = tmp_path / "originals"
    assets.mkdir(parents=True)
    originals.mkdir()
    paths = {}
    for name, root, color in (("source", assets, "red"), ("control", assets, "black"),
                              ("original_source", originals, "red"), ("target", originals, "green")):
        path = root / f"{name}.png"
        Image.new("RGB", (4, 4), color).save(path)
        paths[name] = path
    points = assets / "requested.json"
    write_json(points, {"points": [[0, 0]], "scores": [1.]})
    raw = originals / "pairs.jsonl"
    raw.write_text('{"split":"train"}\n')
    condition = {"sample_id": "one", "split": "train", "source_rgb": str(paths["source"]),
                 "reference_pose_images": [str(paths["control"])], "control_image": str(paths["control"])}
    truth = {"sample_id": "one", "target_rgb": str(paths["target"]), "requested_keypoints": [str(points)]}
    for split in ("train", "val", "test", "overfit6"):
        for kind, row in (("conditions", condition), ("supervision", truth)):
            (data / f"{split}_{kind}.jsonl").write_text(json.dumps(row) + "\n" if split in ("train", "overfit6") else "")
    lock = {"protocol": "pilot_b_v1", "counts": {"train": 1, "val": 0, "test": 0},
            "selected_manifest_sha256": {p.name: digest(p) for p in data.glob("*.jsonl")},
            "manifest_sha256": {str(raw): digest(raw)},
            "required_asset_sha256": {str(paths[key]): digest(paths[key]) for key in ("original_source", "target")}}
    write_json(data / "lock.json", lock)
    registry = {"protocol": "pilot_b_v1", "lock_sha256": digest(data / "lock.json"),
                "condition_count": 1, "asset_count": 3,
                "asset_sha256": {str(p): digest(p) for p in (paths["source"], paths["control"], points)}}
    write_json(data / "prepared_assets_sha256.json", registry)
    return tmp_path, paths


def module():
    from src.pose_control.qwen21_union import integrity
    integrity.verify_workspace_inputs.cache_clear()
    return integrity


def test_intact_workspace_returns_json_report_without_mutations(tmp_path):
    api = module()
    root, _ = workspace(tmp_path)
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    report = api.verify_workspace_inputs(root)
    assert report["verified"] is True
    assert report["lock_sha256"] == digest(root / "data/lock.json")
    assert report["prepared_registry_sha256"] == digest(root / "data/prepared_assets_sha256.json")
    assert report["prepared_asset_count"] == 3 and report["original_asset_count"] == 2
    assert report["condition_count"] == 1
    json.dumps(report)
    assert before == {str(path): digest(path) for path in root.rglob("*") if path.is_file()}


def test_tampered_prepared_png_fails_even_after_cache_reencoded(tmp_path):
    api = module()
    root, paths = workspace(tmp_path)
    Image.new("RGB", (4, 4), "blue").save(paths["control"])
    (root / "cache").mkdir()
    (root / "cache/reencoded_condition.pt").write_bytes(b"cache reflects the edited control")
    with pytest.raises(ValueError, match="prepared asset.*checksum"):
        api.verify_workspace_inputs(root)


def test_changed_original_target_is_rejected(tmp_path):
    api = module()
    root, paths = workspace(tmp_path)
    Image.new("RGB", (4, 4), "blue").save(paths["target"])
    with pytest.raises(ValueError, match="original asset.*checksum"):
        api.verify_workspace_inputs(root)


def test_stale_registry_lock_binding_is_rejected(tmp_path):
    api = module()
    root, _ = workspace(tmp_path)
    path = root / "data/prepared_assets_sha256.json"
    registry = json.loads(path.read_text())
    registry["lock_sha256"] = "0" * 64
    write_json(path, registry)
    with pytest.raises(ValueError, match="registry.*lock"):
        api.verify_workspace_inputs(root)


def test_modified_selected_manifest_is_rejected(tmp_path):
    api = module()
    root, _ = workspace(tmp_path)
    with (root / "data/train_conditions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="selected manifest.*checksum"):
        api.verify_workspace_inputs(root)


def test_registry_cannot_omit_a_required_prepared_file(tmp_path):
    api = module()
    root, paths = workspace(tmp_path)
    path = root / "data/prepared_assets_sha256.json"
    registry = json.loads(path.read_text())
    registry["asset_sha256"].pop(str(paths["control"]))
    registry["asset_count"] = 2
    write_json(path, registry)
    with pytest.raises(ValueError, match="registry.*inventory"):
        api.verify_workspace_inputs(root)


def test_missing_original_asset_is_rejected(tmp_path):
    api = module()
    root, _ = workspace(tmp_path)
    lock_path = root / "data/lock.json"
    lock = json.loads(lock_path.read_text())
    lock["required_asset_sha256"][str(root / "absent.png")] = "0" * 64
    write_json(lock_path, lock)
    registry_path = root / "data/prepared_assets_sha256.json"
    registry = json.loads(registry_path.read_text())
    registry["lock_sha256"] = digest(lock_path)
    write_json(registry_path, registry)
    with pytest.raises(ValueError, match="original asset.*missing"):
        api.verify_workspace_inputs(root)


def test_cache_is_per_process_resettable_and_returns_defensive_report(tmp_path):
    api = module()
    root, _ = workspace(tmp_path)
    first = api.verify_workspace_inputs(root)
    first["verified"] = False
    assert api.verify_workspace_inputs(root)["verified"] is True
    assert api.verify_workspace_inputs.cache_info().hits == 1
    api.verify_workspace_inputs.cache_clear()
    assert api.verify_workspace_inputs.cache_info().currsize == 0


def test_integrity_import_has_no_torch_or_image_runtime_dependency():
    check = "import sys; from src.pose_control.qwen21_union.integrity import verify_workspace_inputs; assert not {'torch','numpy','PIL'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

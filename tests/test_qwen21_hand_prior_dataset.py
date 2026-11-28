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

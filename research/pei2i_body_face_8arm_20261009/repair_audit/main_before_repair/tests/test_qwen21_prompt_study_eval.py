from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image


@pytest.fixture
def evaluation():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prompt_study_eval", root / "scripts/eval/eval_qwen21_joint_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_generation_keeps_A_and_original_target_size(evaluation):
    a = Image.new("RGB", (608, 1088))
    args = SimpleNamespace(
        resolution="native",
        align_mode="native",
        steps=30,
        use_kv_cache=True,
    )
    kw = evaluation.generation_arguments(
        a,
        args,
        prompt="Raise both arms",
        generator=None,
        target_size=(736, 1024),
    )
    assert kw["image"] == [a]
    assert kw["prompt"] == "Raise both arms"
    assert (kw["width"], kw["height"]) == (736, 1024)
    assert not {"latents", "target", "gt_image"} & kw.keys()


def test_eval_arguments_study_mode(evaluation, monkeypatch):
    import sys

    argv = [
        "eval.py",
        "--checkpoint", "checkpoint.pt",
        "--pairs", "pairs.jsonl",
        "--features", "features",
        "--cache", "cache",
        "--output", "output",
        "--resolution", "native",
        "--study-lock", "lock.json",
        "--arm", "head_pb",
        "--prompt-manifest", "prompts.jsonl",
        "--study-step", "128",
        "--study-split", "val",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = evaluation.arguments()
    assert args.study_lock == Path("lock.json")
    assert args.arm == "head_pb"
    assert args.study_step == 128
    assert args.study_split == "val"
    assert args.label == "head_pb_step_128"


def test_crop_normalized_native_resolutions(evaluation):
    import numpy as np

    # Test 604x1080
    img1 = Image.new("RGB", (604, 1080), color=(100, 150, 200))
    box = [0.2, 0.1, 0.4, 0.3]
    cropped1, pixels1 = evaluation.crop_normalized(img1, box, side=128)
    assert cropped1 is not None
    assert cropped1.size == (128, 128)
    assert pixels1[0] == int(np.floor(0.2 * 604))
    assert pixels1[1] == int(np.floor(0.1 * 1080))
    assert pixels1[2] == int(np.ceil(0.4 * 604))
    assert pixels1[3] == int(np.ceil(0.3 * 1080))

    # Test 736x1024
    img2 = Image.new("RGB", (736, 1024), color=(50, 80, 110))
    cropped2, pixels2 = evaluation.crop_normalized(img2, box, side=128)
    assert cropped2 is not None
    assert pixels2[0] == int(np.floor(0.2 * 736))
    assert pixels2[1] == int(np.floor(0.1 * 1024))
    assert pixels2[2] == int(np.ceil(0.4 * 736))
    assert pixels2[3] == int(np.ceil(0.3 * 1024))


def test_compute_bg_l1_background_preservation(evaluation):
    src = Image.new("RGB", (100, 100), color=(200, 200, 200))
    # Identical output: bg L1 is 0.0
    mask_a = Image.new("L", (100, 100), color=0)  # no foreground
    l1, reason = evaluation.compute_bg_l1(src, src, mask_a=mask_a)
    assert reason is None
    assert l1 == pytest.approx(0.0)

    # Different background
    gen = Image.new("RGB", (100, 100), color=(100, 100, 100))
    l1_diff, reason = evaluation.compute_bg_l1(src, gen, mask_a=mask_a)
    assert reason is None
    assert l1_diff == pytest.approx(100.0 / 255.0, abs=1e-3)


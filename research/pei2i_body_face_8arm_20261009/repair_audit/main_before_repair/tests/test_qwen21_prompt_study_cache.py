from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest

from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache
from scripts.precompute_qwen21_prompt_study_cache import cache_split


def test_source_cache_is_sensitive_to_compiled_prompt(tmp_path: Path):
    cache = Qwen21IPERCache(tmp_path, resolution="native", preprocessing="native_v1")
    a = Image.new("RGB", (608, 1088), "red")
    # Different prompts produce different source cache paths
    p1 = cache._path(a, "source", prompt="Raise arms")
    p2 = cache._path(a, "source", prompt="Lower arms")
    assert p1 != p2
    # Target cache path does not depend on prompt
    t1 = cache._path(a, "target", prompt="Raise arms")
    t2 = cache._path(a, "target", prompt="Lower arms")
    assert t1 == t2


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FakeBank:
    def __init__(self, prompts):
        self.prompts = prompts

    def prompt_for(self, pair_id):
        return self.prompts[pair_id]


class FakeCache:
    def __init__(self):
        self.sources = []
        self.targets = []

    def source(self, pipe, img, prompt=None):
        self.sources.append((img, prompt))

    def target(self, pipe, img):
        self.targets.append(img)


import torch


def test_cache_split_never_calls_target_for_val_or_test():
    img_tensor = torch.zeros(3, 100, 100)
    samples = [
        {"pair_metadata": {"pair_id": "p1"}, "src_image": img_tensor, "tgt_image": img_tensor},
        {"pair_metadata": {"pair_id": "p2"}, "src_image": img_tensor, "tgt_image": img_tensor},
    ]
    dataset = FakeDataset(samples)
    bank = FakeBank({"p1": "Prompt 1", "p2": "Prompt 2"})

    # Val split: target must never be called!
    cache_val = FakeCache()
    counts_val = cache_split(None, cache_val, dataset, bank, split="val")
    assert counts_val["source_calls"] == 2
    assert counts_val["target_calls"] == 0
    assert len(cache_val.targets) == 0
    assert [p for _, p in cache_val.sources] == ["Prompt 1", "Prompt 2"]

    # Train split: target is called
    cache_train = FakeCache()
    counts_train = cache_split(None, cache_train, dataset, bank, split="train")
    assert counts_train["source_calls"] == 2
    assert counts_train["target_calls"] == 2
    assert len(cache_train.targets) == 2


def test_precompute_cache_cli_execution(tmp_path: Path, monkeypatch, capsys):
    import sys
    import scripts.precompute_qwen21_prompt_study_cache as cache_cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data").mkdir()
    (run_dir / "prompts").mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["cache.py", "--run", str(run_dir), "--protocols", "p0", "--splits", "train"],
    )
    cache_cli.main()
    out = capsys.readouterr().out
    assert "Precompute cache finished:" in out


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
    counts_val = cache_split(object(), cache_val, dataset, bank, split="val")
    assert counts_val["source_calls"] == 2
    assert counts_val["target_calls"] == 0
    assert len(cache_val.targets) == 0
    assert [p for _, p in cache_val.sources] == ["Prompt 1", "Prompt 2"]

    # Train split: target is called
    cache_train = FakeCache()
    counts_train = cache_split(object(), cache_train, dataset, bank, split="train")
    assert counts_train["source_calls"] == 2
    assert counts_train["target_calls"] == 2
    assert len(cache_train.targets) == 2


def test_precompute_cache_cli_rejects_missing_split(tmp_path: Path, monkeypatch, capsys):
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
    with pytest.raises(FileNotFoundError, match="split"):
        cache_cli.main()
    out = capsys.readouterr().out
    assert "Precompute cache finished:" not in out


def test_cache_split_requires_loaded_pipeline():
    with pytest.raises(ValueError, match="pipeline"):
        cache_split(None, FakeCache(), FakeDataset([]), FakeBank({}), split="val")


def test_pipeline_load_failure_is_fatal(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import scripts.precompute_qwen21_prompt_study_cache as cli
    def broken(*args, **kwargs):
        raise RuntimeError("Qwen model load failed")
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        QwenImage21Pipeline=SimpleNamespace(from_pretrained=broken)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="Qwen model load failed"):
        cli.load_pipeline(tmp_path)


def test_cache_missing_cuda_fails_before_pipeline_allocation(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import scripts.precompute_qwen21_prompt_study_cache as cli
    def forbidden(*args, **kwargs):
        pytest.fail("Pipeline allocated before CUDA prerequisite was checked")
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        QwenImage21Pipeline=SimpleNamespace(from_pretrained=forbidden)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        cli.load_pipeline(tmp_path)


@pytest.mark.parametrize("module_name", ["scripts.precompute_qwen21_prompt_study_cache", "scripts.precompute_qwen21_pei2i_prompts"])
def test_direct_test_cli_requires_all_eight_arms(tmp_path, monkeypatch, module_name):
    import sys
    import importlib
    cli = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", ["precompute.py", "--run", str(tmp_path), "--splits", "test", "--protocols", "p0"])
    with pytest.raises(RuntimeError, match="eight-arm"):
        cli.main()


def test_cache_manifest_counts_only_actual_calls(tmp_path, monkeypatch):
    import json
    import sys
    import scripts.precompute_qwen21_prompt_study_cache as cli
    import src.pose_control.qwen21_full.execution as execution
    from src.pose_control.qwen21_full.prompt_study.prompts import make_p0_record
    data = tmp_path / "data"
    data.mkdir()
    (data / "val_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    prompt = tmp_path / "prompts/p0/val.jsonl"
    prompt.parent.mkdir(parents=True)
    prompt.write_text(json.dumps(make_p0_record("p1", "val")) + "\n")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    samples = [{"pair_id":"p1", "pair_metadata":{"pair_id":"p1"},
                "src_image":torch.zeros(3, 16, 16), "tgt_image":torch.zeros(3, 16, 16)}]
    monkeypatch.setattr(execution, "QwenFullDataset", lambda *args, **kwargs: FakeDataset(samples))
    monkeypatch.setattr(cli, "pair_metadata", lambda row: {"pair_id": row["pair_id"]})
    monkeypatch.setattr(cli, "load_pipeline", lambda path: object())
    class FileCache(FakeCache):
        def __init__(self, path, **kwargs):
            super().__init__()
            self.path = path
        def _path(self, image, kind, **kwargs):
            return self.path / (kind + ".pt")
        def source(self, pipe, img, prompt=None):
            super().source(pipe, img, prompt)
            torch.save({"prompt_embeds":torch.zeros(1, 2, 4), "prompt_mask":torch.ones(1, 2, dtype=torch.bool),
                        "image_mask":torch.ones(1, 2, dtype=torch.bool), "reference_latent":torch.zeros(1, 1, 64),
                        "token_hw":(1, 1)}, self._path(img, "source"))
        def target(self, pipe, img):
            super().target(pipe, img)
            torch.save(torch.zeros(1, 1, 64), self._path(img, "target"))
    monkeypatch.setattr(cli, "Qwen21IPERCache", FileCache)
    monkeypatch.setattr(sys, "argv", ["cache.py", "--run",str(tmp_path),"--splits","val","--protocols","p0", "--model",str(model_dir)])
    cli.main()
    assert (tmp_path / "cache/p0/manifest.json").is_file(), "real cache evidence is required"
    manifest = json.loads((tmp_path / "cache/p0/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["splits"]["val"]["source_calls"] == 1
    assert manifest["splits"]["val"]["target_calls"] == 0
    assert manifest["splits"]["val"]["prompt_manifest_sha256"] == cli.sha256_file(prompt)
    assert hasattr(cli, "validate_cache_manifest"), "selfcheck requires a cache provenance validator"
    cli.validate_cache_manifest(tmp_path, "p0", ("val",))
    source = tmp_path / manifest["splits"]["val"]["files"][0]["source"]["path"]
    original = source.read_bytes()
    source.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="cache|Cache"):
        cli.validate_cache_manifest(tmp_path, "p0", ("val",))
    source.write_bytes(original)
    torch.save({"prompt_embeds": torch.zeros(1, 2, 4)}, source)
    manifest["splits"]["val"]["files"][0]["source"]["sha256"] = cli.sha256_file(source)
    cli.atomic_json(tmp_path / "cache/p0/manifest.json", manifest)
    with pytest.raises(ValueError, match="cache|Cache"):
        cli.validate_cache_manifest(tmp_path, "p0", ("val",))
    source.write_bytes(original)
    manifest["splits"]["val"]["files"][0]["source"]["sha256"] = cli.sha256_file(source)
    cli.atomic_json(tmp_path / "cache/p0/manifest.json", manifest)
    (data / "train_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    (prompt.parent / "train.jsonl").write_text(json.dumps(make_p0_record("p1", "train")) + "\n")
    monkeypatch.setattr(sys, "argv", ["cache.py", "--run",str(tmp_path),"--splits","train","--protocols","p0", "--model",str(model_dir)])
    cli.main()
    manifest = json.loads((tmp_path / "cache/p0/manifest.json").read_text())
    assert manifest["splits"]["val"]["prompt_manifest_sha256"] == cli.sha256_file(prompt)
    assert manifest["splits"]["train"]["target_calls"] == 1
    cli.validate_cache_manifest(tmp_path, "p0", ("val", "train"))
    (model_dir / "weights.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="model provenance"):
        cli.main()
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    def broken_source(*args, **kwargs):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr(FileCache, "source", broken_source)
    with pytest.raises(RuntimeError, match="encoder failed"):
        cli.main()
    failed = json.loads((tmp_path / "cache/p0/manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["splits"]["train"]["status"] == "failed"
    assert failed["splits"]["train"]["source_calls"] == 0
    assert failed["splits"]["val"]["status"] == "complete"


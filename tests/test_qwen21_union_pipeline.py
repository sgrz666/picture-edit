"""The CPU import path must not import upstream models or optional audio."""
import subprocess
import sys
import os
from pathlib import Path

import pytest


def test_pipeline_module_is_lazy():
    code = "import sys; import src.pose_control.qwen21_union.pipeline; assert 'videox_fun' not in sys.modules; assert 'diffusers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_invalid_build_request_fails_before_loading_models():
    from src.pose_control.qwen21_union.pipeline import build_pipeline

    with pytest.raises(ValueError, match="kind"):
        build_pipeline(kind="unknown")
    with pytest.raises(ValueError, match="scope"):
        build_pipeline(kind="union", scope="invalid")


def test_reference_options_accept_guidance_alias_and_reject_scope_drift():
    import types
    from src.pose_control.qwen21_union.pipeline import normalize_inference_options

    transformer = types.SimpleNamespace(injection_scope="target")
    assert normalize_inference_options(1.0, 2.0, "target", transformer) == 2.0
    assert normalize_inference_options(1.5, None, None, transformer) == 1.5
    with pytest.raises(ValueError, match="scope"):
        normalize_inference_options(1.0, None, "joint", transformer)
    with pytest.raises(ValueError, match="guidance"):
        normalize_inference_options(2.0, 3.0, "target", transformer)


def test_source_commit_is_verified_before_import(tmp_path):
    from src.pose_control.qwen21_union.pipeline import verify_source

    with pytest.raises(RuntimeError, match="fixed commit"):
        verify_source(tmp_path)


def test_model_fingerprint_hashes_actual_weights_and_reuses_unchanged_reads(tmp_path, monkeypatch):
    from src.pose_control.qwen21_union import pipeline

    for component in ("vae", "text_encoder", "processor", "transformer", "scheduler"):
        (tmp_path / component).mkdir()
        (tmp_path / component / "config.json").write_text('{"version":1}')
    (tmp_path / "vae/model.safetensors").write_bytes(b"actual vae weights")
    (tmp_path / "transformer/model.safetensors").write_bytes(b"actual transformer weights")
    reads = []
    original = pipeline.sha256_file

    def counted(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(pipeline, "sha256_file", counted)
    first = pipeline.model_asset_signature(tmp_path)
    read_count = len(reads)
    assert read_count == 7
    assert pipeline.model_asset_signature(tmp_path) == first
    assert len(reads) == read_count
    (tmp_path / "transformer/model.safetensors").write_bytes(b"changed transformer weights")
    changed = pipeline.model_asset_signature(tmp_path)
    assert changed["fingerprint"] != first["fingerprint"]
    assert len(reads) == read_count + 1
    encoder = pipeline.encoder_asset_signature(tmp_path)
    assert set(encoder["components"]) == {"vae", "text_encoder", "processor"}
    assert encoder["fingerprint"] != first["fingerprint"]


def test_cached_pipeline_sets_up_transformer_before_offload_without_vlm(tmp_path, monkeypatch):
    import types
    import torch
    from src.pose_control.qwen21_union import pipeline

    events = []

    class Model(torch.nn.Linear):
        @classmethod
        def from_pretrained(cls, path, subfolder, **kwargs):
            assert subfolder != "text_encoder", "cached inference must not load VLM"
            events.append(subfolder)
            return cls(1, 1)

    class Component:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Pipe:
        def __init__(self, **components):
            vars(self).update(components)

    def setup(model):
        assert all(not p.requires_grad for p in model.parameters())
        model.lora = torch.nn.Parameter(torch.ones(1))
        events.append("setup")
        return {"installed": True}

    def offload(pipe, *args, **kwargs):
        assert pipe.transformer.lora.requires_grad
        assert pipe.text_encoder is None
        assert events[-1] == "setup"
        events.append("offload")

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.FlowMatchEulerDiscreteScheduler = Component
    fake_diffusers.QwenImage21Transformer2DModel = Model
    fake_diffusers.QwenImage21Pipeline = Pipe
    fake_diffusers.__version__ = "test"
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    models = types.SimpleNamespace(AutoencoderKLQwenImage21=Model, Qwen3VLForConditionalGeneration=Model,
                                   Qwen3VLProcessor=Component)
    monkeypatch.setattr(pipeline, "_upstream", lambda root: (tmp_path, models, None,
                                                          types.SimpleNamespace(apply_gpu_memory_mode=offload)))
    pipe, audit = pipeline.build_pipeline("native", device="cuda", base_model=tmp_path,
                                          cached=True, transformer_setup=setup)
    assert events[-2:] == ["setup", "offload"]
    assert audit["transformer_setup"] == {"installed": True}
    assert audit["backend"] == "diffusers-native"


@pytest.mark.skipif(not Path(os.environ.get("QWEN21_UNION_SOURCE", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/source/VideoX-Fun")).is_dir(),
                    reason="pinned existing remote assets are not mounted")
def test_pinned_models_import_and_native_cache_contract():
    import inspect
    import diffusers
    from src.pose_control.qwen21_union.pipeline import _upstream, scoped_transformer_class

    root, models, pipelines, _ = _upstream()
    assert str(root).endswith("VideoX-Fun")
    cls = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    assert cls.__mro__[1] is models.QwenImage21ControlTransformer2DModel
    assert hasattr(pipelines.QwenImage21ControlPipeline, "encode_prompt")
    native = inspect.signature(diffusers.QwenImage21Transformer2DModel.forward).parameters
    assert {"hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "img_shapes", "img_mask"} <= set(native)
    assert "librosa" not in sys.modules

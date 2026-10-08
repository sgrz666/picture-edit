"""CPU checks for frozen BFS loading, using real safetensors and torch layers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn


def _api():
    path = Path(__file__).resolve().parents[1] / "src/pose_control/qwen21_full/frozen_bfs_lora.py"
    if not path.exists():
        pytest.fail("frozen BFS loading API has not been implemented")
    spec = importlib.util.spec_from_file_location("frozen_bfs_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LoraLinear(nn.Module):
    """Small differentiable PEFT-shaped layer; never loads a diffusion model."""

    def __init__(self, base, name, a, b, *, base_rounding=False):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({name: nn.Linear(a.shape[1], a.shape[0], bias=False)})
        self.lora_B = nn.ModuleDict({name: nn.Linear(b.shape[1], b.shape[0], bias=False)})
        self.scaling = {name: 1.0}
        self.active_adapters = [name]
        self.disable_adapters = False
        self.merged_adapters = []
        with torch.no_grad():
            for target, value in ((self.lora_A[name].weight, a), (self.lora_B[name].weight, b)):
                if base_rounding:
                    value = value.to(base.weight.dtype)
                target.copy_(value)

    def get_base_layer(self):
        return self.base_layer

    def forward(self, value):
        out = nn.functional.linear(value, self.base_layer.weight.float())
        if not self.disable_adapters:
            for name in self.active_adapters:
                out = out + self.scaling[name] * self.lora_B[name](self.lora_A[name](value))
        return out


class _Transformer(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_q = nn.Linear(4, 3, bias=False)
        block.img_mlp = nn.Module()
        block.img_mlp.gate_layer = nn.Linear(4, 6, bias=False)
        block.img_mlp.proj = nn.Linear(4, 6, bias=False)
        block.img_mlp.out = nn.Linear(6, 4, bias=False)
        self.transformer_blocks = nn.ModuleList([block])
        self.to(dtype=dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, value):
        return self.transformer_blocks[0].attn.to_q(value)


class _Pipeline:
    def __init__(self, dtype=torch.float32, *, glitch=None, base_rounding=False):
        self.transformer = _Transformer(dtype)
        self.text_encoder = nn.Linear(4, 4)
        self.vae = nn.Linear(4, 4)
        self.glitch = glitch
        self.base_rounding = base_rounding
        self.loaded_state = None
        self.load_calls = 0

    def load_lora_weights(self, state, adapter_name):
        self.load_calls += 1
        self.loaded_state = {key: value.clone() for key, value in state.items()}
        modules = sorted({key.removeprefix("transformer.").split(".lora_")[0] for key in state})
        for index, name in enumerate(modules):
            if self.glitch == "missing" and index == 0:
                continue
            base = self.transformer.get_submodule(name)
            a = state[f"transformer.{name}.lora_A.weight"]
            b = state[f"transformer.{name}.lora_B.weight"]
            layer = _LoraLinear(base, adapter_name, a, b, base_rounding=self.base_rounding)
            if index == 0:
                if self.glitch == "value":
                    with torch.no_grad():
                        layer.lora_A[adapter_name].weight.add_(0.1)
                if self.glitch == "shape":
                    layer.lora_B[adapter_name] = nn.Linear(b.shape[1], b.shape[0] + 1, bias=False)
                if self.glitch == "disabled":
                    layer.disable_adapters = True
                if self.glitch == "merged":
                    layer.merged_adapters = [adapter_name]
                if self.glitch == "bias":
                    layer.lora_B[adapter_name].bias = nn.Parameter(torch.ones(b.shape[0]))
            parent_name, child = name.rsplit(".", 1)
            setattr(self.transformer.get_submodule(parent_name), child, layer)
        if self.glitch == "extra_adapter":
            mlp = self.transformer.transformer_blocks[0].img_mlp
            mlp.out = _LoraLinear(mlp.out, "rogue", torch.ones(2, 6), torch.ones(4, 2))

    def set_adapters(self, names, adapter_weights):
        for layer in self.transformer.modules():
            if isinstance(layer, _LoraLinear):
                layer.active_adapters = list(names)
                layer.scaling[names[0]] = adapter_weights[0] + (0.25 if self.glitch == "scale" else 0)

    def fuse_lora(self, *args, **kwargs):
        pytest.fail("frozen BFS must remain unfused")


def _raw_state():
    prefix = "diffusion_model.transformer_blocks.0."
    return {
        prefix + "attn.to_q.lora_A.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11 + 0.013,
        prefix + "attn.to_q.lora_B.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2) / 13 + 0.019,
        prefix + "img_mlp.gate_up.lora_A.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 17 + 0.021,
        prefix + "img_mlp.gate_up.lora_B.weight": torch.arange(24, dtype=torch.float32).reshape(12, 2) / 19 + 0.027,
    }


def _checkpoint(tmp_path, state=None):
    path = tmp_path / "bfs.safetensors"
    save_file(_raw_state() if state is None else state, str(path))
    return path


def test_plain_qwen_is_frozen_without_freezing_adapter_or_pipeline_neighbors():
    api = _api()
    pipe = _Pipeline()
    adapter = nn.Linear(4, 4)
    report = api.load_frozen_bfs(pipe, None)

    assert report["variant"] == "qwen"
    assert report["file_sha256"] is None
    assert report["tensors_checked"] == 0
    assert report["freeze"]["all_parameters_frozen"]
    assert report["freeze"]["trainable_parameter_names"] == []
    assert all(not p.requires_grad for p in pipe.transformer.parameters())
    assert all(p.requires_grad for p in adapter.parameters())
    assert all(p.requires_grad for p in pipe.text_encoder.parameters())
    assert all(p.requires_grad for p in pipe.vae.parameters())
    assert not pipe.transformer.training
    assert pipe.load_calls == 0
    pipe.transformer(adapter(torch.ones(2, 4))).sum().backward()
    assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad).all()
    assert adapter.weight.grad.abs().sum() > 0
    json.dumps(report, allow_nan=False)


def test_real_fused_gate_first_conversion_loading_and_report(tmp_path):
    api = _api()
    path = _checkpoint(tmp_path)
    pipe = _Pipeline()
    report = api.load_frozen_bfs(pipe, path, adapter_name="frozen", strength=0.75)

    prefix = "transformer.transformer_blocks.0.img_mlp."
    raw = _raw_state()
    torch.testing.assert_close(pipe.loaded_state[prefix + "gate_layer.lora_A.weight"], raw["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"])
    torch.testing.assert_close(pipe.loaded_state[prefix + "proj.lora_A.weight"], pipe.loaded_state[prefix + "gate_layer.lora_A.weight"])
    torch.testing.assert_close(pipe.loaded_state[prefix + "gate_layer.lora_B.weight"], raw["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"][:6])
    torch.testing.assert_close(pipe.loaded_state[prefix + "proj.lora_B.weight"], raw["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"][6:])
    assert report["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["source_tensors"] == 4
    assert report["tensors_checked"] == 6
    assert report["modules_checked"] == 3
    assert report["gate_first"] is True
    assert report["module_scales"] == [0.75]
    assert report["all_values_match_after_dtype_conversion"]
    assert report["freeze"]["all_parameters_frozen"]
    assert report["loaded_lora_sha256"] == api.frozen_transformer_digest(pipe.transformer)
    assert all(not p.requires_grad for p in pipe.transformer.parameters())
    json.dumps(report, allow_nan=False)


def test_bf16_base_rounding_is_accepted_and_remains_differentiable(tmp_path):
    api = _api()
    pipe = _Pipeline(torch.bfloat16, base_rounding=True)
    report = api.load_frozen_bfs(pipe, _checkpoint(tmp_path))
    assert report["tensors_cast_through_base_dtype"] == 6
    adapter = nn.Linear(4, 4)
    pipe.transformer(adapter(torch.ones(1, 4))).square().mean().backward()
    assert adapter.weight.grad is not None and adapter.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in pipe.transformer.parameters())


@pytest.mark.parametrize("failure", ["unknown", "missing_pair", "rank", "fused_shape", "source_nonfinite", "collision"])
def test_invalid_source_mapping_is_rejected_before_pipeline_mutation(tmp_path, failure):
    api = _api()
    state = _raw_state()
    prefix = "diffusion_model.transformer_blocks.0."
    if failure == "unknown":
        state[prefix + "unknown.lora_A.weight"] = torch.ones(2, 4)
        state[prefix + "unknown.lora_B.weight"] = torch.ones(3, 2)
    elif failure == "missing_pair":
        state.pop(prefix + "attn.to_q.lora_B.weight")
    elif failure == "rank":
        state[prefix + "attn.to_q.lora_B.weight"] = torch.ones(3, 3)
    elif failure == "fused_shape":
        state[prefix + "img_mlp.gate_up.lora_B.weight"] = torch.ones(13, 2)
    elif failure == "source_nonfinite":
        state[prefix + "attn.to_q.lora_A.weight"][0, 0] = float("nan")
    elif failure == "collision":
        state[prefix + "img_mlp.proj.lora_A.weight"] = torch.ones(2, 4)
        state[prefix + "img_mlp.proj.lora_B.weight"] = torch.ones(6, 2)
    pipe = _Pipeline()
    with pytest.raises((ValueError, RuntimeError)):
        api.load_frozen_bfs(pipe, _checkpoint(tmp_path, state))
    assert pipe.load_calls == 0


@pytest.mark.parametrize("glitch", ["missing", "value", "shape", "scale", "disabled", "merged", "bias", "extra_adapter"])
def test_failed_actual_load_or_effective_scale_is_rejected(tmp_path, glitch):
    api = _api()
    pipe = _Pipeline(glitch=glitch)
    with pytest.raises(RuntimeError):
        api.load_frozen_bfs(pipe, _checkpoint(tmp_path))
    assert all(not p.requires_grad for p in pipe.transformer.parameters())


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1.0])
def test_invalid_strength_is_rejected_before_load(tmp_path, strength):
    api = _api()
    pipe = _Pipeline()
    with pytest.raises(ValueError):
        api.load_frozen_bfs(pipe, _checkpoint(tmp_path), strength=strength)
    assert pipe.load_calls == 0


def test_digest_detects_lora_changes_and_optional_base_changes(tmp_path):
    api = _api()
    pipe = _Pipeline()
    api.load_frozen_bfs(pipe, _checkpoint(tmp_path))
    initial = api.frozen_transformer_digest(pipe.transformer)
    full = api.frozen_transformer_digest(pipe.transformer, lora_only=False)
    with torch.no_grad():
        pipe.transformer.transformer_blocks[0].img_mlp.out.weight.add_(1)
    assert api.frozen_transformer_digest(pipe.transformer) == initial
    assert api.frozen_transformer_digest(pipe.transformer, lora_only=False) != full
    with torch.no_grad():
        pipe.transformer.transformer_blocks[0].attn.to_q.lora_A["bfs_joint"].weight.add_(0.01)
    assert api.frozen_transformer_digest(pipe.transformer) != initial


def test_digest_refuses_trainable_transformer():
    api = _api()
    with pytest.raises(RuntimeError, match="frozen"):
        api.frozen_transformer_digest(_Transformer())


@pytest.mark.parametrize("path_kind", ["plain", "bfs"])
def test_existing_lora_is_rejected_instead_of_misreported_as_fresh(tmp_path, path_kind):
    api = _api()
    pipe = _Pipeline()
    path = _checkpoint(tmp_path)
    api.load_frozen_bfs(pipe, path)
    with pytest.raises(RuntimeError):
        api.load_frozen_bfs(pipe, None if path_kind == "plain" else path)


def test_actual_diffusers_peft_load_on_tiny_qwen21_cpu(tmp_path):
    """Runs where the Qwen21 diffusers build is installed; no model download/GPU."""
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
        pytest.skip("Qwen-Image-2.1 diffusers build is not installed")
    from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin

    class TinyPipeline(diffusers.DiffusionPipeline, QwenImageLoraLoaderMixin):
        def __init__(self, transformer):
            super().__init__()
            self.register_modules(transformer=transformer)

    transformer = diffusers.QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=3, axes_dims_rope=(2, 2, 4),
    ).to(dtype=torch.bfloat16)
    block = transformer.transformer_blocks[0]
    generator = torch.Generator(device="cpu").manual_seed(1708)
    rank = 2
    state = {}
    for name, weight in (("attn.to_q", block.attn.to_q.weight),
                         ("img_mlp.gate_up", torch.cat((block.img_mlp.gate_layer.weight, block.img_mlp.proj.weight)))):
        for part, shape in (("A", (rank, weight.shape[1])), ("B", (weight.shape[0], rank))):
            state[f"diffusion_model.transformer_blocks.0.{name}.lora_{part}.weight"] = torch.randn(shape, generator=generator)
    pipe = TinyPipeline(transformer)
    report = _api().load_frozen_bfs(pipe, _checkpoint(tmp_path, state))
    assert report["modules_checked"] == 3
    assert report["module_scales"] == [1.0]
    assert report["freeze"]["all_parameters_frozen"]
    value = torch.ones(2, block.attn.to_q.get_base_layer().weight.shape[1], dtype=torch.bfloat16, requires_grad=True)
    block.attn.to_q(value).float().square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in transformer.parameters())

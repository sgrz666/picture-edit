"""执行实际 ReferenceUnionPipeline.__call__ 的 CPU CFG 回归；仅替代大模型组件。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler


class PredictionModel:
    config = SimpleNamespace(in_channels=64, causal_condition=True)
    transformer_blocks = [None] * 2

    def __init__(self):
        self.calls = []
        self.positive = torch.zeros(1, 4, 64)
        self.negative = torch.zeros_like(self.positive)
        self.positive[0, :, :2] = torch.tensor(
            [[3.0, 4.0], [0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]
        )
        self.negative[0, :, :2] = torch.tensor(
            [[1.0, -2.0], [0.0, 0.0], [2.0, 4.0], [1.0, -3.0]]
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        output = torch.full_like(kwargs["hidden_states"], 123.0)
        output[:, -4:] = (
            self.positive
            if kwargs["encoder_hidden_states"][0, 0, 0] > 0
            else self.negative
        )
        return (output,)


class RecordingScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self):
        super().__init__(use_dynamic_shifting=True)
        self.predictions = []

    def step(self, model_output, timestep, sample, **kwargs):
        self.predictions.append(model_output.clone())
        return super().step(model_output, timestep, sample, **kwargs)


class PipelineComponents:
    _execution_device = "cpu"
    vae_scale_factor = 16
    latent_channels = 64
    text_encoder = object()

    def __init__(self):
        self.transformer, self.scheduler = PredictionModel(), RecordingScheduler()
        self.image_processor = SimpleNamespace(
            resize=lambda image, width, height: image.resize((width, height)),
            preprocess=lambda image, width, height: torch.zeros(1, 3, height, width),
        )
        self.mask_processor = SimpleNamespace(
            preprocess=lambda image, width, height: torch.ones(1, 1, height, width)
        )

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def check_inputs(self, *args):
        pass

    def encode_prompt(self, *, image, prompt, prompt_embeds, **kwargs):
        negative = prompt == "negative" or prompt_embeds is not None
        length = 11 if negative else 7
        embeds = torch.full((1, length, 8), -1.0 if negative else 1.0)
        slots = torch.zeros(1, length, dtype=torch.bool)
        slots[:, 1 if negative else 0] = True
        return embeds, torch.ones(1, length, dtype=torch.bool), slots

    def prepare_latents(self, *args):
        return torch.zeros(1, 4, 64), torch.ones(1, 4, 64)

    def _encode_vae_image(self, pixels, generator):
        return torch.zeros(1, 64, 1, 2, 2)

    def _pack_latents(self, value, *args):
        return value.flatten(2).transpose(1, 2)

    def maybe_free_model_hooks(self):
        pass

    def progress_bar(self, **kwargs):
        class Progress:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self):
                pass

        return Progress()


@pytest.fixture
def reference_pipeline(monkeypatch):
    """加载磁盘上真实采样循环；不导入或下载 VideoX 大模型。"""
    root, package, official = (
        ModuleType(name)
        for name in (
            "videox_fun",
            "videox_fun.pipeline",
            "videox_fun.pipeline.pipeline_qwenimage21_control",
        )
    )
    root.__path__, package.__path__ = [], []
    root.pipeline, package.pipeline_qwenimage21_control = package, official
    package.QwenImage21ControlPipeline = PipelineComponents
    official.calculate_dimensions = lambda area, aspect: (32, 32, 1.0)
    official.calculate_shift = lambda *args: 0.5

    def retrieve(scheduler, steps, device, *, sigmas, mu):
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        return scheduler.timesteps, steps

    official.retrieve_timesteps = retrieve
    official.randn_tensor, official.F, official.np = torch.randn, F, np
    official.logger = SimpleNamespace()
    official.QwenImagePipelineOutput = lambda **kwargs: SimpleNamespace(**kwargs)
    for module in (root, package, official):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = (
        Path(__file__).parents[1]
        / "src/pose_control/qwen21_union/_reference_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "src.pose_control.qwen21_union._reference_pipeline_cpu_cfg", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReferenceUnionPipeline


def sample(cls, *, scale=2.0, negative="negative", **options):
    pipe = cls()
    image = Image.new("RGB", (32, 32), "green")
    result = pipe(
        prompt="positive",
        negative_prompt=negative,
        true_cfg_scale=scale,
        image=image,
        reference_images=[image],
        control_image=image,
        mask_image=Image.new("L", (32, 32), 255),
        height=32,
        width=32,
        output_resolution=32,
        num_inference_steps=1,
        output_type="latent",
        **options
    )
    return pipe, result.images


def test_actual_reference_loop_cfg_rescales_target_tokens_and_is_zero_safe(
    reference_pipeline,
):
    pipe, result = sample(reference_pipeline, cfg_norm_rescale=True)
    model = pipe.transformer
    guided = model.negative + 2 * (model.positive - model.negative)
    expected = guided * (
        model.positive.norm(dim=-1, keepdim=True)
        / guided.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    assert torch.equal(pipe.scheduler.predictions[0], expected)
    assert torch.isfinite(result).all()
    assert torch.equal(result, -expected)
    assert expected[0, 0].norm() == 5
    assert not expected[0, 1:].any()  # zero positive, zero guided and both-zero cases.
    positive, negative = model.calls
    assert positive["encoder_hidden_states"].shape[1] == 7
    assert negative["encoder_hidden_states"].shape[1] == 11
    assert torch.equal(positive["control_context"], negative["control_context"])
    assert positive["control_context"].shape == (1, 8, 129)
    assert positive["kv_cache"] is negative["kv_cache"] is None


def test_actual_reference_loop_old_union_default_remains_unrescaled(reference_pipeline):
    default, image = sample(reference_pipeline)
    disabled, explicitly_disabled = sample(reference_pipeline, cfg_norm_rescale=False)
    expected = default.transformer.negative + 2 * (
        default.transformer.positive - default.transformer.negative
    )
    assert torch.equal(default.scheduler.predictions[0], expected)
    assert torch.equal(image, explicitly_disabled)
    assert torch.equal(disabled.scheduler.predictions[0], expected)


@pytest.mark.parametrize(
    "scale,negative", [(1.0, "negative"), (1.0, None), (2.0, None)]
)
def test_actual_reference_loop_without_cfg_is_unchanged(
    reference_pipeline, scale, negative
):
    enabled, result = sample(
        reference_pipeline, scale=scale, negative=negative, cfg_norm_rescale=True
    )
    disabled, baseline = sample(
        reference_pipeline, scale=scale, negative=negative, cfg_norm_rescale=False
    )
    assert torch.equal(result, baseline)
    assert torch.equal(enabled.scheduler.predictions[0], enabled.transformer.positive)
    assert len(enabled.transformer.calls) == len(disabled.transformer.calls) == 1

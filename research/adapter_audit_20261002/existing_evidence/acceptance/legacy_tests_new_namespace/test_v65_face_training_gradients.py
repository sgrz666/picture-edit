from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from train_iper_v65_face import (
    FP32MasterAdamW,
    checkpoint_float_dtype,
    compute_face_validation_metrics,
    decode_latents_to_pixels,
    make_fixed_training_state,
)


class _DifferentiableVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.25)

    def decode(self, value: torch.Tensor, *, return_dict: bool = False):
        assert return_dict is False
        return (value * 3.0,)


class _NoGradPipelineWrapper:
    def __init__(self) -> None:
        self.vae = _DifferentiableVAE()
        self.wrapper_calls = 0

    @torch.no_grad()
    def latents_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        self.wrapper_calls += 1
        return latents * 99.0


def test_training_decode_bypasses_no_grad_pipeline_wrapper() -> None:
    pipe = _NoGradPipelineWrapper()
    latents = torch.ones(1, 2, 3, 3, requires_grad=True)

    pixels = decode_latents_to_pixels(pipe, latents, differentiable=True)
    pixels.sum().backward()

    assert pipe.wrapper_calls == 0
    assert latents.grad is not None
    torch.testing.assert_close(latents.grad, torch.full_like(latents, 1.5))


def test_inference_decode_can_use_pipeline_wrapper() -> None:
    pipe = _NoGradPipelineWrapper()
    pixels = decode_latents_to_pixels(
        pipe,
        torch.ones(1, 2, 3, 3),
        differentiable=False,
    )

    assert pipe.wrapper_calls == 1
    torch.testing.assert_close(pixels, torch.full_like(pixels, 99.0))


class _FlattenEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1)


class _LandmarkEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(1, 2, 3))
        base = images.new_tensor([[0.0, 0.0], [1.0, 1.0]])
        output = base[None].expand(images.shape[0], -1, -1).clone()
        output[..., 0] += mean[:, None]
        return output


class _PairwiseLPIPS(nn.Module):
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (predicted - target).square().mean(dim=(1, 2, 3))


def test_fixed_comparison_metrics_cover_identity_lpips_and_landmark_nme() -> None:
    target = torch.ones(1, 3, 8, 8)
    face_on = target.clone()
    face_off = torch.zeros_like(target)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    metrics = compute_face_validation_metrics(
        face_on,
        face_off,
        target,
        boxes,
        identity_evaluator=_FlattenEncoder(),
        lpips_evaluator=_PairwiseLPIPS(),
        landmark_evaluator=_LandmarkEncoder(),
    )

    assert metrics["face_on_similarity"] == pytest.approx(1.0)
    assert metrics["face_on_lpips"] == pytest.approx(0.0)
    assert metrics["face_on_landmark_nme"] == pytest.approx(0.0)
    assert metrics["face_off_similarity"] == pytest.approx(0.0)
    assert metrics["face_off_lpips"] > 0.0
    assert metrics["face_off_landmark_nme"] > 0.0


def test_overfit_state_can_pin_a_low_noise_texture_timestep() -> None:
    state = make_fixed_training_state(
        (1, 16, 64, 64),
        seed=42,
        max_timestep=650,
        fixed_timestep=150,
    )
    assert state["timestep"].tolist() == [150]
    with pytest.raises(ValueError, match="fixed_timestep"):
        make_fixed_training_state(
            (1, 16, 64, 64),
            seed=42,
            max_timestep=650,
            fixed_timestep=651,
        )


def test_checkpoint_dtype_is_inferred_before_strict_v64_migration() -> None:
    checkpoint = {
        "adapter_state_dict": {
            "weight": torch.zeros(2, dtype=torch.bfloat16),
            "counter": torch.zeros(1, dtype=torch.long),
        }
    }
    assert checkpoint_float_dtype(checkpoint) == torch.bfloat16
    checkpoint["adapter_state_dict"]["other"] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="floating dtypes"):
        checkpoint_float_dtype(checkpoint)


def test_fp32_master_adamw_accumulates_sub_bf16_updates() -> None:
    parameter = nn.Parameter(torch.tensor([0.02, 1.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW(
        [{"params": [parameter], "lr": 1e-5, "name": "face"}],
        weight_decay=0.0,
    )
    initial_master = optimizer.master_parameters[0].detach().clone()

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

    master = optimizer.master_parameters[0]
    assert master.dtype == torch.float32
    assert torch.count_nonzero(master.detach() != initial_master) == 2
    assert parameter[0] != torch.tensor(0.02, dtype=torch.bfloat16)
    state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ]
    assert state_tensors
    assert all(value.dtype == torch.float32 for value in state_tensors)

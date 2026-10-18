from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from src.pose_control.qwen21_full.checkpoint import DETAIL_ONLY_PREFIXES, FACE_ONLY_PREFIXES, HAND_ONLY_PREFIXES
from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.qwen21_full.face.conditions import PreparedFaceConditioning
from src.pose_control.qwen21_full.qwen_adapter import (
    LAYER_GROUPS, Qwen21FullAdapter, Qwen21FullControlContext,
)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(*, learned=True, face=False):
    torch.manual_seed(12)
    legacy = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32, geometry_channels=16, context_input_dim=24,
        pooled_input_dim=20, num_heads=4, enable_face_adapter=face,
    )
    if learned:
        for head in legacy.control_core.geometry_zero_heads:
            nn.init.normal_(head.weight, std=.02)
        if face:
            for head in legacy.control_interface.face_adapter.zero_heads:
                nn.init.normal_(head.weight, std=.02)
    return Qwen21FullAdapter(legacy, qwen_hidden_size=16, qwen_text_dim=24, pooled_input_dim=20)


def make_prepared(model, *, batch=1):
    raw = dict(
        normal_a=torch.zeros(batch, 3, 32, 32), pose_heatmap_a=torch.zeros(batch, 25, 32, 32),
        part_onehot_a=torch.zeros(batch, 14, 32, 32), smplx_global_a=torch.zeros(batch, 26),
        human_mask_a=torch.ones(batch, 1, 32, 32), task_id=torch.zeros(batch, dtype=torch.long),
    )
    bundle = model.condition_injector(**raw)
    return model.prepare_conditioning(
        condition_bundle=bundle, source_latent=torch.randn(batch, 16, 64), target_token_hw=(4, 4),
    )


def runtime(model, prepared=None):
    return dict(
        noisy_target=torch.randn(1, 16, 64), prepared=make_prepared(model) if prepared is None else prepared,
        prompt_embeds=torch.randn(1, 5, 24), prompt_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
        sigma=torch.tensor([.5]), target_token_hw=(4, 4),
    )


def test_lossless_initial_latent_bridge_and_grid():
    model = make_model()
    value = torch.randn(2, 16, 64)
    legacy = model.latent_to_legacy(value, (4, 4))
    assert legacy.shape == (2, 16, 8, 8)
    torch.testing.assert_close(torch.nn.functional.pixel_unshuffle(legacy, 2).flatten(2).transpose(1, 2), value)
    with pytest.raises(ValueError):
        model.latent_to_legacy(value, (2, 4))


def test_inherited_core_nonzero_raw_zero_output_and_real_padding_removal():
    model = make_model()
    args = runtime(model)
    first = model.build_residuals(**args)
    assert any(value.abs().sum() > 0 for value in first.raw_residuals.geometry)
    assert all(torch.count_nonzero(value) == 0 for value in first.residuals)
    changed = dict(args, prompt_embeds=args["prompt_embeds"].clone())
    changed["prompt_embeds"][:, 3:] = float("nan")
    second = model.build_residuals(**changed)
    for a, b in zip(first.raw_residuals.geometry, second.raw_residuals.geometry):
        torch.testing.assert_close(a, b)


def test_zero_output_bridge_learns_then_reaches_inherited_core():
    model = make_model()
    optimizer = torch.optim.SGD(model.output_bridges["geometry"].parameters(), lr=.1)
    args = runtime(model)
    output = model.build_residuals(**args)
    sum((value - 1).square().mean() for value in output.residuals).backward()
    assert model.output_bridges["geometry"].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.legacy.control_core.shared_block.parameters())
    optimizer.step()
    model.zero_grad(set_to_none=True)
    args = runtime(model)
    output = model.build_residuals(**args)
    sum((value - 1).square().mean() for value in output.residuals).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.legacy.control_core.shared_block.parameters())
    assert model.latent_bridge.weight.grad.abs().sum() > 0
    assert model.text_bridge.weight.grad.abs().sum() > 0


def test_untrained_branch_does_not_have_double_zero_gradient_block():
    model = make_model(learned=False)
    assert torch.count_nonzero(model.output_bridges["geometry"].weight) > 0
    output = model.build_residuals(**runtime(model))
    assert all(torch.count_nonzero(value) == 0 for value in output.residuals)
    sum((value - 1).square().mean() for value in output.residuals).backward()
    assert any(head.weight.grad.abs().sum() > 0 for head in model.legacy.control_core.geometry_zero_heads)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(64, 16)
        self.transformer_blocks = nn.ModuleList(nn.Sequential(nn.Linear(16, 16), nn.Tanh()) for _ in range(32))
        self.gradient_checkpointing = False

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask, img_shapes, **kwargs):
        value = self.img_in(hidden_states)
        for block in self.transformer_blocks:
            value = checkpoint(block, value, use_reentrant=False) if self.gradient_checkpointing else block(value)
        return (value,)


def fake_inputs():
    return dict(hidden_states=torch.randn(1, 32, 64), timestep=torch.tensor([.5]),
                encoder_hidden_states=torch.randn(1, 5, 24),
                encoder_hidden_states_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
                img_shapes=[[(1, 4, 4), (1, 4, 4)]])


def test_32_layer_mapping_target_only_and_gradient_checkpoint_lifetime():
    assert LAYER_GROUPS == (tuple(range(0, 6)), tuple(range(6, 11)), tuple(range(11, 16)),
                            tuple(range(16, 22)), tuple(range(22, 27)), tuple(range(27, 32)))
    model = make_model()
    backbone = FakeQwen().requires_grad_(False)
    inputs = fake_inputs()
    baseline = backbone(**inputs)[0]
    with Qwen21FullControlContext(backbone, model, make_prepared(model)) as context:
        zero = backbone(**inputs)[0]
    torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
    assert list(context.call_counts.values()) == [1] * 32
    nn.init.normal_(model.output_bridges["geometry"].weight, std=.02)
    backbone.gradient_checkpointing = True
    with set_checkpoint_early_stop(False), Qwen21FullControlContext(backbone, model, make_prepared(model)) as context:
        changed = backbone(**inputs)[0]
        torch.testing.assert_close(changed[:, :16], baseline[:, :16], rtol=0, atol=0)
        assert not torch.equal(changed[:, 16:], baseline[:, 16:])
        changed[:, 16:].square().mean().backward()
    assert model.output_bridges["geometry"].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert not getattr(backbone, "_pic_edit_qwen21_control_active")
    assert sum(context.call_counts.values()) > 32


def test_step_schedule_reuses_cfg_timestep_and_cleans_up_on_failure():
    model, backbone = make_model(), FakeQwen()
    inputs = fake_inputs()
    with Qwen21FullControlContext(backbone, model, make_prepared(model), progress_mode="step", num_inference_steps=3) as context:
        backbone(**inputs)
        assert context.output.diagnostics["denoise_progress"] == 0
        backbone(**inputs)
        assert context.output.diagnostics["denoise_progress"] == 0
        inputs["timestep"] = torch.tensor([.25])
        backbone(**inputs)
        assert context.output.diagnostics["denoise_progress"] == .5
    with pytest.raises(ValueError):
        with Qwen21FullControlContext(backbone, model, make_prepared(model), target_token_hw=(8, 8)):
            backbone(**inputs)
    assert not getattr(backbone, "_pic_edit_qwen21_control_active")


def test_strict_body_load_preserves_all_original_keys_and_rejects_missing(tmp_path):
    source = make_model()
    additions = DETAIL_ONLY_PREFIXES + FACE_ONLY_PREFIXES + HAND_ONLY_PREFIXES
    body = {key: value.detach().to(torch.bfloat16) for key, value in source.legacy.state_dict().items()
            if not key.startswith(additions)}
    path = tmp_path / "body.pt"
    torch.save({"adapter_state_dict": body, "step": 20000}, path)
    destination = make_model(learned=False)
    report = destination.load_legacy_checkpoints(path)
    assert report["body_tensors_loaded"] == len(body)
    assert report["body_step"] == 20000
    for key, value in body.items():
        torch.testing.assert_close(destination.legacy.state_dict()[key], value.float(), rtol=0, atol=0)
    before = copy.deepcopy(destination.state_dict())
    body.pop(next(iter(body)))
    torch.save({"adapter_state_dict": body}, path)
    with pytest.raises(RuntimeError, match="key mismatch"):
        destination.load_legacy_checkpoints(path)
    for key, value in before.items():
        torch.testing.assert_close(destination.state_dict()[key], value, rtol=0, atol=0)


def prepared_face():
    gate = torch.tensor([[1., 0.]])
    return PreparedFaceConditioning(
        spatial_features=torch.randn(1, 2, 512, 16, 16) * gate[..., None, None, None],
        identity_tokens=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        texture_tokens=torch.randn(1, 2, 8, 512) * gate[..., None, None],
        texture_delta=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        geometry_tokens=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        face_masks=torch.ones(1, 2, 128, 128) * gate[..., None, None],
        target_boxes=torch.tensor([[[.25, .25, .75, .75], [0., 0., 0., 0.]]]),
        face_valid=gate.bool(),
    ).validate()


def test_face_zero_boundary_disabled_fallback_roi_and_gradient():
    model = make_model(face=True)
    nn.init.normal_(model.output_bridges["geometry"].weight, std=.02)
    prepared = make_prepared(model)
    args = runtime(model, prepared)
    args["sigma"] = torch.tensor([.2])
    body = model.build_residuals(**args)
    args["prepared"] = replace(prepared, face=prepared_face())
    zero_face = model.build_residuals(**args)
    for before, after in zip(body.residuals, zero_face.residuals):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert zero_face.diagnostics["face_active_rows"] == 1
    optimizer = torch.optim.SGD(model.output_bridges["face"].parameters(), lr=.1)
    sum((value - 1).square().mean() for value in zero_face.residuals).backward()
    assert model.output_bridges["face"].weight.grad.abs().sum() > 0
    optimizer.step()
    model.zero_grad(set_to_none=True)
    args["prepared"] = replace(make_prepared(model), face=prepared_face())
    on = model.build_residuals(**args)
    off = model.build_residuals(**args, face_strength=0)
    outside = torch.ones(4, 4, dtype=torch.bool)
    outside[1:3, 1:3] = False
    for controlled, baseline in zip(on.residuals, off.residuals):
        delta = controlled - baseline
        assert torch.count_nonzero(delta[:, outside.flatten()]) == 0
        assert delta.abs().sum() > 0
    sum((value - 1).square().mean() for value in on.residuals).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.legacy.control_interface.face_adapter.shared_block.parameters())
    empty_face = prepared_face()
    for name, value in vars(empty_face).items():
        setattr(empty_face, name, torch.zeros_like(value))
    args["prepared"] = replace(args["prepared"], face=empty_face)
    invalid = model.build_residuals(**args)
    invalid_off = model.build_residuals(**args, face_strength=0)
    assert invalid.diagnostics["face_active_rows"] == 0
    for a, b in zip(invalid.residuals, invalid_off.residuals):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_face_checkpoint_cannot_overwrite_body(tmp_path):
    source = make_model(face=True)
    additions = DETAIL_ONLY_PREFIXES + FACE_ONLY_PREFIXES + HAND_ONLY_PREFIXES
    body = {key: value.detach().clone() for key, value in source.legacy.state_dict().items()
            if not key.startswith(additions)}
    face = {key: value.detach().clone() for key, value in source.legacy.state_dict().items()}
    for key, value in face.items():
        if not key.startswith(FACE_ONLY_PREFIXES):
            value.add_(100)
    body_path, face_path = tmp_path / "body.pt", tmp_path / "face.pt"
    torch.save({"adapter_state_dict": body}, body_path)
    torch.save({"adapter_state_dict": face}, face_path)
    destination = make_model(face=True)
    report = destination.load_legacy_checkpoints(body_path, face_path)
    assert report["face_nonface_tensors_ignored"] > 0
    for key, value in body.items():
        torch.testing.assert_close(destination.legacy.state_dict()[key], value, rtol=0, atol=0)
    for key, value in face.items():
        if key.startswith(FACE_ONLY_PREFIXES):
            torch.testing.assert_close(destination.legacy.state_dict()[key], value, rtol=0, atol=0)
    face['unknown_unapproved_parameter'] = torch.zeros(1)
    torch.save({'adapter_state_dict': face}, face_path)
    with pytest.raises(RuntimeError, match='Unknown Face checkpoint keys'):
        destination.load_legacy_checkpoints(body_path, face_path)

"""CPU checks for Face-only ablation and correctly aligned flow-stage capture."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_qwen21_full_bridge import (
    FakeQwen, fake_inputs, make_model, make_prepared, prepared_face, runtime,
)
from src.pose_control.qwen21_full.qwen_adapter import (
    Qwen21FullControlContext, flow_prediction_x0, validate_face_group_gains,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def learned_face_runtime():
    model = make_model(face=True).eval()
    nn.init.normal_(model.output_bridges['geometry'].weight, std=.02)
    nn.init.normal_(model.output_bridges['face'].weight, std=.02)
    args = runtime(model)
    args.update(prepared=replace(args['prepared'], face=prepared_face()),
                sigma=torch.tensor([.2]), hand_mode='off')
    return model, args


def test_default_diagnostics_preserve_prediction_and_checkpoint_keys_exactly():
    model, args = learned_face_runtime()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    normal = model.build_residuals(**args)
    diagnostic = model.build_residuals(**args, face_group_gains=[1] * 6, record_face_injection=True)
    assert normal.face_injected_residuals is None
    assert diagnostic.face_injected_residuals is not None
    for a, b in zip(normal.residuals, diagnostic.residuals):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_one_group_disabled_preserves_all_raw_stages_and_other_injections():
    model, args = learned_face_runtime()
    full = model.build_residuals(**args, record_face_injection=True)
    gains = [1.] * 6
    gains[2] = 0.
    limited = model.build_residuals(**args, face_group_gains=gains, record_face_injection=True)
    body = model.build_residuals(**args, face_strength=0)
    for branch in ('geometry', 'interaction', 'face'):
        for a, b in zip(getattr(full.raw_residuals, branch), getattr(limited.raw_residuals, branch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index, (a, b) in enumerate(zip(full.residuals, limited.residuals)):
        expected = body.residuals[index] if index == 2 else a
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
    assert torch.count_nonzero(limited.face_injected_residuals[2]) == 0
    assert all(torch.count_nonzero(value) > 0 for i, value in enumerate(limited.face_injected_residuals) if i != 2)


def test_bfloat16_all_ones_history_and_all_zero_body_remain_exact():
    model, args = learned_face_runtime()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        full = model.build_residuals(**args)
        explicit = model.build_residuals(**args, face_group_gains=[1] * 6, record_face_injection=True)
        body = model.build_residuals(**args, face_strength=0)
        zero = model.build_residuals(**args, face_group_gains=[0] * 6)
        partial = model.build_residuals(**args, face_group_gains=[1, 1, 0, 1, 1, 1])
    for expected, actual in zip(full.residuals, explicit.residuals):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(body.residuals, zero.residuals):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for index, (normal, base, delta) in enumerate(zip(full.residuals, body.residuals,
                                                   explicit.face_injected_residuals)):
        torch.testing.assert_close(normal, base + delta, rtol=0, atol=0)
        torch.testing.assert_close(partial.residuals[index], base if index == 2 else normal, rtol=0, atol=0)


def test_all_groups_off_and_invalid_face_equal_body_with_zero_roi_leakage():
    model, args = learned_face_runtime()
    off = model.build_residuals(**args, face_strength=0)
    all_off = model.build_residuals(**args, face_group_gains=[0] * 6, record_face_injection=True)
    for a, b in zip(off.residuals, all_off.residuals):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    full = model.build_residuals(**args, record_face_injection=True)
    outside = torch.ones(4, 4, dtype=torch.bool)
    outside[1:3, 1:3] = False
    assert all(torch.count_nonzero(value[:, outside.flatten()]) == 0 for value in full.face_injected_residuals)
    face = prepared_face()
    for name, value in vars(face).items():
        setattr(face, name, torch.zeros_like(value))
    invalid = model.build_residuals(**dict(args, prepared=replace(args['prepared'], face=face)),
                                    face_group_gains=[0, 1, 1, 1, 1, 1])
    for a, b in zip(off.residuals, invalid.residuals):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_detailed_layer_statistics_use_scheduled_delta_and_preserve_prefix():
    model, args = learned_face_runtime()
    backbone = FakeQwen().eval()
    inputs = fake_inputs()
    inputs['timestep'] = args['sigma']
    with Qwen21FullControlContext(backbone, model, args['prepared'], face_strength=0) as context:
        body = backbone(**inputs)[0]
    with Qwen21FullControlContext(backbone, model, args['prepared'], detailed_face_diagnostics=True,
                                 face_group_gains=[1, 1, 0, 1, 1, 1]) as context:
        controlled = backbone(**inputs)[0]
    torch.testing.assert_close(controlled[:, :16], body[:, :16], rtol=0, atol=0)
    assert len(context.injection_diagnostics) == 32
    for row in context.injection_diagnostics:
        expected = context.output.face_injected_residuals[row['face_group']]
        assert row['face_injected_max_abs'] == float(expected.detach().abs().max())
        assert row['face_roi_tokens'] == 4
        assert row['target_hidden_face_roi_rms'] > 0
        if row['face_group'] == 2:
            assert row['face_injected_local_rms'] == 0
            assert row['face_to_target_hidden_rms_ratio'] == 0
        else:
            assert row['face_injected_local_rms'] > 0


class VelocityQwen(FakeQwen):
    def __init__(self):
        super().__init__()
        self.velocity_out = nn.Linear(16, 64)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask, img_shapes, **kwargs):
        value = super().forward(hidden_states, timestep, encoder_hidden_states,
                                encoder_hidden_states_mask, img_shapes, **kwargs)[0]
        return (self.velocity_out(value),)


def test_stage_capture_uses_cloned_pre_update_latent_and_same_velocity_physical_sigma():
    model, backbone = make_model(), VelocityQwen()
    inputs = fake_inputs()
    physical = [.801, .501, .201]
    with Qwen21FullControlContext(backbone, model, make_prepared(model), num_inference_steps=3,
                                 diagnostic_stage_steps=[1, 3],
                                 diagnostic_sigma_provider=lambda step: torch.tensor(physical[step])) as context:
        expected = []
        for step, sigma in enumerate((.8, .5, .2)):
            inputs['timestep'] = torch.tensor([sigma])
            before = inputs['hidden_states'][:, -16:].clone()
            output = backbone(**inputs)[0][:, -16:]
            if step + 1 in (1, 3):
                expected.append(flow_prediction_x0(before, output, physical[step]))
            # Simulate scheduler's in-place change; stored inputs must not alias.
            inputs['hidden_states'].add_(100)
    assert [row['sampling_step'] for row in context.stage_predictions] == [1, 3]
    for row, target in zip(context.stage_predictions, expected):
        torch.testing.assert_close(row['predicted_x0'], target, rtol=0, atol=0)
        torch.testing.assert_close(row['predicted_x0'], flow_prediction_x0(
            row['latent_before_update'], row['velocity'], row['flow_sigma']), rtol=0, atol=0)
        assert row['flow_sigma'].item() != row['model_timestep'].item()
        assert not row['predicted_x0'].requires_grad


def test_stage_capture_rejects_cfg_duplicate_instead_of_saving_wrong_velocity():
    model, backbone = make_model(), VelocityQwen()
    inputs = fake_inputs()
    with pytest.raises(ValueError, match='one Transformer call'):
        with Qwen21FullControlContext(backbone, model, make_prepared(model), diagnostic_stage_steps=[1]):
            backbone(**inputs)
            backbone(**inputs)
    assert not getattr(backbone, '_pic_edit_qwen21_control_active')


@pytest.mark.parametrize('gains', [[1] * 5, [1, 1, 1, 1, 1, -1], [1, 1, 1, 1, 1, float('nan')]])
def test_invalid_group_gains_are_rejected(gains):
    with pytest.raises(ValueError, match='six finite nonnegative'):
        validate_face_group_gains(gains)

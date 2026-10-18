"""Behavioral checks for a Face objective that cannot hide damage in the canvas."""

from __future__ import annotations

import pytest
import torch

from src.pose_control.qwen21_full.execution import face_velocity_loss


def _prediction_with_face(canvas_tokens: int, outside_error: float):
    # Hold one two-token face fixed while changing the surrounding canvas.
    prediction = torch.full((1, canvas_tokens, 3), outside_error, dtype=torch.float64)
    prediction[:, :2] = torch.tensor([[1., -2., 3.], [-4., 5., -6.]], dtype=torch.float64)
    prediction.requires_grad_()
    mask = torch.zeros(1, canvas_tokens, 1, dtype=torch.float64)
    mask[:, :2] = 1
    return prediction, mask


def test_roi_loss_and_face_gradient_do_not_shrink_when_canvas_grows():
    results = []
    for canvas_tokens in (8, 64, 1024):
        prediction, mask = _prediction_with_face(canvas_tokens, outside_error=20.)
        objective, _, _, _ = face_velocity_loss(prediction.square(), mask, mode='roi')
        objective.backward()
        results.append((objective.detach(), prediction.grad[:, :2].clone()))
        torch.testing.assert_close(prediction.grad[:, 2:], torch.zeros_like(prediction.grad[:, 2:]),
                                   rtol=0, atol=0)
    for objective, face_gradient in results[1:]:
        torch.testing.assert_close(objective, results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(face_gradient, results[0][1], rtol=0, atol=0)
    assert torch.count_nonzero(results[0][1]) == 6


def test_fixing_the_background_cannot_hide_or_improve_face_error():
    broken_background, mask = _prediction_with_face(1024, outside_error=20.)
    perfect_background, _ = _prediction_with_face(1024, outside_error=0.)
    before = face_velocity_loss(broken_background.square(), mask, mode='roi')
    after = face_velocity_loss(perfect_background.square(), mask, mode='roi')
    assert after[1] < before[1]  # The whole-image score does improve.
    assert after[3] == 0
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    before[0].backward()
    after[0].backward()
    torch.testing.assert_close(perfect_background.grad[:, :2], broken_background.grad[:, :2],
                               rtol=0, atol=0)


def test_historical_global_mode_preserves_value_and_gradient_exactly():
    prediction, mask = _prediction_with_face(64, outside_error=7.)
    reference = prediction.detach().clone().requires_grad_()
    objective, global_score, _, _ = face_velocity_loss(prediction.square(), mask)
    historical_objective = reference.square().mean()
    torch.testing.assert_close(objective, historical_objective, rtol=0, atol=0)
    torch.testing.assert_close(global_score, historical_objective, rtol=0, atol=0)
    objective.backward()
    historical_objective.backward()
    torch.testing.assert_close(prediction.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_roi_mode_rejects_an_empty_face_in_any_batch_row(batch_size):
    squared = torch.ones(batch_size, 8, 3)
    mask = torch.ones(batch_size, 8, 1)
    mask[-1] = 0
    with pytest.raises(ValueError, match='nonempty face in every sample'):
        face_velocity_loss(squared, mask, mode='roi')


def test_global_mode_remains_available_without_a_face():
    squared = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    mask = torch.zeros(1, 8, 1)
    objective, _, face_score, _ = face_velocity_loss(squared, mask, mode='global')
    torch.testing.assert_close(objective, squared.mean(), rtol=0, atol=0)
    assert face_score == 0

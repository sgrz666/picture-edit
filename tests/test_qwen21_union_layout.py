"""CPU contracts for the official Qwen 2.1 joint image/text layout."""
import torch
import pytest


def test_multiple_reference_slots_and_padding():
    from src.pose_control.qwen21_union.conditions import expand_image_slots, build_control_context

    # Three references; image slots are interleaved with text and a padded text slot.
    slots = torch.tensor([[False, True, False, True, True, False, True]])
    image_mask, target_mask = expand_image_slots(slots, [(1, 2, 2)] * 4)
    assert image_mask.tolist() == [False] + [True] * 4 + [False] + [True] * 8 + [False] + [True] * 4
    assert target_mask.sum().item() == 4
    assert target_mask.nonzero().flatten().tolist() == list(range(15, 19))
    reference = torch.randn(1, 12, 64)
    control = torch.randn(1, 4, 64)
    context = build_control_context(reference, control)
    assert context.shape == (1, 16, 129)
    assert torch.equal(context[:, :12], torch.zeros(1, 12, 129))
    assert torch.equal(context[:, 12:, :64], control)
    assert torch.count_nonzero(context[:, :, 64:]) == 0


def test_image_slots_must_match_complete_reference_and_target_shapes():
    from src.pose_control.qwen21_union.conditions import expand_image_slots

    with pytest.raises(ValueError, match="token"):
        expand_image_slots(torch.tensor([[True, False, True]]), [(1, 2, 2)] * 3)


def test_scope_masks_only_residual_and_keeps_gradient():
    from src.pose_control.qwen21_union.pipeline import scoped_transformer_class

    class Official(torch.nn.Module):
        def forward_control(self, control_joint, x_joint, kwargs):
            self.received = (control_joint, x_joint, kwargs)
            return (control_joint + x_joint, 2 * control_joint + x_joint)

    model = scoped_transformer_class(Official)()
    control = torch.randn(1, 7, 3, requires_grad=True)
    joint = torch.randn_like(control)
    mask = torch.tensor([False, False, False, True, True, False, True])
    kwargs = {"target_token_mask": mask}
    expected = (control + joint, 2 * control + joint)
    model.injection_scope = "joint"
    actual = model.forward_control(control, joint, kwargs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    model.injection_scope = "target"
    actual = model.forward_control(control, joint, kwargs)
    assert model.received[0] is control and model.received[1] is joint
    assert model.received[2] is kwargs
    for result, original in zip(actual, expected):
        assert torch.equal(result[:, mask], original[:, mask])
        assert torch.count_nonzero(result[:, ~mask]) == 0
    sum(t.sum() for t in actual).backward()
    assert torch.count_nonzero(control.grad[:, mask]) > 0
    assert torch.count_nonzero(control.grad[:, ~mask]) == 0
    with pytest.raises(ValueError, match="official"):
        model.forward_control(control, joint, {})


def test_fractional_letterbox_content_coverage():
    from src.pose_control.qwen21_union.conditions import valid_target_tokens

    weights = valid_target_tokens((512, 512), (376, 512), (68, 0), vae_scale_factor=16)
    assert weights.shape == (1, 1024)
    assert torch.all((weights >= 0) & (weights <= 1))
    assert (weights > 0).sum() == 32 * 24
    assert weights.sum() == 376 * 512 / 256
    assert torch.any((weights > 0) & (weights < 1))

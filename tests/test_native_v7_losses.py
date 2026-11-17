"""Numerical and gradient tests for paper-inspired supervision, without models."""
import pytest
import torch


def test_person_fm_balances_people_instead_of_pixel_area():
    from src.pose_control.qwen21_full.native_v7.losses import paper_region_losses
    prediction = torch.tensor([[[1.], [1.], [1.], [3.]]], requires_grad=True)
    owner = torch.tensor([[[1., 0.], [1., 0.], [1., 0.], [0., 1.]]])
    result = paper_region_losses(prediction, torch.zeros_like(prediction), torch.ones(1, 4), owner, torch.zeros_like(owner))
    assert result['person_fm'].item() == pytest.approx(5.)
    assert result['person_count'] == 2
    assert result['contact_count'] == 0
    result['person_fm'].backward()
    assert prediction.grad[0, 3].item() == pytest.approx(3.)


def test_contact_fm_uses_visible_support_and_single_partial_coverage():
    from src.pose_control.qwen21_full.native_v7.losses import paper_region_losses
    prediction = torch.tensor([[[2.], [4.], [100.]]], requires_grad=True)
    coverage = torch.tensor([[.5, 1., 0.]])
    owner = torch.tensor([[[.5, 0.], [1., 0.], [0., 0.]]])
    contact = torch.tensor([[[1., 0.], [1., 1.], [1., 1.]]])
    result = paper_region_losses(prediction, torch.zeros_like(prediction), coverage, owner, contact)
    assert result['contact_fm'].item() == pytest.approx(12.)
    assert result['contact_count'] == 1
    result['contact_fm'].backward()
    assert prediction.grad[0, 2].item() == 0.


def test_empty_regions_are_differentiable_zero():
    from src.pose_control.qwen21_full.native_v7.losses import paper_region_losses
    prediction = torch.randn(1, 3, 4, requires_grad=True)
    support = torch.zeros(1, 3, 2)
    result = paper_region_losses(prediction, torch.zeros_like(prediction), torch.ones(1, 3), support, support)
    (result['person_fm'] + result['contact_fm']).backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_identity_contrastive_penalizes_swaps_and_freezes_references():
    from src.pose_control.qwen21_full.native_v7.losses import identity_contrastive
    reference = torch.eye(2)[None].requires_grad_()
    aligned = torch.tensor([[[1., .1], [.1, 1.]]], requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    good, count = identity_contrastive(aligned, reference, valid, temperature=.2)
    swapped, _ = identity_contrastive(aligned.flip(1), reference, valid, temperature=.2)
    assert count == 2 and swapped > good
    good.backward()
    assert torch.isfinite(aligned.grad).all() and aligned.grad.count_nonzero()
    assert reference.grad is None


def test_identity_contrastive_does_not_invent_missing_negatives():
    from src.pose_control.qwen21_full.native_v7.losses import identity_contrastive
    generated = torch.randn(1, 2, 4, requires_grad=True)
    loss, count = identity_contrastive(generated, torch.randn_like(generated), torch.tensor([[True, False]]))
    assert count == 0 and loss.item() == 0.
    loss.backward()
    assert not generated.grad.count_nonzero()


def test_paper_loss_config_validates_and_roundtrips():
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    cfg = PaperLossConfig()
    assert cfg.person_fm > 0 and cfg.contact_fm > 0 and cfg.identity_contrastive > 0
    assert PaperLossConfig.from_dict(cfg.to_dict()) == cfg
    for raw in ({'person_fm': -1.}, {'temperature': 0.}, {'identity_contrastive': float('nan')}, {'unknown': 1}):
        with pytest.raises(ValueError):
            PaperLossConfig.from_dict(raw)


def face_loss_case():
    """Tiny differentiable decoder implementing the frozen Qwen64 protocol."""
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    from src.pose_control.qwen21_full.native_v5.contracts import NativeTrainingSupervisionV5, SUPERVISION_SCHEMA
    class Decoder(nn.Module):
        dtype = torch.float32
        config = SimpleNamespace(latents_mean=[0.] * 64, latents_std=[1.] * 64)

        def __init__(self):
            super().__init__()
            self.calls = 0

        def decode(self, latent, return_dict=False):
            self.calls += 1
            return (F.interpolate(latent[:, :3, 0], size=(32, 32), mode='bilinear', align_corners=False).tanh()[:, :, None],)

    class Evaluator(nn.Module):
        def forward(self, image):
            return torch.cat((image.mean((2, 3)), image.std((2, 3))), 1)

    pipe = SimpleNamespace(vae=Decoder(), vae_scale_factor=16,
        _unpack_latents=lambda t, h, w, f: t.transpose(1, 2).reshape(t.shape[0], 64, 1, h // 16, w // 16))
    weights = torch.zeros(1, 6, 32, 32)
    weights[:, 0, :, :16] = 1
    weights[:, 1, :, 16:] = 1
    points = torch.tensor([[[[3., 10.], [12., 10.], [8., 16.], [4., 24.], [11., 24.]],
                            [[19., 10.], [28., 10.], [24., 16.], [20., 24.], [27., 24.]]]])
    supervision = NativeTrainingSupervisionV5(weights, torch.ones(1, 2, dtype=torch.bool), points, torch.tensor([True]),
        ({'schema': SUPERVISION_SCHEMA, 'role': 'training_only', 'driver_record_sha256': 'driver',
          'driver_rgb_sha256': 'target', 'target_canvas_id': 'canvas', 'target_person_ids': ['a', 'b']},)).validate()
    return pipe, Evaluator(), supervision


def test_gt_alignment_shares_one_decode_for_id_and_contrastive_gradients():
    from src.pose_control.qwen21_full.native_v5.losses import auxiliary_loss, LocalLossConfig
    from src.pose_control.qwen21_full.native_v7.losses import aligned_identity_regularizer
    pipe, evaluator, supervision = face_loss_case()
    prediction = torch.randn(1, 4, 64, requires_grad=True)
    target = torch.rand(1, 3, 32, 32, requires_grad=True)
    base, parts = auxiliary_loss(pipe, torch.zeros_like(prediction), prediction, torch.tensor([.2]), target,
        supervision, LocalLossConfig(weights=(0., 0., .1, 0.)), step=4, evaluators=(evaluator, None),
        identity_regularizer=aligned_identity_regularizer(.2))
    assert parts['contrastive_eligible'] == 2 and pipe.vae.calls == 1
    (base + .1 * parts['identity_contrastive']).backward()
    assert prediction.grad.abs().sum() > 0 and target.grad is None


def test_v7_full_objective_exercises_fm_id_contrastive_and_sigma_gate():
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7, TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    pipe, evaluator, supervision = face_loss_case()
    backend = object.__new__(NativeBackendV7)
    backend.config = TrainingConfigV7()
    backend.pipe = pipe
    backend.loss_context = {'owner': torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]]),
                            'contact': torch.ones(1, 4, 2)}
    prediction = torch.randn(1, 4, 64, requires_grad=True)
    target = torch.zeros_like(prediction)
    counts = {'velocity': 1, 'face': 0, 'hand': 0, 'identity': 2, 'perceptual': 0, 'paper_identity': 2}
    loss, parts = backend.compute_loss(prediction, target, torch.ones(1, 4), supervision,
        torch.zeros_like(prediction), torch.tensor([.2]), torch.rand(1, 3, 32, 32), LossSpec(identity=.1), counts, 4, (evaluator, None))
    assert parts['person_count'] == 2 and parts['contact_count'] == 2
    assert parts['contrastive_eligible'] == 2 and pipe.vae.calls == 1
    loss.backward()
    assert prediction.grad.abs().sum() > 0
    prediction.grad = None
    counts = {**counts, 'identity': 0, 'paper_identity': 0}
    high, parts = backend.compute_loss(prediction, target, torch.ones(1, 4), supervision,
        torch.zeros_like(prediction), torch.tensor([.8]), torch.rand(1, 3, 32, 32), LossSpec(identity=.1), counts, 4, (evaluator, None))
    assert parts['eligible'] == 0 and pipe.vae.calls == 1
    high.backward()
    assert prediction.grad.abs().sum() > 0

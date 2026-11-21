"""Exercise real rollout prediction without a GT latent or GPU model load."""
from contextlib import nullcontext
from types import SimpleNamespace

import torch


def test_rollout_predictor_does_not_require_target_latent(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.training import backend as module

    backend = object.__new__(module.NativeBackend)
    backend.device = torch.device('cpu')
    backend.config = SimpleNamespace(diagnostics=False)
    backend.runtime = SimpleNamespace(encode=lambda *args, **kwargs: object())
    backend.pipe = SimpleNamespace(transformer=lambda **kwargs: (kwargs['hidden_states'] * 2,))
    monkeypatch.setattr(module, 'bind_native_full_v62', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(module, 'transformer_inputs', lambda pair, latent, sigma: {'hidden_states': latent})
    monkeypatch.setattr(module.torch, 'autocast', lambda *args, **kwargs: nullcontext())
    pair = {'source_latent': torch.zeros(1, 1, 64, dtype=torch.bfloat16),
            'target_token_hw': (1, 1), 'reference_roles': ['source'], 'reference_canvas_ids': ['source']}
    latent = torch.ones(1, 1, 64, requires_grad=True)
    prediction, coverage = backend.predict_velocity(
        {'valid_image_mask': torch.ones(1, 1, 16, 16)}, pair,
        SimpleNamespace(poselink_strength=1.), torch.tensor([.5]), latent, object(),
        SimpleNamespace(identity=False, identity_strength=0.))

    assert prediction.dtype == pair['source_latent'].dtype
    assert prediction.shape == latent.shape and prediction.eq(2).all()
    assert coverage.shape == (1, 1) and coverage.eq(1).all()
    prediction.float().sum().backward()
    assert torch.equal(latent.grad, torch.full_like(latent, 2))

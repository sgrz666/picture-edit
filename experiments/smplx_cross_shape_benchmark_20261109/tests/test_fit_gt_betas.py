import numpy as np
import pytest

from fit_gt_betas import fit_one


def test_fit_recovers_known_beta_and_translation():
    rng = np.random.default_rng(0)
    n = 120
    template = rng.normal(size=(n, 3))
    shapedirs = rng.normal(size=(n, 3, 10)) * 0.1
    true_beta = rng.normal(size=10) * 0.5
    true_t = np.array([0.1, -0.2, 0.3])
    gt = template + np.einsum("i,vci->vc", true_beta, shapedirs) + true_t
    beta, transl, fitted, residual = fit_one(gt, template, shapedirs, 10)
    assert np.allclose(beta, true_beta, atol=1e-8)
    assert np.allclose(transl, true_t, atol=1e-8)
    assert residual.mean() < 1e-6
    assert fitted.shape == gt.shape

import numpy as np
import pytest

import metrics


def test_v2v_identity_translation_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    assert metrics.v2v_mm(a, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.v2v_mm(a + 3.0, a) == pytest.approx(0.0, abs=1e-9)
    b = a * 1.01
    err = metrics.v2v_mm(b, a)
    assert err > 0.0
    # No scale alignment: a 1% scaled 1.7 m body must show a visible error.
    assert err > 1.0


def test_p2p20k_identity_and_translation():
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    v = rng.normal(size=(30, 3))
    reg = sp.eye(30, format="csr")
    assert metrics.p2p20k_mm(v, v, reg) == pytest.approx(0.0, abs=1e-9)
    assert metrics.p2p20k_mm(v + 1.0, v, reg) == pytest.approx(0.0, abs=1e-9)


def test_beta_to_verts():
    rng = np.random.default_rng(2)
    template = rng.normal(size=(20, 3))
    dirs = rng.normal(size=(20, 3, 10))
    out = metrics.beta_to_verts(template, dirs, np.zeros(10))
    assert np.allclose(out[0], template)
    beta = rng.normal(size=10)
    out2 = metrics.beta_to_verts(template, dirs, beta)
    assert not np.allclose(out2[0], template)


def test_masks():
    a = np.zeros((100, 100), dtype=bool)
    a[20:60, 20:60] = True
    b = np.zeros((100, 100), dtype=bool)
    b[20:60, 20:60] = True
    c = np.zeros((100, 100), dtype=bool)
    c[70:90, 70:90] = True
    assert metrics.mask_iou(a, b) == pytest.approx(1.0)
    assert metrics.mask_iou(a, c) == pytest.approx(0.0)
    assert metrics.contour_chamfer_px(a, b) == pytest.approx(0.0, abs=1e-6)
    assert metrics.contour_chamfer_px(a, c) > 0


def test_paired_summary_direction():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    b = a - 2.0
    s = metrics.paired_summary(a, b, n_boot=1000, seed=3)
    assert s["mean_diff"] == pytest.approx(-2.0)
    assert s["wins_b"] == 5
    assert s["ci95"][0] <= s["mean_diff"] <= s["ci95"][1]

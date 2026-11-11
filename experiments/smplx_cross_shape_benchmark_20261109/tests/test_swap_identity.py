import numpy as np

import metrics


def test_beta_swap_identity_and_wrong_body():
    rng = np.random.default_rng(0)
    template = rng.normal(size=(30, 3))
    dirs = rng.normal(size=(30, 3, 10))
    b_a = rng.normal(size=10) * 0.3
    b_b = rng.normal(size=10) * 0.3

    # Identity swap: the same beta must reproduce the same mesh exactly.
    v1 = metrics.beta_to_verts(template, dirs, b_b)
    v2 = metrics.beta_to_verts(template, dirs, b_b)
    assert np.array_equal(v1, v2)
    assert metrics.v2v_mm(v1[0], v2[0]) == 0.0

    # Wrong body: A's beta must not be confused with B's beta.
    va = metrics.beta_to_verts(template, dirs, b_a)
    assert metrics.v2v_mm(va[0], v1[0]) > 0.0

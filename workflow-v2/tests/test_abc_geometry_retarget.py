"""CPU contracts for A/B identity preservation and shared-camera placement."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.core import map_roles, mix_parameters, place_pair, bounded_contact_shift


def params(value):
    return {k: np.full(n, value, np.float32) for k, n in {
        'global_orient':3, 'body_pose':63, 'left_hand_pose':45,
        'right_hand_pose':45, 'jaw_pose':3, 'expression':10,
        'betas':10, 'transl':3}.items()}


def test_roles_use_image_left_right_and_allow_explicit_swap():
    assert map_roles(np.array([[5., 0, 2], [-3, 0, 2]]), {'A':'left','B':'right'}) == [1,0]
    assert map_roles(np.array([[5., 0, 2], [-3, 0, 2]]), {'A':'right','B':'left'}) == [0,1]
    with pytest.raises(ValueError, match='distinct'):
        map_roles(np.array([[0.,0,2],[1,0,2]]), {'A':'left','B':'left'})


def test_mix_keeps_identity_and_face_but_takes_driver_body_and_hands():
    source, driver = params(1), params(2)
    out = mix_parameters(source, driver, np.ones(45)*.25, np.ones(45)*.5)
    for k in ('betas','expression','jaw_pose'):
        np.testing.assert_array_equal(out[k], source[k])
    for k in ('global_orient','body_pose'):
        np.testing.assert_array_equal(out[k], driver[k])
    np.testing.assert_allclose(out['left_hand_pose'], 1.75)
    np.testing.assert_allclose(out['right_hand_pose'], 1.5)
    assert not np.shares_memory(source['betas'], out['betas'])


def test_pair_placement_is_one_shared_translation_and_fits_camera():
    vertices=np.array([[[-1,-1,1],[-.2,1,1]],[[.2,-1,1],[1,1,1]]],np.float64)
    k=np.array([[800,0,320],[0,800,240],[0,0,1]],np.float64)
    moved, delta, diagnostics=place_pair(vertices,k,(480,640),margin=.05)
    np.testing.assert_allclose(moved-vertices,np.broadcast_to(delta, vertices.shape),atol=1e-7)
    uv=moved@k.T;uv=uv[...,:2]/uv[...,2:]
    assert uv[...,0].min()>=32-1e-4 and uv[...,0].max()<=608+1e-4
    assert uv[...,1].min()>=24-1e-4 and uv[...,1].max()<=456+1e-4
    assert diagnostics['placement_scale']==1.0


def test_contact_correction_is_bounded_and_does_not_invent_candidates():
    a=np.array([[0.,0,2],[0,1,2]])
    b=np.array([[.2,0,2],[.2,1,2]])
    shifts, diag=bounded_contact_shift([a,b],np.array([[0,0,1,0]]),np.array([[.01,0,0]]),max_shift=.05)
    assert np.linalg.norm(shifts[0])<=.05+1e-8
    assert np.linalg.norm(shifts[1])<=.05+1e-8
    assert diag['after_mean_m']<diag['before_mean_m']
    shifts, diag=bounded_contact_shift([a,b],np.empty((0,4),int),np.empty((0,3)))
    np.testing.assert_array_equal(shifts,0)
    assert diag['status']=='uncertain_no_reliable_candidates'

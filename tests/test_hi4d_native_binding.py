"""Checks that prevent coordinate and person-assignment confounds in the audit."""
import numpy as np

from scripts.eval.eval_hi4d_native_binding import (
    CASES, MAPPINGS, METHODS, OPENPOSE_TO_COCO, build_prompt, formal_jobs, skeleton,
)
from src.data.letterbox import Letterbox


def test_official_wholebody_reordering_round_trip():
    # Reproduce Wholebody's documented conversion using distinguishable COCO IDs.
    points=np.arange(133,dtype=float)
    points=np.insert(points,17,-1)
    points[[1,2,3,4,6,7,8,9,10,12,13,14,15,16,17]]=points[[17,6,8,10,7,9,12,14,16,13,15,2,1,4,3]]
    np.testing.assert_array_equal(points[list(OPENPOSE_TO_COCO)],np.arange(17))


def test_full_canvas_alignment_retains_relative_contact():
    t=Letterbox.from_size((940,1280),1024)
    two_hands=np.array([[.45,.4],[.45,.4]],dtype=float)
    projected=t.points(two_hands)*1024
    np.testing.assert_array_equal(projected[0],projected[1])
    np.testing.assert_allclose(t.unpad_points(t.points(two_hands)),two_hands,atol=1e-12)
    assert t.resized_height==1024 and t.resized_width==752 and t.left==136


def test_swap_preserves_reference_roles_and_only_reassigns_people():
    case={'source':{'descriptors':['red shirt','blue shirt']},
          'target':{'descriptors':['black shirt','white shirt']},'actions':['raise hands','hands at waist']}
    for method in METHODS:
        direct=build_prompt(case,method,'direct');swap=build_prompt(case,method,'swap')
        assert 'Assign this exact person to target role S1. Required pose: raise hands' in direct
        assert 'Assign this exact person to target role S2. Required pose: hands at waist' in swap
        assert direct.split('\n\n')[0]==swap.split('\n\n')[0]
        if method!='rgb':assert 'black shirt' not in direct and 'white shirt' not in direct
    assert MAPPINGS['swap']==tuple(reversed(MAPPINGS['direct']))


def test_protocol_has_exactly_48_unique_jobs_and_cross_person_inputs():
    jobs=formal_jobs([{'pair_id':f'hi4d_{i:02d}'} for i in range(6)])
    assert len(jobs)==48 and sum(j['resolution']==512 for j in jobs)==12
    assert len({tuple(j.values()) for j in jobs})==48
    assert all(a[0]!=b[0] for a,b in CASES)
    assert not {v[0] for c in CASES for v in c}&{'pair00','pair02'}


def test_rendering_does_not_center_an_isolated_person(tmp_path):
    xy=np.full((134,2),[700.,500.],dtype=np.float32)
    xy[:18,0]=np.linspace(650,750,18)
    path=tmp_path/'pose.npz'
    np.savez(path,xy_native=xy,scores=np.ones(134),width=940,height=1280)
    image,core,t=skeleton(path,0,1024)
    yy,xx=np.nonzero(core.max(-1))
    assert xx.min()>600 and image.size==(1024,1024)
    np.testing.assert_allclose(t.points(xy/np.array([940,1280]))[0]*1024,[656.,400.],atol=1e-4)

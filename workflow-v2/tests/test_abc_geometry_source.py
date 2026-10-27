"""Source masks and canonical canvas must retain the detected person evidence."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_source_refuses_multiple_people_instead_of_guessing():
    from abc_source_worker import require_single_person
    with pytest.raises(ValueError,match='exactly one'):
        require_single_person([{'box_xyxy':[0,0,2,2]},{'box_xyxy':[2,0,4,2]}],'A')
    p={'box_xyxy':[0,0,2,2]}
    assert require_single_person([p],'A') is p


def test_mask_selection_scores_pose_support_and_rejects_empty():
    from abc_source_worker import select_person_mask
    masks=np.zeros((2,8,8),bool);masks[0,1:3,1:3]=1;masks[1,1:7,1:7]=1
    points=np.array([[2,2],[4,4],[5,5]],float)
    chosen,diag=select_person_mask(masks,[.95,.9],points,np.ones(3))
    assert np.array_equal(chosen,masks[1])
    assert diag['pose_support'] ==1
    with pytest.raises(ValueError,match='empty'):
        select_person_mask(np.zeros((1,8,8),bool),[1],points,np.ones(3))

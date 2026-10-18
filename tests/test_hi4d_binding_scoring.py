import numpy as np
import pytest
from scripts.eval.eval_hi4d_native_binding import OPENPOSE_TO_COCO, points_for
from scripts.eval.score_hi4d_native_binding import pose_compare,checked_review,sha


def test_pose_compares_anatomical_order_and_valid_frame(tmp_path):
    p=tmp_path/'target.npz'
    xy=np.column_stack((np.linspace(100,800,134),np.linspace(100,1100,134)))
    np.savez(p,xy_native=xy,scores=np.ones(134),width=940,height=1280)
    mapped,_,transform=points_for(p,512)
    detector={'coco17_at_512':mapped[np.asarray(OPENPOSE_TO_COCO)].tolist(),'coco17_scores':[1.]*17}
    assert pose_compare(detector,p)['pose_pck_002']==1.
    assert pose_compare(detector,p)['pose_nme']==0.
    detector['coco17_at_512']=(np.asarray(detector['coco17_at_512'])+[20,0]).tolist()
    result=pose_compare(detector,p)
    assert result['pose_pck_002']==0.
    x0,y0,x1,y1=transform.rect
    assert result['pose_nme']==pytest.approx(20/np.hypot(x1-x0,y1-y0))


def test_missing_identity_is_not_silently_reassigned(tmp_path):
    p=tmp_path/'target.npz';np.savez(p,xy_native=np.zeros((134,2)),scores=np.ones(134),width=940,height=1280)
    assert pose_compare(None,p)['pose_pck_002'] is None


def test_same_detection_cannot_represent_both_source_people(tmp_path):
    p=tmp_path/'image.png';p.write_bytes(b'fixed input')
    review={'images':{str(p):{'reviewed':True,'image_sha256':sha(p),'correspondence':'correct',
        'people':[{'source_id':0,'person_detection_index':0},{'source_id':1,'person_detection_index':0}]}}}
    with pytest.raises(ValueError,match='Duplicate/invalid person detection'):
        checked_review(review,str(p),{'people':[{}]})

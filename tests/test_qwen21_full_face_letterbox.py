"""Verify Face follows RGB without stretching, double transforms or padding loss."""
import json

import pytest
import torch

from src.data.letterbox import Letterbox
from src.pose_control.qwen21_full.face.conditions import FaceFineCondition
from src.pose_control.qwen21_full.face_coordinates import (
    batch_alignment_records, transform_face_condition, transform_from_record, valid_image_tokens,
)
from src.pose_control.qwen21_full.face_supervision import (
    SCHEMA, FaceSupervisionStore, detection_record,
)
from src.pose_control.qwen21_full.execution import face_velocity_loss


def condition():
    return FaceFineCondition(
        landmarks=torch.tensor([.25,.2,.8]).expand(1,2,72,3).clone(),
        jaw_pose=torch.tensor([.1,.2,.3]).expand(1,2,3).clone(),
        expression=torch.arange(10.).expand(1,2,10).clone(),
        source_boxes=torch.tensor([.1,.2,.4,.5]).expand(1,2,4).clone(),
        target_boxes=torch.tensor([.2,.3,.5,.6]).expand(1,2,4).clone(),
        face_valid=torch.tensor([[True,False]]),source_indices=torch.tensor([[0,1]]))


def test_face_uses_distinct_source_and_target_transforms():
    c=condition();a=Letterbox.from_size((604,1080),512);b=Letterbox.from_size((1000,500),512)
    untouched=c.landmarks.clone();jaw=c.jaw_pose.clone();expression=c.expression.clone()
    source=c.source_boxes.clone();target=c.target_boxes.clone()
    mapped=transform_face_condition(c,[{'source':a.metadata(),'target':b.metadata()}])
    torch.testing.assert_close(mapped.source_boxes[0,0],a.points(source[0,0].reshape(2,2)).reshape(4),rtol=0,atol=0)
    torch.testing.assert_close(mapped.target_boxes[0,0],b.points(target[0,0].reshape(2,2)).reshape(4),rtol=0,atol=0)
    torch.testing.assert_close(mapped.landmarks[0,0],b.points(untouched[0,0]),rtol=0,atol=0)
    assert torch.equal(mapped.landmarks[...,2],untouched[...,2])
    assert torch.equal(mapped.jaw_pose,jaw) and torch.equal(mapped.expression,expression)
    assert torch.equal(mapped.landmarks[:,1],untouched[:,1])
    assert torch.equal(mapped.source_boxes[:,1],source[:,1])


def test_square_image_and_legacy_resize_have_identical_face_coordinates():
    c=condition();before=c.landmarks.clone();box=c.target_boxes.clone()
    square=Letterbox.from_size((1024,1024),512)
    assert transform_face_condition(c,None) is c
    mapped=transform_face_condition(c,[{'source':square.metadata(),'target':square.metadata()}])
    assert torch.equal(mapped.landmarks,before) and torch.equal(mapped.target_boxes,box)


def test_rejects_transform_inconsistent_with_rgb_rounding():
    record=Letterbox.from_size((604,1080),512).metadata();record['left']+=1
    with pytest.raises(ValueError,match='differs'):
        transform_from_record(record)


def test_batch_alignment_roundtrip():
    a=Letterbox.from_size((604,1080),512)
    record={'source':a.metadata(),'target':a.metadata()}
    assert batch_alignment_records({'alignment_meta':[json.dumps(record)]},1)==[record]
    assert batch_alignment_records({},1) is None
    with pytest.raises(ValueError):
        batch_alignment_records({'alignment_meta':[json.dumps(record)]},2)


def test_detection_size_and_five_points_follow_letterbox_not_stretch():
    bbox=[200,200,300,300,.9];points=[[210,220],[290,220],[250,250],[220,280],[280,280]]
    record=detection_record(bbox,points,[190,190,310,310],(1080,604),
                            score_threshold=.5,iou_threshold=.2,align_mode='letterbox')
    transform=Letterbox.from_size((604,1080),512)
    expected=transform.points(torch.tensor(points)/torch.tensor([604,1080]))*512
    assert record['short_side_output_px']['512']==pytest.approx(100*512/1080,rel=.005)
    torch.testing.assert_close(torch.tensor(record['landmarks5_output_px']['512']),expected,atol=0,rtol=0)
    legacy=detection_record(bbox,points,[190,190,310,310],(1080,604),score_threshold=.5,iou_threshold=.2)
    assert legacy['short_side_output_px']['512']==pytest.approx(min(100*512/604,100*512/1080))
    assert legacy['bbox_output_px']['512']!=record['bbox_output_px']['512']


def test_auxiliary_supervision_rejects_wrong_preprocessing(tmp_path):
    path=tmp_path/'supervision.json';path.write_text(json.dumps({'schema':SCHEMA,'align_mode':'letterbox',
        'records':{'00001/b':{'by_resolution':{'512':{'valid':True}}}}}))
    store=FaceSupervisionStore(path)
    row={'sequence_id':'00001','source_stem':'a','target_stem':'b','source_rgb':'a.png','target_rgb':'b.png'}
    assert store.valid_mask([row],512,align_mode='letterbox').tolist()==[True]
    with pytest.raises(ValueError,match='preprocessing'):
        store.valid_mask([row],512,align_mode='resize')


def test_face_objective_ignores_invalid_padding_tokens():
    transform=Letterbox.from_size((604,1080),512)
    valid=valid_image_tokens({'valid_image_mask':transform.valid_mask().unsqueeze(0)},32,'cpu')
    error=torch.ones(1,1024,3,requires_grad=True)
    mask=valid.clone()
    objective,*_=face_velocity_loss(error.square(),mask,'roi');objective.backward()
    assert objective==1
    assert torch.count_nonzero(error.grad[valid.expand_as(error)==0])==0

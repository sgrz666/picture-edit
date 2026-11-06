import json
import numpy as np
import pytest
from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_v5.training.io import file_sha


def cache(tmp_path,*,canvas=None):
    from PIL import Image
    from scripts.data.build_native_spatial_weights import save_npz,digest
    rgb=tmp_path/'gt.png';Image.new('RGB',(31,45),'gray').save(rgb)
    spec=canvas or native_pad((45,31));valid=spec.valid_mask()[0].numpy()
    frame=dict(frame_id='f',rgb_path=str(rgb),rgb_sha256=file_sha(rgb),original_hw=[45,31],
               loss_mask=dict(hand_a=False,hand_b=False),hand_raw_pose_usable=dict(hand_a=False,hand_b=False))
    frames=tmp_path/'frames.jsonl';frames.write_text(json.dumps(frame)+'\n',encoding='utf8')
    meta=dict(schema='native_spatial_components_v1',gt_rgb_sha256=frame['rgb_sha256'],target_canvas_id=spec.fingerprint(),
              canvas=spec.metadata(),frame_id='f',geometry_method='sapiens_v1',source_frame_sha256=digest(frame))
    npz=tmp_path/'components.npz';save_npz(npz,dict(body=valid*.5,joint=valid*.2,hand=valid*0,valid=valid),meta)
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(schema='native_spatial_index_v1',
          entries={frame['rgb_sha256']+':'+spec.fingerprint():dict(file=dict(path=str(npz),sha256=file_sha(npz)),metadata=meta)})),encoding='utf8')
    return frames,index,rgb


def test_audit_checks_actual_gt_maps_and_qwen64_baseline(tmp_path):
    from scripts.audit.audit_native_spatial_weights import audit
    frames,index,_=cache(tmp_path)
    report=audit(index,frames)
    assert report['passed'] and report['frames']==1 and report['qwen_channels']==64
    assert report['padding_weight_max']==0 and report['zero_coefficients_baseline_max_error']<1e-6


def test_audit_rejects_rgb_changed_after_generation(tmp_path):
    from scripts.audit.audit_native_spatial_weights import audit
    frames,index,rgb=cache(tmp_path)
    with rgb.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='GT|RGB'):
        audit(index,frames)


def test_audit_refuses_partial_index_for_full_manifest(tmp_path):
    from scripts.audit.audit_native_spatial_weights import audit
    frames,index,_=cache(tmp_path)
    with frames.open('a',encoding='utf8') as stream:stream.write(json.dumps({'frame_id':'missing'})+'\n')
    with pytest.raises(ValueError,match='frame|[Cc]overage'):
        audit(index,frames)


def test_audit_rejects_self_consistent_cache_on_wrong_original_grid(tmp_path):
    from scripts.audit.audit_native_spatial_weights import audit
    frames,index,_=cache(tmp_path,canvas=native_pad((44,30)))
    with pytest.raises(ValueError,match='original|RGB|Canvas'):
        audit(index,frames)


def test_audit_accepts_verified_coarse_hand_with_old_fine_gate_closed(tmp_path):
    import hashlib
    from PIL import Image
    from scripts.audit.audit_native_spatial_weights import audit
    from scripts.data.build_native_spatial_weights import save_npz,digest
    from src.pose_control.qwen21_full.native_v5.spatial_masks import regional_maps
    frames,_,_=cache(tmp_path)
    f=json.loads(frames.read_text());points=[[12.,20.]]*134
    hand=[[10.+i%5,18.+i//5] for i in range(21)]
    points[113:134]=hand;points[4]=hand[0]
    f.update(candidate=points,effective_confidence=[.9]*134,valid_keypoint_mask=[True]*134,
             hand_a_present=True,hand_b_present=False,hand_a_status='visible_blurry',
             hand_block_slices={'hand_a':[113,134],'hand_b':[92,113]},
             hand_body_wrist_indices={'hand_a':4,'hand_b':7},hand_landmarks21={'hand_a':None,'hand_b':None})
    spec=native_pad((45,31));body=np.ones((45,31),np.uint8)
    maps,status=regional_maps(f,body,spec,hand_policy='visible_region_v2')
    metadata=dict(schema='native_spatial_components_v1',gt_rgb_sha256=f['rgb_sha256'],target_canvas_id=spec.fingerprint(),
                  canvas=spec.metadata(),frame_id='f',geometry_method='sapiens_v1',source_frame_sha256=digest(f),
                  hand_policy='visible_region_v2',region_status=status,
                  segmentation={'body_mask_pixel_sha256':hashlib.sha256(body.tobytes()).hexdigest()})
    Image.fromarray(body*255).save(tmp_path/'body_mask.png')
    npz=tmp_path/'v2.npz';save_npz(npz,maps,metadata)
    index=tmp_path/'index2.json';index.write_text(json.dumps(dict(schema='native_spatial_index_v1',entries={f['rgb_sha256']+':'+spec.fingerprint():
          dict(file=dict(path=str(npz),sha256=file_sha(npz)),metadata=metadata)})),encoding='utf8')
    source=tmp_path/'frames2.jsonl';source.write_text(json.dumps(f)+'\n',encoding='utf8')
    report=audit(index,source)
    assert report['passed'] and report['hand_support_frames']==1


def test_audit_rejects_entry_policy_disagreeing_with_generation_policy(tmp_path):
    from scripts.audit.audit_native_spatial_weights import audit
    frames,index,_=cache(tmp_path)
    obj=json.loads(index.read_text());obj['config']={'hand_policy':'visible_region_v2'}
    path=tmp_path/'mismatched_policy.json';path.write_text(json.dumps(obj),encoding='utf8')
    with pytest.raises(ValueError,match='policy|Policy'):
        audit(path,frames)

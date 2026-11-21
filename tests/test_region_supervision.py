"""Binding and pixel-support contracts for immutable training-only sidecars."""
import json
import numpy as np
import pytest
import torch
from src.data.canvas_contract import native_pad


def write_case(tmp_path):
    from src.pose_control.qwen21_full.region_supervision import write_region_sidecar
    spec=native_pad((15,29));shape=spec.canvas_hw
    meta={'schema':'native_region_supervision_v1','target_rgb_sha256':'a'*64,
          'canvas':spec.metadata(),'canvas_id':spec.fingerprint(),'person_ids':['p0',None],
          'hand_slot_order':['A-left','A-right','B-left','B-right'],
          'producer':'test-fixture','annotation_sha256':'b'*64,'frame_id':'frame0'}
    arrays={'valid':spec.valid_mask()[0].numpy()}
    for name,size in {'person':2,'joint':2,'hands':4,'background':1}.items():
        arrays[name]=np.zeros((size,*shape),np.float32)
        arrays[name+'_quality']=np.zeros(size,np.float32)
    arrays['person'][0]=arrays['valid'];arrays['person_quality'][0]=1
    arrays['hands'][0,:2,:3]=1;arrays['hands_quality'][0]=.5
    return write_region_sidecar(tmp_path/'region.npz',arrays,meta),spec


def test_region_read_binds_gt_canvas_person_and_pools_once(tmp_path):
    from src.pose_control.qwen21_full.region_supervision import read_region_sidecar
    ref,spec=write_case(tmp_path)
    payload=read_region_sidecar(ref,target_rgb_sha256='a'*64,canvas=spec.metadata(),person_ids=['p0',None])
    hw=spec.metadata()['token_hw']
    assert payload['supports']['hands'].shape==(4,hw[0]*hw[1])
    torch.testing.assert_close(payload['supports']['hands'][0].sum(),torch.tensor(6/256))
    torch.testing.assert_close(payload['coverage'],torch.nn.functional.avg_pool2d(spec.valid_mask(),16).flatten(1))
    assert payload['quality']['hands'][0]==.5


@pytest.mark.parametrize('change',['rgb','canvas','person','file'])
def test_misbound_regions_rejected(tmp_path,change):
    from src.pose_control.qwen21_full.region_supervision import read_region_sidecar
    ref,spec=write_case(tmp_path)
    kwargs={'target_rgb_sha256':'a'*64,'canvas':spec.metadata(),'person_ids':['p0',None]}
    if change=='rgb':kwargs['target_rgb_sha256']='c'*64
    elif change=='person':kwargs['person_ids']=['other',None]
    elif change=='canvas':kwargs['canvas']=native_pad((17,29)).metadata()
    else:ref={**ref,'sha256':'d'*64}
    with pytest.raises(ValueError):read_region_sidecar(ref,**kwargs)


def test_sidecar_writer_refuses_overwrite(tmp_path):
    from src.pose_control.qwen21_full.region_supervision import write_region_sidecar
    ref,spec=write_case(tmp_path)
    with np.load(ref['path'],allow_pickle=False) as a:
        metadata=json.loads(a['metadata'].item());arrays={k:a[k] for k in a.files if k!='metadata'}
    with pytest.raises(FileExistsError):write_region_sidecar(ref['path'],arrays,metadata)


def test_v7_regional_mode_rejects_default_additive_person_contact(tmp_path):
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.config import StageSpec,RouteSpec,LossSpec
    stage=StageSpec(phase='S0',routes=(RouteSpec('body','body',1.,loss=LossSpec(region_mode='uniform')),))
    with pytest.raises(ValueError):TrainingConfigV7(model_path=str(tmp_path),stage=stage)

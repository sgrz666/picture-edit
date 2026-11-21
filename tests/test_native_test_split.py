"""A real test role must never become a training or validation source."""
import json
from types import SimpleNamespace

import pytest


def test_test_source_cannot_receive_training_weight_or_enter_a_route():
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec,StageSpec,RouteSpec
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    test=SourceSpec('heldout',role='test')
    cfg=TrainingConfigV7(sources=(SourceSpec('train',weight=1.),test))
    assert TrainingConfigV7.from_dict(cfg.record()).sources[-1].role=='test'
    with pytest.raises(ValueError,match='test'):
        SourceSpec('heldout',role='test',weight=1.)
    with pytest.raises(ValueError,match='training sources'):
        TrainingConfigV7(sources=cfg.sources,stage=StageSpec(phase='S0',routes=(RouteSpec('body','body',1.,source_ids=('heldout',)),)))


@pytest.mark.parametrize('overlap',['train','validation'])
def test_test_groups_and_all_rgb_hashes_are_disjoint(overlap):
    from src.pose_control.qwen21_full.native_v62.training.data import validate_evaluation_split
    cfg=SimpleNamespace(experiment_mode='standard')
    groups={'train':{'tr'},'validation':{'val'},'observation':set(),'test':{'te'}}
    images={k:{k+'-rgb'} for k in groups};targets={k:set() for k in groups}
    validate_evaluation_split(cfg,groups,targets,images)
    with pytest.raises(ValueError,match='test'):
        validate_evaluation_split(cfg,{**groups,'test':groups[overlap]},targets,images)
    with pytest.raises(ValueError,match='test'):
        validate_evaluation_split(cfg,groups,targets,{**images,'test':images[overlap]})


def test_dataplan_keeps_test_rows_out_of_optimizer_pools(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v62.training import data as module
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec,StageSpec,RouteSpec
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.io import write_json
    from src.pose_control.qwen21_full.native_v62.training.sampling import DeterministicSampler
    from src.pose_control.qwen21_full.native_v62 import conditioning
    class Dataset:
        def __init__(self,path,**kwargs):self.rows=[json.loads(open(path).read())];self.role=self.rows[0]['pair_id']
        def __len__(self):return 1
        def __getitem__(self,index):
            return {'native_scene_kind':'single','local_condition_v4':SimpleNamespace(metadata=[{
                'driver_rgb_sha256':self.role+'-target','source_rgb_sha256':self.role+'-source'}])}
    monkeypatch.setattr(module,'NativeFullDatasetV62',Dataset)
    monkeypatch.setattr(module,'sample_status',lambda *a:{'eligible':True,'face_area':0,'hand_area':0,'auxiliary_faces':0})
    monkeypatch.setattr(conditioning,'read_condition',lambda *a:{'profile':'single_2ref'})
    sources=[]
    for role in ('train','validation','test'):
        manifest=write_json(tmp_path/(role+'.jsonl'),{'pair_id':role,'qwen_condition':{'synthetic':'reader boundary only'}})
        groups=write_json(tmp_path/(role+'.groups.json'),{role:role+'-note'})
        sources.append(SourceSpec(role,role=role,manifest=manifest,groups=groups,weight=1. if role=='train' else None))
    cfg=TrainingConfigV7(sources=tuple(sources),qwen_condition_profile='single_2ref',
        stage=StageSpec(phase='S0',routes=(RouteSpec('body','body',1.),)))
    plan=module.DataPlan(cfg)
    assert any(r['role']=='test' for r in plan.records)
    assert set(plan.pools)=={('body','train')}
    sampler=DeterministicSampler(cfg,plan.pools)
    assert {sampler.draw()[1] for _ in range(30)}=={'train'}

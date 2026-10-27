"""Fixed-pair observations never enter the optimizer's sampling pool."""
from dataclasses import replace
import pytest
from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, SourceSpec
from src.pose_control.qwen21_full.native_v5.training.data import validate_evaluation_split
from src.pose_control.qwen21_full.native_v5.training.sampling import DeterministicSampler


def test_observation_requires_explicit_overfit_experiment():
    with pytest.raises(ValueError, match='observation'):
        TrainingConfig(sources=(SourceSpec('observe', role='observation'),))
    c = TrainingConfig(experiment_mode='fixed_pair_overfit', sources=(
        SourceSpec('train', weight=1.), SourceSpec('observe', role='observation')))
    assert TrainingConfig.from_dict(c.record()).signature() == c.signature()
    s = DeterministicSampler(c, {('body', 'train'): [0]})
    assert all(s.draw()[1] == 'train' for _ in range(10))
    assert 'validation source' not in c.missing()
    assert 'validation source' in TrainingConfig().missing()


def test_standard_group_isolation_is_not_relaxed():
    with pytest.raises(ValueError, match='leakage'):
        validate_evaluation_split(TrainingConfig(), {'train': {'video'}, 'validation': {'video'}, 'observation': set()}, {})


def test_same_scene_observation_is_explicit_and_target_disjoint():
    c = TrainingConfig(experiment_mode='fixed_pair_overfit')
    groups = {'train': {'video'}, 'validation': set(), 'observation': {'video'}}
    validate_evaluation_split(c, groups, {'train': {'sha-a'}, 'observation': {'sha-b'}})
    with pytest.raises(ValueError, match='target'):
        validate_evaluation_split(c, groups, {'train': {'sha-a'}, 'observation': {'sha-a'}})


def test_generation_and_condition_protocol_enter_training_signature():
    c = TrainingConfig()
    assert replace(c, qwen_condition_profile='single_2ref').signature() != c.signature()
    assert replace(c, generation_every_updates=30).signature() != c.signature()


def test_continuous_stage_gate_requires_recorded_user_authorization(tmp_path):
    import json
    import hashlib
    from src.pose_control.qwen21_full.native_v5.training.config import stage_preset
    from src.pose_control.qwen21_full.native_v5.training.validation import check_body_baseline
    def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    grad=tmp_path/'grad.json';grad.write_text(json.dumps({'level':'grad','passed':True,'production_passed':True}))
    authorization={'source':'direct_user_message','text':'continue the approved 300 updates'}
    report=tmp_path/'body.json';report.write_text(json.dumps({'body_accepted':True,'manual_acknowledgment':False,
        'checkpoint_sha256':'a'*64,'acceptance_kind':'technical_pilot_gate','continuous_stage_authorization':authorization,
        's0_gradient_report':ref(grad)}))
    c=replace(TrainingConfig(),stage=stage_preset('S1','face',budget=60),body_baseline_report=ref(report))
    with pytest.raises(ValueError):check_body_baseline(c,{'sha256':'a'*64})
    check_body_baseline(replace(c,continuous_stage_authorization=authorization),{'sha256':'a'*64})


def test_head_warmup_gate_only_requires_effectively_enabled_training_branches():
    import torch
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.native_v5.training.validation import check_warmed_heads
    rows=[SimpleNamespace(name='face',runtime_group='face_heads',role='head'),
        SimpleNamespace(name='interaction',runtime_group='interaction_heads',role='head')]
    catalog=SimpleNamespace(rows=rows,parameters={'face':torch.ones(2),'interaction':torch.zeros(2)})
    sample={'role':'train','eligible':True,'branch_usable':{'face':True,'hand':False,'detail':False,'interaction':False},
        'effective_regions':{'detail':[False]*6}}
    data=SimpleNamespace(records=[sample])
    assert check_warmed_heads(catalog,{'face','interaction'},data)==['face_heads']
    sample['branch_usable']['interaction']=True
    with pytest.raises(ValueError,match='interaction_heads'):
        check_warmed_heads(catalog,{'face','interaction'},data)
    catalog.parameters['interaction'].fill_(1)
    assert check_warmed_heads(catalog,{'face','interaction'},data)==['face_heads','interaction_heads']
    sample['role']='observation'
    assert check_warmed_heads(catalog,{'face','interaction'},data)==[]

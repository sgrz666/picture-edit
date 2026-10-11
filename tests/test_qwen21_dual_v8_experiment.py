import copy
import json
from pathlib import Path

import pytest
import torch


def contract():
    return dict(model_id='/model', backbone_sha256='backbone', model_config_sha256='config',
        generation_sha256='generation', source_features_sha256='features', geometry_sha256='geometry',
        supervision_sha256='supervision', supervision_content={'target_rgb_sha256':'rgb',
            'target_masks_sha256':['maskA','maskB'], 'person_uids':['A','B']},
        warm_start_sha256='warm', prompt='hug', seed=42, effective_batch=4,
        frozen_scopes=['spatial','person','role'], optimizer={'name':'AdamW','lr':1e-4,
            'weight_decay':.01,'betas':[.9,.999],'eps':1e-8},
        precision_policy={'bf16_reduced_precision_reduction':False}, code_sha256={'train':'code'})


@pytest.mark.parametrize('field,value', [('model_id','/other'),('backbone_sha256','other'),
    ('supervision_content',{'target_rgb_sha256':'changed'}),('seed',43),
    ('optimizer',{'name':'AdamW','lr':9e-4}),('geometry_sha256','changed')])
def test_resume_rejects_different_immutable_experiment(field,value):
    from src.pose_control.qwen21_full.dual.experiment import validate_resume_contract
    old=contract();current=copy.deepcopy(old);current[field]=value
    with pytest.raises(ValueError,match=field):
        validate_resume_contract({'run_contract':old,'step':50},current,requested_steps=100)


def test_resume_accepts_same_contract_and_rejects_step_rewind():
    from src.pose_control.qwen21_full.dual.experiment import validate_resume_contract
    assert validate_resume_contract({'run_contract':contract(),'step':50},contract(),requested_steps=100)==50
    with pytest.raises(ValueError,match='step'):
        validate_resume_contract({'run_contract':contract(),'step':300},contract(),requested_steps=100)


def test_resume_rejects_missing_contract():
    from src.pose_control.qwen21_full.dual.experiment import validate_resume_contract
    with pytest.raises(ValueError,match='contract'):
        validate_resume_contract({'step':50},contract(),requested_steps=100)


def test_resume_optimizer_matches_requested_lr_before_state_is_loaded():
    from src.pose_control.qwen21_full.dual.experiment import validate_optimizer_contract
    p=torch.nn.Parameter(torch.zeros(1))
    saved=torch.optim.AdamW([p],lr=1e-4,weight_decay=.01).state_dict()
    validate_optimizer_contract(saved,contract()['optimizer'])
    wrong=copy.deepcopy(contract()['optimizer']);wrong['lr']=9e-4
    with pytest.raises(ValueError,match='lr'):validate_optimizer_contract(saved,wrong)


def test_resume_metrics_reject_older_checkpoint_duplicate_and_broken_tail(tmp_path):
    from src.pose_control.qwen21_full.dual.experiment import validate_metrics_resume
    path=tmp_path/'metrics.jsonl'
    path.write_text('\n'.join(json.dumps({'step':step}) for step in [1,2,3])+'\n')
    validate_metrics_resume(path,3)
    with pytest.raises(ValueError,match='tail'):validate_metrics_resume(path,2)
    path.write_text('\n'.join(json.dumps({'step':step}) for step in [1,2,2,3])+'\n')
    with pytest.raises(ValueError,match='consecutive'):validate_metrics_resume(path,3)
    path.write_text('{broken')
    with pytest.raises(ValueError,match='metrics'):validate_metrics_resume(path,3)


def test_fresh_resume_directory_can_begin_after_parent_checkpoint(tmp_path):
    from src.pose_control.qwen21_full.dual.experiment import validate_metrics_resume
    path=tmp_path/'metrics.jsonl'
    validate_metrics_resume(path,50)
    path.write_text('\n'.join(json.dumps({'step':step}) for step in [51,52])+'\n')
    validate_metrics_resume(path,52)


def test_pilot_code_contract_covers_legacy_execution_dependencies():
    from src.pose_control.qwen21_full.dual.experiment import pilot_code_sha256
    code=pilot_code_sha256(Path(__file__).resolve().parents[1])
    for suffix in ('qwen_runtime.py','qwen_adapter.py','geometry_encoder.py','conditions.py',
        'reasoning/config.py','reasoning/person_geometry.py','reasoning/blocks.py','deepgen_adapter.py'):
        assert 'src/pose_control/qwen21_full/'+suffix in code


def test_pilot_code_hash_changes_when_native_input_builder_changes(tmp_path):
    from src.pose_control.qwen21_full.dual.experiment import pilot_code_sha256
    for path in ('scripts/train/train_qwen21_dual_v8_pilot.py','src/data/hi4d_smplx_dataset.py',
        'scripts/precompute/precompute_hi4d_smplx_controls.py','src/pose_control/qwen21_full/qwen_runtime.py'):
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('old')
    previous=pilot_code_sha256(tmp_path)
    (tmp_path/'src/pose_control/qwen21_full/qwen_runtime.py').write_text('changed')
    current=pilot_code_sha256(tmp_path)
    assert previous['src/pose_control/qwen21_full/qwen_runtime.py']!=current['src/pose_control/qwen21_full/qwen_runtime.py']


@pytest.mark.parametrize('corruption', ['shape','nan','count','step','unknown_state'])
def test_resume_rejects_damaged_optimizer_before_loading(corruption):
    from src.pose_control.qwen21_full.dual.experiment import validate_optimizer_contract
    p=torch.nn.Parameter(torch.zeros(2));optimizer=torch.optim.AdamW([p],lr=1e-4,weight_decay=.01)
    p.grad=torch.ones_like(p);optimizer.step()
    state=optimizer.state_dict()
    if corruption=='shape':state['state'][0]['exp_avg']=torch.zeros(1)
    elif corruption=='nan':state['state'][0]['exp_avg_sq'][0]=float('nan')
    elif corruption=='count':state['param_groups'][0]['params'].append(1)
    elif corruption=='step':state['state'][0]['step'].fill_(float('nan'))
    else:state['state'][999]=state['state'][0]
    with pytest.raises(ValueError,match='optimizer'):
        validate_optimizer_contract(state,contract()['optimizer'],parameters=[p])


def test_resume_accepts_valid_moments_and_unused_parameter_without_state():
    from src.pose_control.qwen21_full.dual.experiment import validate_optimizer_contract
    parameters=[torch.nn.Parameter(torch.zeros(2)),torch.nn.Parameter(torch.zeros(3))]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    parameters[0].grad=torch.ones(2);optimizer.step()
    validate_optimizer_contract(optimizer.state_dict(),contract()['optimizer'],parameters=parameters)


def test_native_cache_digest_hashes_tensor_content_instead_of_torch_archive_name(tmp_path):
    from src.pose_control.qwen21_full.dual.experiment import tensor_cache_sha256
    value={'latent':torch.randn(1,2,64,dtype=torch.bfloat16),'token_hw':(32,32),'valid':torch.ones(2,dtype=torch.bool)}
    paths=[tmp_path/'pid1.tmp',tmp_path/'pid2.tmp']
    for path in paths:torch.save(value,path)
    assert paths[0].read_bytes()!=paths[1].read_bytes()
    loaded=[torch.load(path,weights_only=True) for path in paths]
    assert tensor_cache_sha256(loaded[0])==tensor_cache_sha256(loaded[1])
    loaded[1]['latent'][0,0,0]+=1
    assert tensor_cache_sha256(loaded[0])!=tensor_cache_sha256(loaded[1])


@pytest.mark.parametrize('corruption', ['order','rollback'])
def test_resume_rejects_swapped_same_shape_moments_and_optimizer_step_rollback(corruption):
    from src.pose_control.qwen21_full.dual.experiment import validate_optimizer_contract
    parameters=[torch.nn.Parameter(torch.zeros(2)),torch.nn.Parameter(torch.zeros(2))]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    for parameter in parameters:parameter.grad=torch.ones(2)
    optimizer.step();state=optimizer.state_dict()
    if corruption=='order':state['param_groups'][0]['params'].reverse()
    with pytest.raises(ValueError,match='optimizer'):
        validate_optimizer_contract(state,contract()['optimizer'],parameters=parameters,checkpoint_step=1 if corruption=='order' else 50)

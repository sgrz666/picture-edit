"""Atomic frozen-geometry import and restricted checkpoint deserialization."""
import pathlib
import pickle
import os

import pytest
import torch

from src.pose_control.qwen21_full.dual.checkpoint import (
    load_checkpoint, save_checkpoint, warm_start_from_body,
)
from src.pose_control.qwen21_full.dual.controller import DualV8Adapter

torch.set_num_threads(1)

FROZEN_PREFIXES = ('geometry_encoder.spatial_encoder.','reasoner.person_reasoner.')


def tiny_adapter():
    return DualV8Adapter(width=32,hidden_size=64,rank=8,layer_indices=(3,8))


def warm_payload(model):
    state = {}
    for name,value in model.state_dict().items():
        if name.startswith('geometry_encoder.spatial_encoder.'):
            source = name.replace('geometry_encoder.spatial_encoder.',
                                  'condition_injector.spatial_encoder.',1)
            state[source] = torch.full_like(value,.25)
        elif name.startswith('reasoner.person_reasoner.'):
            state[name] = torch.full_like(value,.3)
        elif name == 'reasoner.shared_role_offset':
            state['reasoner.role_embedding.weight'] = torch.stack(
                (torch.full_like(value,.4),torch.full_like(value,-.5)))
    return {'adapter_state_dict':state,'step':19500}


def test_strict_frozen_warm_start_reports_all_scopes_and_copies_only_shared_role():
    model = tiny_adapter()
    payload = warm_payload(model)
    report = warm_start_from_body(model,payload,
        require_frozen_coverage=True,expected_source_step=19500)
    coverage = report['frozen_coverage']
    frozen = {name:value for name,value in model.state_dict().items()
              if name.startswith(FROZEN_PREFIXES) or name=='reasoner.shared_role_offset'}
    assert coverage['required'] and coverage['complete']
    assert coverage['expected_tensor_count']==coverage['matched_tensor_count']==len(frozen)
    assert coverage['expected_numel']==coverage['matched_numel']==sum(value.numel() for value in frozen.values())
    assert coverage['missing']==[]
    assert all(scope['complete'] for scope in coverage['scopes'].values())
    assert report['source_step']==report['expected_source_step']==19500
    assert report['source_step_verified']
    assert torch.all(model.reasoner.shared_role_offset==.4)


@pytest.mark.parametrize('damage',['missing','shape','nonfinite','step','role_missing','role_disabled'])
def test_strict_import_rejects_incomplete_or_wrong_checkpoint_before_any_mutation(damage):
    model = tiny_adapter()
    payload = warm_payload(model)
    state = payload['adapter_state_dict']
    key = next(name for name in state if name.endswith('weight') and name!='reasoner.role_embedding.weight')
    if damage=='missing': del state[key]
    elif damage=='shape': state[key] = state[key].flatten()[:1]
    elif damage=='nonfinite': state[key].flatten()[0] = float('nan')
    elif damage=='step': payload['step'] = 20000
    elif damage=='role_missing': del state['reasoner.role_embedding.weight']
    snapshot = {name:value.clone() for name,value in model.state_dict().items()}
    with pytest.raises(ValueError):
        warm_start_from_body(model,payload,copy_shared_role=damage!='role_disabled',
            require_frozen_coverage=True,expected_source_step=19500)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in snapshot.items())
    assert not hasattr(model,'warm_start_report')


def test_default_partial_import_remains_explicitly_incomplete():
    model = tiny_adapter()
    payload = warm_payload(model)
    name,value = next(iter(payload['adapter_state_dict'].items()))
    report = warm_start_from_body(model,{'adapter_state_dict':{name:value},'step':19500})
    coverage = report['frozen_coverage']
    assert not coverage['required'] and not coverage['complete']
    assert coverage['matched_tensor_count']==1
    assert len(coverage['missing'])==coverage['expected_tensor_count']-1
    assert not report['source_step_verified']


def test_new_v8_save_load_uses_weights_only_and_keeps_provenance(tmp_path,monkeypatch):
    model = tiny_adapter()
    report = warm_start_from_body(model,warm_payload(model),
        require_frozen_coverage=True,expected_source_step=19500)
    path = tmp_path/'dual_v8.pt'
    save_checkpoint(path,model,{'model_id':'native-qwen21'},step=3)
    actual_load = torch.load
    loads = []
    def restricted_load(*args,**kwargs):
        loads.append(kwargs)
        return actual_load(*args,**kwargs)
    monkeypatch.setattr(torch,'load',restricted_load)
    loaded,metadata,payload = load_checkpoint(path)
    assert len(loads)==1 and loads[0]['weights_only'] is True
    assert metadata['loading_report']==report and payload['step']==3
    assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in model.state_dict().items())


class UnapprovedMetadata:
    pass


def test_new_v8_load_rejects_unapproved_pickle_globals(tmp_path):
    model = tiny_adapter()
    path = tmp_path/'dual_v8.pt'
    payload = save_checkpoint(path,model,{'model_id':'native-qwen21'})
    payload['unapproved'] = UnapprovedMetadata()
    torch.save(payload,path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)


@pytest.mark.skipif(os.name=='nt',reason='real old checkpoint stores PosixPath')
def test_body_safe_loader_allows_only_bounded_posix_path_and_does_not_leak_globals(tmp_path,monkeypatch):
    model = tiny_adapter()
    payload = warm_payload(model)
    payload['training_options'] = {'run_dir':pathlib.PosixPath('/original/body/run')}
    path = tmp_path/'body.pt'
    torch.save(payload,path)
    prior = torch.serialization.get_safe_globals()
    actual_load = torch.load
    loads = []
    def restricted_load(*args,**kwargs):
        loads.append(kwargs)
        return actual_load(*args,**kwargs)
    monkeypatch.setattr(torch,'load',restricted_load)
    report = warm_start_from_body(model,path,
        require_frozen_coverage=True,expected_source_step=19500)
    assert loads and all(item['weights_only'] is True for item in loads)
    assert report['frozen_coverage']['complete']
    assert torch.serialization.get_safe_globals()==prior

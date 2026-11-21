"""A paired pilot must differ only in its explicitly enabled preservation loss."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'experiments/hi4d_qwen21_dual_20260930/native-v7-source-preservation-20261121/scripts/run_pilot.py'
    spec=importlib.util.spec_from_file_location('source_preservation_pilot',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def configs():
    base={'initialization':{'path':'/data/parent.pt','sha256':'a'*64},'seed':42,'sources':[{'source_id':'train'}],
        'stage':{'phase':'S3','budget_updates':12,'routes':[{'controls':{},'loss':{'auxiliary_every':1,'auxiliary_sigma_max':.5}}]},
        'model_manifest':{'sha256':'b'*64},'preservation':None,'preservation_index':None}
    control={k:v for k,v in base.items() if not k.startswith('preservation')}
    modified=deepcopy(base);modified['preservation']={'source_identity':.2,'background':.1}
    modified['preservation_index']={'path':'/data/index.json','sha256':'c'*64}
    return control,modified


def test_paired_pilot_accepts_loss_only_change():
    assert module().validate_pair(*configs())==12


@pytest.mark.parametrize('field',('seed','parent','budget','sources','pose','missing_index'))
def test_paired_pilot_rejects_confounding_or_unbound_changes(field):
    control,modified=configs()
    if field=='seed':modified['seed']=43
    if field=='parent':modified['initialization']['sha256']='d'*64
    if field=='budget':modified['stage']['budget_updates']=300
    if field=='sources':modified['sources'][0]['source_id']='other'
    if field=='pose':modified['stage']['routes'][0]['controls']['body_strength']=4
    if field=='missing_index':modified['preservation_index']=None
    with pytest.raises(ValueError):module().validate_pair(control,modified)

"""A reviewed subset must never be published as a full-dataset binding."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import pytest


def validator():
    default=Path(__file__).resolve().parents[1]/'experiments/native-spatial-supervision/scripts/finalize_spatial_full.py'
    spec=importlib.util.spec_from_file_location('spatial_full_finalize_test',os.environ.get('NATIVE_SPATIAL_FINALIZER',str(default)))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_full_scope


def inputs():
    donor={'path':'/data/full/index.json','sha256':'a'*64};candidate={'path':'/data/full-v2/index.json','sha256':'b'*64}
    source={'path':'/data/full/frames.jsonl','sha256':'f'*64}
    baseline={'entries':{'one':{},'two':{}},'config':{'frames':source}}
    approved={'config':{'body_cache_index':donor}}
    new={'entries':{'one':{},'two':{}},'config':{'frames':source}}
    audit={'passed':True,'index':candidate,'frames':2,'source_frames':2,'source_manifest_sha256':'f'*64}
    return baseline,approved,new,audit,donor,candidate


def test_full_scope_binds_complete_donor_and_source_manifest():
    validator()(*inputs())


@pytest.mark.parametrize('bad',['subset_donor','source_count','source_sha','candidate_source','index_binding'])
def test_subset_or_mismatched_scope_is_rejected(bad):
    baseline,approved,new,audit,donor,candidate=deepcopy(inputs())
    if bad=='subset_donor':
        baseline['entries'].pop('two');new['entries'].pop('two');audit.update(frames=1,source_frames=1)
        donor={'path':'/data/subset/index.json','sha256':'c'*64}
    elif bad=='source_count':audit['source_frames']=3
    elif bad=='source_sha':audit['source_manifest_sha256']='c'*64
    elif bad=='candidate_source':new['config']['frames']={'path':'/data/subset/frames.jsonl','sha256':'d'*64}
    else:audit['index']={'path':'/data/another/index.json','sha256':'e'*64}
    with pytest.raises(ValueError):validator()(baseline,approved,new,audit,donor,candidate)

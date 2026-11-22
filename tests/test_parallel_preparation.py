"""Independent cache inputs must match the eventual Native training contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module(name,monkeypatch):
    p=Path(__file__).parents[1]/f'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/{name}.py'
    monkeypatch.syspath_prepend(str(p.parent));spec=importlib.util.spec_from_file_location('xhs_'+name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_sam_gate_prefers_a_qualified_candidate_and_marks_missing_annotation(monkeypatch):
    m=module('sam_quality',monkeypatch)
    masks=np.ones((2,4,4),bool);scores=np.array([.99,.7]);coverage=np.array([.6,.8])
    index,accepted=m.choose_mask(masks,scores,coverage)
    assert index==1 and accepted
    index,accepted=m.choose_mask(masks,scores,np.array([.6,.5]))
    assert index==0 and not accepted


def test_unqualified_segmentation_has_no_spatial_gt_or_background_metric(monkeypatch):
    from src.data.canvas_contract import native_pad
    m=module('assets',monkeypatch);spec=native_pad((32,64))
    sd={'points134_raw':[[15,15]]*134,'scores134_raw':[.9]*134,'valid_mask134':[True]*134,
        'bbox_xyxy':[1,1,60,30],'height':32,'width':64}
    arrays,meta=m.build_regions(sd,np.ones((32,64),np.uint8),spec,['person','__inactive__'],'a'*64,'b'*64,.7,segmentation_accepted=False)
    assert arrays['valid'].sum()>0
    for family in ('person','joint','hands','background'):
        assert not arrays[family].any() and not arrays[family+'_quality'].any()
    assert meta['segmentation_accepted'] is False


def test_cache_binding_checks_exact_condition_set_and_native_provenance(monkeypatch):
    m=module('cache_preparation',monkeypatch)
    native={'x\0p':{'condition_sha256':'cond','source_rgb_sha256':'s','driver_rgb_sha256':'t',
        'source_canvas_id':'sc','target_canvas_id':'tc'}}
    cache={'condition_bindings':{'x\0p':'cond'},'entries':{'x\0p':{**native['x\0p'],'file':{'path':'unused'}}}}
    m.validate_native_bindings(cache,native)
    with pytest.raises(ValueError,match='provenance'):
        m.validate_native_bindings(cache,{'x\0p':{**native['x\0p'],'driver_rgb_sha256':'other'}})
    with pytest.raises(ValueError,match='condition'):
        m.validate_native_bindings(cache,{})


def test_reuse_verifies_file_bytes_without_rejecting_typed_schema(tmp_path,monkeypatch):
    m=module('assets',monkeypatch);p=tmp_path/'asset.npz';p.write_bytes(b'frozen')
    typed={**m.ref(p),'schema':'native_smplx_asset_v4'}
    m.verify_artifact(typed)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact'):m.verify_artifact(typed)

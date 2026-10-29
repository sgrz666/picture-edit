"""Quality comparisons must change one declared control, not stage evidence."""
import importlib.util
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def test_control_variants_are_independent_and_keep_depth_detail():
    from abc_variants import control_records
    full, base = control_records('full')
    detail, same_base = control_records('no_source_detail')
    assert same_base == base and base['detail'] is True
    assert {k for k in full if full[k] != detail[k]} == {'detail'}
    body, body_base = control_records('body_interaction')
    assert not body['face'] and not body['hand'] and not body['detail']
    assert body['interaction'] and body_base == base
    only, only_base = control_records('body_only')
    assert not only['interaction']
    assert not any(only_base[k] for k in ('interaction', 'raster', 'part_contact'))
    assert only_base['identity'] and only_base['depth_mode'] == 'full_geometry'
    with pytest.raises(ValueError):
        control_records('typo')


def test_stage_tags_are_siblings_and_preserve_legacy_paths():
    from abc_variants import generation_key
    assert generation_key('native_v5', 'full', None) == 'native_v5'
    assert generation_key('native_v5', 'no_hand', None) == 'native_v5__no_hand'
    assert generation_key('native_v5', 'full', 's42_full') == 'native_v5__s42_full'
    with pytest.raises(ValueError):
        generation_key('native_v5', 'full', '../escape')


def test_cli_rejects_native_ablation_on_baseline():
    from abc_runner import parse_args
    accepted=parse_args(['--source-a','a','--source-b','b','--interaction','c',
        '--checkpoint','weights','--variant','no_hand','--generation-tag','s42'])
    assert accepted.variant=='no_hand' and accepted.generation_tag=='s42'
    with pytest.raises(SystemExit):
        parse_args(['--source-a', 'a', '--source-b', 'b', '--interaction', 'c',
                    '--mode', 'qwen_baseline', '--variant', 'no_hand'])


def test_campaign_matrix_has_fixed_counts_and_unique_jobs():
    path = HERE / 'scripts/quality-baseline/script/campaign.py'
    assert path.exists(), 'quality campaign must be implemented'
    spec = importlib.util.spec_from_file_location('quality_campaign', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    jobs = module.matrix()
    assert len(jobs) == 68
    assert {p: sum(j['phase'] == p for j in jobs) for p in
            ('smoke', 'formal', 'swap', 'diagnose')} == dict(smoke=12, formal=36, swap=4, diagnose=16)
    assert len({(j['case'], j['resolution'], j['tag']) for j in jobs}) == 68


def test_observer_never_changes_tensor_or_requires_grad():
    import torch
    from src.pose_control.qwen21_full.native_v5.observation import ResidualObserver, observe_residual
    value = torch.randn(1, 8, 16, requires_grad=True)
    original = value.detach().clone()
    observer = ResidualObserver()
    with observer.capture():
        observe_residual('body', value, layer=5, sigma=torch.tensor([.5]), gain=torch.ones(1))
    assert torch.equal(value.detach(), original)
    assert value.requires_grad and value.grad is None
    assert observer.records[0]['rms'] > 0
    assert observer.records[0]['layer'] == 5
    assert not any(torch.is_tensor(x) for row in observer.records for x in row.values())


def test_background_support_excludes_canvas_padding_and_people():
    import numpy as np
    from abc_quality import background_support
    valid = np.zeros((64, 64), bool); valid[8:56, 8:56] = True
    source = np.zeros_like(valid); source[20:30, 20:30] = True
    target = np.zeros((2, 64, 64), bool); target[0, 35:45, 35:45] = True
    support = background_support(valid, source, target, dilation=1)
    assert not support[0].any() and not support[:, 0].any()
    assert not support[source].any() and not support[target.any(0)].any()
    assert support[10, 10]


def test_contact_projection_proxy_reports_missing_detection():
    import numpy as np
    from abc_quality import contact_projection_proxy
    joints=np.zeros((2,137,3));joints[...,2]=2.
    joints[0,20]=[.2,.1,2.];joints[1,21]=[.22,.1,2.]
    points=np.array([[[.2,.1,2.],[.22,.1,2.]]])
    arrays={'joints_137_cam':joints,'K':np.diag([100.,100.,1.]),
        'contact_indices':np.array([[0,10,1,20]]),'contact_points_cam':points}
    requested=[np.zeros((134,2)),np.zeros((134,2))]
    result=contact_projection_proxy(arrays,requested,[None,None])
    assert result['comparable'] == 0 and result['missing_endpoints'] == 2
    assert result['mean_relative_error_px'] is None


def test_missing_body_points_are_not_counted_as_correct():
    from abc_quality import pose_coverage
    value=pose_coverage([{'body_joints':18,'pck@0.2':1.}, {'comparable':False}])
    assert value['detected_points']==18 and value['requested_points']==36
    assert value['coverage']==.5 and value['pck_02_all_requested']==.5


def test_report_pairs_diagnostics_to_same_seed_full_control():
    directory=HERE/'scripts/quality-baseline/script'
    sys.path.insert(0,str(directory))
    from quality_report import ablation_pairs
    def row(phase,label,variant):
        return {'job':dict(phase=phase,case='hug',seed=42,label=label,variant=variant,mode='native_v5',diagnostic='none'),
            'evaluation':{'pose':{'persons':[]}},'image':label}
    rows=[row('formal','full','full'),row('diagnose','no_hand','no_hand')]
    pairs=ablation_pairs(rows)
    assert len(pairs)==1 and pairs[0]['comparison']=='no_hand_vs_full'
    assert pairs[0]['seed']==42


def test_geometry_audit_rejects_owner_depth_disagreement():
    import numpy as np
    from abc_quality import geometry_consistency
    depth=np.stack([np.ones((2,2)),np.ones((2,2))*2])
    owner=np.zeros((2,2,2),bool);owner[1]=True
    values=dict(depth=depth,mask=np.ones_like(owner),visible_owner=owner,
        normal_cam=np.zeros((2,2,2,3)),combined_depth=np.ones((2,2))*2,combined_normal_cam=np.zeros((2,2,3)))
    with pytest.raises(ValueError,match='Owner'):
        geometry_consistency(values)


def test_source_feature_cache_reuses_only_verified_binding(tmp_path):
    import numpy as np
    from abc_feature_cache import frozen_source_features
    calls=[]
    def build():calls.append(1);return {'feature':np.ones((2,3),np.float32)}
    one=frozen_source_features(tmp_path,{'rgb':'a','canvas':'x'},build)
    two=frozen_source_features(tmp_path,{'rgb':'a','canvas':'x'},build)
    assert len(calls)==1 and np.array_equal(one['feature'],two['feature'])
    payload=next(tmp_path.rglob('*.npz'));payload.write_bytes(b'corrupted')
    with pytest.raises(ValueError):frozen_source_features(tmp_path,{'rgb':'a','canvas':'x'},build)
    frozen_source_features(tmp_path,{'rgb':'b','canvas':'x'},build)
    assert len(calls)==2


def test_pinned_asset_snapshot_preserves_content_identity_and_rejects_drift(tmp_path):
    from abc_runner import asset_identity,sha256
    path=tmp_path/'model.bin';path.write_bytes(b'weights')
    stat=path.stat();snapshot={str(path.resolve()):dict(sha256=sha256(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)}
    assert asset_identity(tmp_path,snapshot=snapshot)==asset_identity(tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen asset'):
        asset_identity(tmp_path,snapshot=snapshot)

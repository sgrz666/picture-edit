import importlib.util
from pathlib import Path
import pytest
import torch


def driver():
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/rl.py'
    spec=importlib.util.spec_from_file_location('bilibili_region_rl_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_reference_weight_context_restores_parameter_objects_on_failure():
    from src.pose_control.qwen21_full.native_rl.runner import FrozenReference

    net = torch.nn.Linear(2, 1)
    reference = FrozenReference(net.parameters(), storage_device='cpu')
    with torch.no_grad():
        net.weight.add_(1.)
    current = net.weight.detach().clone()
    ids=[id(p) for p in net.parameters()]
    with torch.no_grad(), pytest.raises(RuntimeError, match='controlled'):
        with reference.use():
            assert torch.equal(net.weight, reference.weights[0])
            raise RuntimeError('controlled reference-probe failure')
    assert [id(p) for p in net.parameters()]==ids
    assert torch.equal(net.weight, current)


def test_regional_reward_columns_keep_generation_misses_and_reject_invalid_measurements():
    score = driver().score_columns
    applicable = {'body': True, 'hands': False, 'identity': True, 'background': False}
    row = {'pose': {'body': {'reward': .5}, 'hands': {'reward': None}},
           'identity': {'identity_reward': 0.}, 'pixel': {'background_reward': None}}
    assert score([row], applicable) == {'body': [.5], 'identity': [0.]}
    row['identity']['identity_reward'] = None
    with pytest.raises(ValueError, match='fixed target eligibility'):
        score([row], applicable)


def test_regional_gate_uses_independent_validation_metrics_and_rejects_missing_values():
    from src.pose_control.qwen21_full.native_rl.validation import regressions

    metrics = driver().ode_metrics
    summary = {'pose': {'body': {'reward': .88}, 'hands': {'training_reward': .75, 'gesture_reward': .60}},
               'identity': {'identity_reward': {'mean': .91}}, 'pixel': {'background_reward': {'mean': .7}}}
    baseline = metrics(summary)
    assert baseline == {'pose': .88, 'hands': .75, 'hands_gesture': .60, 'identity': .91, 'background': .7}
    del summary['pose']['hands']['gesture_reward']
    summary['identity']['identity_reward']['mean'] = .87
    assert regressions(metrics(summary), baseline, {'pose': .02, 'identity': .02, 'hands_gesture': .02}) == [
        'identity', 'hands_gesture']


def test_regional_driver_rolls_back_and_resumes_terminal_gate_without_updates(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace as Box

    from PIL import Image
    from src.pose_control.qwen21_full.native_rl import runner
    from src.pose_control.qwen21_full.native_v62.training import checkpointing, data, io, validation
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    from src.pose_control.qwen21_full.native_v7 import training

    module = driver()
    parent = tmp_path/'parent.pt'
    torch.save({'training_state': {'config': {'stage': {'stage_id': 'S3'}}, 'global_update': 8,
                                  'stage_update': 8, 'parent': None, 'stage_origin': None}}, parent)
    data_dir, report_dir = tmp_path/'data', tmp_path/'report'
    data_dir.mkdir()
    report_dir.mkdir()
    (report_dir/'model-identity-sdpose.json').write_text('{}')
    (data_dir/'evaluation.json').write_text('[{"pair_id":"train","split":"train"},{"pair_id":"val","split":"val"}]')
    cfg_path = tmp_path/'cfg.json'
    cfg_path.write_text('{}')
    route = Box(name='joint')
    cfg = Box(initialization={'path': str(parent), 'sha256': file_sha(parent)}, signature=lambda: 'config',
              stage=Box(routes=[route], optimizer=Box(scheduler='constant', clip_norm=1.)),
              output_root=tmp_path/'output/grpo', generation_steps=20, generation_seed=42)
    condition = {'path': str(cfg_path), 'sha256': file_sha(cfg_path)}
    val_data = Box(records=[{'source_id': 'v', 'index': 0, 'pair_id': 'val', 'role': 'validation', 'eligible': True}],
                   conditions={('v',0):(condition,{})}, sample=lambda *_: {})

    class Policy(torch.nn.Linear):
        def checkpoint_payload(self, *, metadata):
            return {'state_dict': self.state_dict()}

    def start(*_):
        runtime = Policy(1, 1, bias=False)
        catalog = Box(parameters={'weight': runtime.weight}, signature=lambda: 'catalog')
        backend = Box(runtime=runtime, active={'weight'}, catalog=catalog, pipe=Box(transformer=torch.nn.Linear(1,1)),
                      base_state_names=set(), data=val_data, identity={'code':'test'},
                      pair=lambda *_: ({'pair_metadata':{'pair_id':'train'},'native_scene_kind':'single'}, {}, {}))
        return {'backend':backend,'optimizer':torch.optim.AdamW(runtime.parameters()),'global_update':8,
                'journal':Box(append=lambda *_: None,state_dict=lambda: {}),
                'sampler':Box(draw=lambda:('joint','t',0),state_dict=lambda: {},load_state_dict=lambda *_: None)}

    def evaluator(_, **options):
        backend = 'adaface_kprpe' if options.get('adaface_model') else 'glintr100'
        return Box(prepare=lambda *_a,**_k:{'eligible':{'body':True}}, identity_backend=backend,
                   model_identity={'sha256':backend}, background_index=None)

    def evaluate(root, manifest, folder, metric, targets):
        images = json.loads(manifest.read_text())['records']
        return {'records':[{**image,'pose':{'body':{'reward':i/4},'hands':{'reward':None}},
                            'identity':{'identity_reward':None},'pixel':{'background_reward':None}}
                           for i,image in enumerate(images)]}

    ode_scores = iter((.9, .9, .7, .7))
    observed = []
    def evaluate_ode(root, backend, route, cohort, folder, metric, targets, **kwargs):
        assert cohort[0]['pair_id'] == 'val' and metric.identity_backend == 'glintr100'
        observed.append(folder.name)
        return {'pose':next(ode_scores), 'identity':.9}

    updates = []
    def update(optimizer, groups, active, frozen, **kwargs):
        with torch.no_grad():
            active[0].add_(1.)
        updates.append(1)
        return {'max_ratio_error':0.,'grad_norm':1.}

    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *_:None)
    monkeypatch.setattr(training,'TrainingConfigV7',Box(from_dict=lambda _:cfg))
    monkeypatch.setattr(training,'NativeV7Trainer',lambda _:Box(_start=start))
    monkeypatch.setitem(sys.modules,'evaluate',Box(IdentityPixelEvaluator=evaluator))
    monkeypatch.setattr(data,'route_options',lambda *_:({},{}))
    monkeypatch.setattr(checkpointing,'environment',lambda:{})
    monkeypatch.setattr(io,'verify_prefix',lambda *_:None)
    monkeypatch.setattr(validation,'frozen_digest',lambda *_:{})
    monkeypatch.setattr(module,'audit_rewards',lambda *a:a[2])
    monkeypatch.setattr(module,'evaluate',evaluate)
    monkeypatch.setattr(module,'evaluate_ode',evaluate_ode)
    monkeypatch.setattr(runner,'NativeRollout',lambda *a,**k:Box(sde_step=k['sde_step'],predict=None,
        generate=lambda *a,**k:(Image.new('RGB',(2,2)),Box(state=torch.zeros(1)))))
    monkeypatch.setattr(runner,'fixed_reference_means',lambda *a,**k:{})
    monkeypatch.setattr(runner,'update_policy',update)
    args = Box(root=tmp_path,config=cfg_path,parent=parent,report=report_dir/'first.json',resume=None,updates=5,
               adaface_model='test',adaface_rpe_extension='test',reward_profile='regional-legacy',
               spatial_model=None,spatial_weight=0.,spatial_std_floor=None,sde_indices=[1,2,3],val_limit=1,
               validate_every=1,patience=2,ode_pose_device='cpu',pose_drop=.02,id_drop=.02,hands_drop=.02,
               two_person_drop=.02,background_drop=.02,reward_arguments=[])
    module.train(args,None)
    result = json.loads(args.report.read_text())
    assert result['stop']=='ode_regression' and result['selected_update']==1 and len(updates)==3
    assert result['checkpoint'] != result['last_checkpoint']
    assert observed==['sft','update_000001','update_000002','update_000003']
    args.resume=Path(result['last_checkpoint']['path'])
    args.report=report_dir/'recovered.json'
    module.train(args,None)
    assert json.loads(args.report.read_text()) == result and len(updates)==3


def test_regional_spatial_group_is_committed_in_rollout_order_before_advantage(tmp_path):
    import json
    from types import SimpleNamespace

    module = driver()
    fixed = SimpleNamespace(record=lambda: {'roles': ['source']})
    records = [{'path': f'{i}.png', 'sha256': str(i), 'pose': {'body': {'reward': .5}, 'hands': {'reward': None}},
                'identity': {'identity_reward': None}, 'pixel': {'background_reward': None}} for i in range(2)]

    def score(target, items):
        assert target is fixed
        assert [item['sha256'] for item in items] == ['0', '1']
        return [{'id': i, 'reward': score} for i, score in enumerate((.42, .84))]

    module.score_spatial_group(SimpleNamespace(score_group=score), fixed, records, tmp_path)
    rows = json.loads((tmp_path / 'spatial.json').read_text())['candidates']
    assert [row['file']['sha256'] for row in rows] == ['0', '1']
    assert module.score_columns(records, {'body': True, 'spatial': True}) == {'body': [.5, .5], 'spatial': [.42, .84]}

    folder = tmp_path / 'incomplete'
    with pytest.raises(ValueError):
        module.score_spatial_group(SimpleNamespace(score_group=lambda *_: rows[:1]), fixed, records, folder)
    assert (folder / 'evaluation-failed.json').is_file() and not (folder / 'spatial.json').exists()


def test_background_scores_source_on_verified_support_and_disables_unknown_alignment(tmp_path):
    import numpy as np
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_rl.background import prepare_background
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha

    canvas = native_pad((64, 64))
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    gt = np.full_like(src, 190)
    gt[25:35, 28:35] = 255
    source_mask = src[:, :, 0].copy()
    source_mask[25:35, 18:25] = 255
    refs = {}
    for name, image in (('source_rgb', src), ('target_rgb', gt), ('source_mask', source_mask)):
        path = tmp_path / f'{name}.png'
        Image.fromarray(image).save(path)
        refs[name] = {'path': str(path), 'sha256': file_sha(path)}
    pair = {**refs, 'source_canvas': canvas.metadata(), 'target_canvas': canvas.metadata()}
    target_mask = gt[:, :, 0] > 190
    assert prepare_background({**pair, 'source_mask': None}, Image.fromarray(src), target_mask)[0] is None
    pair['background_alignment'] = {'method': 'identity', 'source_rgb_sha256': refs['source_rgb']['sha256'],
                                    'target_rgb_sha256': refs['target_rgb']['sha256'], 'canvas_id': canvas.fingerprint()}
    bg, reason = prepare_background(pair, Image.fromarray(src), target_mask)
    assert reason is None and bg.score(src)['background_reward'] == pytest.approx(1.)
    assert bg.score(src)['background_reward'] > bg.score(gt)['background_reward']
    assert bg.score(np.full_like(src, 255))['background_reward'] < .001
    assert prepare_background({**pair, 'source_mask_accepted': False}, Image.fromarray(src), target_mask) == (
        None, 'source_foreground_unqualified')

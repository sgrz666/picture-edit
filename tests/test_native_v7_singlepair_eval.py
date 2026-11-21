"""CPU checks for the pilot's scoring and immutable report boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
import weakref
import sys

import numpy as np
import pytest


class FakePosePipeline:
    pass


def evaluator():
    path = Path(__file__).resolve().parents[1] / 'experiments/hi4d_qwen21_dual_20260930/native-v7-singlepair-overfit-20261117/scripts/evaluate_run.py'
    spec = importlib.util.spec_from_file_location('singlepair_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requested():
    points = np.zeros((2, 21, 2))
    for person, x in enumerate((180, 60)):  # Slot 0 deliberately to the right.
        points[person, :, 0] = x
        points[person, :, 1] = 90
        points[person, (15, 16), 1] = 40
    return points, np.ones((2, 21), dtype=bool)


def detection(points):
    result = np.zeros((134, 2))
    result[2:14] = points[np.array((16, 18, 20, 15, 17, 19, 1, 4, 7, 0, 3, 6))]
    return {'points': result, 'scores': np.ones(134), 'box_xyxy': [0, 0, 255, 255], 'box_score': .9}


def test_anatomical_mapping_and_hip_assignment_preserve_requested_identity():
    module = evaluator()
    points, valid = requested()
    people = [detection(points[1]), detection(points[0])]
    result = module.pose_metrics(points, valid, people, (256, 256), raw_count=3)
    assert module.BODY21_COMMON == (16, 18, 20, 15, 17, 19, 1, 4, 7, 0, 3, 6)
    assert result['assignment'] == [1, 0]
    assert result['raw_persons_detected'] == 3
    assert result['all']['pck@0.1'] == 1
    assert result['all']['mpjpe_px'] == 0
    assert result['persons'][0]['torso_length_px'] == 50


@pytest.mark.parametrize('count', (0, 1, 3))
def test_failed_person_count_retains_null_main_metrics(count):
    module = evaluator()
    points, valid = requested()
    result = module.pose_metrics(points, valid, [detection(points[0])] * count, (256, 256), raw_count=count)
    assert result['all'] is None
    assert result['assignment'] is None
    assert result['reason'] == 'foreground_person_count_not_two'


def test_normalized_scores_confidence_and_canvas_are_applied_per_joint():
    module = evaluator()
    points, valid = requested()
    people = [detection(points[0]), detection(points[1])]
    people[0]['points'][2:14, 0] += 7.5  # 0.15 requested torso lengths.
    people[0]['scores'][2] = .29
    people[0]['points'][3] = [-1, 30]
    valid[0, 20] = False
    result = module.pose_metrics(points, valid, people, (256, 256), raw_count=2)
    assert result['persons'][0]['valid_joints'] == 9
    assert result['persons'][0]['pck@0.1'] == 0
    assert result['persons'][0]['pck@0.2'] == 1
    assert result['persons'][0]['nme'] == pytest.approx(.15)
    assert result['all']['valid_joints'] == 21
    assert result['all']['nme'] == pytest.approx(9 * .15 / 21)


def test_missing_hips_does_not_use_face_or_left_to_right_fallback():
    module = evaluator()
    points, valid = requested()
    people = [detection(points[0]), detection(points[1])]
    people[0]['scores'][8] = 0
    result = module.pose_metrics(points, valid, people, (256, 256), raw_count=2)
    assert result['all'] is None
    assert result['reason'] == 'hip_center_unavailable'


def test_ambiguous_assignment_and_all_unscorable_joints_stay_explicit():
    module = evaluator()
    points, valid = requested()
    points[1] = points[0]
    people = [detection(points[0]), detection(points[1])]
    people[0]['scores'][:] = 0
    people[1]['scores'][:] = 0
    for person in people:
        person['scores'][[8, 11]] = 1
    result = module.pose_metrics(points, valid, people, (256, 256), raw_count=2)
    assert result['ambiguous'] is True
    assert result['all']['valid_joints'] == 4  # Hips remain measurable.
    valid[:, [15, 16]] = False
    result = module.pose_metrics(points, valid, people, (256, 256), raw_count=2)
    assert result['all'] is None
    assert result['persons'][0]['reason'] == 'requested_torso_unavailable'


def test_background_excludes_both_body_supports_and_padding():
    module = evaluator()
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    generated = source.copy()
    generated[0, 0] = 200  # Source foreground.
    generated[1, 1] = 200  # Target foreground.
    generated[:, 4] = 200  # Padding.
    valid = np.ones((4, 5), bool)
    valid[:, 4] = False
    src_mask = np.zeros((2, 1, 4, 5), bool)
    target = np.zeros((1, 2, 1, 4, 5), bool)
    src_mask[0, 0, 0, 0] = True
    target[0, 1, 0, 1, 1] = True
    result = module.background_metrics(source, generated, valid, src_mask, target)
    assert result == {'valid_pixels': 14, 'mae_rgb_0_255': 0., 'psnr_db': None}
    generated[2, 2] = 10
    result = module.background_metrics(source, generated, valid, src_mask, target)
    assert result['mae_rgb_0_255'] == pytest.approx(10 / 14)
    assert result['psnr_db'] == pytest.approx(10 * np.log10(255**2 / (100 / 14)))


def test_generation_records_are_verified_and_report_directory_never_reused(tmp_path):
    module = evaluator()
    artifact = tmp_path / 'actual.png'
    artifact.write_bytes(b'image bytes')
    record = {'path': str(artifact), 'sha256': module.reference(artifact)['sha256'],
              'source_id': 'train', 'pair_id': 'p', 'variant': 'native', 'seed': 42}
    manifest = tmp_path / 'generation.json'
    manifest.write_text(json.dumps({'records': [record]}))
    assert module.generation_records([manifest])[0]['record'] == record
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        module.generation_records([manifest])
    folder = tmp_path / 'new-report'
    module.new_report(folder)
    with pytest.raises(FileExistsError):
        module.new_report(folder)


def test_json_rejects_nonfinite_values_and_never_overwrites(tmp_path):
    module = evaluator()
    path = tmp_path / 'report.json'
    module.write_json(path, {'unavailable': None})
    with pytest.raises(FileExistsError):
        module.write_json(path, {})
    with pytest.raises(ValueError):
        module.write_json(tmp_path / 'invalid.json', {'bad': float('nan')})


def test_identity_swap_uses_known_slots_and_mean_margin():
    module = evaluator()
    result = module.identity_summary(np.array([[.7, .75], [.74, .7]]))
    assert result['diagonal_mean'] == pytest.approx(.7)
    assert result['swap_suspected'] is True
    result = module.identity_summary(np.array([[.7, .72], [.71, .7]]))
    assert result['swap_suspected'] is False


def test_identity_metrics_loads_torchscript_without_module_requires_grad(tmp_path, monkeypatch):
    import torch
    module = evaluator()
    path = tmp_path / 'identity.ts'
    torch.jit.trace(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3)).save(str(path))
    prepared = tmp_path / 'prepared.json'
    prepared.write_text(json.dumps({'qualification': []}))
    loaded = []
    original_load = torch.jit.load
    def capture(*args, **kwargs):
        model = original_load(*args, **kwargs)
        loaded.append(model)
        return model
    monkeypatch.setattr(torch.jit, 'load', capture)
    monkeypatch.setattr(module, 'generation_records', lambda _: [])
    args = SimpleNamespace(identity_model=path, prepared=prepared, generation_report=['unused'])
    result = module.identity_metrics(args, {'sources': []})
    assert result['records'] == []
    assert len(loaded) == 1 and loaded[0].training is False
    assert all(not parameter.requires_grad for parameter in loaded[0].parameters())


def test_identity_metrics_gt_control_uses_loaded_canvas_once_per_pair(tmp_path, monkeypatch):
    import torch
    from PIL import Image
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62 import data
    from src.pose_control.qwen21_full.native_v5 import losses
    module = evaluator()
    model_path = tmp_path / 'identity.ts'
    model_path.write_bytes(b'test evaluator')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}\n')
    generated = tmp_path / 'generated.png'
    Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(generated)
    canvas = CanvasSpec((32, 30), (32, 30), (32, 32)).metadata()
    sample = {'tgt_image': torch.full((3, 32, 32), -.5), 'target_canvas': canvas,
              'local_condition_v4': SimpleNamespace(source_indices=torch.tensor([[1, 0]]),
                  metadata=[{'target_person_ids': ['A', 'B']}]),
              'pair_metadata': {'pair_id': 'p', 'target_rgb_sha256': 'target-sha'}}
    class Dataset:
        rows = [{'pair_id': 'p'}]
        def __init__(self, manifest, *, load_target, data_profile):
            assert load_target is True, 'GT must come from the SHA-verified typed reader'
        def __getitem__(self, index):
            return sample
    prepared = tmp_path / 'prepared.json'
    expected = [[.83, .02], [.01, .82]]
    prepared.write_text(json.dumps({'qualification': [{'pair_id': 'p', 'source_target_similarity': expected}]}))
    calls = []
    def aligned(sample, pixels, qualification, model):
        calls.append(int(pixels[0, 0, 0]))
        return module.identity_summary(np.array(expected if calls[-1] == 64 else [[.1, .02], [.01, .2]]))
    monkeypatch.setattr(module, 'aligned_identity', aligned)
    monkeypatch.setattr(data, 'NativeFullDatasetV62', Dataset)
    monkeypatch.setattr(losses, 'frozen_evaluators', lambda *a, **k: (object(), None))
    monkeypatch.setattr(module, 'generation_records', lambda _: [
        {'record': {'path': str(generated), 'source_id': 'train', 'pair_id': 'p', 'variant': variant}}
        for variant in ('qwen_only', 'native')])
    result = module.identity_metrics(SimpleNamespace(identity_model=model_path, prepared=prepared,
        generation_report=['unused']), {'sources': [{'source_id': 'train', 'manifest': module.reference(manifest), 'data_profile': 'hybrid_hi4d'}]})
    assert calls.count(64) == 1 and calls.count(128) == 2
    control = result['positive_controls'][0]
    assert control['source_indices'] == [1, 0]
    assert control['gt_identity']['diagonal'] == [.83, .82]
    assert control['diagonal_delta_vs_preparation'] == [0., 0.]


@pytest.mark.parametrize('options,uses_budget,device', [(['--pose-device', 'cpu'], False, 'cpu'), ([], True, 'cuda')])
def test_image_metrics_cpu_bypasses_cuda_budget_and_default_keeps_it(tmp_path, monkeypatch, options, uses_budget, device):
    module = evaluator()
    config = tmp_path / 'config.json'
    config.write_text('{}')
    budget_calls = []
    def budget(fraction):
        budget_calls.append(fraction)
        return {'device': 'cuda', 'memory_fraction': fraction}
    monkeypatch.setattr(module, 'gpu_budget', budget)
    monkeypatch.setattr(module, 'image_metrics', lambda args, config: {'pose_device': args.pose_device})
    output = tmp_path / 'report'
    module.main(['--action', 'image-metrics', '--config', str(config), '--report-dir', str(output)] + options)
    report = json.loads((output / 'evaluation.json').read_text())
    assert bool(budget_calls) is uses_budget
    assert report['execution']['device'] == report['pose_device'] == device


def test_image_metrics_forwards_cpu_to_sdpose_runtime(tmp_path, monkeypatch):
    import src.evaluation
    module = evaluator()
    code = tmp_path / 'src/evaluation/sdpose.py'
    code.parent.mkdir(parents=True)
    code.write_text('# test SDPose interface')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/SDPOSE_POSE_EVAL_SPEC.md').write_text('test spec')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'yolo11x.pt').write_bytes(b'YOLO')
    calls = []
    def load_runtime(*, device):
        calls.append(device)
        return SimpleNamespace(device=device, ckpt_dir=weights, pipe=FakePosePipeline())
    sdpose = SimpleNamespace(__file__=str(code), load_runtime=load_runtime)
    monkeypatch.setitem(sys.modules, 'src.evaluation.sdpose', sdpose)
    monkeypatch.setattr(src.evaluation, 'sdpose', sdpose, raising=False)
    monkeypatch.setattr(module, 'generation_records', lambda _: [])
    result = module.image_metrics(SimpleNamespace(generation_report=['unused'], pose_device='cpu', min_area=0), {'sources': []})
    assert calls == ['cpu'] and result['pose_device'] == 'cpu'
    assert result['per_detection_call_seed'] == 24042
    assert result['pose_pipeline_implementation']['path'] == str(Path(__file__).resolve())


def test_per_detection_seed_repeats_same_image_and_restores_callers_rng():
    import torch
    module = evaluator()
    sdpose = SimpleNamespace(detect_people=lambda image, runtime, **kwargs: torch.rand(8))
    runtime = SimpleNamespace(device='cpu')
    image = np.zeros((16, 16, 3), np.uint8)
    assert not torch.equal(sdpose.detect_people(image, runtime), sdpose.detect_people(image, runtime))
    before = torch.random.get_rng_state().clone()
    first = module.seeded_detect_people(sdpose, image, runtime, return_stats=True)
    second = module.seeded_detect_people(sdpose, image, runtime, return_stats=True)
    assert torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())


def test_per_detection_seed_restores_rng_when_detector_fails():
    import torch
    module = evaluator()
    def fail(*args, **kwargs):
        torch.rand(8)
        raise RuntimeError('real detector failed')
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='real detector failed'):
        module.seeded_detect_people(SimpleNamespace(detect_people=fail), None, SimpleNamespace(device='cpu'))
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('device,index', [('cuda:3', 3), ('cuda', 2)])
def test_detection_seed_selects_only_runtime_cuda_rng_without_gpu(device, index, monkeypatch):
    import torch
    module = evaluator()
    original_fork = torch.random.fork_rng
    devices_seen, devices_entered, seeds = [], [], []
    @contextmanager
    def cpu_fork(*, devices):
        devices_seen.append(devices)
        with original_fork(devices=[]):
            yield
    def cuda_device(selected):
        devices_entered.append(selected)
        return nullcontext()
    monkeypatch.setattr(torch.random, 'fork_rng', cpu_fork)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device', cuda_device)
    monkeypatch.setattr(torch.cuda, 'manual_seed', seeds.append)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: pytest.fail('Other CUDA RNGs must remain untouched'))
    before = torch.random.get_rng_state().clone()
    sdpose = SimpleNamespace(detect_people=lambda *a, **k: torch.rand(8))
    first = module.seeded_detect_people(sdpose, None, SimpleNamespace(device=device))
    second = module.seeded_detect_people(sdpose, None, SimpleNamespace(device=device))
    assert torch.equal(first, second) and torch.equal(before, torch.random.get_rng_state())
    assert devices_seen == [[index], [index]] and devices_entered == [index, index]
    assert seeds == [24042, 24042]


def test_gt_pose_audit_reads_true_target_and_keeps_failed_case(tmp_path, monkeypatch):
    import torch
    import src.evaluation
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62 import data
    evaluation = evaluator()
    path = Path(evaluation.__file__).with_name('audit_pose_control.py')
    spec = importlib.util.spec_from_file_location('audit_gt_pose', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    points, valid = requested()
    canvas = CanvasSpec((256, 256), (256, 256), (256, 256)).metadata()
    class Dataset:
        def __init__(self, manifest, *, load_target, data_profile):
            assert load_target is True
            self.source = Path(manifest).stem
            self.rows = [{'pair_id': self.source + '-pair'}]
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {'src_image': torch.zeros(3, 256, 256), 'tgt_image': torch.zeros(3, 256, 256),
                    'source_canvas': canvas, 'target_canvas': canvas,
                    'joint_uv': torch.tensor(points[0]), 'joint_uv_b': torch.tensor(points[1]),
                    'projection_valid': torch.tensor(valid[0]), 'projection_valid_b': torch.tensor(valid[1]),
                    'pair_metadata': {'pair_id': self.rows[0]['pair_id'], 'target_rgb_sha256': 'actual-GT'},
                    'local_condition_v4': SimpleNamespace(source_indices=torch.tensor([[0, 1]]),
                        metadata=[{'target_person_ids': ['A', 'B']}])}
    sources = []
    for source in ('train', 'observation'):
        manifest = tmp_path / (source + '.jsonl')
        manifest.write_text('{}\n')
        sources.append({'source_id': source, 'manifest': evaluation.reference(manifest), 'data_profile': 'hybrid_hi4d'})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'sources': sources}))
    code = tmp_path / 'src/evaluation/sdpose.py'
    code.parent.mkdir(parents=True)
    code.write_text('# test SDPose')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'yolo11x.pt').write_bytes(b'YOLO')
    calls, random_values = [], []
    def detect(image, runtime, **options):
        assert image.dtype == np.uint8 and image.shape == (256, 256, 3) and image[0, 0, 0] == 128
        assert options == {'order': 'geometry', 'sort_by': 'score', 'return_stats': True, 'min_area': 0.}
        calls.append(1)
        random_values.append(float(torch.rand(())))
        return ([detection(points[1]), detection(points[0])], 2) if len(calls) == 1 else ([], 0)
    def load(*, device):
        assert device == 'cpu'
        return SimpleNamespace(device='cpu', dtype=torch.float32, ckpt_dir=weights, pipe=FakePosePipeline())
    sdpose = SimpleNamespace(__file__=str(code), load_runtime=load, detect_people=detect)
    monkeypatch.setitem(sys.modules, 'src.evaluation.sdpose', sdpose)
    monkeypatch.setattr(src.evaluation, 'sdpose', sdpose, raising=False)
    monkeypatch.setattr(data, 'NativeFullDatasetV62', Dataset)
    args = SimpleNamespace(config=config, report_dir=tmp_path / 'report', evaluator=Path(evaluation.__file__))
    result = audit.run(args, evaluation)
    assert len(calls) == 2 and len(result['records']) == 2
    assert random_values[0] == random_values[1]
    assert result['per_detection_call_seed'] == 24042
    assert result['execution']['device'] == 'cpu'
    assert result['records'][0]['pose']['all']['pck@0.1'] == 1
    assert result['records'][1]['pose']['all'] is None
    assert result['records'][1]['pose']['reason'] == 'foreground_person_count_not_two'
    assert 'passed' not in result
    assert (args.report_dir / 'evaluation.json').is_file()
    with pytest.raises(FileExistsError):
        audit.run(args, evaluation)


def test_fixed_probe_real_count_and_evaluator_contracts_and_model_release(tmp_path, monkeypatch):
    import torch
    from src.pose_control.qwen21_full.native_v7 import training
    from src.pose_control.qwen21_full.native_v62.training import checkpointing
    module = evaluator()
    identity = tmp_path / 'identity.ts'
    torch.jit.trace(torch.nn.Linear(1, 1).eval(), torch.ones(1, 1)).save(str(identity))
    initial, trained = tmp_path / 'initial.pt', tmp_path / 'trained.pt'
    cfg = training.TrainingConfigV7(stage=training.stage_preset_v7('S3', budget=120),
        initialization={'path': str(trained)}, evaluators={'identity': module.reference(identity)})
    torch.save({'value': 0.}, initial)
    torch.save({'value': 1., 'training_state': {'global_update': 300, 'config': cfg.record()}}, trained)
    route = cfg.stage.routes[0]
    class Plan:
        def __init__(self, config):
            self.datasets = {'train': [None], 'observation': [None]}
            self.status = {(route.name, source, 0): {'face_area': 0, 'hand_area': 0, 'auxiliary_faces': 2}
                           for source in self.datasets}
    references, noises, gates = [], [], []
    class Backend(training.NativeBackendV7):
        def __init__(self, config, data, *, saved, purpose):
            assert purpose == 'grad'
            assert all(ref() is None for ref in references), 'Previous model was retained at the next constructor'
            self.config, self.data, self.device = config, data, torch.device('cpu')
            self.runtime = torch.nn.Linear(1, 1)
            self.pipe = SimpleNamespace(vae=torch.nn.Linear(1, 1))
            self.pipe.vae.use_tiling = False
            self.evaluators = [None, None]
            self.identity = {'metadata_only': saved['value']}
            self.value = saved['value']
            references.extend((weakref.ref(self), weakref.ref(self.runtime), weakref.ref(self.pipe.vae)))
        def pair(self, source, index):
            sample = {'native_scene_kind': 'dual', 'pair_metadata': {'pair_id': source + '-pair'}}
            supervision = SimpleNamespace(to=lambda _: None)
            return sample, {'training_supervision_v5': supervision, 'tgt_image': torch.zeros(1, 3, 2, 2)}, {'target_latent': torch.zeros(1, 2, 64)}
        def forward(self, batch, pair, route, sigma, noise, options, base):
            assert self.runtime.training is False and not torch.is_grad_enabled()
            assert self.evaluators[0] is not None
            assert all(not p.requires_grad for p in self.evaluators[0].parameters())
            assert options.face and options.hand and options.interaction and base.identity
            noises.append(noise.clone())
            return torch.zeros_like(noise), torch.zeros_like(noise), torch.ones(1, 2), noise
        def compute_loss(self, prediction, target, coverage, supervision, noisy, sigma, target_rgb,
                         loss_spec, counts, update, evaluators, *, spatial=None):
            assert update == 4 and counts['velocity'] == 1 and evaluators[0] is self.evaluators[0]
            gates.append((float(sigma[0]), counts['identity'], counts['paper_identity']))
            loss = torch.tensor(self.value)
            return loss, {'weighted_micro_objective': loss, 'window_denominators': counts}
    monkeypatch.setattr(training, 'DataPlanV7', Plan)
    monkeypatch.setattr(training, 'NativeBackendV7', Backend)
    monkeypatch.setattr(checkpointing, 'load_committed', lambda ref, schema: (torch.load(ref['path'], mmap=True, weights_only=True), ref, None))
    monkeypatch.setattr(checkpointing, 'validate_resume', lambda *a, **k: pytest.fail('Resume validation is not evaluation'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    result = module.fixed_probe(SimpleNamespace(initial_checkpoint=initial), cfg.record())
    assert len(result['records']['initial']) == len(result['records']['trained']) == 6
    assert all(torch.equal(noise, noises[0]) for noise in noises)
    assert gates[0][1:] == (2, 2) and gates[1][1:] == gates[2][1:] == (0, 0)
    assert all(row['trained_minus_initial']['weighted_micro_objective'] == 1 for row in result['deltas'])
    assert all(ref() is None for ref in references)
    json.dumps(result, allow_nan=False)

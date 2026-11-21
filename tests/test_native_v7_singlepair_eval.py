"""CPU checks for the pilot's scoring and immutable report boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


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

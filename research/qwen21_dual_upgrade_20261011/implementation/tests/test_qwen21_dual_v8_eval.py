"""Evaluation contracts that do not need the native model or downloaded weights."""
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image
import pytest

from src.evaluation.qwen21_dual_v8_metrics import (
    EvaluationTarget, PersonDetection, aggregate_quality, appearance_binding,
    assign_by_geometry, background_error, mesh_contact_proxy, pck,
    whole_square_dino_inputs,
)
from scripts.eval.eval_qwen21_dual_v8 import (
    configuration_sha256, planned_runs, reusable_output, save_residual_heatmaps,
    target_replacement_probe,
)


def person(box, xy, scores=None):
    xy = np.asarray(xy, dtype=np.float32)
    return PersonDetection(np.asarray(box), xy, np.ones(len(xy)) if scores is None else np.asarray(scores))


def test_geometry_assignment_preserves_actor_correspondence_with_swapped_detection_order():
    boxes = np.array([[0, 0, 30, 80], [70, 0, 100, 80]])
    xy = np.array([[[10, 10], [20, 40]], [[80, 10], [90, 40]]])
    found = [person(boxes[1], xy[1]), person(boxes[0], xy[0])]
    assert assign_by_geometry(found, boxes, xy, np.ones((2, 2)), (100, 100))['indices'] == [1, 0]


def test_geometry_assignment_reports_missing_actor_and_does_not_accept_remote_third_person():
    boxes = np.array([[0, 0, 30, 80], [70, 0, 100, 80]])
    xy = np.array([[[10, 10], [20, 40]], [[80, 10], [90, 40]]])
    found = [person(boxes[0], xy[0]), person([300, 0, 330, 80], [[310, 10], [320, 40]])]
    assignment = assign_by_geometry(found, boxes, xy, np.ones((2, 2)), (100, 100))
    assert assignment['indices'] == [0, -1]
    assert assignment['unassigned_detection_indices'] == [1]


def test_pck_missing_detected_joints_count_as_failure_for_an_assigned_person():
    target = np.array([[10, 10], [30, 30], [50, 50]])
    result = pck(person([0, 0, 100, 100], target, [.9, .1, .1]), target, [.9, .9, .9])
    assert result['eligible_joints'] == 3
    assert result['detected_joints'] == 1
    assert result['pck_005'] == pytest.approx(1 / 3)
    assert result['pck_005_hits'] == 1


def test_pck_missing_person_is_na_with_separate_eligible_denominator():
    result = pck(None, np.zeros((3, 2)), [.9, .9, .25])
    assert result['pck_005'] is None
    assert result['eligible_joints'] == 2
    assert result['failure'] == 'person_not_detected'


def test_pck_excludes_fitted_prior_confidence_point_two_five():
    points = np.array([[1, 1], [400, 400]])
    result = pck(person([0, 0, 512, 512], np.zeros((2, 2))), points, [.9, .25])
    assert result['eligible_joints'] == 1
    assert result['pck_005'] == 1.


def test_appearance_swap_margin_does_not_rematch_identity():
    result = appearance_binding(np.eye(2), np.eye(2)[::-1])
    assert result['original_correspondence_mean'] == 0.
    assert result['swapped_correspondence_mean'] == 1.
    assert result['margin_original_minus_swapped'] == -1.
    assert result['swap_detected'] is True


def test_aggregate_uses_joint_denominators_and_reports_missing_people():
    good = pck(person([0, 0, 20, 20], [[0, 0]]), np.zeros((1, 2)), [.9])
    bad = pck(person([0, 0, 20, 20], [[100, 100]] * 3), np.zeros((3, 2)), [.9] * 3)
    missing = pck(None, np.zeros((2, 2)), [.9] * 2)
    rows = [{'people': [{'body': good}, {'body': bad}], 'detected_people': 2},
            {'people': [{'body': missing}], 'detected_people': 3}]
    result = aggregate_quality(rows)
    assert result['body']['pck_005'] == .25
    assert result['body']['scored_joint_denominator'] == 4
    assert result['body']['eligible_joint_denominator_all'] == 6
    assert result['body']['failed_people'] == 1
    assert result['third_person_rate'] == .5


def test_background_error_excludes_both_source_and_target_people():
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    output = source.copy()
    source_masks = np.zeros((2, 8, 8))
    truth_masks = np.zeros((2, 8, 8))
    source_masks[0, :2, :2] = 1
    truth_masks[1, 6:, 6:] = 1
    output[:2, :2] = 255
    output[6:, 6:] = 255
    result = background_error(output, source, source_masks, truth_masks, dilation=0)
    assert result['mae_rgb_0_1'] == 0.
    assert result['pixel_denominator'] == 56
    assert result['target_gt_masks_used_only_by_evaluator'] is True


def test_mesh_contact_requires_two_generated_reconstructions():
    result = mesh_contact_proxy([np.zeros((14, 3))], np.arange(14), [0, -1], [[4, 21]])
    assert result['part_pair_distances'] is None
    assert result['failure'] == 'fewer_than_two_generated_meshes'


def test_mesh_contact_uses_assigned_generated_surfaces_and_scale_normalization():
    vertices = np.zeros((14, 3)); vertices[:, 1] = np.linspace(0, 2, 14)
    other = vertices.copy(); other[:, 0] = .2
    result = mesh_contact_proxy([vertices, other], np.arange(14), [0, 1], [[4, 18]])
    assert result['part_pair_distances'][0]['distance_m_proxy'] == pytest.approx(.2)
    assert result['part_pair_distances'][0]['distance_over_mean_height'] == pytest.approx(.1)
    assert result['mesh_source'] == 'generated_rgb_multihmr_camera_space'


def test_dino_preprocessing_keeps_entire_square_crop():
    calls = []
    def processor(**kwargs):
        calls.append(kwargs)
        return 'tensor'
    assert whole_square_dino_inputs(processor, Image.new('RGB', (512, 512))) == 'tensor'
    assert calls[0]['images'].size == (224, 224)
    assert calls[0]['do_resize'] is False
    assert calls[0]['do_center_crop'] is False


@pytest.mark.parametrize('missing_actor', [0, 1])
def test_face_crop_failure_does_not_erase_valid_other_actor(missing_actor):
    from scripts.eval.eval_qwen21_dual_v8 import _face_embeddings
    from types import SimpleNamespace
    rgb = np.zeros((512, 512, 3), np.uint8)
    masks = np.zeros((2, 512, 512), np.float32)
    masks[1 - missing_actor, 80:140, 200:260] = 1
    rgb[80:140, 200:260] = [40, 70, 120]
    calls = []
    value = np.array([1., 0.])
    metric = SimpleNamespace(embedding=lambda crop: calls.append(crop) or value)
    embeddings, failures = _face_embeddings(metric, rgb, masks)
    assert embeddings[missing_actor] is None
    assert 'insufficient' in failures[missing_actor]
    assert embeddings[1 - missing_actor] is value
    assert failures[1 - missing_actor] is None
    assert len(calls) == 1


def test_face_actor_isolation_preserves_square_exclusive_crop_pixels():
    from scripts.eval.eval_qwen21_dual_v8 import _face_embeddings
    from src.data.hi4d_smplx_dataset import source_person_crops
    from types import SimpleNamespace
    rgb = np.indices((512, 512)).sum(0).astype(np.uint8)[..., None].repeat(3, axis=-1)
    masks = np.zeros((2, 512, 512), np.float32)
    masks[0, 80:160, 200:280] = 1; masks[1, 120:220, 240:340] = 1
    expected, _, _ = source_person_crops(rgb, masks)
    calls = []
    metric = SimpleNamespace(embedding=lambda crop: calls.append(np.asarray(crop)) or np.ones(2))
    _, failures = _face_embeddings(metric, rgb, masks)
    assert failures == [None, None]
    assert all(np.array_equal(actual, target) for actual, target in zip(calls, expected))


def test_gt_detector_calibration_is_evaluator_only_and_keeps_missing_actor():
    from scripts.eval.eval_qwen21_dual_v8 import _detector_calibration
    from types import SimpleNamespace
    rgb = np.zeros((512, 512, 3), np.uint8)
    boxes = np.array([[0, 0, 30, 80], [70, 0, 100, 80]])
    xy = np.array([[[10, 10], [20, 40]], [[80, 10], [90, 40]]])
    calls = []
    metric = SimpleNamespace(detect=lambda image: calls.append(np.asarray(image)) or [person(boxes[1], xy[1])])
    result = _detector_calibration(metric, rgb, boxes, xy, np.ones((2, 2)), ('A', 'B'))
    assert result['detected_people'] == 1
    assert result['assignment']['indices'] == [-1, 0]
    assert result['undetected_person_uids'] == ['A']
    assert result['evaluator_only'] is True
    assert result['used_to_modify_generated_detections'] is False
    assert len(calls) == 1 and np.array_equal(calls[0], rgb)


def test_gt_detector_calibration_failure_is_na():
    from scripts.eval.eval_qwen21_dual_v8 import _detector_calibration
    from types import SimpleNamespace
    def fail(image): raise RuntimeError('detector failed')
    result = _detector_calibration(SimpleNamespace(detect=fail), np.zeros((512, 512, 3), np.uint8),
        np.zeros((2, 4)), np.zeros((2, 2, 2)), np.ones((2, 2)), ('A', 'B'))
    assert result['detected_people'] is None
    assert result['undetected_person_uids'] is None
    assert 'detector failed' in result['failure']


def test_target_replacement_really_generates_again_without_passing_truth():
    generation = object()
    truth = EvaluationTarget(np.zeros((8, 8, 3), np.uint8), np.zeros((2, 8, 8)), {})
    calls = []
    def generator(pipe, received, *, adapter, **kwargs):
        assert received is generation
        assert set(kwargs) == {'steps', 'cfg', 'use_kv_cache'}
        calls.append(received)
        return Image.new('RGB', (8, 8), (7, 8, 9))
    result = target_replacement_probe(generator, None, generation, truth,
                                      steps=40, cfg=1., use_kv_cache=True)
    assert len(calls) == 2
    assert result['pixel_exact'] is True
    assert result['evaluator_target_before_sha256'] != result['evaluator_target_after_sha256']
    assert result['fresh_generation_calls'] == 2


def test_output_resume_requires_matching_configuration_and_image_hash(tmp_path):
    image = tmp_path / 'output.png'
    Image.new('RGB', (8, 8)).save(image)
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    digest = configuration_sha256({'seed': 42, 'geometry': 'hug62'})
    image.with_suffix('.json').write_text(json.dumps({'configuration_sha256': digest,
                                                    'output_sha256': file_sha256(image)}))
    assert reusable_output(image, digest) is not None
    with pytest.raises(ValueError, match='configuration'):
        reusable_output(image, configuration_sha256({'seed': 43}))
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='image'):
        reusable_output(image, digest)


def rescore_image_case(tmp_path):
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    image = tmp_path / 'native_short_seed42.png'; Image.new('RGB', (512, 512)).save(image)
    configuration = dict(key=image.stem, method='native', prompt_kind='short', prompt='hug', seed=42,
        code_sha256='old_scoring_and_generation_code', steps=40, geometry_sha256='geometry',
        checkpoint_sha256=None, precision_policy={'policy': 'same'}, runtime_versions={'torch': 'same'},
        native_code_sha256={'pipeline': 'same'})
    record = dict(output=str(image), output_sha256=file_sha256(image), configuration=configuration,
        configuration_sha256=configuration_sha256(configuration))
    image.with_suffix('.json').write_text(json.dumps(record))
    return image, record, dict(configuration, code_sha256='new_scoring_code')


def test_rescore_preserves_original_generation_configuration_and_hashes(tmp_path):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    image, original, expected = rescore_image_case(tmp_path)
    record, provenance = runner.validate_rescore_record(image, expected)
    assert record['configuration'] == original['configuration']
    assert record['configuration_sha256'] == original['configuration_sha256']
    assert provenance['original_configuration_sha256'] == original['configuration_sha256']
    assert provenance['original_record_sha256'] == runner.file_sha256(image.with_suffix('.json'))
    assert provenance['original_output_sha256'] == original['output_sha256']


@pytest.mark.parametrize('changed', ['steps', 'geometry_sha256', 'runtime_versions', 'native_code_sha256',
                                   'checkpoint_sha256', 'precision_policy', 'digest', 'png'])
def test_rescore_rejects_changed_generation_conditions_or_original_artifacts(tmp_path, changed):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    image, original, expected = rescore_image_case(tmp_path)
    if changed == 'digest':
        original['configuration_sha256'] = 'tampered'
        image.with_suffix('.json').write_text(json.dumps(original))
    elif changed == 'png':
        Image.new('RGB', (512, 512), 'red').save(image)
    else:
        expected[changed] = 'changed'
    with pytest.raises(ValueError, match='rescore'):
        runner.validate_rescore_record(image, expected)


def test_rescore_requires_new_output_directory_outside_original_tree(tmp_path):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    original = tmp_path / 'original'; original.mkdir()
    for destination in (original, original / 'rescore'):
        with pytest.raises(ValueError, match='rescore'):
            runner.validate_rescore_destination(original, destination)
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(ValueError, match='new'):
        runner.validate_rescore_destination(original, existing)
    runner.validate_rescore_destination(original, tmp_path / 'new')


def test_rescore_checks_frozen_generation_sources_using_checkpoint_contract(tmp_path):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    paths = ['src/pose_control/qwen21_full/qwen_runtime.py', 'src/data/hi4d_smplx_dataset.py']
    for name in paths:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# frozen\n')
    metadata = {'run_contract': {'code_sha256': {name: runner.file_sha256(tmp_path / name) for name in paths}}}
    result = runner.validate_rescore_generation_code(metadata, tmp_path)
    assert set(result) == set(paths)
    (tmp_path / paths[0]).write_text('# changed generator\n')
    with pytest.raises(ValueError, match='generation source'):
        runner.validate_rescore_generation_code(metadata, tmp_path)


def test_multihmr_worker_request_verifies_image_sha_camera_and_output_scope(tmp_path):
    import scripts.eval.multihmr_rgb_worker as worker
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    image = tmp_path / 'image.png'; Image.new('RGB', (512, 512)).save(image)
    request_path = tmp_path / 'worker' / 'request.json'; request_path.parent.mkdir()
    request = dict(schema_version='dual_v8_multihmr_request_v1', root=str(tmp_path), K512=np.eye(3).tolist(),
        geometry_sidecar_sha256='a' * 64, joint_to_part=[index % 14 for index in range(55)],
        outputs=[dict(key='native_short_seed42', image=str(image), image_sha256=file_sha256(image),
            original_configuration_sha256='b' * 64, original_record_sha256='c' * 64,
            output_npz=str(request_path.parent / 'native_short_seed42.npz'))])
    worker.validate_request(request, request_path)
    request['outputs'][0]['image_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='image SHA'):
        worker.validate_request(request, request_path)
    request['outputs'][0]['image_sha256'] = file_sha256(image)
    request['K512'][0][0] = float('nan')
    with pytest.raises(ValueError, match='K512'):
        worker.validate_request(request, request_path)
    request['K512'] = np.eye(3).tolist()
    request['outputs'][0]['output_npz'] = str(tmp_path / 'escape.npz')
    with pytest.raises(ValueError, match='output scope'):
        worker.validate_request(request, request_path)


def test_multihmr_worker_loads_once_and_binds_raw_exports_for_batch(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import torch
    import scripts.eval.multihmr_rgb_worker as worker
    root = tmp_path / 'root'; repo = root / 'add/tools/multi-hmr'
    weights = repo / 'models/multiHMR'; weights.mkdir(parents=True)
    (weights / 'multiHMR_896_L.pt').write_bytes(b'test_fixture_not_a_model')
    dino = repo.parent / 'dinov2_e1277af'; dino.mkdir(); (dino / 'hubconf.py').write_text('# local source\n')
    work = tmp_path / 'isolated_worker'; work.mkdir()
    rows = []
    for index in range(2):
        image = tmp_path / f'image{index}.png'; Image.new('RGB', (512, 512)).save(image)
        rows.append(dict(key=f'image{index}', image=str(image), image_sha256=worker.file_sha256(image),
            original_configuration_sha256=str(index) * 64, original_record_sha256='c' * 64,
            output_npz=str(work / f'image{index}.npz')))
    request = dict(schema_version='dual_v8_multihmr_request_v1', root=str(root), K512=np.eye(3).tolist(),
        geometry_sidecar_sha256='a' * 64, joint_to_part=[index % 14 for index in range(55)], outputs=rows)
    request_path = work / 'request.json'; request_path.write_text(json.dumps(request))
    calls = []
    lbs = torch.zeros(10475, 55); lbs[:, 0] = 1
    body = SimpleNamespace(lbs_weights=lbs, faces=np.zeros((1, 3), np.int32))
    model = SimpleNamespace(img_size=896, smpl_layer={'neutral_10': SimpleNamespace(bm_x=body)})
    model.eval = lambda: model
    model.parameters = lambda: iter([torch.zeros(1)])
    demo = ModuleType('demo_smplx_only')
    demo.load_model = lambda *args, **kwargs: calls.append('load') or model
    demo.open_image = lambda *args, **kwargs: (None, None)
    def forward(received, image, K, **kwargs):
        assert received is model
        calls.append('forward')
        return [{'v3d': torch.zeros(10475, 3), 'j3d': torch.zeros(55, 3), 'j2d': torch.zeros(55, 2)}]
    demo.forward_model = forward
    constants = ModuleType('utils.constants'); constants.CACHE_DIR_MULTIHMR = str(weights); constants.SMPLX_DIR = str(repo / 'models')
    monkeypatch.setitem(sys.modules, 'utils', ModuleType('utils'))
    monkeypatch.setitem(sys.modules, 'utils.constants', constants)
    monkeypatch.setitem(sys.modules, 'demo_smplx_only', demo)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['worker', '--request', str(request_path), '--device', 'cpu'])
    worker.main()
    result = json.loads((work / 'results.json').read_text())
    assert calls == ['load', 'forward', 'forward']
    assert result['provenance']['model_load_count'] == 1
    assert result['provenance']['request_sha256'] == worker.file_sha256(request_path)
    assert result['provenance']['inference']['parameter_dtypes'] == ['torch.float32']
    assert result['provenance']['inference']['output_array_dtype'] == 'float32'
    assert result['provenance']['inference']['forward_autocast'] == 'cuda_autocast_disabled_on_cpu'
    for original, record in zip(rows, result['records']):
        assert record['failure'] is None
        assert record['original_configuration_sha256'] == original['original_configuration_sha256']
        assert record['original_output_sha256'] == original['image_sha256']
        assert record['output_npz_sha256'] == worker.file_sha256(record['output_npz'])
        with np.load(record['output_npz'], allow_pickle=False) as arrays:
            assert int(arrays['person_count']) == 1 and arrays['part_labels'].shape == (10475,)


def test_raw_generated_mesh_quality_reuses_geometry_assignment_and_missing_mesh_na():
    from src.evaluation.qwen21_dual_v8_metrics import generated_mesh_quality
    K = np.eye(3)
    human = {'v3d': np.array([[10, 0, 1], [20, 80, 1]]), 'j2d': np.zeros((55, 2)), 'j3d': np.zeros((55, 3))}
    result = generated_mesh_quality([human], np.array([0, 1]), K, 512,
        np.array([[0, 0, 30, 80], [70, 0, 100, 80]]), np.zeros((2, 24, 2)), np.zeros((2, 24)), [[4, 18]])
    assert result['actor_assignment'] == [0, -1]
    assert result['reconstructed_people'] == 1
    assert result['part_pair_distances'] is None
    assert result['failure'] == 'fewer_than_two_generated_meshes'


@pytest.mark.parametrize('tamper', [None, 'binding', 'npz', 'sidecar', 'init'])
def test_parent_authenticates_isolated_worker_raw_generation_binding(tmp_path, monkeypatch, tamper):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.eval.eval_qwen21_dual_v8 as runner
    from src.pose_control.qwen21_full.dual.schema import SMPLX_JOINT_TO_PART
    output = tmp_path / 'evaluation'; output.mkdir()
    image = output / 'native_short_seed42.png'; Image.new('RGB', (512, 512)).save(image)
    record = dict(output=str(image), output_sha256=runner.file_sha256(image),
        configuration={'key': image.stem}, configuration_sha256='b' * 64,
        generation_provenance={'original_record_sha256': 'c' * 64})
    geometry = tmp_path / 'geometry.npz'
    geometry.with_suffix('.json').write_text(json.dumps({'K512': np.eye(3).tolist()}))
    args = SimpleNamespace(output=output, root=tmp_path, mesh_python=Path(sys.executable), mesh_device='cpu')
    calls = []
    def fake_subprocess(command):
        calls.append(command)
        if tamper == 'init':
            return SimpleNamespace(returncode=2)
        request_path = Path(command[command.index('--request') + 1])
        request = json.loads(request_path.read_text()); expected = request['outputs'][0]
        assert request['joint_to_part'] == list(SMPLX_JOINT_TO_PART)
        lbs = np.zeros((10475, 55), np.float32); lbs[:, 0] = 1
        labels = np.asarray(SMPLX_JOINT_TO_PART)[lbs.argmax(1)]
        body_path = request_path.with_name('body.npz')
        np.savez_compressed(body_path, lbs_weights=lbs, part_labels=labels)
        provenance = dict(request_sha256=runner.file_sha256(request_path),
            worker_source_sha256=runner.file_sha256(command[1]), body_partition_path=str(body_path),
            body_partition_sha256=runner.file_sha256(body_path), image_size=896)
        raw_path = Path(expected['output_npz'])
        np.savez_compressed(raw_path, person_count=np.asarray(1), image_size=np.asarray(896),
            K512=np.eye(3, dtype=np.float32), part_labels=labels,
            person_0_v3d=np.zeros((10475, 3)), person_0_j3d=np.zeros((127, 3)), person_0_j2d=np.zeros((127, 2)))
        row = dict(key=expected['key'], original_output_sha256=expected['image_sha256'],
            original_configuration_sha256=expected['original_configuration_sha256'],
            original_record_sha256=expected['original_record_sha256'], request_sha256=provenance['request_sha256'],
            worker_provenance=provenance, output_npz=str(raw_path), output_npz_sha256=runner.file_sha256(raw_path),
            reconstructed_people=1, failure=None)
        if tamper == 'binding': row['original_output_sha256'] = 'e' * 64
        runner.write_json(raw_path.with_suffix('.json'), dict(row, failure='tampered') if tamper == 'sidecar' else row)
        runner.write_json(request_path.with_name('results.json'), dict(schema_version='dual_v8_multihmr_result_v1',
            provenance=provenance, records=[row]))
        if tamper == 'npz': raw_path.write_bytes(b'tampered')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', fake_subprocess)
    if tamper is None:
        raw, provenance = runner.run_mesh_worker(args, [record], geometry)
        assert raw[image.stem]['humans'][0]['j3d'].shape == (127, 3)
        assert raw[image.stem]['raw_worker_record']['original_record_sha256'] == 'c' * 64
        assert provenance['result_manifest_sha256'] == runner.file_sha256(provenance['result_manifest'])
    else:
        with pytest.raises((ValueError, RuntimeError), match='Multi-HMR'):
            runner.run_mesh_worker(args, [record], geometry)
    assert len(calls) == 1 and calls[0][-2:] == ['--device', 'cpu']


def test_legacy_qwen_runtime_and_nested_sources_change_evaluation_reuse_key(tmp_path, monkeypatch):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    filenames = ['src/pose_control/qwen21_full/qwen_runtime.py',
        'src/pose_control/qwen21_full/reasoning/deep/reasoner.py',
        'scripts/eval/eval_qwen21_dual_v8.py', 'src/evaluation/qwen21_dual_v8_metrics.py',
        'src/data/hi4d_smplx_dataset.py']
    for name in filenames:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# original\n')
    monkeypatch.setattr(runner, 'PROJECT', tmp_path)
    monkeypatch.setattr(runner, '__file__', str(tmp_path / filenames[2]))
    original = runner._code_signature()
    original_key = configuration_sha256({'code_sha256': original, 'seed': 42})
    for name in filenames[:2]:
        path = tmp_path / name
        path.write_text('# changed generation code\n')
        changed = runner._code_signature()
        assert changed != original
        assert configuration_sha256({'code_sha256': changed, 'seed': 42}) != original_key
        path.write_text('# original\n')


def test_runtime_version_change_invalidates_evaluation_reuse_key(monkeypatch):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    versions = {'torch': '2.8.0', 'diffusers': '0.35.0', 'transformers': '4.56.0'}
    monkeypatch.setattr(runner, 'version', lambda name: versions.get(name, '1.0.0'), raising=False)
    original = runner._runtime_versions()
    original_key = configuration_sha256({'runtime_versions': original})
    for name in ('torch', 'diffusers', 'transformers'):
        previous = versions[name]; versions[name] = previous + '.changed'
        assert configuration_sha256({'runtime_versions': runner._runtime_versions()}) != original_key
        versions[name] = previous


def test_native_class_sources_match_without_loading_model_weights(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import scripts.eval.eval_qwen21_dual_v8 as runner
    source = tmp_path / 'native.py'
    source.write_text('class Pipeline: pass\nclass Transformer: pass\nclass VAE: pass\nclass TextEncoder: pass\n')
    spec = importlib.util.spec_from_file_location('test_eval_native_classes', source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module); spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'diffusers', module)
    declarations = {'_class_name': 'Pipeline', 'transformer': [spec.name, 'Transformer'],
        'vae': [spec.name, 'VAE'], 'text_encoder': [spec.name, 'TextEncoder']}
    (tmp_path / 'model_index.json').write_text(json.dumps(declarations))
    expected = runner._native_code_sha256(tmp_path)
    pipeline = module.Pipeline()
    pipeline.transformer, pipeline.vae, pipeline.text_encoder = module.Transformer(), module.VAE(), module.TextEncoder()
    actual = runner._native_code_sha256(tmp_path, pipeline)
    assert actual == expected
    assert set(actual) == {'pipeline', 'transformer', 'vae', 'text_encoder'}
    assert all(isinstance(value, str) and len(value) == 64 for value in actual.values())
    source.write_text(source.read_text() + '# installed native source changed\n')
    assert runner._native_code_sha256(tmp_path) != expected


def test_default_comparisons_are_both_prompts_for_both_methods_for_three_seeds():
    runs = planned_runs([42, 43, 44], 'short hug', 'detailed hug')
    assert len(runs) == 12
    assert {run['method'] for run in runs} == {'native', 'dual_v8'}
    assert {run['prompt_kind'] for run in runs} == {'short', 'detailed'}
    assert {run['seed'] for run in runs} == {42, 43, 44}


def test_detector_failure_is_na_and_not_scored_as_zero_people():
    result = aggregate_quality([{'people': [], 'detected_people': None}])
    assert result['third_person_rate'] is None
    assert result['third_person_denominator'] == 0
    assert result['detection_failed_outputs'] == 1


@pytest.mark.parametrize('changed', ['target_rgb', 'target_mask', 'missing_declared_hash',
                                    'wrong_frame', 'wrong_geometry_hash', 'row_count'])
def test_evaluator_truth_uses_strict_supervision_validator(tmp_path, changed):
    from test_qwen21_dual_v8_data import supervised_case
    data, generation_manifest, manifest, _, metadata = supervised_case(tmp_path)
    truth = EvaluationTarget.read(manifest)
    assert truth.metadata['supervision_manifest_sha256'] == data.file_sha256(manifest)
    assert truth.person_masks.shape == (2, 512, 512)
    if changed == 'target_rgb':
        Image.new('RGB', (32, 32), 'blue').save(metadata['target_rgb'])
    elif changed == 'target_mask':
        Image.new('L', (32, 32), 0).save(metadata['target_masks'][1])
    elif changed == 'missing_declared_hash':
        del metadata['target_rgb_sha256']
    elif changed == 'wrong_frame':
        metadata['target_frame'] = 63
    elif changed == 'wrong_geometry_hash':
        metadata['target_geometry_sha256'] = 'f' * 64
    if changed == 'row_count':
        manifest.write_text(json.dumps([metadata, metadata]))
    else:
        manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='supervision'):
        EvaluationTarget.read(manifest)


@pytest.mark.parametrize('missing', [None, '', 'not_a_sha256'])
def test_positive_checkpoint_requires_declared_backbone_hash(missing):
    import scripts.eval.eval_qwen21_dual_v8 as runner
    payload = {'step': 300, 'metadata': {'architecture_version': 'dual_v8'}}
    if missing is not None:
        payload['metadata']['backbone_sha256'] = missing
    with pytest.raises(ValueError, match='backbone_sha256'):
        runner.expected_backbone_sha256(payload)


def test_residual_png_handles_actual_batched_map_and_labels_normalization(tmp_path):
    from types import SimpleNamespace
    adapter = SimpleNamespace(last_residual_maps={3: {'rms': np.ones((1, 4, 4)),
        'sigma': .2, 'timestep': 200., 'target_hw': (4, 4)}})
    path = tmp_path / 'residual.png'
    result = save_residual_heatmaps(adapter, path)
    assert path.exists()
    assert result['maps'][0]['maximum'] == 1.
    assert result['maps'][0]['sigma'] == .2
    assert result['maps'][0]['normalization'] == 'per_layer_map_max_for_display'


def test_main_loads_native_once_and_metrics_only_reuses_all_matching_outputs(tmp_path, monkeypatch):
    """Only native/model dependencies are fake; runner IO/resumption is real."""
    import sys
    from types import ModuleType, SimpleNamespace
    import torch
    import scripts.eval.eval_qwen21_dual_v8 as runner
    from src.pose_control.qwen21_full.dual.conditions import ControlStrengths

    source = tmp_path / 'source.png'; Image.new('RGB', (512, 512)).save(source)
    masks = []
    for index in range(2):
        path = tmp_path / f'mask{index}.png'; Image.new('L', (512, 512)).save(path); masks.append(path)
    features = tmp_path / 'features.pt'; features.write_bytes(b'source_only')
    geometry_path = tmp_path / 'hug_geometry.npz'; geometry_path.write_bytes(b'geometry')
    manifest = tmp_path / 'generation.json'; manifest.write_text('{}')
    checkpoint = tmp_path / 'trained.pt'; checkpoint.write_bytes(b'dual_v8_step300')
    @dataclass(frozen=True)
    class Generation:
        source: object
        geometry: object
        prompts: tuple = ('short hug',)
        seeds: tuple = (42,)
    amodal = torch.zeros(1, 2, 1, 16, 16); amodal[0, 0, 0, 1:15, 1:7] = 1; amodal[0, 1, 0, 1:15, 9:15] = 1
    generated = Generation(SimpleNamespace(rgb=torch.zeros(1, 3, 512, 512),
        masks=torch.zeros(1, 2, 1, 512, 512), person_uids=(('actor0', 'actor1'),)),
        SimpleNamespace(amodal_mask=amodal, pose=torch.zeros(1, 2, 25, 16, 16), geometry_hash='geometry_hash'))
    row = dict(source_rgb=str(source), source_masks=list(map(str, masks)), source_features=str(features), target_geometry=str(geometry_path))
    class Dataset:
        def __init__(self, path, supervision_manifest=None):
            self.rows = [row]; self.supervision_manifest = supervision_manifest
        def __len__(self): return 1
        def generation(self): return generated
        def supervision(self, index=0):
            assert self.supervision_manifest is not None
            truth_reads.append(len(calls))
            return SimpleNamespace(rgb=torch.zeros(1, 3, 512, 512),
                person_masks=torch.zeros(1, 2, 1, 512, 512), metadata={})
    fake_data = ModuleType('src.data.hi4d_smplx_dataset')
    fake_data.Hi4DSMPLXDataset = Dataset
    fake_data.letterbox_affine = lambda width, height: np.eye(3)
    fake_runtime = ModuleType('src.pose_control.qwen21_full.dual.runtime')
    fake_runtime.DETAILED_HUG_PROMPT = 'detailed hug'
    loads, calls, truth_reads = [], [], []
    pipe = SimpleNamespace(last_dual_v8_diagnostics={'velocity_finite': True}, transformer=object())
    def load_native(path): loads.append(path); return pipe
    def generate(received_pipe, value, **kwargs):
        assert received_pipe is pipe
        assert isinstance(value, Generation)
        assert 'evaluation_target' not in kwargs
        calls.append((value.prompts[0], value.seeds[0], kwargs['adapter'] is not None))
        return Image.new('RGB', (512, 512), (value.seeds[0], 0, 0))
    fake_runtime.load_native, fake_runtime.generate = load_native, generate
    fake_runtime.prepared_for_qwen = lambda *args: None
    backbone_state = {'expected': 'a' * 64, 'actual': 'a' * 64}
    backbone_checks = []
    def backbone_hash(module):
        backbone_checks.append(module)
        return backbone_state['actual']
    fake_runtime.tensor_model_sha256 = backbone_hash
    adapter = SimpleNamespace(eval=lambda: None, last_diagnostics={})
    fake_checkpoint = ModuleType('src.pose_control.qwen21_full.dual.checkpoint')
    model = tmp_path / 'model'
    (model / 'transformer').mkdir(parents=True)
    (model / 'transformer/config.json').write_text('{}')
    fake_checkpoint.load_checkpoint = lambda *args, **kwargs: (adapter, {'model_id': str(model.resolve())}, {'step': 300})
    monkeypatch.setitem(sys.modules, fake_data.__name__, fake_data)
    monkeypatch.setitem(sys.modules, fake_runtime.__name__, fake_runtime)
    monkeypatch.setitem(sys.modules, fake_checkpoint.__name__, fake_checkpoint)
    checkpoint_reads = []
    def precheck_checkpoint(*args, **kwargs):
        checkpoint_reads.append(kwargs)
        paths = list((runner.PROJECT / 'src/pose_control/qwen21_full').rglob('*.py'))
        paths += [runner.PROJECT / 'src/data/hi4d_smplx_dataset.py']
        paths += [runner.PROJECT / name for name in ('src/__init__.py', 'src/data/__init__.py', 'src/pose_control/__init__.py') if (runner.PROJECT / name).exists()]
        return {'step': 300, 'metadata': {'architecture_version': 'dual_v8',
            'backbone_sha256': backbone_state['expected'], 'model_id': str(model.resolve()),
            'model_config_sha256': runner.file_sha256(model / 'transformer/config.json'),
            'run_contract': {'code_sha256': {path.relative_to(runner.PROJECT).as_posix(): runner.file_sha256(path) for path in paths}}}}
    monkeypatch.setattr(torch, 'load', precheck_checkpoint)
    monkeypatch.setattr(runner, 'export_controls', lambda *args: [])
    monkeypatch.setattr(runner, '_wholebody_truth', lambda *args: (None, None, {'available': False}))
    runtime_versions = {'torch': 'test_v1', 'diffusers': 'test_v1', 'transformers': 'test_v1'}
    native_code = {'pipeline': 'c' * 64, 'transformer': 'd' * 64, 'vae': 'e' * 64, 'text_encoder': 'f' * 64}
    monkeypatch.setattr(runner, '_runtime_versions', lambda: dict(runtime_versions), raising=False)
    monkeypatch.setattr(runner, '_native_code_sha256', lambda *args: dict(native_code), raising=False)
    def unavailable(*args): raise FileNotFoundError('no_test_model')
    monkeypatch.setattr(runner, 'DWPosePeople', unavailable)
    output = tmp_path / 'evaluation'
    argv = ['eval', '--generation', str(manifest), '--evaluation-target', str(tmp_path / 'truth.json'),
        '--checkpoint', str(checkpoint), '--output', str(output), '--model', str(model), '--seeds', '42',
        '--skip-dino', '--no-face-metrics', '--no-target-replacement-check', '--no-geometry-switch-check']
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    assert len(loads) == 1
    assert len(calls) == 4
    assert truth_reads == [4]
    summary = json.loads((output / 'run.json').read_text())
    assert summary['checkpoint']['payload_step'] == 300
    assert len(summary['output_records']) == 4
    from src.pose_control.qwen21_full.dual.precision import precision_policy
    assert summary['precision_policy'] == precision_policy()
    assert checkpoint_reads[0]['weights_only'] is True
    image_record = json.loads((output / 'dual_v8_short_seed42.json').read_text())
    assert image_record['precision_policy'] == precision_policy()
    assert image_record['configuration']['precision_policy'] == precision_policy()
    assert image_record['configuration']['backbone_sha256'] == backbone_state['expected']
    assert image_record['configuration']['runtime_versions'] == runtime_versions
    assert image_record['configuration']['native_code_sha256'] == native_code
    assert len(backbone_checks) == 1
    monkeypatch.setattr(sys, 'argv', argv + ['--metrics-only'])
    runner.main()
    assert len(loads) == 1
    assert len(calls) == 4
    assert len(backbone_checks) == 1
    originals = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    rescored = tmp_path / 'rescored'
    original_signature = runner._code_signature
    monkeypatch.setattr(runner, '_code_signature', lambda: 'changed_scoring_code')
    rescoring_argv = list(argv)
    rescoring_argv[rescoring_argv.index('--output') + 1] = str(rescored)
    monkeypatch.setattr(sys, 'argv', rescoring_argv + ['--rescore-from', str(output)])
    runner.main()
    assert len(loads) == 1 and len(calls) == 4 and len(backbone_checks) == 1
    assert all(path.read_bytes() == value for path, value in originals.items())
    rescored_summary = json.loads((rescored / 'run.json').read_text())
    assert rescored_summary['rescore_provenance']['original_probes_file_sha256'] == runner.file_sha256(output / 'probes.json')
    assert rescored_summary['scoring_provenance']['code_sha256']
    assert (rescored / 'original_probes.json').read_bytes() == (output / 'probes.json').read_bytes()
    copied_record = json.loads((rescored / 'dual_v8_short_seed42.json').read_text())
    assert copied_record['configuration'] == image_record['configuration']
    assert copied_record['configuration_sha256'] == image_record['configuration_sha256']
    assert copied_record['generation_provenance']['original_output_sha256'] == image_record['output_sha256']
    assert copied_record['scoring_provenance']['generated_rgb_sha256'] == image_record['output_sha256']
    monkeypatch.setattr(runner, '_code_signature', original_signature)
    monkeypatch.setattr(sys, 'argv', argv + ['--metrics-only'])
    runtime_versions['torch'] = 'test_v2'
    with pytest.raises(ValueError, match='configuration'):
        runner.main()
    assert len(loads) == 1
    assert len(calls) == 4
    runtime_versions['torch'] = 'test_v1'
    native_code['transformer'] = '9' * 64
    with pytest.raises(ValueError, match='configuration'):
        runner.main()
    assert len(loads) == 1
    assert len(calls) == 4
    native_code['transformer'] = 'd' * 64
    # The model path is identical, but a different checkpoint backbone binding
    # must invalidate even a metrics-only reuse key.
    backbone_state['expected'] = 'b' * 64
    with pytest.raises(ValueError, match='configuration'):
        runner.main()
    assert len(calls) == 4
    # Ordinary inference must reject actual changed native weights before any
    # cached image reuse or new generation, even without --native-checksum.
    backbone_state['expected'] = 'a' * 64
    backbone_state['actual'] = 'b' * 64
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='backbone'):
        runner.main()
    assert len(calls) == 4

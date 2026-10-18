import importlib
import inspect
import json

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module('src.data.hi4d_dual_prompt_pilot')
    except ModuleNotFoundError as exc:
        if exc.name == 'src.data.hi4d_dual_prompt_pilot':
            pytest.fail('source-only dual prompt pilot data module is not implemented')
        raise


def split_manifest():
    return {'train_pairs': ['pair12', 'pair14'], 'val_pairs': ['pair22'],
            'test_pairs': ['pair00']}


def sample(split='train', pair='pair12', frame=33):
    return dict(split=split, pair=pair, source_action='highfive12',
                target_action='highfive12', source_frame_id=28,
                target_frame_id=frame, camera_id=4)


def anchors():
    masks = np.zeros((2, 64, 64), bool)
    masks[0, 10:50, 40:55] = True
    masks[1, 12:55, 5:25] = True
    return module().source_person_anchors(['pair12_actor0', 'pair12_actor1'], masks)


def geometry(height=0.7):
    joints = np.zeros((2, 55, 3), np.float32)
    joints[:, :, 1] = .8
    joints[:, 0, 1] = .9
    joints[:, 15, 1] = 1.6
    joints[:, 16:18, 1] = 1.4
    joints[:, 20:22, 1] = height
    joints[0, :, 0] = -.25
    joints[1, :, 0] = .25
    return dict(person_uids=['pair12_actor0', 'pair12_actor1'], joints_world=joints,
                body_confidence=np.full((2, 22), .9), hand_confidence=np.full((2, 2), .25),
                root_orient=np.zeros((2, 3)), contact_primitives=[], geometry_sha256='a' * 64)


def test_authoritative_split_rejects_test_pair_in_training():
    with pytest.raises(ValueError, match='membership'):
        module().validate_split_selection([sample(pair='pair00')], split_manifest())


def test_authoritative_split_rejects_pair_overlap_and_duplicate_frame():
    invalid = split_manifest()
    invalid['val_pairs'].append('pair12')
    with pytest.raises(ValueError, match='overlap'):
        module().validate_split_selection([sample()], invalid)
    with pytest.raises(ValueError, match='duplicate'):
        module().validate_split_selection([sample(), sample()], split_manifest())


def test_source_uid_binding_survives_target_left_right_swap():
    prompts = module().build_action_prompts(person_uids=geometry()['person_uids'],
        source_anchors=anchors(), action='highfive12', geometry=geometry())
    assert 'A is the person on the right in the source image' in prompts['structured']
    assert 'B is the person on the left in the source image' in prompts['structured']
    wrong = anchors()[::-1]
    with pytest.raises(ValueError, match='UID'):
        module().build_action_prompts(person_uids=geometry()['person_uids'],
            source_anchors=wrong, action='highfive12', geometry=geometry())


@pytest.mark.parametrize('field', ['target_rgb', 'target_masks', 'target_mask', 'caption'])
def test_prompt_geometry_cannot_accept_target_truth(field):
    values = geometry()
    values[field] = 'secret ground truth'
    with pytest.raises(ValueError, match='forbidden'):
        module().build_action_prompts(person_uids=values['person_uids'],
            source_anchors=anchors(), action='highfive12', geometry=values)


def test_low_confidence_hand_details_are_omitted_and_action_does_not_invent_contact():
    prompts = module().build_action_prompts(person_uids=geometry()['person_uids'],
        source_anchors=anchors(), action='highfive12', geometry=geometry())
    assert 'finger' not in prompts['structured'].lower()
    assert 'palms touch' not in prompts['structured'].lower()
    assert 'hands near waist level' in prompts['structured']
    assert prompts['provenance']['omitted'] == ['finger_articulation: unobserved/model prior']
    assert prompts == module().build_action_prompts(person_uids=geometry()['person_uids'],
        source_anchors=anchors(), action='highfive12', geometry=geometry())


def test_highfive_phases_are_derived_from_reliable_wrist_height():
    data = module()
    low = data.build_action_prompts(person_uids=geometry()['person_uids'], source_anchors=anchors(),
                                    action='highfive12', geometry=geometry(.7))
    high = data.build_action_prompts(person_uids=geometry()['person_uids'], source_anchors=anchors(),
                                     action='highfive12', geometry=geometry(1.8))
    assert low['structured'] != high['structured']
    assert 'hands above head level' in high['structured']
    uncertain = geometry(1.8)
    uncertain['body_confidence'][:, 20:22] = .25
    prompt = data.build_action_prompts(person_uids=uncertain['person_uids'], source_anchors=anchors(),
                                     action='highfive12', geometry=uncertain)
    assert 'above head level' not in prompt['structured']


def test_true_smplx_topology_and_source_shape_identity_are_required():
    data = module()
    source = dict(vertices_world=np.zeros((2, 10475, 3)), joints_world=np.zeros((2, 55, 3)),
                  betas=np.zeros((2, 10)), actor_ids=np.array([0, 1]), faces=np.zeros((20908, 3), int))
    target = {k: v.copy() for k, v in source.items()}
    data.validate_fitted_identity(source, target)
    target['vertices_world'] = np.zeros((2, 6890, 3))
    with pytest.raises(ValueError, match='SMPL-X'):
        data.validate_fitted_identity(source, target)
    target['vertices_world'] = source['vertices_world'].copy()
    target['betas'][0, 0] = .01
    with pytest.raises(ValueError, match='source actor beta'):
        data.validate_fitted_identity(source, target)


def test_geometry_and_fit_apis_have_no_target_truth_inputs():
    data = module()
    assert 'target_masks' not in inspect.signature(data.export_target_controls).parameters
    assert 'target_rgb' not in inspect.signature(data.fit_smpl_frame).parameters
    assert 'target_masks' not in inspect.signature(data.fit_smpl_frame).parameters


def test_dwpose_letterbox_and_missing_prior_confidence_are_explicit():
    data = module()
    xy = np.full((2, 134, 2), [470., 640.], np.float32)
    scores = np.ones((2, 134), np.float32)
    scores[0, 4] = 0
    prior = np.full((2, 24, 2), [123., 144.], np.float32)
    points, confidence, kinds = data.body_pose_points(xy, scores, 940, 1280, prior)
    np.testing.assert_allclose(points[1, 0], [256., 256.])
    np.testing.assert_allclose(points[0, 4], [123., 144.])
    assert confidence[0, 4] == .25
    assert kinds[0][4] == 'fitted_smplx_prior'


def test_materialized_source_inputs_are_exact_512_letterbox(tmp_path):
    from PIL import Image
    data = module()
    rgb = np.full((128, 94, 3), 73, np.uint8)
    rgb_path = tmp_path / 'source.png'
    Image.fromarray(rgb).save(rgb_path)
    masks = []
    for slot in range(2):
        mask = np.zeros((128, 94), np.uint8)
        mask[30:100, 10 + 40 * slot:30 + 40 * slot] = 255
        path = tmp_path / f'mask{slot}.png'
        Image.fromarray(mask).save(path)
        masks.append(path)
    output = tmp_path / 'assets' / 'sources' / 'pair12' / 'cam4' / 'highfive12' / '000028'
    result = data.materialize_source_assets(rgb_path, masks, output)
    assert Image.open(result['source_rgb']).size == (512, 512)
    assert all(Image.open(p).size == (512, 512) for p in result['source_masks'])
    value = np.asarray(Image.open(result['source_rgb']))
    assert np.all(value[:, :60] == 0)
    assert value[256, 256, 0] == 73
    assert result['preprocessing']['native_size'] == [94, 128]


def test_existing_output_is_rejected_before_reading_or_fitting(tmp_path):
    with pytest.raises(FileExistsError, match='new output'):
        module().prepare_dataset(root=tmp_path / 'does-not-exist', output=tmp_path,
                                 dino=tmp_path, vae=tmp_path, device='cpu', fit_steps=1)


def test_dataset_preparation_requires_selection_lock_before_model_load(tmp_path, monkeypatch):
    data = module()
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split_manifest()))
    monkeypatch.setattr(data, 'load_selected_rows', lambda root: ([], split_path, {}))
    with pytest.raises(FileNotFoundError, match='selection lock'):
        data.prepare_dataset(root=tmp_path, output=tmp_path / 'new-data', dino=tmp_path,
                             vae=tmp_path, device='cpu', fit_steps=1)
    assert not (tmp_path / 'new-data').exists()


@pytest.mark.parametrize('tamper', ['split', 'manifest', 'row'])
def test_selection_lock_rejects_changed_original_content(tmp_path, tamper):
    data = module()
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split_manifest()))
    record = dict(split='train', original_manifest_line=445,
                  original_manifest_sha256='a' * 64, original_row_sha256='b' * 64)
    lock = dict(selection_locked_before_training=True, split_manifest_sha256=data._sha(split_path),
                rows=[dict(split='train', one_based_line=445, manifest_sha256='a' * 64, row_sha256='b' * 64)])
    if tamper == 'split':
        lock['split_manifest_sha256'] = 'c' * 64
    elif tamper == 'manifest':
        lock['rows'][0]['manifest_sha256'] = 'c' * 64
    else:
        lock['rows'][0]['row_sha256'] = 'c' * 64
    path = tmp_path / 'lock.json'
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='split|changed'):
        data._validate_selection_lock(path, [record], split_path)


def test_export_uses_individual_wrist_residual_not_good_body_average(tmp_path):
    """Exercise the actual export->prompt path without a CUDA renderer."""
    data = module()
    joints = geometry(1.8)['joints_world']
    errors = np.full((2, 22), 5., np.float32)
    errors[0, 20] = 180.
    fitted = dict(vertices_world=np.zeros((2, 10475, 3), np.float32),
                  joints_world=joints, betas=np.zeros((2, 10), np.float32),
                  actor_ids=np.array([0, 1]), faces=np.zeros((20908, 3), np.int32),
                  global_orient=np.zeros((2, 3), np.float32),
                  body_joint_mm=errors, body_joint_mean_mm=errors.mean(-1))

    class FixtureRenderer:
        def render(self, vertices, R, t, K):
            return dict(mask=np.ones((512, 512), bool), depth=np.full((512, 512), 3., np.float32),
                        normal=np.zeros((3, 512, 512), np.float32),
                        part=np.zeros((14, 512, 512), np.float32),
                        vertex_normals_camera=np.zeros((10475, 3), np.float32))

    paths = []
    for name in ('fit.npz', 'source.npz', 'raw.npz', 'poseA.npz', 'poseB.npz'):
        path = tmp_path / name
        path.write_bytes(b'fixture provenance')
        paths.append(path)
    identity = dict(pair='pair12', action='highfive12', source_frame=28,
                    target_frame=82, camera=4, person_uids=geometry()['person_uids'])
    path, exported = data.export_target_controls(source_fit=fitted, target_fit=fitted,
        raw_smpl=dict(contact=np.zeros((2, 6890))),
        camera_calibration=dict(K=np.array([[500., 0, 256], [0, 500, 256], [0, 0, 1]]),
                                extrinsics=np.c_[np.eye(3), [0, 0, 3]]),
        pose_observations=dict(xy_native=np.full((2, 134, 2), 256.), scores=np.ones((2, 134)),
                               width=512, height=512, paths=list(map(str, paths[3:]))),
        renderer=FixtureRenderer(), labels=np.arange(10475) % 14, heights=np.array([1.7, 1.7]),
        output=tmp_path, identity=identity, fit_path=paths[0], source_fit_path=paths[1], raw_smpl_path=paths[2])
    assert errors.mean(-1).max() < 15
    assert exported['body_confidence'][0, 20] < .6
    assert exported['body_confidence'][0, 21] > .6
    prompt = data.build_action_prompts(person_uids=identity['person_uids'], source_anchors=anchors(),
                                     action='highfive12', geometry=exported)
    assert 'A keeps the hands above head level' not in prompt['structured']
    assert 'A keeps the right hand above head level' in prompt['structured']
    assert not any(entry['feature'] == 'A_left_elbow' for entry in prompt['provenance']['used'])
    with np.load(path) as controls:
        assert controls['part_confidence'][0, 3] < .6
        assert controls['part_confidence'][0, 4] <= .25
        np.testing.assert_array_equal(controls['body_joint_mm'], errors)

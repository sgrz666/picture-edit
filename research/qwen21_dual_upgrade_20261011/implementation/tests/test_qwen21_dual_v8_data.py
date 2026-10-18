import importlib
import json

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module('src.data.hi4d_smplx_dataset')
    except ModuleNotFoundError as exc:
        if exc.name == 'src.data.hi4d_smplx_dataset':
            pytest.fail('SMPL-X generation-only dataset is not implemented')
        raise


def test_letterbox_camera_uses_same_affine_as_points():
    data = module()
    affine = data.letterbox_affine(940, 1280)
    assert affine.shape == (3, 3)
    assert affine[0, 0] == affine[1, 1]
    K = np.array([[1000., 0, 470], [0, 1000, 640], [0, 0, 1]])
    point = np.array([.2, .1, 3.])
    raw = K @ point
    direct = affine @ (raw / raw[2])
    adjusted = affine @ K @ point
    np.testing.assert_allclose(direct[:2], adjusted[:2] / adjusted[2])


def test_smplx_parts_reject_old_smpl_topology_and_keep_hands():
    data = module()
    with pytest.raises(ValueError, match='10475'):
        data.smplx_part_labels(np.ones((6890, 55)))
    weights = np.zeros((10475, 55))
    weights[:, 0] = 1
    weights[0] = 0
    weights[0, 25] = 1
    weights[1] = 0
    weights[1, 40] = 1
    labels = data.smplx_part_labels(weights)
    assert labels[0] == 4 and labels[1] == 7
    assert labels.max() < 14


def test_depth_owner_is_symmetric_and_uses_metric_near_surface():
    data = module()
    masks = np.ones((2, 4, 4), dtype=bool)
    depths = np.stack((np.full((4, 4), 2.), np.full((4, 4), 3.)))
    owners = data.visible_owner(masks, depths)
    assert np.all(owners[0] == 1) and not owners[1].any()
    np.testing.assert_array_equal(data.visible_owner(masks[::-1], depths[::-1])[::-1], owners)
    ties = data.visible_owner(masks, depths * 0 + 2)
    assert np.all(ties == .5)


def test_global_and_directed_relative_use_rotation_matrices():
    data = module()
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', [[.2,.3,.4],[-.4,.2,.1]]).as_rotvec()
    masks = np.zeros((2,512,512), dtype=bool)
    masks[0,100:400,100:220] = True
    masks[1,110:410,250:390] = True
    global32, relative14 = data.geometry_vectors(np.zeros((2,10)), rot,
        np.array([[0.,0,2],[.5,0,3]]), np.eye(3), np.zeros(3),
        np.array([[500.,0,256],[0,500,256],[0,0,1]]), masks, np.array([1.7,1.8]))
    assert global32.shape == (2,32) and relative14.shape == (2,14)
    np.testing.assert_allclose(relative14[0,:3], -relative14[1,:3])
    RA,RB = Rotation.from_rotvec(rot).as_matrix()
    expected = RA.T @ RB
    np.testing.assert_allclose(relative14[0,3:9], np.concatenate([expected[:,0],expected[:,1]]), atol=1e-6)


def test_generation_manifest_rejects_target_truth_paths():
    data = module()
    with pytest.raises(ValueError, match='target_rgb'):
        data.validate_generation_manifest({'target_rgb':'hidden.jpg'})


def test_dataset_generation_does_not_open_supervision(tmp_path):
    data = module()
    generation = tmp_path/'generation.json'
    generation.write_text(json.dumps([]))
    dataset = data.Hi4DSMPLXDataset(generation, tmp_path/'missing_truth.json')
    assert len(dataset) == 0
    with pytest.raises(FileNotFoundError):
        dataset.supervision(0)


def test_gaussian_heatmaps_match_body25_convention():
    data = module()
    xy = np.full((24,2), 128., dtype=np.float32)
    score = np.ones(24, dtype=np.float32)
    score[4] = 0
    heat = data.body25_heatmaps(xy, score)
    assert heat.shape == (25,512,512)
    assert heat[0,128,128] == 1 and heat[24,128,128] == 1
    assert not heat[4].any()


def test_cuda_renderer_camera_projection_and_nearer_depth():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('requires the production CUDA rasterizer')
    data = module()
    K = np.array([[32.,0,16],[0,32,16],[0,0,1]],np.float32)
    xy = np.array([[4.,3.],[15.,3.],[4.,14.]],np.float32)
    z = 2.
    vertices = np.c_[(xy-16)/32*z,np.full(3,z)].astype(np.float32)
    renderer = data.SMPLXRenderer(np.array([[0,1,2]]),np.ones(3,dtype=np.int64))
    result = renderer.render(vertices,np.eye(3),np.zeros(3),K,size=32)
    assert result['mask'][6,7]
    assert not result['mask'][25,7]  # catches a vertical raster flip
    np.testing.assert_allclose(result['depth'][result['mask']],2.,atol=1e-5)
    assert result['part'][1,6,7]==1
    assert np.isfinite(result['normal']).all()


def test_source_crops_remove_other_person_and_keep_whole_silhouette():
    data = module()
    rgb = np.full((512,512,3),30,np.uint8)
    masks = np.zeros((2,512,512),np.float32)
    masks[0,100:450,80:180]=1
    masks[1,100:450,180:280]=1
    rgb[masks[0]>0]=[255,0,0]
    rgb[masks[1]>0]=[0,255,0]
    crops,coverage,boxes = data.source_person_crops(rgb,masks)
    assert crops.shape==(2,512,512,3) and coverage.shape==(2,512,512)
    assert not ((crops[0,...,1]>200)&(crops[0,...,0]<50)).any()
    assert np.nonzero(coverage[0])[0].min()>0 and np.nonzero(coverage[0])[0].max()<511
    assert boxes.shape==(2,4)


def test_hi4d_contact_correspondence_is_one_based_with_zero_absent():
    data = module()
    contact = np.array([[0,4,2,0],[2,0,0,2]],np.float64)
    pairs = data.hi4d_contact_pairs(contact)
    assert set(map(tuple,pairs.tolist()))=={(1,3),(2,1),(1,0)}


def manifest_row():
    return dict(source_rgb='source.png', source_masks=['a.png', 'b.png'],
        source_features='source_features.pt', target_geometry='geometry.npz',
        source_features_sha256='0' * 64,
        person_uids=['actor_a', 'actor_b'], prompt='hug', seed=42,
        resolution=512, target_geometry_known=True)


def test_generation_manifest_requires_explicit_versions():
    with pytest.raises(ValueError, match='versions'):
        module().validate_generation_manifest(manifest_row())


@pytest.mark.parametrize('key,value', [('geometry', 'smpl_dual_v1'),
    ('coordinate_convention', 'camera_only'), ('rotation6d_convention', 'rows'),
    ('partition_sha256', '0' * 64), ('source', 'source_v0'),
    ('crop', 'centre_crop'), ('encoder', 'unknown_encoder')])
def test_generation_manifest_rejects_mixed_schema_conventions(key, value):
    data = module()
    row = manifest_row()
    row['versions'] = data.generation_versions()
    row['versions'][key] = value
    with pytest.raises(ValueError, match=key):
        data.validate_generation_manifest(row)


def test_canonical_partition_hash_binds_the_part_order_and_55_joint_rule():
    import hashlib
    data = module()
    encoded = json.dumps(data.SMPLX14_PARTITION_SPEC, sort_keys=True,
        separators=(',', ':')).encode('utf-8')
    assert hashlib.sha256(encoded).hexdigest() == data.SMPLX14_PARTITION_SHA256
    assert data.SMPLX14_PARTITION_SPEC['joint_to_part'] == data.SMPLX_TO_PART.tolist()
    assert len(data.SMPLX14_PARTITION_SPEC['part_names']) == 14


def test_anatomical_geometry_uses_standard_smplx_joint_centres_and_bone_directions():
    data = module()
    joints = np.arange(2 * 144 * 3, dtype=np.float32).reshape(2, 144, 3) / 100
    orientations, part_joints, valid = data.anatomical_part_geometry(joints)
    indices = [15, 9, 16, 18, 20, 17, 19, 21, 1, 4, 7, 2, 5, 8]
    np.testing.assert_array_equal(part_joints, joints[:, indices])
    expected = joints[:, 25:40].mean(1) - joints[:, 20]
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(orientations[:, 4], expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(orientations, axis=-1), 1, atol=1e-6)
    assert valid.all()


def test_degenerate_anatomical_bone_is_explicitly_invalid_and_zero():
    data = module()
    joints = np.arange(2 * 144 * 3, dtype=np.float32).reshape(2, 144, 3)
    joints[0, 18] = joints[0, 16]
    orientations, part_joints, valid = data.anatomical_part_geometry(joints)
    assert not valid[0, 2]
    assert not orientations[0, 2].any() and not part_joints[0, 2].any()
    assert valid[1, 2]


@pytest.mark.parametrize('key,value', [('schema_version', 'smplx_dual_v0'),
    ('coordinate_convention', 'camera_only'), ('rotation6d_convention', 'rows'),
    ('partition_sha256', '0' * 64), ('source_shape_semantics', 'target_betas')])
def test_geometry_payload_rejects_wrong_metadata_before_tensor_loading(key, value):
    data = module()
    payload = data.geometry_schema_metadata()
    payload['source_shape_shared'] = np.array(True)
    payload[key] = np.array(value)
    with pytest.raises(ValueError, match=key):
        data.validate_geometry_schema(payload, data.generation_versions())


def test_geometry_payload_requires_metadata_in_npz():
    data = module()
    with pytest.raises(ValueError, match='schema_version'):
        data.validate_geometry_schema({}, data.generation_versions())


def test_geometry_validator_cannot_accept_matching_but_unsupported_manifest_and_payload():
    data = module()
    versions = data.generation_versions()
    versions['geometry'] = 'unverified_geometry'
    payload = dict(data.geometry_schema_metadata(),source_shape_shared=np.array(True))
    payload['schema_version'] = versions['geometry']
    with pytest.raises(ValueError, match='versions.geometry'):
        data.validate_geometry_schema(payload,versions)


def cached_source(tmp_path):
    import torch
    from PIL import Image
    data = module()
    rgb = tmp_path / 'source.png'
    masks = [tmp_path / 'a.png', tmp_path / 'b.png']
    Image.new('RGB', (32, 32), (50, 60, 70)).save(rgb)
    for i, path in enumerate(masks):
        mask = np.zeros((32, 32), np.uint8)
        mask[4:28, 2 + i * 16:14 + i * 16] = 255
        Image.fromarray(mask).save(path)
    row = manifest_row()
    row.update(source_rgb=str(rgb), source_masks=list(map(str, masks)),
        source_features=str(tmp_path / 'source_features.pt'),
        target_geometry=str(tmp_path / 'geometry.npz'), versions=data.generation_versions())
    features = dict(person_latents=torch.zeros(2, 64, 32, 32),
        latent_masks=torch.ones(2, 1, 32, 32), dino_patches=torch.zeros(2, 256, 1536),
        dino_valid=torch.ones(2, 256, dtype=torch.bool),
        metadata=data.source_cache_metadata(rgb, masks, row['person_uids']))
    return row, features


@pytest.mark.parametrize('damage', ['missing_metadata', 'rgb_changed', 'mask_changed',
    'uid_swapped', 'encoder_changed', 'crop_changed', 'schema_changed'])
def test_source_feature_cache_rejects_stale_or_misbound_metadata(tmp_path, damage):
    import torch
    from PIL import Image
    data = module()
    row, features = cached_source(tmp_path)
    if damage == 'missing_metadata':
        del features['metadata']
    elif damage == 'rgb_changed':
        Image.new('RGB', (32, 32), 'red').save(row['source_rgb'])
    elif damage == 'mask_changed':
        Image.new('L', (32, 32), 255).save(row['source_masks'][1])
    elif damage == 'uid_swapped':
        features['metadata']['person_uids'].reverse()
    else:
        features['metadata'][{'encoder_changed':'encoder_version',
            'crop_changed':'crop_version', 'schema_changed':'schema_version'}[damage]] = 'wrong'
    torch.save(features, row['source_features'])
    row['source_features_sha256'] = data.file_sha256(row['source_features'])
    path = tmp_path / 'generation.json'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='source feature cache'):
        data.Hi4DSMPLXDataset(path).generation()


def test_source_cache_validates_same_files_and_manifest_versions(tmp_path):
    data = module()
    row, features = cached_source(tmp_path)
    data.validate_source_feature_cache(features, row)


def test_source_cache_validator_cannot_accept_matching_but_unsupported_versions(tmp_path):
    data = module()
    row, features = cached_source(tmp_path)
    row['versions']['encoder'] = 'unverified_encoder'
    features['metadata']['encoder_version'] = 'unverified_encoder'
    with pytest.raises(ValueError, match='source feature cache.*versions.encoder'):
        data.validate_source_feature_cache(features,row)


def test_loader_checks_actual_pt_file_hash_before_unpickling(tmp_path):
    data = module()
    row, _ = cached_source(tmp_path)
    from pathlib import Path
    Path(row['source_features']).write_bytes(b'this is not the cached PT payload')
    path = tmp_path / 'generation.json'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='source feature cache SHA256'):
        data.Hi4DSMPLXDataset(path).generation()


def test_precompute_writes_schema_and_anatomical_provenance(tmp_path, monkeypatch):
    data = module()
    precompute = importlib.import_module('scripts.precompute.precompute_hi4d_smplx_controls')
    work = tmp_path / 'work'
    out = tmp_path / 'output'
    for path in [work / 'observations', work / 'fit_v1', out,
        tmp_path / 'datasets/Hi4D_pilot_v1/raw/pair14/talk14/smpl']:
        path.mkdir(parents=True)
    np.savez(work / 'fit_v1/smplx_world.npz', marker=np.array(1))
    np.savez(work / 'observations/talk14_000032_cam4.npz', width=512, height=512,
        K=np.array([[500.,0,256],[0,500.,256],[0,0,1]]),
        extrinsics=np.c_[np.eye(3), np.array([0, 0, 2.])])
    np.savez(tmp_path / 'datasets/Hi4D_pilot_v1/raw/pair14/talk14/smpl/000032.npz',
        contact=np.zeros((2, 6890), np.int64))
    labels = np.arange(10475) % 14
    vertices = np.ones((2, 10475, 3), np.float32)
    joints = np.arange(2 * 144 * 3, dtype=np.float32).reshape(2, 144, 3) / 100
    fit = dict(vertices_world=vertices, joints_world=joints,
        betas=np.zeros((2, 10), np.float32), global_orient=np.zeros((2, 3), np.float32),
        actions=np.array(['talk14', 'talk14']), frame_ids=np.array([32,32]), actor_ids=np.array([0,1]))
    np.savez(work / 'fit_v1/smplx_world.npz', **fit)
    class Renderer:
        def render(self, v, R, t, K):
            return dict(mask=np.ones((512,512), bool), depth=np.ones((512,512), np.float32),
                normal=np.zeros((3,512,512), np.float32), part=np.zeros((14,512,512), np.float32),
                vertex_normals_camera=np.zeros_like(v))
    monkeypatch.setattr(precompute, 'pose_observations',
        lambda *args: (np.zeros((2,25,512,512), np.float32), [['test'] * 24] * 2))
    path = precompute.controls(tmp_path, work, out, fit, np.array([0,1]),
        'talk14', 32, 4, labels, np.array([1.7,1.8]), Renderer())
    with np.load(path, allow_pickle=False) as payload:
        data.validate_geometry_schema(payload, data.generation_versions())
        assert payload['part_orientations'].shape == (2,14,3)
        np.testing.assert_array_equal(payload['part_joints'], joints[:, [15,9,16,18,20,17,19,21,1,4,7,2,5,8]])
        assert payload['source_shape_shared'].item() is True
    sidecar = json.loads(path.with_suffix('.json').read_text())
    assert sidecar['partition_sha256'] == data.SMPLX14_PARTITION_SHA256
    assert sidecar['geometry_sha256'] == data.file_sha256(path)
    assert sidecar['anatomical_joint_indices'] == [15,9,16,18,20,17,19,21,1,4,7,2,5,8]
    # Exercise the real loader, including the required anatomical fields and
    # explicit version propagation, after precompute has written a valid cache.
    import torch
    row, features = cached_source(tmp_path)
    row['target_geometry'] = str(path)
    row['person_uids'] = ['pair14_actor0', 'pair14_actor1']
    features['metadata'] = data.source_cache_metadata(row['source_rgb'], row['source_masks'], row['person_uids'])
    torch.save(features, row['source_features'])
    row['source_features_sha256'] = data.file_sha256(row['source_features'])
    manifest = tmp_path / 'generation.json'
    manifest.write_text(json.dumps(row))
    loaded = data.Hi4DSMPLXDataset(manifest, tmp_path/'missing_supervision.json').generation()
    assert loaded.geometry.version == row['versions']['geometry']
    assert loaded.source.version == row['versions']['source']
    np.testing.assert_allclose(loaded.geometry.part_joints[0].numpy(), joints[:, [15,9,16,18,20,17,19,21,1,4,7,2,5,8]])


def test_verified_legacy_upgrade_rejects_mismatched_feature_sha(tmp_path):
    import torch
    data = module()
    precompute = importlib.import_module('scripts.precompute.precompute_hi4d_smplx_controls')
    row, features = cached_source(tmp_path)
    torch.save(features, row['source_features'])
    (tmp_path / 'generation.json').write_text(json.dumps(row))
    (tmp_path / 'source_features.json').write_text(json.dumps(dict(source_features_sha256='0' * 64)))
    with pytest.raises(ValueError, match='legacy source feature SHA256'):
        precompute.upgrade_verified_pilot_controls(tmp_path, tmp_path / 'new', tmp_path / 'fit.npz')


def test_verified_legacy_upgrade_preserves_encoded_tensors(tmp_path):
    import torch
    from pathlib import Path
    from PIL import Image
    data = module()
    precompute = importlib.import_module('scripts.precompute.precompute_hi4d_smplx_controls')
    row, features = cached_source(tmp_path)
    del features['metadata']
    torch.save(features, row['source_features'])
    rgb = (data.read_rgb(row['source_rgb']).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    masks = np.stack([data.read_mask(p)[0].numpy() for p in row['source_masks']])
    crops, _, _ = data.source_person_crops(rgb, masks)
    for i, crop in enumerate(crops):
        Image.fromarray(crop).save(tmp_path / f'source_person_{i}.png')
    dino, vae = tmp_path / 'dino', tmp_path / 'vae'
    dino.mkdir(); vae.mkdir()
    for path in [dino / 'model.safetensors', vae / 'config.json']:
        path.write_bytes(b'verified checkpoint fixture')
    sidecar = dict(source_sha256=data.file_sha256(row['source_rgb']),
        source_mask_sha256=[data.file_sha256(p) for p in row['source_masks']],
        source_features_sha256=data.file_sha256(row['source_features']),
        dino_path=str(dino), vae_path=str(vae), dino_whole_crop_size=224, dino_min_patch_coverage=.5,
        dino_sha256=data.file_sha256(dino / 'model.safetensors'),
        vae_config_sha256=data.file_sha256(vae / 'config.json'))
    (tmp_path / 'source_features.json').write_text(json.dumps(sidecar))
    joints = np.arange(2 * 144 * 3, dtype=np.float32).reshape(2,144,3)/100
    fit = dict(betas=np.zeros((2,10),np.float32),global_orient=np.zeros((2,3),np.float32),
        joints_world=joints,actions=np.array(['talk14','talk14']),frame_ids=np.array([32,32]),actor_ids=np.array([0,1]))
    fit_path = tmp_path / 'fit.npz'
    np.savez(fit_path, **fit)
    K = np.array([[500.,0,256],[0,500.,256],[0,0,1]])
    E = np.c_[np.eye(3), np.array([0,0,2.])]
    heights = np.array([1.7,1.8])
    amodal = np.ones((2,512,512),bool)
    global32, relative14 = data.geometry_vectors(fit['betas'],fit['global_orient'],joints[:,0],
        E[:,:3],E[:,3],K,amodal,heights)
    geometry = tmp_path / 'talk14_000032_cam4_geometry.npz'
    np.savez(geometry,global32=global32,relative14=relative14,amodal_mask=amodal[:,None],
        part_positions=np.ones((2,14,3),np.float32),part_valid=np.ones((2,14),bool),
        part_confidence=np.ones((2,14),np.float32),person_uids=np.asarray(row['person_uids']))
    geometry.with_suffix('.json').write_text(json.dumps(dict(version='smplx_dual_v1',
        source_shape_shared=True,smplx_sha256=data.file_sha256(fit_path),
        K512=K.tolist(),extrinsics=E.tolist(),source_neutral_heights=heights.tolist())))
    row['target_geometry'] = str(geometry)
    row['versions'] = dict(source='source_v1',geometry='smplx_dual_v1')
    for name in ['generation.json','generation_talk_control.json']:
        (tmp_path / name).write_text(json.dumps(row))
    weights = np.zeros((10475,55),np.float32)
    weights[:,0] = 1
    model_path = tmp_path / 'SMPLX_NEUTRAL.npz'
    # Production SMPLX_NEUTRAL.npz uses `weights`; the library exposes that
    # array as model.lbs_weights after constructing the runtime object.
    np.savez(model_path,weights=weights)
    out = tmp_path / 'upgraded'
    precompute.upgrade_verified_pilot_controls(tmp_path,out,fit_path,smplx_model_path=model_path)
    upgraded = torch.load(out / 'source_features.pt',weights_only=True)
    for key in features:
        torch.testing.assert_close(upgraded[key],features[key],rtol=0,atol=0)
    new_row = json.loads((out / 'generation.json').read_text())
    data.validate_generation_manifest(new_row)
    data.validate_source_feature_cache(upgraded,new_row)
    with np.load(new_row['target_geometry']) as payload:
        data.validate_geometry_schema(payload,new_row['versions'])
        np.testing.assert_array_equal(payload['global32'],global32)
        assert payload['part_orientations'].shape == (2,14,3)
    assert data.file_sha256(row['source_features']) == sidecar['source_features_sha256']
    assert Path(new_row['source_features']).parent == out


def supervised_case(tmp_path):
    from PIL import Image
    data = module()
    source = tmp_path/'sources/pair14/cam4/talk14/000032'
    target = tmp_path/'targets/pair14/hug14/cam4/000062'
    source.mkdir(parents=True); target.mkdir(parents=True)
    for base in (source,target):
        Image.new('RGB',(32,32),(50,60,70)).save(base/'rgb.jpg')
    source_masks=[source/'mask_A.png',source/'mask_B.png']
    target_masks=[target/'gt_mask_A.png',target/'gt_mask_B.png']
    for paths in (source_masks,target_masks):
        for i,path in enumerate(paths):
            mask=np.zeros((32,32),np.uint8)
            mask[4:28,2+i*16:14+i*16]=255
            Image.fromarray(mask).save(path)
    geometry=tmp_path/'hug14_000062_cam4_geometry.npz'
    uids=['pair14_actor0','pair14_actor1']
    np.savez(geometry,**data.geometry_schema_metadata(),source_shape_shared=np.array(True),
        person_uids=np.asarray(uids),marker=np.array(1))
    sidecar=dict(data.geometry_schema_metadata(),geometry_sha256=data.file_sha256(geometry),
        pair='pair14',action='hug14',target_frame=62,source_frame=32,camera=4,person_uids=uids)
    geometry.with_suffix('.json').write_text(json.dumps(sidecar))
    generation=manifest_row()
    generation.update(source_rgb=str(source/'rgb.jpg'),source_masks=list(map(str,source_masks)),
        person_uids=uids,target_geometry=str(geometry),versions=data.generation_versions())
    truth=dict(schema_version='dual_supervision_v1',target_rgb=str(target/'rgb.jpg'),
        target_masks=list(map(str,target_masks)),target_rgb_sha256=data.file_sha256(target/'rgb.jpg'),
        target_mask_sha256=[data.file_sha256(p)for p in target_masks],
        target_geometry=str(geometry),target_geometry_sha256=data.file_sha256(geometry),
        target_geometry_sidecar_sha256=data.file_sha256(geometry.with_suffix('.json')),
        pair='pair14',action='hug14',source_frame=32,target_frame=62,camera=4,person_uids=uids)
    generation_path=tmp_path/'generation.json'
    supervision_path=tmp_path/'supervision.json'
    generation_path.write_text(json.dumps(generation))
    supervision_path.write_text(json.dumps(truth))
    return data,generation_path,supervision_path,generation,truth


@pytest.mark.parametrize('damage', ['rgb_content','mask_content','pair','action','frame','source_frame',
    'camera','uid','geometry_content','geometry_path','sidecar_content','missing_hash'])
def test_supervision_rejects_changed_content_or_generation_binding(tmp_path,damage):
    from pathlib import Path
    from PIL import Image
    data,generation_path,supervision_path,generation,truth=supervised_case(tmp_path)
    if damage=='rgb_content':
        Image.new('RGB',(32,32),'red').save(truth['target_rgb'])
    elif damage=='mask_content':
        Image.new('L',(32,32),255).save(truth['target_masks'][0])
    elif damage=='geometry_content':
        with Path(truth['target_geometry']).open('ab') as stream:
            stream.write(b'changed geometry content')
    elif damage=='geometry_path':
        alternate=tmp_path/'alternate_geometry.npz'
        alternate.write_bytes(Path(truth['target_geometry']).read_bytes())
        generation['target_geometry']=str(alternate)
        generation_path.write_text(json.dumps(generation))
    elif damage=='sidecar_content':
        sidecar=Path(truth['target_geometry']).with_suffix('.json')
        sidecar.write_text(sidecar.read_text()+'\n')
    elif damage=='missing_hash':
        del truth['target_rgb_sha256']
    else:
        key={'frame':'target_frame','uid':'person_uids'}.get(damage,damage)
        truth[key]=({'pair':'pair99','action':'talk14','target_frame':63,'source_frame':31,
            'camera':3,'person_uids':['pair14_actor1','pair14_actor0']})[key]
    supervision_path.write_text(json.dumps(truth))
    with pytest.raises(ValueError,match='supervision'):
        data.Hi4DSMPLXDataset(generation_path,supervision_path).supervision()


def test_supervision_returns_verified_content_hashes_for_checkpoint_binding(tmp_path):
    data,generation_path,supervision_path,_,truth=supervised_case(tmp_path)
    loaded=data.Hi4DSMPLXDataset(generation_path,supervision_path).supervision()
    assert loaded.rgb.shape==(1,3,512,512)
    assert loaded.person_masks.shape==(1,2,1,512,512)
    assert loaded.metadata['target_rgb_sha256']==data.file_sha256(truth['target_rgb'])
    assert loaded.metadata['target_mask_sha256']==[data.file_sha256(p)for p in truth['target_masks']]
    assert loaded.metadata['target_geometry_sha256']==data.file_sha256(truth['target_geometry'])
    assert loaded.metadata['supervision_manifest_sha256']==data.file_sha256(supervision_path)


def test_supervision_cannot_relabel_unbound_geometry_sidecar(tmp_path):
    from pathlib import Path
    data,generation_path,supervision_path,_,truth=supervised_case(tmp_path)
    sidecar=Path(truth['target_geometry']).with_suffix('.json')
    metadata=json.loads(sidecar.read_text())
    del metadata['action']
    sidecar.write_text(json.dumps(metadata))
    truth['target_geometry_sidecar_sha256']=data.file_sha256(sidecar)
    supervision_path.write_text(json.dumps(truth))
    with pytest.raises(ValueError,match='supervision.*action'):
        data.Hi4DSMPLXDataset(generation_path,supervision_path).supervision()


def test_v3_supervision_binding_reuses_v2_encoded_and_geometry_content(tmp_path):
    import shutil
    import torch
    from pathlib import Path
    data,generation_path,supervision_path,generation,truth=supervised_case(tmp_path)
    precompute=importlib.import_module('scripts.precompute.precompute_hi4d_smplx_controls')
    features=tmp_path/'source_features.pt'
    torch.save(dict(metadata=data.source_cache_metadata(generation['source_rgb'],generation['source_masks'],generation['person_uids'])),features)
    generation.update(source_features=str(features),source_features_sha256=data.file_sha256(features))
    generation_path.write_text(json.dumps(generation))
    (tmp_path/'generation_talk_control.json').write_text(json.dumps(generation))
    legacy_truth={key:truth[key]for key in ('target_rgb','target_masks','pair','action','source_frame','target_frame','camera')}
    supervision_path.write_text(json.dumps(legacy_truth))
    root=tmp_path/'root'
    work=root/'research/hi4d_smplx_pilot_20261011'
    (work/'fit_v1').mkdir(parents=True); (work/'observations').mkdir()
    fit=work/'fit_v1/smplx_world.npz'
    np.savez(fit,actions=np.asarray(['talk14','talk14','hug14','hug14']),
        frame_ids=np.array([32,32,62,62]),actor_ids=np.array([0,1,0,1]))
    obs=work/'observations/hug14_000062_cam4.npz'
    K=np.array([[500.,0,256],[0,500.,256],[0,0,1]])
    E=np.c_[np.eye(3),np.array([0,0,2.])]
    np.savez(obs,width=512,height=512,K=K,extrinsics=E)
    raw=root/'datasets/Hi4D_pilot_v1/raw/pair14/hug14'
    (raw/'smpl').mkdir(parents=True)
    contact=raw/'smpl/000062.npz'
    np.savez(contact,contact=np.zeros((2,6890),np.int64))
    for image,path in [(truth['target_rgb'],raw/'images/4/000062.jpg'),
        (truth['target_masks'][0],raw/'seg/img_seg_mask/4/0/000062.png'),
        (truth['target_masks'][1],raw/'seg/img_seg_mask/4/1/000062.png')]:
        path.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(image,path)
    geometry=Path(generation['target_geometry'])
    sidecar=json.loads(geometry.with_suffix('.json').read_text())
    for key in ('pair','action','target_frame','source_frame','camera','person_uids'):
        del sidecar[key]
    sidecar.update(smplx_sha256=data.file_sha256(fit),observation_sha256=data.file_sha256(obs),
        contact_source_sha256=data.file_sha256(contact),K512=K.tolist(),extrinsics=E.tolist())
    geometry.with_suffix('.json').write_text(json.dumps(sidecar))
    original_geometry_sha=data.file_sha256(geometry)
    original_features_sha=data.file_sha256(features)
    original_supervision_sha=data.file_sha256(supervision_path)
    out=tmp_path/'v3'
    precompute.bind_verified_pilot_supervision(tmp_path,out,root)
    new_generation=json.loads((out/'generation.json').read_text())
    assert new_generation['source_features']==str(features)
    assert data.file_sha256(new_generation['target_geometry'])==original_geometry_sha
    loaded=data.Hi4DSMPLXDataset(out/'generation.json',out/'supervision.json').supervision()
    assert loaded.metadata['target_geometry_sha256']==original_geometry_sha
    assert loaded.metadata['action']=='hug14' and loaded.metadata['target_frame']==62
    assert data.file_sha256(features)==original_features_sha
    assert data.file_sha256(supervision_path)==original_supervision_sha
    from src.evaluation.qwen21_dual_v8_metrics import EvaluationTarget
    evaluation=EvaluationTarget.read(out/'supervision.json')
    assert evaluation.metadata['target_frame']==62 and evaluation.metadata['camera']==4
    assert evaluation.rgb.shape==(512,512,3)

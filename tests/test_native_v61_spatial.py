"""CPU-only raw spatial contracts; never read RGB or training supervision."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.native_local_condition import NativeLocalConditionV4
from src.pose_control.qwen21_full.native_v6.contracts import PoseGeometry


def api():
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61, PoseSpatialConditionV61
    from src.pose_control.qwen21_full.native_v61.spatial import build_spatial_condition
    return PoseLinkConfigV61, PoseSpatialConditionV61, build_spatial_condition


def geometry_for(hw=(8, 8), people=(True, False), valid=None):
    h, w = hw
    valid = torch.ones(1, 1, h * 16, w * 16) if valid is None else valid
    projected = torch.zeros(1, h * w, 2)
    projected[:, 2 * w + 2, 0] = 1
    return PoseGeometry(torch.zeros(1, h * w, 512), torch.zeros(1, 2, h * w, 512),
        projected.clone(), projected, F.avg_pool2d(valid, 16).flatten(1),
        torch.zeros_like(projected), hw, torch.tensor([people]), ('canvas',))


def batch_for(geometry, valid=None):
    h, w = geometry.target_token_hw
    return {'valid_image_mask': torch.ones(1, 1, h * 16, w * 16) if valid is None else valid}


def local_for(hw=(128, 128)):
    def z(*shape, boolean=False):
        return torch.zeros(1, *shape, dtype=torch.bool if boolean else torch.float32)
    h, w = hw
    return NativeLocalConditionV4(
        person_valid=torch.tensor([[True, False]]), source_indices=torch.tensor([[0, -1]]),
        source_slot_ids=torch.tensor([[0, 1]]), face_landmarks=z(2, 72, 3), face_point_valid=z(2, 72, boolean=True),
        jaw_pose=z(2, 3), jaw_valid=z(2, boolean=True), expression=z(2, 10), expression_valid=z(2, boolean=True),
        face_boxes=z(2, 4), face_roi_valid=z(2, boolean=True), hand_pose=z(2, 2, 45),
        hand_pose_valid=z(2, 2, boolean=True), hand_keypoints=z(2, 2, 21, 3), hand_point_valid=z(2, 2, 21, boolean=True),
        mesh_bps=z(2, 2, 1024), mesh_valid=z(2, 2, boolean=True), hand_boxes=z(2, 2, 4),
        hand_roi_valid=z(2, 2, boolean=True), wrist_z=z(2, 2), region_masks=z(6, h, w), visibility=z(6, h, w),
        local_z=z(6, h, w), local_z_valid=z(6, h, w, boolean=True), part_visible=z(6, h, w, boolean=True),
        person_owner=z(2, h, w), scene_z=z(1, h, w), scene_z_valid=z(1, h, w, boolean=True),
        body_height=torch.tensor([[1.8, 0.]]), metadata=({'schema': 'native_local_condition_v4',
            'expression_mode': 'off', 'normal_space': 'camera_cv', 'hand_pose_convention': 'unavailable',
            'target_canvas_id': 'canvas', 'source_canvas_id': 'source-canvas', 'driver_record_sha256': 'driver-sha',
            'camera_id': 'camera', 'scale_group_id': 'scale', 'source_rgb_sha256': 'source-sha',
            'source_person_ids': ['a', '__inactive__'], 'target_person_ids': ['a', '__inactive__']},)).validate()


def repair_for():
    mask = torch.zeros(1, 2, 128, 128)
    mask[:, 0, 100:110, 100:110] = 1
    return {'schema': 'native_pose_repair_v61', 'role': 'source_repair', 'mask': mask,
        'metadata': ({'target_canvas_id': 'canvas', 'source_rgb_sha256s': ['source-sha'],
            'target_person_ids': ['a', '__inactive__'], 'driver_record_sha256': 'driver-sha'},)}


def test_config_roundtrip_preserves_all_v6_flags_and_defaults():
    Config, _, _ = api()
    config = Config(use_contact=False, use_role=False)
    assert (config.width, config.heads, len(config.injection_layers)) == (512, 8, 6)
    assert (config.global_token_budget, config.joint_token_budget, config.local_token_budget) == (128, 64, 64)
    assert (config.support_expand_px, config.edit_expand_px, config.feather_px) == (16, 16, 8)
    assert Config.from_dict(config.to_dict()) == config
    assert Config(global_token_budget=1, joint_token_budget=0, local_token_budget=0)


@pytest.mark.parametrize('overrides', [dict(max_pose_tokens_per_reference=257),
    dict(max_pose_tokens_per_reference=1024, global_token_budget=256, joint_token_budget=128, local_token_budget=128)])
def test_config_cannot_exceed_the_approved_reference_budget(overrides):
    Config, _, _ = api()
    with pytest.raises(ValueError, match='256'):
        Config(**overrides)


def test_no_geometry_support_has_an_explicit_diagnostic():
    Config, _, build = api()
    geometry = geometry_for()
    geometry.projected.zero_(); geometry.owner.zero_()
    spatial = build(batch_for(geometry), geometry, Config())
    assert spatial.metadata[0]['key_support_empty'] is True
    assert spatial.metadata[0]['key_support_reason'] == 'no_valid_geometry'


@pytest.mark.parametrize('kwargs', [dict(geometry_key_mask=1), dict(edit_region_gate=None),
    dict(multiscale_tokens='yes'), dict(global_token_budget=-1), dict(joint_token_budget=1.5),
    dict(global_token_budget=0, joint_token_budget=0, local_token_budget=0),
    dict(global_token_budget=256), dict(support_expand_px=-1), dict(edit_expand_px=True), dict(feather_px=.5)])
def test_config_rejects_invalid_flags_radii_and_budgets(kwargs):
    Config, _, _ = api()
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_support_is_per_person_bounded_soft_and_zero_in_distant_background():
    Config, _, build = api()
    geometry = geometry_for()
    condition = build(batch_for(geometry), geometry, Config())
    assert condition.validate(geometry) is condition
    assert condition.key_support.shape == (1, 2, 128, 128)
    assert condition.target_token_hw == (8, 8)
    assert condition.key_support[0, 0, 40, 40] == 1
    assert not condition.key_support[:, 1].any()
    assert not condition.key_support[:, :, 100:, 100:].any()
    assert ((condition.key_support > 0) & (condition.key_support < 1)).any()
    assert condition.metadata[0]['source_repair'] == 'missing'
    assert not condition.joint_valid.any() and not condition.local_valid.any()


def test_joint_uv_is_pixel_input_normalized_only_when_valid_and_inside_canvas():
    Config, _, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    uv = torch.zeros(1, 21, 2)
    uv[:, 9] = torch.tensor([80., 96.])
    uv[:, 7] = torch.tensor([64., 80.])
    uv[:, 14] = torch.tensor([32., 16.])
    uv[:, 0] = torch.tensor([128., 1.])
    valid = torch.zeros(1, 21, dtype=torch.bool)
    valid[:, [0, 7, 9, 14]] = True
    batch.update(joint_uv=uv, projection_valid=valid)
    condition = build(batch, geometry, Config())
    assert torch.equal(condition.joint_uv[0, 0, 9], torch.tensor([.625, .75]))
    assert not condition.joint_valid[0, 0, 0] and not condition.joint_uv[0, 0, 0].any()
    assert not condition.joint_valid[:, 1].any()
    assert condition.local_valid[0, 0].tolist() == [False, False, True, True, True]
    assert torch.allclose(condition.local_boxes[0, 0, 2], torch.tensor([.5, .625, .75, .875]))
    assert torch.allclose(condition.local_boxes[0, 0, 3], torch.tensor([.375, .5, .625, .75]))


def test_local_boxes_use_raw_roi_validity_and_never_invent_hand_observations():
    Config, _, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    local = local_for()
    local.hand_boxes[0, 0, 0] = torch.tensor([-.1, .1, .2, .3])
    local.hand_roi_valid[0, 0, 0] = True
    local.hand_boxes[0, 0, 1] = torch.tensor([.7, .7, .9, .9])
    local.face_boxes[0, 0] = torch.tensor([.1, .1, .4, .4])
    local.face_roi_valid[0, 0] = True
    batch['local_condition_v4'] = local
    condition = build(batch, geometry, Config())
    assert condition.local_valid[0, 0].tolist() == [True, False, False, False, True]
    assert torch.equal(condition.local_boxes[0, 0, 0], torch.tensor([0., .1, .2, .3]))
    assert not condition.local_boxes[:, 1].any()
    assert not condition.key_support[0, 0, 110:, 110:].any()


def test_gate_is_valid_pixel_conditional_mean_so_coverage_is_applied_once():
    Config, _, build = api()
    valid = torch.zeros(1, 1, 128, 128)
    valid[:, :, 5:123, 5:123] = 1
    geometry = geometry_for(valid=valid)
    geometry.projected[:, :, 0] = 1
    condition = build(batch_for(geometry, valid), geometry, Config(feather_px=0))
    covered = geometry.coverage > 0
    assert torch.equal(condition.edit_gate[covered], torch.ones_like(condition.edit_gate[covered]))
    assert torch.equal(condition.edit_gate * geometry.coverage, geometry.coverage)
    assert not (condition.key_support * (1 - valid)).any()


def test_explicit_aligned_source_repair_extends_only_gate_and_never_reads_training_fields():
    Config, _, build = api()
    geometry = geometry_for()
    class RawBatch(dict):
        def get(self, name, *args):
            assert name not in ('tgt_image', 'training_supervision_v5')
            return super().get(name, *args)
    batch = RawBatch(batch_for(geometry))
    batch.update(local_condition_v4=local_for(), pose_repair_v61=repair_for(),
        training_supervision_v5=object(), tgt_image=object())
    condition = build(batch, geometry, Config())
    assert condition.edit_gate[0, -1] > 0
    assert not condition.key_support[0, :, 100:, 100:].any()
    assert condition.metadata[0]['source_repair'] == 'aligned'


@pytest.mark.parametrize('field,value', [('target_canvas_id', 'wrong'), ('source_rgb_sha256s', ['wrong']),
    ('target_person_ids', ['__inactive__', 'a']), ('driver_record_sha256', 'wrong')])
def test_repair_provenance_mismatch_rejects(field, value):
    Config, _, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    repair = repair_for()
    repair['metadata'][0][field] = value
    batch.update(local_condition_v4=local_for(), pose_repair_v61=repair)
    with pytest.raises(ValueError, match=field):
        build(batch, geometry, Config())


@pytest.mark.parametrize('change', ['role', 'schema', 'dtype', 'shape', 'range', 'nan', 'missing-provenance'])
def test_explicit_malformed_repair_is_rejected(change):
    Config, _, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    batch['local_condition_v4'] = local_for()
    repair = repair_for()
    if change in ('role', 'schema'):
        repair[change] = 'training_supervision_v5'
    elif change == 'dtype': repair['mask'] = repair['mask'].double()
    elif change == 'shape': repair['mask'] = repair['mask'][:, :1]
    elif change == 'range': repair['mask'][0, 0, 0, 0] = 2
    elif change == 'nan': repair['mask'][0, 0, 0, 0] = float('nan')
    elif change == 'missing-provenance': batch.pop('local_condition_v4')
    batch['pose_repair_v61'] = repair
    with pytest.raises(ValueError): build(batch, geometry, Config())


def test_no_geometry_keys_remain_zero_even_when_repair_can_gate():
    Config, _, build = api()
    geometry = geometry_for()
    geometry.projected.zero_()
    batch = batch_for(geometry)
    batch.update(local_condition_v4=local_for(), pose_repair_v61=repair_for())
    condition = build(batch, geometry, Config())
    assert not condition.key_support.any()
    assert condition.edit_gate.any()


def test_record_transforms_preserve_gradients_fp32_masks_and_discrete_types():
    Config, Record, build = api()
    geometry = geometry_for()
    condition = build(batch_for(geometry), geometry, Config())
    support = condition.key_support.clone().requires_grad_()
    condition = replace(condition, key_support=support)
    selected = Record.stack([condition, condition]).to(torch.bfloat16).index_select(torch.tensor([1, 0]))
    assert selected.validate() is selected
    assert selected.valid_pixels.dtype == selected.key_support.dtype == selected.edit_gate.dtype == torch.float32
    assert selected.joint_valid.dtype == selected.local_valid.dtype == torch.bool
    assert selected.canvas_ids == ('canvas', 'canvas')
    selected.key_support.sum().backward()
    assert torch.equal(support.grad, torch.full_like(support, 2))
    selected.metadata[0]['source_repair'] = 'mutated'
    assert condition.metadata[0]['source_repair'] == 'missing'


def test_record_and_builder_reject_grid_canvas_coverage_and_person_mismatches():
    Config, Record, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    condition = build(batch, geometry, Config())
    with pytest.raises(ValueError, match='coverage'):
        condition.validate(replace(geometry, coverage=geometry.coverage * .5))
    with pytest.raises(ValueError, match='canvas'):
        condition.validate(replace(geometry, canvas_ids=('other',)))
    with pytest.raises(ValueError): Record.stack([condition, replace(condition, key_support=torch.zeros(1, 2, 16, 1024))])
    batch['local_condition_v4'] = replace(local_for(), person_valid=torch.tensor([[True, True]]))
    with pytest.raises(ValueError): build(batch, geometry, Config())


def test_canonical_canvas_must_match_pixels_grid_id_and_coverage():
    Config, _, build = api()
    canvas = CanvasSpec((118, 118), (118, 118), (128, 128), (5, 5))
    valid = canvas.valid_mask()[None]
    geometry = replace(geometry_for(valid=valid), canvas_ids=(canvas.fingerprint(),))
    batch = batch_for(geometry, valid)
    batch['target_canvas'] = [canvas.metadata()]
    assert build(batch, geometry, Config()).validate(geometry)
    batch['valid_image_mask'] = torch.ones_like(valid)
    with pytest.raises(ValueError): build(batch, geometry, Config())


def test_joint_in_padding_cannot_seed_support_or_become_a_valid_sparse_key():
    Config, _, build = api()
    valid = torch.zeros(1, 1, 128, 128)
    valid[:, :, 32:96, 32:96] = 1
    geometry = geometry_for(valid=valid)
    geometry.projected.zero_()
    batch = batch_for(geometry, valid)
    uv = torch.full((1, 21, 2), 30.)
    batch.update(joint_uv=uv, projection_valid=torch.ones(1, 21, dtype=torch.bool))
    condition = build(batch, geometry, Config())
    assert not condition.joint_valid.any()
    assert not condition.local_valid.any()
    assert not condition.key_support.any()


def test_source_bank_repair_binds_two_ordered_source_images_and_local_mapping(tmp_path):
    from tests.test_native_v6_data import _bank, _local
    Config, _, build = api()
    bank = _bank(tmp_path)
    canvas = CanvasSpec((128, 128), (128, 128), (128, 128))
    local = _local(bank, canvas, {'sha256': 'target-rgb'})
    geometry = replace(geometry_for(people=(True, True)), canvas_ids=(canvas.fingerprint(),))
    batch = batch_for(geometry)
    repair = repair_for()
    repair['metadata'][0].update(target_canvas_id=canvas.fingerprint(),
        source_rgb_sha256s=[source.rgb_sha256 for source in bank.sources],
        target_person_ids=['person0', 'person1'], driver_record_sha256='target_geometry')
    batch.update(source_bank_v5=bank, local_condition_v4=local, pose_repair_v61=repair)
    assert build(batch, geometry, Config()).metadata[0]['source_repair'] == 'aligned'
    swapped = deepcopy(repair)
    swapped['metadata'][0]['source_rgb_sha256s'].reverse()
    batch['pose_repair_v61'] = swapped
    with pytest.raises(ValueError, match='source_rgb_sha256s'):
        build(batch, geometry, Config())
    batch['pose_repair_v61'] = repair
    batch['local_condition_v4'] = replace(local, source_indices=torch.tensor([[1, 0]]))
    with pytest.raises(ValueError, match='source'):
        build(batch, geometry, Config())


def test_dual_raw_uv_person_order_and_invalid_boxes_stay_independent():
    Config, _, build = api()
    geometry = geometry_for(people=(True, True))
    geometry.projected.zero_()
    batch = batch_for(geometry)
    va = torch.zeros(1, 21, dtype=torch.bool)
    vb = va.clone()
    va[:, 0] = vb[:, 1] = True
    ua, ub = torch.zeros(1, 21, 2), torch.zeros(1, 21, 2)
    ua[:, 0], ub[:, 1] = torch.tensor([16., 16.]), torch.tensor([112., 112.])
    batch.update(joint_uv=ua, projection_valid=va, joint_uv_b=ub, projection_valid_b=vb)
    condition = build(batch, geometry, Config())
    assert condition.joint_valid[0, 0, 0] and condition.joint_valid[0, 1, 1]
    assert not condition.key_support[0, 0, 90:, 90:].any()
    assert not condition.key_support[0, 1, :40, :40].any()


def test_offline_repair_validation_is_raw_identity_preserving_and_accepts_unbatched_canvas_mask():
    from src.pose_control.qwen21_full.native_v61.spatial import validate_repair_condition
    repair = repair_for()
    batch = {'local_condition_v4': local_for(), 'valid_image_mask': torch.ones(1, 128, 128)}
    assert validate_repair_condition(repair, batch) is repair
    bad = deepcopy(repair)
    bad['metadata'][0]['driver_record_sha256'] = 'other'
    with pytest.raises(ValueError, match='driver_record_sha256'):
        validate_repair_condition(bad, batch)


def test_offline_single_source_repair_rejects_false_source_person_mapping():
    from src.pose_control.qwen21_full.native_v61.spatial import validate_repair_condition
    local = local_for()
    local.metadata[0]['source_person_ids'] = ['someone-else', '__inactive__']
    with pytest.raises(ValueError, match='source.*person'):
        validate_repair_condition(repair_for(), {'local_condition_v4': local})


def test_spatial_to_never_rounds_a_valid_boundary_uv_onto_padding():
    Config, _, build = api()
    geometry = geometry_for()
    batch = batch_for(geometry)
    uv = torch.zeros(1, 21, 2)
    uv[:, 0] = torch.tensor([127.9, 127.9])
    valid = torch.zeros(1, 21, dtype=torch.bool)
    valid[:, 0] = True
    batch.update(joint_uv=uv, projection_valid=valid)
    condition = build(batch, geometry, Config()).to(dtype=torch.bfloat16)
    assert condition.joint_uv.dtype == torch.float32
    assert condition.local_boxes.dtype == torch.float32
    assert condition.validate() is condition


@pytest.mark.parametrize('key,value', [('source_canvas_id', 'other-canvas'),
    ('source_rgb_sha256', 'other-rgb'), ('source_asset_sha256', 'other-geometry'),
    ('source_person_ids', ['person1', 'person0'])])
def test_offline_repair_rejects_source_bank_and_local_provenance_drift(tmp_path, key, value):
    from tests.test_native_v6_data import _bank, _local
    from src.pose_control.qwen21_full.native_v61.spatial import validate_repair_condition
    bank = _bank(tmp_path)
    canvas = CanvasSpec((128, 128), (128, 128), (128, 128))
    local = _local(bank, canvas, {'sha256': 'target-rgb'})
    repair = repair_for()
    repair['metadata'][0].update(target_canvas_id=canvas.fingerprint(),
        source_rgb_sha256s=[source.rgb_sha256 for source in bank.sources],
        target_person_ids=['person0', 'person1'], driver_record_sha256='target_geometry')
    local.metadata[0][key] = value
    with pytest.raises(ValueError, match='source'):
        validate_repair_condition(repair, {'local_condition_v4': local, 'source_bank_v5': bank})

"""V6.2 CPU routing and conditional branch-gate contracts."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.pose_control.qwen21_full.native_local_condition import NativeLocalConditionV4
from src.pose_control.qwen21_full.native_v6.contracts import PoseGeometry
from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61, PoseSpatialConditionV61
from src.pose_control.qwen21_full.native_v61.spatial import build_spatial_condition


def api():
    from src.pose_control.qwen21_full.native_v62.contracts import (
        PoseLinkConfigV62, RoutingConditionV62, BranchSpatialGatesV62)
    from src.pose_control.qwen21_full.native_v62.conditions import (
        build_routing_conditions, build_branch_gates, routing_permissions)
    return PoseLinkConfigV62, RoutingConditionV62, BranchSpatialGatesV62, build_routing_conditions, build_branch_gates, routing_permissions


def local_for(people=(True, True)):
    def z(*shape, boolean=False):
        return torch.zeros(1, *shape, dtype=torch.bool if boolean else torch.float32)
    ids = ['alice', 'bob' if people[1] else '__inactive__']
    return NativeLocalConditionV4(
        person_valid=torch.tensor([people]), source_indices=torch.tensor([[0, 1 if people[1] else -1]]),
        source_slot_ids=torch.tensor([[0, 1]]), face_landmarks=z(2, 72, 3), face_point_valid=z(2, 72, boolean=True),
        jaw_pose=z(2, 3), jaw_valid=z(2, boolean=True), expression=z(2, 10), expression_valid=z(2, boolean=True),
        face_boxes=z(2, 4), face_roi_valid=z(2, boolean=True), hand_pose=z(2, 2, 45),
        hand_pose_valid=z(2, 2, boolean=True), hand_keypoints=z(2, 2, 21, 3), hand_point_valid=z(2, 2, 21, boolean=True),
        mesh_bps=z(2, 2, 1024), mesh_valid=z(2, 2, boolean=True), hand_boxes=z(2, 2, 4), hand_roi_valid=z(2, 2, boolean=True),
        wrist_z=z(2, 2), region_masks=z(6, 32, 64), visibility=z(6, 32, 64), local_z=z(6, 32, 64),
        local_z_valid=z(6, 32, 64, boolean=True), part_visible=z(6, 32, 64, boolean=True), person_owner=z(2, 32, 64),
        scene_z=z(1, 32, 64), scene_z_valid=z(1, 32, 64, boolean=True), body_height=torch.tensor([[1.8, 1.7 if people[1] else 0.]]),
        metadata=({'schema': 'native_local_condition_v4', 'expression_mode': 'off', 'normal_space': 'camera_cv',
            'hand_pose_convention': 'unavailable', 'target_canvas_id': 'canvas', 'source_canvas_id': 'source-canvas',
            'driver_record_sha256': 'driver-sha', 'camera_id': 'camera', 'scale_group_id': 'scale',
            'source_rgb_sha256': 'source-sha', 'source_person_ids': list(ids), 'target_person_ids': list(ids)},)).validate()


def fixture(people=(True, True), valid=None):
    valid = torch.ones(1, 1, 32, 64) if valid is None else valid
    owner = torch.zeros(1, 8, 2)
    geometry = PoseGeometry(torch.zeros(1, 8, 512), torch.zeros(1, 2, 8, 512), owner,
        owner.clone(), F.avg_pool2d(valid, 16).flatten(1), owner.clone(), (2, 4), torch.tensor([people]), ('canvas',)).validate()
    batch = {'valid_image_mask': valid, 'local_condition_v4': local_for(people)}
    spatial = build_spatial_condition(batch, geometry, PoseLinkConfigV61(support_expand_px=0, edit_expand_px=0, feather_px=0))
    return batch, geometry, spatial


def repair_for(people=(True, True)):
    return {'schema': 'native_pose_repair_v61', 'role': 'source_repair', 'mask': torch.zeros(1, 2, 32, 64),
        'metadata': ({'target_canvas_id': 'canvas', 'source_rgb_sha256s': ['source-sha'],
            'target_person_ids': ['alice', 'bob' if people[1] else '__inactive__'], 'driver_record_sha256': 'driver-sha'},)}


def test_configuration_defaults_roundtrip_and_rejects_invalid_values():
    Config, *_ = api()
    config = Config()
    assert (config.width, config.heads, len(config.injection_layers), config.max_pose_tokens_per_reference) == (512, 8, 6, 256)
    assert config.person_routing and config.body_spatial_gate and config.interaction_spatial_gate
    assert config.owner_threshold == .75 and config.routing_epsilon == 1e-6
    assert Config.from_dict(config.to_dict()) == config
    for values in ({'person_routing': 1}, {'body_spatial_gate': None}, {'interaction_spatial_gate': 0},
            {'owner_threshold': .5}, {'owner_threshold': 1.01}, {'owner_threshold': float('nan')},
            {'routing_epsilon': 0}, {'routing_epsilon': float('inf')}, {'max_pose_tokens_per_reference': 257}):
        with pytest.raises(ValueError): Config(**values)


def permission_inputs():
    query_ids = torch.tensor([[0, 1, -1, 0]])
    query_parts = torch.zeros(1, 4, 2, 14)
    query_parts[0, :, 0, 2] = 1
    query_parts[0, :, 1, 3] = 1
    key_ids = torch.tensor([[0, 1, 1, -1, 0]])
    key_parts = torch.zeros(1, 5, 14)
    key_parts[0, 0, 2] = 1
    key_parts[0, 1, 3] = 1
    key_parts[0, 2, 4] = 1
    key_parts[0, 4, 2] = 1
    positive = torch.zeros(1, 14, 14, dtype=torch.bool)
    forbidden = torch.zeros_like(positive)
    valid = torch.tensor([[True, True, True, True, False]])
    return query_ids, query_parts, key_ids, key_parts, positive, forbidden, valid


def test_unknown_contacts_and_unknown_keys_do_not_open_hard_cross_person_reads():
    *_, permissions = api()
    args = permission_inputs()
    result = permissions(*args)
    assert result.dtype == torch.bool
    assert result[0].tolist() == [[True, False, False, False, False], [False, True, True, False, False],
        [True, True, True, False, False], [True, False, False, False, False]]


def test_contact_only_opens_matching_parts_and_transposes_reverse_direction():
    *_, permissions = api()
    args = permission_inputs()
    args[4][0, 2, 3] = True
    result = permissions(*args)
    assert result[0, 0, 1] and result[0, 1, 0]
    assert not result[0, 0, 2]
    args[5][0, 2, 3] = True
    vetoed = permissions(*args)
    assert not vetoed[0, 0, 1] and not vetoed[0, 1, 0]
    assert vetoed[0, 0, 0]  # forbidden relations do not disable the same person


def test_permissions_no_legal_keys_stays_all_false_and_zero_parts_do_not_open():
    *_, permissions = api()
    args = list(permission_inputs())
    args[4].fill_(True)
    args[1].zero_()
    assert not permissions(*args)[0, 0, 1]
    args[6].zero_()
    assert not permissions(*args).any()


def test_visible_owner_wins_then_projected_then_spatial_and_uncertainty_is_not_argmax():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    geometry.owner[0, :4] = torch.tensor([[.75, .25], [.5, .5], [.6, 0.], [.74, .26]])
    geometry.projected[0, :6] = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [.5, .5]])
    support = spatial.key_support.clone()
    support[0, 0, 16:, 32:48] = 1
    spatial = replace(spatial, key_support=support)
    record = build(batch, geometry, spatial, Config())
    assert record.query_person_ids.tolist() == [[0, -1, 0, -1, 1, -1, 0, -1]]
    assert record.person_ids == (('alice', 'bob'),)
    assert record.validate(geometry) is record
    assert record.query_parts.shape == (1, 8, 2, 14)
    assert record.metadata[0]['local_spatial_policy'] == 'inherited_owned_roi'


def test_repair_owner_is_validated_last_fallback_and_explicit_invalid_repair_rejects():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    repair = repair_for()
    repair['mask'][0, 1, :16, :16] = 1
    batch['pose_repair_v61'] = repair
    record = build(batch, geometry, spatial, Config())
    assert record.query_person_ids[0, 0] == 1
    repair['metadata'][0]['driver_record_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='driver_record_sha256'): build(batch, geometry, spatial, Config())


@pytest.mark.parametrize('change', ['target', 'source', 'canvas', 'active', 'raw-person'])
def test_stable_person_mapping_and_canvas_are_checked(change):
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    local = batch['local_condition_v4']
    if change == 'target': local.metadata[0]['target_person_ids'].reverse()
    elif change == 'source': local.metadata[0]['source_person_ids'][0] = 'other'
    elif change == 'canvas': local.metadata[0]['target_canvas_id'] = 'other'
    elif change == 'active': batch['local_condition_v4'] = replace(local, person_valid=torch.tensor([[True, False]]))
    elif change == 'raw-person': batch['pose_provenance'] = ({'person_id': 'wrong', 'canvas_id': 'canvas', 'record_sha256': 'driver-sha'},)
    with pytest.raises(ValueError): build(batch, geometry, spatial, Config())


def test_routing_requires_raw_stable_ids_and_disabled_mode_declares_missing_provenance():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    batch.pop('local_condition_v4')
    with pytest.raises(ValueError, match='local_condition_v4'): build(batch, geometry, spatial, Config())
    record = build(batch, geometry, spatial, Config(person_routing=False))
    assert record.person_ids == (('', ''),)
    assert record.query_person_ids.eq(-1).all()
    assert record.metadata[0]['person_ids_status'] == 'missing_routing_disabled'


def test_padding_and_inactive_person_never_get_hard_owner():
    Config, _, _, build, *_ = api()
    valid = torch.ones(1, 1, 32, 64)
    valid[..., :16, :16] = 0
    batch, geometry, spatial = fixture((True, False), valid)
    geometry.owner[:, :, 1] = 1  # inactive evidence must never become a real person
    record = build(batch, geometry, spatial, Config())
    assert record.query_person_ids.eq(-1).all()
    assert not record.part_tokens[:, 1].any()


def test_body_gate_requires_explicit_legal_repair_even_when_empty():
    Config, _, _, _, build, _ = api()
    batch, geometry, spatial = fixture()
    gate = build(batch, geometry, spatial, Config())
    assert torch.equal(gate.body, torch.ones_like(gate.body))
    assert gate.metadata[0]['body_gate_reason'] == 'missing_repair_legacy_coverage'
    batch['pose_repair_v61'] = repair_for()
    gate = build(batch, geometry, spatial, Config())
    assert not gate.body.any()
    assert gate.metadata[0]['body_gate_reason'] == 'explicit_repair'
    batch['pose_repair_v61']['role'] = 'training_supervision_v5'
    with pytest.raises(ValueError): build(batch, geometry, spatial, Config())


def test_interaction_gate_is_conditional_body_union_not_contact_only():
    Config, _, _, _, build, _ = api()
    valid = torch.ones(1, 1, 32, 64)
    valid[..., :8, :16] = 0
    batch, geometry, spatial = fixture(valid=valid)
    support = spatial.key_support.clone()
    support[0, 0, 8:16, :16] = 1
    support[0, 1, 16:, 48:] = 1
    spatial = replace(spatial, key_support=support)
    gate = build(batch, geometry, spatial, Config())
    assert gate.interaction[0, 0] == 1 and gate.interaction[0, -1] == 1
    assert gate.interaction[0, 1] == 0 and not geometry.contact.any()
    assert torch.equal(gate.interaction * geometry.coverage, F.avg_pool2d(support.amax(1, keepdim=True), 16).flatten(1))
    assert gate.metadata[0]['local_spatial_policy'] == 'inherited_owned_roi'
    batch, geometry, spatial = fixture((True, False))
    assert not build(batch, geometry, spatial, Config()).interaction.any()
    disabled = build({}, geometry, spatial, Config(body_spatial_gate=False, interaction_spatial_gate=False))
    assert disabled.body.eq(1).all() and disabled.interaction.eq(1).all()


def test_condition_transforms_preserve_exact_dtypes_gradients_and_metadata():
    Config, Routing, Gates, build, build_gates, _ = api()
    batch, geometry, spatial = fixture()
    record = build(batch, geometry, spatial, Config())
    leaf = record.part_tokens.clone().requires_grad_()
    record = replace(record, part_tokens=leaf)
    selected = Routing.stack([record, record]).to(torch.bfloat16).index_select(torch.tensor([1, 0]))
    assert selected.validate() is selected
    assert selected.part_tokens.dtype == torch.float32
    assert selected.query_person_ids.dtype == torch.int64 and selected.contact_positive.dtype == torch.bool
    selected.part_tokens.sum().backward()
    assert leaf.grad.eq(2).all()
    selected.metadata[0]['local_spatial_policy'] = 'changed'
    assert record.metadata[0]['local_spatial_policy'] == 'inherited_owned_roi'
    gates = build_gates(batch, geometry, spatial, Config())
    leaf = gates.body.clone().requires_grad_()
    selected_gates = Gates.stack([replace(gates, body=leaf)] * 2).to(dtype=torch.float16).index_select(torch.tensor([0, 1]))
    assert selected_gates.body.dtype == torch.float32
    selected_gates.body.sum().backward()
    assert leaf.grad.eq(2).all()
    assert selected_gates.validate() is selected_gates
    with pytest.raises(ValueError): selected_gates.validate(geometry)
    with pytest.raises(ValueError): record.validate(replace(geometry, canvas_ids=('other',)))
    with pytest.raises(ValueError): replace(record, part_tokens=record.part_tokens.half()).validate()
    with pytest.raises(ValueError): replace(gates, body=gates.body.half()).validate()


def depth_for(person, part):
    from src.pose_control.qwen21_full.depth_condition_v2 import (
        DepthFusionContext, DepthConditionV2, CONTEXT_SCHEMA, CONDITION_SCHEMA, FEATURE_SCHEMA, DETAIL_CHANNELS)
    blank = torch.zeros(1, 1, 32, 64)
    false = blank.bool()
    surface = part.sum(1, keepdim=True).bool()
    context = DepthFusionContext(torch.eye(3)[None], torch.tensor([2.]), torch.tensor([1.8]),
        torch.zeros(1, 3, 32, 64), false, part, surface.clone(), surface, torch.tensor([True]), torch.tensor([False]),
        ({'schema': CONTEXT_SCHEMA, 'normal_basis': 'unknown', 'canvas_id': 'canvas', 'pose_record_id': person,
            'units': 'model', 'scale_status': 'shared'},))
    metadata = ({'schema': CONDITION_SCHEMA, 'feature_schema': FEATURE_SCHEMA,
        'detail_channels': list(DETAIL_CHANNELS), 'kind': 'camera_z', 'polarity': 'near_one',
        'source_id': 'source', 'canvas_id': 'canvas', 'pose_canvas_id': 'canvas', 'camera_id': 'camera',
        'pose_record_id': person, 'pose_record_sha256': 'driver-sha', 'person_id': person, 'scale_group_id': 'scale'},)
    return DepthConditionV2(blank, false, torch.ones_like(false), torch.tensor([False]), blank.clone(), false.clone(),
        false.clone(), blank.clone(), torch.tensor([False]), context, torch.zeros(1, len(DETAIL_CHANNELS), 32, 64), metadata).validate()


def dual_for():
    from src.pose_control.qwen21_full.dual_depth_condition import DualDepthConditionV3, DualContactRelations, CONDITION_SCHEMA
    parts = torch.zeros(1, 2, 14, 32, 64)
    parts[0, 0, 2, :16, :16] = 1
    parts[0, 1, 3, :16, 16:32] = 1
    first, second = [depth_for(person, parts[:, p]) for p, person in enumerate(('alice', 'bob'))]
    blank = torch.zeros(1, 1, 32, 64)
    false = blank.bool()
    projected = torch.stack((first.context.smpl_mask, second.context.smpl_mask), 1)
    contacts = DualContactRelations(torch.tensor([[[0, 5000, 1, 6000]]]), torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), ({'kind': 'raw_vertices'},))
    return DualDepthConditionV3(first, second, blank, false, projected, torch.zeros_like(projected),
        torch.zeros_like(projected, dtype=torch.float32), torch.zeros(1, 8, 32, 64), false.clone(),
        torch.zeros_like(projected), false.clone(), torch.zeros(1, 14, 14), torch.zeros(1, 14, 14, dtype=torch.bool),
        torch.zeros(1, 14, 14, 2, 3), torch.zeros(1, 14, 14, dtype=torch.bool), torch.zeros(1, 14, 14, 4), contacts,
        torch.zeros(1, 2, dtype=torch.long), ({'schema': CONDITION_SCHEMA, 'mode': 'full_geometry', 'pair_id': 'pair',
            'canvas_id': 'canvas', 'camera_id': 'camera', 'scale_group_id': 'scale', 'record_sha256': 'driver-sha',
            'person_ids': ['alice', 'bob']},)).validate()


def test_typed_dual_parts_are_area_pooled_and_unmapped_vertex_contacts_do_not_open():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    dual = dual_for()
    batch['dual_depth'] = dual
    record = build(batch, geometry, spatial, Config())
    assert record.part_tokens[0, 0, 0, 2] == 1 and record.part_tokens[0, 1, 1, 3] == 1
    assert record.part_tokens.sum() == 2
    assert not record.contact_positive.any() and not record.contact_forbidden.any()
    assert record.metadata[0]['ignored_unmapped_vertex_contacts'] == 1
    dual.part_contact[0, 2, 3, :2] = 1
    record = build(batch, geometry, spatial, Config())
    assert record.contact_positive[0, 2, 3]
    dual.part_contact[0, 2, 3, 2:] = 1
    dual.part_contact[0, 5, 6, 1] = 1
    record = build(batch, geometry, spatial, Config())
    assert not record.contact_positive[0, 2, 3] and record.contact_forbidden[0, 2, 3]
    assert record.contact_forbidden[0, 5, 6]


def test_unknown_part_pixels_remain_zero_and_area_pooling_keeps_gradients():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    dual = dual_for()
    dual.person_a.context.part_valid.zero_()
    leaf = dual.person_b.context.part_onehot.clone().requires_grad_()
    dual = replace(dual, person_b=replace(dual.person_b, context=replace(dual.person_b.context, part_onehot=leaf)))
    batch['dual_depth'] = dual
    record = build(batch, geometry, spatial, Config())
    assert not record.part_tokens[:, 0].any()
    record.part_tokens.sum().backward()
    assert leaf.grad is not None and leaf.grad[0, 3, 0, 16] == 1 / 256


@pytest.mark.parametrize('field', ['person_ids', 'canvas_id', 'record_sha256'])
def test_raw_dual_provenance_must_match_local_order_and_target_canvas(field):
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    dual = dual_for()
    if field == 'person_ids': dual.metadata[0]['person_ids'].reverse()
    else: dual.metadata[0][field] = 'wrong'
    batch['dual_depth'] = dual
    with pytest.raises(ValueError): build(batch, geometry, spatial, Config())


def test_single_depth_has_only_person_zero_parts():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture((True, False))
    parts = torch.zeros(1, 14, 32, 64)
    parts[0, 4, :8, :16] = 1
    batch['depth_condition_v2'] = depth_for('alice', parts)
    record = build(batch, geometry, spatial, Config())
    assert record.part_tokens[0, 0, 0, 4] == .5
    assert not record.part_tokens[:, 1].any()
    assert not record.contact_positive.any()


def test_single_depth_checks_distinct_depth_and_pose_canvas_fingerprints():
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.depth_condition import canvas_identity
    Config, _, _, build, *_ = api()
    batch, geometry, _ = fixture((True, False))
    canvas = native_pad((32, 64))
    geometry = replace(geometry, canvas_ids=(canvas.fingerprint(),))
    batch['target_canvas'] = [canvas.metadata()]
    batch['local_condition_v4'].metadata[0]['target_canvas_id'] = canvas.fingerprint()
    depth = depth_for('alice', torch.zeros(1, 14, 32, 64))
    depth.metadata[0].update(canvas_id=canvas_identity(canvas.metadata()), pose_canvas_id=canvas.fingerprint())
    depth.context.metadata[0]['canvas_id'] = canvas_identity(canvas.metadata())
    assert depth.metadata[0]['canvas_id'] != depth.metadata[0]['pose_canvas_id']
    batch['depth_condition_v2'] = depth
    spatial = build_spatial_condition(batch, geometry, PoseLinkConfigV61())
    build(batch, geometry, spatial, Config())
    depth.metadata[0]['canvas_id'] = 'foreign-depth-canvas'
    depth.context.metadata[0]['canvas_id'] = 'foreign-depth-canvas'
    with pytest.raises(ValueError, match='Canvas'):
        build(batch, geometry, spatial, Config())


def test_all_new_features_off_do_not_require_new_dual_provenance():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture()
    batch['dual_depth'] = dual_for()
    batch.pop('local_condition_v4')
    record = build(batch, geometry, spatial, Config(person_routing=False,body_spatial_gate=False,interaction_spatial_gate=False))
    assert record.query_person_ids.eq(-1).all() and not record.contact_positive.any()


def test_single_depth_list_keeps_the_existing_frontend_input_form():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture((True,False))
    parts = torch.zeros(1,14,32,64);parts[:,4,:8,:16]=1
    depth = depth_for('alice',parts)
    batch['depth_condition_v2'] = depth
    expected = build(batch,geometry,spatial,Config())
    batch['depth_condition_v2'] = [depth]
    actual = build(batch,geometry,spatial,Config())
    assert torch.equal(expected.part_tokens,actual.part_tokens)


def test_single_part_evidence_rejects_foreign_driver_geometry():
    Config, _, _, build, *_ = api()
    batch, geometry, spatial = fixture((True,False))
    depth = depth_for('alice',torch.zeros(1,14,32,64))
    depth.metadata[0]['pose_record_sha256']='foreign-driver'
    batch['depth_condition_v2']=depth
    with pytest.raises(ValueError):build(batch,geometry,spatial,Config())


def test_gate_rejects_a_repair_bound_to_another_canvas_even_if_local_agrees():
    Config, _, _, _, build, _ = api()
    batch, geometry, spatial = fixture()
    batch['pose_repair_v61'] = repair_for()
    batch['local_condition_v4'].metadata[0]['target_canvas_id'] = 'other'
    batch['pose_repair_v61']['metadata'][0]['target_canvas_id'] = 'other'
    with pytest.raises(ValueError, match='canvas|Canvas'): build(batch, geometry, spatial, Config())


def test_source_bank_and_local_bindings_keep_explicit_order_without_invented_ids(tmp_path):
    from tests.test_native_v6_data import _bank, _local
    from src.data.canvas_contract import native_pad
    Config, _, _, build, *_ = api()
    bank = _bank(tmp_path)
    canvas = native_pad((32, 64))
    local = _local(bank, canvas, {'sha256': 'target'})
    _, geometry, _ = fixture()
    geometry = replace(geometry, canvas_ids=(canvas.fingerprint(),))
    batch = {'valid_image_mask': torch.ones(1, 1, 32, 64), 'local_condition_v4': local, 'source_bank_v5': bank}
    spatial = build_spatial_condition(batch, geometry, PoseLinkConfigV61())
    assert build(batch, geometry, spatial, Config()).person_ids == (('person0', 'person1'),)
    local.metadata[0]['source_rgb_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='source_rgb_sha256'): build(batch, geometry, spatial, Config())
    local.metadata[0]['source_rgb_sha256'] = bank.sources[0].rgb_sha256
    batch['source_bank_v5'] = replace(bank, target_qwen_image_indices=torch.tensor([[1, 0]]),
        target_person_ids=(('person1', 'person0'),))
    with pytest.raises(ValueError, match='mapping'): build(batch, geometry, spatial, Config())


@pytest.mark.parametrize('person',['person_a','person_b'])
def test_dual_child_parts_reject_foreign_driver_geometry(person):
    Config, _, _, build, *_ = api()
    batch,geometry,spatial=fixture()
    depth=dual_for();getattr(depth,person).metadata[0]['pose_record_sha256']='foreign-child-driver'
    batch['dual_depth']=depth
    with pytest.raises(ValueError):build(batch,geometry,spatial,Config())


def test_routing_off_does_not_rebind_unused_legacy_local_identity():
    Config, _, _, build, build_gates, *_ = api()
    batch,geometry,_=fixture()
    batch['local_condition_v4'].metadata[0]['source_person_ids'][0]='legacy-unbound-source'
    spatial=build_spatial_condition(batch,geometry,PoseLinkConfigV61())
    config=Config(person_routing=False,body_spatial_gate=False,interaction_spatial_gate=False)
    record=build(batch,geometry,spatial,config)
    gates=build_gates(batch,geometry,spatial,config)
    assert record.query_person_ids.eq(-1).all()
    assert gates.body.eq(1).all() and gates.interaction.eq(1).all()

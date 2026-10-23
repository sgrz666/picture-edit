"""Global Geometry v2 contracts; no model downloads, optimizers or training."""
import copy
import importlib
import math

import pytest
import torch


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def api():
    assert importlib.util.find_spec('src.pose_control.qwen21_full.body_pose') is not None, '21-joint contract missing'
    return importlib.import_module('src.pose_control.qwen21_full.body_pose')


def raw_inputs(dual=False):
    result = dict(normal_a=torch.rand(1, 3, 32, 64) * 2 - 1,
                  pose_heatmap_a=torch.rand(1, 25, 32, 64), part_onehot_a=torch.zeros(1, 14, 32, 64),
                  human_mask_a=torch.ones(1, 1, 32, 64), smplx_global_a=torch.randn(1, 26),
                  task_id=torch.tensor([int(dual)]))
    result['part_onehot_a'][:, 1] = 1
    if dual:
        result.update({k[:-1] + 'b': v.clone() for k, v in result.copy().items() if k.endswith('_a')})
    return result


def raw(batch=1, dual=False):
    result = {k: v.expand(batch, *v.shape[1:]).clone() for k, v in raw_inputs(dual=dual).items()}
    for person in ('a', 'b') if dual else ('a',):
        joints = torch.randn(batch, 22, 3)
        joints[..., 2] += 5
        result.update({f'body_pose_{person}': torch.randn(batch, 63) * .1,
                       f'joints_cam_{person}': joints,
                       f'joint_uv_{person}': torch.rand(batch, 21, 2) * torch.tensor([64., 32.]),
                       f'fk_valid_{person}': torch.ones(batch, 22, dtype=torch.bool),
                       f'projection_valid_{person}': torch.ones(batch, 21, dtype=torch.bool),
                       f'body_height_{person}': torch.ones(batch) * 1.7,
                       f'smpl_bbox_height_{person}': torch.ones(batch) * 30})
    return result


def test_rotation_columns_zero_near_pi_and_roundtrip():
    p = api()
    aa = torch.tensor([[0., 0., 0.], [math.pi - 1e-6, 0., 0.], [.3, -.5, 1.2]], dtype=torch.float64)
    matrix = p.axis_angle_to_matrix(aa)
    six = p.matrix_to_rotation_6d(matrix)
    assert torch.equal(six[0], torch.tensor([1., 0., 0., 0., 1., 0.], dtype=six.dtype))
    torch.testing.assert_close(six, torch.cat([matrix[..., :, 0], matrix[..., :, 1]], -1))
    torch.testing.assert_close(p.rotation_6d_to_matrix(six), matrix, atol=1e-12, rtol=1e-12)
    x = torch.zeros(2, 3, requires_grad=True)
    p.axis_angle_to_matrix(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_topology_keeps_pelvis_paths():
    p = api()
    distance = p.joint_tree_distances()
    assert distance.shape == (21, 21)
    assert distance[0, 1] == 2  # left hip -> pelvis -> right hip
    assert distance[0, 2] == 2  # left hip -> pelvis -> spine1
    assert distance.max() < 22 and torch.equal(distance, distance.T)


@pytest.mark.parametrize('rotation', ['rotation6d_columns', 'axis_angle'])
def test_encoder_shape_and_rejects_smpl69(rotation):
    p = api()
    from src.pose_control.qwen21_full.token_encoders import BodyPoseTokenEncoder
    encoder = BodyPoseTokenEncoder(p.BodyPoseConfig(rotation=rotation))
    data = raw()
    args = [data['body_pose_a'], data['joints_cam_a'], data['joint_uv_a'], data['fk_valid_a'],
            data['projection_valid_a'], data['body_height_a'], torch.tensor([[32., 64.]])]
    assert encoder(*args).shape == (1, 21, 256)
    with pytest.raises(ValueError, match='63'):
        encoder(torch.zeros(1, 69), *args[1:])


def test_native_default_requires_pose_legacy_is_explicit():
    api()
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    model = Qwen21NativeGeometry()
    with pytest.raises(ValueError, match='[Bb]ody|[Pp]ose'):
        model.condition_injector(**raw_inputs())
    assert model.get_extra_state()['schema'].endswith('_v2')
    old = Qwen21NativeGeometry(legacy=True)
    assert old.get_extra_state()['schema'].endswith('_v1')
    old.condition_injector(**raw_inputs()).validate()


def test_zero_joint_delta_exact_global_parity_and_bundle_subsets():
    api()
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    model = Qwen21NativeGeometry().eval()
    old = Qwen21NativeGeometry(legacy=True).eval()
    old.load_state_dict({k: v for k, v in model.state_dict().items() if k in old.state_dict() and k != '_extra_state'} |
                        {'_extra_state': old.get_extra_state()}, strict=True)
    data = raw(batch=2, dual=True)
    data['person_b_valid'] = torch.tensor([False, True])
    data['task_id'] = torch.tensor([0, 1])
    bundle = model.condition_injector(**data).validate()
    assert bundle.person_a_joint_tokens.shape == (2, 21, 256)
    assert bundle.person_a_global_tokens.shape == (2, 4, 256)
    selected = bundle.index_select(torch.tensor([1])).to(dtype=torch.float64).validate()
    assert selected.person_b_projection_valid.dtype == torch.bool
    torch.testing.assert_close(selected.person_b_joint_uv.float(), bundle.person_b_joint_uv[1:])
    with torch.no_grad():
        a = model(bundle, target_token_hw=(2, 4))
        b = old(bundle, target_token_hw=(2, 4))
    torch.testing.assert_close(a.geometry, b.geometry, atol=0, rtol=0)
    torch.testing.assert_close(a.interaction, b.interaction, atol=0, rtol=0)
    assert a.geometry.shape == (2, 8, 512)


def test_spatial_bias_follows_pixel_locations_and_all_mask_skips():
    p = api()
    from src.pose_control.qwen21_full.reasoning.person_geometry import JointGeometryAttention
    m = JointGeometryAttention(512, p.BodyPoseConfig())
    q = torch.randn(2, 8, 512, requires_grad=True)
    j = torch.randn(2, 21, 256, requires_grad=True)
    uv = torch.full((2, 21, 2), 24.)
    uv[:, 0] = torch.tensor([8., 8.])
    uv[:, 1] = torch.tensor([56., 24.])
    valid = torch.ones(2, 21, dtype=torch.bool)
    valid[1] = False
    bias = m.spatial_bias((2, 4), uv, torch.tensor([16., 16.]), torch.tensor([[32., 64.], [32., 64.]]))
    assert bias[0, 0, 0] > bias[0, 0, 1]
    assert bias[0, -1, 1] > bias[0, -1, 0]
    # Exactly one zero-initialized projection. Activate it only in-memory for gradient reachability.
    torch.nn.init.normal_(m.projection.weight, std=.01)
    delta = m(q, j, uv, valid, torch.tensor([16., 16.]), torch.tensor([[32., 64.], [32., 64.]]), (2, 4))
    assert torch.isfinite(delta).all() and torch.count_nonzero(delta[1]) == 0
    delta.square().sum().backward()
    assert j.grad[0].abs().sum() > 0 and torch.count_nonzero(j.grad[1]) == 0
    assert m.last_projectable.tolist() == [True, False]


def test_v1_checkpoint_migration_strict_keys_and_pose_trainability():
    api()
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    old = NativeDepthRuntime(Qwen21NativeGeometry(legacy=True), Qwen21NativeBody())
    saved = old.checkpoint_payload(metadata={'fixture': 'same canonical inputs'})
    with pytest.raises(ValueError, match='legacy'):
        NativeDepthRuntime.from_native_checkpoint(saved)
    restored = NativeDepthRuntime.from_native_checkpoint(saved, legacy=True)
    upgraded = NativeDepthRuntime.from_native_checkpoint(saved, upgrade_pose=True)
    assert restored.geometry.legacy and not upgraded.geometry.legacy
    assert all(p.requires_grad for p in upgraded.geometry.pose_parameters())
    assert not upgraded.geometry.condition_injector.global_encoder.network[1].weight.requires_grad
    broken = copy.deepcopy(saved)
    del broken['state_dict']['geometry.reasoner.role_embedding.weight']
    with pytest.raises(ValueError, match='keys'):
        NativeDepthRuntime.from_native_checkpoint(broken, upgrade_pose=True)


def test_tiny_actual_qwen_reaches_pose_once_without_prefix_or_base_change():
    api()
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody, NativeBodyConfig
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig
    geometry = Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(hidden_dim=32))
    system = NativeDepthRuntime(geometry, Qwen21NativeBody(NativeBodyConfig(hidden_dim=16, geometry_dim=32, width=8)))
    data = raw()
    batch = {k[:-2] if k.endswith('_a') else k: v for k, v in data.items()}
    batch['pose_provenance'] = [{'record_id': 'fixture', 'record_sha256': 'fixture'}]
    model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=32,
              num_attention_heads=2, attention_head_dim=8, context_in_dim=8,
              axes_dims_rope=(2, 2, 4), mlp_ratio=2).eval()
    original = list(model.parameters())
    for head in system.body.output_heads:
        torch.nn.init.normal_(head.weight, std=.03)
    torch.nn.init.normal_(geometry.reasoner.person_reasoner.joint_attention.projection.weight, std=.02)
    calls = []
    handle = geometry.condition_injector.body_pose_encoder.register_forward_hook(lambda *args: calls.append(1))
    features = system.encode(batch, target_token_hw=(2, 4))
    handle.remove()
    assert features.geometry.requires_grad and len(calls) == 1
    values = {'hidden_states': torch.randn(1, 12, 64), 'encoder_hidden_states': torch.randn(1, 3, 8),
              'encoder_hidden_states_mask': torch.ones(1, 3, dtype=torch.bool), 'timestep': torch.tensor([.5]),
              'img_shapes': [[(1, 2, 2), (1, 2, 4)]],
              'img_mask': torch.tensor([[True, False, False, True, True]]), 'return_dict': False}
    with torch.no_grad():
        baseline = model(**values)[0]
    install_native_body(model, system.body)
    model.enable_gradient_checkpointing()
    with bind_native_body(model, features):
        result = model(**values)[0]
    torch.testing.assert_close(result[:, :-8], baseline[:, :-8], atol=0, rtol=0)
    # 12 image positions + 2 expanded text positions; both are the base model's.
    assert result.shape == baseline.shape == (1, 14, 64)
    result[:, -8:].square().mean().backward()
    for module in (geometry.condition_injector.body_pose_encoder,
                   geometry.reasoner.person_reasoner.joint_attention):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(not p.requires_grad and p.grad is None for p in original)


def test_new_training_entry_requires_pose_and_explicit_legacy():
    from scripts.train.train_qwen21_native_depth import arguments, validate_arguments
    argv = ['--pairs', 'pairs', '--output', 'run', '--full-checkpoint', 'full']
    with pytest.raises(ValueError, match='pose-manifest'):
        validate_arguments(arguments(argv))
    validate_arguments(arguments(argv + ['--pose-manifest', 'pose.jsonl']))
    validate_arguments(arguments(argv + ['--geometry-version', 'legacy_v1']))


def test_strict_native_header_and_unknown_state_keys_are_rejected():
    api()
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody, NativeBodyConfig
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig
    r = NativeDepthRuntime(Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(hidden_dim=32)),
                           Qwen21NativeBody(NativeBodyConfig(hidden_dim=16, geometry_dim=32, width=8)))
    saved = r.checkpoint_payload(metadata={})
    broken = {**saved, 'architecture': 'qwen21_native_depth_runtime_v1'}
    with pytest.raises(ValueError, match='architecture'):
        NativeDepthRuntime.from_native_checkpoint(broken)
    broken = copy.deepcopy(saved)
    broken['state_dict']['undeclared.weight'] = torch.ones(1)
    with pytest.raises(ValueError, match='keys'):
        r.load_native_weights(broken, include_depth=True)


def test_projection_validity_cannot_claim_an_outside_joint_is_visible():
    p = api()
    from src.pose_control.qwen21_full.token_encoders import BodyPoseTokenEncoder
    data = raw()
    data['joint_uv_a'][0, 0, 0] = 1000
    with pytest.raises(ValueError, match='projection'):
        BodyPoseTokenEncoder()(data['body_pose_a'], data['joints_cam_a'], data['joint_uv_a'],
                              data['fk_valid_a'], data['projection_valid_a'], data['body_height_a'], torch.tensor([[32., 64.]]))


def test_dual_runtime_requires_separate_provenance_and_collate_keeps_rows():
    api()
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig
    from src.pose_control.qwen21_full.data_runtime import qwen_full_collate
    r = NativeDepthRuntime(Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(hidden_dim=32)),
                           Qwen21NativeBody(NativeBodyConfig(hidden_dim=16, geometry_dim=32, width=8)))
    data = raw(dual=True)
    batch = {k[:-2] if k.endswith('_a') else k: v for k, v in data.items()}
    batch['pose_provenance'] = [{'record_id': 'A', 'record_sha256': 'A', 'canvas_id': 'first'}]
    with pytest.raises(ValueError, match='provenance'):
        r.encode(batch, target_token_hw=(2, 4))
    batch['pose_provenance_b'] = [{'record_id': 'B', 'record_sha256': 'B', 'canvas_id': 'first'}]
    features = r.encode(batch, target_token_hw=(2, 4))
    assert features.person_valid.tolist() == [[True, True]]
    batch['pose_provenance'][0]['canvas_id'] = 'second'  # same raw data, a different canvas is allowed
    r.encode(batch, target_token_hw=(2, 4))
    rows = [{'value': torch.tensor(i), 'pair_metadata': {'pair_id': str(i)},
             'pose_provenance': {'record_id': str(i), 'record_sha256': str(i)}} for i in range(2)]
    collated = qwen_full_collate(rows)
    assert isinstance(collated['pose_provenance'], list)
    assert collated['pose_provenance'][1]['record_id'] == '1'

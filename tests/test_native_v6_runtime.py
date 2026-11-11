"""CPU contracts; no datasets, pretrained weights, GPU, or optimizer steps."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def test_v6_api_is_independent():
    from src.pose_control.qwen21_full.native_v6.runtime import NativeFullRuntimeV6, ARCHITECTURE
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    assert ARCHITECTURE == 'native_full_v6'
    assert NativeFullRuntimeV6 is not NativeFullRuntimeV5
    with pytest.raises(ValueError, match='V6'):
        NativeFullRuntimeV6.from_checkpoint({'architecture': 'native_full_v5'})


def test_contact_support_is_local_and_unknown_stays_zero():
    from src.pose_control.qwen21_full.native_v6.frontend import contact_support
    part = torch.zeros(1, 2, 14, 32, 64)
    part[:, 0, 2, :16, :16] = 1
    part[:, 1, 3, :16, 16:32] = 1
    flags = torch.zeros(1, 14, 14, 4)
    depth = SimpleNamespace(person_part=part, part_contact=flags)
    assert not contact_support(depth).any()
    flags[:, 2, 3, :2] = 1
    support = contact_support(depth)
    assert support.shape == (1, 8, 2)
    assert support[0, 0, 0] == 1 and support[0, 1, 1] == 1
    assert not support[0, 4:].any()
    flags[:, 2, 3, 0] = 0
    assert not contact_support(depth).any()


def test_frontend_preserves_pre_identity_geometry_and_runs_reasoner_once(monkeypatch):
    from torch import nn
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5, NativeConfigV5
    from src.pose_control.qwen21_full.native_v6 import frontend
    people = torch.tensor([[True, False]])
    raw = torch.randn(1, 8, 512, requires_grad=True)
    state = SimpleNamespace(person_valid=people, person_tokens=torch.randn(1, 2, 8, 512),
                            person_token_mask=people[:, :, None].expand(-1, -1, 8))
    calls = []
    bundle = SimpleNamespace(person_valid=people, person_a_global_tokens=torch.zeros(1, 4, 256),
                             person_a_joint_tokens=torch.zeros(1, 21, 256))
    class Geometry(nn.Module):
        def reasoner(self, bundle):
            calls.append('reasoner')
            return state
        def project_state(self, state, target_token_hw):
            return NativeGeometryFeatures(raw, torch.zeros_like(raw), target_token_hw, people)
    base = nn.Module()
    base.geometry = Geometry()
    base.family_embedding = nn.Embedding(2, 256)
    base.identity_binder = object()
    base.identity_fusion = lambda value, update, context, valid: update
    base.identity_provenance = {}
    runtime = SimpleNamespace(base=base, config=NativeConfigV5())
    class Identity:
        source_indices = torch.tensor([[0, -1]])
        source_valid = people
        metadata = ({'target_person_ids': ['a', '__inactive__']},)
        def to(self, *args): return self
        def validate(self): return self
    identity = Identity()
    batch = {'native_scene_kind': 'single', 'model_family_a': torch.zeros(1, dtype=torch.long),
        'single_owner_weights': torch.ones(1, 1, 32, 64), 'valid_image_mask': torch.ones(1, 1, 32, 64),
        'pose_provenance': [{'canvas_id': 'canvas', 'person_id': 'a'}],
        'source_indices': identity.source_indices, 'identity_metadata': identity.metadata}
    monkeypatch.setattr(frontend, 'build_condition_bundle', lambda *a, **k: bundle)
    monkeypatch.setattr(frontend, 'identity_update', lambda *a, **k: torch.full_like(raw, 7.))
    features, _, _, pose = frontend.encode_frontend(runtime, batch, (2, 4),
        DualControlOptions(depth_mode='off', detail=False, interaction=False, raster=False, part_contact=False),
        NativeControlOptionsV5(), identity)
    assert calls == ['reasoner']
    assert torch.equal(pose.geometry, raw)
    assert torch.allclose(features.geometry, raw + 7.)
    pose.geometry.sum().backward()
    assert raw.grad.abs().sum() > 0


def test_v6_training_default_prepare_never_constructs_model():
    from src.pose_control.qwen21_full.native_v6.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v6.training.trainer import NativeV6Trainer
    config = TrainingConfig()
    report = NativeV6Trainer(config).prepare()
    assert report['schema'] == 'native_v6_training_v1'
    assert not report['model_constructed'] and not report['optimizer_created']
    assert not report['training_started']
    assert config.qwen_condition_profile == 'single_2ref'
    assert config.stage.routes[0].branch == 'poselink'


def test_v6_training_accepts_five_reference_profile_and_spatial_loss():
    from src.pose_control.qwen21_full.native_v6.training.config import TrainingConfig, LossSpec
    config = replace(TrainingConfig(), qwen_condition_profile='dual_source_5ref')
    assert TrainingConfig.from_dict(config.record()) == config
    assert LossSpec(spatial_joint=.5).spatial_joint == .5
    with pytest.raises(ValueError):
        replace(config, qwen_condition_profile='legacy_single')


def test_migration_and_strict_round_trip_cover_all_parameters():
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody, NativeBodyConfig
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.native_v6.runtime import NativeFullRuntimeV6
    cfg = NativeConfigV5(verification_only=True, hidden_dim=32, num_layers=6, injection_layers=tuple(range(6)))
    body = Qwen21NativeBody(NativeBodyConfig(hidden_dim=32, num_layers=6, injection_layers=tuple(range(6))))
    old = NativeFullRuntimeV5(NativeDualRuntime(body=body), cfg, profile='frozen')
    payload = old.checkpoint_payload(metadata={'test': 'synthetic'})
    new = NativeFullRuntimeV6.from_v5_checkpoint(payload)
    assert new.get_extra_state()['architecture'] == 'native_full_v6'
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items() if torch.is_tensor(v))
    rows = [item for group in new.named_parameter_groups().values() for item in group]
    assert {n for n, _ in rows} == set(dict(new.named_parameters()))
    assert len({id(p) for _, p in rows}) == len(rows)
    new.set_trainability('poselink-heads')
    assert all(n.startswith('poselink.output_heads.') for n, p in new.named_parameters() if p.requires_grad)
    saved = new.checkpoint_payload(metadata={})
    assert 'training_state' not in saved
    rebuilt = NativeFullRuntimeV6.from_checkpoint(saved)
    assert rebuilt.get_extra_state() == new.get_extra_state()
    assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in new.state_dict().items() if torch.is_tensor(v))
    bad = dict(saved, runtime_config={**saved['runtime_config'], 'qwen_tokens_added': 1})
    with pytest.raises(ValueError):
        NativeFullRuntimeV6.from_checkpoint(bad)


def test_training_defaults_and_poselink_warmup_gate():
    from src.pose_control.qwen21_full.native_v6.training.config import stage_preset, RouteSpec
    from src.pose_control.qwen21_full.native_v6.training.validation import check_warmed_heads
    assert stage_preset('S0').routes[0].poselink_strength == 0
    assert RouteSpec('joint', 'joint').poselink_strength == 1
    assert RouteSpec('hand', 'hand').poselink_strength == 0
    for phase in ('S1', 'S2'):
        assert all(r.loss.face == r.loss.hand == r.loss.identity == r.loss.perceptual == 0
                   for r in stage_preset(phase).routes)
    row = SimpleNamespace(name='head', runtime_group='poselink_heads', role='head')
    catalog = SimpleNamespace(rows=[row], parameters={'head': torch.zeros(2, 2)})
    data = SimpleNamespace(records=[{'role': 'train', 'eligible': True, 'branch_usable': {'poselink': True}}])
    with pytest.raises(ValueError, match='poselink_heads'):
        check_warmed_heads(catalog, {'head'}, data)
    catalog.parameters['head'][0, 0] = 1
    assert check_warmed_heads(catalog, {'head'}, data) == ['poselink_heads']


def test_stage_advance_uses_declared_adam_hyperparameters_without_step():
    from src.pose_control.qwen21_full.native_v6.training.config import OptimizerSpec
    from src.pose_control.qwen21_full.native_v6.training.parameters import ParameterCatalog
    parameter = torch.nn.Parameter(torch.ones(2))
    previous = torch.optim.AdamW([{'params': [parameter], 'name': 'test', 'names': ['p']}],
                                betas=(.9, .999), eps=1e-8)
    spec = OptimizerSpec(betas=(.8, .99), eps=1e-6, weight_decay=.02)
    current = torch.optim.AdamW([{'params': [parameter], 'name': 'test', 'names': ['p']}],
                               betas=spec.betas, eps=spec.eps)
    current.load_state_dict(previous.state_dict())
    catalog = SimpleNamespace(group_spec=lambda value: [{'name': 'test', 'names': ['p'],
                                                       'lr': 2e-5, 'weight_decay': value.weight_decay}])
    ParameterCatalog.set_stage_lrs(catalog, current, spec)
    assert current.param_groups[0]['betas'] == spec.betas
    assert current.param_groups[0]['eps'] == spec.eps
    assert current.param_groups[0]['lr'] == 2e-5
    assert current.param_groups[0]['weight_decay'] == spec.weight_decay
    assert current.state == {}

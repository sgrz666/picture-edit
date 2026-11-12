"""V6.1 versioning and safe execution interfaces use synthetic CPU weights only."""
from dataclasses import replace
import pytest
import torch


def test_default_prepare_has_independent_schema_and_no_execution():
    from src.pose_control.qwen21_full.native_v61.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v61.training.trainer import NativeV61Trainer
    cfg = TrainingConfig()
    report = NativeV61Trainer(cfg).prepare()
    assert report['schema'] == 'native_v61_training_v1'
    assert not report['model_constructed'] and not report['optimizer_created'] and not report['training_started']
    assert cfg.stage.routes[0].branch == 'poselink'
    assert TrainingConfig.from_dict(cfg.record()) == cfg


def test_migration_strict_readback_and_catalog():
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_v6.runtime import NativeFullRuntimeV6
    from src.pose_control.qwen21_full.native_v6.contracts import PoseLinkConfig
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v61.training.parameters import ParameterCatalog
    from src.pose_control.qwen21_full.native_v61.training.config import stage_preset
    layers = tuple(range(6))
    cfg = NativeConfigV5(verification_only=True, hidden_dim=32, num_layers=6, injection_layers=layers)
    base = NativeDualRuntime(body=Qwen21NativeBody(NativeBodyConfig(hidden_dim=32, num_layers=6, injection_layers=layers)))
    old = NativeFullRuntimeV6(base, cfg, pose_config=PoseLinkConfig(hidden_dim=32, width=32, heads=4,
        injection_layers=layers, max_pose_tokens_per_reference=16), profile='frozen')
    saved = old.checkpoint_payload(metadata={})
    saved['training_state'] = {'must_not_resume': True}
    new = NativeFullRuntimeV61.from_v6_checkpoint(saved)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items() if torch.is_tensor(v))
    assert new.pose_config.global_token_budget + new.pose_config.joint_token_budget + new.pose_config.local_token_budget <= 16
    assert new.get_extra_state()['architecture'] == 'native_full_v61'
    catalog = ParameterCatalog(new)
    active = catalog.apply(stage_preset('S1', 'poselink'))
    assert all(name.startswith('poselink.output_heads.') for name in active)
    active = catalog.apply(stage_preset('S2', 'poselink'))
    assert 'poselink.token_type_embedding.weight' in active
    payload = new.checkpoint_payload(metadata={})
    assert 'training_state' not in payload
    rebuilt = NativeFullRuntimeV61.from_checkpoint(payload)
    assert rebuilt.get_extra_state() == new.get_extra_state()
    assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in new.state_dict().items() if torch.is_tensor(v))
    with pytest.raises(ValueError): NativeFullRuntimeV6.from_checkpoint(payload)
    with pytest.raises(ValueError): NativeFullRuntimeV61.from_checkpoint(saved)

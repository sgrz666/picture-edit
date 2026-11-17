"""Strict version migration and stage membership for all new V7 parameters."""
import pytest
import torch


def test_inference_ablation_updates_effective_flags_and_header_without_rebuilding_parameters():
    from torch import nn
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    from src.pose_control.qwen21_full.native_v7.poselink import PoseLinkAttentionV7
    from src.pose_control.qwen21_full.native_v7.local import NativeLocalControllerV7
    # Keep the production V7 controllers/config/header path, without allocating
    # frozen geometry encoders that this inference-only setter never touches.
    class HeaderBase(nn.Module):
        def get_extra_state(self):return {'unchanged_base_header':True}
    runtime=NativeFullRuntimeV7.__new__(NativeFullRuntimeV7);nn.Module.__init__(runtime)
    runtime.config=NativeConfigV5(verification_only=True,hidden_dim=16,num_layers=6,injection_layers=tuple(range(6)))
    runtime.base=HeaderBase();runtime.bps_convention=None;runtime.face_parameter_basis=None
    runtime.pose_config=PoseLinkConfigV7(hidden_dim=16,width=8,heads=2,injection_layers=tuple(range(6)))
    runtime.poselink=PoseLinkAttentionV7(runtime.pose_config)
    runtime.face_controller=NativeLocalControllerV7(runtime.config)
    runtime.hand_controller=NativeLocalControllerV7(runtime.config,hand=True)
    parameters=dict(runtime.named_parameters())
    values={name:parameter.detach().clone() for name,parameter in parameters.items()}
    assert runtime.set_ablation(identity_modulation=False,spatial_rope=False) is runtime
    assert runtime.pose_config is runtime.poselink.config
    assert runtime.pose_config.mask_pose_fusion and runtime.pose_config.geometry_contact
    assert not runtime.pose_config.identity_modulation and not runtime.pose_config.spatial_rope
    assert not runtime.face_controller.identity_modulation_enabled
    assert not runtime.hand_controller.identity_modulation_enabled
    assert runtime.get_extra_state()['pose_config']==runtime.pose_config.to_dict()
    assert runtime.set_ablation(mask_pose_fusion=False,geometry_contact=False) is runtime
    assert not any(getattr(runtime.pose_config,name) for name in
        ('mask_pose_fusion','spatial_rope','geometry_contact','identity_modulation'))
    assert runtime.set_ablation(identity_modulation=True) is runtime
    assert runtime.face_controller.identity_modulation_enabled and runtime.hand_controller.identity_modulation_enabled
    assert runtime.pose_config.identity_modulation and not runtime.pose_config.mask_pose_fusion
    previous=runtime.pose_config
    for invalid in ({'person_routing':False},{'unknown':True},{'mask_pose_fusion':1},
            {'identity_modulation':False,'spatial_rope':'yes'}):
        with pytest.raises(ValueError):runtime.set_ablation(**invalid)
        assert runtime.pose_config is previous and runtime.poselink.config is previous
        assert runtime.face_controller.identity_modulation_enabled and runtime.hand_controller.identity_modulation_enabled
    assert all(dict(runtime.named_parameters())[name] is parameter for name,parameter in parameters.items())
    assert all(torch.equal(values[name],parameter) for name,parameter in parameters.items())


def test_v62_weights_migrate_and_v7_checkpoint_rebuild_is_strict():
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody,NativeBodyConfig
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.native_v62.contracts import PoseLinkConfigV62
    from src.pose_control.qwen21_full.native_v62.runtime import NativeFullRuntimeV62
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    from src.pose_control.qwen21_full.native_v62.training.parameters import ParameterCatalog
    from src.pose_control.qwen21_full.native_v62.training.config import stage_preset
    layers=tuple(range(6));cfg=NativeConfigV5(verification_only=True,hidden_dim=16,num_layers=6,injection_layers=layers)
    base=NativeDualRuntime(body=Qwen21NativeBody(NativeBodyConfig(hidden_dim=16,num_layers=6,injection_layers=layers)))
    old=NativeFullRuntimeV62(base,cfg,pose_config=PoseLinkConfigV62(hidden_dim=16,width=8,heads=2,
        injection_layers=layers,max_pose_tokens_per_reference=16,global_token_budget=8,joint_token_budget=4,local_token_budget=4))
    saved=old.checkpoint_payload(metadata={});saved['training_state']={'optimizer':'do not inherit'}
    new=NativeFullRuntimeV7.from_v62_checkpoint(saved)
    assert new.get_extra_state()['architecture']=='native_full_v7'
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items() if torch.is_tensor(v))
    report=new.inheritance_report
    assert report['optimizer_inherited'] is False and report['rng_inherited'] is False
    assert set(report['inherited'])|set(report['adapted'])|set(report['new_initialized'])==set(dict(new.named_parameters()))
    assert set(report['new_initialized']) and any('identity_modulation' in n for n in report['new_initialized'])
    catalog=ParameterCatalog(new)
    head_names={r.name for r in catalog.rows if r.runtime_group=='poselink_heads'}
    assert all('output_heads' in n for n in head_names)
    assert any('contact_' in n for n in head_names)
    s1=catalog.apply(stage_preset('S1','poselink'));assert head_names==s1
    s2=catalog.apply(stage_preset('S2','poselink'))
    assert all(n in s2 for n in report['new_initialized'] if n.startswith('poselink.'))
    s3=catalog.apply(stage_preset('S3'))
    assert set(report['new_initialized'])<=s3
    payload=new.checkpoint_payload(metadata={});assert 'training_state' not in payload
    rebuilt=NativeFullRuntimeV7.from_checkpoint(payload)
    assert rebuilt.get_extra_state()==new.get_extra_state()
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in new.state_dict().items() if torch.is_tensor(v))
    with pytest.raises(ValueError,match='V7|V6.2|v62'):NativeFullRuntimeV7.from_checkpoint(saved)
    with pytest.raises(ValueError):NativeFullRuntimeV62.from_checkpoint(payload)
    malformed=dict(payload,runtime_config=dict(payload['runtime_config'],contact_bridge='changed'))
    with pytest.raises(ValueError,match='header|schema'):NativeFullRuntimeV7.from_checkpoint(malformed)

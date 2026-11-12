"""Real tiny Qwen blocks; inactive encoder construction is unnecessary here."""
from dataclasses import replace
from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tests.test_native_v6_integration import case
from tests.test_native_v61_poselink import spatial_for


@pytest.fixture(scope='module')
def installed():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_v5.strength import NativeStrengthController
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    from src.pose_control.qwen21_full.native_v61.integration import install_native_full_v61
    torch.manual_seed(6101)
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    config = NativeConfigV5(verification_only=True, hidden_dim=32, num_layers=6, injection_layers=tuple(range(6)))
    # Exact modules consumed by the installer; checkpoint tests construct full encoders.
    runtime = NativeFullRuntimeV61.__new__(NativeFullRuntimeV61); nn.Module.__init__(runtime)
    runtime.config = config; runtime.base = nn.Module()
    runtime.base.body = Qwen21NativeBody(NativeBodyConfig(hidden_dim=32, num_layers=6,
                                                        injection_layers=tuple(range(6)))).requires_grad_(False)
    runtime.face_controller = nn.Identity(); runtime.hand_controller = nn.Identity()
    runtime.interaction_head = nn.Identity(); runtime.strength = NativeStrengthController(config).requires_grad_(False)
    runtime.pose_config = PoseLinkConfigV61(hidden_dim=32, width=32, heads=4, injection_layers=tuple(range(6)),
        max_pose_tokens_per_reference=16, global_token_budget=8, joint_token_budget=4, local_token_budget=4,
        query_chunk_size=3)
    runtime.poselink = PoseLinkAttentionV61(runtime.pose_config)
    bare = QwenImage21Transformer2DModel(num_layers=6, attention_head_dim=16, num_attention_heads=2,
                                       context_in_dim=32, axes_dims_rope=(2, 6, 8))
    model = deepcopy(bare)
    install_native_full_v61(model, runtime)
    yield bare.eval(), model.eval(), runtime
    torch.set_num_threads(threads)


def control_case(roles):
    from src.pose_control.qwen21_full.native_v61.prepared import PreparedNativeFullControlV61
    old, options, values = case(roles)
    spatial = spatial_for(old.pose)
    active = old.pose.person_valid
    spatial.key_support.mul_(active[:, :, None, None])
    spatial.joint_valid.logical_and_(active[:, :, None])
    spatial.local_valid.logical_and_(active[:, :, None])
    spatial.validate(old.pose)
    control = PreparedNativeFullControlV61(old.core, old.relation, old.relation_valid,
        old.local, old.metadata, old.pose, spatial).validate()
    return control, options, values


@pytest.mark.parametrize('roles', [
    ('source', 'target_pose'),
    ('source', 'combined_pose', 'person_0_pose', 'person_1_pose'),
    ('source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose')])
def test_zero_heads_preserve_real_qwen_for_all_reference_modes(installed, roles):
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    bare, model, _ = installed
    control, options, inputs = control_case(roles)
    with torch.no_grad():
        expected = bare(**inputs)[0]
        with bind_native_full_v61(model, control, options=options, valid_mask=control.pose.coverage,
                reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles)):
            actual = model(**inputs)[0]
    assert torch.equal(expected, actual)


def test_trained_poselink_preserves_reference_prefix_and_freezing(installed):
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    bare, model, runtime = installed
    roles = ('source', 'target_pose')
    control, options, inputs = control_case(roles)
    try:
        with torch.no_grad():
            for head in runtime.poselink.output_heads.values(): head.weight.normal_(0, .02)
            expected = bare(**inputs)[0]
            with bind_native_full_v61(model, control, options=options, valid_mask=control.pose.coverage,
                    reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles)):
                actual = model(**inputs)[0]
        assert torch.equal(expected[:, :-8], actual[:, :-8])
        assert not torch.equal(expected[:, -8:], actual[:, -8:])
        assert all(not p.requires_grad for n,p in model.named_parameters()
                   if not n.startswith(('native_body.', 'native_v61_')))
    finally:
        with torch.no_grad():
            for head in runtime.poselink.output_heads.values(): head.weight.zero_()


def test_two_forwards_preserve_spatial_context_in_checkpoint_backward(installed):
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    _, model, runtime = installed
    model.gradient_checkpointing = True
    model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    runtime.poselink.train()
    runtime.poselink.zero_grad(set_to_none=True)
    try:
        losses = []
        for roles in (('source', 'target_pose'), ('source', 'combined_pose', 'person_0_pose', 'person_1_pose')):
            control, options, inputs = control_case(roles)
            with bind_native_full_v61(model, control, options=options, valid_mask=control.pose.coverage,
                    reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles)):
                losses.append(model(**inputs)[0][:, -8:].square().mean())
        sum(losses).backward()
        assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0
                   for head in runtime.poselink.output_heads.values())
        assert model._native_v61_execution.get() is None
        assert model._native_v61_binding.get() is None
    finally:
        model.gradient_checkpointing = False
        runtime.poselink.zero_grad(set_to_none=True); runtime.poselink.eval()

"""Real tiny Qwen21 blocks on CPU; no pretrained models or experiment data."""
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint


@pytest.fixture(scope='module')
def installed():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody, NativeBodyConfig
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.integration import install_native_full_v5
    from src.pose_control.qwen21_full.native_v6.contracts import PoseLinkConfig
    from src.pose_control.qwen21_full.native_v6.runtime import NativeFullRuntimeV6
    from src.pose_control.qwen21_full.native_v6.integration import install_native_full_v6
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(166)
    config = NativeConfigV5(verification_only=True, hidden_dim=32, num_layers=6, injection_layers=tuple(range(6)))
    body = Qwen21NativeBody(NativeBodyConfig(hidden_dim=32, num_layers=6, injection_layers=tuple(range(6))))
    old = NativeFullRuntimeV5(NativeDualRuntime(body=body), config, profile='frozen')
    with torch.no_grad():
        for head in old.body.output_heads:
            head.weight.normal_(0, .005)
    new = NativeFullRuntimeV6.from_v5_checkpoint(old.checkpoint_payload(metadata={}),
        pose_config=PoseLinkConfig(hidden_dim=32, width=32, heads=4, injection_layers=tuple(range(6)), query_chunk_size=3))
    new.set_trainability('poselink-full')
    bare = QwenImage21Transformer2DModel(num_layers=6, attention_head_dim=16, num_attention_heads=2,
        context_in_dim=32, axes_dims_rope=(2, 6, 8))
    legacy = deepcopy(bare)
    modern = deepcopy(bare)
    install_native_full_v5(legacy, old)
    install_native_full_v6(modern, new)
    yield legacy.eval(), modern.eval(), old, new
    torch.set_num_threads(previous_threads)


def case(roles, *, value=1.):
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    from src.pose_control.qwen21_full.native_local_condition import PreparedNativeFullControlV4
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5
    from src.pose_control.qwen21_full.native_v6.prepared import PreparedNativeFullControlV6
    from src.pose_control.qwen21_full.native_v6.contracts import PoseGeometry
    dual = len(roles) > 2
    people = torch.tensor([[True, dual]])
    options = NativeControlOptionsV5(face=False, hand=False, detail=False, interaction=False)
    geometry = torch.full((1, 8, 512), value)
    relation = torch.zeros_like(geometry)
    features = NativeGeometryFeatures(geometry, relation, (2, 4), people)
    metadata = ({'options': asdict(options)},)
    core = PreparedNativeFullControlV4(features, None, metadata)
    owner = torch.zeros(1, 8, 2)
    owner[:, :, 0] = 1
    if dual:
        owner[:, 4:, 0] = 0
        owner[:, 4:, 1] = 1
    pose = PoseGeometry(geometry, geometry[:, None].expand(-1, 2, -1, -1).clone(), owner, owner.clone(),
        torch.ones(1, 8), torch.zeros_like(owner), (2, 4), people, ('canvas',), relation)
    control = PreparedNativeFullControlV6(core, relation, torch.zeros(1, 8, dtype=torch.bool), None, metadata, pose).validate()
    # Adjacent image slots still belong to different images; text is interleaved.
    slots = [False] + [True, True] * len(roles) + [False] + [True, True]
    inputs = dict(hidden_states=torch.randn(1, 8 * (len(roles) + 1), 64), timestep=torch.tensor([.5]),
        encoder_hidden_states=torch.randn(1, len(slots) - 2, 32),
        encoder_hidden_states_mask=torch.ones(1, len(slots) - 2, dtype=torch.bool),
        img_mask=torch.tensor([slots]), img_shapes=[[(1, 2, 4)] * (len(roles) + 1)], return_dict=False)
    return control, options, inputs


@pytest.mark.parametrize('roles', [
    ('source', 'target_pose'),
    ('source', 'combined_pose', 'person_0_pose', 'person_1_pose'),
    ('source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose')])
def test_zero_poselink_matches_v5_on_real_qwen(installed, roles):
    from src.pose_control.qwen21_full.native_v5.contracts import PreparedNativeFullControlV5
    from src.pose_control.qwen21_full.native_v5.integration import bind_native_full_v5
    from src.pose_control.qwen21_full.native_v6.integration import bind_native_full_v6
    legacy, model, _, new = installed
    control, options, values = case(roles)
    old_control = PreparedNativeFullControlV5(control.core, control.relation, control.relation_valid, None, control.metadata)
    with torch.no_grad(), bind_native_full_v5(legacy, old_control, options=options, valid_mask=control.pose.coverage):
        expected = legacy(**values)[0]
    with torch.no_grad(), bind_native_full_v6(model, control, options=options, valid_mask=control.pose.coverage, reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles)):
        actual = model(**values)[0]
    assert torch.equal(expected, actual)
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith(('native_body.', 'native_v6_')))


def test_nonzero_poselink_changes_only_target_prefix_is_exact(installed):
    from src.pose_control.qwen21_full.native_v6.integration import bind_native_full_v6
    model, runtime = installed[1], installed[3]
    roles = ('source', 'target_pose')
    control, options, values = case(roles)
    try:
        with torch.no_grad():
            for head in runtime.poselink.output_heads.values():
                head.weight.normal_(0, .02)
        with bind_native_full_v6(model, control, options=options, valid_mask=control.pose.coverage, reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles), poselink_strength=0.):
            baseline = model(**values)[0]
        with bind_native_full_v6(model, control, options=options, valid_mask=control.pose.coverage, reference_roles=roles, reference_canvas_ids=('canvas',)*len(roles)):
            result = model(**values)[0]
        assert torch.equal(baseline[:, :-8], result[:, :-8])
        assert not torch.equal(baseline[:, -8:], result[:, -8:])
    finally:
        with torch.no_grad():
            for head in runtime.poselink.output_heads.values():
                head.weight.zero_()


def test_nonreentrant_checkpoint_retains_each_forward_context(installed):
    from src.pose_control.qwen21_full.native_v6.integration import bind_native_full_v6
    model, runtime = installed[1], installed[3]
    model.gradient_checkpointing = True
    model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    runtime.zero_grad(set_to_none=True)
    try:
        results = []
        for amount in (1., 2.):
            control, options, values = case(('source', 'target_pose'), value=amount)
            with bind_native_full_v6(model, control, options=options, valid_mask=control.pose.coverage,
                                     reference_roles=('source', 'target_pose'), reference_canvas_ids=('canvas','canvas')):
                results.append(model(**values)[0][:, -8:].square().mean())
        sum(results).backward()
        assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0
                   for head in runtime.poselink.output_heads.values())
        assert model._native_v6_execution.get() is None
        assert model._native_v6_binding.get() is None
    finally:
        model.gradient_checkpointing = False
        runtime.zero_grad(set_to_none=True)


def test_same_grid_foreign_pose_canvas_is_rejected(installed):
    from src.pose_control.qwen21_full.native_v6.integration import bind_native_full_v6
    model = installed[1]
    roles = ('source', 'target_pose')
    control, options, values = case(roles)
    with pytest.raises(ValueError, match='Canvas'):
        with bind_native_full_v6(model, control, options=options, valid_mask=control.pose.coverage,
                                reference_roles=roles, reference_canvas_ids=('source', 'foreign')):
            model(**values)

from dataclasses import replace

import pytest
import torch

from tests.test_contact_hand_adapter import condition
from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter, HandMesh


@pytest.fixture(autouse=True, scope="module")
def small_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def adapter(**kwargs):
    return ContactHandAdapter(AdapterConfig(width=32, heads=4, intra_layers=1, inter_layers=1,
                                            roi_size=32, **kwargs))


def with_3d():
    sample = condition()
    valid = sample.visible.clone()
    visible = valid.clone()
    visible[1, 4] = False
    xyz = torch.full((4, 21, 3), float("nan"))
    xyz[valid] = torch.stack((torch.arange(42) % 4 * .007, torch.arange(42) % 5 * .006,
                              torch.arange(42) % 3 * .003 + .55), -1)
    mano_valid = torch.tensor([False, True, False, False])
    return replace(sample, visible=visible, joint_2d_valid=valid, joint_3d_valid=valid,
                   joint_occluded=valid & ~visible, hand_present=valid.any(-1), joints_3d=xyz,
                   mano_shape=torch.zeros(4, 10),
                   mano_wrist_rotation_6d=torch.tensor([[1, 0, 0, 0, 1, 0.]]).expand(4, -1),
                   mano_palm_scale=torch.full((4,), .075), mano_valid=mano_valid)


def test_occluded_joints_keep_3d_anatomy_and_receive_less_spatial_context():
    sample = with_3d().validate()
    assert sample.projected_joints()[1, 4] and sample.anatomy_joints()[1, 4]
    assert not sample.active_joints()[1, 4]
    assert sample.spatial_joint_confidence()[1, 4] < sample.spatial_joint_confidence()[1, 3]
    model = adapter()
    assert model.joint_graph.input.in_features == 19
    before = model.prepare(sample)
    assert before.tokens.shape[0] == 57  # Two hand hierarchies plus one MANO token.
    assert torch.isfinite(before.tokens).all()
    changed = sample.joints_3d.clone()
    changed[1, 4, 2] += .02
    after = model.prepare(replace(sample, joints_3d=changed))
    assert not torch.equal(before.tokens, after.tokens)
    torch.testing.assert_close(before.support, after.support)
    with pytest.raises(ValueError, match="occlu"):
        replace(sample, joint_occluded=sample.joint_2d_valid).validate()


def test_joint_sampling_and_identity_have_independent_spatial_and_side_paths():
    model = adapter()
    spatial_calls = []
    handle = model.spatial_encoder.token_sampler.register_forward_hook(lambda _, args, out: spatial_calls.append(out))
    memory = model.prepare(condition())
    handle.remove()
    assert spatial_calls
    samples = spatial_calls[0]
    assert samples.shape[-1] == 32 and not torch.equal(samples[0], samples[-1])
    assert model.person_embedding.num_embeddings == model.side_embedding.num_embeddings == 2
    assert memory.anchor_strength is not None


def test_low_quality_surface_candidate_keeps_unknown_state():
    sample = with_3d()
    mesh = HandMesh(torch.rand(4, 778, 3) * .1, torch.tensor([[0, 1], [1, 2]]),
                    torch.arange(128).reshape(64, 2), torch.ones(4, 64, 2) * 16,
                    sample.hand_present, (None, "common", None, "common"))
    pair = replace(sample.pairs[0], surface_indices=torch.tensor([[2, 4], [3, 5]]),
                   surface_features=torch.tensor([[.005, -.9, -1., 1.], [.005, -.9, -1., 1.]]),
                   surface_quality=torch.tensor([[.5, .5, .5, .5], [0., 1., 1., 1.]]))
    graph = adapter().prepare(replace(sample, mesh=mesh, pairs=(pair,))).graph
    assert graph.surface_features[:, 2].eq(-1).all()
    expected = .5 ** 4 * torch.exp(torch.tensor(-.5 * (.005 / .008) ** 2))
    torch.testing.assert_close(graph.surface_probability, torch.tensor([expected, 0., expected, 0.]))


def test_dynamic_peers_are_updated_with_gradients():
    model = adapter()
    memory = model.prepare(condition())
    before = memory.tokens.detach()
    routing = (memory.support[memory.slots][:, memory.roi].T > 0)
    bias = torch.zeros_like(routing, dtype=torch.float).masked_fill(~routing, -torch.inf)
    hidden = torch.randn(len(memory.roi), model.config.base_width)
    after = model.feedback(before, memory, hidden, torch.tensor([.4, .0]), bias)
    after.square().mean().backward()
    assert model.feedback.peer_attention.query.weight.grad.abs().sum() > 0
    assert torch.equal(memory.tokens.detach(), before)


def test_tiny_contact_confidence_does_not_join_distant_hands():
    sample = condition(hw=(64, 256))
    sample.joints[3, :, 0] += 180
    sample.support[3] = sample.support[3].roll(180, dims=1)
    pair = replace(sample.pairs[0], part_confidence=torch.full((16, 16), 1e-5))
    memory = adapter().prepare(replace(sample, pairs=(pair,)))
    assert not memory.graph.neighbors[1, 3]


def test_old_mesh_pack_can_keep_fitted_but_invisible_hands():
    sample = condition()
    sample.visible[1] = False
    mesh = HandMesh(torch.zeros(4, 778, 3), torch.tensor([[0, 1], [1, 2]]),
                    torch.arange(128).reshape(64, 2), torch.ones(4, 64, 2) * 16,
                    torch.tensor([False, True, False, False]))
    old = replace(sample, mesh=mesh)
    memory = adapter(architecture_version=2).prepare(old)
    assert memory.slots.unique().tolist() == [3]

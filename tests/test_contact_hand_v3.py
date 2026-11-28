from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from src.pose_control.contact_hand_adapter import ContactHandAdapter, bind_hand_condition, install_hand_adapter
from tests.test_contact_hand_adapter import condition, fixture_model


@pytest.fixture(autouse=True, scope="module")
def small_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def v3(**overrides):
    base, adapter, inputs = fixture_model()
    return base, ContactHandAdapter(replace(adapter.config, architecture_version=3, **overrides)), inputs


def test_overlapping_instances_are_query_selected_without_losing_slot_memory():
    from src.pose_control.contact_hand_adapter.routing import InstanceSpatialAttention

    router = InstanceSpatialAttention(2, 2)
    with torch.no_grad():
        for projection in (router.query, router.key, router.value):
            projection.weight.copy_(torch.eye(2))
        router.metadata.weight.zero_()
    instances = torch.tensor([[[2., 0.], [0., 2.], [20., 20.], [0., 0.]]]).expand(2, -1, -1)
    query = torch.tensor([[5., 0.], [0., 5.]])
    support = torch.tensor([[.5, .5, 0., 0.]]).expand(2, -1)
    selected, weights = router(query, instances, support, torch.zeros(2, 4, 5))
    assert weights[0, 0] > .99 and weights[1, 1] > .99
    assert not weights[:, 2:].any()
    assert selected[0, 0] > selected[0, 1] and selected[1, 1] > selected[1, 0]


def test_v3_retains_spatial_instances_and_contact_types():
    _, model, _ = v3()
    sample = condition()
    sample.support[3, 8:24, 20:28] = 1
    memory = model.prepare(sample)
    assert memory.spatial.shape == (len(memory.roi), 4, 32)
    assert memory.spatial_metadata.shape == (len(memory.roi), 4, 5)
    assert (memory.support[:, memory.roi] > 0).sum(0).max() == 2
    assert (memory.token_types == 3).sum() == 2
    assert (memory.support.sum(0) <= 1).all()


def test_depth_bias_is_only_comparable_with_shared_camera_evidence():
    from src.pose_control.contact_hand_adapter import HandMesh
    from src.pose_control.contact_hand_adapter.routing import spatial_evidence

    sample = condition()
    mask = torch.zeros(4, 5, 64, 96, dtype=torch.bool)
    mask[[1, 3], 0, 1:8, 1:8] = True
    data = torch.zeros(4, 5, 64, 96)
    data[1, 0] = .3
    data[3, 0] = .5
    mesh = HandMesh(torch.zeros(4, 778, 3), torch.tensor([[0], [1]]),
                    torch.zeros(64, 1, dtype=torch.long), torch.zeros(4, 64, 2),
                    torch.tensor([False, True, False, True]), (None, "shared", None, "shared"))
    graph = v3()[1].prepare(sample).graph
    roi = torch.tensor([0])
    shared = replace(sample, geometry_maps=data, map_valid=mask, mesh=mesh).validate()
    evidence = spatial_evidence(shared, graph, roi)
    assert evidence[0, 1, 1] > evidence[0, 3, 1]
    unaligned = replace(shared, mesh=replace(mesh, camera_ids=(None, "front", None, "rear")))
    assert not spatial_evidence(unaligned, graph, roi)[..., 1].any()


def test_renderer_only_present_hand_has_local_and_prior_support():
    from src.pose_control.contact_hand_adapter.integration import _prior_support
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan

    sample = condition()
    visible = torch.zeros_like(sample.visible)
    mask = torch.zeros(4, 5, 64, 96, dtype=torch.bool)
    mask[1, 4, 12:23, 30:41] = True
    dense = torch.zeros(4, 5, 64, 96)
    sample = replace(sample, visible=visible, support=None, pairs=(), geometry_maps=dense,
                     map_valid=mask, hand_present=torch.tensor([False, True, False, False]))
    memory = v3()[1].prepare(sample)
    assert memory.roi.numel() and _prior_support(sample).any()
    assert make_local_plan(sample, size=64).slots == (1,)


def test_part_ablation_excludes_surface_evidence_from_peer_routing():
    from src.pose_control.contact_hand_adapter import HandMesh
    from src.pose_control.contact_hand_adapter.geometry import build_contact_graph

    sample = condition(hw=(64, 256))
    sample.joints[3, :, 0] += 180
    sample.support[3] = sample.support[3].roll(180, dims=1)
    mesh = HandMesh(torch.zeros(4, 778, 3), torch.tensor([[0], [1]]), torch.zeros(64, 1, dtype=torch.long),
                    torch.zeros(4, 64, 2), torch.tensor([False, True, False, True]), (None, "a", None, "a"))
    pair = replace(sample.pairs[0], part_state=torch.full((16, 16), -1, dtype=torch.int8),
                   surface_indices=torch.tensor([[0, 0]]), surface_features=torch.tensor([[.002, -1., -1., .8]]))
    sample = replace(sample, mesh=mesh, pairs=(pair,)).validate()
    assert build_contact_graph(sample, sample.support).neighbors[1, 3]
    assert not build_contact_graph(sample, sample.support, contact_level="part").neighbors[1, 3]


def test_small_local_canvas_keeps_rounded_reference_grids_above_vlm_minimum():
    from types import SimpleNamespace
    from PIL import Image
    from src.pose_control.contact_hand_adapter.runtime import qwen_reference_resolution, qwen_reference_size

    processor = SimpleNamespace(size=SimpleNamespace(shortest_edge=65536, longest_edge=16777216))
    pipe = SimpleNamespace(processor=SimpleNamespace(image_processor=processor))
    references = [Image.new("RGB", (192, 320)), Image.new("RGB", (256, 256))]
    assert qwen_reference_resolution(pipe, references, (256, 256)) == 288
    assert qwen_reference_resolution(pipe, references, (512, 352)) == 512
    for image in references:
        width, height = qwen_reference_size(image, 288)
        assert width * height >= processor.size.shortest_edge


def test_gate_initial_order_and_update_gradient():
    from src.pose_control.contact_hand_adapter.routing import TypedDenoiseGate

    module = TypedDenoiseGate(8, 3)
    hidden = torch.zeros(2, 8)
    early = module(hidden, hidden, torch.ones(4), torch.tensor([.95]), 0)
    late = module(hidden, hidden, torch.ones(4), torch.tensor([.05]), 2)
    assert early[0, 0] > early[0, 4] and late[0, 4] > late[0, 0]
    gate = module(hidden, hidden, torch.ones(4), torch.tensor([.45]), 1)
    gate.square().mean().backward()
    assert module.phase_logits.grad is not None and module.phase_logits.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ("compressed", "low_rank", "direct"))
def test_all_residual_arms_are_identity_at_initialization_and_train_on_qwen(mode):
    base, model, inputs = v3(residual_mode=mode)
    native = deepcopy(base)
    install_hand_adapter(base, model)
    with bind_hand_condition(base, condition()):
        actual = base(**inputs)[0]
    assert torch.equal(native(**inputs)[0], actual)
    actual.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.output.parameters())
    assert all(p.grad is None for p in native.parameters())

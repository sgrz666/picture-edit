"""CPU contracts for V6 PoseLink, independent of diffusers and model weights."""
from dataclasses import replace
import math

import pytest
import torch

from src.pose_control.qwen21_full.native_v6.contracts import PoseGeometry, PoseLinkConfig
from src.pose_control.qwen21_full.native_v6.pose_layout import build_pose_layout
from src.pose_control.qwen21_full.native_v6.poselink import PoseLinkAttention


class MetadataModel:
    """The native Qwen metadata algorithm, including adjacent image boundaries."""
    @staticmethod
    def build_token_metadata(mask, shapes):
        indices = mask.nonzero().flatten()
        lengths = torch.tensor([math.prod(s) for s in shapes], device=mask.device)
        if int(lengths.sum()) != len(indices):
            raise ValueError("image lengths differ")
        ids = torch.full_like(mask, -1, dtype=torch.long)
        ids[indices] = torch.repeat_interleave(torch.arange(len(shapes)), lengths)
        target = ids == len(shapes) - 1
        return ids, target


ROLES = {
    2: ("source", "target_pose"),
    4: ("source", "combined_pose", "person_0_pose", "person_1_pose"),
    5: ("source_a", "source_b", "person_0_pose", "person_1_pose", "combined_pose"),
}


def layout_for(count=4, shapes=None):
    shapes = shapes or [(1, 2, 2)] * (count + 1)
    slots = [False]
    for i, shape in enumerate(shapes):
        slots.extend([True] * (math.prod(shape) // 4))
        if i < count and i % 2 == 0:
            slots.extend([False, False])
    return build_pose_layout(MetadataModel(), torch.tensor([slots]), [shapes], ROLES[count])


def geometry_for(hw=(2, 2), batch=1):
    n = math.prod(hw)
    owner = torch.zeros(batch, n, 2)
    owner[:, : max(1, n // 2), 0] = 1
    owner[:, max(1, n // 2):, 1] = 1
    return PoseGeometry(
        torch.randn(batch, n, 512), torch.randn(batch, 2, n, 512),
        owner, owner.clone(), torch.ones(batch, n), torch.zeros(batch, n, 2),
        hw, torch.ones(batch, 2, dtype=torch.bool), tuple(f"canvas-{i}" for i in range(batch)),
        interaction=torch.randn(batch, n, 512),
    )


def config(**kwargs):
    return PoseLinkConfig(hidden_dim=16, width=8, heads=2, injection_layers=(0, 2),
                          query_chunk_size=2, max_pose_tokens_per_reference=4, **kwargs)


def inputs(layout=None, geometry=None):
    layout = layout or layout_for()
    geometry = geometry or geometry_for(layout.target_token_hw)
    joint = torch.randn(geometry.geometry.shape[0], layout.joint_length, 16)
    target = joint[:, layout.target_indices].clone()
    return target, joint, layout, geometry


def activate(model):
    with torch.no_grad():
        for head in model.output_heads.values():
            head.weight.normal_(0, .2)


def test_config_roundtrip_and_validation():
    value = config()
    assert PoseLinkConfig.from_dict(value.to_dict()) == value
    for kwargs in ({"heads": 3}, {"query_chunk_size": 0}, {"max_pose_tokens_per_reference": 0},
                   {"injection_layers": (0, 0)}, {"use_contact": 1}):
        with pytest.raises(ValueError):
            PoseLinkConfig(**kwargs)


def test_geometry_batch_transforms_preserve_discrete_types_and_gradients():
    first = geometry_for()
    first.geometry.requires_grad_()
    selected = PoseGeometry.stack([first, first]).to(dtype=torch.float64).index_select(torch.tensor([1, 0]))
    assert selected.validate() is selected
    assert selected.person_valid.dtype == torch.bool
    assert selected.geometry.dtype == torch.float64
    assert selected.canvas_ids == ("canvas-0", "canvas-0")
    selected.geometry.sum().backward()
    assert torch.equal(first.geometry.grad, torch.full_like(first.geometry, 2))
    with pytest.raises(ValueError, match="coverage"):
        replace(first, coverage=torch.full_like(first.coverage, 1.1)).validate()
    with pytest.raises(ValueError):
        PoseGeometry.stack([first, geometry_for((1, 4))])


@pytest.mark.parametrize("count", [2, 4, 5])
def test_layout_selects_actual_image_ids_excluding_interleaved_text_and_sources(count):
    layout = layout_for(count)
    chosen = torch.cat([ref.indices for ref in layout.references])
    expected = {i for i, role in enumerate(ROLES[count]) if "pose" in role}
    assert set(layout.image_ids[chosen].tolist()) == expected
    assert torch.equal(layout.target_mask.nonzero().flatten(), layout.target_indices)
    for ref in layout.references:
        assert torch.equal(ref.indices, (layout.image_ids == ref.image_id).nonzero().flatten())
        assert ref.indices.numel() == math.prod(ref.token_hw)
    # At least two images are adjacent: each still has a distinct ID.
    assert any(a.indices[-1] + 1 == b.indices[0] for a, b in zip(layout.references, layout.references[1:])) or count == 2


def test_layout_rejects_mislabeled_roles_and_shape_disagreement():
    with pytest.raises(ValueError, match="roles"):
        build_pose_layout(MetadataModel(), torch.ones(1, 3, dtype=torch.bool),
                          [[(1, 2, 2)] * 3], ("source", "person_0_pose"))
    with pytest.raises(ValueError):
        build_pose_layout(MetadataModel(), torch.ones(1, 2, dtype=torch.bool),
                          [[(1, 2, 2)] * 3], ROLES[2])


def test_forward_rejects_forged_source_key_indices_and_mismatched_reference_grid():
    model = PoseLinkAttention(config())
    target, joint, layout, geo = inputs()
    ref = layout.references[0]
    source = (layout.image_ids == 0).nonzero().flatten()
    for corrupted in (replace(ref, indices=source), replace(ref, token_hw=(1, 4))):
        forged = replace(layout, references=(corrupted,) + layout.references[1:])
        with pytest.raises(ValueError):
            model(target, joint, forged, geo, layer=0, sigma=torch.tensor([.4]))


def test_token_budget_and_query_chunks_bound_each_attention_matrix(monkeypatch):
    shapes = [(1, 2, 2), (1, 6, 8), (1, 4, 8), (1, 8, 8), (1, 4, 6)]
    layout = layout_for(4, shapes)
    model = PoseLinkAttention(config()); activate(model)
    target, joint, layout, geo = inputs(layout)
    seen = []
    original = model._attend_chunk
    def observed(query, key, *args):
        seen.append((query.shape[2], key.shape[2]))
        return original(query, key, *args)
    monkeypatch.setattr(model, "_attend_chunk", observed)
    model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5])).sum().backward()
    assert seen
    assert all(q <= 2 and k <= len(layout.references) * 4 for q, k in seen)


def test_mixed_batch_validity_and_checkpoint_recomputation_survive_other_forward():
    torch.manual_seed(101)
    model = PoseLinkAttention(config()); activate(model)
    geo = geometry_for(batch=2)
    geo.person_valid[1] = False
    target, joint, layout, geo = inputs(geometry=geo)
    joint.requires_grad_(); target.requires_grad_()
    first = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5, .5]))
    expected = first.detach().clone()
    model(target, joint, layout, geo, layer=2, sigma=torch.tensor([.9, .9]))
    first.square().sum().backward()
    assert torch.equal(first.detach(), expected)
    assert first[0].abs().sum() > 0 and not first[1].any()
    assert not target.grad[1].any() and not joint.grad[1].any()


def test_zero_head_identity_and_first_backward_then_full_upstream_gradients():
    torch.manual_seed(41)
    model = PoseLinkAttention(config())
    target, joint, layout, geo = inputs()
    target.requires_grad_(); joint.requires_grad_(); geo.geometry.requires_grad_(); geo.person_tokens.requires_grad_()
    first = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5]))
    assert torch.count_nonzero(first) == 0
    first.square().sum().add(first.sum()).backward()
    assert model.output_heads["0"].weight.grad.abs().sum() > 0
    assert model.query_projection.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    activate(model)
    model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5])).square().sum().backward()
    for value in (target.grad, joint.grad[:, torch.cat([r.indices for r in layout.references])],
                  geo.geometry.grad, geo.person_tokens.grad, model.query_projection.weight.grad):
        assert value.abs().sum() > 0
    source = (layout.image_ids == 0) | (layout.image_ids == -1)
    assert torch.count_nonzero(joint.grad[:, source]) == 0


def test_all_invalid_keys_stay_finite_and_zero_even_after_head_training():
    model = PoseLinkAttention(config()); activate(model)
    target, joint, layout, geo = inputs()
    with torch.no_grad():
        model.output_heads["0"].bias.fill_(4)
    geo = replace(geo, person_valid=torch.zeros_like(geo.person_valid))
    residual = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.8]))
    assert torch.isfinite(residual).all() and torch.count_nonzero(residual) == 0
    residual.sum().backward()


def test_rectangular_alignment_pooling_and_padding_do_not_contaminate_output():
    torch.manual_seed(7)
    shapes = [(1, 2, 2), (1, 4, 6), (1, 2, 6)]
    layout = layout_for(2, shapes)
    geo = geometry_for((2, 6))
    geo.coverage[:, :,] = torch.tensor([[1, 1, 1, 0, 0, 0] * 2])
    model = PoseLinkAttention(config()); activate(model)
    target, joint, layout, geo = inputs(layout, geo)
    expected = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.4]))
    poisoned = joint.clone()
    ref = layout.references[0]
    invalid_ref = (torch.arange(24).reshape(4, 6)[:, 3:]).flatten()
    poisoned[:, ref.indices[invalid_ref]] = 1e12
    poisoned_geo = replace(geo, geometry=geo.geometry.clone(), person_tokens=geo.person_tokens.clone(),
                           interaction=geo.interaction.clone())
    for field in (poisoned_geo.geometry, poisoned_geo.interaction):
        field[:, geo.coverage[0] == 0] = 1e12
    poisoned_geo.person_tokens[:, :, geo.coverage[0] == 0] = 1e12
    actual = model(target, poisoned, layout, poisoned_geo, layer=0, sigma=torch.tensor([.4]))
    assert torch.equal(actual[:, geo.coverage[0] == 0], torch.zeros(1, 6, 16))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_strength_and_chunking_are_independent_and_deterministic():
    model = PoseLinkAttention(config()); activate(model)
    target, joint, layout, geo = inputs()
    one = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.3]))
    twice = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.3]), strength=2.)
    assert torch.allclose(twice, one * 2)
    assert not model(target, joint, layout, geo, layer=2, sigma=torch.tensor([.3]), strength=0).any()
    with pytest.raises(ValueError):
        model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.3]), strength=-1)
    larger = PoseLinkAttention(replace(config(), query_chunk_size=100))
    larger.load_state_dict(model.state_dict())
    assert torch.allclose(larger(target, joint, layout, geo, layer=0, sigma=torch.tensor([.3])), one, atol=1e-6)


@pytest.mark.parametrize("flag", ["use_geometry", "use_spatial", "use_role", "use_interaction"])
def test_feature_flags_change_only_their_requested_term(flag):
    torch.manual_seed(61)
    model = PoseLinkAttention(config()); activate(model)
    target, joint, layout, geo = inputs()
    disabled = PoseLinkAttention(replace(config(), **{flag: False}))
    disabled.load_state_dict(model.state_dict())
    full = model(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5]))
    ablated = disabled(target, joint, layout, geo, layer=0, sigma=torch.tensor([.5]))
    assert not torch.allclose(full, ablated, atol=1e-6)
    if flag in ("use_geometry", "use_interaction"):
        changed = replace(geo, **({"geometry": torch.randn_like(geo.geometry),
                                   "person_tokens": torch.randn_like(geo.person_tokens)} if flag == "use_geometry"
                                  else {"interaction": torch.randn_like(geo.interaction)}))
        assert torch.equal(ablated, disabled(target, joint, layout, changed, layer=0, sigma=torch.tensor([.5])))


def test_contact_relaxes_cross_person_penalty_only_at_both_local_endpoints():
    model = PoseLinkAttention(config())
    xy = torch.tensor([[[0., 0.], [1., 1.]]])
    owner = torch.tensor([[[1., 0.], [0., 1.]]])
    absent = torch.zeros_like(owner)
    local = owner.clone()
    base = model._attention_bias(xy, xy, owner, owner, absent, absent)
    one_side = model._attention_bias(xy, xy, owner, owner, local, absent)
    both = model._attention_bias(xy, xy, owner, owner, local, local)
    assert torch.equal(base, one_side)
    assert (both[:, :, 0, 1] > base[:, :, 0, 1]).all()
    assert torch.equal(both[:, :, 0, 0], base[:, :, 0, 0])
    disabled = PoseLinkAttention(replace(config(), use_contact=False))
    disabled.load_state_dict(model.state_dict())
    assert torch.equal(base, disabled._attention_bias(xy, xy, owner, owner, local, local))

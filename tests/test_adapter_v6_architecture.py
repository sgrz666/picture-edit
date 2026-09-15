from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.pose_control.v6.conditions import TaskSpec, UnifiedAdapterCondition
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.experts import DualPersonExpert, TaskRouter
from src.pose_control.v6.geometry_encoder import SharedGeometryEncoder
from src.pose_control.v6.token_encoders import (
    AppearanceTokenEncoder,
    GlobalGeometryTokenEncoder,
)


def make_condition(
    *,
    batch_size: int = 1,
    num_people: int = 1,
    image_size: int = 64,
    latent_size: int = 8,
    interaction_type: int = 0,
) -> UnifiedAdapterCondition:
    generator = torch.Generator().manual_seed(11)
    person_valid = torch.zeros(batch_size, 2, dtype=torch.bool)
    person_valid[:, :num_people] = True
    modality_valid = person_valid[..., None].expand(-1, -1, 5).clone()

    silhouette = torch.zeros(batch_size, 2, 1, image_size, image_size)
    silhouette[:, 0, :, :, : image_size // 2 + 4] = 1
    if num_people == 2:
        silhouette[:, 1, :, :, image_size // 2 - 4 :] = 1

    contact_pairs = torch.zeros(batch_size, 4, 4)
    contact_pairs[:, 0] = torch.tensor([0.0, 7.0, 1.0, 10.0])
    contact_valid = torch.zeros(batch_size, 4, dtype=torch.bool)
    if num_people == 2:
        contact_valid[:, 0] = True

    return UnifiedAdapterCondition(
        normal=torch.randn(batch_size, 2, 3, image_size, image_size, generator=generator),
        depth=torch.rand(batch_size, 2, 1, image_size, image_size, generator=generator),
        skeleton=torch.rand(batch_size, 2, 3, image_size, image_size, generator=generator),
        silhouette=silhouette,
        person_valid=person_valid,
        source_person_latents=torch.randn(
            batch_size, 2, 16, latent_size, latent_size, generator=generator
        ),
        smpl_pose6d=torch.randn(batch_size, 2, 52, 6, generator=generator),
        source_betas=torch.randn(batch_size, 2, 10, generator=generator),
        camera=torch.randn(batch_size, 2, 7, generator=generator),
        task_spec=TaskSpec(
            interaction_type=torch.full((batch_size,), interaction_type, dtype=torch.long),
            declared_num_people=torch.full((batch_size,), num_people, dtype=torch.long),
            prompts=tuple("two people shake hands" for _ in range(batch_size)),
        ),
        part_ids=torch.randint(
            0, 24, (batch_size, 2, image_size, image_size), generator=generator
        ),
        modality_valid=modality_valid,
        source_indices=torch.arange(2).expand(batch_size, -1).clone(),
        contact_maps=torch.rand(
            batch_size, 2, image_size, image_size, generator=generator
        ),
        contact_pairs=contact_pairs,
        contact_valid=contact_valid,
    )


def swap_people(condition: UnifiedAdapterCondition) -> UnifiedAdapterCondition:
    swapped = copy.deepcopy(condition)
    for name in (
        "normal",
        "depth",
        "skeleton",
        "silhouette",
        "person_valid",
        "source_person_latents",
        "smpl_pose6d",
        "source_betas",
        "camera",
        "part_ids",
        "modality_valid",
        "source_indices",
        "contact_maps",
    ):
        value = getattr(swapped, name)
        if value is not None:
            setattr(swapped, name, value[:, [1, 0]].clone())
    pairs = swapped.contact_pairs.clone()
    pairs[..., 0] = 1 - pairs[..., 0]
    pairs[..., 2] = 1 - pairs[..., 2]
    swapped.contact_pairs = pairs
    return swapped


def test_condition_contract_and_router_are_explicit() -> None:
    condition = make_condition(batch_size=2, num_people=1, interaction_type=4)
    condition.validate()

    route = TaskRouter().resolve(condition.person_valid, condition.task_spec)
    assert route.num_people.tolist() == [1, 1]
    assert route.single_mask.tolist() == [True, True]
    assert route.dual_mask.tolist() == [False, False]
    # Explicit metadata wins over the prompt text (which says handshake).
    assert route.interaction_ids.tolist() == [4, 4]

    bad = copy.deepcopy(condition)
    bad.person_valid[:] = False
    with pytest.raises(ValueError, match="one or two"):
        bad.validate()


def test_geometry_encoder_is_shared_multiscale_and_masks_missing_modalities() -> None:
    condition = make_condition(num_people=1)
    encoder = SharedGeometryEncoder(base_channels=8, output_channels=32, num_parts=24)
    encoder.eval()

    pyramid = encoder(condition)
    assert pyramid.level_256.shape == (1, 2, 8, 32, 32)
    assert pyramid.level_128.shape == (1, 2, 16, 16, 16)
    assert pyramid.level_64.shape == (1, 2, 32, 8, 8)
    assert torch.count_nonzero(pyramid.level_64[:, 1]) == 0

    hidden_normal = copy.deepcopy(condition)
    hidden_normal.modality_valid[..., 0] = False
    first = encoder(hidden_normal).level_64
    hidden_normal.normal = torch.randn_like(hidden_normal.normal) * 1000
    second = encoder(hidden_normal).level_64
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_geometry_and_appearance_encoders_emit_eight_tokens_per_person() -> None:
    condition = make_condition(num_people=2)
    geometry = GlobalGeometryTokenEncoder(token_dim=48)
    appearance = AppearanceTokenEncoder(in_channels=16, token_dim=48, token_count=8)

    geometry_tokens = geometry(condition)
    appearance_tokens = appearance(condition.source_person_latents, condition.person_valid)
    assert geometry_tokens.shape == (1, 2, 8, 48)
    assert appearance_tokens.shape == (1, 2, 8, 48)

    # The target condition deliberately exposes only source_betas; target_betas
    # must not become part of the API and accidentally change identity/shape.
    assert not hasattr(condition, "target_betas")


def test_dual_expert_is_swap_equivariant_and_contact_sensitive() -> None:
    torch.manual_seed(3)
    expert = DualPersonExpert(
        feature_channels=16,
        cross_attention_dim=32,
        token_dim=32,
        num_heads=4,
        contact_pair_dim=4,
        contact_token_count=8,
    ).eval()
    condition = make_condition(num_people=2, image_size=32)
    features = torch.randn(1, 2, 16, 8, 8)
    person_tokens = torch.randn(1, 2, 16, 32)

    output = expert(features, person_tokens, condition)
    swapped = swap_people(condition)
    swapped_output = expert(features[:, [1, 0]], person_tokens[:, [1, 0]], swapped)
    torch.testing.assert_close(
        output.person_features[:, [1, 0]],
        swapped_output.person_features,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(output.scene_feature, swapped_output.scene_feature, atol=1e-5, rtol=1e-5)

    without_contact = copy.deepcopy(condition)
    without_contact.contact_maps.zero_()
    without_contact.contact_valid.zero_()
    no_contact_output = expert(features, person_tokens, without_contact)
    assert not torch.allclose(output.contact_tokens, no_contact_output.contact_tokens)
    assert not torch.allclose(output.scene_feature, no_contact_output.scene_feature)


def test_single_route_ignores_person_b_and_contact_branches() -> None:
    torch.manual_seed(5)
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        context_input_dim=24,
        pooled_input_dim=20,
        num_heads=4,
    ).eval()
    first = make_condition(num_people=1, image_size=32)
    second = copy.deepcopy(first)
    second.normal[:, 1].normal_(mean=100, std=20)
    second.depth[:, 1].uniform_(100, 200)
    second.source_person_latents[:, 1].normal_(mean=-50, std=10)
    second.contact_maps.uniform_(100, 200)
    second.contact_pairs.uniform_(100, 200)
    second.contact_valid[:] = True

    prepared_first = model.prepare_conditioning(first)
    prepared_second = model.prepare_conditioning(second)
    torch.testing.assert_close(prepared_first.scene_feature, prepared_second.scene_feature)
    torch.testing.assert_close(prepared_first.adapter_tokens, prepared_second.adapter_tokens)
    assert torch.count_nonzero(prepared_first.contact_token_mask) == 0


def test_zero_heads_emit_six_target_only_residuals_and_alignment_is_exact() -> None:
    torch.manual_seed(7)
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        context_input_dim=24,
        pooled_input_dim=20,
        num_heads=4,
    ).eval()
    condition = make_condition(num_people=2, image_size=32, latent_size=8)
    target = torch.randn(1, 16, 8, 8)
    source_scene = torch.randn(1, 16, 8, 8)
    references = [[torch.randn(16, 8, 8), torch.randn(16, 4, 4)]]

    residuals = model(
        target_latents=target,
        condition=condition,
        source_scene_latents=source_scene,
        cond_hidden_states=references,
        encoder_hidden_states=torch.randn(1, 5, 24),
        pooled_projections=torch.randn(1, 20),
        timestep=torch.tensor([500]),
    )
    target_tokens = (8 // 2) ** 2
    full_tokens = target_tokens + (8 // 2) ** 2 + (4 // 2) ** 2
    assert len(residuals) == 6
    assert all(item.shape == (1, full_tokens, 32) for item in residuals)
    assert all(torch.count_nonzero(item) == 0 for item in residuals)
    assert all(torch.count_nonzero(head.weight) == 0 for head in model.control_core.zero_heads)
    assert all(torch.count_nonzero(head.bias) == 0 for head in model.control_core.zero_heads)

    with torch.no_grad():
        model.control_core.zero_heads[0].weight.fill_(0.01)
    residuals = model(
        target_latents=target,
        condition=condition,
        source_scene_latents=source_scene,
        cond_hidden_states=references,
        encoder_hidden_states=torch.randn(1, 5, 24),
        pooled_projections=torch.randn(1, 20),
        timestep=torch.tensor([500]),
    )
    assert torch.count_nonzero(residuals[0][:, :target_tokens]) > 0
    assert torch.count_nonzero(residuals[0][:, target_tokens:]) == 0


def test_architecture_config_and_full_parameter_count_is_reportable() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "adapter_v6_architecture.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["control_core"]["stages"] == 6
    assert payload["control_core"]["rank"] == 64
    assert payload["deepgen_injection"]["block_ranges"] == [
        [0, 3],
        [4, 7],
        [8, 11],
        [12, 15],
        [16, 19],
        [20, 22],
    ]

    deepgen_config = SimpleNamespace(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=tuple(range(13)),
        qk_norm="rms_norm",
    )
    with torch.device("meta"):
        model = UnifiedSMPLXAdapterV6.from_deepgen_config(deepgen_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable > 0


def test_from_deepgen_freezes_backbone_and_copies_first_block() -> None:
    # Tiny builder exercises the ownership rule without allocating the real
    # 24-layer backbone; the real-model smoke test covers the SD3 copy path.
    backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        context_input_dim=24,
        pooled_input_dim=20,
        num_heads=4,
        deepgen_backbone=backbone,
    )
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_pipeline_freeze_helper_covers_vae_vlm_and_connector() -> None:
    pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(4, 4),
        vae=torch.nn.Linear(4, 4),
        lmm=torch.nn.Linear(4, 4),
        connector_module=torch.nn.Linear(4, 4),
    )
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipeline)
    for name in ("transformer", "vae", "lmm", "connector_module"):
        component = getattr(pipeline, name)
        assert not component.training
        assert all(not parameter.requires_grad for parameter in component.parameters())

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.pose_control.v6.conditions import AdapterIdentityCondition, ContactRelationBatch, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.experts import TaskRouter
from src.pose_control.v6.reasoning import InternalControlState, SMPLXAdapterReasoner


def test_smoke_report_names_preserve_backend_results() -> None:
    from scripts.smoke_adapter_v6 import default_report_name

    assert default_report_name("single", "native", False) == "adapter_v6_smoke_single.json"
    assert default_report_name("dual", "native", False) == "adapter_v6_smoke_dual.json"
    assert default_report_name("dual", "champ", True) == "adapter_v6_smoke_dual_champ.json"


def build_tiny() -> UnifiedSMPLXAdapterV6:
    return UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        context_input_dim=24,
        pooled_input_dim=20,
        num_heads=4,
    ).eval()


def make_raw_conditions(*, batch_size: int = 1, num_people: int = 1, image_size: int = 32) -> dict:
    generator = torch.Generator().manual_seed(31)

    def person(side: str) -> dict:
        mask = torch.zeros(batch_size, 1, image_size, image_size)
        if side == "a":
            mask[..., 3:-3, 2 : image_size // 2 + 3] = 1
        else:
            mask[..., 3:-3, image_size // 2 - 3 : -2] = 1
        normal = (torch.rand(batch_size, 3, image_size, image_size, generator=generator) * 2 - 1) * mask
        pose = torch.zeros(batch_size, 25, image_size, image_size)
        pose[:, 0, image_size // 2, image_size // 2] = 1
        part = torch.zeros(batch_size, 14, image_size, image_size)
        part[:, 1] = mask[:, 0]
        return {
            f"normal_{side}": normal,
            f"pose_heatmap_{side}": pose,
            f"part_onehot_{side}": part,
            f"smplx_global_{side}": torch.randn(batch_size, 26, generator=generator),
            f"human_mask_{side}": mask,
            f"depth_{side}": torch.rand(batch_size, 1, image_size, image_size, generator=generator),
        }

    values = person("a")
    values["task_id"] = torch.full((batch_size,), int(TaskType.SINGLE), dtype=torch.long)
    if num_people == 2:
        values.update(person("b"))
        values["task_id"] = torch.full((batch_size,), int(TaskType.HANDSHAKE), dtype=torch.long)
        values["relative_geometry"] = torch.randn(batch_size, 11, generator=generator)
        values["contact_raster"] = torch.zeros(batch_size, 2, image_size, image_size)
        center = image_size // 2
        values["contact_raster"][..., center - 2 : center + 2, center - 2 : center + 2] = 1
        valid = torch.zeros(batch_size, 8, dtype=torch.bool)
        valid[:, 0] = True
        values["contact_relations"] = ContactRelationBatch(
            src_person=torch.zeros(batch_size, 8, dtype=torch.long),
            src_part=torch.ones(batch_size, 8, dtype=torch.long) * 4,
            dst_person=torch.ones(batch_size, 8, dtype=torch.long),
            dst_part=torch.ones(batch_size, 8, dtype=torch.long) * 7,
            contact_type=torch.ones(batch_size, 8, dtype=torch.long),
            distance=torch.zeros(batch_size, 8, 1),
            valid_mask=valid,
        )
    return values


def make_runtime(model: UnifiedSMPLXAdapterV6, *, num_people: int, image_size: int = 32, latent_size: int = 8):
    bundle = model.condition_injector(**make_raw_conditions(num_people=num_people, image_size=image_size))
    generator = torch.Generator().manual_seed(47)
    identity = AdapterIdentityCondition(
        source_person_latents=torch.randn(1, 2, 16, latent_size, latent_size, generator=generator),
        source_indices=torch.tensor([[0, 1]]),
    )
    return bundle, identity


def test_adapter_public_boundary_consumes_bundle_and_separate_identity() -> None:
    model = build_tiny()
    assert hasattr(model, "condition_injector")
    parameters = inspect.signature(model.prepare_conditioning).parameters
    assert "condition_bundle" in parameters
    assert "identity_condition" in parameters
    assert "condition" not in parameters
    bundle, identity = make_runtime(model, num_people=1)
    prepared = model.prepare_conditioning(bundle, identity)
    assert prepared.scene_feature.shape == (1, 16, 2, 2)
    assert prepared.route.num_people.tolist() == [1]
    assert isinstance(prepared.control_state, InternalControlState)
    assert isinstance(model.reasoner, SMPLXAdapterReasoner)
    assert model.reason_conditions(bundle) is not None


def test_legacy_raw_condition_contract_is_removed() -> None:
    from src.pose_control.v6 import conditions, geometry_encoder, token_encoders

    assert not hasattr(conditions, "UnifiedAdapterCondition")
    assert not hasattr(conditions, "TaskSpec")
    assert not hasattr(geometry_encoder, "SharedGeometryEncoder")
    assert not hasattr(token_encoders, "GlobalGeometryTokenEncoder")
    assert not hasattr(token_encoders, "ContactRelationEncoder")


def test_router_uses_only_bundle_person_count() -> None:
    route = TaskRouter().resolve(torch.tensor([1, 2]))
    assert route.num_people.tolist() == [1, 2]
    assert route.single_mask.tolist() == [True, False]
    assert route.dual_mask.tolist() == [False, True]


def test_old_experts_and_raw_condition_bridge_are_removed() -> None:
    from src.pose_control.v6 import condition_bridge, experts

    assert not hasattr(experts, "SinglePersonExpert")
    assert not hasattr(experts, "DualPersonExpert")
    assert not hasattr(condition_bridge, "ConditionBundleBridge")


def test_single_route_ignores_invalid_person_b_contact_and_appearance() -> None:
    torch.manual_seed(5)
    model = build_tiny()
    raw = make_raw_conditions(num_people=2)
    raw["person_b_valid"] = torch.tensor([False])
    raw["task_id"] = torch.tensor([int(TaskType.SINGLE)])
    first_bundle = model.condition_injector(**raw)
    changed_raw = copy.deepcopy(raw)
    changed_raw["normal_b"].mul_(-1)
    changed_raw["smplx_global_b"].add_(100)
    changed_raw["contact_raster"].uniform_(0, 1)
    changed_bundle = model.condition_injector(**changed_raw)
    first_identity = AdapterIdentityCondition(torch.randn(1, 2, 16, 8, 8))
    changed_identity = copy.deepcopy(first_identity)
    changed_identity.source_person_latents[:, 1].normal_(mean=100, std=20)
    first = model.prepare_conditioning(first_bundle, first_identity)
    second = model.prepare_conditioning(changed_bundle, changed_identity)
    torch.testing.assert_close(first.scene_feature, second.scene_feature)
    torch.testing.assert_close(first.adapter_tokens, second.adapter_tokens)
    assert torch.count_nonzero(first.contact_token_mask) == 0
    assert torch.count_nonzero(first.control_state.interaction_feature) == 0


def test_source_appearance_remains_condition_sensitive() -> None:
    torch.manual_seed(6)
    model = build_tiny()
    bundle, identity = make_runtime(model, num_people=1)
    changed = copy.deepcopy(identity)
    changed.source_person_latents[:, 0, 0].add_(3)
    first = model.prepare_conditioning(bundle, identity)
    second = model.prepare_conditioning(bundle, changed)
    assert not torch.allclose(first.adapter_tokens, second.adapter_tokens)
    torch.testing.assert_close(
        first.control_state.person_tokens, second.control_state.person_tokens
    )


def test_control_core_context_contains_only_bound_person_tokens() -> None:
    model = build_tiny()
    bundle, identity = make_runtime(model, num_people=2)
    prepared = model.prepare_conditioning(bundle, identity)
    assert prepared.adapter_tokens.shape == (1, 24, 32)
    assert prepared.adapter_token_mask.shape == (1, 24)
    per_person_mask = prepared.adapter_token_mask.reshape(1, 2, 12)
    assert per_person_mask[..., 4:].all()
    assert per_person_mask[..., :4].any(dim=-1).all()


def test_zero_heads_emit_six_target_only_residuals_and_alignment_is_exact() -> None:
    torch.manual_seed(7)
    model = build_tiny()
    bundle, identity = make_runtime(model, num_people=2)
    target = torch.randn(1, 16, 8, 8)
    source_scene = torch.randn(1, 16, 8, 8)
    references = [[torch.randn(16, 8, 8), torch.randn(16, 4, 4)]]
    kwargs = dict(
        target_latents=target, condition_bundle=bundle, identity_condition=identity,
        source_scene_latents=source_scene, cond_hidden_states=references,
        encoder_hidden_states=torch.randn(1, 5, 24),
        pooled_projections=torch.randn(1, 20), timestep=torch.tensor([500]),
    )
    residuals = model(**kwargs)
    target_tokens = (8 // 2) ** 2
    full_tokens = target_tokens + (8 // 2) ** 2 + (4 // 2) ** 2
    assert len(residuals) == 6
    assert all(item.shape == (1, full_tokens, 32) for item in residuals)
    assert all(torch.count_nonzero(item) == 0 for item in residuals)
    assert all(torch.count_nonzero(head.weight) == 0 for head in model.control_core.zero_heads)
    with torch.no_grad():
        model.control_core.zero_heads[0].weight.fill_(0.01)
    residuals = model(**kwargs)
    assert torch.count_nonzero(residuals[0][:, :target_tokens]) > 0
    assert torch.count_nonzero(residuals[0][:, target_tokens:]) == 0


def test_contact_padding_mask_does_not_couple_batch_rows() -> None:
    torch.manual_seed(71)
    model = build_tiny()
    raw = make_raw_conditions(batch_size=2, num_people=2)
    raw["contact_relations"].valid_mask[0, 1:] = False
    raw["contact_relations"].valid_mask[1] = True
    bundle = model.condition_injector(**raw)
    identity = AdapterIdentityCondition(
        source_person_latents=torch.randn(2, 2, 16, 8, 8),
        source_indices=torch.tensor([[0, 1], [0, 1]]),
    )
    target = torch.randn(2, 16, 8, 8)
    source_scene = torch.randn(2, 16, 8, 8)
    text = torch.randn(2, 5, 24)
    pooled = torch.randn(2, 20)
    timestep = torch.tensor([500, 500])
    with torch.no_grad():
        model.control_core.zero_heads[0].weight.copy_(torch.eye(32))
        model.control_core.zero_heads[0].bias.zero_()

    batched = model(
        target_latents=target,
        condition_bundle=bundle,
        identity_condition=identity,
        source_scene_latents=source_scene,
        cond_hidden_states=None,
        encoder_hidden_states=text,
        pooled_projections=pooled,
        timestep=timestep,
    )[0][0]
    first = torch.tensor([0])
    standalone = model(
        target_latents=target.index_select(0, first),
        condition_bundle=bundle.index_select(first),
        identity_condition=identity.index_select(first),
        source_scene_latents=source_scene.index_select(0, first),
        cond_hidden_states=None,
        encoder_hidden_states=text.index_select(0, first),
        pooled_projections=pooled.index_select(0, first),
        timestep=timestep.index_select(0, first),
    )[0][0]

    torch.testing.assert_close(batched, standalone, atol=1e-5, rtol=1e-5)


def test_architecture_config_reports_parameters_without_a_size_gate() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "adapter_v6_architecture.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["control_core"]["stages"] == 6
    assert payload["control_core"]["rank"] == 64
    assert payload["parameter_policy"]["limit"] is None
    assert payload["condition_injection"]["pose_channels"] == 25
    assert payload["condition_injection"]["part_channels"] == 14
    assert payload["dual_person_reasoner"]["depth_aware_scene_fusion"] is False
    assert payload["dual_person_reasoner"]["depth_order_fusion_stage"] == "deferred-v2"
    deepgen_config = SimpleNamespace(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24,
        attention_head_dim=64, num_attention_heads=24, joint_attention_dim=4096,
        caption_projection_dim=1536, pooled_projection_dim=2048, out_channels=16,
        pos_embed_max_size=384, dual_attention_layers=tuple(range(13)), qk_norm="rms_norm",
    )
    with torch.device("meta"):
        model = UnifiedSMPLXAdapterV6.from_deepgen_config(deepgen_config)
    assert model.trainable_parameter_count > 0


def test_smoke_parameter_report_uses_reasoner_instead_of_removed_experts() -> None:
    from scripts.smoke_adapter_v6 import parameter_report

    report = parameter_report(build_tiny())
    assert report["reasoner"] > 0
    assert report["reasoner_control_bridge"] > 0
    assert "single_expert" not in report
    assert "dual_expert" not in report


def test_from_deepgen_freezes_backbone() -> None:
    backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32, geometry_channels=16, context_input_dim=24,
        pooled_input_dim=20, num_heads=4, deepgen_backbone=backbone,
    )
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_builders_forward_condition_backend_options() -> None:
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        context_input_dim=24,
        pooled_input_dim=20,
        num_heads=4,
        condition_use_depth=True,
        normal_backend="champ",
        depth_backend="champ",
    )
    assert model.condition_injector.normal_backend == "champ"
    assert model.condition_injector.depth_backend == "champ"
    assert model.condition_injector.use_depth


def test_pipeline_freeze_helper_covers_vae_vlm_and_connector() -> None:
    pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(4, 4), vae=torch.nn.Linear(4, 4),
        lmm=torch.nn.Linear(4, 4), connector_module=torch.nn.Linear(4, 4),
    )
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipeline)
    for name in ("transformer", "vae", "lmm", "connector_module"):
        component = getattr(pipeline, name)
        assert not component.training
        assert all(not parameter.requires_grad for parameter in component.parameters())

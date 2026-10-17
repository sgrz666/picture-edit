from __future__ import annotations

import inspect
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_bridge import ReasonerControlBridge
from .condition_injector import SMPLXConditionInjector
from .conditions import AdapterIdentityCondition, ConditionBundle
from .control_core import SharedRecurrentControlCore
from .detail import (
    DetailReferenceBatch,
    FaceHandDetailCondition,
    FaceHandDetailPreparer,
    HandDetailCondition,
    hand_to_legacy_detail,
    legacy_to_face_and_hand,
    references_to_hand_only,
)
from .face import FaceConditioningPreparer, FaceFineCondition, FaceReferenceFeatures
from .face.adapter import FaceControlAdapter
from .hand import HandConditioningPreparer, HandControlAdapter, HandFineCondition, HandReferenceFeatures
from .interface import (
    DeepGenControlInterface,
    DeepGenControlOutput,
    PreparedControlConditioning,
    StrengthScheduleConfig,
    align_target_residuals,
)
from .interface.deepgen_interface import latent_token_count
from .reasoning import AdapterReasoningConfig, InternalControlState, SMPLXAdapterReasoner
from .token_encoders import AppearanceTokenEncoder, PersonTokenBinder


class UnifiedSMPLXAdapterV6(nn.Module):
    """Static SMPL-X reasoning plus dynamic context-aware DeepGen control."""

    def __init__(
        self,
        control_core: SharedRecurrentControlCore,
        *,
        geometry_channels: int = 128,
        cross_attention_dim: int = 512,
        expert_heads: int = 8,
        condition_use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        reasoning_config: AdapterReasoningConfig | None = None,
        num_transformer_layers: int = 24,
        context_pre_only_blocks: tuple[int, ...] | None = None,
        geometry_schedule: StrengthScheduleConfig | None = None,
        interaction_schedule: StrengthScheduleConfig | None = None,
        detail_schedule: StrengthScheduleConfig | None = None,
        face_schedule: StrengthScheduleConfig | None = None,
        face_texture_schedule: StrengthScheduleConfig | None = None,
        enable_face_adapter: bool = False,
        enable_hand_adapter: bool = False,
        face_dim: int = 512,
        face_rank: int = 64,
        face_heads: int = 8,
        face_resampler_depth: int = 4,
        face_perceiver_depth: int = 4,
    ) -> None:
        super().__init__()
        token_dim = control_core.hidden_dim
        self.condition_injector = SMPLXConditionInjector(
            use_depth=condition_use_depth,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
        )
        if reasoning_config is None:
            reasoning_config = AdapterReasoningConfig(
                hidden_dim=cross_attention_dim,
                global_attention_heads=expert_heads,
                cross_person_heads=expert_heads,
                contact_attention_heads=expert_heads,
            )
        self.reasoner = SMPLXAdapterReasoner(reasoning_config)
        self.condition_bridge = ReasonerControlBridge(
            reasoning_dim=reasoning_config.hidden_dim,
            geometry_channels=geometry_channels,
            token_dim=token_dim,
        )
        self.geometry_token_projection = nn.Linear(
            reasoning_config.hidden_dim, token_dim
        )
        self.appearance_token_encoder = AppearanceTokenEncoder(token_dim=token_dim)
        self.person_token_binder = PersonTokenBinder(token_dim=token_dim)
        self.detail_preparer = FaceHandDetailPreparer(token_dim=token_dim)
        self.control_interface = DeepGenControlInterface(
            control_core,
            num_transformer_layers=num_transformer_layers,
            context_pre_only_blocks=context_pre_only_blocks,
            geometry_schedule=geometry_schedule,
            interaction_schedule=interaction_schedule,
            detail_schedule=detail_schedule,
            face_schedule=face_schedule,
            face_texture_schedule=face_texture_schedule,
            enable_face_controller=enable_face_adapter,
        )
        self.hand_preparer: HandConditioningPreparer | None = None
        if enable_hand_adapter:
            hand_dim = face_dim if token_dim != 1536 else 512
            self.hand_preparer = HandConditioningPreparer(dim=hand_dim, resampler_depth=1 if token_dim != 1536 else 2, heads=face_heads)
            self.control_interface.hand_adapter = HandControlAdapter(dim=hand_dim, deepgen_dim=token_dim, rank=face_rank if token_dim != 1536 else 64, heads=face_heads)
            self.control_interface.strength_controller.enable_v66_hand()
        self.face_preparer: FaceConditioningPreparer | None = None
        if enable_face_adapter:
            self.face_preparer = FaceConditioningPreparer(
                resampler_depth=face_resampler_depth,
                perceiver_depth=face_perceiver_depth,
            )
            self.control_interface.face_adapter = FaceControlAdapter(
                face_dim=face_dim,
                deepgen_dim=token_dim,
                rank=face_rank,
                heads=face_heads,
            )
        self.geometry_channels = geometry_channels
        self.reasoning_config = reasoning_config

    @property
    def control_core(self) -> SharedRecurrentControlCore:
        return self.control_interface.control_core

    @staticmethod
    def _freeze_backbone(backbone: nn.Module | None) -> None:
        if backbone is None:
            return
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def freeze_deepgen_pipeline_components(cls, pipeline) -> None:
        for name in (
            "transformer",
            "vae",
            "lmm",
            "vlm",
            "connector_module",
            "connector",
        ):
            cls._freeze_backbone(getattr(pipeline, name, None))

    @classmethod
    def build_tiny(
        cls,
        *,
        hidden_dim: int = 32,
        geometry_channels: int = 16,
        context_input_dim: int = 24,
        pooled_input_dim: int = 20,
        num_heads: int = 4,
        deepgen_backbone: nn.Module | None = None,
        condition_use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        reasoning_config: AdapterReasoningConfig | None = None,
        enable_face_adapter: bool = False,
        enable_hand_adapter: bool = False,
        face_dim: int = 32,
    ) -> "UnifiedSMPLXAdapterV6":
        cls._freeze_backbone(deepgen_backbone)
        core = SharedRecurrentControlCore.build_tiny(
            hidden_dim=hidden_dim,
            condition_channels=16 + geometry_channels,
            context_input_dim=context_input_dim,
            pooled_input_dim=pooled_input_dim,
            rank=max(4, hidden_dim // 4),
        )
        return cls(
            core,
            geometry_channels=geometry_channels,
            cross_attention_dim=max(hidden_dim, geometry_channels),
            expert_heads=num_heads,
            condition_use_depth=condition_use_depth,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
            reasoning_config=reasoning_config,
            enable_face_adapter=enable_face_adapter,
            enable_hand_adapter=enable_hand_adapter,
            face_dim=face_dim,
            face_rank=max(4, face_dim // 4),
            face_heads=num_heads,
            face_resampler_depth=1,
            face_perceiver_depth=1,
        )

    @classmethod
    def from_deepgen_config(
        cls,
        config,
        *,
        condition_use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        reasoning_config: AdapterReasoningConfig | None = None,
        enable_face_adapter: bool = True,
        enable_hand_adapter: bool = False,
    ) -> "UnifiedSMPLXAdapterV6":
        core = SharedRecurrentControlCore.from_deepgen_config(config)
        return cls(
            core,
            condition_use_depth=condition_use_depth,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
            reasoning_config=reasoning_config,
            enable_face_adapter=enable_face_adapter,
            enable_hand_adapter=enable_hand_adapter,
            num_transformer_layers=config.num_layers,
            context_pre_only_blocks=(config.num_layers - 1,),
        )

    @classmethod
    def from_deepgen(
        cls,
        deepgen_transformer: nn.Module,
        *,
        condition_use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        reasoning_config: AdapterReasoningConfig | None = None,
        enable_face_adapter: bool = True,
        enable_hand_adapter: bool = False,
    ) -> "UnifiedSMPLXAdapterV6":
        if "block_controlnet_hidden_states" not in inspect.signature(
            deepgen_transformer.forward
        ).parameters:
            raise ValueError(
                "DeepGen transformer must accept block_controlnet_hidden_states"
            )
        cls._freeze_backbone(deepgen_transformer)
        core = SharedRecurrentControlCore.from_deepgen(deepgen_transformer)
        pre_only = tuple(
            index
            for index, block in enumerate(deepgen_transformer.transformer_blocks)
            if bool(block.context_pre_only)
        )
        return cls(
            core,
            condition_use_depth=condition_use_depth,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
            reasoning_config=reasoning_config,
            enable_face_adapter=enable_face_adapter,
            enable_hand_adapter=enable_hand_adapter,
            num_transformer_layers=len(deepgen_transformer.transformer_blocks),
            context_pre_only_blocks=pre_only,
        )

    @classmethod
    def from_deepgen_pipeline(
        cls,
        pipeline,
        *,
        condition_use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        reasoning_config: AdapterReasoningConfig | None = None,
        enable_face_adapter: bool = True,
        enable_hand_adapter: bool = False,
    ) -> "UnifiedSMPLXAdapterV6":
        cls.freeze_deepgen_pipeline_components(pipeline)
        return cls.from_deepgen(
            pipeline.transformer,
            condition_use_depth=condition_use_depth,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
            reasoning_config=reasoning_config,
            enable_face_adapter=enable_face_adapter,
            enable_hand_adapter=enable_hand_adapter,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def reason_conditions(self, condition_bundle: ConditionBundle) -> InternalControlState:
        return self.reasoner(condition_bundle)

    @staticmethod
    def _pool_person_tokens(
        state: InternalControlState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, people, token_count, channels = state.person_tokens.shape
        height, width = state.geometry_feature.shape[-2:]
        if token_count != height * width:
            raise ValueError("person token count must match reasoning feature size")
        features = state.person_tokens.reshape(
            batch_size * people, height, width, channels
        ).permute(0, 3, 1, 2)
        mask = state.person_token_mask.reshape(batch_size * people, 1, height, width)
        weights = mask.to(features.dtype)
        numerator = F.adaptive_avg_pool2d(features * weights, (2, 2))
        denominator = F.adaptive_avg_pool2d(weights, (2, 2))
        pooled_mask = denominator > 0
        pooled = numerator / denominator.clamp_min(1e-6)
        pooled = pooled * pooled_mask.to(pooled.dtype)
        pooled = pooled.flatten(2).transpose(1, 2).reshape(
            batch_size, people, 4, channels
        )
        pooled_mask = pooled_mask.flatten(1).reshape(batch_size, people, 4)
        return pooled, pooled_mask

    def prepare_control(
        self,
        *,
        state: InternalControlState,
        identity_condition: AdapterIdentityCondition,
        source_scene_latents: torch.Tensor,
        target_latent_hw: tuple[int, int],
        detail_condition: FaceHandDetailCondition | None = None,
        detail_references: DetailReferenceBatch | None = None,
        hand_condition: HandDetailCondition | None = None,
        hand_references: DetailReferenceBatch | None = None,
        hand_features: HandReferenceFeatures | None = None,
        face_condition: FaceFineCondition | None = None,
        face_references: FaceReferenceFeatures | None = None,
    ) -> PreparedControlConditioning:
        state.validate()
        batch_size = state.batch_size
        if identity_condition.source_person_latents.shape[:3] != (batch_size, 2, 16):
            raise ValueError("source_person_latents must have shape [B,2,16,h,w]")
        if (
            identity_condition.source_indices is not None
            and identity_condition.source_indices.shape != (batch_size, 2)
        ):
            raise ValueError("source_indices must have shape [B,2]")
        if source_scene_latents.ndim != 4 or source_scene_latents.shape[:2] != (
            batch_size,
            16,
        ):
            raise ValueError("source_scene_latents must have shape [B,16,h,w]")
        if len(target_latent_hw) != 2 or min(target_latent_hw) <= 0:
            raise ValueError("target_latent_hw must contain two positive dimensions")

        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        state = state.to(device=device, dtype=dtype)
        identity_condition = identity_condition.to(device=device, dtype=dtype)
        source_scene_latents = source_scene_latents.to(device=device, dtype=dtype)

        if detail_condition is not None and hand_condition is not None:
            raise ValueError("use either legacy detail_condition or hand_condition, not both")
        if detail_references is not None and hand_references is not None:
            raise ValueError("use either legacy detail_references or hand_references, not both")
        legacy_face_condition = None
        if detail_condition is not None:
            detail_condition = detail_condition.to(device=device, dtype=dtype)
            legacy_face_condition, hand_condition = legacy_to_face_and_hand(
                detail_condition
            )
            if detail_references is not None:
                hand_references = references_to_hand_only(
                    detail_references.to(device=device, dtype=dtype)
                )
        elif detail_references is not None:
            raise ValueError("detail references require a detail condition")
        if face_condition is None and face_references is not None:
            face_condition = legacy_face_condition
        if (face_condition is None) != (face_references is None):
            raise ValueError("face_condition and face_references must be provided together")

        expected_source_indices = identity_condition.effective_source_indices()

        def validate_binding(name, condition, person_valid) -> None:
            condition.validate()
            if condition.batch_size != batch_size:
                raise ValueError(f"{name} condition batch does not match control state")
            if torch.any(person_valid & ~state.person_valid):
                raise ValueError(
                    f"each {name}-valid person must be valid in the control state"
                )
            if torch.any(
                person_valid & (condition.source_indices != expected_source_indices)
            ):
                raise ValueError(
                    f"{name} source_indices must match identity source_indices for every valid person"
                )

        if hand_condition is not None:
            hand_condition = hand_condition.to(device=device, dtype=dtype)
            validate_binding("detail", hand_condition, hand_condition.person_valid)
            detail_condition = hand_to_legacy_detail(hand_condition).validate()
            if hand_references is None:
                hand_references = DetailReferenceBatch(
                    images=torch.zeros(
                        batch_size,
                        2,
                        3,
                        1,
                        3,
                        224,
                        224,
                        device=device,
                        dtype=dtype,
                    ),
                    reference_valid=torch.zeros(
                        batch_size, 2, 3, 1, device=device, dtype=torch.bool
                    ),
                )
            else:
                hand_references = references_to_hand_only(
                    hand_references.to(device=device, dtype=dtype)
                )
        elif hand_references is not None:
            raise ValueError("hand references require a hand condition")
        if hand_features is not None and hand_condition is None:
            raise ValueError("hand_features require a hand condition")

        if face_condition is not None:
            if self.face_preparer is None or self.control_interface.face_adapter is None:
                raise ValueError("face conditioning requires an enabled face adapter")
            face_condition = face_condition.to(device=device, dtype=dtype)
            assert face_references is not None
            face_references = face_references.to(device=device, dtype=dtype)
            validate_binding("face", face_condition, face_condition.face_valid)
            face_references.validate()
            if face_references.batch_size != batch_size:
                raise ValueError("face reference batch does not match control state")
        bridged = self.condition_bridge(state)
        geometry_scene = F.interpolate(
            bridged.geometry_scene,
            size=target_latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        interaction_scene = F.interpolate(
            bridged.interaction_scene,
            size=target_latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        source_scene = F.interpolate(
            source_scene_latents,
            size=target_latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        geometry_condition = torch.cat((source_scene, geometry_scene), dim=1)
        interaction_condition = torch.cat(
            (torch.zeros_like(source_scene), interaction_scene), dim=1
        )
        interaction_condition = interaction_condition * state.interaction_valid[
            :, None, None, None
        ].to(interaction_condition.dtype)

        pooled, pooled_mask = self._pool_person_tokens(state)
        geometry_tokens = self.geometry_token_projection(pooled)
        geometry_tokens = geometry_tokens * pooled_mask[..., None].to(
            geometry_tokens.dtype
        )
        appearance_tokens = self.appearance_token_encoder(
            identity_condition.source_person_latents, state.person_valid
        )
        person_binding = self.person_token_binder.binding_for(
            identity_condition.effective_source_indices(), state.person_valid
        )
        person_tokens, person_token_mask = self.person_token_binder(
            geometry_tokens,
            appearance_tokens,
            state.person_valid,
            identity_condition.effective_source_indices(),
            geometry_token_mask=pooled_mask,
            person_binding=person_binding,
        )
        adapter_tokens = person_tokens.flatten(1, 2)
        adapter_token_mask = person_token_mask.flatten(1, 2)
        adapter_tokens = adapter_tokens * adapter_token_mask[..., None].to(
            adapter_tokens.dtype
        )
        prepared_detail = None
        if detail_condition is not None:
            if target_latent_hw[0] % self.control_core.patch_size or target_latent_hw[1] % self.control_core.patch_size:
                raise ValueError("target latent size must be divisible by control patch_size")
            assert hand_references is not None
            prepared_detail = self.detail_preparer(
                detail_condition,
                hand_references,
                target_latent_hw,
                (
                    target_latent_hw[0] // self.control_core.patch_size,
                    target_latent_hw[1] // self.control_core.patch_size,
                ),
                identity_condition.source_person_latents,
                person_binding=person_binding,
            )
        prepared_face = None
        if face_condition is not None:
            assert self.face_preparer is not None
            assert face_references is not None
            prepared_face = self.face_preparer(face_condition, face_references)
        prepared_hand = None
        if hand_features is not None:
            if self.hand_preparer is None or self.control_interface.hand_adapter is None:
                raise ValueError("hand features require an enabled v66 adapter")
            hand_features = hand_features.to(device=device, dtype=dtype)
            fine = hand_condition if isinstance(hand_condition, HandFineCondition) else HandFineCondition.from_legacy(hand_condition)
            prepared_hand = self.hand_preparer(fine, hand_features)
        return PreparedControlConditioning(
            geometry_condition=geometry_condition,
            interaction_condition=interaction_condition,
            adapter_tokens=adapter_tokens,
            adapter_token_mask=adapter_token_mask,
            person_count=state.person_count,
            interaction_valid=state.interaction_valid,
            control_state=state,
            detail=prepared_detail,
            face=prepared_face,
            hand=prepared_hand,
        ).validate()

    def prepare_conditioning(
        self,
        condition_bundle: ConditionBundle,
        identity_condition: AdapterIdentityCondition,
        source_scene_latents: torch.Tensor,
        target_latent_hw: tuple[int, int],
        detail_condition: FaceHandDetailCondition | None = None,
        detail_references: DetailReferenceBatch | None = None,
        hand_condition: HandDetailCondition | None = None,
        hand_references: DetailReferenceBatch | None = None,
        hand_features: HandReferenceFeatures | None = None,
        face_condition: FaceFineCondition | None = None,
        face_references: FaceReferenceFeatures | None = None,
    ) -> PreparedControlConditioning:
        condition_bundle.validate()
        identity_condition.validate(condition_bundle)
        return self.prepare_control(
            state=self.reason_conditions(condition_bundle),
            identity_condition=identity_condition,
            source_scene_latents=source_scene_latents,
            target_latent_hw=target_latent_hw,
            detail_condition=detail_condition,
            detail_references=detail_references,
            hand_condition=hand_condition,
            hand_references=hand_references,
            hand_features=hand_features,
            face_condition=face_condition,
            face_references=face_references,
        )

    def forward(
        self,
        *,
        target_latents: torch.Tensor,
        prepared: PreparedControlConditioning,
        cond_hidden_states: Optional[Sequence[Sequence[torch.Tensor]]],
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        denoise_progress: float | torch.Tensor,
        geometry_strength: float | torch.Tensor = 1.0,
        interaction_strength: float | torch.Tensor = 0.8,
        detail_strength: float | torch.Tensor = 1.0,
        face_strength: float | torch.Tensor = 1.0,
        hand_strength: float | torch.Tensor = 1.0,
        hand_mode: str = "legacy",
        joint_attention_kwargs: Optional[dict] = None,
    ) -> DeepGenControlOutput:
        parameter = next(self.parameters())
        return self.control_interface(
            target_latents=target_latents.to(dtype=parameter.dtype),
            prepared=prepared.to(
                device=target_latents.device, dtype=parameter.dtype
            ),
            cond_hidden_states=cond_hidden_states,
            encoder_hidden_states=encoder_hidden_states.to(
                device=target_latents.device, dtype=parameter.dtype
            ),
            pooled_projections=pooled_projections.to(
                device=target_latents.device, dtype=parameter.dtype
            ),
            timestep=timestep.to(device=target_latents.device),
            denoise_progress=denoise_progress,
            geometry_strength=geometry_strength,
            interaction_strength=interaction_strength,
            detail_strength=detail_strength,
            face_strength=face_strength,
            hand_strength=hand_strength,
            hand_mode=hand_mode,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    def forward_deepgen(
        self,
        deepgen_transformer: nn.Module,
        *,
        adapter_kwargs: dict,
        transformer_kwargs: dict,
    ):
        output = self(**adapter_kwargs)
        kwargs = dict(transformer_kwargs)
        kwargs["block_controlnet_hidden_states"] = (
            output.block_controlnet_hidden_states
        )
        return deepgen_transformer(**kwargs), output


__all__ = [
    "UnifiedSMPLXAdapterV6",
    "align_target_residuals",
    "latent_token_count",
]

from __future__ import annotations

import threading
from typing import Any

import torch

from .conditions import AdapterIdentityCondition, ConditionBundle
from .deepgen_adapter import UnifiedSMPLXAdapterV6
from .detail import DetailReferenceBatch, FaceHandDetailCondition, HandDetailCondition
from .face import FaceFineCondition, FaceReferenceFeatures
from .hand import HandReferenceFeatures
from .interface import PreparedControlConditioning


class ControlledDeepGenPipeline:
    """Inject dynamic Adapter controls without modifying DeepGen runtime code."""

    def __init__(self, *, pipeline, adapter: UnifiedSMPLXAdapterV6) -> None:
        self.pipeline = pipeline
        self.adapter = adapter
        self._state_lock = threading.Lock()
        self._active = False
        self.last_control_diagnostics: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        return self._active

    def _enter(self) -> None:
        with self._state_lock:
            if self._active:
                raise RuntimeError(
                    "ControlledDeepGenPipeline does not support nested or concurrent calls"
                )
            self._active = True

    def _exit(self) -> None:
        with self._state_lock:
            self._active = False

    def _vae_scale_factor(self) -> int:
        configured = getattr(self.pipeline, "vae_scale_factor", None)
        if configured is not None:
            return int(configured)
        vae = getattr(self.pipeline, "vae", None)
        block_channels = getattr(getattr(vae, "config", None), "block_out_channels", None)
        if block_channels is None:
            raise ValueError("cannot infer DeepGen VAE scale factor")
        return 2 ** (len(block_channels) - 1)

    def _prepare(
        self,
        *,
        condition_bundle: ConditionBundle,
        identity_condition: AdapterIdentityCondition,
        source_scene_latents: torch.Tensor,
        height: int,
        width: int,
        detail_condition: FaceHandDetailCondition | None = None,
        detail_references: DetailReferenceBatch | None = None,
        hand_condition: HandDetailCondition | None = None,
        hand_references: DetailReferenceBatch | None = None,
        hand_features: HandReferenceFeatures | None = None,
        face_condition: FaceFineCondition | None = None,
        face_references: FaceReferenceFeatures | None = None,
    ) -> PreparedControlConditioning:
        scale = self._vae_scale_factor()
        if height % scale or width % scale:
            raise ValueError("height and width must be divisible by the VAE scale factor")
        return self.adapter.prepare_conditioning(
            condition_bundle,
            identity_condition,
            source_scene_latents,
            (height // scale, width // scale),
            detail_condition=detail_condition,
            detail_references=detail_references,
            hand_condition=hand_condition,
            hand_references=hand_references,
            hand_features=hand_features,
            face_condition=face_condition,
            face_references=face_references,
        )

    def __call__(
        self,
        *,
        condition_bundle: ConditionBundle | None = None,
        identity_condition: AdapterIdentityCondition | None = None,
        source_scene_latents: torch.Tensor | None = None,
        prepared_control: PreparedControlConditioning | None = None,
        geometry_strength: float = 1.0,
        interaction_strength: float = 0.8,
        detail_strength: float = 1.0,
        face_strength: float = 1.0,
        hand_strength: float = 1.0,
        hand_mode: str = "legacy",
        detail_condition: FaceHandDetailCondition | None = None,
        detail_references: DetailReferenceBatch | None = None,
        hand_condition: HandDetailCondition | None = None,
        hand_references: DetailReferenceBatch | None = None,
        hand_features: HandReferenceFeatures | None = None,
        face_condition: FaceFineCondition | None = None,
        face_references: FaceReferenceFeatures | None = None,
        cache_adapter_outputs: bool = True,
        num_inference_steps: int = 28,
        **pipeline_kwargs,
    ):
        external = pipeline_kwargs.pop("block_controlnet_hidden_states", None)
        if external is not None:
            raise ValueError("external block_controlnet_hidden_states would duplicate control")
        control_scale = float(pipeline_kwargs.pop("control_scale", 1.0))
        if control_scale != 1.0:
            raise ValueError("control_scale must remain 1.0; use branch strengths instead")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        height = int(pipeline_kwargs.get("height", 512))
        width = int(pipeline_kwargs.get("width", 512))
        if prepared_control is None:
            if condition_bundle is None or identity_condition is None or source_scene_latents is None:
                raise ValueError(
                    "condition_bundle, identity_condition, and source_scene_latents are required"
                )
            if cache_adapter_outputs:
                prepared_control = self._prepare(
                    condition_bundle=condition_bundle,
                    identity_condition=identity_condition,
                    source_scene_latents=source_scene_latents,
                    height=height,
                    width=width,
                    detail_condition=detail_condition,
                    detail_references=detail_references,
                    hand_condition=hand_condition,
                    hand_references=hand_references,
                    hand_features=hand_features,
                    face_condition=face_condition,
                    face_references=face_references,
                )
        else:
            prepared_control.validate()

        self._enter()
        diagnostics: list[dict[str, Any]] = []
        step_index = 0

        def current_prepared() -> PreparedControlConditioning:
            if prepared_control is not None:
                return prepared_control
            assert condition_bundle is not None
            assert identity_condition is not None
            assert source_scene_latents is not None
            return self._prepare(
                condition_bundle=condition_bundle,
                identity_condition=identity_condition,
                source_scene_latents=source_scene_latents,
                height=height,
                width=width,
                detail_condition=detail_condition,
                detail_references=detail_references,
                hand_condition=hand_condition,
                hand_references=hand_references,
                hand_features=hand_features,
                face_condition=face_condition,
                face_references=face_references,
            )

        def inject_control(module, args, kwargs):
            del module
            nonlocal step_index
            if kwargs.get("block_controlnet_hidden_states") is not None:
                raise ValueError("transformer already received external block control")
            if step_index >= num_inference_steps:
                raise RuntimeError("transformer was called more times than denoising steps")
            required = (
                "hidden_states",
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
            )
            missing = [name for name in required if name not in kwargs]
            if missing:
                raise ValueError(f"DeepGen transformer call is missing {missing}")
            progress = step_index / max(num_inference_steps - 1, 1)
            output = self.adapter(
                target_latents=kwargs["hidden_states"],
                prepared=current_prepared(),
                cond_hidden_states=kwargs.get("cond_hidden_states"),
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                pooled_projections=kwargs["pooled_projections"],
                timestep=kwargs["timestep"],
                denoise_progress=progress,
                geometry_strength=geometry_strength,
                interaction_strength=interaction_strength,
                detail_strength=detail_strength,
                face_strength=face_strength,
                hand_strength=hand_strength,
                hand_mode=hand_mode,
                joint_attention_kwargs=kwargs.get("joint_attention_kwargs"),
            )
            kwargs["block_controlnet_hidden_states"] = (
                output.block_controlnet_hidden_states
            )
            diagnostics.append(output.diagnostics)
            step_index += 1
            return args, kwargs

        handle = None
        try:
            handle = self.pipeline.transformer.register_forward_pre_hook(
                inject_control, with_kwargs=True
            )
            result = self.pipeline(
                num_inference_steps=num_inference_steps,
                block_controlnet_hidden_states=None,
                control_scale=1.0,
                **pipeline_kwargs,
            )
            self.last_control_diagnostics = diagnostics
            try:
                result.control_diagnostics = diagnostics
            except (AttributeError, TypeError):
                pass
            return result
        finally:
            if handle is not None:
                handle.remove()
            self._exit()


__all__ = ["ControlledDeepGenPipeline"]

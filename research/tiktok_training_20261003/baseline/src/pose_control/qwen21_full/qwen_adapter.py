"""Qwen boundary adapters around the unchanged, complete V6 control architecture.

The copied V6 modules retain their 1536-dimensional internal representations.
Only latent/text input coordinates, residual output coordinates, and injection
into the official Qwen target stream are adapted here.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .checkpoint import DETAIL_ONLY_PREFIXES, FACE_ONLY_PREFIXES, HAND_ONLY_PREFIXES
from .conditions import AdapterIdentityCondition, ConditionBundle
from .control_core import BranchControlResiduals
from .deepgen_adapter import UnifiedSMPLXAdapterV6
from .interface.outputs import PreparedControlConditioning


BRANCH_NAMES = ("geometry", "interaction", "detail", "face", "hand")
LAYER_GROUPS = tuple(tuple(i for i in range(32) if 6 * i // 32 == g) for g in range(6))


def resize_face_residuals_from_reference_grid(
    residuals: tuple[torch.Tensor, ...], *, reference_token_hw: tuple[int, int],
    target_token_hw: tuple[int, int], prepared_face,
) -> tuple[torch.Tensor, ...]:
    """Reuse supervised scatter locations without changing the retained ROI module.

    The existing Face residuals already contain their feathered mask. Resizing
    can spread support beyond the target boxes, so apply a hard box union at
    actual target-token centers; applying the feather again would attenuate it
    twice. Equal grids bypass both interpolation and masking exactly.
    """
    if tuple(reference_token_hw) == tuple(target_token_hw):
        return residuals
    source_height, source_width = reference_token_hw
    target_height, target_width = target_token_hw
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError('Face reference and target token grids must be positive')
    boxes = prepared_face.target_boxes.float()
    y = (torch.arange(target_height, device=boxes.device, dtype=boxes.dtype) + .5) / target_height
    x = (torch.arange(target_width, device=boxes.device, dtype=boxes.dtype) + .5) / target_width
    inside = (
        (x[None, None, None, :] >= boxes[:, :, 0, None, None])
        & (x[None, None, None, :] <= boxes[:, :, 2, None, None])
        & (y[None, None, :, None] >= boxes[:, :, 1, None, None])
        & (y[None, None, :, None] <= boxes[:, :, 3, None, None])
        & prepared_face.face_valid[..., None, None]
    ).any(dim=1).flatten(1)[..., None]
    resized = []
    for residual in residuals:
        if residual.ndim != 3 or residual.shape[:2] != (boxes.shape[0], source_height * source_width):
            raise ValueError('Face residual shape does not match the reference token grid')
        channels_first = residual.transpose(1, 2).reshape(
            residual.shape[0], residual.shape[-1], source_height, source_width)
        value = F.interpolate(channels_first, size=target_token_hw,
                              mode='bilinear', align_corners=False)
        value = value.flatten(2).transpose(1, 2)
        resized.append(value * inside.to(value.dtype))
    return tuple(resized)


@dataclass
class QwenFullControlOutput:
    residuals: tuple[torch.Tensor, ...]
    diagnostics: dict[str, Any]
    raw_residuals: BranchControlResiduals

    @property
    def block_controlnet_hidden_states(self):
        return self.residuals


class Qwen21FullAdapter(nn.Module):
    def __init__(
        self, legacy: UnifiedSMPLXAdapterV6, *, qwen_hidden_size: int = 4096,
        qwen_text_dim: int = 4096, pooled_input_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.legacy = legacy
        self.qwen_hidden_size = qwen_hidden_size
        self.latent_bridge = nn.Conv2d(64, 64, 1)
        nn.init.eye_(self.latent_bridge.weight[:, :, 0, 0])
        nn.init.zeros_(self.latent_bridge.bias)
        context_dim = legacy.control_core.context_embedder.in_features
        self.text_bridge = nn.Linear(qwen_text_dim, context_dim)
        if qwen_text_dim == context_dim:
            nn.init.eye_(self.text_bridge.weight)
            nn.init.zeros_(self.text_bridge.bias)
        self.pooled_bridge = nn.Linear(qwen_text_dim, pooled_input_dim)
        self.output_bridges = nn.ModuleDict({
            name: nn.Linear(legacy.control_core.hidden_dim, qwen_hidden_size, bias=False)
            for name in BRANCH_NAMES
        })
        self.loading_report: dict[str, Any] = {}
        self.reset_output_bridges()

    @property
    def condition_injector(self):
        return self.legacy.condition_injector

    @property
    def active_block_mapping(self):
        return LAYER_GROUPS

    def _branch_heads(self, name: str):
        if name in {"geometry", "interaction", "detail"}:
            return getattr(self.legacy.control_core, name + "_branch").zero_heads
        adapter = getattr(self.legacy.control_interface, name + "_adapter")
        return None if adapter is None else adapter.zero_heads

    def reset_output_bridges(self) -> None:
        """Keep exactly one zero boundary; two consecutive zero maps cannot learn."""
        report = {}
        for name, projection in self.output_bridges.items():
            heads = self._branch_heads(name)
            nonzero = heads is not None and any(
                bool(torch.count_nonzero(p.detach())) for p in heads.parameters()
            )
            if nonzero:
                nn.init.zeros_(projection.weight)
                initialization = "zero_projection_nonzero_inherited_heads"
            else:
                nn.init.xavier_uniform_(projection.weight)
                initialization = "xavier_projection_zero_or_absent_heads"
            report[name] = {"initialization": initialization, "heads_nonzero": nonzero}
        self.loading_report["output_initialization"] = report

    @staticmethod
    def _load_state(path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = checkpoint.get("adapter_state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint adapter_state_dict must be a mapping")
        return checkpoint, state

    @classmethod
    def from_checkpoints(
        cls, deepgen_config_path, body_checkpoint, face_checkpoint=None, *, enable_hand: bool = True,
        legacy_kwargs: Mapping[str, Any] | None = None,
    ) -> "Qwen21FullAdapter":
        config_path = Path(deepgen_config_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        config = SimpleNamespace(**json.loads(config_path.read_text()))
        options = {"enable_face_adapter": True, "enable_hand_adapter": enable_hand}
        options.update(dict(legacy_kwargs or {}))
        legacy = UnifiedSMPLXAdapterV6.from_deepgen_config(config, **options)
        model = cls(legacy)
        model.load_legacy_checkpoints(body_checkpoint, face_checkpoint)
        return model

    def load_legacy_checkpoints(self, body_checkpoint, face_checkpoint=None) -> dict[str, Any]:
        expected = self.legacy.state_dict()
        body_meta, body = self._load_state(body_checkpoint)
        new_prefixes = DETAIL_ONLY_PREFIXES + FACE_ONLY_PREFIXES + HAND_ONLY_PREFIXES
        required_body = {key for key in expected if not key.startswith(new_prefixes)}
        if set(body) != required_body:
            raise RuntimeError(
                f"Body checkpoint key mismatch: missing={sorted(required_body-set(body))}, "
                f"unexpected={sorted(set(body)-required_body)}"
            )
        selected = dict(body)
        face_count = 0
        face_ignored = 0
        face_keys = []
        ignored_keys = []
        if face_checkpoint is not None:
            _, face_state = self._load_state(face_checkpoint)
            unknown = set(face_state) - set(expected)
            if unknown:
                raise RuntimeError(f'Unknown Face checkpoint keys cannot be silently ignored: {sorted(unknown)}')
            face = {key: value for key, value in face_state.items() if key.startswith(FACE_ONLY_PREFIXES)}
            required_face = {key for key in expected if key.startswith(FACE_ONLY_PREFIXES)}
            if set(face) != required_face:
                raise RuntimeError(
                    f"Face-only checkpoint key mismatch: missing={sorted(required_face-set(face))}, "
                    f"unexpected={sorted(set(face)-required_face)}"
                )
            selected.update(face)
            face_count, face_ignored = len(face), len(face_state) - len(face)
            face_keys = sorted(face)
            ignored_keys = sorted(set(face_state) - set(face))
        for key, value in selected.items():
            if not torch.is_tensor(value) or value.shape != expected[key].shape:
                raise RuntimeError(f"incompatible inherited tensor: {key}")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite inherited tensor: {key}")
        # Preflight above completes before any state mutation. Dtype conversion to
        # FP32 is deliberate; all inherited bfloat16 values remain exactly representable.
        result = self.legacy.load_state_dict(selected, strict=False)
        allowed_missing = set(expected) - set(selected)
        if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
            raise RuntimeError("unexpected discrepancy after preflighted legacy load")
        self.loading_report = {
            "body_checkpoint": str(body_checkpoint), "body_tensors_loaded": len(body),
            "body_step": body_meta.get("step", body_meta.get("metadata", {}).get("global_step")),
            "face_checkpoint": None if face_checkpoint is None else str(face_checkpoint),
            "face_tensors_loaded": face_count, "face_nonface_tensors_ignored": face_ignored,
            "body_keys_loaded": sorted(body), "face_keys_loaded": face_keys,
            "face_nonface_keys_ignored": ignored_keys,
            "new_legacy_keys": sorted(allowed_missing),
            "dtype_conversion": "inherited tensors copied into FP32 training parameters",
        }
        self.reset_output_bridges()
        return self.loading_report

    def latent_to_legacy(self, latent: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        h, w = map(int, token_hw)
        if latent.ndim != 3 or latent.shape[1:] != (h * w, 64):
            raise ValueError("Qwen latent must have shape [B,h*w,64]")
        value = latent.to(device=self.latent_bridge.weight.device, dtype=self.latent_bridge.weight.dtype)
        value = value.transpose(1, 2).reshape(value.shape[0], 64, h, w)
        return F.pixel_shuffle(self.latent_bridge(value), 2)

    def prepare_conditioning(
        self, *, condition_bundle: ConditionBundle, source_latent: torch.Tensor,
        target_token_hw: tuple[int, int], source_token_hw: tuple[int, int] | None = None,
        source_person_latents: torch.Tensor | None = None, source_indices: torch.Tensor | None = None,
        face_condition=None, face_references=None, hand_condition=None,
        hand_references=None, hand_features=None, detail_condition=None, detail_references=None,
    ) -> PreparedControlConditioning:
        source_hw = target_token_hw if source_token_hw is None else source_token_hw
        source = self.latent_to_legacy(source_latent, source_hw)
        batch = source.shape[0]
        if source_person_latents is None:
            if bool(condition_bundle.person_valid[:, 1].any()):
                raise ValueError("dual-person control requires explicitly bound source_person_latents")
            persons = torch.stack((source, torch.zeros_like(source)), dim=1)
        else:
            if source_person_latents.shape[:2] != (batch, 2):
                raise ValueError("source_person_latents must have shape [B,2,N,64]")
            persons = self.latent_to_legacy(source_person_latents.flatten(0, 1), source_hw)
            persons = persons.reshape(batch, 2, *persons.shape[1:])
        if source_indices is None:
            source_indices = torch.arange(2, device=source.device)[None].expand(batch, -1)
        identity = AdapterIdentityCondition(persons, source_indices=source_indices)
        return self.legacy.prepare_conditioning(
            condition_bundle=condition_bundle, identity_condition=identity,
            source_scene_latents=source,
            target_latent_hw=(target_token_hw[0] * 2, target_token_hw[1] * 2),
            face_condition=face_condition, face_references=face_references,
            hand_condition=hand_condition, hand_references=hand_references,
            hand_features=hand_features, detail_condition=detail_condition, detail_references=detail_references,
        )

    def build_residuals(
        self, *, noisy_target: torch.Tensor, prepared: PreparedControlConditioning,
        prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None,
        sigma: torch.Tensor, target_token_hw: tuple[int, int],
        denoise_progress: float | torch.Tensor | None = None,
        geometry_strength=1.0, interaction_strength=0.8, detail_strength=1.0,
        face_strength=1.0, hand_strength=1.0, hand_mode: str = "legacy",
        face_reference_token_hw: tuple[int, int] | None = None,
    ) -> QwenFullControlOutput:
        if hand_mode not in {"legacy", "v66", "off"}:
            raise ValueError("invalid hand_mode")
        if face_reference_token_hw is not None:
            if (not isinstance(face_reference_token_hw, (tuple, list))
                    or len(face_reference_token_hw) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                           for value in face_reference_token_hw)):
                raise ValueError('face_reference_token_hw must contain two positive integers')
            face_reference_token_hw = tuple(face_reference_token_hw)
        target = self.latent_to_legacy(noisy_target, target_token_hw)
        batch = target.shape[0]
        prepared = prepared.validate().expand_to_batch(batch)
        sigma = sigma.to(target).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.shape != (batch,) or not torch.isfinite(sigma).all() or bool(((sigma < 0) | (sigma > 1)).any()):
            raise ValueError("sigma must be finite, [B], and within [0,1]")
        progress = 1 - sigma if denoise_progress is None else torch.as_tensor(denoise_progress, device=target.device, dtype=target.dtype)
        if progress.ndim == 0:
            progress = progress.expand(batch)
        if progress.shape != (batch,):
            raise ValueError("denoise_progress must be scalar or [B]")
        embeddings = prompt_embeds.to(device=target.device, dtype=self.text_bridge.weight.dtype)
        if embeddings.shape[0] != batch:
            raise ValueError("prompt batch must match target batch")
        mask = torch.ones(embeddings.shape[:2], device=target.device, dtype=torch.bool) if prompt_mask is None else prompt_mask.to(device=target.device, dtype=torch.bool)
        if mask.shape != embeddings.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("every prompt must have at least one valid condition token")
        # Masked positions may contain arbitrary values; remove them before either
        # pooling or attention. The inherited JointAttnProcessor ignores masks.
        valid_embeddings = embeddings.masked_fill(~mask[..., None], 0)
        pooled = self.pooled_bridge(valid_embeddings.sum(1) / mask.sum(1, keepdim=True))
        interface = self.legacy.control_interface
        controller = interface.strength_controller

        def sample_gate(value):
            value = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            if value.ndim == 0:
                return value.expand(batch)
            if value.shape != (batch,):
                raise ValueError("control strength must be scalar or [B]")
            return value

        detail_active = torch.zeros(batch, device=target.device, dtype=torch.bool)
        face_active = torch.zeros_like(detail_active)
        hand_active = torch.zeros_like(detail_active)
        if prepared.detail is not None and hand_mode == "legacy":
            face_valid = prepared.detail.region_valid[:, :, 0].any(dim=1)
            hand_valid = prepared.detail.region_valid[:, :, 1:].any(dim=(1, 2))
            region_active = (face_valid & (sample_gate(face_strength) != 0)) | (hand_valid & (sample_gate(hand_strength) != 0))
            detail_active = prepared.detail.detail_valid & (controller.detail_schedule.multiplier(progress) != 0) & (sample_gate(detail_strength) != 0) & region_active
        if prepared.face is not None:
            face_active = prepared.face.face_valid.any(dim=1) & (controller.face_schedule.multiplier(progress) != 0) & (sample_gate(detail_strength) != 0) & (sample_gate(face_strength) != 0)
        if hand_mode == "v66":
            if prepared.hand is None:
                raise ValueError("v66 hand mode requires prepared hand features")
            hand_active = prepared.hand.hand_valid.any(dim=(1, 2)) & (controller.hand_schedule.multiplier(progress) != 0) & (sample_gate(detail_strength) != 0) & (sample_gate(hand_strength) != 0)
        face_texture_scale = controller.face_texture_schedule.multiplier(progress)
        hand_texture_scale = controller.hand_texture_schedule.multiplier(progress)
        reference_face_scatter = (prepared.face is not None
                                 and face_reference_token_hw is not None
                                 and face_reference_token_hw != tuple(target_token_hw))
        rows = []
        for row in range(batch):
            idx = torch.tensor([row], device=target.device)
            row_prepared = prepared.index_select(idx)
            row_face_active = face_active[row:row + 1]
            # Keep the original Body/other branches on the actual grid. The
            # optional Face-only scatter is built explicitly, with no hooks or
            # mutation of the copied modules or global runtime state.
            raw_row = interface.build_raw_residuals(
                target_latents=target[row:row + 1], prepared=row_prepared,
                encoder_hidden_states=self.text_bridge(embeddings[row:row + 1, mask[row]]),
                pooled_projections=pooled[row:row + 1], timestep=sigma[row:row + 1] * 1000,
                detail_active=detail_active[row:row + 1],
                face_active=torch.zeros_like(row_face_active) if reference_face_scatter else row_face_active,
                face_texture_scale=face_texture_scale[row:row + 1], hand_active=hand_active[row:row + 1],
                hand_texture_scale=hand_texture_scale[row:row + 1], hand_mode=hand_mode,
            )
            if reference_face_scatter and bool(row_face_active.any()):
                reference_face = interface.face_adapter(
                    row_prepared.face, target_token_hw=face_reference_token_hw,
                    timestep=sigma[row:row + 1] * 1000,
                    texture_scale=face_texture_scale[row:row + 1], face_active=row_face_active)
                resized_face = resize_face_residuals_from_reference_grid(
                    reference_face, reference_token_hw=face_reference_token_hw,
                    target_token_hw=target_token_hw, prepared_face=row_prepared.face)
                raw_row = replace(raw_row, face=resized_face).validate()
            rows.append(raw_row)
        branches = {}
        for name in BRANCH_NAMES:
            values = [getattr(row, name) for row in rows]
            if values[0] is None:
                if any(value is not None for value in values):
                    raise RuntimeError("inconsistent optional control branch across rows")
                branches[name] = None
            else:
                branches[name] = tuple(torch.cat([value[g] for value in values], dim=0) for g in range(6))
        raw = BranchControlResiduals(target_token_hw=target_token_hw, **branches).validate()
        projected = BranchControlResiduals(
            target_token_hw=target_token_hw,
            **{name: None if values is None else tuple(self.output_bridges[name](value) for value in values) for name, values in branches.items()},
        ).validate()
        detail_masks = None
        if prepared.detail is not None:
            masks = prepared.detail.region_masks.flatten(0, 2)[:, None]
            masks = F.interpolate(masks, size=target_token_hw, mode="bilinear", align_corners=False)
            detail_masks = masks[:, 0].reshape(batch, 2, 3, *target_token_hw)
        scaled, diagnostics = controller(
            projected, denoise_progress=progress, geometry_strength=geometry_strength,
            interaction_strength=interaction_strength, detail_strength=detail_strength,
            face_strength=face_strength, hand_strength=hand_strength,
            detail_region_masks=detail_masks, hand_mode=hand_mode,
        )
        diagnostics.update({
            "target_token_hw": list(target_token_hw), "active_block_mapping": [list(group) for group in LAYER_GROUPS],
            "face_reference_token_hw": None if face_reference_token_hw is None else list(face_reference_token_hw),
            "face_scatter_resized": reference_face_scatter,
            "detail_active_rows": int(detail_active.sum()), "face_active_rows": int(face_active.sum()),
            "hand_active_rows": int(hand_active.sum()),
            "raw_branch_rms": {name: None if values is None else [float(value.detach().float().square().mean().sqrt()) for value in values] for name, values in branches.items()},
            "residual_rms": [float(value.detach().float().square().mean().sqrt()) for value in scaled],
        })
        return QwenFullControlOutput(scaled, diagnostics, raw)


class Qwen21FullControlContext(AbstractContextManager):
    """Keep alive until backward completes when Qwen block checkpointing is used."""

    def __init__(
        self, transformer: nn.Module, adapter: Qwen21FullAdapter,
        prepared_or_factory: PreparedControlConditioning | Callable[[dict], PreparedControlConditioning],
        *, target_token_hw: tuple[int, int] | None = None,
        progress_mode: str = "sigma", num_inference_steps: int | None = None,
        denoise_progress: float | torch.Tensor | None = None, **strengths,
    ) -> None:
        if progress_mode not in {"sigma", "step"}:
            raise ValueError("progress_mode must be sigma or step")
        if progress_mode == "step" and (num_inference_steps is None or num_inference_steps < 1):
            raise ValueError("step progress requires num_inference_steps")
        self.transformer, self.adapter = transformer, adapter
        self.prepared_or_factory = prepared_or_factory
        self.expected_hw = target_token_hw
        self.progress_mode, self.num_inference_steps = progress_mode, num_inference_steps
        self.denoise_progress = denoise_progress
        self.strengths = strengths
        self.handles = []
        self.output: QwenFullControlOutput | None = None
        self.call_counts = {index: 0 for index in range(32)}
        self.forward_count = 0
        self.step_diagnostics = []
        self.injection_diagnostics = []
        self._step_index = -1
        self._last_sigma = None
        self._prompt = self._prompt_mask = None
        self.target_tokens: int | None = None

    def _before_transformer(self, module, args, kwargs):
        inputs = dict(inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments)
        # Wrappers with **kwargs need flattening; the official forward is explicit.
        if "kwargs" in inputs:
            inputs.update(inputs.pop("kwargs"))
        shapes = inputs.get("img_shapes")
        if not shapes:
            raise ValueError("Qwen control requires native img_shapes")
        if any(sample[-1] != shapes[0][-1] for sample in shapes):
            raise ValueError("all samples must share a target token layout")
        frame, h, w = shapes[0][-1]
        if frame != 1 or min(h, w) <= 0:
            raise ValueError("only single-frame image target layouts are supported")
        if self.expected_hw is not None and (h, w) != tuple(self.expected_hw):
            raise ValueError("native target token grid differs from requested grid")
        self.target_tokens = h * w
        hidden = inputs["hidden_states"]
        if hidden.shape[1] < self.target_tokens:
            raise ValueError("native hidden_states does not contain the target latent")
        if inputs.get("encoder_hidden_states") is not None:
            self._prompt = inputs["encoder_hidden_states"]
            self._prompt_mask = inputs.get("encoder_hidden_states_mask")
        if self._prompt is None:
            raise ValueError("no cached or current Qwen condition embeddings")
        sigma = inputs["timestep"].reshape(-1)
        # CFG may invoke the transformer twice at the same denoising timestep.
        current_sigma = sigma.detach().cpu()
        if self._last_sigma is None or not torch.equal(current_sigma, self._last_sigma):
            self._step_index += 1
            self._last_sigma = current_sigma.clone()
        progress = self.denoise_progress
        if progress is None and self.progress_mode == "step":
            progress = min(self._step_index / max(self.num_inference_steps - 1, 1), 1.0)
        prepared = self.prepared_or_factory(inputs) if callable(self.prepared_or_factory) else self.prepared_or_factory
        self.output = self.adapter.build_residuals(
            noisy_target=hidden[:, -self.target_tokens:], prepared=prepared,
            prompt_embeds=self._prompt, prompt_mask=self._prompt_mask,
            sigma=sigma, target_token_hw=(h, w), denoise_progress=progress, **self.strengths,
        )
        self.step_diagnostics.append({
            'index': self._step_index, 'sigma': current_sigma.tolist(),
            'progress_mode': self.progress_mode,
            'progress': self.output.diagnostics.get('denoise_progress'),
            'face_schedule_multiplier': self.output.diagnostics.get('face_schedule_multiplier'),
            'face_texture_schedule_multiplier': self.output.diagnostics.get('face_texture_schedule_multiplier'),
            'raw_branch_rms': self.output.diagnostics.get('raw_branch_rms'),
            'residual_rms': self.output.diagnostics.get('residual_rms'),
        })
        self.forward_count += 1

    def _after_block(self, layer):
        def inject(module, args, output):
            if self.output is None or self.target_tokens is None:
                raise RuntimeError("control residuals were not prepared before the Qwen block")
            if not torch.is_tensor(output) or output.ndim != 3 or output.shape[-1] != self.adapter.qwen_hidden_size or output.shape[1] < self.target_tokens:
                raise ValueError("unexpected Qwen block output shape")
            self.call_counts[layer] += 1
            delta = self.output.residuals[6 * layer // 32].to(output)
            if delta.shape != output[:, -self.target_tokens:].shape:
                raise ValueError("residual and target token shapes differ")
            prefix = output.shape[1] - self.target_tokens
            if layer in (0, 6, 11, 16, 22, 27):
                # Local statistics avoid diluting a large face update over the full image.
                group = 6 * layer // 32
                face = self.output.raw_residuals.face
                face_projected = None if face is None else self.adapter.output_bridges['face'](face[group]).detach()
                item = {'step': self._step_index, 'layer': layer,
                        'target_hidden_rms': float(output[:, prefix:].detach().float().square().mean().sqrt()),
                        'total_delta_max_abs': float(delta.detach().float().abs().max())}
                if face_projected is not None:
                    support = face_projected.abs().amax(-1) > 0
                    item['face_projected_local_rms'] = float(face_projected[support].float().square().mean().sqrt()) if bool(support.any()) else 0.
                    item['face_projected_max_abs'] = float(face_projected.float().abs().max())
                self.injection_diagnostics.append(item)
            target = output[:, prefix:] + delta
            return target if prefix == 0 else torch.cat((output[:, :prefix], target), dim=1)
        return inject

    def __enter__(self):
        if getattr(self.transformer, "_pic_edit_qwen21_control_active", False):
            raise RuntimeError("another Qwen control context is active")
        if len(self.transformer.transformer_blocks) != 32:
            raise ValueError("Qwen21 full control requires 32 native blocks")
        self.transformer._pic_edit_qwen21_control_active = True
        self.handles.append(self.transformer.register_forward_pre_hook(self._before_transformer, with_kwargs=True))
        for layer, block in enumerate(self.transformer.transformer_blocks):
            self.handles.append(block.register_forward_hook(self._after_block(layer)))
        return self

    def __exit__(self, exception_type, exception, traceback):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.transformer._pic_edit_qwen21_control_active = False
        return False

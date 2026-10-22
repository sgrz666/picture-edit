"""Full's learned geometry frontend without the legacy SD3 control core.

This opt-in endpoint preserves both resolution paths before the old 128-channel
projection. It does not resize conditions, generate Qwen residuals, or load Union.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .condition_injector import SMPLXConditionInjector
from .conditions import ConditionBundle
from .reasoning import AdapterReasoningConfig, InternalControlState, SMPLXAdapterReasoner


SCHEMA = "qwen21_full_native_geometry_v1"
SOURCE_ARCHITECTURE = "qwen21_full_v1"
_FULL_PREFIXES = (
    ("condition_injector.", "legacy.condition_injector."),
    ("reasoner.", "legacy.reasoner."),
    ("geometry_high_downsample.", "legacy.condition_bridge.geometry_high_downsample."),
    ("interaction_high_downsample.", "legacy.condition_bridge.interaction_high_downsample."),
)


@dataclass(frozen=True)
class NativeGeometryFeatures:
    """Row-major target-grid features; person_valid is not a canvas mask."""

    geometry: torch.Tensor
    interaction: torch.Tensor
    target_token_hw: tuple[int, int]
    person_valid: torch.Tensor


class Qwen21NativeGeometry(nn.Module):
    """Independent native-stem frontend. Production width is 512.

    Construct with the original reasoning configuration when migrating a
    non-default Full model. All checkpoint imports create independent FP32 CPU
    parameters; call .to(device) afterwards. Depth is intentionally unsupported
    in this first migration, rather than changing the old fusion input shape.
    """

    def __init__(self, *, reasoning_config: AdapterReasoningConfig | None = None):
        super().__init__()
        config = reasoning_config or AdapterReasoningConfig()
        if not isinstance(config, AdapterReasoningConfig):
            raise ValueError("reasoning_config must be AdapterReasoningConfig")
        self.condition_injector = SMPLXConditionInjector(use_depth=False, normal_backend="native")
        self.reasoner = SMPLXAdapterReasoner(config)
        self.geometry_high_downsample = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1, bias=False)
        self.interaction_high_downsample = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1, bias=False)
        self.inheritance_report: dict = {}

    def get_extra_state(self):
        return {"schema": SCHEMA, "reasoning_config": asdict(self.reasoner.config),
                "normal_backend": "native", "use_depth": False, "grid_policy": "exact"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Native geometry schema/config does not match this model")

    @classmethod
    def from_full_adapter(cls, full):
        """Warm-start from a Full object without sharing parameters or buffers."""
        legacy = getattr(full, "legacy", None)
        if legacy is None or not hasattr(full, "qwen_hidden_size"):
            raise ValueError("Expected Qwen21FullAdapter, not a simplified or legacy adapter")
        if legacy.condition_injector.use_depth:
            raise ValueError("Depth-enabled Full requires a separate migration; this endpoint keeps depth off")
        if legacy.condition_injector.normal_backend != "native":
            raise ValueError("Only the existing native normal stem is supported")
        result = cls.from_full_checkpoint(
            {"architecture": SOURCE_ARCHITECTURE, "adapter_state_dict": full.state_dict()},
            reasoning_config=legacy.reasoner.config,
        )
        result.train(full.training)
        return result

    @classmethod
    def from_full_checkpoint(cls, checkpoint: Mapping, *, reasoning_config: AdapterReasoningConfig | None = None):
        """Import the four frontend prefixes from a loaded Full checkpoint.

        No old Full/SD3 model is constructed. The caller owns file loading (and
        may use mmap=True); optimizer/RNG state is ignored, never resumed.
        Unknown keys *within* the selected prefixes are rejected, as are missing,
        nonfinite, non-floating, or incompatible tensors. Other Full branches
        are intentionally excluded and counted in the report.
        """
        if not isinstance(checkpoint, Mapping) or checkpoint.get("architecture") != SOURCE_ARCHITECTURE:
            raise ValueError(f"Expected source architecture {SOURCE_ARCHITECTURE}")
        state = checkpoint.get("adapter_state_dict")
        if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
            raise ValueError("Full checkpoint must contain an adapter_state_dict mapping with string keys")
        # A caller may be inside init_empty_weights / a meta-device context.
        # Explicitly materialize this small frontend before copying learned
        # weights; load_state_dict into meta parameters would silently do nothing.
        with torch.device("cpu"):
            result = cls(reasoning_config=reasoning_config).float()
        expected = {key: value for key, value in result.state_dict().items() if key != "_extra_state"}
        selected = {}
        for key, value in state.items():
            for new_prefix, old_prefix in _FULL_PREFIXES:
                if key.startswith(old_prefix):
                    selected[new_prefix + key[len(old_prefix):]] = value
                    break
        missing, extra = sorted(expected.keys() - selected.keys()), sorted(selected.keys() - expected.keys())
        if missing or extra:
            raise ValueError(f"Full frontend keys differ: missing={missing}, extra={extra}")
        for key, value in selected.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
                raise ValueError(f"Full frontend tensor shape differs: {key}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Full frontend tensor must be finite floating point: {key}")
        result.load_state_dict({**selected, "_extra_state": result.get_extra_state()}, strict=True)
        result.inheritance_report = {
            "source_architecture": SOURCE_ARCHITECTURE,
            "schema": SCHEMA,
            "loaded_tensor_count": len(selected),
            "loaded_element_count": sum(value.numel() for value in selected.values()),
            "parameter_count": sum(parameter.numel() for parameter in result.parameters()),
            "ignored_key_count": len(state) - len(selected),
            "source_prefixes": tuple(old for _, old in _FULL_PREFIXES),
        }
        return result

    def forward(self, bundle: ConditionBundle, *, target_token_hw: tuple[int, int]) -> NativeGeometryFeatures:
        return self.project_state(self.reasoner(bundle), target_token_hw=target_token_hw)

    def project_state(self, state: InternalControlState, *, target_token_hw: tuple[int, int]) -> NativeGeometryFeatures:
        """Fuse native-grid low/high state without interpolation or masking background.

        The caller must supply conditions already transformed into the target
        canvas. Equal grid sizes are necessary, not proof of geometric alignment.
        """
        if (not isinstance(target_token_hw, (tuple, list)) or len(target_token_hw) != 2
                or any(type(size) is not int or size <= 0 for size in target_token_hw)):
            raise ValueError("target_token_hw must contain two positive integers")
        target_token_hw = tuple(target_token_hw)
        state.validate()
        actual_hw = tuple(state.geometry_feature.shape[-2:])
        if actual_hw != target_token_hw or target_token_hw[0] * target_token_hw[1] % 4:
            raise ValueError(f"Native grid {actual_hw} must equal target_token_hw={target_token_hw} with four-token Qwen slots; no implicit resize")
        if state.geometry_feature.shape[1] != self.reasoner.config.hidden_dim:
            raise ValueError("Native geometry width differs from reasoning configuration")
        for name in ("geometry_feature", "geometry_highres", "interaction_feature", "interaction_highres"):
            value = getattr(state, name)
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating point features")
        geometry = state.geometry_feature + self.geometry_high_downsample(state.geometry_highres)
        interaction = state.interaction_feature + self.interaction_high_downsample(state.interaction_highres)
        interaction = interaction * state.interaction_valid[:, None, None, None].to(interaction.dtype)
        return NativeGeometryFeatures(
            geometry.flatten(2).transpose(1, 2).contiguous(),
            interaction.flatten(2).transpose(1, 2).contiguous(),
            target_token_hw,
            state.person_valid.clone(),
        )

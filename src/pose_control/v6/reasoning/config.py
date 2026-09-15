from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterReasoningConfig:
    """Widths and attention layout for Adapter-internal reasoning."""

    condition_dim: int = 256
    hidden_dim: int = 512
    reasoning_downsample_factor: int = 2

    global_attention_heads: int = 8
    global_attention_layers: int = 1
    cross_person_heads: int = 8
    cross_person_layers: int = 2
    contact_attention_heads: int = 8
    contact_attention_layers: int = 1

    ffn_ratio: int = 4
    dropout: float = 0.0
    contact_gate_init: float = -4.0

    def __post_init__(self) -> None:
        if self.condition_dim != 256:
            raise ValueError("ConditionBundle fixes condition_dim at 256")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.reasoning_downsample_factor != 2:
            raise ValueError("reasoning V1 supports downsample factor 2 only")
        for name in (
            "global_attention_heads",
            "cross_person_heads",
            "contact_attention_heads",
        ):
            heads = getattr(self, name)
            if heads <= 0 or self.hidden_dim % heads:
                raise ValueError(f"hidden_dim must be divisible by {name}")
        for name in (
            "global_attention_layers",
            "cross_person_layers",
            "contact_attention_layers",
            "ffn_ratio",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

"""Optional frozen Qwen-Image-2.1 I2I hand LoRA and its target-only attention residual."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_PROJECTIONS = {"query": "to_q", "key": "to_k", "value": "to_v", "output": "to_out.0"}
_OFFICIAL_WIDTH, _OFFICIAL_LAYERS, _OFFICIAL_RANK = 4096, 32, 16


class PriorProjection(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)

    def forward(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.up(self.down(value.to(self.down.weight.dtype))).to(value.dtype)


class HandPrior(nn.Module):
    def __init__(self, *, width, layers, rank, sha256):
        super().__init__()
        self.width = width
        self.rank = rank
        self.sha256 = sha256
        self.blocks = nn.ModuleList(nn.ModuleDict({name: PriorProjection(width, rank) for name in _PROJECTIONS})
                                    for _ in range(layers))

    def provenance(self):
        return dict(sha256=self.sha256, width=self.width, layers=len(self.blocks), rank=self.rank, alpha=self.rank,
                    projections=list(_PROJECTIONS.values()), schedule="linear_clean_1_minus_sigma",
                    routing="target_hand_token_coverage")


def load_hand_prior(path, *, sha256):
    """Load only SHA-pinned official rank-16/alpha-16 Q/K/V/O weights as a separate frozen module."""
    from safetensors.torch import load

    if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Expected a lowercase SHA256 for the Qwen hand prior")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Qwen hand prior SHA256 differs from the pinned artifact")
    header_end = 8 + int.from_bytes(payload[:8], "little")
    try:
        header = json.loads(payload[8:header_end])
        metadata = json.loads(header["__metadata__"]["lora_adapter_metadata"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Qwen hand prior must include official LoRA safetensors metadata") from exc
    if (not isinstance(metadata, dict) or metadata.get("transformer.r") != _OFFICIAL_RANK
            or metadata.get("transformer.lora_alpha") != _OFFICIAL_RANK
            or set(metadata.get("transformer.target_modules") or ()) != set(_PROJECTIONS.values())
            or any(metadata.get(f"transformer.{name}") for name in
                   ("alpha_pattern", "rank_pattern", "use_rslora", "use_dora", "use_qalora", "lora_bias",
                    "lora_dropout", "fan_in_fan_out"))
            or metadata.get("transformer.bias", "none") != "none"):
        raise ValueError("Qwen hand prior metadata must describe standard rank-16/alpha-16 Q/K/V/O LoRA")

    tensors = load(payload)
    expected = {f"transformer.transformer_blocks.{i}.attn.{projection}.lora_{side}.weight":
                (f"blocks.{i}.{name}.{member}.weight", shape)
                for i in range(_OFFICIAL_LAYERS) for name, projection in _PROJECTIONS.items()
                for side, member, shape in (("A", "down", (_OFFICIAL_RANK, _OFFICIAL_WIDTH)),
                                            ("B", "up", (_OFFICIAL_WIDTH, _OFFICIAL_RANK)))}
    if expected.keys() != tensors.keys():
        raise ValueError("Qwen hand prior must contain Q/K/V/O A and B matrices for all 32 transformer layers")
    state = {}
    for key, (name, shape) in expected.items():
        value = tensors[key]
        if (value.shape != shape or value.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid Qwen hand prior tensor {key}")
        state[name] = value
    with torch.random.fork_rng(devices=()):
        prior = HandPrior(width=_OFFICIAL_WIDTH, layers=_OFFICIAL_LAYERS, rank=_OFFICIAL_RANK, sha256=sha256)
    prior.load_state_dict(state, strict=True)
    return prior.requires_grad_(False).eval()


def prior_attention_delta(attention, block, hidden, positions, gains, projected, rotary, key_valid):
    """Recompute attention for ROI query rows with gated Q/K/V and add the gated O update."""
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    def split(value):
        return value.unflatten(-1, (attention.heads, -1))

    def rotate(value, table):
        return value if table is None else apply_rotary_emb_qwen(value, table, use_real=False)

    base_query, base_key, base_value = (projected[name] for name in ("query", "key", "value"))
    query_table = None if rotary is None else rotary[positions]
    roi_hidden = hidden[:, positions]
    gain = gains[None, :, None].to(base_value)
    query = base_query + block["query"](roi_hidden) * gain
    key = base_key[:, positions] + block["key"](roi_hidden) * gain
    value = base_value[:, positions] + block["value"](roi_hidden) * gain
    original_query = rotate(attention.norm_q(split(base_query)).to(base_value.dtype), query_table)
    adapted_query = rotate(attention.norm_q(split(query)).to(base_value.dtype), query_table)
    original_key = rotate(attention.norm_k(split(base_key)).to(base_value.dtype), rotary)
    adapted_keys = rotate(attention.norm_k(split(key)).to(base_value.dtype), query_table)
    mask = None if key_valid is None else key_valid[:, None, None, :]

    def attend(q, k, v):
        result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                attn_mask=mask)
        return result.transpose(1, 2).flatten(2)

    original = attend(original_query, original_key, split(base_value))
    adapted = attend(adapted_query, original_key.index_copy(1, positions, adapted_keys),
                     split(base_value.index_copy(1, positions, value)))
    return F.linear(adapted - original, attention.to_out[0].weight) + block["output"](adapted) * gain

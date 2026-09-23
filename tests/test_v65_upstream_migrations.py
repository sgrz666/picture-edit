from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import math
from pathlib import Path
import sys
import types

import torch


if importlib.util.find_spec("diffusers") is None:
    diffusers = types.ModuleType("diffusers")
    diffusers_models = types.ModuleType("diffusers.models")
    diffusers.__spec__ = ModuleSpec("diffusers", loader=None)
    diffusers_models.__spec__ = ModuleSpec("diffusers.models", loader=None)

    class _SD3ControlNetModel(torch.nn.Module):
        pass

    diffusers_models.SD3ControlNetModel = _SD3ControlNetModel
    diffusers.models = diffusers_models
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.models"] = diffusers_models


def test_runtime_modules_directly_reuse_pinned_upstream_adaptations() -> None:
    from third_party.v65_face.mcld import FacePoseGuider2D
    from third_party.v65_face.stableanimator import FacePerceiver
    from third_party.v65_face.visual_persona import (
        IPAttnProcessor2_0,
        Resampler,
    )
    from src.pose_control.v6.face.adapter import FaceAttentionBlock
    from src.pose_control.v6.face.content_encoder import FacePerceiver as RuntimeFacePerceiver
    from src.pose_control.v6.face.resampler import Resampler as RuntimeResampler
    from src.pose_control.v6.face.spatial import FaceSpatialEncoder

    assert RuntimeResampler is Resampler
    assert RuntimeFacePerceiver is FacePerceiver
    assert issubclass(FaceSpatialEncoder, FacePoseGuider2D)
    assert issubclass(FaceAttentionBlock, IPAttnProcessor2_0)


def test_visual_persona_attention_matches_pinned_core_equations() -> None:
    from third_party.v65_face.visual_persona import PerceiverAttention

    torch.manual_seed(101)
    module = PerceiverAttention(dim=32, dim_head=8, heads=4).eval()
    context = torch.randn(2, 7, 32)
    latents = torch.randn(2, 3, 32)

    actual = module(context, latents)
    normalized_context = module.norm1(context)
    normalized_latents = module.norm2(latents)
    query = module.to_q(normalized_latents)
    key, value = module.to_kv(
        torch.cat((normalized_context, normalized_latents), dim=-2)
    ).chunk(2, dim=-1)

    def reshape(value: torch.Tensor) -> torch.Tensor:
        return value.view(2, value.shape[1], 4, 8).transpose(1, 2)

    query, key, value = map(reshape, (query, key, value))
    stable_scale = 1.0 / math.sqrt(math.sqrt(module.dim_head))
    weights = torch.softmax(
        ((query * stable_scale) @ (key * stable_scale).transpose(-2, -1)).float(),
        dim=-1,
    ).to(query.dtype)
    expected = module.to_out(
        (weights @ value).transpose(1, 2).reshape(2, 3, 32)
    )
    torch.testing.assert_close(actual, expected)


def test_stableanimator_fusion_keeps_shortcut_and_zero_delta_contract() -> None:
    from third_party.v65_face.stableanimator import FusionFaceId

    torch.manual_seed(103)
    fusion = FusionFaceId(
        cross_attention_dim=32,
        id_embeddings_dim=16,
        clip_embeddings_dim=24,
        num_tokens=4,
        heads=4,
        depth=1,
    ).eval()
    identity = torch.randn(2, 16)
    appearance = torch.randn(2, 8, 24)
    projected = fusion.project_identity(identity)
    delta = fusion.fusion_model(projected, appearance)
    assert torch.count_nonzero(delta) == 0
    torch.testing.assert_close(
        fusion(identity, appearance, shortcut=True),
        projected,
    )


def test_xdyna_cfg_order_and_optional_residual_sum() -> None:
    from third_party.v65_face.xdyna import (
        add_optional_face_residual,
        repeat_condition_batch,
    )

    condition = torch.tensor([[1.0], [2.0]])
    torch.testing.assert_close(
        repeat_condition_batch(condition, 2),
        torch.tensor([[1.0], [2.0], [1.0], [2.0]]),
    )
    body = torch.tensor([1.0, 2.0])
    face = torch.tensor([0.25, 0.50])
    torch.testing.assert_close(add_optional_face_residual(body, face), body + face)
    assert add_optional_face_residual(body, None) is body


def test_vendored_sources_record_fixed_commits_and_license_status() -> None:
    root = Path(__file__).resolve().parents[1] / "third_party" / "v65_face"
    provenance = (root / "UPSTREAM_SOURCES.md").read_text(encoding="utf-8")
    assert "020e7f23a768410e0183ce943397d96a609f4ec1" in provenance
    assert "d393d09284cac20570466ae3c7ad137b10648bdc" in provenance
    assert "6279ff7c601c11dff5e075410ed34df2973da088" in provenance
    assert "9a54f8e9b90c195eb1f21641c791896bcefe4ce0" in provenance
    assert "no repository-level license" in provenance
    assert (root / "licenses" / "STABLEANIMATOR-MIT.txt").is_file()
    assert (root / "licenses" / "X-DYNA-APACHE-2.0.txt").is_file()

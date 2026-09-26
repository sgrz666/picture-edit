import hashlib
import pytest
import torch
import torch.nn as nn

from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.checkpoint import freeze_for_detail_training
from src.pose_control.v6.detail.conditions import FaceHandDetailCondition, DetailReferenceBatch
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType


def compute_frozen_sha256(adapter: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(adapter.named_parameters()):
        if not param.requires_grad:
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_detail_zero_heads_exact_match_step_0():
    torch.manual_seed(42)
    adapter = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    freeze_for_detail_training(adapter)

    # Ensure zero heads are zero
    for head in adapter.control_core.detail_branch.zero_heads:
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

    # 32x32 inputs downsampled by 8 -> 4x4 spatial (even)
    B = 1
    H, W = 32, 32
    target_latents = torch.randn(B, 16, H, W)
    ref_latents = torch.randn(B, 16, H, W)
    source_person_latents = torch.zeros(B, 2, 16, H, W)
    source_person_latents[:, 0] = ref_latents
    identity = AdapterIdentityCondition(source_person_latents=source_person_latents)

    # Condition Injector builds ConditionBundle with bounded inputs
    bundle = adapter.condition_injector(
        normal_a=torch.zeros(B, 3, H, W),
        pose_heatmap_a=torch.zeros(B, 25, H, W),
        part_onehot_a=torch.zeros(B, 14, H, W),
        smplx_global_a=torch.zeros(B, 26),
        human_mask_a=torch.ones(B, 1, H, W),
        task_id=torch.tensor([int(TaskType.SINGLE)]),
    )

    # Detail condition
    face_kps = torch.zeros(B, 2, 68, 3)
    face_kps[:, 0, :, :2] = 0.5
    face_kps[:, 0, :, 2] = 1.0

    hand_kps = torch.zeros(B, 2, 2, 21, 3)
    hand_kps[:, 0, :, :, :2] = 0.5
    hand_kps[:, 0, :, 2] = 1.0

    detail_cond = FaceHandDetailCondition(
        face_keypoints=face_kps,
        hand_keypoints=hand_kps,
        smplx_detail=torch.zeros(B, 2, 103),
        source_boxes=torch.tensor([[[[0.1, 0.1, 0.3, 0.3]] * 3, [[0.0, 0.0, 0.0, 0.0]] * 3]]),
        target_boxes=torch.tensor([[[[0.2, 0.2, 0.4, 0.4]] * 3, [[0.0, 0.0, 0.0, 0.0]] * 3]]),
        region_valid=torch.tensor([[[True, True, True], [False, False, False]]]),
        source_indices=torch.tensor([[0, 1]]),
    )

    ref_images = torch.zeros(B, 2, 3, 1, 3, 224, 224)
    ref_valid = torch.zeros(B, 2, 3, 1, dtype=torch.bool)
    ref_valid[:, 0, :, 0] = True
    detail_refs = DetailReferenceBatch(
        images=ref_images,
        reference_valid=ref_valid,
    )

    # 1. Prepare conditioning with detail
    prep_on = adapter.prepare_conditioning(
        condition_bundle=bundle,
        identity_condition=identity,
        source_scene_latents=ref_latents,
        target_latent_hw=(H, W),
        detail_condition=detail_cond,
        detail_references=detail_refs,
    )

    # 2. Prepare conditioning without detail
    prep_off = adapter.prepare_conditioning(
        condition_bundle=bundle,
        identity_condition=identity,
        source_scene_latents=ref_latents,
        target_latent_hw=(H, W),
        detail_condition=None,
        detail_references=None,
    )

    common_args = dict(
        target_latents=target_latents,
        cond_hidden_states=[[ref_latents[0]]],
        encoder_hidden_states=torch.randn(B, 5, 24),
        pooled_projections=torch.randn(B, 20),
        timestep=torch.tensor([500.0]),
        denoise_progress=torch.tensor([0.5]),
        geometry_strength=1.0,
        interaction_strength=0.0,
    )

    out_on = adapter(prepared=prep_on, detail_strength=1.0, **common_args)
    out_off = adapter(prepared=prep_off, detail_strength=0.0, **common_args)

    for h_on, h_off in zip(out_on.block_controlnet_hidden_states, out_off.block_controlnet_hidden_states):
        torch.testing.assert_close(h_on, h_off, atol=0.0, rtol=0.0)


def test_gradient_isolation_and_hash():
    torch.manual_seed(42)
    adapter = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    trainable = freeze_for_detail_training(adapter)

    hash_before = compute_frozen_sha256(adapter)

    B = 1
    H, W = 32, 32
    target_latents = torch.randn(B, 16, H, W)
    ref_latents = torch.randn(B, 16, H, W)
    source_person_latents = torch.zeros(B, 2, 16, H, W)
    source_person_latents[:, 0] = ref_latents
    identity = AdapterIdentityCondition(source_person_latents=source_person_latents)

    bundle = adapter.condition_injector(
        normal_a=torch.zeros(B, 3, H, W),
        pose_heatmap_a=torch.zeros(B, 25, H, W),
        part_onehot_a=torch.zeros(B, 14, H, W),
        smplx_global_a=torch.zeros(B, 26),
        human_mask_a=torch.ones(B, 1, H, W),
        task_id=torch.tensor([int(TaskType.SINGLE)]),
    )

    detail_cond = FaceHandDetailCondition(
        face_keypoints=torch.full((B, 2, 68, 3), 0.5),
        hand_keypoints=torch.full((B, 2, 2, 21, 3), 0.5),
        smplx_detail=torch.zeros(B, 2, 103),
        source_boxes=torch.tensor([[[[0.1, 0.1, 0.3, 0.3]] * 3, [[0.0, 0.0, 0.0, 0.0]] * 3]]),
        target_boxes=torch.tensor([[[[0.2, 0.2, 0.4, 0.4]] * 3, [[0.0, 0.0, 0.0, 0.0]] * 3]]),
        region_valid=torch.tensor([[[True, True, True], [False, False, False]]]),
        source_indices=torch.tensor([[0, 1]]),
    )

    ref_images = torch.zeros(B, 2, 3, 1, 3, 224, 224)
    ref_valid = torch.zeros(B, 2, 3, 1, dtype=torch.bool)
    ref_valid[:, 0, :, 0] = True
    detail_refs = DetailReferenceBatch(
        images=ref_images,
        reference_valid=ref_valid,
    )

    prep = adapter.prepare_conditioning(
        condition_bundle=bundle,
        identity_condition=identity,
        source_scene_latents=ref_latents,
        target_latent_hw=(H, W),
        detail_condition=detail_cond,
        detail_references=detail_refs,
    )

    out = adapter(
        target_latents=target_latents,
        prepared=prep,
        cond_hidden_states=[[ref_latents[0]]],
        encoder_hidden_states=torch.randn(B, 5, 24),
        pooled_projections=torch.randn(B, 20),
        timestep=torch.tensor([500.0]),
        denoise_progress=torch.tensor([0.5]),
        geometry_strength=1.0,
        interaction_strength=0.0,
        detail_strength=1.0,
    )

    loss = sum(h.square().sum() for h in out.block_controlnet_hidden_states)
    loss.backward()

    # Verify gradients
    for name, param in adapter.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Trainable param {name} has None grad!"
            assert torch.isfinite(param.grad).all(), f"Trainable param {name} has non-finite grad!"
        else:
            assert param.grad is None, f"Frozen param {name} received gradient!"

    # Verify SHA-256
    hash_after = compute_frozen_sha256(adapter)
    assert hash_before == hash_after, "Frozen parameters SHA-256 changed!"

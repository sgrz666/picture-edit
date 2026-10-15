"""Smoke tests for the TikTok V6.5 Face Adapter training pipeline.

All tests use synthetic data and the tiny-adapter builder so they can run
on CPU without DeepGen weights, real TikTok images, or face-evaluator
checkpoints.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "diffusers" not in sys.modules:
    try:
        _spec = importlib.util.find_spec("diffusers")
    except ValueError:
        _spec = None
    if _spec is None:
        _d = types.ModuleType("diffusers")
        _dm = types.ModuleType("diffusers.models")
        _d.__spec__ = ModuleSpec("diffusers", loader=None)
        _dm.__spec__ = ModuleSpec("diffusers.models", loader=None)

        class _SD3ControlNetModel(nn.Module):
            pass

        _dm.SD3ControlNetModel = _SD3ControlNetModel
        _d.models = _dm
        sys.modules["diffusers"] = _d
        sys.modules["diffusers.models"] = _dm

# ──────────────────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────────────────
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.face.conditions import FaceFineCondition, FaceReferenceFeatures
from src.pose_control.v6.detail.conditions import DetailReferenceBatch, FaceHandDetailCondition
from src.pose_control.v6.detail.hand import HandDetailCondition, hand_to_legacy_detail


class TestTikTokTrainingImports(unittest.TestCase):
    """Verify all public symbols from the TikTok training script are importable."""

    def test_imports(self):
        from train_tiktok_v65_face import (
            get_cached_conditions,
            parse_args,
            run_training,
        )
        # These are re-exported from iPER training script
        from train_tiktok_v65_face import (
            FP32MasterAdamW,
            build_face_optimizer,
            compute_v65_face_losses,
            make_fixed_training_state,
            rasterize_box_mask,
        )
        self.assertTrue(callable(get_cached_conditions))
        self.assertTrue(callable(parse_args))
        self.assertTrue(callable(run_training))
        self.assertTrue(callable(compute_v65_face_losses))


class TestParseArgs(unittest.TestCase):
    """Verify the TikTok CLI parses correctly."""

    def test_minimal_args(self):
        from train_tiktok_v65_face import parse_args

        args = parse_args([
            "--sequence", "00001",
            "--source-stem", "0014",
            "--target-stem", "0074",
        ])
        self.assertEqual(args.sequence, "00001")
        self.assertEqual(args.source_stem, "0014")
        self.assertEqual(args.target_stem, "0074")
        self.assertEqual(args.stage, "overfit")
        self.assertEqual(args.max_steps, 1500)
        self.assertEqual(args.warmup_steps, 1000)

    def test_custom_args(self):
        from train_tiktok_v65_face import parse_args

        args = parse_args([
            "--sequence", "00042",
            "--source-stem", "0001",
            "--target-stem", "0050",
            "--max-steps", "100",
            "--warmup-steps", "20",
            "--resolution", "256",
            "--dtype", "fp32",
            "--device", "cpu",
            "--disable-face-id-loss",
            "--disable-face-perceptual-loss",
            "--disable-face-lpips-metric",
            "--disable-face-landmark-metric",
        ])
        self.assertEqual(args.sequence, "00042")
        self.assertEqual(args.max_steps, 100)
        self.assertEqual(args.warmup_steps, 20)
        self.assertEqual(args.resolution, 256)
        self.assertEqual(args.dtype, "fp32")
        self.assertTrue(args.disable_face_id_loss)
        self.assertTrue(args.disable_face_perceptual_loss)


class TestGetCachedConditions(unittest.TestCase):
    """Test get_cached_conditions with a mock DeepGen pipeline."""

    def _make_mock_pipe(self, *, num_returns: int = 4):
        pipe = mock.MagicMock(spec=["pixels_to_latents", "encode_prompt"])
        ref = torch.randn(1, 16, 64, 64)
        pipe.pixels_to_latents.return_value = ref

        seq = torch.randn(1, 8, 24)
        pooled = torch.randn(1, 20)

        if num_returns == 4:
            # SD3-style: (embeds, neg_embeds, pooled, neg_pooled)
            pipe.encode_prompt.return_value = (
                seq,
                torch.zeros_like(seq),
                pooled,
                torch.zeros_like(pooled),
            )
        elif num_returns == 2:
            pipe.encode_prompt.return_value = (seq, pooled)
        return pipe, ref, seq, pooled

    def test_sd3_style_four_returns(self):
        from train_tiktok_v65_face import get_cached_conditions

        pipe, expected_ref, expected_seq, expected_pooled = self._make_mock_pipe(num_returns=4)
        batch = {"src_image": torch.randn(1, 3, 512, 512)}
        device = torch.device("cpu")
        dtype = torch.float32

        with tempfile.TemporaryDirectory() as tmpdir:
            ref, seq, pooled = get_cached_conditions(
                pipe, batch, "test prompt", tmpdir, device, dtype
            )
            self.assertEqual(ref.shape, expected_ref.shape)
            self.assertEqual(seq.shape, expected_seq.shape)
            self.assertEqual(pooled.shape, expected_pooled.shape)
            torch.testing.assert_close(ref, expected_ref)
            torch.testing.assert_close(seq, expected_seq)
            torch.testing.assert_close(pooled, expected_pooled)

    def test_two_returns(self):
        from train_tiktok_v65_face import get_cached_conditions

        pipe, _, expected_seq, expected_pooled = self._make_mock_pipe(num_returns=2)
        batch = {"src_image": torch.randn(1, 3, 512, 512)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref, seq, pooled = get_cached_conditions(
                pipe, batch, "test prompt", tmpdir, torch.device("cpu"), torch.float32
            )
            torch.testing.assert_close(seq, expected_seq)
            torch.testing.assert_close(pooled, expected_pooled)

    def test_disk_cache_roundtrip(self):
        """Second call must load from cache, not re-encode."""
        from train_tiktok_v65_face import get_cached_conditions

        pipe, _, _, _ = self._make_mock_pipe(num_returns=4)
        batch = {"src_image": torch.randn(1, 3, 512, 512)}
        device = torch.device("cpu")
        dtype = torch.float32

        with tempfile.TemporaryDirectory() as tmpdir:
            ref1, seq1, pooled1 = get_cached_conditions(
                pipe, batch, "test prompt", tmpdir, device, dtype
            )
            # Reset call counts
            pipe.pixels_to_latents.reset_mock()
            pipe.encode_prompt.reset_mock()

            ref2, seq2, pooled2 = get_cached_conditions(
                pipe, batch, "test prompt", tmpdir, device, dtype
            )
            # Should have loaded from cache, not called encode again
            pipe.pixels_to_latents.assert_not_called()
            pipe.encode_prompt.assert_not_called()
            torch.testing.assert_close(ref1, ref2, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(seq1, seq2, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(pooled1, pooled2, atol=2e-2, rtol=1e-2)

    def test_missing_encode_prompt_raises(self):
        from train_tiktok_v65_face import get_cached_conditions

        pipe = mock.MagicMock(spec=[])
        pipe.pixels_to_latents = mock.MagicMock(return_value=torch.randn(1, 16, 64, 64))
        del pipe.encode_prompt

        batch = {"src_image": torch.randn(1, 3, 512, 512)}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                get_cached_conditions(pipe, batch, "prompt", tmpdir, torch.device("cpu"), torch.float32)


class TestFaceLossComputation(unittest.TestCase):
    """Verify compute_v65_face_losses shapes and gradient flow."""

    def test_losses_shape_and_gradient(self):
        from train_tiktok_v65_face import compute_v65_face_losses

        prediction = torch.randn(1, 16, 64, 64, requires_grad=True)
        target = torch.randn(1, 16, 64, 64)
        face_mask = torch.zeros(1, 1, 64, 64)
        face_mask[:, :, 20:40, 20:40] = 1.0

        losses = compute_v65_face_losses(prediction, target, face_mask)
        self.assertIn("total", losses)
        self.assertIn("flow", losses)
        self.assertIn("face_id", losses)
        self.assertIn("face_perceptual", losses)
        self.assertIn("outside", losses)
        self.assertEqual(losses["total"].ndim, 0)
        losses["total"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(prediction.grad.abs().sum() > 0)

    def test_outside_reference_constraint(self):
        from train_tiktok_v65_face import compute_v65_face_losses

        prediction = torch.randn(1, 16, 8, 8, requires_grad=True)
        target = torch.randn(1, 16, 8, 8)
        face_mask = torch.zeros(1, 1, 8, 8)
        face_mask[:, :, 2:6, 2:6] = 1.0
        outside_ref = torch.randn(1, 16, 8, 8)

        losses = compute_v65_face_losses(
            prediction, target, face_mask, outside_reference=outside_ref
        )
        self.assertGreater(float(losses["outside"].detach()), 0.0)


class TestFixedTrainingState(unittest.TestCase):
    """Verify determinism of fixed training state."""

    def test_determinism(self):
        from train_tiktok_v65_face import make_fixed_training_state

        shape = (1, 16, 64, 64)
        s1 = make_fixed_training_state(shape, seed=42, max_timestep=650, fixed_timestep=150)
        s2 = make_fixed_training_state(shape, seed=42, max_timestep=650, fixed_timestep=150)
        torch.testing.assert_close(s1["noise"], s2["noise"])
        torch.testing.assert_close(s1["timestep"], s2["timestep"])

    def test_different_seeds(self):
        from train_tiktok_v65_face import make_fixed_training_state

        shape = (1, 16, 64, 64)
        s1 = make_fixed_training_state(shape, seed=42, max_timestep=650)
        s2 = make_fixed_training_state(shape, seed=99, max_timestep=650)
        self.assertFalse(torch.allclose(s1["noise"], s2["noise"]))


class TestRasterizeBoxMask(unittest.TestCase):
    """Verify face mask rasterization."""

    def test_box_mask_shape(self):
        from train_tiktok_v65_face import rasterize_box_mask

        boxes = torch.tensor([[0.2, 0.3, 0.8, 0.7]])  # xyxy normalized
        mask = rasterize_box_mask(boxes, height=64, width=64)
        self.assertEqual(mask.shape, (1, 1, 64, 64))
        self.assertTrue(mask.sum() > 0)

    def test_empty_box(self):
        from train_tiktok_v65_face import rasterize_box_mask

        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # degenerate box
        mask = rasterize_box_mask(boxes, height=32, width=32)
        self.assertEqual(mask.shape, (1, 1, 32, 32))


class TestTinyAdapterGradientFlow(unittest.TestCase):
    """End-to-end gradient flow through tiny adapter with face branch enabled."""

    def _build_synthetic_inputs(self, B: int = 1, img_size: int = 32, latent_size: int = 8):
        """Create synthetic TikTok-shaped inputs for the tiny adapter."""
        adapter = UnifiedSMPLXAdapterV6.build_tiny(
            hidden_dim=32,
            geometry_channels=16,
            context_input_dim=24,
            pooled_input_dim=20,
            num_heads=4,
            condition_use_depth=False,
            enable_face_adapter=True,
            face_dim=32,
        )

        # Geometry conditions
        normal_a = torch.randn(B, 3, img_size, img_size).clamp(-1.0, 1.0)
        pose_heatmap_a = torch.rand(B, 25, img_size, img_size)
        part_onehot_a = torch.zeros(B, 14, img_size, img_size)
        part_onehot_a[:, 0, :, :img_size // 2] = 1.0
        smplx_global_a = torch.randn(B, 26)
        human_mask_a = torch.ones(B, 1, img_size, img_size)
        task_id = torch.tensor([int(TaskType.SINGLE)], dtype=torch.long)

        bundle = adapter.condition_injector(
            normal_a=normal_a,
            pose_heatmap_a=pose_heatmap_a,
            part_onehot_a=part_onehot_a,
            smplx_global_a=smplx_global_a,
            human_mask_a=human_mask_a,
            task_id=task_id,
        )

        # Identity condition
        ref_latent = torch.randn(B, 16, latent_size, latent_size)
        src_person_latents = torch.zeros(B, 2, 16, latent_size, latent_size)
        src_person_latents[:, 0] = ref_latent
        identity = AdapterIdentityCondition(
            source_person_latents=src_person_latents,
            source_indices=torch.tensor([[0, 1]]),
        )

        # Hand detail (stub)
        hand_keypoints = torch.zeros(B, 2, 2, 21, 3)
        hand_pose = torch.zeros(B, 2, 2, 45)
        hand_boxes_src = torch.zeros(B, 2, 2, 4)
        hand_boxes_tgt = torch.zeros_like(hand_boxes_src)
        hand_valid = torch.zeros(B, 2, 2, dtype=torch.bool)
        hand_cond = hand_to_legacy_detail(
            HandDetailCondition(
                hand_keypoints=hand_keypoints,
                hand_pose=hand_pose,
                source_boxes=hand_boxes_src,
                target_boxes=hand_boxes_tgt,
                region_valid=hand_valid,
                source_indices=torch.tensor([[0, 1]]),
            ).validate()
        ).validate()
        hand_refs = DetailReferenceBatch(
            images=torch.zeros(B, 2, 3, 1, 3, 224, 224),
            reference_valid=torch.zeros(B, 2, 3, 1, dtype=torch.bool),
        ).validate()

        # Face condition
        landmarks = torch.zeros(B, 2, 72, 3)
        landmarks[0, 0, :, 0] = torch.linspace(0.3, 0.7, 72)
        landmarks[0, 0, :, 1] = torch.linspace(0.2, 0.6, 72)
        landmarks[0, 0, :, 2] = 1.0
        face_cond = FaceFineCondition(
            landmarks=landmarks,
            jaw_pose=torch.zeros(B, 2, 3),
            expression=torch.zeros(B, 2, 10),
            source_boxes=torch.tensor([[0.25, 0.15, 0.75, 0.65]]).unsqueeze(0).expand(B, 2, 4).clone(),
            target_boxes=torch.tensor([[0.25, 0.15, 0.75, 0.65]]).unsqueeze(0).expand(B, 2, 4).clone(),
            face_valid=torch.tensor([[True, False]]),
            source_indices=torch.tensor([[0, 1]]),
        ).validate()
        face_refs = FaceReferenceFeatures(
            arcface=torch.randn(B, 2, 1, 512, dtype=torch.float16),
            dino_patches=torch.randn(B, 2, 1, 256, 1536, dtype=torch.float16),
            reference_valid=torch.tensor([[[True], [False]]]),
        ).validate()

        return (
            adapter,
            bundle,
            identity,
            ref_latent,
            hand_cond,
            hand_refs,
            face_cond,
            face_refs,
            latent_size,
        )

    def test_forward_backward_with_face(self):
        """Verify gradient flow through geometry + face branches."""
        from train_tiktok_v65_face import (
            compute_v65_face_losses,
            rasterize_box_mask,
            select_face_trainable_parameters,
        )

        (
            adapter,
            bundle,
            identity,
            ref_latent,
            hand_cond,
            hand_refs,
            face_cond,
            face_refs,
            latent_size,
        ) = self._build_synthetic_inputs()

        # Freeze everything except face
        selected = select_face_trainable_parameters(adapter, phase="full")
        self.assertGreater(len(selected), 0, "must select face trainable parameters")

        prepared = adapter.prepare_conditioning(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=ref_latent,
            target_latent_hw=(latent_size, latent_size),
            detail_condition=hand_cond,
            detail_references=hand_refs,
            face_condition=face_cond,
            face_references=face_refs,
        )

        target_latents = torch.randn(1, 16, latent_size, latent_size)
        encoder_hidden_states = torch.randn(1, 4, 24)
        pooled_projections = torch.randn(1, 20)
        timesteps = torch.tensor([500.0])

        output = adapter(
            target_latents=target_latents,
            prepared=prepared,
            cond_hidden_states=None,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
            denoise_progress=0.5,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=1.0,
        )

        self.assertEqual(len(output.block_controlnet_hidden_states), 6)

        # Compute linear loss on residuals so zero-initialized heads receive non-zero upstream gradient
        loss = sum(r.sum() for r in output.block_controlnet_hidden_states)
        if adapter.control_interface.strength_controller.face_log_group_scale is not None:
            loss = loss + adapter.control_interface.strength_controller.face_log_group_scale.sum()
        loss.backward()

        # Check face adapter received gradients
        face_params_with_grad = [
            name
            for name, p in adapter.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        ]
        self.assertGreater(
            len(face_params_with_grad),
            0,
            "face parameters must receive non-zero gradients",
        )

    def test_face_strength_zero_ablation(self):
        """face_strength=0 should produce identical residuals to no-face adapter."""
        (
            adapter,
            bundle,
            identity,
            ref_latent,
            hand_cond,
            hand_refs,
            face_cond,
            face_refs,
            latent_size,
        ) = self._build_synthetic_inputs()

        prepared = adapter.prepare_conditioning(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=ref_latent,
            target_latent_hw=(latent_size, latent_size),
            detail_condition=hand_cond,
            detail_references=hand_refs,
            face_condition=face_cond,
            face_references=face_refs,
        )

        target_latents = torch.randn(1, 16, latent_size, latent_size)
        encoder_hidden_states = torch.randn(1, 4, 24)
        pooled = torch.randn(1, 20)
        ts = torch.tensor([500.0])

        with torch.no_grad():
            out_on = adapter(
                target_latents=target_latents,
                prepared=prepared,
                cond_hidden_states=None,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=ts,
                denoise_progress=0.5,
                face_strength=1.0,
            )
            out_off = adapter(
                target_latents=target_latents,
                prepared=prepared,
                cond_hidden_states=None,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=ts,
                denoise_progress=0.5,
                face_strength=0.0,
            )

        # Zero-initialized face heads → face_strength=1.0 should still be all zeros initially
        for on_r, off_r in zip(
            out_on.block_controlnet_hidden_states,
            out_off.block_controlnet_hidden_states,
        ):
            torch.testing.assert_close(
                on_r, off_r,
                msg="With zero-initialized face heads, ON and OFF must be identical at initialization",
            )


class TestFP32MasterAdamW(unittest.TestCase):
    """Verify FP32MasterAdamW maintains precision."""

    def test_basic_step(self):
        from train_tiktok_v65_face import FP32MasterAdamW

        param = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW(
            [{"params": [param], "lr": 1e-3}],
            weight_decay=0.01,
        )

        # Simulate a gradient step
        param.grad = torch.randn_like(param)
        initial = param.detach().clone()
        optimizer.step()
        self.assertFalse(torch.allclose(param, initial), "parameter must change after step")
        self.assertEqual(param.dtype, torch.bfloat16)

    def test_zero_grad(self):
        from train_tiktok_v65_face import FP32MasterAdamW

        param = nn.Parameter(torch.randn(4, dtype=torch.float32))
        optimizer = FP32MasterAdamW(
            [{"params": [param], "lr": 1e-3}],
            weight_decay=0.0,
        )
        param.grad = torch.ones_like(param)
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()

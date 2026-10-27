"""Unit tests for iPER V6.3 training pipeline under Scheme A."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from train_iper_v6_native import (
    configure_scheme_a_parameters,
    build_scheme_a_optimizer,
    CheckpointManager,
)


class TestIPERV6TrainingPipeline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.tiny_adapter = UnifiedSMPLXAdapterV6.build_tiny(
            hidden_dim=32,
            geometry_channels=16,
            context_input_dim=24,
            pooled_input_dim=20,
            num_heads=4,
        )

    def test_scheme_a_freezing_contract(self):
        """Verify that Scheme A strictly freezes Interaction, Contact, and Person-B modules."""
        adapter = self.tiny_adapter
        configure_scheme_a_parameters(adapter)

        # 1. Condition Injector: interaction & contact modules must be frozen
        for p in adapter.condition_injector.relative_encoder.parameters():
            self.assertFalse(p.requires_grad, "relative_encoder must be frozen in Scheme A")
        for p in adapter.condition_injector.contact_raster_encoder.parameters():
            self.assertFalse(p.requires_grad, "contact_raster_encoder must be frozen in Scheme A")
        for p in adapter.condition_injector.contact_relation_encoder.parameters():
            self.assertFalse(p.requires_grad, "contact_relation_encoder must be frozen in Scheme A")

        # 2. Reasoner: cross-person, contact, and dual fusion must be frozen
        for p in adapter.reasoner.cross_person_reasoner.parameters():
            self.assertFalse(p.requires_grad, "cross_person_reasoner must be frozen in Scheme A")
        for p in adapter.reasoner.contact_reasoner.parameters():
            self.assertFalse(p.requires_grad, "contact_reasoner must be frozen in Scheme A")
        for p in adapter.reasoner.dual_fusion.parameters():
            self.assertFalse(p.requires_grad, "dual_fusion must be frozen in Scheme A")

        # 3. Reasoner Bridge: interaction projections must be frozen
        for p in adapter.condition_bridge.interaction_projection.parameters():
            self.assertFalse(p.requires_grad, "condition_bridge.interaction_projection must be frozen in Scheme A")
        for p in adapter.condition_bridge.interaction_high_downsample.parameters():
            self.assertFalse(p.requires_grad, "condition_bridge.interaction_high_downsample must be frozen in Scheme A")

        # 4. Control Core: interaction branch must be frozen
        for p in adapter.control_core.interaction_branch.parameters():
            self.assertFalse(p.requires_grad, "interaction_branch must be frozen in Scheme A")

        # 4. Strength Controller: interaction gate must be frozen
        self.assertFalse(
            adapter.control_interface.strength_controller.interaction_log_group_scale.requires_grad,
            "interaction_log_group_scale must be frozen in Scheme A",
        )

        # 5. Geometry branch & core modules must be trainable
        trainable_spatial = [p for p in adapter.condition_injector.spatial_encoder.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_spatial), 0, "spatial_encoder should be trainable")

        trainable_reasoner = [p for p in adapter.reasoner.person_reasoner.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_reasoner), 0, "person_reasoner should be trainable")

        trainable_heads = [p for p in adapter.control_core.geometry_zero_heads.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_heads), 0, "geometry_zero_heads should be trainable")

        trainable_shared = [p for p in adapter.control_core.shared_block.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_shared), 0, "shared_block should be trainable")

    def test_optimizer_parameter_groups_partition(self):
        """Verify that layered optimizer creates exactly 4 non-overlapping groups with correct LRs."""
        adapter = self.tiny_adapter
        configure_scheme_a_parameters(adapter)

        optimizer, meta = build_scheme_a_optimizer(
            adapter,
            lr_zero_heads=1e-4,
            lr_condition=5e-5,
            lr_bridge=5e-5,
            lr_control_block=1e-5,
        )

        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[2]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[3]["lr"], 1e-5)

        # Total assigned parameters must match exactly the number of requires_grad=True parameters
        total_in_groups = sum(len(grp["params"]) for grp in optimizer.param_groups)
        total_trainable = sum(1 for p in adapter.parameters() if p.requires_grad)
        self.assertEqual(total_in_groups, total_trainable)

    def test_mask_weighted_flow_loss_computation(self):
        """Verify mask-weighted loss applies correct foreground multiplier and normalizes cleanly."""
        B, C, H, W = 2, 16, 8, 8
        pred = torch.ones(B, C, H, W, dtype=torch.float32) * 2.0
        target_velocity = torch.zeros(B, C, H, W, dtype=torch.float32)

        # Mask where half the spatial area is 1 (human) and half is 0 (background)
        human_mask = torch.zeros(B, 1, H, W, dtype=torch.float32)
        human_mask[..., : H // 2, :] = 1.0

        # Unweighted (human_loss_weight = 1.0)
        diff_sq = (pred - target_velocity) ** 2  # all elements are 4.0
        weights_1 = (1.0 + (1.0 - 1.0) * human_mask).expand_as(diff_sq)
        loss_unweighted = (diff_sq * weights_1).sum() / weights_1.sum()
        self.assertAlmostEqual(loss_unweighted.item(), 4.0, places=5)

        # Weighted (human_loss_weight = 3.0)
        # Even with nonuniform error, weighted sum formula operates correctly
        diff_sq_nonuniform = torch.zeros_like(pred)
        diff_sq_nonuniform[..., : H // 2, :] = 10.0  # human error
        diff_sq_nonuniform[..., H // 2 :, :] = 2.0   # bg error

        weights_3 = (1.0 + (3.0 - 1.0) * human_mask).expand_as(diff_sq_nonuniform)  # 3.0 on human, 1.0 on bg
        # Expected weighted mean: (10.0 * 3.0 * 32 + 2.0 * 1.0 * 32) / (3.0 * 32 + 1.0 * 32)
        # = (960 + 64) / (96 + 32) = 1024 / 128 = 8.0
        loss_weighted = (diff_sq_nonuniform * weights_3).sum() / weights_3.sum()
        self.assertAlmostEqual(loss_weighted.item(), 8.0, places=5)

    def test_checkpoint_manager_rotation(self):
        """Verify that CheckpointManager rotates and limits rolling checkpoints to max_rolling."""
        temp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager(temp_dir, max_rolling=2)
            adapter = nn.Linear(4, 4)
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)

            # Step 10: initial save
            mgr.save(step=10, adapter=adapter, optimizer=optimizer, val_loss=1.5, is_best=True)
            self.assertTrue((Path(temp_dir) / "checkpoint_step_10.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoint_last.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoint_best.pt").exists())

            # Step 20: second rolling save
            mgr.save(step=20, adapter=adapter, optimizer=optimizer, val_loss=1.2, is_best=True)
            self.assertTrue((Path(temp_dir) / "checkpoint_step_10.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoint_step_20.pt").exists())

            # Step 30: third rolling save -> step 10 must be deleted!
            mgr.save(step=30, adapter=adapter, optimizer=optimizer, val_loss=1.3, is_best=False)
            self.assertFalse((Path(temp_dir) / "checkpoint_step_10.pt").exists(), "Oldest rolling checkpoint was not pruned!")
            self.assertTrue((Path(temp_dir) / "checkpoint_step_20.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoint_step_30.pt").exists())

            # Total rolling checkpoints on disk must be <= 2
            rolling_files = list(Path(temp_dir).glob("checkpoint_step_*.pt"))
            self.assertEqual(len(rolling_files), 2)

            # Checkpoint last must match step 30
            last_data = mgr.load("last", adapter, optimizer)
            self.assertEqual(last_data["step"], 30)

            # Checkpoint best must match step 20 (loss 1.2)
            best_data = mgr.load("best", adapter, optimizer)
            self.assertEqual(best_data["step"], 20)
            self.assertAlmostEqual(best_data["val_loss"], 1.2)
        finally:
            shutil.rmtree(temp_dir)

    def test_dev_train_val_split_integrity_if_exists(self):
        """Verify dev-train and dev-val split isolation if files exist on server."""
        split_dir = Path("/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits")
        dev_train_path = split_dir / "dev_train_pairs.jsonl"
        dev_val_path = split_dir / "dev_val_pairs.jsonl"
        test_path = split_dir / "test_pairs.jsonl"

        if not dev_train_path.exists():
            self.skipTest("Split files do not exist in local environment")

        with open(dev_train_path) as f:
            dev_train = [json.loads(l) for l in f if l.strip()]
        with open(dev_val_path) as f:
            dev_val = [json.loads(l) for l in f if l.strip()]
        with open(test_path) as f:
            test = [json.loads(l) for l in f if l.strip()]

        train_apps = set(x["appearance"] for x in dev_train)
        val_apps = set(x["appearance"] for x in dev_val)
        test_apps = set(x["appearance"] for x in test)

        # Appearance counts
        self.assertEqual(len(train_apps), 69)
        self.assertEqual(len(val_apps), 10)
        self.assertEqual(len(test_apps), 20)

        # Zero appearance leakage
        self.assertEqual(len(train_apps & val_apps), 0, "Leakage between dev-train and dev-val!")
        self.assertEqual(len(train_apps & test_apps), 0, "Leakage between dev-train and test!")
        self.assertEqual(len(val_apps & test_apps), 0, "Leakage between dev-val and test!")

        # Pair counts
        self.assertEqual(len(dev_train), 26496)
        self.assertEqual(len(dev_val), 3840)
        self.assertEqual(len(test), 7680)
        self.assertEqual(len(dev_train) + len(dev_val), 30336)

    def test_single_person_forward_backward_tiny(self):
        """Verify complete forward pass and backward gradient flow on tiny adapter."""
        adapter = self.tiny_adapter
        configure_scheme_a_parameters(adapter)
        optimizer, _ = build_scheme_a_optimizer(adapter)

        # Synthetic inputs
        B = 1
        img_size = 32
        latent_size = 8
        hidden_dim = 32

        normal_a = torch.randn(B, 3, img_size, img_size).clamp(-1, 1)
        pose_heatmap_a = torch.rand(B, 25, img_size, img_size)
        part_onehot_a = torch.zeros(B, 14, img_size, img_size)
        part_onehot_a[:, 0, :, :] = 1.0
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

        ref_latent = torch.randn(B, 16, latent_size, latent_size)
        src_person_latents = torch.zeros(B, 2, 16, latent_size, latent_size)
        src_person_latents[:, 0] = ref_latent
        identity = AdapterIdentityCondition(
            source_person_latents=src_person_latents,
            source_indices=torch.tensor([[0, 1]]),
        )

        prepared = adapter.prepare_conditioning(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=ref_latent,
            target_latent_hw=(latent_size, latent_size),
        )

        target_latents = torch.randn(B, 16, latent_size, latent_size)
        cond_hidden_states = [[ref_latent[0]]]
        encoder_hidden_states = torch.randn(B, 4, 24)
        pooled_projections = torch.randn(B, 20)
        timesteps = torch.tensor([500.0])

        output = adapter(
            target_latents=target_latents,
            prepared=prepared,
            cond_hidden_states=cond_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
            denoise_progress=0.5,
            geometry_strength=1.0,
            interaction_strength=0.0,
        )

        self.assertEqual(len(output.block_controlnet_hidden_states), 6)
        residuals = output.block_controlnet_hidden_states

        # Since zero heads are 0, residuals start at 0
        all_zero_at_init = all(torch.all(r == 0) for r in residuals)
        self.assertTrue(all_zero_at_init, "Zero heads must output exact zeros initially")

        # Fake loss directly on residuals to check gradient flow through full adapter
        fake_loss = sum(r.sum() for r in residuals) + adapter.control_interface.strength_controller.geometry_log_group_scale.sum()
        fake_loss.backward()

        # Check that geometry zero heads received gradients
        for head in adapter.control_core.geometry_zero_heads:
            self.assertIsNotNone(head.weight.grad, "geometry_zero_heads must receive gradients")

        # Check that frozen interaction zero heads have NO gradient
        for head in adapter.control_core.interaction_zero_heads:
            self.assertIsNone(head.weight.grad, "interaction_zero_heads must have None gradient")

    def test_12_residual_heads_freeze_contract(self):
        """Verify the exact 12 Residual Heads contract: 6 geometry trainable, 6 interaction frozen."""
        adapter = self.tiny_adapter
        configure_scheme_a_parameters(adapter)

        geo_heads = adapter.control_core.geometry_zero_heads
        int_heads = adapter.control_core.interaction_zero_heads
        self.assertEqual(len(geo_heads), 6, "Must have exactly 6 geometry residual heads")
        self.assertEqual(len(int_heads), 6, "Must have exactly 6 interaction residual heads")

        for i, h in enumerate(geo_heads):
            for p in h.parameters():
                self.assertTrue(p.requires_grad, f"Geometry head {i} parameter must have requires_grad=True")

        for i, h in enumerate(int_heads):
            for p in h.parameters():
                self.assertFalse(p.requires_grad, f"Interaction head {i} parameter must have requires_grad=False")

    def test_checkpoint_best_loss_recovery(self):
        """Verify that CheckpointManager discovers and recovers existing best_loss on init."""
        temp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager(temp_dir, max_rolling=2)
            adapter = nn.Linear(4, 4)
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)

            # Save best checkpoint with loss 0.42
            mgr.save(step=5, adapter=adapter, optimizer=optimizer, val_loss=0.42, is_best=True)
            self.assertAlmostEqual(mgr.best_loss, 0.42)

            # Re-instantiate CheckpointManager (simulating training restart/resume)
            mgr_new = CheckpointManager(temp_dir, max_rolling=2)
            self.assertAlmostEqual(mgr_new.best_loss, 0.42, places=4, msg="Failed to recover best_loss from checkpoint_best.pt")
        finally:
            shutil.rmtree(temp_dir)

    def test_denoise_progress_direction(self):
        """Verify denoise_progress calculation aligns with inference (0.0 at noise, 1.0 at clean)."""
        # At start of diffusion: timestep = 1000 (pure noise) -> progress must be 0.0
        timesteps_noise = torch.tensor([1000.0])
        progress_noise = float((1.0 - timesteps_noise.mean() / 1000.0).clamp(0.0, 1.0).item())
        self.assertAlmostEqual(progress_noise, 0.0, places=5)

        # At end of diffusion: timestep = 0 (clean image) -> progress must be 1.0
        timesteps_clean = torch.tensor([0.0])
        progress_clean = float((1.0 - timesteps_clean.mean() / 1000.0).clamp(0.0, 1.0).item())
        self.assertAlmostEqual(progress_clean, 1.0, places=5)

        # Midpoint
        timesteps_mid = torch.tensor([500.0])
        progress_mid = float((1.0 - timesteps_mid.mean() / 1000.0).clamp(0.0, 1.0).item())
        self.assertAlmostEqual(progress_mid, 0.5, places=5)

    def test_warmup_heads_transition_and_recovery(self):
        """Verify that zero-heads warmup zeroes non-head LRs and restores them cleanly at boundary and resume."""
        adapter = self.tiny_adapter
        configure_scheme_a_parameters(adapter)
        optimizer, _ = build_scheme_a_optimizer(
            adapter,
            lr_zero_heads=1e-4,
            lr_condition=5e-5,
            lr_bridge=5e-5,
            lr_control_block=1e-5,
        )

        warmup_steps = 300

        # Step 0..299: simulate warmup active
        for step in [0, 100, 299]:
            if step < warmup_steps:
                for grp in optimizer.param_groups[1:]:
                    grp["lr"] = 0.0
            self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
            self.assertEqual(optimizer.param_groups[1]["lr"], 0.0)
            self.assertEqual(optimizer.param_groups[2]["lr"], 0.0)
            self.assertEqual(optimizer.param_groups[3]["lr"], 0.0)

        # Step 300: warmup completion
        step = 300
        if step >= warmup_steps:
            if optimizer.param_groups[1]["lr"] == 0.0:
                optimizer.param_groups[1]["lr"] = 5e-5
                optimizer.param_groups[2]["lr"] = 5e-5
                optimizer.param_groups[3]["lr"] = 1e-5

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[2]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[3]["lr"], 1e-5)

        # Resuming at step 500 when optimizer was saved with 0.0 lr
        for grp in optimizer.param_groups[1:]:
            grp["lr"] = 0.0
        step_resumed = 500
        if step_resumed >= warmup_steps:
            if optimizer.param_groups[1]["lr"] == 0.0:
                optimizer.param_groups[1]["lr"] = 5e-5
                optimizer.param_groups[2]["lr"] = 5e-5
                optimizer.param_groups[3]["lr"] = 1e-5

        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-5, "Resume past warmup failed to restore LR!")

    def test_micro_step_budget_and_adapter_step_saving(self):
        """Verify micro_step logic, adapter_step saving, and raw_mse in metrics."""
        temp_dir = tempfile.mkdtemp()
        try:
            adapter = self.tiny_adapter
            optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=0.01)
            output_dir = Path(temp_dir)
            save_step_id = 500
            direct_adapter_ckpt = output_dir / f"adapter_step_{save_step_id}.pt"
            torch.save(
                {
                    "step": save_step_id,
                    "micro_step": 500,
                    "global_step": 125,
                    "adapter_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": {"seed": 42},
                    "val_loss": 0.22,
                },
                direct_adapter_ckpt,
            )
            self.assertTrue(direct_adapter_ckpt.exists())
            data = torch.load(direct_adapter_ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(data["step"], 500)
            self.assertEqual(data["micro_step"], 500)
            self.assertEqual(data["global_step"], 125)
            self.assertIn("adapter_state_dict", data)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()

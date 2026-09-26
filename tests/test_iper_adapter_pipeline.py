"""Unit test for iPER adapter pipeline, token alignment, and gradient flow."""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adapter.pose_adapter import PoseConditionAdapter
from src.integration.champ_deepgen import align_control_residuals
from src.data.iper_dataset import IPERPoseDataset


class TestIPERAdapterPipeline(unittest.TestCase):

    def setUp(self):
        self.resolution = 512
        self.target_tokens = (self.resolution // 16) ** 2  # 1024
        self.reference_tokens = (self.resolution // 16) ** 2  # 1024
        self.hidden_dim = 1536
        self.adapter = PoseConditionAdapter(
            in_channels=8,
            hidden_dim=self.hidden_dim,
            num_injection_layers=6,
            num_blocks=4,
        )

    def test_adapter_initialization_zero_conv(self):
        """Verify zero-conv initialization produces exact zeros at init."""
        dummy_control = torch.randn(1, 8, self.resolution, self.resolution)
        residuals = self.adapter(dummy_control)

        self.assertEqual(len(residuals), 6)
        for i, res in enumerate(residuals):
            self.assertEqual(res.shape, (1, self.target_tokens, self.hidden_dim))
            max_abs = res.abs().max().item()
            self.assertEqual(max_abs, 0.0, f"Layer {i} residual is not zero initialized!")

    def test_token_alignment_zero_padding(self):
        """Verify align_control_residuals pads reference token positions with zeros."""
        dummy_control = torch.randn(1, 8, self.resolution, self.resolution)
        # Give non-zero weights for testing alignment
        with torch.no_grad():
            for zc in self.adapter.zero_convs:
                zc.weight.fill_(1.0)

        residuals = self.adapter(dummy_control)
        aligned = align_control_residuals(
            residuals,
            target_tokens=self.target_tokens,
            reference_tokens=self.reference_tokens,
            batch_size=1,
        )

        self.assertEqual(len(aligned), 6)
        total_tokens = self.target_tokens + self.reference_tokens  # 2048
        for i, res in enumerate(aligned):
            self.assertEqual(res.shape, (1, total_tokens, self.hidden_dim))
            # Target tokens (0 to 1024) should have signal
            target_part = res[:, :self.target_tokens, :]
            # Reference tokens (1024 to 2048) must be strictly zero
            ref_part = res[:, self.target_tokens:, :]
            self.assertEqual(ref_part.abs().max().item(), 0.0,
                             f"Reference tokens at layer {i} are not zero-padded!")
            self.assertGreater(target_part.abs().max().item(), 0.0,
                               f"Target tokens at layer {i} have no signal!")

    def test_adapter_gradient_flow(self):
        """Verify gradients flow backward through all adapter layers and zero_convs."""
        dummy_control = torch.randn(1, 8, self.resolution, self.resolution, requires_grad=True)
        residuals = self.adapter(dummy_control)
        aligned = align_control_residuals(
            residuals,
            target_tokens=self.target_tokens,
            reference_tokens=self.reference_tokens,
            batch_size=1,
        )

        loss = sum(r.sum() for r in aligned)
        loss.backward()

        # Check adapter parameter gradients
        for name, param in self.adapter.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} received no gradient!")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf!")

    def test_iper_dataset_loading_if_exists(self):
        """Verify IPERPoseDataset loads a sample with correct 8-channel control map and RGB shapes."""
        train_pairs = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits/train_pairs.jsonl"
        sampled_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt"
        assets_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"
        if not os.path.exists(train_pairs):
            self.skipTest(f"{train_pairs} does not exist in this environment")

        dataset = IPERPoseDataset(
            pairs_jsonl=train_pairs,
            sampled_root=sampled_root,
            assets_root=assets_root,
            resolution=self.resolution,
            augment=False,
        )
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(sample["src_image"].shape, (3, self.resolution, self.resolution))
        self.assertEqual(sample["tgt_image"].shape, (3, self.resolution, self.resolution))
        self.assertEqual(sample["control_map"].shape, (8, self.resolution, self.resolution))
        self.assertIn("appearance", sample)
        self.assertIn("source_stem", sample)
        self.assertIn("target_stem", sample)


if __name__ == "__main__":
    unittest.main()


"""Tests for iPER V6.3 evaluation parity and metric comparability against 130.28M baseline."""

import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from eval_iper_v6 import compute_metrics


class TestIPERV6EvalParity(unittest.TestCase):

    def test_compute_metrics_numerical_consistency(self):
        """Verify that compute_metrics returns exact PSNR, SSIM, and MAE matching baseline formulas."""
        # Create synthetic images
        arr1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
        arr2 = np.ones((64, 64, 3), dtype=np.uint8) * 120
        img1 = Image.fromarray(arr1)
        img2 = Image.fromarray(arr2)

        metrics = compute_metrics(img1, img2)
        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("ssim_skimage", metrics)

        # Difference is exactly 20.0 everywhere
        self.assertAlmostEqual(metrics["mae"], 20.0, places=4)
        self.assertAlmostEqual(metrics["mse"], 400.0, places=4)
        expected_psnr = 10.0 * np.log10(255.0 ** 2 / 400.0)
        self.assertAlmostEqual(metrics["psnr"], expected_psnr, places=4)

        # Baseline grayscale formula:
        # mu1 = 100, mu2 = 120, sig1 = 0, sig2 = 0, sig_gt = 0
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        expected_ssim = (2 * 100 * 120 + c1) * c2 / ((100**2 + 120**2 + c1) * c2)
        self.assertAlmostEqual(metrics["ssim"], expected_ssim, places=4)

    def test_baseline_50_eval_pairs_matches_baseline_json(self):
        """Verify baseline_50_eval_pairs.jsonl matches exactly the 50 evaluated samples in baseline eval_metrics.json."""
        pairs_file = Path("/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl")
        metrics_file = Path("/home/shangguanrz/project/pic-edit/experiments/iper_adapter_finetune_v1/eval_results/eval_metrics.json")

        if not pairs_file.exists() or not metrics_file.exists():
            self.skipTest("Files only exist on remote server")

        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        with open(metrics_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        self.assertEqual(len(pairs), 50)
        per_sample = base_data["per_sample"]
        self.assertEqual(len(per_sample), 50)

        for i in range(50):
            self.assertEqual(pairs[i]["appearance"], per_sample[i]["appearance"])
            self.assertEqual(pairs[i]["source"]["stem"], per_sample[i]["source_stem"])
            self.assertEqual(pairs[i]["target"]["stem"], per_sample[i]["target_stem"])


if __name__ == "__main__":
    unittest.main()

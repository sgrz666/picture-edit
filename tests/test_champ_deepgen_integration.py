import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.integration.champ_deepgen import (
    align_control_residuals,
    image_diagnostics,
    load_champ_sample,
)


def _write_rgb(path: Path, channels: tuple[int, int, int], size: int = 16) -> None:
  array = np.zeros((size, size, 3), dtype=np.uint8)
  for index, value in enumerate(channels):
    array[:, :, index] = value
  Image.fromarray(array, mode="RGB").save(path)


def _write_gray(path: Path, value: int, size: int = 16) -> None:
  Image.fromarray(np.full((size, size), value, dtype=np.uint8), mode="L").save(path)


def _write_natural_rgb(path: Path, size: int = 16) -> None:
  yy, xx = np.mgrid[:size, :size]
  array = np.stack(
      [
          (xx * 13 + yy * 3) % 256,
          (xx * 5 + yy * 17 + 31) % 256,
          (xx * 19 + yy * 7 + 63) % 256,
      ],
      axis=-1,
  ).astype(np.uint8)
  Image.fromarray(array, mode="RGB").save(path)


class ChampSampleTests(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)
    _write_natural_rgb(self.root / "real_src_frame_0.png")
    _write_natural_rgb(self.root / "real_tgt_frame_60.png")
    _write_rgb(self.root / "ctrl_normal.png", (10, 20, 30))
    _write_gray(self.root / "ctrl_depth.png", 40)
    _write_rgb(self.root / "ctrl_openpose.png", (50, 60, 70))
    _write_gray(self.root / "ctrl_semantic.png", 80)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_loads_eight_channels_in_fixed_order(self):
    sample = load_champ_sample(self.root, resolution=16)

    self.assertEqual(tuple(sample.control.shape), (1, 8, 16, 16))
    means = sample.control.mean(dim=(0, 2, 3))
    expected = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80]) / 255.0
    torch.testing.assert_close(means, expected)
    self.assertEqual(sample.source.size, (16, 16))
    self.assertEqual(sample.target.size, (16, 16))

  def test_rejects_binary_mask_as_target_rgb(self):
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[:, 8:, :] = 255
    Image.fromarray(mask, mode="RGB").save(self.root / "real_tgt_frame_60.png")

    with self.assertRaisesRegex(ValueError, "target RGB looks like a mask"):
      load_champ_sample(self.root, resolution=16)

  def test_reports_missing_control_path(self):
    missing = self.root / "ctrl_depth.png"
    missing.unlink()

    with self.assertRaisesRegex(FileNotFoundError, "ctrl_depth.png"):
      load_champ_sample(self.root, resolution=16)


class ResidualAlignmentTests(unittest.TestCase):

  def test_pads_only_reference_tokens(self):
    residuals = [torch.ones(1, 4, 3), torch.full((1, 4, 3), 2.0)]

    aligned = align_control_residuals(
        residuals,
        target_tokens=4,
        reference_tokens=4,
        batch_size=1,
    )

    self.assertEqual(tuple(aligned[0].shape), (1, 8, 3))
    torch.testing.assert_close(aligned[0][:, :4], torch.ones(1, 4, 3))
    torch.testing.assert_close(aligned[0][:, 4:], torch.zeros(1, 4, 3))
    torch.testing.assert_close(aligned[1][:, :4], torch.full((1, 4, 3), 2.0))

  def test_rejects_wrong_batch(self):
    with self.assertRaisesRegex(ValueError, "expected batch 1"):
      align_control_residuals(
          [torch.ones(2, 4, 3)],
          target_tokens=4,
          reference_tokens=4,
          batch_size=1,
      )

  def test_rejects_wrong_target_token_count(self):
    with self.assertRaisesRegex(ValueError, r"expected \[B,4,D\]"):
      align_control_residuals(
          [torch.ones(1, 5, 3)],
          target_tokens=4,
          reference_tokens=4,
          batch_size=1,
      )


class ImageDiagnosticsTests(unittest.TestCase):

  def test_flags_near_white_output_as_invalid(self):
    image = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8), mode="RGB")

    result = image_diagnostics(image)

    self.assertFalse(result["valid"])
    self.assertGreater(result["near_white_ratio"], 0.95)

  def test_accepts_non_degenerate_output(self):
    yy, xx = np.mgrid[:16, :16]
    array = np.stack([xx * 16, yy * 16, (xx + yy) * 8], axis=-1).astype(np.uint8)

    result = image_diagnostics(Image.fromarray(array, mode="RGB"))

    self.assertTrue(result["valid"])
    self.assertGreater(result["std"], 5.0)


class RunnerContractTests(unittest.TestCase):

  def test_runner_uses_formal_pipeline_control_interface(self):
    source = Path("run_champ_single_overfit.py").read_text(encoding="utf-8")

    self.assertNotIn("transformer.forward =", source)
    self.assertIn("block_controlnet_hidden_states=aligned_residuals", source)
    self.assertIn("load_champ_sample", source)
    self.assertIn("align_control_residuals", source)

  def test_runner_supports_validation_without_loading_model(self):
    source = Path("run_champ_single_overfit.py").read_text(encoding="utf-8")

    validation_guard = source.index("if args.validate_only:")
    model_load = source.index("DiffusionPipeline.from_pretrained")
    self.assertLess(validation_guard, model_load)

  def test_runner_records_numerical_output_diagnostics(self):
    source = Path("run_champ_single_overfit.py").read_text(encoding="utf-8")

    self.assertIn("diagnostics.json", source)
    self.assertIn("image_diagnostics", source)
    self.assertIn("residual_rms", source)


if __name__ == "__main__":
  unittest.main()

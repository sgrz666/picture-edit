import unittest
from pathlib import Path


PATCH_PATH = Path("patches/deepgen_pose_control.patch")


class DeepGenPosePatchTests(unittest.TestCase):

  def test_patch_exposes_control_arguments_on_both_pipeline_layers(self):
    patch = PATCH_PATH.read_text(encoding="utf-8")

    self.assertGreaterEqual(patch.count("block_controlnet_hidden_states"), 7)
    self.assertGreaterEqual(patch.count("control_scale"), 4)
    self.assertIn("block_controlnet_hidden_states=control_states", patch)
    self.assertIn("block_controlnet_hidden_states=block_controlnet_hidden_states", patch)

  def test_patch_handles_cfg_batch_and_scale_once(self):
    patch = PATCH_PATH.read_text(encoding="utf-8")

    self.assertIn("state = state * float(control_scale)", patch)
    self.assertEqual(patch.count("state = state * float(control_scale)"), 1)
    self.assertIn("state = torch.cat([state, state], dim=0)", patch)
    self.assertIn("expected_control_batch", patch)

  def test_patch_does_not_add_runtime_forward_replacement(self):
    patch = PATCH_PATH.read_text(encoding="utf-8")

    self.assertNotIn("transformer.forward =", patch)


if __name__ == "__main__":
  unittest.main()

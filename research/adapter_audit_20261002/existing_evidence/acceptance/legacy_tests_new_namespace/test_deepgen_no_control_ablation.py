import unittest
from pathlib import Path


RUNNER = Path("run_deepgen_no_control_ablation.py")


class NoControlRunnerContractTests(unittest.TestCase):

  def test_runner_exists_and_uses_audited_no_control_call(self):
    self.assertTrue(RUNNER.is_file())
    source = RUNNER.read_text(encoding="utf-8")
    self.assertIn("call_deepgen_without_control", source)
    self.assertNotIn("PoseConditionAdapter", source)
    self.assertNotIn("block_controlnet_hidden_states=", source)
    self.assertNotIn("control_scale=", source)

  def test_runner_writes_required_ablation_artifacts(self):
    self.assertTrue(RUNNER.is_file())
    source = RUNNER.read_text(encoding="utf-8")
    for filename in (
        "03_DEEPGEN_NO_CONTROL.png",
        "04_ABLATION_COMPARISON_GRID.png",
        "config.json",
        "diagnostics.json",
        "summary.json",
    ):
      self.assertIn(filename, source)


if __name__ == "__main__":
  unittest.main()

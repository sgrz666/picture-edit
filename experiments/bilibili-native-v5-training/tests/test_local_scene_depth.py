"""CPU regression for PNG quantization versus same-fit local occlusion depths."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.native_full_assets import write_record
from src.pose_control.qwen21_full.native_local_control import resolve_local_ownership


@dataclass
class Local:
    region_masks: torch.Tensor
    visibility: torch.Tensor
    region_valid: torch.Tensor
    person_valid: torch.Tensor
    boxes: torch.Tensor
    person_owner: torch.Tensor
    body_height: torch.Tensor
    local_z: torch.Tensor
    local_z_valid: torch.Tensor
    scene_z: torch.Tensor
    scene_z_valid: torch.Tensor
    part_visible: torch.Tensor
    metadata: tuple

    def validate(self):
        assert self.scene_z.shape == self.scene_z_valid.shape == (1, 1, 32, 32)
        assert self.scene_z.dtype == torch.float32 and self.scene_z_valid.dtype == torch.bool
        assert not (self.scene_z[self.scene_z_valid] <= 0).any()
        assert not (self.scene_z[~self.scene_z_valid] != 0).any()
        return self


class FloatRenderer:
    def __init__(self):
        self.calls = []

    def render(self, person, camera, hw):
        self.calls.append((person, camera.copy(), tuple(hw)))
        return {"z": np.full(hw, 2.0009, np.float32), "valid": np.ones(hw, bool)}


class WrappedPublishedRenderer:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def render(self, *args):
        self.calls += 1
        raise AssertionError("Published PNG/Bounded wrappers must be bypassed for local float scene")


class LocalSceneContracts(unittest.TestCase):
    def setUp(self):
        path = ROOT / "experiments/bilibili-native-v5-training/scripts/local_scene_depth.py"
        self.assertTrue(path.is_file(), "LocalSceneDepth is not implemented")
        spec = importlib.util.spec_from_file_location("local_scene_depth", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.spec = CanvasSpec((64, 32), (32, 16), (32, 32), (8, 0), policy="aspect_preserving")
        self.person = {"mesh_cam": np.array([[-.2, -.2, 2.], [.2, -.2, 2.], [0., .2, 2.]], np.float32),
                       "faces": np.array([[0, 1, 2]]), "metadata": {"person_id": "person0"}}
        self.asset = {"people": [self.person], "K": np.array([[80., 0., 16.], [0., 90., 32.], [0., 0., 1.]], np.float32),
                      "metadata": {"original_hw": [64, 32], "geometry_method": "sapiens_v1"}}
        self.driver = write_record(self.directory / "driver.npz", self.asset, schema="native_smplx_asset_v4")
        self.source = write_record(self.directory / "source.npz", self.asset, schema="native_smplx_asset_v4")
        support = torch.zeros(1, 6, 32, 32)
        support[:, 0] = self.spec.valid_mask()
        z = support * 2.0009
        self.local = Local(support, torch.ones_like(support), torch.tensor([[True, False, False, False, False, False]]),
            torch.tensor([[True, False]]), torch.tensor([[[0., 0., 1., 1.]] * 6]),
            torch.cat((self.spec.valid_mask()[None], torch.zeros(1, 1, 32, 32)), 1),
            torch.tensor([[1.8, 0.]]), z, support.bool(), torch.full((1, 1, 32, 32), 2.),
            torch.ones(1, 1, 32, 32, dtype=torch.bool), support.bool(),
            ({"source_asset_sha256": self.source["sha256"], "target_canvas_id": self.spec.fingerprint(),
              "existing": {"value": 1}},))
        self.sample = {"depth_condition_v2": torch.full((1, 1, 32, 32), 2.),
                       "normal": torch.zeros(3, 32, 32), "pair_metadata": {"pair_id": "p"}}
        self.renderer = FloatRenderer()
        self.calls = 0

    def job(self):
        return {"source_asset": self.source, "driver_asset": self.driver, "target_canvas": self.spec.metadata(),
                "source_indices": [0, -1], "source_slot_ids": [0, 1], "expression_mode": "driver",
                "distances": "absent_single_person"}

    def original(self, job, *, renderer):
        self.calls += 1
        return self.sample, self.local

    def test_quantized_visible_surface_recovers_without_changing_body_values(self):
        self.assertEqual(float(resolve_local_ownership(self.local).sum()), 0.)
        original_scene = self.local.scene_z.clone()
        helper = self.module.LocalSceneDepth(self.original)
        sample, local = helper.assemble(self.job(), renderer=self.renderer)
        self.assertGreater(float(resolve_local_ownership(local).sum()), 0.)
        self.assertIs(sample, self.sample)
        self.assertTrue(torch.equal(sample["depth_condition_v2"], torch.full((1, 1, 32, 32), 2.)))
        self.assertTrue(torch.equal(self.local.scene_z, original_scene))
        self.assertIs(local.local_z, self.local.local_z)
        self.assertTrue(local.scene_z_valid[:, :, :, 8:24].all())
        self.assertFalse(local.scene_z_valid[:, :, :, :8].any())
        self.assertEqual(float(local.scene_z[:, :, :, :8].sum()), 0.)
        self.assertEqual(self.calls, 1)

    def test_true_occlusion_stays_hidden_even_when_part_visible_is_true(self):
        self.local.local_z[self.local.local_z_valid] = 2.01
        sample, local = self.module.LocalSceneDepth(self.original).assemble(self.job(), renderer=self.renderer)
        self.assertEqual(float(resolve_local_ownership(local).sum()), 0.)
        self.assertTrue(local.scene_z_valid[:, :, :, 8:24].all())

    def test_unwraps_renderer_and_uses_complete_driver_at_exact_target_camera_grid(self):
        selected = WrappedPublishedRenderer(self.renderer)
        bounded = WrappedPublishedRenderer(selected)
        _, local = self.module.LocalSceneDepth(self.original).assemble(self.job(), renderer=bounded)
        self.assertEqual((bounded.calls, selected.calls, len(self.renderer.calls)), (0, 0, 1))
        person, camera, hw = self.renderer.calls[0]
        np.testing.assert_array_equal(person["mesh_cam"], self.person["mesh_cam"])
        np.testing.assert_array_equal(person["faces"], self.person["faces"])
        np.testing.assert_array_equal(camera, self.spec.camera(self.asset["K"]).astype(np.float32))
        self.assertEqual(hw, self.spec.canvas_hw)
        meta = local.metadata[0]["local_scene_depth"]
        self.assertEqual(meta["source_asset_sha256"], self.driver["sha256"])
        self.assertEqual(meta["source_role"], "driver")
        self.assertEqual(meta["target_canvas_id"], self.spec.fingerprint())
        self.assertIn("float", meta["policy"])
        local.metadata[0]["existing"]["value"] = 2
        self.assertEqual(self.local.metadata[0]["existing"]["value"], 1)
        self.assertEqual(local.metadata[0]["source_asset_sha256"], self.source["sha256"])

    def test_bad_source_or_driver_sha_is_rejected_before_render(self):
        for key in ("source_asset", "driver_asset"):
            with self.subTest(key=key):
                job = self.job()
                job[key] = {**job[key], "sha256": "wrong"}
                with self.assertRaises(ValueError):
                    self.module.LocalSceneDepth(self.original).assemble(job, renderer=self.renderer)
        self.assertEqual(self.renderer.calls, [])

    def test_unsupported_protocol_and_dual_driver_explicitly_fall_back(self):
        for update in ({"source_indices": [1, 0]}, {"expression_mode": "source"}, {"future_control": True}):
            with self.subTest(update=update):
                result = self.module.LocalSceneDepth(self.original).assemble({**self.job(), **update}, renderer=self.renderer)
                self.assertIs(result[0], self.sample)
                self.assertIs(result[1], self.local)
        dual = deepcopy(self.asset)
        dual["people"].append(deepcopy(self.person))
        reference = write_record(self.directory / "dual.npz", dual, schema="native_smplx_asset_v4")
        result = self.module.LocalSceneDepth(self.original).assemble({**self.job(), "driver_asset": reference}, renderer=self.renderer)
        self.assertIs(result[1], self.local)
        self.assertEqual(self.renderer.calls, [])

    def test_wrong_canvas_or_invalid_rendered_depth_is_rejected(self):
        job = self.job()
        job["target_canvas"] = CanvasSpec((32, 32), (32, 32), (32, 32)).metadata()
        with self.assertRaises(ValueError):
            self.module.LocalSceneDepth(self.original).assemble(job, renderer=self.renderer)

        class Invalid(FloatRenderer):
            def render(self, person, camera, hw):
                result = super().render(person, camera, hw)
                result["z"][0, 8] = np.nan
                return result
        with self.assertRaises(ValueError):
            self.module.LocalSceneDepth(self.original).assemble(self.job(), renderer=Invalid())


if __name__ == "__main__":
    unittest.main()

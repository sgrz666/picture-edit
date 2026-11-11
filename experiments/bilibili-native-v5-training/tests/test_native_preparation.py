"""Offline contracts for selected geometry and immutable publishing; no model."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_native as native


class PreparationContracts(unittest.TestCase):
    def test_publication_resumes_only_when_content_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            original = native.publish(path, [{"pair_id": "p"}], True)
            self.assertEqual(original, native.publish(path, [{"pair_id": "p"}], True))
            with self.assertRaises(ValueError):
                native.publish(path, [{"pair_id": "changed"}], True)
            config = Path(temp) / "config.json"
            first = native.publish(config, {"sources": ({"weight": 1.},), "betas": (.9, .999)})
            self.assertEqual(first, native.publish(config, {"sources": ({"weight": 1.},), "betas": (.9, .999)}))

    def test_global_conditions_use_selected_pngs_and_cv_normal_basis(self):
        import cv2
        import numpy as np
        class Renderer:
            def render(self, person, camera, hw):
                return {"z": np.ones(hw, np.float32) * 9,
                        "normal": np.zeros((*hw, 3), np.float32),
                        "valid": np.ones(hw, bool), "part": np.ones(hw, np.uint8)}
        with tempfile.TemporaryDirectory() as temp:
            depth = Path(temp) / "depth.png"
            normal = Path(temp) / "normal.png"
            cv2.imwrite(str(depth), np.full((2, 2), 1234, np.uint16))
            cv2.imwrite(str(normal), np.broadcast_to(np.array([255, 127, 127], np.uint8), (2, 2, 3)))
            renderer = native.SelectedGeometryRenderer(Renderer())
            person = {"mesh_cam": np.ones((3, 3), np.float32), "faces": np.array([[0, 1, 2]])}
            camera = np.eye(3)
            renderer.register(person, camera, (2, 2), depth, normal)
            result = renderer.render(person, camera, (2, 2))
            np.testing.assert_allclose(result["z"], 1.234)
            self.assertLess(float(result["normal"][0, 0, 2]), -.99)
            self.assertEqual(renderer.used, {str(depth)})
            # Partial submesh uses the selected fit's renderer, not a whole-body PNG.
            result = renderer.render({**person, "faces": np.empty((0, 3), int)}, camera, (2, 2))
            np.testing.assert_allclose(result["z"], 9.)

    def test_rounding_boundary_only_limits_geometry_and_keeps_original_maps(self):
        import cv2
        import numpy as np
        class Renderer:
            def render(self, person, camera, hw):
                valid = np.ones(hw, bool)
                valid[0, 0] = False
                return {"z": np.ones(hw, np.float32), "normal": np.zeros((*hw, 3), np.float32),
                        "valid": valid, "part": valid.astype(np.uint8)}
        with tempfile.TemporaryDirectory() as temp:
            depth, normal = Path(temp) / "depth.png", Path(temp) / "normal.png"
            cv2.imwrite(str(depth), np.full((2, 2), 1234, np.uint16))
            cv2.imwrite(str(normal), np.full((2, 2, 3), 127, np.uint8))
            original = depth.read_bytes(), normal.read_bytes()
            renderer = native.SelectedGeometryRenderer(Renderer())
            person = {"mesh_cam": np.ones((3, 3), np.float32), "faces": np.array([[0, 1, 2]])}
            renderer.register(person, np.eye(3), (2, 2), depth, normal)
            result = renderer.render(person, np.eye(3), (2, 2))
            self.assertEqual(int(result["valid"].sum()), 3)
            self.assertEqual(renderer.boundary_pixels[str(depth)], 1)
            self.assertEqual(original, (depth.read_bytes(), normal.read_bytes()))


if __name__ == "__main__":
    unittest.main()

"""Input isolation and camera-coordinate contracts for the four-frame pilot."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contracts import (  # noqa: E402
    fit_record,
    model_to_original,
    project,
    safe_child,
    sha256_file,
    validate_boxes,
    write_json_exclusive,
)


class FitRecordTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "id": "pair00_frame000001",
            "image": "/data/example.jpg",
            "image_sha256": "a" * 64,
            "size": [1280, 720],
            "K": [[1000, 0, 640], [0, 900, 360], [0, 0, 1]],
        }

    def test_only_whitelisted_input_and_metadata_reach_fitting(self):
        metadata = {"split": "test", "category": "hug", "camera_id": "4", "frame_id": 1}
        self.row.update(metadata)
        self.row.update({
            "joints_3d": [[[1, 2, 3]]], "smpl_gt": "secret.npz",
            "mesh": "secret.obj", "extrinsics": [[1]], "unknown": "drop",
        })
        before = copy.deepcopy(self.row)
        result = fit_record(self.row)
        self.assertEqual(set(result), {
            "id", "image", "image_sha256", "size", "K",
            "split", "category", "camera_id", "frame_id",
        })
        self.assertEqual({key: result[key] for key in metadata}, metadata)
        self.assertEqual(self.row, before)

    def test_absent_optional_metadata_is_not_invented(self):
        self.assertEqual(fit_record(self.row), self.row)

    def test_every_required_field_is_checked(self):
        for key in self.row:
            with self.subTest(key=key):
                missing = {k: v for k, v in self.row.items() if k != key}
                with self.assertRaisesRegex(ValueError, key):
                    fit_record(missing)

    def test_invalid_frame_values_are_rejected(self):
        cases = [
            ("id", ""), ("image", ""), ("image_sha256", "not-a-hash"),
            ("size", [0, 720]), ("size", [1280, -1]),
            ("size", [1280.5, 720]), ("size", [1280]),
            ("K", np.eye(2).tolist()),
            ("K", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ("K", [[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ("K", [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]]),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    fit_record({**self.row, key: value})


class BoxTests(unittest.TestCase):
    def test_two_boxes_return_float64_and_preserve_image_boundaries(self):
        boxes = [[0, 0, 1280, 720], [100, 80, 300, 600]]
        result = validate_boxes(boxes, [1280, 720])
        self.assertEqual(result.dtype, np.dtype("float64"))
        np.testing.assert_array_equal(result, boxes)

    def test_invalid_boxes_are_rejected_instead_of_clipped(self):
        good = [100, 80, 300, 600]
        cases = [
            [], [good], [good, good, good], [[1, 2, 3], good],
            [[300, 80, 100, 600], good], [[100, 80, 100, 600], good],
            [[100, 600, 300, 80], good], [[-1, 80, 300, 600], good],
            [[100, 80, 1281, 600], good], [[100, 80, 300, 721], good],
            [[100, float("nan"), 300, 600], good],
            [[100, 80, float("inf"), 600], good],
        ]
        for boxes in cases:
            with self.subTest(boxes=boxes):
                with self.assertRaises(ValueError):
                    validate_boxes(boxes, [1280, 720])

    def test_invalid_image_size_is_rejected(self):
        for size in ([0, 720], [1280], [1280, float("inf")], [1280.5, 720]):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    validate_boxes([[0, 0, 1, 1], [1, 1, 2, 2]], size)


class CoordinateTests(unittest.TestCase):
    def test_projection_handles_arbitrary_leading_dimensions(self):
        xyz = np.array([[[1, 2, 2], [-2, 1, 4]], [[0, 0, 1], [3, -1, 2]]])
        K = [[1000, 10, 640], [0, 900, 360], [0, 0, 1]]
        expected = [[[1150, 1260], [142.5, 585]], [[640, 360], [2135, -90]]]
        np.testing.assert_allclose(project(xyz, K), expected)
        np.testing.assert_allclose(project([1, 2, 2], K), expected[0][0])

    def test_nonpositive_or_nonfinite_depth_is_rejected(self):
        for depth in (0, -1, float("nan"), float("inf")):
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    project([[1, 2, depth]], np.eye(3))
        with self.assertRaises(ValueError):
            project([[float("inf"), 2, 3]], np.eye(3))

    def test_malformed_projection_inputs_are_rejected(self):
        for xyz, K in (([1, 2], np.eye(3)), ([[1, 2, 3]], np.eye(2))):
            with self.subTest(xyz=xyz, K=K):
                with self.assertRaises(ValueError):
                    project(xyz, K)

    def test_resize_and_padding_roundtrip_for_non_square_image(self):
        # 1280x720 -> 640x360, then centered in a 640x640 canvas.
        original = np.array([[[0, 0], [1280, 720]], [[640, 360], [100, 200]]])
        scale, offset = 0.5, np.array([0, 140])
        model = original * scale + offset
        np.testing.assert_allclose(model_to_original(model, scale, offset), original)
        # A landscape canvas can also have horizontal padding.
        np.testing.assert_allclose(
            model_to_original(original * 0.25 + [100, 0], 0.25, [100, 0]), original
        )

    def test_invalid_inverse_affine_inputs_are_rejected(self):
        for scale in (0, -1, float("nan"), float("inf"), [0.5]):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    model_to_original([[0, 0]], scale, [0, 140])
        for offset in ([0], [0, 1, 2], [0, float("nan")]):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    model_to_original([[0, 0]], 0.5, offset)
        for xy in ([0, 0, 0], [[0, float("inf")]]):
            with self.subTest(xy=xy):
                with self.assertRaises(ValueError):
                    model_to_original(xy, 0.5, [0, 140])


class FileTests(unittest.TestCase):
    def test_safe_child_resolves_normal_sample_and_arm_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            self.assertEqual(
                safe_child(root, "pair00_cam4_000001", "joint", "result.json"),
                (root / "pair00_cam4_000001" / "joint" / "result.json").resolve(),
            )

    def test_safe_child_rejects_traversal_and_absolute_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = [
                "", ".", "..", "../escape", "sample/../escape", "sample/id",
                "sample\\id", "C:\\escape", "C:escape", "/tmp/escape",
                "\\\\server\\share", "arm:camera", str(root.resolve()),
            ]
            for component in invalid:
                with self.subTest(component=component):
                    with self.assertRaises(ValueError):
                        safe_child(root, "sample", component)
            with self.assertRaises(ValueError):
                safe_child(root)

    def test_safe_child_rejects_existing_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            outside = Path(directory) / "root-other"
            root.mkdir()
            outside.mkdir()
            link = root / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")
            with self.assertRaises(ValueError):
                safe_child(root, "escape", "result.json")

    def test_json_write_is_exclusive_and_keeps_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            payload = {"label": "双人", "score": 0.25}
            write_json_exclusive(target, payload)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"overwritten": True})
            self.assertEqual(target.read_bytes(), before)

    def test_nonfinite_json_is_rejected_without_creating_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        write_json_exclusive(target, {"nested": [value]})
                    self.assertFalse(target.exists())

    def test_sha256_reads_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "input.bin"
            data = b"hi4d\x00\xff\r\n"
            target.write_bytes(data)
            self.assertEqual(sha256_file(target), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()

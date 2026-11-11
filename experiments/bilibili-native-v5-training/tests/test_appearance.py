"""Source appearance contracts on CPU; never load a model or start training."""
import builtins
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


POINTS = [[30.0, 30.0], [70.0, 30.0], [50.0, 50.0], [35.0, 75.0], [65.0, 75.0]]


def frame():
    return {"original_hw": [100, 200],
            "detection": {"face": {"landmarks5": copy.deepcopy(POINTS)}},
            "loss_mask": {"body": True, "face": True, "hand_a": True, "hand_b": True},
            "hand_raw_pose_usable": {"hand_a": True, "hand_b": True}}


def models():
    return {"arcface_model": "unopened.onnx", "arcface_sha256": "a" * 64,
            "dino_model": "unopened-dino", "dino_files": {"config.json": "b" * 64}, "device": "cpu"}


def crop_row(frame_id):
    source = frame()
    source["frame_id"] = frame_id
    return {"frame": source, "face_box": [0.1, 0.1, 0.3, 0.3],
            "hand_boxes": [[0.15, 0.5, 0.35, 0.7], [0.65, 0.5, 0.85, 0.7]]}


def prime_sample(frame_id):
    return {"frame": {**frame(), "frame_id": frame_id}, "rgb": {"path": "unopened.png", "sha256": "c" * 64},
            "canvas": object(), "asset": {"sha256": "d" * 64}, "empty_references": object()}


class AppearanceContracts(unittest.TestCase):
    def load(self):
        path = Path(__file__).parents[1] / "scripts/appearance.py"
        self.assertTrue(path.is_file(), "The resident Source-only feature extractor is not implemented")
        spec = importlib.util.spec_from_file_location("bilibili_appearance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_import_and_constructor_do_not_import_heavy_dependencies(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "PIL", "onnxruntime", "transformers", "cv2"}:
                raise AssertionError("Eager dependency import: " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            module = self.load()
            module.SourceFeatureExtractor({"arcface_model": "unopened.onnx", "arcface_sha256": "a" * 64,
                                           "dino_model": "unopened-dino", "dino_files": {"config.json": "b" * 64},
                                           "device": "cpu"})

    def test_five_points_come_from_source_face_detector(self):
        module = self.load()
        source = frame()
        source["candidate"] = [[999.0, 999.0]] * 134
        self.assertEqual(module.source_five_points(source), tuple(map(tuple, POINTS)))
        self.assertEqual(source["detection"]["face"]["landmarks5"], POINTS)

    def test_invalid_or_degenerate_five_points_are_unavailable(self):
        module = self.load()
        for points in (None, [[1.0, 2.0]] * 5, [[float(i), float(i)] for i in range(5)],
                       [[float("nan"), 1.0], *POINTS[1:]], POINTS[:4], [[True, 2.0], *POINTS[1:]]):
            with self.subTest(points=points):
                source = frame()
                source["detection"]["face"]["landmarks5"] = points
                self.assertIsNone(module.source_five_points(source))
        self.assertIsNone(module.source_five_points({"detection": {"face": None}}))

    def test_alignment_uses_canvas_pixel_points_on_original_pixels(self):
        module = self.load()

        class Canvas:
            original_hw = (100, 200)
            canvas_hw = (128, 128)

            def pixel_points(self, points):
                self.seen = points.tolist()
                return points * [0.5, 0.5] + [14.0, 39.0]

        canvas = Canvas()
        result = module.arcface_canvas_points(frame(), canvas)
        self.assertEqual(canvas.seen, POINTS)
        self.assertEqual(result.tolist(), [[x * 0.5 + 14.0, y * 0.5 + 39.0] for x, y in POINTS])

    def test_outside_original_image_alignment_is_unavailable(self):
        module = self.load()
        source = frame()
        source["detection"]["face"]["landmarks5"][0][0] = -1.0
        self.assertIsNone(module.source_five_points(source, image_hw=(100, 200)))

    def test_hand_a_is_anatomical_right_and_hand_b_is_left(self):
        module = self.load()
        source = frame()
        source["loss_mask"]["hand_a"] = False
        policy = module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3], [[0.1, 0.5, 0.3, 0.7]] * 2)
        self.assertEqual(policy["hand_dino"], (True, False))
        source["loss_mask"]["hand_a"] = True
        source["loss_mask"]["hand_b"] = False
        self.assertEqual(module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3],
                                                         [[0.1, 0.5, 0.3, 0.7]] * 2)["hand_dino"], (False, True))

    def test_raw_pose_and_roi_gate_each_hand_independently_without_mutating_flags(self):
        module = self.load()
        source = frame()
        source["hand_raw_pose_usable"]["hand_b"] = False
        before = copy.deepcopy(source)
        policy = module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3], [[0.1, 0.5, 0.3, 0.7]] * 2)
        self.assertEqual(policy["hand_dino"], (False, True))
        self.assertEqual(source, before)
        policy = module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3],
                                                  [[0.1, 0.5, 0.3, 0.7], [1.1, 0.2, 1.2, 0.5]])
        self.assertEqual(policy["hand_dino"], (False, False))

    def test_face_identity_and_texture_have_independent_validity(self):
        module = self.load()
        source = frame()
        source["detection"]["face"]["landmarks5"] = None
        policy = module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3], [[0.0] * 4] * 2)
        self.assertFalse(policy["arcface"])
        self.assertTrue(policy["face_dino"])
        source = frame()
        policy = module.source_appearance_policy(source, [0.0] * 4, [[0.0] * 4] * 2)
        self.assertTrue(policy["arcface"])
        self.assertFalse(policy["face_dino"])
        source["loss_mask"]["face"] = False
        self.assertFalse(module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3],
                                                         [[0.0] * 4] * 2)["arcface"])

    def test_unclipped_intersecting_roi_is_valid_but_degenerate_roi_is_not(self):
        module = self.load()
        self.assertTrue(module.valid_canvas_roi([-0.2, 0.3, 1.2, 0.8]))
        for box in ([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.5, 1.2], [1.0, 0.1, 1.5, 0.4],
                    [float("nan"), 0.0, 0.5, 0.5], [0.1, 0.2, 0.4]):
            with self.subTest(box=box):
                self.assertFalse(module.valid_canvas_roi(box))

    def test_missing_raw_flags_preserve_legacy_loss_mask_policy(self):
        module = self.load()
        source = frame()
        del source["hand_raw_pose_usable"]
        self.assertEqual(module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3],
                                                         [[0.1, 0.5, 0.3, 0.7]] * 2)["hand_dino"], (True, True))

    def test_nonboolean_release_flags_are_rejected(self):
        module = self.load()
        for key in ("loss_mask", "hand_raw_pose_usable"):
            source = frame()
            source[key]["hand_a"] = "false"
            with self.assertRaises(ValueError):
                module.source_appearance_policy(source, [0.1, 0.1, 0.3, 0.3], [[0.1, 0.5, 0.3, 0.7]] * 2)

    def test_batch_plan_preserves_each_original_modality_policy_and_anatomical_side(self):
        module = self.load()
        rows = [crop_row("A"), crop_row("B")]
        rows[0]["frame"]["loss_mask"]["hand_a"] = False
        rows[1]["frame"]["hand_raw_pose_usable"]["hand_b"] = False
        rows[1]["frame"]["detection"]["face"]["landmarks5"] = None
        before = copy.deepcopy(rows)
        tasks = [task for batch in module.plan_dino_batches(rows, max_batch=16) for task in batch]
        self.assertEqual([(task["frame_id"], task["region"]) for task in tasks],
                         [("A", "face"), ("A", "left"), ("B", "face"), ("B", "right")])
        for task in tasks:
            self.assertEqual(task["key"], module.dino_cache_key(task["frame_id"], task["box"]))
        self.assertEqual(rows, before)

    def test_batch_plan_limits_crop_batch_size_independently_of_frame_count(self):
        module = self.load()
        batches = module.plan_dino_batches([crop_row(str(i)) for i in range(3)], max_batch=2)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 2, 1])
        self.assertEqual({task["frame_id"] for batch in batches for task in batch}, {"0", "1", "2"})

    def test_batch_keys_use_exact_float32_roi_and_keep_frames_separate(self):
        module = self.load()
        a = module.dino_cache_key("A", [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(a, module.dino_cache_key("A", [0.1000000000001, 0.2, 0.3, 0.4]))
        self.assertNotEqual(a, module.dino_cache_key("B", [0.1, 0.2, 0.3, 0.4]))
        self.assertNotEqual(a, module.dino_cache_key("A", [0.11, 0.2, 0.3, 0.4]))

    def test_prime_rejects_missing_and_duplicate_entries_before_loading_models(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())
        for key in ("frame", "rgb", "canvas", "asset", "empty_references"):
            sample = prime_sample("A")
            del sample[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                extractor.prime_dino([sample])
        with self.assertRaises(ValueError):
            extractor.prime_dino([prime_sample("A"), prime_sample("A")])
        self.assertIsNone(extractor._dino)

    def test_prime_rejects_oversized_frames_and_invalid_batch_sizes(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())
        with self.assertRaises(ValueError):
            extractor.prime_dino([prime_sample(str(i)) for i in range(17)])
        for size in (0, -1, 1.5, True, 17):
            with self.subTest(size=size), self.assertRaises(ValueError):
                extractor.prime_dino([], max_batch=size)
        self.assertIsNone(extractor._dino)

    def test_patch_cache_is_consumed_only_for_the_active_frame_and_exact_roi(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())
        key = module.dino_cache_key("A", [0.1, 0.2, 0.3, 0.4])
        expected = object()
        extractor._dino_cache[key] = expected
        extractor._active_frame_id = "A"
        self.assertIs(extractor._patches(None, [0.1, 0.2, 0.3, 0.4]), expected)
        self.assertNotIn(key, extractor._dino_cache)

    def test_unprimed_patch_uses_real_inference_path_after_unclipped_crop(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())

        class Image:
            width, height = 200, 100

            def crop(self, box):
                self.box = box
                return "actual-pil-crop"

        image = Image()
        with patch.object(extractor, "_dino_patches", return_value=["actual-forward-output"]) as forward:
            self.assertEqual(extractor._patches(image, [-0.1, 0.2, 0.3, 0.4]), "actual-forward-output")
            forward.assert_called_once_with(["actual-pil-crop"])
        self.assertEqual(image.box, (-20, 20, 60, 40))

    def test_identical_roi_for_multiple_modalities_is_kept_until_last_consumer(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())
        box = [0.1, 0.2, 0.3, 0.4]
        key = module.dino_cache_key("A", box)
        expected = object()
        extractor._dino_cache[key] = expected
        extractor._dino_cache_uses[key] = 2
        extractor._active_frame_id = "A"
        self.assertIs(extractor._patches(None, box), expected)
        self.assertIn(key, extractor._dino_cache)
        self.assertIs(extractor._patches(None, box), expected)
        self.assertNotIn(key, extractor._dino_cache)

    def test_extract_cleans_only_its_frame_cache_on_success_and_failure(self):
        module = self.load()
        for failure in (False, True):
            with self.subTest(failure=failure):
                extractor = module.SourceFeatureExtractor(models())
                keys = {fid: module.dino_cache_key(fid, [0.1, 0.2, 0.3, 0.4]) for fid in ("A", "B")}
                extractor._dino_cache = {key: object() for key in keys.values()}
                extractor._dino_cache_uses = {key: 1 for key in keys.values()}
                extractor._primed_images = {fid: object() for fid in keys}
                with patch.object(extractor, "_extract_one", side_effect=RuntimeError("real failure") if failure else None,
                                  return_value="actual-reference"):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "real failure"):
                            extractor.extract({"frame_id": "A"}, None, None, None, None)
                    else:
                        self.assertEqual(extractor.extract({"frame_id": "A"}, None, None, None, None), "actual-reference")
                self.assertIsNone(extractor._active_frame_id)
                self.assertNotIn(keys["A"], extractor._dino_cache)
                self.assertNotIn("A", extractor._primed_images)
                self.assertIn(keys["B"], extractor._dino_cache)
                self.assertIn("B", extractor._primed_images)

    def test_priming_does_not_overwrite_unconsumed_previous_frame(self):
        module = self.load()
        extractor = module.SourceFeatureExtractor(models())
        extractor._primed_images["A"] = object()
        with self.assertRaisesRegex(ValueError, "Consume the previous"):
            extractor.prime_dino([])

    def test_batch_planner_rejects_missing_and_duplicate_frame_rows(self):
        module = self.load()
        with self.assertRaises(ValueError):
            module.plan_dino_batches([crop_row("A"), crop_row("A")])
        with self.assertRaises(ValueError):
            module.plan_dino_batches([{"frame": frame()}])


if __name__ == "__main__":
    unittest.main()

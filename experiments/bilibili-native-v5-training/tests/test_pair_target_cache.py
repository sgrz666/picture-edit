"""CPU-only equivalence contracts against the real Native single-person assembler."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_full_assets import sha256, write_record
from src.pose_control.qwen21_full.native_full_data import assemble_full_cache
from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA


class Renderer:
    def __init__(self):
        self.calls = []

    def render(self, person, camera, hw):
        self.calls.append(person["tag"])
        h, w = hw
        part = np.ones((h, w), np.uint8)
        part[:, :person["tag"] % max(1, w)] = 2
        normal = np.zeros((h, w, 3), np.float32)
        normal[..., 2] = -1
        return {"z": np.full((h, w), 2., np.float32), "valid": np.ones((h, w), bool),
                "normal": normal, "part": part}


class TargetCacheContracts(unittest.TestCase):
    def setUp(self):
        path = ROOT / "experiments/bilibili-native-v5-training/scripts/pair_target_cache.py"
        self.assertTrue(path.is_file(), "SingleTargetCache is not implemented")
        spec = importlib.util.spec_from_file_location("pair_target_cache", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.renderer = Renderer()
        self.calls = 0
        self.source_a = self.asset("source_a", (32, 32), 1)
        self.source_b = self.asset("source_b", (32, 64), 2)
        self.target_a = self.asset("target_a", (32, 32), 9)
        self.target_b = self.asset("target_b", (64, 32), 10)

    def original(self, job, *, renderer):
        self.calls += 1
        return assemble_full_cache(job, renderer=renderer)

    def asset(self, name, hw, tag, person_id="same_person"):
        h, w = hw
        rgb_path = self.directory / (name + ".png")
        Image.new("RGB", (w, h), (tag, 30, 40)).save(rgb_path)
        rgb = {"path": str(rgb_path), "sha256": sha256(rgb_path)}
        mask_path = self.directory / (name + "_mask.png")
        mask = np.zeros(hw, np.uint8)
        mask[tag % h:, :] = 255
        Image.fromarray(mask).save(mask_path)
        mask_ref = {"path": str(mask_path), "sha256": sha256(mask_path)}
        mesh = np.array([[-.3, -.3, 2.], [.3, -.3, 2.], [0., .3, 2.]] * 3, np.float32)
        joints = np.tile([0., 0., 2.], (22, 1)).astype(np.float32)
        person = {"tag": tag, "mesh_cam": mesh, "joints_cam": joints, "faces": np.arange(9).reshape(3, 3),
            "face_part": np.array([0, 4, 7]), "body_height": 1.8, "body_pose": np.zeros(63, np.float32),
            "global26": np.zeros(26, np.float32), "jaw_pose": np.zeros(3, np.float32),
            "expression_full": np.zeros(10, np.float32), "hand_pose": np.zeros((2, 45), np.float32),
            "face72_cam": np.tile([0., 0., 2.], (72, 1)).astype(np.float32), "face72_valid": np.ones(72, bool),
            "hand21_cam": np.tile([0., 0., 2.], (2, 21, 1)).astype(np.float32), "hand21_valid": np.ones((2, 21), bool),
            "wrist_cam": np.array([[0., 0., 2.]] * 2, np.float32), "wrist_rotation": np.array([np.eye(3)] * 2, np.float32),
            "hand_vertices": [np.array([3, 4, 5]), np.array([6, 7, 8])], "standard_hand_length": [1., 1.],
            "metadata": {"person_id": person_id, "expression_basis_sha256": "basis", "expression_retargeted": False}}
        asset = {"people": [person], "K": np.array([[32., 0., w / 2], [0., 32., h / 2], [0., 0., 1.]], np.float32),
            "metadata": {"original_hw": list(hw), "units": "model_unit", "scale_status": "model_scale",
                "camera_id": name + ":camera", "scale_group_id": "same_scale", "rgb_sha256": rgb["sha256"],
                "counterfactual_geometry": False}}
        asset_ref = write_record(self.directory / (name + "_asset.npz"), asset, schema="native_smplx_asset_v4")
        canvas = native_pad(hw)
        metadata = ({"schema": REFERENCE_SCHEMA, "role": "source_only", "reference_sources": [{"geometry_sha256": asset_ref["sha256"],
            "rgb": rgb, "person_ids": [person_id, "__inactive__"], "canvas": canvas.metadata(),
            "canvas_id": canvas.fingerprint(), "extra_source_metadata": name}]},)
        references = NativeLocalReferencesV4(torch.zeros(1, 2, 1, 512), torch.zeros(1, 2, 1, dtype=torch.bool),
            torch.zeros(1, 2, 1, 256, 1536), torch.zeros(1, 2, 1, dtype=torch.bool),
            torch.zeros(1, 2, 2, 1, 256, 1536), torch.zeros(1, 2, 2, 1, dtype=torch.bool),
            torch.zeros(1, 2, 4), torch.zeros(1, 2, 2, 4), metadata)
        reference_ref = write_record(self.directory / (name + "_references.npz"), references,
                                     schema="native_local_reference_v4")
        return {"asset": asset_ref, "raw": asset, "rgb": rgb, "mask": mask_ref, "canvas": canvas.metadata(),
                "references": reference_ref, "reference_object": references}

    def job(self, source=None, target=None, pair_id="p1"):
        source, target = source or self.source_a, target or self.target_a
        return {"pair_id": pair_id, "source_asset": source["asset"], "driver_asset": target["asset"],
            "source_canvas": source["canvas"], "target_canvas": target["canvas"], "source_masks": [source["mask"]],
            "source_rgb": source["rgb"], "target_rgb": target["rgb"], "source_indices": [0, -1], "source_slot_ids": [0, 1],
            "units": "model_unit", "scale_status": "model_scale", "references": source["references"],
            "expression_mode": "driver", "distances": "absent_single_person"}

    def assertRecordEqual(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected), "Tensor contents differ")
            self.assertEqual(actual.dtype, expected.dtype)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif is_dataclass(expected):
            self.assertIs(type(actual), type(expected))
            for field in fields(expected):
                self.assertRecordEqual(getattr(actual, field.name), getattr(expected, field.name))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertRecordEqual(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertRecordEqual(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_hit_changes_source_and_pair_provenance_exactly_like_original(self):
        cache = self.module.SingleTargetCache(self.original)
        first, local_first = cache.assemble(self.job(), renderer=self.renderer)
        job = self.job(self.source_b, pair_id="p2")
        actual = cache.assemble(job, renderer=self.renderer)
        expected = assemble_full_cache(job, renderer=self.renderer)
        self.assertRecordEqual(actual, expected)
        self.assertEqual((self.calls, cache.hits, cache.misses), (1, 1, 1))
        self.assertIs(actual[0]["pose_heatmap"], first["pose_heatmap"])
        self.assertIs(actual[1].hand_keypoints, local_first.hand_keypoints)
        self.assertNotEqual(first["source_person_mask"].shape, actual[0]["source_person_mask"].shape)

    def test_caller_metadata_mutation_does_not_pollute_template(self):
        cache = self.module.SingleTargetCache(self.original)
        sample, local = cache.assemble(self.job(), renderer=self.renderer)
        sample["pair_metadata"]["unwanted"] = ["first"]
        sample["identity_metadata"]["source_person_ids"][0] = "wrong"
        sample["pose_provenance"]["record_id"] = "wrong"
        sample["depth_condition_v2"].metadata[0]["source_id"] = "wrong"
        sample["depth_condition_v2"].context.metadata[0]["pose_record_id"] = "wrong"
        local.metadata[0]["reference_sources"][0]["geometry_sha256"] = "wrong"
        job = self.job(self.source_b, pair_id="p2")
        actual = cache.assemble(job, renderer=self.renderer)
        self.assertRecordEqual(actual, assemble_full_cache(job, renderer=self.renderer))
        actual[0]["pair_metadata"]["unwanted_again"] = True
        actual[1].metadata[0]["bps"][0][0]["seed"] = -1
        third = self.job(pair_id="p3")
        self.assertRecordEqual(cache.assemble(third, renderer=self.renderer), assemble_full_cache(third, renderer=self.renderer))

    def test_target_switch_evicts_the_only_cached_target(self):
        cache = self.module.SingleTargetCache(self.original)
        for target in (self.target_a, self.target_b, self.target_a):
            job = self.job(target=target)
            self.assertRecordEqual(cache.assemble(job, renderer=self.renderer), assemble_full_cache(job, renderer=self.renderer))
        self.assertEqual((self.calls, cache.hits, cache.misses), (3, 0, 3))

    def test_target_options_and_renderer_invalidate_the_cache(self):
        for updates in ({"bps_seed": 123}, {"dual_config": {"owner_tolerance": .002}},
                        {"target_canvas": native_pad((32, 32), pad_multiple=64).metadata()}):
            with self.subTest(updates=updates):
                cache = self.module.SingleTargetCache(self.original)
                cache.assemble(self.job(), renderer=self.renderer)
                changed = {**self.job(pair_id="changed"), **updates}
                self.assertRecordEqual(cache.assemble(changed, renderer=self.renderer),
                                       assemble_full_cache(changed, renderer=self.renderer))
                self.assertEqual(cache.misses, 2)
        cache = self.module.SingleTargetCache(self.original)
        cache.assemble(self.job(), renderer=self.renderer)
        cache.assemble(self.job(), renderer=Renderer())
        self.assertEqual(cache.misses, 2)

    def test_unsupported_jobs_call_the_original_without_interpretation(self):
        sentinel = object()
        calls = []
        cache = self.module.SingleTargetCache(lambda job, *, renderer: calls.append(job) or sentinel)
        for update in ({"expression_mode": "source"}, {"source_indices": [1, 0]},
                       {"source_slot_ids": [1, 0]}, {"future_control": 1}, {"distances": "other"}):
            job = {**self.job(), **update}
            self.assertIs(cache.assemble(job, renderer=self.renderer), sentinel)
        self.assertEqual(len(calls), 5)

    def test_dual_driver_or_source_falls_back(self):
        sentinel = object()
        for field, fixture in (("driver_asset", self.target_a), ("source_asset", self.source_a)):
            raw = deepcopy(fixture["raw"])
            raw["people"].append(deepcopy(raw["people"][0]))
            asset = write_record(self.directory / (field + "_dual.npz"), raw, schema="native_smplx_asset_v4")
            cache = self.module.SingleTargetCache(lambda job, *, renderer: sentinel)
            self.assertIs(cache.assemble({**self.job(), field: asset}, renderer=self.renderer), sentinel)

    def test_hit_rechecks_all_source_and_target_file_hashes(self):
        fields_to_refs = ("source_asset", "driver_asset", "source_rgb", "target_rgb", "references")
        for field in fields_to_refs:
            with self.subTest(field=field):
                cache = self.module.SingleTargetCache(self.original)
                job = self.job()
                cache.assemble(job, renderer=self.renderer)
                path = Path(job[field]["path"])
                before = path.read_bytes()
                try:
                    with path.open("ab") as stream:
                        stream.write(b"changed")
                    with self.assertRaises(ValueError):
                        cache.assemble(job, renderer=self.renderer)
                finally:
                    path.write_bytes(before)
        cache = self.module.SingleTargetCache(self.original)
        cache.assemble(self.job(), renderer=self.renderer)
        job = self.job()
        job["source_masks"] = [{**job["source_masks"][0], "sha256": "bad"}]
        with self.assertRaises(ValueError):
            cache.assemble(job, renderer=self.renderer)

    def test_hit_rejects_source_identity_geometry_and_rgb_dimension_mismatch(self):
        cache = self.module.SingleTargetCache(self.original)
        cache.assemble(self.job(), renderer=self.renderer)
        wrong_person = self.asset("wrong_person", (32, 32), 3, person_id="someone_else")
        with self.assertRaisesRegex(ValueError, "identity"):
            cache.assemble(self.job(wrong_person), renderer=self.renderer)
        changed = deepcopy(self.source_b["reference_object"])
        changed.metadata[0]["reference_sources"][0]["geometry_sha256"] = "unbound"
        bad_ref = write_record(self.directory / "bad_references.npz", changed, schema="native_local_reference_v4")
        bad_job = self.job(self.source_b)
        bad_job["references"] = bad_ref
        with self.assertRaisesRegex(ValueError, "geometry"):
            cache.assemble(bad_job, renderer=self.renderer)
        bad_job = self.job(self.source_b)
        bad_job["source_canvas"] = self.source_a["canvas"]
        with self.assertRaises(ValueError):
            cache.assemble(bad_job, renderer=self.renderer)

    def test_float_scene_composition_keeps_target_provenance_and_rebinds_each_source(self):
        from src.data.canvas_contract import CanvasSpec
        scene_path = ROOT / "experiments/bilibili-native-v5-training/scripts/local_scene_depth.py"
        scene_spec = importlib.util.spec_from_file_location("composed_local_scene_depth", scene_path)
        scene_module = importlib.util.module_from_spec(scene_spec)
        scene_spec.loader.exec_module(scene_module)

        def v1(fixture, name):
            # Independent assets bind their references to the new v1 SHA; existing fixtures stay intact.
            raw = deepcopy(fixture["raw"])
            raw["metadata"]["geometry_method"] = "sapiens_v1"
            asset = write_record(self.directory / (name + "_v1.npz"), raw, schema="native_smplx_asset_v4")
            references = deepcopy(fixture["reference_object"])
            references.metadata[0]["reference_sources"][0]["geometry_sha256"] = asset["sha256"]
            reference = write_record(self.directory / (name + "_v1_references.npz"), references,
                                     schema="native_local_reference_v4")
            return {**fixture, "asset": asset, "raw": raw, "references": reference}

        class FloatRenderer(Renderer):
            def render(self, person, camera, hw):
                result = super().render(person, camera, hw)
                result["z"].fill(2.0009)
                return result

        class PublishedRenderer:
            def __init__(self):
                self.delegate = FloatRenderer()

            def render(self, person, camera, hw):
                result = self.delegate.render(person, camera, hw)
                if person["tag"] == 9 and tuple(hw) == (64, 64) and len(person["faces"]) == 3:
                    result["z"].fill(2.)  # Selected full-resolution PNG quantization is kept for Body.
                return result

        source_a, source_b = v1(self.source_a, "combo_source_a"), v1(self.source_b, "combo_source_b")
        target = v1(self.asset("combo_target", (64, 64), 9), "combo_target")
        canvas = CanvasSpec((64, 64), (32, 32), (32, 32), policy="aspect_preserving").metadata()
        renderer = PublishedRenderer()
        aligned = scene_module.LocalSceneDepth(self.original)
        cache = self.module.SingleTargetCache(aligned.assemble)
        first_job = {**self.job(source_a, target, pair_id="composed_1"), "target_canvas": canvas}
        first, first_local = cache.assemble(first_job, renderer=renderer)
        self.assertTrue(torch.equal(first["depth_condition_v2"].camera_z, torch.full((1, 1, 32, 32), 2.)))
        self.assertTrue(torch.equal(first_local.scene_z, torch.full((1, 1, 32, 32), 2.0009)))
        first["pair_metadata"]["caller_only"] = True
        first["identity_metadata"]["source_person_ids"][0] = "polluted"
        first["pose_provenance"]["record_id"] = "polluted"
        first["depth_condition_v2"].context.metadata[0]["pose_record_id"] = "polluted"
        first_local.metadata[0]["local_scene_depth"]["policy"] = "polluted"
        first_local.metadata[0]["reference_sources"][0]["geometry_sha256"] = "polluted"

        second_job = {**self.job(source_b, target, pair_id="composed_2"), "target_canvas": canvas}
        second, second_local = cache.assemble(second_job, renderer=renderer)
        expected = scene_module.LocalSceneDepth(assemble_full_cache).assemble(second_job, renderer=renderer)
        self.assertRecordEqual((second, second_local), expected)
        self.assertIs(second_local.scene_z, first_local.scene_z)
        scene = second_local.metadata[0]["local_scene_depth"]
        self.assertEqual(scene["source_asset_sha256"], target["asset"]["sha256"])
        self.assertEqual(scene["policy"], scene_module.POLICY)
        self.assertEqual(second_local.metadata[0]["source_asset_sha256"], source_b["asset"]["sha256"])
        self.assertEqual(second_local.metadata[0]["source_rgb_sha256"], source_b["rgb"]["sha256"])
        self.assertEqual(second["identity_metadata"]["source_canvas_id"], native_pad((32, 64)).fingerprint())
        second_local.metadata[0]["local_scene_depth"]["source_asset_sha256"] = "polluted_again"
        second["pair_metadata"]["caller_only_again"] = True
        third_job = {**first_job, "pair_id": "composed_3"}
        self.assertRecordEqual(cache.assemble(third_job, renderer=renderer),
            scene_module.LocalSceneDepth(assemble_full_cache).assemble(third_job, renderer=renderer))
        self.assertEqual((cache.hits, cache.misses, aligned.aligned, self.calls), (2, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()

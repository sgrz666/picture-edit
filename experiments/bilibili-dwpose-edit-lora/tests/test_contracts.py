import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import core
except ModuleNotFoundError:
    core = None


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'The experiment data/condition contract is not implemented')

    def test_native_canvas_keeps_all_content(self):
        image = Image.new('RGB', (1080, 1920), 'white')
        prepared, affine = core.letterbox(image, (384, 672))
        self.assertEqual(prepared.size, (384, 672))
        self.assertEqual(affine['resized'], [378, 672])
        np.testing.assert_allclose(core.transform_points([[0, 0], [1080, 1920]], affine),
                                   [[3, 0], [381, 672]])

    def test_dwpose_uses_published_visibility_not_raw_score_threshold(self):
        record = {'candidate': [[50, 50]] * 134, 'valid_keypoint_mask': [False] * 134,
                  'confidence': [0.1] * 134, 'coordinate_space': 'original_pixel'}
        record['valid_keypoint_mask'][1] = True
        points, valid = core.pose_points(record, (100, 100))
        self.assertTrue(valid[1])
        self.assertEqual(int(valid.sum()), 1)

    def test_out_of_canvas_and_nan_are_not_clamped_into_visible_points(self):
        record = {'candidate': [[50, 50]] * 134, 'valid_keypoint_mask': [True] * 134}
        record['candidate'][0] = [100, 20]
        record['candidate'][1] = [float('nan'), 20]
        _, valid = core.pose_points(record, (100, 100))
        self.assertFalse(valid[0])
        self.assertFalse(valid[1])

    def test_missing_visibility_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'valid_keypoint_mask'):
            core.pose_points({'candidate': [[0, 0]] * 134}, (100, 100))

    def test_skeleton_has_opaque_black_background(self):
        points = np.zeros((134, 2))
        image = core.render_skeleton(points, np.zeros(134, dtype=bool), (384, 672))
        self.assertEqual(image.convert('RGBA').getpixel((0, 0)), (0, 0, 0, 255))

    def test_full_split_counts_and_cross_split_videos(self):
        rows = {'train': [{'pair_id': 'a', 'group_id': 'one'}],
                'val': [{'pair_id': 'b', 'group_id': 'one'}], 'test': []}
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            core.audit_splits(rows)

    def test_reused_frame_across_splits_is_rejected(self):
        rows = {'train': [{'pair_id': 'a', 'group_id': 'one', 'source_frame_id': 'same'}],
                'val': [{'pair_id': 'b', 'group_id': 'two', 'target_frame_id': 'same'}]}
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            core.audit_splits(rows)

    def test_pilot_is_deterministic_and_has_same_source_multiple_targets(self):
        rows = [dict(pair_id=f'{g}-{a}-{b}', group_id=str(g), source_frame_id=f'{g}-A{a}',
                     target_frame_id=f'{g}-B{b}')
                for g in range(10) for a in range(2) for b in range(6)]
        first = core.select_pilot(rows, videos=8, targets_per_source=4, seed=42)
        self.assertEqual(first, core.select_pilot(rows, 8, 4, 42))
        self.assertEqual(len(first), 32)
        self.assertEqual(len({r['group_id'] for r in first}), 8)
        for group in {r['group_id'] for r in first}:
            self.assertEqual(len({r['source_frame_id'] for r in first if r['group_id'] == group}), 1)

    def test_cache_identity_includes_pose_prompt_model_and_target(self):
        base = {'condition_sha256': ['a', 'pose'], 'target_sha256': 'b', 'prompt': 'short'}
        key = core.cache_key(base, {'model': 'v1', 'canvas': [384, 672]})
        for field, value in [('condition_sha256', ['a', 'other']), ('target_sha256', 'c'),
                             ('prompt', 'different')]:
            changed = copy.deepcopy(base)
            changed[field] = value
            self.assertNotEqual(key, core.cache_key(changed, {'model': 'v1', 'canvas': [384, 672]}))
        self.assertNotEqual(key, core.cache_key(base, {'model': 'v2', 'canvas': [384, 672]}))

    def test_changed_asset_bytes_are_rejected_even_if_manifest_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ('appearance.dat', 'pose.dat', 'target.dat')]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]))
            row = {'condition_images': [str(p) for p in paths[:2]],
                   'condition_sha256': [core.sha256(p) for p in paths[:2]],
                   'target_image': str(paths[2]), 'target_sha256': core.sha256(paths[2])}
            core.verify_row_assets(row)
            paths[1].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                core.verify_row_assets(row)

    def test_joint_layout_is_two_conditions_then_target(self):
        result = core.joint_layout([(42, 24), (42, 24)], (42, 24), 504)
        self.assertEqual(result['img_shapes'], [(1, 42, 24)] * 3)
        self.assertEqual(result['target_tokens'], 1008)
        self.assertEqual(result['target_slots'], 252)
        self.assertEqual(result['condition_tokens'], 2016)

    def test_layout_rejects_vlm_vae_slot_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'slots'):
            core.joint_layout([(42, 24), (42, 24)], (42, 24), 503)

    def test_latent_geometry_requires_even_spatial_dimensions(self):
        with self.assertRaises(ValueError):
            core.joint_layout([(41, 24)], (42, 24), 246)

    def test_new_outputs_cannot_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            core.write_json(path, {'first': True})
            with self.assertRaises(FileExistsError):
                core.write_json(path, {'second': True})

    def test_pose_metric_uses_dwpose134_not_body25_joint_indices(self):
        reference = np.zeros((134, 2))
        reference[5] = [40, 60]  # DWPose left shoulder -> COCO slot 5.
        reference[11] = [200, 300]  # DWPose left hip -> COCO slot 11.
        reference[13] = [300, 500]  # DWPose left ankle -> COCO slot 15.
        detected = np.zeros((17, 2))
        detected[5], detected[11], detected[15] = [40, 60], [200, 300], [300, 500]
        result = core.pose_stats((detected, np.ones(17)), reference, np.ones(134, dtype=bool), (384, 672))
        self.assertEqual(result['pck_005_diagonal'], 1.0)

    def test_failed_pose_detection_is_counted_as_misses(self):
        result = core.pose_stats(None, np.zeros((134, 2)), np.ones(134, dtype=bool), (384, 672))
        self.assertEqual(result['pck_005_diagonal'], 0.0)
        self.assertEqual(result['matched_joints'], 0)
        self.assertEqual(result['target_joints'], 17)


if __name__ == '__main__':
    unittest.main()

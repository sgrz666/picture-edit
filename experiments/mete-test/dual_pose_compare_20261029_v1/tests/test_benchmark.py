import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark.py'


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available = SCRIPT.is_file()
        if cls.available:
            spec = importlib.util.spec_from_file_location('benchmark', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.module
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(self.available, 'benchmark implementation does not exist yet')
        self.b = self.module
        x = np.array([[20, 10], [60, 20], [10, 30], [80, 50], [5, 70], [90, 90],
                      [25, 60], [55, 65], [20, 100], [60, 115], [15, 150], [65, 170]], float)
        self.gt = np.stack([x, x + [200, 0]])
        self.record = {'id': 'example', 'size': [400, 200], 'gt_xy': self.gt.tolist(),
                       'gt_boxes': [[0, 0, 100, 200], [180, 0, 380, 200]],
                       'gt_valid': np.ones((2, 12), bool).tolist(),
                       'occlusion_proxy': np.zeros((2, 12), int).tolist()}

    def prediction(self, points):
        return {'status': 'ok', 'det_count': len(points), 'instances': [
            {'xy': p.tolist(), 'scores': [1.] * 12,
             'joint_names': list(self.b.BODY_NAMES)} for p in points]}

    def test_sapiens_mapping_is_named_not_coco(self):
        meta = {i: {'name': f'other_{i}'} for i in range(308)}
        expected = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
        for name, index in zip(self.b.BODY_NAMES, expected):
            meta[index] = {'name': name}
        self.assertEqual(self.b.named_indices(meta), expected)
        self.assertNotEqual(expected, list(range(5, 17)))
        del meta[62]
        with self.assertRaises(ValueError):
            self.b.named_indices(meta)

    def test_prediction_order_is_not_identity_swap(self):
        a = self.b.score_frame(self.record, self.prediction(self.gt))
        b = self.b.score_frame(self.record, self.prediction(self.gt[::-1]))
        self.assertEqual(a['e2e_pck10'], 1.)
        self.assertEqual(b['e2e_pck10'], 1.)
        self.assertEqual(a['mean_error_px'], b['mean_error_px'])
        self.assertEqual(b['assignment'], [1, 0])

    def test_missing_person_lowers_end_to_end_coverage(self):
        score = self.b.score_frame(self.record, self.prediction(self.gt[:1]))
        self.assertEqual(score['matched_pck10'], 1.)
        self.assertEqual(score['e2e_pck10'], .5)
        self.assertEqual(score['gt_joint_count'], 24)
        self.assertEqual(score['matched_joint_count'], 12)
        self.assertFalse(score['two_person_count_ok'])

    def test_each_person_uses_own_box_scale(self):
        offset = self.gt + [24., 0.]
        score = self.b.score_frame(self.record, self.prediction(offset))
        self.assertEqual(score['e2e_pck10'], .5)

    def test_zero_pose_and_swapped_anatomy_are_not_perfect(self):
        zero = self.b.score_frame(self.record, self.prediction(np.zeros_like(self.gt)))
        swapped = self.gt[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]]
        reversed_score = self.b.score_frame(self.record, self.prediction(swapped))
        self.assertLess(zero['e2e_pck10'], .1)
        self.assertLess(reversed_score['e2e_pck10'], .5)

    def test_strict_metric_separates_near_misses(self):
        exact = self.b.score_frame(self.record, self.prediction(self.gt))
        shifted = self.b.score_frame(self.record, self.prediction(self.gt + [10., 0.]))
        self.assertEqual(exact.get('e2e_pck02'), 1.)
        self.assertEqual(shifted.get('e2e_pck02'), 0.)
        self.assertEqual(shifted['e2e_pck10'], 1.)

    def test_cached_gtbox_retains_roi_binding_diagnostic(self):
        pred = self.prediction(self.gt[::-1])
        pred['mode'] = 'sapiens_gtbox_cached'
        score = self.b.score_frame(self.record, pred)
        self.assertIs(score['roi_assignment_disagrees'], True)

    def test_cross_person_count_needs_a_discriminable_denominator(self):
        gt = self.gt.copy()
        gt[1, 0] = gt[0, 0] + [20., 0.]
        record = dict(self.record, gt_xy=gt.tolist())
        swapped = self.prediction(gt)
        swapped['instances'][1]['xy'][0] = gt[0, 0].tolist()
        swapped['instances'][1]['xy'][1] = gt[0, 1].tolist()
        score = self.b.score_frame(record, swapped)
        self.assertEqual(score['cross_person_joint_candidates'], 2)
        # 2 people x 12 joints minus the two shoulders whose GT sit 20 px apart (below 10% of
        # the 223.6 px diagonal). Only person 1's joint 1 counts as cross among those 22.
        self.assertEqual(score['cross_person_discriminable_joints'], 22)
        self.assertAlmostEqual(score['cross_person_rate_discriminable'], 1 / 22)
        summary = self.b.summarize([score])
        self.assertEqual(summary['cross_person_discriminable_joints'], 22)
        self.assertAlmostEqual(summary['cross_person_rate_discriminable'], 1 / 22)

    def test_no_3d_is_null_not_zero(self):
        score = self.b.score_frame(self.record, self.prediction(self.gt))
        self.assertIsNone(score['root_mpjpe_mm'])
        self.assertIsNone(score['absolute_mpjpe_mm'])

    def test_root_alignment_does_not_hide_absolute_translation(self):
        xyz = np.concatenate([self.gt / 100, np.ones((2, 12, 1)) * 3], axis=-1)
        roots = np.array([[.4, .6, 3.], [2.4, .6, 3.]])
        self.record.update(gt_xyz=xyz.tolist(), gt_pelvis=roots.tolist())
        pred = self.prediction(self.gt)
        pred['scene_camera_available'] = True
        for i, person in enumerate(pred['instances']):
            person.update(xyz=(xyz[i] + [0, 0, 1]).tolist(),
                          pelvis=(roots[i] + [0, 0, 1]).tolist())
        score = self.b.score_frame(self.record, pred)
        self.assertAlmostEqual(score['root_mpjpe_mm'], 0.)
        self.assertAlmostEqual(score['absolute_mpjpe_mm'], 1000.)
        self.assertAlmostEqual(score['pelvis_vector_error_mm'], 0.)

    def test_error_or_empty_output_keeps_full_denominator(self):
        for pred in ({'status': 'error', 'error': 'model failed'}, self.prediction([])):
            score = self.b.score_frame(self.record, pred)
            self.assertEqual(score['e2e_pck10'], 0.)
            self.assertEqual(score['gt_joint_count'], 24)
            self.assertIsNone(score['mean_error_px'])

    def test_duplicate_record_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.b.validate_records([self.record, self.record])

    def test_request_mode_and_id_must_match_prediction(self):
        checker = getattr(self.b, 'validate_prediction', None)
        self.assertIsNotNone(checker)
        record = dict(self.record, image_sha256='a' * 64)
        pred = {'id': record['id'], 'mode': 'dwpose_auto', 'image_sha256': 'a' * 64, 'request_sha256': 'b' * 64}
        checker(record, pred, 'dwpose_auto', 'b' * 64)
        for key in ('id', 'mode', 'image_sha256', 'request_sha256'):
            with self.assertRaises(ValueError):
                checker(record, dict(pred, **{key: 'wrong'}), 'dwpose_auto', 'b' * 64)

    def test_3d_coverage_cannot_hide_missing_person(self):
        xyz = np.concatenate([self.gt / 100, np.ones((2, 12, 1)) * 3], axis=-1)
        roots = np.array([[.4, .6, 3.], [2.4, .6, 3.]])
        self.record.update(gt_xyz=xyz.tolist(), gt_pelvis=roots.tolist())
        pred = self.prediction(self.gt)
        pred['mode'] = 'multihmr'
        pred['instances'][0].update(xyz=xyz[0].tolist(), pelvis=roots[0].tolist())
        score = self.b.score_frame(self.record, pred)
        self.assertEqual(score.get('gt_3d_joint_count'), 24)
        self.assertEqual(score.get('matched_3d_joint_count'), 12)
        self.assertEqual(score.get('e2e_root_pck150mm'), .5)

    def test_occlusion_bucket_keeps_missing_gt_denominator(self):
        self.record['occlusion_proxy'] = np.ones((2, 12), int).tolist()
        row = self.b.score_frame(self.record, self.prediction(self.gt[:1]))
        summary = self.b.summarize([row])
        self.assertEqual(summary['other_mask'].get('gt_joints'), 24)
        self.assertEqual(summary['other_mask'].get('e2e_pck10'), .5)

    def test_guess_fov_arm_gets_no_absolute_metric(self):
        xyz = np.concatenate([self.gt / 100, np.ones((2, 12, 1)) * 3], axis=-1)
        roots = np.array([[.4, .6, 3.], [2.4, .6, 3.]])
        self.record.update(gt_xyz=xyz.tolist(), gt_pelvis=roots.tolist())
        pred = self.prediction(self.gt)
        pred.update(mode='multihmr_fov60', scene_camera_available=True,
                    provenance={'absolute_3d_comparable': False})
        for i, person in enumerate(pred['instances']):
            person.update(xyz=(xyz[i] + [0, 0, 1]).tolist(), pelvis=(roots[i] + [0, 0, 1]).tolist())
        score = self.b.score_frame(self.record, pred)
        self.assertAlmostEqual(score['root_mpjpe_mm'], 0.)
        self.assertIsNone(score['absolute_mpjpe_mm'])
        self.assertIsNone(score['pelvis_vector_error_mm'])

    def test_project_camera_uses_known_extrinsic_direction(self):
        xyz = np.array([[[1., 2., 3.]]])
        k = np.array([[100., 0., 50.], [0., 200., 60.], [0., 0., 1.]])
        e = np.column_stack([np.eye(3), [0., 0., 1.]])
        camera, xy = self.b.project_camera(xyz, k, e)
        np.testing.assert_allclose(camera, [[[1., 2., 4.]]])
        np.testing.assert_allclose(xy, [[[75., 160.]]])


if __name__ == '__main__':
    unittest.main()

"""Integration gates against real fitted models and exported calibrated geometry."""
from pathlib import Path
import json
import os
import unittest
import numpy as np

WORK = Path(__file__).parent
FIT = WORK / os.environ.get('HI4D_SMPLX_FIT', 'fit_v1')


class ExportTests(unittest.TestCase):
    def test_four_bodies_and_shared_identity_shape(self):
        z = np.load(FIT / 'smplx_world.npz')
        self.assertEqual(z['vertices_world'].shape, (4, 10475, 3))
        self.assertEqual(z['body_pose'].shape, (4, 63))
        np.testing.assert_array_equal(z['betas'][:2], z['betas'][2:])
        np.testing.assert_array_equal(z['actor_ids'], [0, 1, 0, 1])
        for k in ['vertices_world', 'joints_world', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl']:
            self.assertTrue(np.isfinite(z[k]).all(), k)

    def test_parameters_reproduce_real_saved_mesh(self):
        import smplx
        import torch
        root = WORK.parents[1]
        model = smplx.create(str(root / 'add/tools/SMPLest-X/human_models/human_model_files'), 'smplx', gender='neutral', batch_size=4, use_pca=False, flat_hand_mean=True, use_face_contour=True)
        z = np.load(FIT / 'smplx_world.npz')
        names = ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'transl', 'leye_pose', 'reye_pose']
        with torch.no_grad():
            result = model(**{k: torch.tensor(z[k]) for k in names})
        np.testing.assert_allclose(result.vertices.numpy(), z['vertices_world'], atol=2e-6)

    def test_body_alignment_improves_for_both_actors_in_both_frames(self):
        z = np.load(FIT / 'smplx_world.npz')
        root = WORK.parents[1]
        j = json.loads((FIT / 'result.json').read_text())
        for frame, (action, fid) in enumerate([('talk14', 32), ('hug14', 62)]):
            truth = np.load(root / f'datasets/Hi4D_pilot_v1/raw/pair14/{action}/smpl/{fid:06d}.npz')['joints_3d'][:, :22]
            errors = np.linalg.norm(z['joints_world'][2 * frame:2 * frame + 2, :22] - truth, axis=-1).mean(-1)
            for actor in range(2):
                self.assertLess(errors[actor], .03)
                self.assertLess(errors[actor] * 1000, j['metrics'][frame]['actors'][actor]['body_joint_before_mm'])

    def test_calibrated_exports_and_camera_mesh_roundtrip(self):
        path = FIT / 'visuals/export_checks.json'
        self.assertTrue(path.exists(), 'Calibrated SMPL-X and skeleton exports have not been implemented')
        if not path.exists():
            return
        checks = json.loads(path.read_text())
        self.assertEqual(len(checks['views']), 16)
        for view in checks['views']:
            self.assertLess(view['mesh_camera_roundtrip_max_m'], 2e-5)
            self.assertLess(view['raster_centroid_error_px'], 30)
            self.assertTrue(view['render_pixels'] > 1000)

    def test_missing_hands_are_marked_as_estimated(self):
        j = json.loads((FIT / 'result.json').read_text())
        self.assertEqual(j['annotation_type'], 'fitted_pseudo_annotation_not_official_ground_truth')
        for actor in j['metrics'][1]['actors']:
            for hand in actor['hands']:
                if hand['points_with_3_views'] < 8:
                    self.assertEqual(hand['status'], 'mostly_model_prior')


if __name__ == '__main__':
    unittest.main()

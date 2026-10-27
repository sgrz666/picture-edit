"""CPU contracts for shared-camera ABC geometry exports."""
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class ABCSceneRenderingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('render.abc_scene')
        except ModuleNotFoundError as error:
            self.fail(f'ABC scene renderer is missing: {error}')

    def test_owner_uses_positive_finite_z_and_keeps_hidden_person_mask(self):
        depth = np.array([[[2., 3., 0., 4.]], [[3., 2., 1., 4.]]])
        masks = np.array([[[True, True, False, True]], [[True, True, True, True]]])
        owner = self.module.visible_ownership(depth, masks)
        np.testing.assert_array_equal(owner[:, 0], [[1, 0, 0, 1], [0, 1, 1, 0]])
        self.assertTrue(masks[1, 0, 0], 'occlusion must not destroy the full person mask')
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.module.visible_ownership(np.array([[[-1.]], [[2.]]]), np.ones((2, 1, 1), bool))

    def test_parts_use_actual_triangle_vertex_labels_and_background_is_separate(self):
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        np.testing.assert_array_equal(self.module.face_labels(faces, [2, 2, 3, 3]), [2, 3])
        with self.assertRaisesRegex(ValueError, '0..13'):
            self.module.face_labels(faces, [0, 0, 14, 14])

    def test_contact_requires_explicit_correspondences_and_measures_target_vertices(self):
        people = [dict(vertices_cam=np.array([[0., 0., 2.], [1., 0., 2.]])),
                  dict(vertices_cam=np.array([[.03, 0., 2.], [2., 0., 2.]]))]
        missing = self.module.contact_evidence(people)
        self.assertEqual(missing['status'], 'unavailable')
        self.assertEqual(missing['indices'].shape, (0, 4))
        known = self.module.contact_evidence(people, [[0, 0, 1, 0]])
        np.testing.assert_allclose(known['distance_m'], [.03])
        self.assertEqual(known['status'], 'explicit_vertex_correspondences')
        with self.assertRaisesRegex(ValueError, 'different'):
            self.module.contact_evidence(people, [[0, 0, 0, 1]])

    def test_projection_uses_shared_pinhole_without_depth_epsilon(self):
        k = np.array([[100., 0., 64.], [0., 200., 32.], [0., 0., 1.]])
        uv = self.module.project_points(np.array([[1., .5, 2.], [0., 0., -1.]]), k)
        np.testing.assert_allclose(uv[0], [114., 82.], atol=1e-10)
        self.assertTrue(np.isnan(uv[1]).all())

    def test_export_keeps_dense_arrays_params_owner_and_contact_evidence(self):
        h, w = 32, 32
        people = []
        for index in range(2):
            people.append(dict(vertices_cam=np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.]]),
                faces=np.array([[0, 1, 2]]), part_vertex=np.array([index, index, index]),
                joints_cam=np.tile([0., 0., 2.], (137, 1)), body_pose=np.zeros(63),
                global_orient=np.zeros(3), transl=np.array([0., 0., 2.]), betas=np.zeros(10)))

        class FixtureRasterizer:
            """A supplied dense raster tests export, never replaces production CUDA."""
            def render(self, person, k, hw):
                mask = np.zeros(hw, bool); mask[4:12, 5:13] = True
                return dict(z=mask.astype(np.float32) * 2., valid=mask,
                    normal=np.stack([mask, mask * 0, mask * 0], -1).astype(np.float32),
                    part=mask.astype(np.uint8) * (int(person['face_part'][0]) + 1))

        with tempfile.TemporaryDirectory() as directory:
            result = self.module.render_scene(people, np.array([[50.,0.,16.],[0.,50.,16.],[0.,0.,1.]]),
                (h, w), directory, rasterizer=FixtureRasterizer())
            with np.load(result['geometry_arrays'], allow_pickle=False) as saved:
                self.assertEqual(saved['normal_cam'].shape, (2, h, w, 3))
                self.assertEqual(saved['joints_cam'].shape, (2, 137, 3))
                self.assertEqual(saved['body_pose'].shape, (2, 63))
                self.assertEqual(saved['visible_owner'][1].sum(), 0)
                self.assertEqual(saved['mask'][1].sum(), 64)
                self.assertEqual(saved['part'][1].max(), 2)
                self.assertEqual(saved['contact_indices'].shape, (0, 4))
            self.assertEqual(set(result['guides']), {'person_0', 'person_1', 'combined'})
            for path in result['guides'].values():
                self.assertTrue(Path(path).is_file())
            self.assertEqual(result['diagnostics']['contact_status'], 'unavailable')
            with self.assertRaises(FileExistsError):
                self.module.render_scene(people, np.eye(3), (h, w), directory,
                    rasterizer=FixtureRasterizer())


if __name__ == '__main__':
    unittest.main()

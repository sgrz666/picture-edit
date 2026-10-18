"""Coordinate, association and triangulation checks for the Hi4D pilot."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('geometry.py')
        if not path.exists():
            cls.g = None
            return
        spec = importlib.util.spec_from_file_location('pilot_geometry', path)
        cls.g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.g)

    def setUp(self):
        self.assertIsNotNone(self.g, 'The calibrated geometry utilities are not implemented yet')

    def test_intrinsics_follow_actual_resize_and_padding(self):
        K = np.array([[1100., 0, 450], [0, 1120., 640], [0, 0, 1]])
        A = self.g.resize_pad_affine((940, 1280), (658, 896), (119, 0))
        X = np.array([[.2, -.3, 3.]])
        native = self.g.project(X, K, np.eye(4)[:3])
        expected = np.c_[native, np.ones(1)] @ A.T
        np.testing.assert_allclose(self.g.project(X, A @ K, np.eye(4)[:3]), expected[:, :2])

    def test_camera_world_roundtrip(self):
        E = np.array([[0., -1, 0, .3], [1, 0, 0, -.2], [0, 0, 1, 2]])
        X = np.array([[.2, .3, .4], [-.1, .4, -.3]])
        np.testing.assert_allclose(self.g.camera_to_world(X @ E[:, :3].T + E[:, 3], E), X, atol=1e-12)

    def test_slot_assignment_uses_geometry_not_left_right(self):
        truth = np.array([[[20., 50], [20, 80]], [[90., 50], [90, 80]]])
        candidates = truth[::-1].copy()
        assigned, cost = self.g.associate(candidates, np.ones((2, 2)), truth)
        self.assertEqual(assigned.tolist(), [1, 0])
        np.testing.assert_allclose(cost, 0)

    def test_missing_person_keeps_slot_missing(self):
        truth = np.array([[[20., 50], [20, 80]], [[90., 50], [90, 80]]])
        assigned, _ = self.g.associate(truth[:1], np.ones((1, 2)), truth)
        self.assertEqual(assigned.tolist(), [0, -1])

    def test_low_confidence_candidates_are_not_bound(self):
        truth = np.zeros((2, 4, 2))
        assigned, _ = self.g.associate(truth.copy(), np.zeros((2, 4)), truth)
        self.assertEqual(assigned.tolist(), [-1, -1])

    def test_triangulation_rejects_outlier_and_missing_views(self):
        K = np.array([[800., 0, 400], [0, 800, 400], [0, 0, 1]])
        Es = np.stack([np.c_[np.eye(3), [x, 0, 3]] for x in [-.7, 0, .7]])
        X = np.array([[.12, .3, .2]])
        uv = np.stack([self.g.project(X, K, E)[0] for E in Es])
        uv[2] += 200
        recovered, valid, error, views = self.g.triangulate(uv[:, None], np.ones((3, 1)), np.repeat(K[None], 3, 0), Es)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(recovered[0], X[0], atol=1e-8)
        self.assertEqual(views[0], 2)
        scores = np.array([[1.], [0.], [0.]])
        _, valid, _, _ = self.g.triangulate(uv[:, None], scores, np.repeat(K[None], 3, 0), Es)
        self.assertFalse(valid[0])

    def test_wholebody_layout_has_correct_sides_and_face_order(self):
        idx = self.g.wholebody134_indices()
        self.assertEqual(len(idx), 134)
        self.assertEqual(idx[:5].tolist(), [55, 12, 17, 19, 21])
        self.assertEqual(idx[18:24].tolist(), [60, 61, 62, 63, 64, 65])
        self.assertEqual(idx[24:41].tolist(), list(range(127, 144)))
        self.assertEqual(idx[41:92].tolist(), list(range(76, 127)))
        self.assertEqual(idx[92:97].tolist(), [20, 37, 38, 39, 66])
        self.assertEqual(idx[113:118].tolist(), [21, 52, 53, 54, 71])


if __name__ == '__main__':
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(ROOT))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'prepare.py').is_file(), 'preparation implementation absent')
        import prepare
        self.p = prepare

    def test_mask_ownership_is_not_inferred_from_same_joint_depth(self):
        a = np.zeros((10, 10), bool)
        b = a.copy()
        a[2, 2] = True
        b[3, 3] = True
        a[4, 4] = b[4, 4] = True
        points = np.array([[[2, 2], [3, 3], [4, 4], [-1, 2]],
                           [[2, 2], [3, 3], [4, 4], [20, 20]]])
        expected = [[0, 1, -1, -1], [1, 0, -1, -1]]
        self.assertEqual(self.p.mask_ownership(points, [a, b]), expected)

    def test_box_is_xyxy_with_positive_extent(self):
        mask = np.zeros((10, 10), bool)
        mask[2:5, 3:7] = True
        self.assertEqual(self.p.mask_box(mask), [3., 2., 7., 5.])
        with self.assertRaises(ValueError):
            self.p.mask_box(np.zeros((10, 10), bool))

    def test_overlap_bins_are_disjoint(self):
        self.assertEqual(self.p.category_from_boxes([[0, 0, 10, 10], [20, 0, 30, 10]])[0], 'separated')
        self.assertEqual(self.p.category_from_boxes([[0, 0, 10, 10], [1, 0, 11, 10]])[0], 'severe')

    def test_temporal_guard_rejects_adjacent_frames(self):
        chosen = [{'pair': 'p', 'action': 'a', 'frame_id': 30}]
        self.assertFalse(self.p.temporally_separate({'pair': 'p', 'action': 'a', 'frame_id': 35}, chosen, 15))
        self.assertTrue(self.p.temporally_separate({'pair': 'p', 'action': 'b', 'frame_id': 35}, chosen, 15))
        self.assertTrue(self.p.temporally_separate({'pair': 'p', 'action': 'a', 'frame_id': 50}, chosen, 15))


if __name__ == '__main__':
    unittest.main()

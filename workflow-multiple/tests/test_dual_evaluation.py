import importlib
import unittest

import numpy as np

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class RegionBoxTests(unittest.TestCase):
    """The region count exists because whole-image detection merges embracing bodies."""

    def setUp(self):
        self.module = importlib.import_module('evaluate_dual')

    def body(self, x, y=0.0):
        points = np.full((134, 2), np.nan)
        points[1] = [x, 100 + y]
        points[8] = [x - 20, 300 + y]
        points[11] = [x + 20, 300 + y]
        return points

    def test_each_requested_person_gets_its_own_padded_box(self):
        boxes = self.module.region_boxes([self.body(200), self.body(500)], (940, 1280))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0][:2], [172, 60])
        self.assertEqual(boxes[0][2:], [228, 340])
        self.assertLess(boxes[0][2], boxes[1][0], 'the two regions should not overlap here')

    def test_regions_are_clamped_to_the_frame_and_degenerate_ones_are_dropped(self):
        boxes = self.module.region_boxes([self.body(5), self.body(935)], (940, 1280))
        self.assertEqual(boxes[0][0], 0)
        self.assertLessEqual(boxes[1][2], 940)
        self.assertIsNone(self.module.region_boxes([np.full((134, 2), np.nan)], (940, 1280))[0])
        tiny = np.full((134, 2), np.nan)
        tiny[1] = [100, 100]
        tiny[8] = [100, 101]
        tiny[11] = [101, 101]
        self.assertIsNone(self.module.region_boxes([tiny], (940, 1280))[0])

    def test_a_region_is_kept_even_when_the_two_people_overlap_heavily(self):
        # The case that motivated this measure: an embrace, where whole-image detection returns one
        # box. Each person's own region must still be cropped and counted separately.
        boxes = self.module.region_boxes([self.body(400), self.body(430)], (940, 1280))
        self.assertTrue(all(box is not None for box in boxes))
        overlap = min(boxes[0][2], boxes[1][2]) - max(boxes[0][0], boxes[1][0])
        self.assertGreater(overlap, 0)


if __name__ == '__main__':
    unittest.main()

import importlib
import unittest

import numpy as np

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class TorsoRatioTests(unittest.TestCase):
    """A measurement tool is only as good as its box and pairing rules."""

    def setUp(self):
        self.module = importlib.import_module('torso_ratio')

    def test_mask_box_is_the_tight_bounds_of_the_silhouette(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:12, 7:19] = 1
        self.assertEqual(self.module.mask_box(mask), [7.0, 5.0, 19.0, 12.0])
        self.assertIsNone(self.module.mask_box(np.zeros((10, 10), dtype=np.uint8)))

    def test_ratio_divides_torso_by_box_height_and_refuses_either_missing(self):
        points = np.full((134, 2), np.nan)
        points[1] = [10, 10]     # neck
        points[8] = [0, 50]      # right hip
        points[11] = [20, 50]    # left hip
        measurement = self.module.ratio(points, [0, 10, 20, 110])
        self.assertAlmostEqual(measurement['torso_px'], 40.0)
        self.assertEqual(measurement['box_height_px'], 100.0)
        self.assertAlmostEqual(measurement['ratio'], 0.4)
        self.assertIsNone(self.module.ratio(points, None)['ratio'])
        self.assertIsNone(self.module.ratio(np.full((134, 2), np.nan), [0, 0, 10, 10])['ratio'])

    def test_the_person_paired_to_a_box_is_the_one_whose_hips_fall_inside_it(self):
        def person(x):
            points = np.full((134, 2), np.nan)
            points[1] = [x, 10]
            points[8] = [x - 1, 50]
            points[11] = [x + 1, 50]
            return {'points': points}

        people = [person(20), person(120)]
        self.assertEqual(self.module.assign_to_box(people, [0, 0, 40, 100]), 0)
        self.assertEqual(self.module.assign_to_box(people, [100, 0, 140, 100]), 1)
        self.assertIsNone(self.module.assign_to_box(people, [300, 0, 400, 100]))

    def test_the_summary_reports_the_spread_not_just_the_centre(self):
        measurements = [{'ratio': value, 'frame': 'A', 'input_sha256': f'sha{index}',
            'box_xyxy': [0.0, 0.0, 10.0, float(100 + index)]}
            for index, value in enumerate((0.25, 0.30, 0.35))]
        summary = self.module.summarise(measurements)
        self.assertEqual(summary['measured'], 3)
        self.assertEqual(summary['samples_raw'], 3)
        self.assertAlmostEqual(summary['median'], 0.30)
        self.assertAlmostEqual(summary['ratio_to_constant'], 0.30 / self.module.CONSTANT)
        self.assertEqual(summary['by_frame']['A']['n'], 3)

    def test_the_same_body_measured_twice_counts_once(self):
        # Re-run directories and pairs that share a source frame must not tighten the spread.
        one = {'ratio': 0.30, 'frame': 'A', 'input_sha256': 'same', 'box_xyxy': [0, 0, 10, 100]}
        twice = dict(one, ratio=0.31)
        summary = self.module.summarise([one, twice])
        self.assertEqual(summary['samples_raw'], 2)
        self.assertEqual(summary['measured'], 1)
        self.assertAlmostEqual(summary['median'], 0.30)


if __name__ == '__main__':
    unittest.main()

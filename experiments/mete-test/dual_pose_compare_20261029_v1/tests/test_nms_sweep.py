import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'nms_sweep.py'
sys.path.insert(0, str(SCRIPT.parent))


class NmsSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available = SCRIPT.is_file()
        if cls.available:
            spec = importlib.util.spec_from_file_location('nms_sweep', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.module
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(self.available, 'nms_sweep implementation missing')
        self.m = self.module

    def test_greedy_nms_keeps_overlap_only_below_threshold(self):
        boxes = [[0, 0, 10, 10], [2, 0, 12, 10], [50, 50, 60, 60]]
        kept = self.m.greedy_nms(boxes, [.9, .8, .7], .5)
        self.assertEqual(kept, [0, 2])
        self.assertEqual(self.m.greedy_nms(boxes, [.9, .8, .7], .9), [0, 1, 2])

    def test_distinct_people_requires_one_to_one_assignment(self):
        gt = [[0, 0, 10, 10], [6, 0, 16, 10]]
        two_on_first = [[0, 0, 10, 10], [1, 0, 11, 10]]
        one_each = [[0, 0, 10, 10], [6, 0, 16, 10]]
        self.assertEqual(self.m.distinct_people(two_on_first, gt), 1)
        self.assertEqual(self.m.distinct_people(one_each, gt), 2)

    def test_sweep_reports_every_threshold(self):
        frames = [{'id': 'a', 'gt_boxes': [[0, 0, 10, 10], [30, 0, 40, 10]],
                   'raw_boxes': [[0, 0, 10, 10], [30, 0, 40, 10]], 'raw_scores': [.9, .8]}]
        rows = self.m.sweep(frames, [.3, .7])
        self.assertEqual([r['nms_thr'] for r in rows], [.3, .7])
        self.assertEqual([r['frames_with_two_distinct_people'] for r in rows], [1, 1])

    def test_top_two_gate_drops_only_extra_boxes(self):
        frames = [{'id': 'a', 'gt_boxes': [[0, 0, 10, 10], [30, 0, 40, 10]],
                   'raw_boxes': [[0, 0, 10, 10], [30, 0, 40, 10], [70, 70, 80, 80]],
                   'raw_scores': [.9, .8, .2]}]
        row = self.m.sweep(frames, [.7])[0]
        self.assertEqual(row['det_box_histogram'], {'3': 1})
        self.assertEqual(row['frames_with_two_distinct_people'], 1)
        self.assertEqual(row['frames_with_exactly_two_matching'], 0)
        self.assertEqual(row['capped_two_distinct_frames'], 1)


if __name__ == '__main__':
    unittest.main()

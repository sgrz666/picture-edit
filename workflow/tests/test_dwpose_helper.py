#!/usr/bin/env python3
"""Contract tests for src.pose_edit.dwpose: the one place the 133->134 convention lives."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit import dwpose


class BoxIouTests(unittest.TestCase):
    def test_identical_and_disjoint_boxes(self):
        self.assertAlmostEqual(dwpose.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(dwpose.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(dwpose.box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


class PickPersonTests(unittest.TestCase):
    def setUp(self):
        self.people = [{'box_xyxy': [0.0, 0.0, 10.0, 10.0], 'tag': 'left'},
            {'box_xyxy': [50.0, 50.0, 90.0, 90.0], 'tag': 'right'}]

    def test_choice_follows_overlap_not_detection_order(self):
        self.assertEqual(dwpose.pick_person(self.people, [52.0, 48.0, 88.0, 92.0])['tag'], 'right')
        self.assertEqual(dwpose.pick_person(list(reversed(self.people)), [1.0, 1.0, 9.0, 9.0])['tag'],
            'left')

    def test_no_overlap_is_no_person(self):
        self.assertIsNone(dwpose.pick_person(self.people, [400.0, 400.0, 420.0, 420.0]))
        self.assertIsNone(dwpose.pick_person([], [0.0, 0.0, 1.0, 1.0]))


class ConversionTests(unittest.TestCase):
    def test_133_slots_become_134_and_the_neck_score_is_the_weaker_shoulder(self):
        raw_points = np.zeros((133, 2))
        raw_scores = np.full((133,), 0.7)
        raw_scores[5], raw_scores[6] = 0.9, 0.4
        points, scores = dwpose.convert_person(raw_points, raw_scores)
        self.assertEqual(points.shape, (134, 2))
        self.assertEqual(scores.shape, (134,))
        self.assertAlmostEqual(scores[1], 0.4)

    def test_neck_point_is_the_shoulder_midpoint(self):
        raw_points = np.zeros((133, 2))
        raw_points[5] = [10.0, 20.0]
        raw_points[6] = [30.0, 40.0]
        points, _ = dwpose.convert_person(raw_points, np.full((133,), 0.5))
        np.testing.assert_allclose(points[1], [20.0, 30.0])


class MeasureTorsoTests(unittest.TestCase):
    def test_measured_person_returns_its_neck_to_hip_length(self):
        points = np.full((134, 2), np.nan)
        points[1] = [10.0, 20.0]
        points[8] = [5.0, 120.0]
        points[11] = [15.0, 120.0]
        original = dwpose.detect_people
        dwpose.detect_people = lambda config, image_bgr, sessions=None: [
            {'box_xyxy': [0.0, 0.0, 20.0, 140.0], 'points': points,
                'scores': np.full((134,), 0.9)}]
        try:
            self.assertAlmostEqual(
                dwpose.measure_torso({}, np.zeros((4, 4, 3)), [0.0, 0.0, 20.0, 140.0]), 100.0)
        finally:
            dwpose.detect_people = original

    def test_measure_torso_returns_none_when_no_person_matches_the_box(self):
        calls = []
        original = dwpose.detect_people

        def fake(config, image_bgr, sessions=None):
            calls.append(image_bgr)
            return []
        dwpose.detect_people = fake
        try:
            self.assertIsNone(dwpose.measure_torso({}, np.zeros((4, 4, 3)), [0, 0, 2, 2]))
        finally:
            dwpose.detect_people = original
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class DualFittingTests(unittest.TestCase):
    """The fit-to-guide path must map joints the same way the fit report does, or not at all."""

    NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
        'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar',
        'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist'] + [f'hand{index}' for index in range(33)]

    def setUp(self):
        self.fitting = importlib.import_module('dual_fitting')
        self.geometry = importlib.import_module('dual_geometry')

    def joints(self, shift=0.0):
        points = np.zeros((len(self.NAMES), 2))
        for index, name in enumerate(self.NAMES):
            points[index] = [index * 10.0 + shift, 100.0 + index * 5.0]
        return points

    def test_only_the_thirteen_mesh_joints_are_placed_and_the_rest_stay_absent(self):
        person = self.fitting.person_from_joints(self.joints(), self.NAMES, (1000, 1000))
        points = person['points']
        self.assertEqual(points.shape, (134, 2))
        for slot in self.fitting.DWPOSE_TO_JOINT:
            self.assertTrue(np.isfinite(points[slot]).all(), f'slot {slot} should be placed')
        for slot in (0, 14, 15, 16, 17, 18, 60, 133):
            self.assertFalse(np.isfinite(points[slot]).any(), f'slot {slot} should stay absent')

    def test_a_skeleton_without_the_required_names_is_refused_not_guessed(self):
        with self.assertRaises(KeyError):
            self.fitting.joint_indices(['pelvis', 'left_hip', 'right_hip'])

    def write_fit(self, directory, positions):
        path = Path(directory) / 'fit.npz'
        j2d = np.array([self.joints(offset) for offset in positions], dtype=np.float32)
        np.savez_compressed(path, j2d_native=j2d,
            joint_names=np.array(self.NAMES, dtype='U32'), j3d=np.zeros((len(positions),
                len(self.NAMES), 3), dtype=np.float32))
        return path

    def test_people_are_ordered_left_to_right_by_hip_centre_like_the_slots_are(self):
        with tempfile.TemporaryDirectory() as directory:
            # Written right-first on purpose: the fit's own row order must not decide the slots.
            path = self.write_fit(directory, [400.0, 0.0])
            people, provenance = self.fitting.load_fit(path, (1000, 1000))
            first = np.nanmean(people[0]['points'][[8, 11]], axis=0)
            second = np.nanmean(people[1]['points'][[8, 11]], axis=0)
            self.assertLess(first[0], second[0])
            self.assertEqual(provenance['fit_person_count'], 2)
            self.assertEqual(provenance['slot_order_by_hip_x'], [1, 0])

    def test_a_fit_that_is_not_exactly_two_people_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fit(directory, [0.0])
            with self.assertRaises(ValueError):
                self.fitting.load_fit(path, (1000, 1000))

    def pair(self):
        return {'B': {'size': [1000, 1000],
            'slot_order': [{'slot': 1, 'hi4d_side': 'A'}, {'slot': 2, 'hi4d_side': 'B'}],
            'keypoints': []}}

    def test_the_guide_builder_draws_the_fit_through_the_same_slot_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fit(directory, [0.0, 400.0])
            persons, meta = self.geometry.guide_persons(self.pair(), Path(directory), 0.5,
                'fit', path)
            self.assertEqual([entry['slot'] for entry in meta], [1, 2])
            self.assertEqual([entry['source'] for entry in meta], ['fit', 'fit'])
            self.assertEqual(len(persons), 2)
            # the further-right fitted body must land in the higher slot, as hip-x order says
            self.assertLess(np.nanmean(persons[0][[8, 11]], axis=0)[0],
                np.nanmean(persons[1][[8, 11]], axis=0)[0])

    def test_an_unknown_source_is_an_error_rather_than_a_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.geometry.guide_persons(self.pair(), Path(directory), 0.5, 'midjourney', None)


if __name__ == '__main__':
    unittest.main()

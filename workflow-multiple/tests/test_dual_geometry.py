import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


def write_keypoints(path, points, scores):
    np.savez(path, xy_native=np.asarray(points, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        xy_normalized=np.zeros((134, 2), dtype=np.float32),
        width=np.int32(940), height=np.int32(1280), generator=np.str_('test'))


def synthetic_person(x_offset, score_value=0.9):
    points = np.full((134, 2), np.nan)
    points[0] = [10 + x_offset, 100]      # nose
    points[1] = [10 + x_offset, 200]      # neck
    points[8] = [0 + x_offset, 500]       # right hip
    points[11] = [20 + x_offset, 500]     # left hip
    points[10] = [0 + x_offset, 800]      # right ankle
    points[13] = [20 + x_offset, 800]     # left ankle
    points[92] = [30 + x_offset, 300]     # left wrist
    return points, np.where(np.isnan(points[:, 0]), 0.0, score_value)


class DualGeometryTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('dual_geometry')
        except ModuleNotFoundError as error:  # pragma: no cover - environment guard
            self.fail(f'dual geometry stage is missing: {error}')

    def test_rigid_transform_preserves_the_distance_between_the_two_people(self):
        first = np.array([[0.0, 0.0], [np.nan, np.nan]])
        second = np.array([[100.0, 0.0], [50.0, 50.0]])
        centre = [50.0, 25.0]
        moved = [self.module.rigid_transform(points, 2.0, centre, [200.0, 100.0])
            for points in (first, second)]
        before = float(np.linalg.norm(first[0] - second[0]))
        after = float(np.linalg.norm(moved[0][0] - moved[1][0]))
        self.assertAlmostEqual(after / before, 2.0)
        self.assertTrue(np.isnan(moved[0][1]).all())

    def test_load_person_drops_points_under_the_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kp.npz'
            points, scores = synthetic_person(0.0)
            points[92] = [30.0, 300.0]
            scores[92] = 0.2
            write_keypoints(path, points, scores)
            xy, kept, valid = self.module.load_person(path, 0.5)
            self.assertEqual(xy.shape, (134, 2))
            self.assertTrue(np.isnan(xy[92]).all(), 'a low-score landmark must be absent')
            self.assertFalse(np.isnan(xy[0]).any())
            with self.assertRaises(ValueError):
                np.savez(path, xy_native=np.zeros((17, 2)), scores=np.ones(17))
                self.module.load_person(path, 0.5)

    def test_measure_pair_and_source_agree_in_canvas_coordinates(self):
        points, _ = synthetic_person(0.0)
        pair = self.module.measure_pair([points])
        self.assertEqual(pair['persons_measured'], 1)
        self.assertAlmostEqual(pair['extent_px'], 800.0 - 100.0)
        source_pair = {'A': {'size': [940, 1280], 'persons': [
            {'region': {'bbox_xyxy': [0, 0, 100, 500], 'centroid_xy': [50, 250]}},
            {'region': {'bbox_xyxy': [200, 0, 300, 900], 'centroid_xy': [250, 450]}}]}}
        canvas = (940, 1280)
        measured, transform = self.module.measure_source(source_pair, canvas)
        self.assertAlmostEqual(measured['extent_px'], (500 + 900) / 2 * transform['scale'])
        self.assertEqual(len(measured['centre_xy']), 2)
        empty = self.module.measure_source({'A': {'size': [940, 1280],
            'persons': [{'region': {'bbox_xyxy': None, 'centroid_xy': None}},
                {'region': {'bbox_xyxy': None, 'centroid_xy': None}}]}}, canvas)
        self.assertIsNone(empty[0], 'no usable person mask means no fitting target')

    def test_combine_guides_adds_no_pixel_outside_the_two_inputs(self):
        first = Image.new('RGB', (40, 40), (0, 0, 0))
        second = Image.new('RGB', (40, 40), (0, 0, 0))
        first.putpixel((5, 5), (255, 0, 0))
        second.putpixel((30, 30), (0, 255, 0))
        combined = np.asarray(self.module.combine_guides([first, second]))
        lit = np.argwhere(combined.any(axis=-1))
        self.assertEqual({tuple(item) for item in lit}, {(5, 5), (30, 30)})

    def test_both_framings_render_and_source_fit_scales_the_pair_as_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_dir = root / 'pairs' / 'demo'
            (pair_dir / 'geometry').mkdir(parents=True)
            for slot, offset in ((1, 0.0), (2, 300.0)):
                points, scores = synthetic_person(offset)
                write_keypoints(pair_dir / f'B_person_{slot}_keypoints.npz', points, scores)
            pair = {'name': 'demo', 'kind': 'same_identity',
                'A': {'size': [940, 1280], 'image': 'A_source.png', 'persons': [
                    {'region': {'bbox_xyxy': [0, 0, 200, 1000], 'centroid_xy': [100, 500]}},
                    {'region': {'bbox_xyxy': [400, 0, 700, 1000], 'centroid_xy': [550, 500]}}]},
                'B': {'size': [940, 1280], 'image': 'B_target.png',
                    'slot_order': [{'slot': 1, 'hi4d_side': 'A', 'hip_x': 10.0,
                        'valid_body_points': 8, 'hip_x_separation': 300.0},
                        {'slot': 2, 'hi4d_side': 'B', 'hip_x': 310.0, 'valid_body_points': 8,
                            'hip_x_separation': 300.0}],
                    'keypoints': [{'slot': 1, 'file': 'B_person_1_keypoints.npz', 'sha256': 'x'},
                        {'slot': 2, 'file': 'B_person_2_keypoints.npz', 'sha256': 'y'}]},
                'scoring': {'score_threshold': 0.5, 'person_assignment': None}}
            canvas = (940, 1280)
            for framing in ('b_canvas', 'source_fit'):
                persons, images, combined, transform, meta = self.module.build_guides(
                    pair, pair_dir, canvas, framing, 0.5)
                with self.subTest(framing=framing):
                    self.assertEqual(len(images), 2)
                    self.assertEqual(combined.size, canvas)
                    self.assertEqual([item['slot'] for item in meta], [1, 2])
                    self.assertTrue(all(person.shape == (134, 2) for person in persons))
                    distance = float(np.linalg.norm(persons[0][8] - persons[1][8]))
                    if framing == 'b_canvas':
                        self.assertNotIn('applied_scale', transform)
                        baseline = distance
                    else:
                        self.assertAlmostEqual(distance / baseline, transform['applied_scale'],
                            places=6, msg='the pair must be scaled by one common factor')
                        self.assertGreater(transform['applied_scale'], 1.0)


if __name__ == '__main__':
    unittest.main()

import importlib
import unittest

import numpy as np
from PIL import Image

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class AppearanceTests(unittest.TestCase):
    """The appearance metric must separate drift from a swap, and never invent either."""

    def setUp(self):
        self.module = importlib.import_module('src.pose_edit.appearance')

    def test_crop_person_clamps_to_the_image_and_rejects_degenerate_boxes(self):
        image = Image.new('RGB', (100, 100), (10, 20, 30))
        crop = self.module.crop_person(image, [-50, -50, 400, 400])
        self.assertEqual(crop.size, (100, 100))
        for box in ([10, 10, 10, 40], [0, 0, 0, 0], [float('nan'), 0, 10, 10], [95, 95, 99, 99]):
            with self.subTest(box=box):
                result = self.module.crop_person(image, box)
                if box == [95, 95, 99, 99]:
                    self.assertIsNotNone(result)
                else:
                    self.assertIsNone(result)

    def test_identity_is_measured_along_the_pose_assignment(self):
        similarity = np.array([[0.9, 0.2], [0.2, 0.85]])
        verdict = self.module.judge_appearance(similarity, [0, 1])
        self.assertTrue(verdict['available'])
        self.assertEqual(verdict['identity_similarity'], [0.9, 0.85])
        self.assertFalse(verdict['identity_swap_suspected'])
        self.assertTrue(verdict['assignment_matches_pose'])

    def test_a_better_appearance_pairing_than_the_pose_pairing_is_a_swap_signal(self):
        similarity = np.array([[0.3, 0.95], [0.92, 0.35]])
        verdict = self.module.judge_appearance(similarity, [0, 1])
        self.assertTrue(verdict['identity_swap_suspected'])
        self.assertEqual(verdict['appearance_optimal_assignment'], [1, 0])
        self.assertFalse(verdict['assignment_matches_pose'])

    def test_a_tie_is_not_reported_as_a_swap(self):
        similarity = np.array([[0.6, 0.61], [0.61, 0.6]])
        verdict = self.module.judge_appearance(similarity, [0, 1])
        self.assertFalse(verdict['identity_swap_suspected'])
        self.assertEqual(verdict['appearance_optimal_assignment'], [1, 0])
        # The other pairing wins by 0.02 in total but only 0.01 per person: float noise from the
        # permutation comparison, not evidence. A one-person spread of the same size would be.
        self.assertAlmostEqual(verdict['swap_margin_per_person'], 0.01, places=12)

    def test_a_per_person_spread_above_the_threshold_is_reported_as_a_swap(self):
        # Person 2 is an appearance tie; person 1 clearly prefers the other pairing. The margin
        # is averaged over people, so one strong disagreement is enough to flag the swap.
        similarity = np.array([[0.50, 0.80], [0.60, 0.60]])
        verdict = self.module.judge_appearance(similarity, [0, 1])
        self.assertTrue(verdict['identity_swap_suspected'])
        self.assertAlmostEqual(verdict['swap_margin_per_person'], 0.15, places=12)

    def test_unusable_inputs_degrade_to_a_reason_instead_of_a_number(self):
        for similarity, mapping in ((np.zeros((0, 0)), [0, 1]),
                                    (np.array([[0.5, 0.4]]), [0, 1]),
                                    (np.full((2, 2), np.nan), [0, 1])):
            with self.subTest(shape=similarity.shape):
                verdict = self.module.judge_appearance(similarity, mapping)
                self.assertFalse(verdict['available'])
                self.assertIn('reason', verdict)

    def test_best_assignment_picks_the_higher_total(self):
        similarity = np.array([[0.2, 0.9], [0.85, 0.3]])
        pairs, total = self.module.best_assignment(similarity)
        self.assertEqual(pairs, [1, 0])
        self.assertAlmostEqual(total, 1.75)


if __name__ == '__main__':
    unittest.main()

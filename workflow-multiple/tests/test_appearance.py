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

    def test_an_extra_body_that_matches_the_target_better_than_the_source_is_an_import_signal(self):
        vectors = {'a1': np.array([1.0, 0.0]), 'a2': np.array([0.0, 1.0]),
            'b1': np.array([0.6, 0.8]), 'like_a': np.array([0.0, 1.0]),
            'like_b': np.array([0.6, 0.8])}
        embed = lambda crop: vectors[crop]
        verdict = self.module.judge_import(embed, ['a1', 'a2'], ['b1'], ['like_a', 'like_b'])
        self.assertTrue(verdict['available'])
        self.assertEqual(verdict['bodies_liking_target_more'], 1)
        self.assertFalse(verdict['bodies'][0]['looks_more_like_target'])
        self.assertAlmostEqual(verdict['bodies'][0]['best_source_similarity'], 1.0)
        self.assertAlmostEqual(verdict['bodies'][1]['best_target_similarity'], 1.0)
        self.assertAlmostEqual(verdict['bodies'][1]['target_minus_source'], 0.2)

    def test_the_import_signal_needs_to_clear_the_margin_not_just_point_the_other_way(self):
        margin = self.module.SWAP_MARGIN_PER_PERSON
        vectors = {'a1': np.array([1.0, 0.0]), 'b1': np.array([np.cos(margin), np.sin(margin)]),
            'body': np.array([1.0, 0.0])}
        embed = lambda crop: vectors[crop]
        verdict = self.module.judge_import(embed, ['a1'], ['b1'], ['body'])
        self.assertAlmostEqual(verdict['bodies'][0]['best_source_similarity'], 1.0)
        self.assertFalse(verdict['bodies'][0]['looks_more_like_target'])
        self.assertEqual(verdict['bodies_liking_target_more'], 0)

    def test_missing_crops_degrade_to_a_reason(self):
        embed = lambda crop: np.array([1.0, 0.0])
        for reference, target, bodies in (([], ['b'], ['g']), (['a'], [], ['g']),
                (['a'], ['b'], [])):
            with self.subTest(reference=len(reference), target=len(target)):
                verdict = self.module.judge_import(embed, reference, target, bodies)
                self.assertFalse(verdict['available'])


if __name__ == '__main__':
    unittest.main()

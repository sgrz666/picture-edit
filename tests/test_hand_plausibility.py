"""Unit fixtures for src/evaluation/hand_plausibility.py.

    cd sg:/home/shangguanrz/project/pic-edit
    /home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest tests/test_hand_plausibility.py -q

The synthetic hand has anatomy-like proportions and a 96 px palm, i.e. above MIN_PALM_PX.
Each mutation is a documented failure mode of pose-conditioned diffusion, not random noise.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation import hand_plausibility as HP


def ideal_hand():
    points = [[0.0, 0.0]] * HP.HAND_SLOT_COUNT
    points[0] = [0.0, 0.0]
    points[1], points[2], points[3], points[4] = [-65, -30], [-85, -55], [-100, -75], [-112, -95]
    points[5], points[6], points[7], points[8] = [-40, -95], [-40, -135], [-40, -160], [-40, -180]
    points[9], points[10], points[11], points[12] = [0, -100], [0, -145], [0, -172], [0, -195]
    points[13], points[14], points[15], points[16] = [28, -95], [28, -133], [28, -158], [28, -178]
    points[17], points[18], points[19], points[20] = [52, -85], [52, -112], [52, -132], [52, -148]
    return [[float(x), float(y)] for x, y in points]


def embed(hand, side='left'):
    points = [[0.0, 0.0]] * HP.DWSLOT_COUNT
    valid = [False] * HP.DWSLOT_COUNT
    start = HP.LEFT_HAND_START if side == 'left' else HP.RIGHT_HAND_START
    for offset, value in enumerate(hand):
        points[start + offset] = list(value)
        valid[start + offset] = True
    return points, valid


class HandPlausibilityTest(unittest.TestCase):

    def test_slot_layout_matches_the_repo_tables(self):
        """Pin to the two places that define hand slots, so a reordering breaks here first."""
        from src.evaluation import no_gt_metrics as M
        self.assertEqual(HP.LEFT_HAND_START, M.GROUPS['left_hand'].start)
        self.assertEqual(HP.RIGHT_HAND_START, M.GROUPS['right_hand'].start)
        self.assertEqual(M.GROUPS['right_hand'].stop, HP.DWSLOT_COUNT)
        self.assertEqual(HP.LEFT_HAND_START + HP.HAND_SLOT_COUNT, HP.RIGHT_HAND_START)

    def test_ideal_hand_is_clean(self):
        result = HP.analyze_hand(ideal_hand(), [True] * HP.HAND_SLOT_COUNT)
        self.assertTrue(result['measurable'], result['flags'])
        self.assertEqual(result['flags'], [])
        self.assertGreaterEqual(result['palm_px'], HP.MIN_PALM_PX)

    def test_fused_fingertips_are_flagged(self):
        hand = ideal_hand()
        hand[8] = list(hand[12])  # index tip on middle tip: the mitten failure
        result = HP.analyze_hand(hand, [True] * HP.HAND_SLOT_COUNT)
        self.assertIn('tips_collapsed:index+middle', result['flags'])

    def test_hyperextended_finger_is_flagged(self):
        hand = ideal_hand()
        hand[20] = [52.0, -400.0]  # pinky more than three palms long
        result = HP.analyze_hand(hand, [True] * HP.HAND_SLOT_COUNT)
        self.assertIn('pinky:finger_length_out_of_range', result['flags'])

    def test_a_curled_finger_is_not_an_anomaly(self):
        """Radial distance is allowed to fall back toward the wrist: that is just a bent finger."""
        hand = ideal_hand()
        hand[12] = [0.0, -150.0]  # middle tip curled below its DIP
        result = HP.analyze_hand(hand, [True] * HP.HAND_SLOT_COUNT)
        self.assertEqual([flag for flag in result['flags'] if 'middle' in flag], [])

    def test_small_hand_is_excluded_not_scored(self):
        hand = [[x * 0.4, y * 0.4] for x, y in ideal_hand()]
        result = HP.analyze_hand(hand, [True] * HP.HAND_SLOT_COUNT)
        self.assertFalse(result['measurable'])
        self.assertEqual(result['flags'], ['hand_too_small_to_judge'])

    def test_unmeasurable_hand_is_excluded_not_blamed(self):
        valid = [False] * HP.HAND_SLOT_COUNT
        for index in range(5):
            valid[index] = True
        result = HP.analyze_hand(ideal_hand(), valid)
        self.assertFalse(result['measurable'])
        self.assertEqual(result['flags'], ['hand_not_measurable'])

    def test_compare_charges_only_new_flags(self):
        clean_points, clean_valid = embed(ideal_hand(), 'left')
        broken = ideal_hand()
        broken[8] = list(broken[12])
        broken_points, broken_valid = embed(broken, 'left')
        broken_analysis = HP.analyze(broken_points, broken_valid)
        same = HP.compare(broken_analysis, broken_analysis)
        self.assertEqual(same['new_in_generated'], [])
        self.assertEqual(same['verdict'], 'clean')
        fresh = HP.compare(HP.analyze(clean_points, clean_valid), broken_analysis)
        self.assertIn('left:tips_collapsed:index+middle', fresh['new_in_generated'])
        self.assertEqual(fresh['verdict'], 'inspect')

    def test_existing_runs_score_without_crashing(self):
        """Real arrays drive the calibration claim: most hands are too small to judge."""
        records = []
        for npz_path in sorted((REPO / 'workflow/runs').rglob('final_keypoints.npz')):
            record = HP.run_on_run(npz_path.parent.parent, write=False)
            if record:
                records.append(record)
        self.assertGreaterEqual(len(records), 1)
        stats = HP.distribution(records)
        self.assertGreater(stats['hands'], 0)
        self.assertLessEqual(stats['palm_px_p5_p50_p95'][1], HP.MIN_PALM_PX,
            'median palm at this resolution is meant to be under the judging threshold; '
            'if hands got bigger, MIN_PALM_PX and the bands need recalibrating')


if __name__ == '__main__':
    unittest.main()

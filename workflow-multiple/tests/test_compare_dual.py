import importlib
import unittest

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class CompareDualTests(unittest.TestCase):
    """A paired comparison must only compare the same pair, and must say when it cannot."""

    def setUp(self):
        self.module = importlib.import_module('compare_dual')

    def fact(self, pair, arm='a', framing='b_canvas', source='dwpose', **values):
        fact = {'pair': pair, 'arm': arm, 'framing': framing, 'source': source,
            'status': 'complete', 'run': f'{pair}.{arm}.{framing}'}
        for name, _ in self.module.METRICS:
            fact[name] = values.get(name)
        return fact

    def test_a_metric_votes_only_when_both_sides_measured_it(self):
        left = self.fact('p1', 'with_b', pck=0.9, mpjpe=0.1)
        right = self.fact('p1', 'skeleton_only', pck=0.4, mpjpe=None)
        row = self.module.differences(left, right, 'arm')
        self.assertEqual(row['metrics']['pck']['verdict'], 'left')
        self.assertEqual(row['metrics']['mpjpe']['verdict'], 'not measured')
        self.assertEqual(row['metrics']['identity']['verdict'], 'not measured')

    def test_a_difference_inside_the_tolerance_is_a_tie_not_a_win(self):
        # Boundary values are chosen away from the tolerance itself: 0.50 - 0.02 is not exactly
        # 0.02 in binary floating point, and a test that lands on the edge tests the rounding.
        left = self.fact('p1', 'a', pck=0.50)
        right = self.fact('p1', 'b', pck=0.49)
        self.assertEqual(self.module.differences(left, right, 'arm')['metrics']['pck']['verdict'],
            'tie')
        beyond = self.fact('p1', 'b', pck=0.47)
        self.assertEqual(self.module.differences(left, beyond, 'arm')['metrics']['pck']['verdict'],
            'left')

    def test_lower_is_better_metrics_are_read_the_other_way(self):
        left = self.fact('p1', 'a', mpjpe=0.30, gap_error=0.5)
        right = self.fact('p1', 'b', mpjpe=0.10, gap_error=0.1)
        row = self.module.differences(left, right, 'arm')
        self.assertEqual(row['metrics']['mpjpe']['verdict'], 'right')
        self.assertEqual(row['metrics']['gap_error']['verdict'], 'right')

    def test_torso_ratio_reports_a_difference_but_never_declares_a_winner(self):
        left = self.fact('p1', 'a', torso_ratio=0.9)
        right = self.fact('p1', 'b', torso_ratio=1.1)
        row = self.module.differences(left, right, 'arm')
        self.assertEqual(row['metrics']['torso_ratio']['verdict'], 'informative only')
        self.assertAlmostEqual(row['metrics']['torso_ratio']['delta'], -0.2)

    def test_a_pair_measured_on_one_side_only_is_listed_as_missing(self):
        # The left side is the alphabetically first arm name; tests use 'a' and 'b' so the
        # direction of every delta stays readable regardless of the real arm names.
        facts = [self.fact('p1', 'a', pck=0.5), self.fact('p1', 'b', pck=0.6),
            self.fact('p2', 'a', pck=0.4)]
        rows, missing = self.module.paired_rows(facts, 'arm')
        self.assertEqual([row['pair'] for row in rows], ['p1'])
        self.assertEqual([item['pair'] for item in missing], ['p2'])
        self.assertEqual(missing[0]['only'], ['a'])

    def test_comparing_framings_holds_the_arm_fixed(self):
        # The bug this pins: b_canvas existed for the with_b arm and source_fit only for the
        # skeleton_only arm. Grouping by (pair, framing) alone compared those two and reported a
        # person-count win for source_fit that was really the arm difference.
        facts = [self.fact('p1', arm='with_b', framing='b_canvas', count_ok=0.0, pck=None),
            self.fact('p1', arm='skeleton_only', framing='b_canvas', count_ok=1.0, pck=0.5),
            self.fact('p1', arm='skeleton_only', framing='source_fit', count_ok=1.0, pck=0.7)]
        rows, missing = self.module.paired_rows(facts, 'framing')
        self.assertEqual(len(rows), 1, 'only the arm that has both framings may be compared')
        self.assertEqual([rows[0]['left'], rows[0]['right']], ['b_canvas', 'source_fit'])
        self.assertEqual(rows[0]['metrics']['pck']['right'], 0.7)
        self.assertEqual(rows[0]['metrics']['count_ok']['verdict'], 'tie')
        self.assertEqual([item['only'] for item in missing], [['b_canvas']])
        self.assertEqual(missing[0]['held'], {'arm': 'with_b', 'source': 'dwpose'})

    def test_comparing_guide_sources_holds_arm_and_framing_fixed(self):
        # A fit-guided run must be compared against the ground-truth-guided run of the same arm in
        # the same framing, and a source run that only exists for the other arm must not vote.
        facts = [self.fact('p1', arm='skeleton_only', framing='b_canvas', source='dwpose',
                pck=0.5, count_ok=1.0),
            self.fact('p1', arm='skeleton_only', framing='b_canvas', source='fit', pck=0.7,
                count_ok=1.0),
            self.fact('p1', arm='with_b', framing='b_canvas', source='fit', pck=0.2)]
        rows, missing = self.module.paired_rows(facts, 'source')
        self.assertEqual(len(rows), 1)
        self.assertEqual([rows[0]['left'], rows[0]['right']], ['dwpose', 'fit'])
        self.assertEqual(rows[0]['metrics']['pck']['verdict'], 'right')
        # the with_b fit run has no with_b ground-truth twin, so it is listed rather than voted
        self.assertEqual([item['only'] for item in missing], [['fit']])
        self.assertEqual(missing[0]['held'], {'arm': 'with_b', 'framing': 'b_canvas'})

    def test_the_same_treatment_measured_twice_uses_the_batch_named_run(self):
        legacy = self.fact('p1', arm='b', framing='b_canvas', pck=0.9, )
        legacy['run'] = 'hand_named_b'
        batch = self.fact('p1', arm='b', framing='b_canvas', pck=0.4)
        other = self.fact('p1', arm='a', framing='b_canvas', pck=0.5)
        rows, _ = self.module.paired_rows([legacy, batch, other], 'arm')
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['metrics']['pck']['right'], 0.4)

    def test_the_summary_counts_verdicts_over_the_pairs_that_have_both_sides(self):
        facts = [self.fact('p1', 'a', pck=0.2, mpjpe=0.9),
            self.fact('p1', 'b', pck=0.8, mpjpe=0.1),
            self.fact('p2', 'a', pck=0.3),
            self.fact('p2', 'b', pck=0.9)]
        rows, _ = self.module.paired_rows(facts, 'arm')
        summary = self.module.summarise(rows)
        self.assertEqual(summary['comparisons'], 2)
        self.assertEqual(summary['by_metric']['pck']['left_wins'], 0)
        self.assertEqual(summary['by_metric']['pck']['right_wins'], 2)
        self.assertAlmostEqual(summary['by_metric']['pck']['median_delta'], -0.6)
        self.assertEqual(summary['by_metric']['mpjpe']['not_measured'], 1)


if __name__ == '__main__':
    unittest.main()

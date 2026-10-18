#!/usr/bin/env python3
"""Contract tests for size_compliance: grouping, skipping, and the ratio itself."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from size_compliance import compliance_groups, group_of, median, summarise


def write_eval(runs, name, persons):
    evaluation = runs / name / 'evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / 'eval_dual.json').write_text(json.dumps({'persons': persons}), encoding='utf-8')


class Naming(unittest.TestCase):
    def test_matrix_cells_are_grouped(self):
        self.assertEqual(group_of('p.skeleton_only.b_canvas'), ('skeleton_only', 'b_canvas', 'dwpose'))
        self.assertEqual(group_of('p.with_b.source_fit'), ('with_b', 'source_fit', 'dwpose'))
        self.assertEqual(group_of('p.skeleton_only.b_canvas.fit'),
            ('skeleton_only', 'b_canvas', 'fit'))

    def test_legacy_directories_are_not_guessed_at(self):
        for name in ('hug_skeleton_only', 'p_b_canvas', 'p'):
            self.assertIsNone(group_of(name))


class Ratios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ratio_is_detected_over_requested(self):
        write_eval(self.runs, 'alpha.with_b.b_canvas',
            [{'requested_torso_px': 200.0, 'detected_torso_px': 190.0}])
        groups = compliance_groups(self.runs)
        self.assertEqual(groups[('with_b', 'b_canvas', 'dwpose')], [0.95])

    def test_persons_missing_either_field_are_dropped(self):
        write_eval(self.runs, 'alpha.with_b.b_canvas', [
            {'requested_torso_px': 200.0, 'detected_torso_px': 210.0},
            {'requested_torso_px': 200.0, 'detected_torso_px': None},
            {'requested_torso_px': 0, 'detected_torso_px': 150.0},
            {}])
        self.assertEqual(len(compliance_groups(self.runs)[('with_b', 'b_canvas', 'dwpose')]), 1)

    def test_legacy_runs_do_not_pollute_a_group(self):
        write_eval(self.runs, 'hug_skeleton_only',
            [{'requested_torso_px': 100.0, 'detected_torso_px': 300.0}])
        self.assertEqual(compliance_groups(self.runs), {})

    def test_summary_reports_median_and_tolerance_bands(self):
        groups = {('skeleton_only', 'b_canvas', 'dwpose'): [0.98, 1.00, 1.02, 1.20]}
        block = summarise(groups)['skeleton_only/b_canvas/dwpose']
        self.assertEqual(block['people'], 4)
        self.assertEqual(block['median'], 1.01)
        self.assertEqual(block['min'], 0.98)
        self.assertEqual(block['max'], 1.20)
        self.assertAlmostEqual(block['within_5_percent'], 0.75)
        self.assertAlmostEqual(block['within_10_percent'], 0.75)

    def test_median_of_even_count_averages_the_middle_pair(self):
        self.assertEqual(median([1.0, 2.0, 3.0, 4.0]), 2.5)
        self.assertEqual(median([1.0, 2.0, 3.0]), 2.0)
        self.assertIsNone(median([]))


if __name__ == '__main__':
    unittest.main(verbosity=2)

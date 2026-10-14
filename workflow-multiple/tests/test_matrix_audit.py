#!/usr/bin/env python3
"""Contract tests for matrix_audit: coverage from artifacts, and paired rows that hold variables."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matrix_audit import build_grid, cells, pairable_rows


def make_run(runs, name, status='complete', summary=None):
    run_dir = runs / name
    (run_dir / 'evaluation').mkdir(parents=True, exist_ok=True)
    (run_dir / 'run.json').write_text(json.dumps({'status': status}), encoding='utf-8')
    if summary is not None:
        (run_dir / 'evaluation' / 'eval_dual.json').write_text(
            json.dumps({'summary': summary}), encoding='utf-8')


class PlannedCells(unittest.TestCase):
    def test_four_guided_cells_plus_one_twin_per_pair(self):
        planned = cells(['p'])
        self.assertEqual(len(planned), 5)
        self.assertEqual({item[1] for item in planned}, {'with_b', 'skeleton_only'})
        self.assertEqual([item[0] for item in planned if item[3] == 'fit'],
            ['p.skeleton_only.b_canvas.fit'])


class Coverage(unittest.TestCase):
    def setUp(self):
        self.pairs = ['alpha_pair', 'beta_pair']
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def grid(self):
        return build_grid(self.pairs, self.runs)

    def test_complete_needs_an_evaluation_too(self):
        make_run(self.runs, 'alpha_pair.with_b.b_canvas', status='complete')
        item = [one for one in self.grid() if one['name'] == 'alpha_pair.with_b.b_canvas'][0]
        self.assertEqual(item['status'], 'complete')
        self.assertFalse(item['covered'])

    def test_partial_run_with_evaluation_is_not_covered(self):
        make_run(self.runs, 'alpha_pair.with_b.b_canvas', status='partial',
            summary={'body_pck@0.2': 0.5, 'identity_similarity': 0.7, 'person_count_ok': 1})
        self.assertFalse([one for one in self.grid()
            if one['name'] == 'alpha_pair.with_b.b_canvas'][0]['covered'])

    def test_missing_metric_is_recorded_as_absent_not_zero(self):
        make_run(self.runs, 'alpha_pair.with_b.b_canvas', summary={'body_pck@0.2': None,
            'identity_similarity': 0.71, 'person_count_ok': 0})
        item = [one for one in self.grid() if one['name'] == 'alpha_pair.with_b.b_canvas'][0]
        self.assertTrue(item['covered'])
        self.assertFalse(item['pose'])
        self.assertTrue(item['identity'])
        self.assertEqual(item['count_ok'], 0)


class PairedRows(unittest.TestCase):
    def setUp(self):
        self.pairs = ['alpha_pair']
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def grid(self):
        return build_grid(self.pairs, self.runs)

    def test_arm_votes_only_within_one_framing_and_one_guide(self):
        for name in ('alpha_pair.with_b.b_canvas', 'alpha_pair.skeleton_only.b_canvas',
                'alpha_pair.skeleton_only.source_fit'):
            make_run(self.runs, name, summary={'body_pck@0.2': 0.9,
                'identity_similarity': 0.7, 'person_count_ok': 1})
        rows, voteable = pairable_rows(self.grid(), 'arm')
        self.assertEqual((rows, voteable), (1, 1))

    def test_a_cell_from_another_framing_does_not_create_a_row(self):
        make_run(self.runs, 'alpha_pair.with_b.b_canvas', summary={'body_pck@0.2': 0.9})
        make_run(self.runs, 'alpha_pair.skeleton_only.source_fit', summary={'body_pck@0.2': 0.8})
        self.assertEqual(pairable_rows(self.grid(), 'arm'), (0, 0))

    def test_rows_need_both_sides_covered(self):
        summary = {'body_pck@0.2': 0.9, 'identity_similarity': 0.7, 'person_count_ok': 1}
        make_run(self.runs, 'alpha_pair.with_b.b_canvas', summary=summary)
        self.assertEqual(pairable_rows(self.grid(), 'arm'), (0, 0))
        make_run(self.runs, 'alpha_pair.skeleton_only.b_canvas', status='partial', summary=summary)
        self.assertEqual(pairable_rows(self.grid(), 'arm'), (0, 0))
        make_run(self.runs, 'alpha_pair.skeleton_only.b_canvas', summary=summary)
        self.assertEqual(pairable_rows(self.grid(), 'arm'), (1, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)

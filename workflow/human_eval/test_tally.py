#!/usr/bin/env python3
"""Fixtures for tally.py: known votes in, known statistics out.

    python -B -m unittest test_tally -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tally


def judgement(qid, pair, annotator, pick):
    return {'annotator': annotator, 'id': qid, 'pair': pair, 'pick': pick,
        'left_sha256': '', 'right_sha256': '', 'ratings': {'left': {}, 'right': {}}}


class TallyTest(unittest.TestCase):

    def setUp(self):
        # Question 1 is won by the candidate on the left, question 2 on the right, so the
        # reference/candidate mapping must flip with the recorded arm, not with the side.
        self.index = [
            {'id': 'q001', 'pair': 'p1', 'left': 'l1.png', 'right': 'r1.png',
                'left_sha256': 'a' * 64, 'right_sha256': 'b' * 64},
            {'id': 'q002', 'pair': 'p2', 'left': 'l2.png', 'right': 'r2.png',
                'left_sha256': 'c' * 64, 'right_sha256': 'd' * 64},
        ]
        self.truth = {'seed': 1, 'reference_arm': 'skeleton_only', 'candidate_arm': 'with_b',
            'items': [
                {'id': 'q001', 'pair': 'p1', 'left_arm': 'with_b', 'right_arm': 'skeleton_only'},
                {'id': 'q002', 'pair': 'p2', 'left_arm': 'skeleton_only', 'right_arm': 'with_b'},
            ]}

    def rows(self, rows_dir):
        path = Path(rows_dir) / 'annotator1.jsonl'
        rows = []
        for question, candidate_side, pick in (('q001', 'left', 'left'), ('q002', 'right', 'right')):
            sha = {'q001': ('a' * 64, 'b' * 64), 'q002': ('c' * 64, 'd' * 64)}[question]
            index = next(item for item in self.index if item['id'] == question)
            row = judgement(question, index['pair'], 'ann1', pick)
            row['left_sha256'], row['right_sha256'] = sha
            row['ratings'] = {side: {axis: 5 for axis in tally.AXES} for side in ('left', 'right')}
            rows.append(row)
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')

    def test_candidate_wins_are_signed_positive(self):
        with tempfile.TemporaryDirectory() as work:
            self.rows(work)
            judgements, excluded = tally.load_judgements(Path(work), self.index, self.truth)
            self.assertEqual(excluded, [])
            summary = tally.summarize(judgements, self.truth)
            self.assertEqual(summary['wins_for_candidate'], 2)
            self.assertEqual(summary['losses'], 0)
            self.assertAlmostEqual(summary['win_rate'], 1.0, places=6)
            self.assertGreater(summary['win_rate_ci95'][0], 0.05)
            self.assertEqual(summary['contract_win_margin'], '+5%')

    def test_hash_mismatch_is_excluded_not_averaged(self):
        with tempfile.TemporaryDirectory() as work:
            self.rows(work)
            stale = json.loads(Path(work, 'annotator1.jsonl').read_text(encoding='utf-8')
                .splitlines()[0])
            stale['left_sha256'] = 'f' * 64
            with Path(work, 'annotator1.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(stale) + '\n')
            judgements, excluded = tally.load_judgements(Path(work), self.index, self.truth)
            self.assertEqual(len(judgements), 2)
            self.assertEqual(excluded[0]['reason'], 'image_hash_mismatch')

    def test_unanimous_choice_gives_kappa_one(self):
        with tempfile.TemporaryDirectory() as work:
            self.rows(work)
            judgements, _ = tally.load_judgements(Path(work), self.index, self.truth)
            self.assertAlmostEqual(tally.fleiss_kappa(
                [[0, 0], [1, 1]]), 1.0, places=6)
            summary = tally.summarize(judgements, self.truth)
            self.assertIsNone(summary['fleiss_kappa_on_choice'])  # one annotator is not agreement

    def test_plausibility_counts_all_four_axes(self):
        with tempfile.TemporaryDirectory() as work:
            self.rows(work)
            judgements, _ = tally.load_judgements(Path(work), self.index, self.truth)
            summary = tally.summarize(judgements, self.truth)
            self.assertAlmostEqual(summary['plausibility_rate_candidate'], 1.0, places=6)
            self.assertEqual(set(summary['axis_fail_rate_candidate']), set(tally.AXES))


if __name__ == '__main__':
    unittest.main()

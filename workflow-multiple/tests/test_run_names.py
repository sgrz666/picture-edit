#!/usr/bin/env python3
"""Run-name contract: every name the batch and the fit driver will generate must be accepted.

The 64-character cap silently rejected nine planned cells on the cross pairs, and the failure looked
like a malformed command line rather than a length rule. This test derives the names from the pair
directories themselves, so adding a long pair name trips it before a batch run loses the cell.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_dual import RUN_NAME_MAX_LENGTH, RUN_NAME_PATTERN


def planned_names():
    pairs = [p.name for p in sorted((ROOT / 'pairs').iterdir()) if (p / 'pair.json').is_file()]
    cross = [p.name for p in sorted((ROOT / 'pairs_cross').iterdir())
        if (p / 'pair.json').is_file()]
    names = []
    for pair in pairs + cross:
        for arm in ('with_b', 'skeleton_only'):
            for framing in ('b_canvas', 'source_fit'):
                names.append(f'{pair}.{arm}.{framing}')
        names.append(f'{pair}.skeleton_only.b_canvas.fit')
    return names


class PlannedNames(unittest.TestCase):
    def test_every_planned_cell_name_is_accepted(self):
        rejected = [name for name in planned_names() if not RUN_NAME_PATTERN.fullmatch(name)]
        self.assertEqual([], rejected,
            f'{len(rejected)} planned run names would be rejected: {rejected[:3]}')

    def test_the_longest_planned_name_is_inside_the_cap(self):
        longest = max(planned_names(), key=len)
        self.assertLessEqual(len(longest), RUN_NAME_MAX_LENGTH)
        self.assertGreater(len(longest), 64,
            'this name is the reason the cap moved; if it shrinks below 65 the guard is vacuous')


class Rejections(unittest.TestCase):
    def test_pathish_names_are_still_rejected(self):
        for name in ('a/b', '../escape', '.hidden', 'with space'):
            self.assertIsNone(RUN_NAME_PATTERN.fullmatch(name), name)

    def test_over_cap_is_rejected(self):
        self.assertIsNone(RUN_NAME_PATTERN.fullmatch('a' * (RUN_NAME_MAX_LENGTH + 1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

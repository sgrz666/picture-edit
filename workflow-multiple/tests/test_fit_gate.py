#!/usr/bin/env python3
"""The hip-gap gate must decide on projected geometry and only report the DWPose gap."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fit_dual_report as report


class GateReference(unittest.TestCase):
    def test_deciding_term_is_the_projected_geometry_error(self):
        self.assertIn('hip_gap_error_norm_gt', report.GATE)
        self.assertEqual(report.GATE['hip_gap_error_norm_gt'], ('max', 0.10))

    def test_dwpose_referenced_gap_is_diagnostic_and_not_deciding(self):
        self.assertNotIn('hip_gap_error_norm', report.GATE)
        self.assertIn('hip_gap_error_norm', report.DIAGNOSTIC)

    def test_the_diagnostic_says_the_landmarks_may_be_at_fault(self):
        self.assertIn('landmarks being wrong', report.DIAGNOSTIC['hip_gap_error_norm'])


class HipGapErrors(unittest.TestCase):
    def test_returns_projected_first_and_dwpose_second(self):
        self.assertEqual(report.hip_gap_errors(
            {'error_norm_gt': 0.02, 'error_norm': 0.38}), (0.02, 0.38))

    def test_missing_projected_reference_does_not_fabricate_a_value(self):
        self.assertEqual(report.hip_gap_errors({'error_norm': 0.38})[0], None)

    def test_absent_record_is_tolerated(self):
        self.assertEqual(report.hip_gap_errors(None), (None, None))


if __name__ == '__main__':
    unittest.main(verbosity=2)

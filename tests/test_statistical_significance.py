"""Statistical edge cases and pairing checks using CPU-only saved metrics."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from statistical_significance import DEFAULT_SOURCE, compare_metrics, holm_adjust, paired_test, run_analysis


class StatisticalTests(unittest.TestCase):
    def test_exact_resolution_and_mixed_signs(self):
        self.assertEqual(paired_test([1, 2, 3])[1], 0.25)
        self.assertEqual(paired_test([1, -2, 3])[1], 0.75)

    def test_ties_zeros_and_floating_point_ranks(self):
        self.assertEqual(paired_test([0, 0, 0])[1], 1)
        self.assertEqual(paired_test([1, 1, 1])[1], 0.25)
        self.assertEqual(paired_test([0, 1, 2])[1], 0.5)
        self.assertEqual(paired_test([0.1, 0.1 + 1e-16, 0.1 - 1e-16])[2], "exhaustive_permutation")

    def test_holm_restores_order_and_is_monotonic(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])
        np.testing.assert_allclose(holm_adjust([0.25] * 12), [1] * 12)

    def test_pairing_does_not_depend_on_csv_order(self):
        frame = pd.read_csv(DEFAULT_SOURCE)
        pd.testing.assert_frame_equal(compare_metrics(frame), compare_metrics(frame.sample(frac=1, random_state=7)))

    def test_incomplete_duplicate_nonfinite_or_wrong_scale_rejected(self):
        frame = pd.read_csv(DEFAULT_SOURCE)
        for bad in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                    frame.assign(accuracy=np.nan), frame.assign(accuracy=99),
                    frame.assign(num_samples=8172), frame.assign(seed=42)]:
            with self.assertRaises(ValueError):
                compare_metrics(bad)

    def test_real_analysis_preserves_source_and_refuses_overwrite(self):
        before = hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "analysis"
            result = run_analysis(DEFAULT_SOURCE, dest)
            self.assertEqual(len(result), 12)
            self.assertFalse(result.reject_h0_holm.any())
            self.assertGreaterEqual(result.p_value.min(), 0.25)
            with self.assertRaises(FileExistsError):
                run_analysis(DEFAULT_SOURCE, dest)
        self.assertEqual(hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()

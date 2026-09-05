import math
import unittest

import numpy as np

from analyze_human_gt_reliability import difference_statistics, icc_a1, relative_difference_percent


class ReliabilityMathTests(unittest.TestCase):
    def test_locked_mdc95_formula(self) -> None:
        first = np.asarray([10.0, 20.0, 30.0])
        second = np.asarray([11.0, 18.0, 33.0])
        result = difference_statistics(first, second)
        expected_sd = float(np.std(np.asarray([1.0, -2.0, 3.0]), ddof=1))
        self.assertAlmostEqual(result["bias"], 2.0 / 3.0)
        self.assertAlmostEqual(result["sd_diff"], expected_sd)
        self.assertAlmostEqual(result["sem"], expected_sd / math.sqrt(2.0))
        self.assertAlmostEqual(result["mdc95"], 1.96 * expected_sd)

    def test_symmetric_relative_difference(self) -> None:
        result = relative_difference_percent(np.asarray([90.0]), np.asarray([110.0]))
        self.assertAlmostEqual(float(result[0]), 20.0)

    def test_identical_ratings_have_perfect_icc(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        self.assertAlmostEqual(icc_a1(values, values), 1.0)


if __name__ == "__main__":
    unittest.main()

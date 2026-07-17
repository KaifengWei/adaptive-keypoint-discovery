#!/usr/bin/env python
"""Small deterministic regressions for topology cleanup and safe interpolation."""

from __future__ import annotations

import unittest

import numpy as np

import g1_prime_phenotype_bridge as bridge


class TopologyBridgeTests(unittest.TestCase):
    def test_short_terminal_spur_is_removed_but_long_organs_remain(self) -> None:
        skeleton = np.zeros((40, 40), dtype=bool)
        skeleton[5:31, 20] = True
        skeleton[15, 20:33] = True
        skeleton[24, 20:24] = True

        pruned, diagnostics = bridge.prune_short_terminal_spurs(skeleton, maximum_length=4.0)

        self.assertFalse(pruned[24, 23])
        self.assertTrue(pruned[5, 20])
        self.assertTrue(pruned[30, 20])
        self.assertTrue(pruned[15, 32])
        self.assertGreater(diagnostics["spur_prune_removed_pixels"], 0)

    def test_shape_preserving_curve_cannot_form_cubic_loop(self) -> None:
        support = np.asarray(
            [
                [219.0, 417.0],
                [221.4, 413.5],
                [223.0, 408.0],
                [225.1, 233.2],
                [235.1, 221.0],
                [248.8, 189.5],
                [340.0, 65.0],
            ]
        )

        curve = bridge.spline_curve(support)

        self.assertGreaterEqual(float(curve[:, 0].min()), float(support[:, 0].min()) - 1e-6)
        self.assertLessEqual(float(curve[:, 0].max()), float(support[:, 0].max()) + 1e-6)
        self.assertGreaterEqual(float(curve[:, 1].min()), float(support[:, 1].min()) - 1e-6)
        self.assertLessEqual(float(curve[:, 1].max()), float(support[:, 1].max()) + 1e-6)
        self.assertLessEqual(
            bridge.polyline_length(curve),
            1.10 * bridge.polyline_length(support),
        )


if __name__ == "__main__":
    unittest.main()

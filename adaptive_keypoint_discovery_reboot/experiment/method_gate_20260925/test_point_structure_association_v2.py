"""Synthetic-only tests; intentionally never open a real val image."""

import math
import unittest

import numpy as np

from point_structure_association_v2 import association_score, build_point_conditioned_graph_v2, roi_square_distance


def fixture(points, *, root_at=None, roi_end=None, empty=False, thick=1):
    shape = (50, 50)
    skeleton = np.zeros(shape, dtype=bool)
    support = np.zeros(shape, dtype=bool)
    if not empty:
        skeleton[20, 5:45] = True
        support[20-thick:21+thick, 5:45] = True
    roi = np.zeros(shape, dtype=bool)
    roi[15:26, 5:45 if roi_end is None else roi_end] = True
    basal = np.zeros(shape, dtype=bool)
    root = np.zeros(shape, dtype=bool)
    shoot = np.zeros(shape, dtype=bool)
    shoot[20, 5:45] = True
    if root_at is not None:
        x = root_at
        root[20, x] = True
        shoot[20, x] = False
    return build_point_conditioned_graph_v2(skeleton, points, 100.0, support, roi, basal, root, shoot)


class FormulaTests(unittest.TestCase):
    def test_square_distance_exact_and_continuous(self):
        xy = np.asarray([[10, 10], [20, 10]], dtype=float)
        self.assertEqual(roi_square_distance(10.2, 10.4, xy), 0.0)
        self.assertAlmostEqual(roi_square_distance(11.2, 10.0, xy), 0.7)
        self.assertAlmostEqual(roi_square_distance(11.2, 11.2, xy), math.hypot(.7, .7))

    def test_formula_and_boundary(self):
        sigma, score = association_score(2.5, 0.0, 0.0, 100.0)
        self.assertAlmostEqual(sigma, 1.25)
        self.assertAlmostEqual(score, math.exp(-2.0))
        wide_sigma, wide_score = association_score(2.5*math.sqrt(2), 0.0, 2.0, 100.0)
        self.assertAlmostEqual(wide_sigma, 1.25*math.sqrt(2))
        self.assertAlmostEqual(wide_score, math.exp(-2.0))

    def test_monotonicity(self):
        for d in (0, .5, 2.0, 3.0):
            for delta in (0, .5, 1.0):
                a = association_score(d, delta, .2, 100)[1]
                self.assertGreaterEqual(a, association_score(d+.1, delta, .2, 100)[1])
                self.assertGreaterEqual(a, association_score(d, delta+.1, .2, 100)[1])
                self.assertLessEqual(a, association_score(d, delta, 1.25, 100)[1])
        self.assertEqual(association_score(0, 0, 0, 100)[1], 1.0)
        with self.assertRaises(ValueError):
            association_score(float('nan'), 0, 0, 100)

    def test_empty_skeleton(self):
        graph, rows = fixture([{"point_id": "a", "x": 12., "y": 20., "score": .7}], empty=True)
        self.assertEqual(graph["diagnostics"]["failure"], "empty_skeleton")
        self.assertEqual(rows[0]["rejection_reason"], "empty_skeleton")

    def test_duplicates_preserve_v1_raw_score(self):
        points = [{"point_id": name, "x": 11., "y": 20., "score": score} for name, score in (("a", .2), ("b", .9), ("c", .9))]
        graph, rows = fixture(points)
        self.assertEqual([n["point_id"] for n in graph["nodes"]], ["b"])
        self.assertEqual([r["status"] for r in rows], ["rejected", "accepted", "rejected"])
        self.assertEqual([r["rejection_reason"] for r in rows], ["duplicate_projection_merged", "", "duplicate_projection_merged"])

    def test_hard_roi_and_underground_veto(self):
        p = {"point_id": "a", "x": 11., "y": 20., "score": .5}
        graph, rows = fixture([p], root_at=11)
        self.assertEqual(rows[0]["rejection_reason"], "root_only_association")
        self.assertEqual(len(graph["nodes"]), 0)
        _, rows = fixture([p], roi_end=10)
        self.assertEqual(rows[0]["rejection_reason"], "outside_frozen_roi")

    def test_weighted_rejection_and_no_synthetic_node(self):
        graph, rows = fixture([{"point_id": "far", "x": 11., "y": 27., "score": .9}])
        self.assertEqual(rows[0]["rejection_reason"], "weighted_association_rejection")
        self.assertEqual(len(graph["nodes"]), 0)
        self.assertEqual(len(graph["edges"]), 0)


if __name__ == "__main__":
    unittest.main()

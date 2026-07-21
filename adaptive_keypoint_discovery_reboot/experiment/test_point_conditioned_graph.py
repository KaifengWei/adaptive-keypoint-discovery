#!/usr/bin/env python
"""Small synthetic tests for keypoint-conditioned graph reconstruction."""

from __future__ import annotations

import unittest

import numpy as np

from point_conditioned_graph import build_point_conditioned_graph


def point(x: float, y: float, score: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "score": score}


class PointConditionedGraphTests(unittest.TestCase):
    def test_line_requires_learned_endpoints(self) -> None:
        skeleton = np.zeros((24, 24), dtype=bool)
        skeleton[12, 3:21] = True
        result = build_point_conditioned_graph(skeleton, [point(3, 12), point(20, 12)], 24.0, 0.05)
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(int(result["edge_union"].sum()), 18)

    def test_no_skeleton_endpoint_autocompletion(self) -> None:
        skeleton = np.zeros((24, 24), dtype=bool)
        skeleton[12, 3:21] = True
        for learned in ([], [point(10, 12)]):
            result = build_point_conditioned_graph(skeleton, learned, 24.0, 0.05)
            self.assertEqual(len(result["edges"]), 0)
            self.assertEqual(int(result["edge_union"].sum()), 0)

    def test_far_point_is_rejected(self) -> None:
        skeleton = np.zeros((24, 24), dtype=bool)
        skeleton[12, 3:21] = True
        result = build_point_conditioned_graph(skeleton, [point(3, 12), point(20, 3)], 24.0, 0.05)
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["diagnostics"]["rejected_point_count"], 1)
        self.assertEqual(len(result["edges"]), 0)

    def test_y_shape_is_spanned_only_when_three_learned_tips_exist(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        for offset in range(9):
            skeleton[12 - offset, 12 - offset] = True
            skeleton[12 - offset, 12 + offset] = True
        skeleton[12:22, 12] = True
        three = build_point_conditioned_graph(
            skeleton, [point(4, 4), point(20, 4), point(12, 21)], 25.0, 0.05
        )
        two = build_point_conditioned_graph(skeleton, [point(4, 4), point(20, 4)], 25.0, 0.05)
        self.assertEqual(len(three["edges"]), 2)
        self.assertGreater(int(three["edge_union"].sum()), int(two["edge_union"].sum()))

    def test_duplicate_projection_keeps_higher_confidence(self) -> None:
        skeleton = np.zeros((24, 24), dtype=bool)
        skeleton[12, 3:21] = True
        result = build_point_conditioned_graph(
            skeleton, [point(8.0, 11.9, 0.2), point(8.1, 12.1, 0.9), point(20, 12)], 24.0, 0.05
        )
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(result["diagnostics"]["merged_duplicate_count"], 1)
        kept = [node for node in result["nodes"] if int(node["skeleton_node"]) != int(result["nodes"][-1]["skeleton_node"])]
        self.assertTrue(any(abs(float(node["score"]) - 0.9) < 1e-9 for node in kept))


if __name__ == "__main__":
    unittest.main()

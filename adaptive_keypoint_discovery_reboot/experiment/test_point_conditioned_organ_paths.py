#!/usr/bin/env python
"""Synthetic tests for learned-node-conditioned organ path decoding."""

from __future__ import annotations

import unittest

import numpy as np

from point_conditioned_graph import build_point_conditioned_graph
from point_conditioned_organ_paths import decode_candidate_organ_paths


def point(x: float, y: float, score: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "score": score}


def masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    shoot = np.zeros(shape, dtype=bool)
    root = np.zeros(shape, dtype=bool)
    shoot[:14] = True
    root[11:] = True
    return shoot, root


class PointConditionedOrganPathTests(unittest.TestCase):
    def test_straight_shoot_decodes_one_main_path(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        skeleton[3:22, 12] = True
        graph = build_point_conditioned_graph(skeleton, [point(12, 21), point(12, 3)], 25.0, 0.05)
        shoot, root = masks(skeleton.shape)
        for node in graph["nodes"]:
            node["organ_region_tolerant"] = "seed_base_root" if node["projected_xy"][1] > 14 else "shoot"
        paths, diagnostics = decode_candidate_organ_paths(graph, shoot, root, 25.0)
        self.assertEqual(diagnostics["failure"], "")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["path_kind"], "main_axis")

    def test_y_shape_decodes_main_and_lateral_path(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        for offset in range(9):
            skeleton[12 - offset, 12 - offset] = True
            skeleton[12 - offset, 12 + offset] = True
        skeleton[12:22, 12] = True
        graph = build_point_conditioned_graph(
            skeleton, [point(4, 4), point(20, 4), point(12, 21)], 25.0, 0.05
        )
        shoot, root = masks(skeleton.shape)
        for node in graph["nodes"]:
            node["organ_region_tolerant"] = "seed_base_root" if node["projected_xy"][1] > 14 else "shoot"
        paths, diagnostics = decode_candidate_organ_paths(graph, shoot, root, 25.0)
        self.assertEqual(diagnostics["failure"], "")
        self.assertEqual(len(paths), 2)
        self.assertEqual({path["path_kind"] for path in paths}, {"main_axis", "lateral_branch"})

    def test_missing_learned_tip_cannot_be_autocompleted(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        for offset in range(9):
            skeleton[12 - offset, 12 - offset] = True
            skeleton[12 - offset, 12 + offset] = True
        skeleton[12:22, 12] = True
        shoot, root = masks(skeleton.shape)
        full = build_point_conditioned_graph(
            skeleton, [point(4, 4), point(20, 4), point(12, 21)], 25.0, 0.05
        )
        missing = build_point_conditioned_graph(skeleton, [point(4, 4), point(12, 21)], 25.0, 0.05)
        for graph in (full, missing):
            for node in graph["nodes"]:
                node["organ_region_tolerant"] = "seed_base_root" if node["projected_xy"][1] > 14 else "shoot"
        full_paths, _ = decode_candidate_organ_paths(full, shoot, root, 25.0)
        missing_paths, _ = decode_candidate_organ_paths(missing, shoot, root, 25.0)
        self.assertEqual(len(full_paths), 2)
        self.assertEqual(len(missing_paths), 1)


if __name__ == "__main__":
    unittest.main()

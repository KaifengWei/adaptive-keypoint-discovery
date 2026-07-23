#!/usr/bin/env python
"""Synthetic tests for learned-node-conditioned organ path decoding."""

from __future__ import annotations

import unittest

import cv2
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

    def test_phenotype_base_is_an_existing_node_near_shoot_side_transition(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        skeleton[3:22, 12] = True
        graph = build_point_conditioned_graph(skeleton, [point(12, 21), point(12, 3)], 25.0, 0.05)
        shoot, root = masks(skeleton.shape)
        phenotype_roi = np.zeros_like(skeleton)
        phenotype_roi[3:22, 10:15] = True
        transition = np.zeros_like(skeleton)
        transition[19:22, 10:15] = True
        for node in graph["nodes"]:
            node["organ_region_tolerant"] = "shoot"
        paths, diagnostics = decode_candidate_organ_paths(
            graph,
            shoot,
            root,
            25.0,
            phenotype_roi_mask=phenotype_roi,
            basal_transition_mask=transition,
        )
        self.assertEqual(diagnostics["failure"], "")
        self.assertEqual(diagnostics["base_selection_rule"], "learned node inside phenotype ROI nearest shoot-side basal transition")
        self.assertEqual(len(paths), 1)

    def test_phenotype_base_does_not_synthesize_a_missing_transition_node(self) -> None:
        skeleton = np.zeros((25, 25), dtype=bool)
        skeleton[3:22, 12] = True
        graph = build_point_conditioned_graph(skeleton, [point(12, 3), point(12, 6)], 25.0, 0.05)
        shoot, root = masks(skeleton.shape)
        phenotype_roi = np.zeros_like(skeleton)
        phenotype_roi[3:22, 10:15] = True
        transition = np.zeros_like(skeleton)
        transition[20:22, 10:15] = True
        for node in graph["nodes"]:
            node["organ_region_tolerant"] = "shoot"
        paths, diagnostics = decode_candidate_organ_paths(
            graph,
            shoot,
            root,
            100.0,
            phenotype_roi_mask=phenotype_roi,
            basal_transition_mask=transition,
        )
        self.assertEqual(paths, [])
        self.assertEqual(diagnostics["failure"], "no_learned_node_near_shoot_side_transition")

    def test_local_scale_keeps_a_short_learned_leaf_that_global_bbox_pruning_drops(self) -> None:
        skeleton = np.zeros((45, 45), dtype=bool)
        skeleton[18:38, 22] = True
        for offset in range(17):
            skeleton[18 - offset, 22 - offset] = True
        for offset in range(9):
            skeleton[18 - offset, 22 + offset] = True
        graph = build_point_conditioned_graph(
            skeleton,
            [point(6, 2), point(30, 10), point(22, 37)],
            400.0,
            0.05,
        )
        for node in graph["nodes"]:
            node["organ_region_tolerant"] = "shoot"
        phenotype_roi = cv2.dilate(
            skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)
        ) > 0
        transition = np.zeros_like(skeleton)
        transition[34:40, 19:26] = True
        shoot = phenotype_roi.copy()
        root = np.zeros_like(skeleton)

        global_paths, global_diagnostics = decode_candidate_organ_paths(
            graph,
            shoot,
            root,
            400.0,
            phenotype_roi_mask=phenotype_roi,
            basal_transition_mask=transition,
            branch_pruning_mode="global_bbox",
        )
        local_paths, local_diagnostics = decode_candidate_organ_paths(
            graph,
            shoot,
            root,
            400.0,
            phenotype_roi_mask=phenotype_roi,
            basal_transition_mask=transition,
            branch_pruning_mode="local_learned_support",
        )
        self.assertEqual(len(global_paths), 1)
        self.assertEqual(len(local_paths), 2)
        self.assertEqual(local_diagnostics["branch_pruning_mode"], "local_learned_support")
        accepted = [
            row for row in local_diagnostics["terminal_branch_decisions"] if row["accepted"]
        ]
        self.assertEqual(len(accepted), 1)
        self.assertLess(
            accepted[0]["minimum_branch_length_px"],
            global_diagnostics["minimum_lateral_branch_length_px"],
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
"""CPU tests for the structure-coverage-enhanced automatic teacher."""

from __future__ import annotations

import argparse
import unittest

import cv2
import numpy as np

import g1_prime_structural_support as gp
from generate_g1prime_pseudolabels import (
    consensus,
    inverse_mapped_records,
    transformed_mask,
)


class StructureCoverageTeacherTests(unittest.TestCase):
    def test_adjacent_endpoint_pixels_merge_but_distinct_terminals_remain(self) -> None:
        endpoints = np.zeros((80, 80), dtype=bool)
        endpoints[10, 10] = True
        endpoints[14, 11] = True
        endpoints[18, 12] = True
        endpoints[60, 60] = True
        centers = gp.merged_endpoint_centers(endpoints, bbox_diag=250.0)
        self.assertEqual(len(centers), 2)

    def test_basal_transition_is_automatic_and_lies_on_the_current_skeleton(self) -> None:
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        cv2.line(image, (15, 64), (112, 64), (20, 150, 20), 9)
        transition = np.zeros((128, 128), dtype=bool)
        transition[54:75, 8:35] = True
        features = np.zeros((9, 9, 2), dtype=np.float32)
        attention = np.zeros((9, 9), dtype=np.float32)
        _, records, _, skeleton, diagnostics = gp.structural_candidates(
            image,
            features,
            attention,
            max_points=30,
            evidence_mode="geometry_only",
            structure_coverage=True,
            basal_transition_mask=transition,
        )
        basal = [
            row for row in records if "basal_transition" in row.get("coverage_roles", [])
        ]
        self.assertEqual(len(basal), 1)
        x, y = int(round(basal[0]["x"])), int(round(basal[0]["y"]))
        self.assertTrue(skeleton[y, x])
        self.assertEqual(diagnostics["structure_coverage_enabled"], 1.0)

    def test_transformed_mask_and_inverse_records_remain_aligned(self) -> None:
        size = 32
        mask = np.zeros((size, size), dtype=bool)
        mask[8:12, 5:9] = True
        matrix = np.asarray(
            [[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        flipped = transformed_mask(mask, matrix, size)
        self.assertEqual(int(flipped.sum()), int(mask.sum()))

        points = np.asarray([[25.0, 9.0], [40.0, 9.0]])
        records = [{"kind": "kept"}, {"kind": "outside"}]
        mapped, mapped_records = inverse_mapped_records(points, records, matrix, size)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped_records[0]["kind"], "kept")
        self.assertTrue(np.allclose(mapped[0], [6.0, 9.0]))

    def test_other_views_prove_stability_but_do_not_duplicate_basal_roles(self) -> None:
        outputs = [
            {
                "points": np.asarray([[10.0, 10.0]]),
                "records": [
                    {
                        "kind": "endpoint",
                        "score": 1.0,
                        "coverage_roles": ["terminal"],
                    }
                ],
            },
            {
                "points": np.asarray([[10.0, 10.0]]),
                "records": [
                    {
                        "kind": "basal_transition",
                        "score": 1.0,
                        "coverage_roles": ["basal_transition"],
                    }
                ],
            },
        ]
        transforms = [
            {"name": "identity", "matrix": np.eye(3)},
            {"name": "brightness", "matrix": np.eye(3)},
        ]
        args = argparse.Namespace(
            size=32,
            no_consistency_filter=False,
            min_presence=0.75,
            max_localization_error=0.025,
        )
        accepted, _ = consensus(outputs, transforms, bbox_diag=20.0, args=args)
        self.assertEqual(accepted[0]["coverage_roles"], ["terminal"])
        self.assertEqual(
            accepted[0]["observed_coverage_roles"],
            ["basal_transition", "terminal"],
        )


if __name__ == "__main__":
    unittest.main()

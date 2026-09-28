"""Synthetic-only checks for the administrative V4 GT freeze adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import freeze_test_gt as freeze


class FreezeSyntheticTests(unittest.TestCase):
    def test_measurable_and_existence_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_stage = freeze.STAGE
            freeze.STAGE = Path(tmp)
            try:
                dataset_id = "synthetic_test_1"
                mask_path = freeze.STAGE / "masks" / "shoot" / "test" / f"{dataset_id}.png"
                mask_path.parent.mkdir(parents=True)
                mask = np.zeros((100, 120), dtype=np.uint8)
                mask[20:70, 10:100] = 255
                Image.fromarray(mask).save(mask_path)
                row = {
                    "dataset_id": dataset_id,
                    "split": "test",
                    "source_frame_id": "synthetic-frame",
                    "output_sha256": "synthetic-image-hash",
                    "shoot_mask_relative_path": f"masks/shoot/test/{dataset_id}.png",
                    "source_crop_box_fullplant": json.dumps([0, 0, 104, 84]),
                    "normalization_padding_pixels": "8",
                }
                base = [10.0, 50.0]
                record = {
                    "blind_id": "SYNTHETICBLINDID1234", "image_sha256": "synthetic-image-hash",
                    "width": 120, "height": 100, "session_id": "synthetic-session",
                    "submitted_at": "2026-01-01T00:00:00Z", "base_xy": base, "image_note": "",
                    "items": [
                        {"gt_id": "GT01", "trace_uuid": "raw-1", "visibility_status": "measurable",
                         "confidence": "high", "occlusion": "none", "interpolation_used": "no",
                         "tip_xy": [80.0, 50.0], "points_px": [base, [50.0, 50.0], [80.0, 50.0]], "note": ""},
                        {"gt_id": "GT02", "trace_uuid": "raw-2", "visibility_status": "visible_unmeasurable",
                         "confidence": "medium", "occlusion": "major", "interpolation_used": "no",
                         "tip_xy": None, "points_px": [], "note": "synthetic occlusion"},
                    ],
                }
                plant, _ = freeze._plant(record, {dataset_id: row},
                                          {"dataset_id": dataset_id, "expected_sha256": "synthetic-image-hash",
                                           "source_frame_id": "synthetic-frame", "record_sha256": "synthetic-record"},
                                          {dataset_id: {"shoot_pixels": str(90 * 50)}})
                self.assertEqual(plant["operational_independent_path_count"], 2)
                self.assertEqual(plant["measurable_reference_path_count"], 1)
                self.assertEqual(len(plant["paths"][0]["points_px"]), 240)
                self.assertTrue(plant["paths"][0]["reference_main_path"])
                self.assertIsNone(plant["structures"][1]["points_px"])
                self.assertAlmostEqual(plant["bbox_diagonal_px"], (90**2 + 50**2) ** 0.5)
                self.assertAlmostEqual(plant["mm_per_output_px_scanner_metadata_derived"], 25.4 / 600)
            finally:
                freeze.STAGE = old_stage


if __name__ == "__main__":
    unittest.main()

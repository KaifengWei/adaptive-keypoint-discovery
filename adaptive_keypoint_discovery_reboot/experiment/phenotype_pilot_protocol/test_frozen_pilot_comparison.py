"""Small integrity checks for the frozen phenotype pilot comparison."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from evaluate_frozen_phenotype_pilot import _source_curve


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "data_stage_clean_v4_fullplant_candidate"


class SourceCoordinateTests(unittest.TestCase):
    def test_inverse_letterbox_matches_all_locked_image_dimensions(self) -> None:
        rows = json.loads((HERE / "phenotype_pilot_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 16)
        for row in rows:
            with self.subTest(dataset_id=row["dataset_id"]):
                width, height = int(row["standardized_width_px"]), int(row["standardized_height_px"])
                image = DATASET / "images" / "val" / f"{row['dataset_id']}.png"
                with Image.open(image) as opened:
                    self.assertEqual(opened.size, (width, height))
                scale = min(518 / width, 518 / height)
                new_width = max(1, round(width * scale))
                new_height = max(1, round(height * scale))
                pad = np.array([(518 - new_width) // 2, (518 - new_height) // 2])
                original = np.array([
                    [0.1 * width, 0.2 * height],
                    [0.5 * width, 0.5 * height],
                    [0.9 * width, 0.8 * height],
                ])
                model = original * scale + pad
                recovered = _source_curve(model.tolist(), width, height)
                np.testing.assert_allclose(recovered, original, atol=1e-9, rtol=0)

    def test_source_curve_rejects_malformed_points(self) -> None:
        with self.assertRaises(ValueError):
            _source_curve([[1, 2, 3]], 1000, 300)


if __name__ == "__main__":
    unittest.main()

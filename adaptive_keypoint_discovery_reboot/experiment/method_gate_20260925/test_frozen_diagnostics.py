"""Non-training checks for frozen val-only diagnostic adapters."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "phenotype_pilot_protocol"))

from evaluate_frozen_phenotype_pilot import _source_curve  # noqa: E402
from phenotype_gt_geometry import trace_metrics  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402
from run_frozen_diagnostics import phenotype  # noqa: E402


class FrozenDiagnosticAdapterTest(unittest.TestCase):
    def test_frozen_student_source_geometry(self) -> None:
        selection = pd.read_csv(HERE / "phenotype_pilot_protocol/phenotype_pilot_selection.csv")
        self.assertEqual(len(selection), 16)
        manifest = pd.read_csv(HERE / "data_stage_clean_v4_fullplant_candidate/manifests/val.csv").set_index("dataset_id")
        paths_by_id: dict[str, list[dict]] = {}
        source = HERE / "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/paths.jsonl"
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            paths_by_id.setdefault(row["dataset_id"], []).append(row)
        observed = 0
        for ident in selection.dataset_id:
            record = manifest.loc[ident].to_dict()
            record["dataset_id"] = ident
            _, mapping, _, _ = load_phenotype_input(HERE / "data_stage_clean_v4_fullplant_candidate", record, 518)
            width, height = int(mapping["source_width"]), int(mapping["source_height"])
            for path in paths_by_id.get(ident, []):
                source_curve = _source_curve(path["full_base_to_tip_path"], width, height)
                own = phenotype([path], 300.0, mapping)[0]
                expected = trace_metrics(source_curve, 300.0 / mapping["scale"])
                self.assertTrue(np.allclose(own["resampled_curve"], expected["resampled_curve"], atol=1e-8))
                self.assertAlmostEqual(own["structural_path_length_px"], expected["structural_path_length_px"], places=8)
                observed += 1
        self.assertGreater(observed, 0)


if __name__ == "__main__":
    unittest.main()

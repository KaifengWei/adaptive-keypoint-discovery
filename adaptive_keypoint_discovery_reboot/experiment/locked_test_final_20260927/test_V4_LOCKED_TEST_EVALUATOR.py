"""Synthetic-only acceptance tests. No V4 test images or predictions are read."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import V4_LOCKED_TEST_EVALUATOR as ev
from phenotype_gt_geometry import divergence_angle, select_reference_main, trace_metrics


def _sha(points: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps(points, separators=(",", ":")).encode()).hexdigest()


def _gt(dataset_id: str = "synthetic_01", *, two: bool = False, existence_only: bool = False,
        unresolved_angle: bool = False) -> dict:
    diag = 80.0
    controls = [
        [[10.0, 50.0], [40.0, 50.0], [80.0, 50.0]],
        [[10.0, 50.0], [40.0, 50.0], [65.0, 20.0]],
    ][:2 if two else 1]
    paths = []
    for index, control in enumerate(controls, 1):
        metric = trace_metrics(control, diag)
        points = metric["resampled_curve"].tolist()
        paths.append({
            "gt_id": f"GT{index:02}", "trace_uuid": _sha(points), "points_px": points,
            "length_px": metric["structural_path_length_px"],
            "structural_path_length_px": metric["structural_path_length_px"],
            "chord_length_px": metric["chord_length_px"],
            "tip_clockwise_angle_deg": metric["tip_clockwise_angle_deg"],
            "visibility_status": "measurable", "reference_main_path": False,
            "divergence_angle_deg": None, "divergence_angle_status": "main_path",
        })
    main = select_reference_main(paths, diag)
    main_curve = next(p["points_px"] for p in paths if p["trace_uuid"] == main)
    for p in paths:
        p["reference_main_path"] = p["trace_uuid"] == main
        if not p["reference_main_path"]:
            angle = divergence_angle(np.asarray(main_curve), np.asarray(p["points_px"]), diag)
            p["divergence_angle_status"] = angle["status"]
            p["divergence_angle_deg"] = angle.get("divergence_angle_deg")
            if unresolved_angle:
                # A branch that the frozen geometry cannot resolve is tested
                # separately by replacing both stored and recomputed curves.
                p["divergence_angle_status"] = "divergence_unresolved"
                p["divergence_angle_deg"] = None
    structures = copy.deepcopy(paths)
    if existence_only:
        structures.append({
            "gt_id": "GT03", "original_trace_uuid": "presence-only",
            "visibility_status": "visible_unmeasurable", "trace_uuid": None,
            "points_px": None, "divergence_angle_status": "geometry_unavailable",
        })
    return {
        "dataset_id": dataset_id, "split": "test", "source_frame_id": "synthetic_frame_A",
        "image_sha256": f"synthetic-image-{dataset_id}", "blind_id": f"SYNTHETIC{dataset_id}",
        "bbox_diagonal_px": diag, "base_xy": [10.0, 50.0],
        "mm_per_output_px_scanner_metadata_derived": 25.4 / 600,
        "reference_main_trace_uuid": main, "paths": paths, "structures": structures,
    }


def _model_path(path_id: str, controls: list[list[float]]) -> dict:
    return {
        "path_id": path_id,
        "full_base_to_tip_path": [[x * 5.18, y * 5.18] for x, y in controls],
        "provenance": {"synthetic": True},
    }


def _prediction(plant: dict, paths: list[dict] | None = None, *, base: list[float] | None = None) -> dict:
    paths = [] if paths is None else paths
    return {
        "dataset_id": plant["dataset_id"], "image_sha256": plant["image_sha256"],
        "source_frame_id": plant["source_frame_id"], "width": 100, "height": 100,
        "paths": paths, "base_xy_model_canvas": base if base is not None else ([51.8, 259.0] if paths else None),
        "basal_node_status": "ok" if paths or base is not None else "no_eligible_basal_node",
        "graph_status": "ok", "decoder_status": "ok" if paths else "zero_path",
        "point_count": 3 if paths else 0, "accepted_node_count": 2 if paths else 0,
        "rejected_node_count": 1 if paths else 0, "graph_node_count": 3 if paths else 0,
        "underground_association_count": 0, "provenance": {"synthetic": True},
    }


class EvaluatorSyntheticTests(unittest.TestCase):
    def test_perfect_match_and_wrong_base(self) -> None:
        gt = _gt()
        path = _model_path("p1", [[10, 50], [40, 50], [80, 50]])
        row, outcomes, source = ev.evaluate_plant("Teacher-direct", gt, _prediction(gt, [path]))
        self.assertEqual((row["matched_count"], row["missed_measurable_count"]), (1, 0))
        self.assertEqual(row["complete_plant_recovery_interval"], [1, 1])
        self.assertAlmostEqual(row["length_error_pct_conditional_mean"], 0, places=6)
        self.assertEqual([x["status"] for x in outcomes], ["matched"])
        self.assertEqual(source["prediction_count"], 1)
        wrong, _, _ = ev.evaluate_plant("Teacher-direct", gt, _prediction(gt, [path], base=[400, 400]))
        self.assertEqual(wrong["base_failure_status"], "base_beyond_0.025D")
        self.assertEqual(wrong["complete_plant_recovery_interval"], [0, 0])
        missing_base = _prediction(gt, [path])
        missing_base["base_xy_model_canvas"] = None
        missing, _, _ = ev.evaluate_plant("Teacher-direct", gt, missing_base)
        self.assertEqual(missing["base_failure_status"], "missing_output_base")

    def test_zero_prediction_and_all_missing_geometry_match(self) -> None:
        gt = _gt()
        row, outcomes, _ = ev.evaluate_plant("Student-B", gt, _prediction(gt))
        self.assertTrue(row["zero_prediction"])
        self.assertTrue(row["zero_path_collapse"])
        self.assertEqual(row["missed_measurable_count"], 1)
        self.assertEqual(row["precision_interval"], [None, None])
        self.assertIsNone(row["length_error_pct_conditional_mean"])
        self.assertEqual(outcomes[0]["status"], "unmatched_session_1")

    def test_partial_recall_and_one_to_one_matching(self) -> None:
        gt = _gt(two=True)
        path = _model_path("p1", [[10, 50], [40, 50], [80, 50]])
        row, outcomes, _ = ev.evaluate_plant("Teacher-direct", gt, _prediction(gt, [path]))
        self.assertEqual(row["matched_count"], 1)
        self.assertEqual(row["measurable_recall"], 0.5)
        self.assertEqual(sum(x["status"] == "matched" for x in outcomes), 1)
        self.assertEqual(sum(x["status"] == "unmatched_session_1" for x in outcomes), 1)
        single = _gt()
        duplicate = _model_path("p2", [[10, 50], [40, 50], [80, 50]])
        two_pred, outcomes, _ = ev.evaluate_plant("Student-D", single, _prediction(single, [path, duplicate]))
        self.assertEqual((two_pred["matched_count"], two_pred["unmatched_prediction_count"]), (1, 1))
        self.assertEqual(two_pred["possible_extra_interval"], [1, 1])
        self.assertEqual(two_pred["precision_interval"], [0.5, 0.5])

    def test_existence_only_interval_and_unresolved_angle(self) -> None:
        gt = _gt(existence_only=True)
        paths = [
            _model_path("p1", [[10, 50], [40, 50], [80, 50]]),
            _model_path("p2", [[10, 50], [30, 70], [70, 85]]),
        ]
        row, outcomes, _ = ev.evaluate_plant("Student-B", gt, _prediction(gt, paths))
        self.assertEqual(row["gt_existence_only_count"], 1)
        self.assertEqual(row["existence_recall_interval"], [0.5, 1.0])
        self.assertEqual(row["precision_interval"], [0.5, 1.0])
        self.assertEqual(row["possible_extra_interval"], [0, 1])
        self.assertIn("gt_visible_unmeasurable", [x["status"] for x in outcomes])
        # Synthetic unresolved branch: force the frozen function to return
        # unresolved by giving the branch the same curve as main with a
        # distinct path identity; its angle is correctly omitted.
        same = _gt(two=True)
        same["paths"][1]["points_px"] = copy.deepcopy(same["paths"][0]["points_px"])
        same["paths"][1]["trace_uuid"] = "synthetic-same-curve-different-identity"
        same["paths"][1]["length_px"] = same["paths"][0]["length_px"]
        same["paths"][1]["structural_path_length_px"] = same["paths"][0]["structural_path_length_px"]
        same["paths"][1]["chord_length_px"] = same["paths"][0]["chord_length_px"]
        same["paths"][1]["tip_clockwise_angle_deg"] = same["paths"][0]["tip_clockwise_angle_deg"]
        same["reference_main_trace_uuid"] = select_reference_main(same["paths"], 80.0)
        for item in same["paths"]:
            item["reference_main_path"] = item["trace_uuid"] == same["reference_main_trace_uuid"]
            item["divergence_angle_status"] = "main_path" if item["reference_main_path"] else "divergence_unresolved"
            item["divergence_angle_deg"] = None
        same["structures"] = copy.deepcopy(same["paths"])
        if any(item["divergence_angle_deg"] is not None for item in ev._gt_traces(same)):
            self.fail("Unresolved GT angle must stay missing")
        duplicate_paths = [
            _model_path("p1", [[10, 50], [40, 50], [80, 50]]),
            _model_path("p2", [[10, 50], [40, 50], [80, 50]]),
        ]
        unresolved, matched, _ = ev.evaluate_plant(
            "Teacher-direct", same, _prediction(same, duplicate_paths)
        )
        self.assertEqual(unresolved["matched_angle_n"], 0)
        self.assertEqual(unresolved["matched_gt_angle_unresolved_count"], 1)
        self.assertEqual(unresolved["matched_pred_angle_unresolved_count"], 1)
        self.assertTrue(all(p["angle_absolute_error_deg"] is None for p in matched))

    def test_shared_gt_identity_for_conditional_pairing(self) -> None:
        gt = _gt(two=True)
        p1 = _model_path("p1", [[10, 50], [40, 50], [80, 50]])
        p2 = _model_path("p2", [[10, 50], [40, 50], [65, 20]])
        teacher, _, _ = ev.evaluate_plant("Teacher-direct", gt, _prediction(gt, [p1]))
        student, _, _ = ev.evaluate_plant("Student-B", gt, _prediction(gt, [p2]))
        d, _, _ = ev.evaluate_plant("Student-D", gt, _prediction(gt, [p1]))
        rows = {method: [row] for method, row in zip(ev.METHODS, (teacher, student, d))}
        paired = ev._paired_rows(rows)
        primary = next(x for x in paired if x["contrast"] == "Student-B - Teacher-direct")
        self.assertEqual(primary["shared_length_path_n"], 0)
        self.assertIsNone(primary["conditional_length_error_diff_pct"])
        secondary = next(x for x in paired if x["contrast"] == "Student-D - Teacher-direct")
        self.assertEqual(secondary["shared_length_path_n"], 1)

    def test_complete_40x3_and_bootstrap_reproducible_with_all_na(self) -> None:
        plants = [_gt(f"synthetic_{i:02d}") for i in range(40)]
        for i, plant in enumerate(plants):
            plant["source_frame_id"] = f"synthetic_frame_{i // 5}"
        bundle = {
            "schema_version": "v4-locked-test-saved-predictions-v1",
            "split": "test", "pipeline_v1_sha256": ev.pipeline_v1_hash(),
            "method_weights_sha256": ev.METHOD_WEIGHTS_SHA256,
            "methods": {method: [_prediction(p) for p in plants] for method in ev.METHODS},
        }
        gt = {"plants": plants}
        first = ev.evaluate_synthetic_or_saved(gt, bundle, n_bootstrap=ev.BOOTSTRAP_N)
        second = ev.evaluate_synthetic_or_saved(gt, bundle, n_bootstrap=ev.BOOTSTRAP_N)
        self.assertEqual(ev.canonical_json(first), ev.canonical_json(second))
        self.assertEqual(len(first["plant_level"]), 120)
        self.assertEqual(len(first["provenance"]), 120)
        self.assertEqual(len(first["paired_plant_level"]), 120)
        length = first["statistics"]["methods"]["Teacher-direct"]["plant_macro"]["conditional_length_error_pct"]
        self.assertEqual(length["n_valid_plants"], 0)
        self.assertEqual(length["plant_cluster_bootstrap"]["ci95"], [None, None])
        self.assertEqual(length["plant_cluster_bootstrap"]["valid_replicates"], 0)
        self.assertEqual(first["statistics"]["bootstrap"]["source_frame_count"], 8)
        self.assertEqual(first["statistics"]["paired_contrasts"]["Student-B - Teacher-direct"]["priority"], "primary")
        figures = first["preregistered_figure_rank"]
        self.assertEqual(len(figures["all_40_ranked"]), 40)
        self.assertEqual(figures["all_40_ranked"][0]["rank"], 1)
        self.assertEqual(figures["all_40_ranked"][19]["rank"], 20)
        self.assertTrue(all(r["missing_matched_length_for_ranking"] for r in figures["all_40_ranked"]))

    def test_prediction_schema_rejects_missing_method_or_plant(self) -> None:
        gt = {"plants": [_gt(f"synthetic_{i:02d}") for i in range(40)]}
        bundle = {
            "schema_version": "v4-locked-test-saved-predictions-v1", "split": "test",
            "pipeline_v1_sha256": ev.pipeline_v1_hash(),
            "method_weights_sha256": ev.METHOD_WEIGHTS_SHA256,
            "methods": {method: [_prediction(p) for p in gt["plants"]] for method in ev.METHODS},
        }
        ev.validate_prediction_bundle(bundle, gt)
        del bundle["methods"]["Student-D"]
        with self.assertRaises(ValueError):
            ev.validate_prediction_bundle(bundle, gt)
        bundle["methods"]["Student-D"] = [_prediction(p) for p in gt["plants"][:-1]]
        with self.assertRaises(ValueError):
            ev.validate_prediction_bundle(bundle, gt)

    def test_inverse_letterbox_and_asset_dimensions(self) -> None:
        # Existing locked inverse-letterbox implementation, including pad.
        point = [[60.0, 40.0]]
        scale = min(518 / 120, 518 / 100)
        pad_y = (518 - round(100 * scale)) // 2
        model = [[point[0][0] * scale, point[0][1] * scale + pad_y]]
        self.assertTrue(np.allclose(ev._source_curve(model, 120, 100), point))
        gt = {"plants": [_gt(f"synthetic_{i:02d}") for i in range(40)]}
        bundle = {
            "schema_version": "v4-locked-test-saved-predictions-v1", "split": "test",
            "pipeline_v1_sha256": ev.pipeline_v1_hash(),
            "method_weights_sha256": ev.METHOD_WEIGHTS_SHA256,
            "methods": {method: [_prediction(p) for p in gt["plants"]] for method in ev.METHODS},
        }
        manifest = {"samples": [
            {"blind_id": p["blind_id"], "width": 100, "height": 100, "image_sha256": p["image_sha256"]}
            for p in gt["plants"]
        ]}
        ev.validate_asset_dimensions(bundle, gt, manifest)
        bundle["methods"]["Student-B"][0]["height"] = 99
        with self.assertRaises(ValueError):
            ev.validate_asset_dimensions(bundle, gt, manifest)

    def test_saved_val_geometry_parity_without_model_rerun(self) -> None:
        experiment = Path(__file__).resolve().parent.parent
        protocol = experiment / "phenotype_pilot_protocol"
        runtime = protocol / "runtime"
        comparison_file = runtime / "method_comparison" / "frozen_pilot_20260924_v5_final" / "frozen_method_comparison.json"
        teacher_paths = runtime / "method_comparison" / "teacher_direct_route_b_local_decoder_val_20260924" / "paths.jsonl"
        val_gt_file = runtime / "final_gt" / "phenotype_pilot_20260924_v1" / "final_gt.json"
        selection_file = protocol / "phenotype_pilot_selection.json"
        if not all(p.is_file() for p in (comparison_file, teacher_paths, val_gt_file, selection_file)):
            self.skipTest("Private historical saved val files not present on this device")
        comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
        self.assertEqual(ev.sha(val_gt_file), comparison["gt_sha256"])
        self.assertEqual(ev.sha(selection_file), comparison["pilot_selection_sha256"])
        self.assertEqual(ev.sha(teacher_paths), comparison["method_source"]["Teacher-direct"]["paths_jsonl_sha256"])
        selection = {r["dataset_id"]: r for r in json.loads(selection_file.read_text(encoding="utf-8"))}
        val_gt = {r["dataset_id"]: r for r in json.loads(val_gt_file.read_text(encoding="utf-8"))["plants"]}
        saved_paths = [json.loads(line) for line in teacher_paths.read_text(encoding="utf-8").splitlines() if line.strip()]
        saved_by_key = {(p["dataset_id"], p["path_id"]): p for p in saved_paths}
        pairs = [(r["dataset_id"], pair) for r in comparison["methods"]["Teacher-direct"]["per_plant"] for pair in r["pairs"]]
        self.assertTrue(pairs)
        # Re-use a saved output only. No forward pass, image open, or test access.
        ident, pair = pairs[0]
        source = saved_by_key[(ident, pair["pred_path_id"])]
        row = {
            "dataset_id": ident,
            "width": int(selection[ident]["standardized_width_px"]),
            "height": int(selection[ident]["standardized_height_px"]),
            "paths": [{"path_id": source["path_id"],
                       "full_base_to_tip_path": source["full_base_to_tip_path"],
                       "provenance": {"saved_val": True}}],
        }
        calculated = ev._prediction_traces("Teacher-direct", row, float(val_gt[ident]["bbox_diagonal_px"]))
        self.assertAlmostEqual(calculated[0]["structural_path_length_px"], pair["pred_length_px"], places=8)


if __name__ == "__main__":
    unittest.main()

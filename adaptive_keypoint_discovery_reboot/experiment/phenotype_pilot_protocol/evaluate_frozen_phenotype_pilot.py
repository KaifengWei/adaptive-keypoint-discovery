"""Compare frozen val-only path methods with the frozen phenotype pilot GT.

Human GT is read-only. Matching and geometry use phenotype-geometry-v1. One
known independent structure has no accepted curve; it is reported as count-only
and never fabricated as a localization/length/angle target.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from phenotype_gt_geometry import (
    clockwise_angle_deg,
    divergence_angle,
    match_cross_session,
    select_reference_main,
    trace_metrics,
)


EXPECTED_GT_SHA256 = "1b3018c83b2695951dfdf2adff10020f9c3e6e739babfe12ad235a88ae7b53b3"
EXPECTED_METHODS = ("Teacher-direct", "Student-B", "Student-D")
LENGTH_MDC95_PCT = 3.72
ANGLE_MDC95_DEG = 17.08


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
    }


def _load_predictions(paths_file: Path, summary_file: Path, gt_ids: set[str]) -> tuple[dict[str, list[dict]], dict]:
    saved = json.loads(summary_file.read_text(encoding="utf-8"))
    required = {
        "images": 40,
        "split": "val",
        "test_images_read": 0,
        "projection_ratio": 0.025,
        "input_domain": "phenotype_roi_v1",
        "branch_pruning_mode": "local_learned_support",
    }
    for key, expected in required.items():
        if saved.get(key) != expected:
            raise ValueError(f"Frozen method mismatch: {paths_file}: {key}={saved.get(key)!r}")
    by_id: dict[str, list[dict]] = defaultdict(list)
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            path = json.loads(line)
            if path["dataset_id"] in gt_ids:
                by_id[path["dataset_id"]].append(path)
    if not set(by_id) <= gt_ids:
        raise ValueError("Method output contains an unexpected GT identity")
    return by_id, {
        "paths_jsonl_sha256": sha(paths_file),
        "saved_summary_sha256": sha(summary_file),
        "frozen_settings": required,
    }


def _gt_traces(plant: dict) -> list[dict]:
    bbox_diag = float(plant["bbox_diagonal_px"])
    result = []
    for path in plant["paths"]:
        # Re-run the same frozen phenotype-geometry-v1 evaluator used for
        # predictions. The stored reference and source coordinates stay intact.
        metrics = trace_metrics(path["points_px"], bbox_diag)
        curve = metrics["resampled_curve"]
        result.append({
            "trace_uuid": path["source_points_sha256"],
            "visibility_status": "measurable",
            "tip_xy": curve[-1].tolist(),
            "tip_clockwise_angle_deg": metrics["tip_clockwise_angle_deg"],
            "resampled_curve": curve,
            "length_px": metrics["structural_path_length_px"],
            "chord_length_px": metrics["chord_length_px"],
            "structural_path_length_px": metrics["structural_path_length_px"],
            "reference_main_path": path["reference_main_path"],
        })
    if result:
        selected = select_reference_main(result, bbox_diag)
        stored = next(trace["trace_uuid"] for trace in result if trace["reference_main_path"])
        if selected != stored:
            raise ValueError(f"GT main-path identity changed on re-evaluation: {plant['dataset_id']}")
        main_curve = next(trace["resampled_curve"] for trace in result if trace["reference_main_path"])
        for trace in result:
            if not trace["reference_main_path"]:
                angle = divergence_angle(main_curve, trace["resampled_curve"], bbox_diag)
                trace["divergence_angle_deg"] = angle.get("divergence_angle_deg")
    return result


def _source_curve(model_points: list[list[float]], width: int, height: int) -> np.ndarray:
    size = 518
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    pad_x, pad_y = (size - new_width) // 2, (size - new_height) // 2
    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Predicted path points must have shape (n, 2)")
    source = (points - np.asarray([pad_x, pad_y])) / scale
    source[:, 0] = np.clip(source[:, 0], 0.0, width - 1.0)
    source[:, 1] = np.clip(source[:, 1], 0.0, height - 1.0)
    return source


def _pred_traces(method: str, dataset_id: str, paths: list[dict], bbox_diag: float, width: int, height: int) -> list[dict]:
    result = []
    for path in paths:
        source_points = _source_curve(path["full_base_to_tip_path"], width, height)
        metrics = trace_metrics(source_points, bbox_diag)
        curve = metrics["resampled_curve"]
        if curve.shape != (240, 2):
            raise ValueError(f"Prediction resampling failed for {dataset_id}")
        result.append({
            "trace_uuid": f"{method}:{dataset_id}:{path['path_id']}",
            "visibility_status": "measurable",
            "tip_xy": curve[-1].tolist(),
            "tip_clockwise_angle_deg": clockwise_angle_deg(curve[0], curve[-1]),
            "resampled_curve": curve,
            "length_px": metrics["structural_path_length_px"],
            "chord_length_px": metrics["chord_length_px"],
            "structural_path_length_px": metrics["structural_path_length_px"],
            "path_id": path["path_id"],
        })
    if result:
        main_uuid = select_reference_main(result, bbox_diag)
        main_curve = next(trace["resampled_curve"] for trace in result if trace["trace_uuid"] == main_uuid)
        for trace in result:
            trace["reference_main_path"] = trace["trace_uuid"] == main_uuid
            if not trace["reference_main_path"]:
                angle = divergence_angle(main_curve, trace["resampled_curve"], bbox_diag)
                trace["divergence_angle_status"] = angle["status"]
                trace["divergence_angle_deg"] = angle.get("divergence_angle_deg")
    return result


def _one_plant(method: str, plant: dict, raw_paths: list[dict], width: int, height: int, mm_per_output_px: float) -> dict:
    dataset_id = plant["dataset_id"]
    bbox_diag = float(plant["bbox_diagonal_px"])
    gt = _gt_traces(plant)
    pred = _pred_traces(method, dataset_id, raw_paths, bbox_diag, width, height)
    matched = match_cross_session(gt, pred, bbox_diag)
    by_gt = {trace["trace_uuid"]: trace for trace in gt}
    by_pred = {trace["trace_uuid"]: trace for trace in pred}
    pairs = []
    for match in matched:
        if match.status != "matched":
            continue
        g = by_gt[match.trace_uuid_session_1]
        p = by_pred[match.trace_uuid_session_2]
        error_px = abs(p["length_px"] - g["length_px"])
        error_symmetric_pct = 100.0 * error_px / max((p["length_px"] + g["length_px"]) / 2.0, 1e-12)
        angle = None
        if not g["reference_main_path"] and not p["reference_main_path"] and g["divergence_angle_deg"] is not None and p.get("divergence_angle_deg") is not None:
            angle = abs(float(p["divergence_angle_deg"]) - float(g["divergence_angle_deg"]))
        pairs.append({
            "gt_path_sha256": g["trace_uuid"],
            "pred_path_id": p["path_id"],
            "matching_cost": match.cost,
            "tip_distance_bbox_normalized": match.tip_norm,
            "curve_distance_bbox_normalized": match.curve_norm,
            "gt_length_px": g["length_px"],
            "pred_length_px": p["length_px"],
            "gt_length_bbox_normalized": g["length_px"] / bbox_diag,
            "pred_length_bbox_normalized": p["length_px"] / bbox_diag,
            "gt_length_mm_derived_from_scanner_metadata": g["length_px"] * mm_per_output_px,
            "pred_length_mm_derived_from_scanner_metadata": p["length_px"] * mm_per_output_px,
            "length_absolute_error_px": error_px,
            "length_absolute_error_bbox_normalized": error_px / bbox_diag,
            "length_absolute_error_mm_derived_from_scanner_metadata": error_px * mm_per_output_px,
            "length_symmetric_relative_error_pct": error_symmetric_pct,
            "length_within_human_mdc95": error_symmetric_pct <= LENGTH_MDC95_PCT,
            "gt_branch": not g["reference_main_path"],
            "pred_branch": not p["reference_main_path"],
            "angle_absolute_error_deg": angle,
            "angle_within_human_mdc95": None if angle is None else angle <= ANGLE_MDC95_DEG,
        })
    missing = plant["operational_independent_path_count"] - plant["measurable_reference_path_count"]
    unmatched_pred = len(pred) - len(pairs)
    if len(pairs) > len(gt) or len(pairs) > len(pred):
        raise ValueError(f"Invalid matching counts for {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "pilot_group": plant["pilot_group"],
        "gt_independent_count": plant["operational_independent_path_count"],
        "gt_geometry_count": len(gt),
        "gt_count_only_no_curve": missing,
        "predicted_path_count": len(pred),
        "count_absolute_error": abs(len(pred) - plant["operational_independent_path_count"]),
        "matched_geometry_count": len(pairs),
        "missed_geometry_count": len(gt) - len(pairs),
        "unmatched_prediction_count": unmatched_pred,
        "unmatched_prediction_count_possible_true_interval": [max(0, unmatched_pred - missing), unmatched_pred],
        "gt_branch_count": sum(not trace["reference_main_path"] for trace in gt),
        "matched_angle_count": sum(pair["angle_absolute_error_deg"] is not None for pair in pairs),
        "pairs": pairs,
    }


def _aggregate(rows: list[dict]) -> dict:
    paths = [pair for row in rows for pair in row["pairs"]]
    gt_geometry = sum(row["gt_geometry_count"] for row in rows)
    gt_total = sum(row["gt_independent_count"] for row in rows)
    pred_total = sum(row["predicted_path_count"] for row in rows)
    matched = len(paths)
    missing_geometry = gt_total - gt_geometry
    unmatched_pred = pred_total - matched
    possible_false_low = sum(row["unmatched_prediction_count_possible_true_interval"][0] for row in rows)
    potentially_true_unmatched = sum(
        min(row["unmatched_prediction_count"], row["gt_count_only_no_curve"]) for row in rows
    )
    angle_values = [pair["angle_absolute_error_deg"] for pair in paths if pair["angle_absolute_error_deg"] is not None]
    return {
        "plants": len(rows),
        "gt_independent_structures": gt_total,
        "gt_complete_geometry": gt_geometry,
        "gt_count_only_no_curve": missing_geometry,
        "predicted_paths": pred_total,
        "geometry_matched": matched,
        "geometry_missed": gt_geometry - matched,
        "unmatched_predictions": unmatched_pred,
        "possible_false_prediction_interval": [possible_false_low, unmatched_pred],
        "geometry_recall": matched / gt_geometry if gt_geometry else None,
        "verified_prediction_precision_interval": [matched / pred_total if pred_total else None, (matched + potentially_true_unmatched) / pred_total if pred_total else None],
        "plant_count_absolute_error": _summary([row["count_absolute_error"] for row in rows]),
        "length_symmetric_relative_error_pct_matched_only": _summary([pair["length_symmetric_relative_error_pct"] for pair in paths]),
        "length_absolute_error_px_matched_only": _summary([pair["length_absolute_error_px"] for pair in paths]),
        "length_absolute_error_bbox_normalized_matched_only": _summary([pair["length_absolute_error_bbox_normalized"] for pair in paths]),
        "length_absolute_error_mm_derived_from_scanner_metadata_matched_only": _summary([pair["length_absolute_error_mm_derived_from_scanner_metadata"] for pair in paths]),
        "length_within_3_72pct_mdc95_matched_only": sum(pair["length_within_human_mdc95"] for pair in paths),
        "gt_branch_with_angle": sum(row["gt_branch_count"] for row in rows),
        "matched_angle_count": len(angle_values),
        "angle_absolute_error_deg_matched_only": _summary(angle_values),
        "angle_within_17_08deg_mdc95_matched_only": sum(pair["angle_within_human_mdc95"] is True for pair in paths),
    }


def evaluate(gt_file: Path, selection_file: Path, methods: dict[str, tuple[Path, Path]], output: Path) -> dict:
    if sha(gt_file) != EXPECTED_GT_SHA256:
        raise ValueError("Frozen GT checksum changed")
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    plants = gt["plants"]
    ids = {plant["dataset_id"] for plant in plants}
    if len(plants) != 16 or len(ids) != 16 or {plant["split"] for plant in plants} != {"val"}:
        raise ValueError("Pilot GT membership changed")
    selection_rows = json.loads(selection_file.read_text(encoding="utf-8"))
    selection = {row["dataset_id"]: row for row in selection_rows}
    if len(selection_rows) != 16 or set(selection) != ids:
        raise ValueError("Locked pilot image dimensions do not match GT identities")
    if any(float(row["source_dpi"]) != 600.0 or float(row["mm_per_output_px"]) <= 0 for row in selection_rows):
        raise ValueError("Pilot scanner-metadata length conversion changed")
    if set(methods) != set(EXPECTED_METHODS):
        raise ValueError("Teacher-direct, Student-B, and Student-D are all required")
    all_rows = {}
    provenance = {}
    for method in EXPECTED_METHODS:
        by_id, source = _load_predictions(*methods[method], ids)
        all_rows[method] = [
            _one_plant(
                method,
                plant,
                by_id.get(plant["dataset_id"], []),
                int(selection[plant["dataset_id"]]["standardized_width_px"]),
                int(selection[plant["dataset_id"]]["standardized_height_px"]),
                float(selection[plant["dataset_id"]]["mm_per_output_px"]),
            )
            for plant in plants
        ]
        provenance[method] = source
    result = {
        "protocol": "phenotype-first-pilot-v2 / phenotype-geometry-v1",
        "scope": "locked 16 V4 val plants; Core12 primary, Diagnostic4 separate",
        "gt_sha256": EXPECTED_GT_SHA256,
        "pilot_selection_sha256": sha(selection_file),
        "prediction_coordinate_transform": "inverse_518_letterbox_to_standardized_source_pixels_before_matching_and_metrics",
        "geometry_evaluation": "GT and predicted full base-to-tip paths both re-evaluated using phenotype-geometry-v1 PCHIP arc-length 240; frozen GT source coordinates and main identity not changed",
        "method_source": provenance,
        "missing_geometry_policy": "One GT structure is count-only. It is not used for localization, length, or angle; unmatched predictions on its plant are an interval rather than unequivocal false positives.",
        "physical_length_status": "derived from scanner metadata; no independent known-size scale calibration yet",
        "human_mdc95": {"length_symmetric_relative_pct": LENGTH_MDC95_PCT, "angle_deg": ANGLE_MDC95_DEG},
        "methods": {
            method: {
                "Core": _aggregate([row for row in all_rows[method] if row["pilot_group"] == "core"]),
                "Diagnostic": _aggregate([row for row in all_rows[method] if row["pilot_group"] == "diagnostic"]),
                "per_plant": all_rows[method],
            }
            for method in EXPECTED_METHODS
        },
        "selection_status": "comparison_generated_not_final_method_selection",
        "gt_modified": False,
        "models_modified": False,
        "test_read": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (output / "frozen_method_comparison.json").write_bytes(payload)
    compact = {method: {group: result["methods"][method][group] for group in ("Core", "Diagnostic")} for method in EXPECTED_METHODS}
    compact["comparison_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "summary.json").write_text(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--teacher-paths", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--student-b-paths", type=Path, required=True)
    parser.add_argument("--student-b-summary", type=Path, required=True)
    parser.add_argument("--student-d-paths", type=Path, required=True)
    parser.add_argument("--student-d-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = {
        "Teacher-direct": (args.teacher_paths, args.teacher_summary),
        "Student-B": (args.student_b_paths, args.student_b_summary),
        "Student-D": (args.student_d_paths, args.student_d_summary),
    }
    print(json.dumps(evaluate(args.gt, args.selection, methods, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

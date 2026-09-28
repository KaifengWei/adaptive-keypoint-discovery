"""Frozen offline evaluator for the V4 locked test.

This module does not import Teacher, Student, Torch, or image readers. It reads
only a frozen human GT JSON and, after separately authorized inference, a
complete bundle of already-saved predictions in the schema below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROTOCOL = EXPERIMENT / "phenotype_pilot_protocol"
sys.path.insert(0, str(PROTOCOL))

from evaluate_frozen_phenotype_pilot import _source_curve  # noqa: E402
from phenotype_gt_geometry import (  # noqa: E402
    GEOMETRY_PROTOCOL_VERSION,
    divergence_angle,
    match_cross_session,
    polyline_length,
    select_reference_main,
    trace_metrics,
)

GT_SHA256 = "8d3f33854075c841020bd063efdefe0550aaaa7b00c746639687ff35e00222ad"
ASSET_MANIFEST_SHA256 = "404b89b87b6ec3ea211069762de98718ad7ea47b3f55c54b6b06b45b74c87c45"
PREREG_SHA256 = "31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9"
GEOMETRY_SHA256 = "c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800"
PILOT_EVALUATOR_SHA256 = "b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8"
PIPELINE_V1_SOURCES = {
    "point_conditioned_graph.py": "3a733ee2fb4213938f92a0f91c969bffe2b4c8e23b82356f07fd03ebba9577ca",
    "point_conditioned_organ_paths.py": "f960485eb6265122955ccc3ffa8a944c2fa66127f8c9c5b58683b46d3c3ab357",
    "phenotype_roi_basal_anchor.py": "d4dfedc3b3fa25c7a086d276d3e4f9ab9728ca7c9a28b2130cb7f678041db696",
    "g1_prime_phenotype_bridge.py": "027a364330888b403823c98ff77deb5e00db8792ad35e0831a45e97138292117",
}
METHODS = ("Teacher-direct", "Student-B", "Student-D")
METHOD_WEIGHTS_SHA256 = {
    "Teacher-direct": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
    "Student-B": "bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2",
    "Student-D": "b904eed30832d1a2c6cc20aca97e3d0140cc4444235c6a3b4b4b606bab17ce4a",
}
CONTRASTS = (
    ("Student-B", "Teacher-direct", "primary"),
    ("Student-D", "Teacher-direct", "secondary"),
    ("Student-D", "Student-B", "secondary"),
)
BOOTSTRAP_N = 10_000
PLANT_SEED = 20260927
FRAME_SEED = 20260928
LENGTH_MDC95_PCT = 3.72
ANGLE_MDC95_DEG = 17.08
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def pipeline_v1_hash() -> str:
    return hashlib.sha256(json.dumps(PIPELINE_V1_SOURCES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources() -> None:
    required = {
        HERE / "V4_LOCKED_TEST_PREREGISTRATION.md": PREREG_SHA256,
        PROTOCOL / "phenotype_gt_geometry.py": GEOMETRY_SHA256,
        PROTOCOL / "evaluate_frozen_phenotype_pilot.py": PILOT_EVALUATOR_SHA256,
        **{EXPERIMENT / name: digest for name, digest in PIPELINE_V1_SOURCES.items()},
    }
    for path, expected in required.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path.name}")


def _point(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def validate_prediction_bundle(bundle: dict, gt: dict) -> None:
    """Require an explicit 40 x 3 saved-output matrix; never infer missing rows."""
    if set(bundle) != {"schema_version", "split", "pipeline_v1_sha256", "method_weights_sha256", "methods"}:
        raise ValueError("Prediction bundle fields changed")
    if (bundle["schema_version"], bundle["split"], bundle["pipeline_v1_sha256"]) != (
        "v4-locked-test-saved-predictions-v1", "test", pipeline_v1_hash()
    ):
        raise ValueError("Prediction bundle version/split/pipeline mismatch")
    if set(bundle["methods"]) != set(METHODS):
        raise ValueError("Prediction bundle must include exactly Teacher-direct/B/D")
    if bundle["method_weights_sha256"] != METHOD_WEIGHTS_SHA256:
        raise ValueError("Saved method-weight provenance does not match the preregistration")
    plants = {p["dataset_id"]: p for p in gt["plants"]}
    if len(plants) != 40 or {p["split"] for p in plants.values()} != {"test"}:
        raise ValueError("Frozen GT membership changed")
    required = {
        "dataset_id", "image_sha256", "width", "height", "source_frame_id",
        "paths", "base_xy_model_canvas", "basal_node_status", "graph_status",
        "decoder_status", "point_count", "accepted_node_count",
        "rejected_node_count", "graph_node_count", "underground_association_count",
        "provenance",
    }
    for method in METHODS:
        rows = bundle["methods"][method]
        if not isinstance(rows, list) or len(rows) != 40:
            raise ValueError(f"{method}: require all 40 saved plants")
        seen = set()
        for row in rows:
            if set(row) != required:
                raise ValueError(f"{method}: plant output fields changed")
            ident = row["dataset_id"]
            if ident not in plants or ident in seen:
                raise ValueError(f"{method}: unknown or duplicate plant")
            seen.add(ident)
            plant = plants[ident]
            if row["image_sha256"] != plant["image_sha256"] or row["source_frame_id"] != plant["source_frame_id"]:
                raise ValueError(f"{method}: image/frame identity mismatch")
            if type(row["width"]) is not int or type(row["height"]) is not int or min(row["width"], row["height"]) < 1:
                raise ValueError(f"{method}: invalid image dimensions")
            if row["basal_node_status"] not in {"ok", "no_eligible_basal_node", "graph_failure"}:
                raise ValueError(f"{method}: invalid basal status")
            if row["graph_status"] not in {"ok", "failed"} or row["decoder_status"] not in {"ok", "zero_path", "failed"}:
                raise ValueError(f"{method}: invalid graph/decoder status")
            if row["base_xy_model_canvas"] is not None and not _point(row["base_xy_model_canvas"]):
                raise ValueError(f"{method}: invalid output base")
            if row["basal_node_status"] != "ok" and row["base_xy_model_canvas"] is not None:
                raise ValueError(f"{method}: failed basal node cannot have output base")
            for count in ("point_count", "accepted_node_count", "rejected_node_count", "graph_node_count", "underground_association_count"):
                if type(row[count]) is not int or row[count] < 0:
                    raise ValueError(f"{method}: invalid {count}")
            if not isinstance(row["provenance"], dict) or not row["provenance"]:
                raise ValueError(f"{method}: provenance must be nonempty")
            if not isinstance(row["paths"], list):
                raise ValueError(f"{method}: paths must be a list, including empty failures")
            if row["decoder_status"] == "zero_path" and row["paths"]:
                raise ValueError(f"{method}: zero_path cannot contain paths")
            if (row["decoder_status"] != "ok" or row["graph_status"] == "failed"
                or row["basal_node_status"] != "ok") and row["paths"]:
                raise ValueError(f"{method}: failed graph/base/decoder cannot contain paths")
            if row["decoder_status"] == "ok" and not row["paths"]:
                raise ValueError(f"{method}: empty paths must retain zero_path/failed status")
            path_ids = set()
            for path in row["paths"]:
                if set(path) != {"path_id", "full_base_to_tip_path", "provenance"}:
                    raise ValueError(f"{method}: path fields changed")
                if not isinstance(path["path_id"], str) or not path["path_id"] or path["path_id"] in path_ids:
                    raise ValueError(f"{method}: duplicate/invalid path ID")
                path_ids.add(path["path_id"])
                if (not isinstance(path["full_base_to_tip_path"], list)
                    or len(path["full_base_to_tip_path"]) < 2
                    or not all(_point(p) for p in path["full_base_to_tip_path"])):
                    raise ValueError(f"{method}: invalid full base-to-tip path")
                if not isinstance(path["provenance"], dict) or not path["provenance"]:
                    raise ValueError(f"{method}: path provenance must be nonempty")


def validate_asset_dimensions(bundle: dict, gt: dict, asset_manifest: dict) -> None:
    """Asset dimensions are administrative metadata, not a second GT source."""
    manifest = asset_manifest.get("samples")
    if not isinstance(manifest, list) or len(manifest) != 40:
        raise ValueError("Locked image asset manifest changed")
    by_blind = {sample["blind_id"]: sample for sample in manifest}
    if len(by_blind) != 40:
        raise ValueError("Duplicate asset blind ID")
    plants = gt["plants"]
    if {p["blind_id"] for p in plants} != set(by_blind):
        raise ValueError("Asset manifest and frozen GT membership differ")
    by_id = {p["dataset_id"]: by_blind[p["blind_id"]] for p in plants}
    for method in METHODS:
        for row in bundle["methods"][method]:
            expected = by_id[row["dataset_id"]]
            if ((row["width"], row["height"]) != (expected["width"], expected["height"])
                or row["image_sha256"] != expected["image_sha256"]):
                raise ValueError(f"{method}: locked image dimensions/hash mismatch")


def _gt_traces(plant: dict) -> list[dict]:
    diag = float(plant["bbox_diagonal_px"])
    traces = []
    for path in plant["paths"]:
        curve = np.asarray(path["points_px"], dtype=np.float64)
        if curve.shape != (240, 2) or not np.isfinite(curve).all():
            raise ValueError("Frozen GT curve invalid")
        if not math.isclose(polyline_length(curve), float(path["length_px"]), abs_tol=1e-6):
            raise ValueError("Frozen GT length changed")
        traces.append({
            "trace_uuid": path["trace_uuid"], "visibility_status": "measurable",
            "tip_xy": curve[-1].tolist(), "tip_clockwise_angle_deg": path["tip_clockwise_angle_deg"],
            "resampled_curve": curve, "structural_path_length_px": path["length_px"],
            "chord_length_px": path["chord_length_px"],
            "reference_main_path": path["reference_main_path"],
            "divergence_angle_deg": path["divergence_angle_deg"],
            "divergence_angle_status": path["divergence_angle_status"],
        })
    if traces:
        if select_reference_main(traces, diag) != plant["reference_main_trace_uuid"]:
            raise ValueError("Frozen GT main identity changed")
        main = next(t for t in traces if t["reference_main_path"])
        for trace in traces:
            if not trace["reference_main_path"]:
                calculated = divergence_angle(main["resampled_curve"], trace["resampled_curve"], diag)
                if calculated["status"] != trace["divergence_angle_status"]:
                    raise ValueError("Frozen GT divergence status changed")
                actual, expected = calculated.get("divergence_angle_deg"), trace["divergence_angle_deg"]
                if (actual is None) != (expected is None) or (actual is not None and abs(actual - expected) > 1e-6):
                    raise ValueError("Frozen GT divergence angle changed")
    return traces


def _prediction_traces(method: str, row: dict, diag: float) -> list[dict]:
    result = []
    for path in row["paths"]:
        source = _source_curve(path["full_base_to_tip_path"], row["width"], row["height"])
        metric = trace_metrics(source, diag)
        curve = metric["resampled_curve"]
        result.append({
            "trace_uuid": f"{method}:{row['dataset_id']}:{path['path_id']}",
            "path_id": path["path_id"], "visibility_status": "measurable",
            "tip_xy": curve[-1].tolist(), "tip_clockwise_angle_deg": metric["tip_clockwise_angle_deg"],
            "resampled_curve": curve, "structural_path_length_px": metric["structural_path_length_px"],
            "chord_length_px": metric["chord_length_px"], "provenance": path["provenance"],
            "reference_main_path": False, "divergence_angle_status": "main_path",
            "divergence_angle_deg": None,
        })
    if result:
        main_uuid = select_reference_main(result, diag)
        main_curve = next(t["resampled_curve"] for t in result if t["trace_uuid"] == main_uuid)
        for path in result:
            path["reference_main_path"] = path["trace_uuid"] == main_uuid
            if not path["reference_main_path"]:
                angle = divergence_angle(main_curve, path["resampled_curve"], diag)
                path["divergence_angle_status"] = angle["status"]
                path["divergence_angle_deg"] = angle.get("divergence_angle_deg")
    return result


def _base_status(plant: dict, row: dict) -> tuple[str, float | None]:
    if row["basal_node_status"] == "no_eligible_basal_node":
        return "no_eligible_basal_node", None
    if row["basal_node_status"] == "graph_failure":
        return "graph_failure", None
    if row["base_xy_model_canvas"] is None:
        return "missing_output_base", None
    if plant["base_xy"] is None:
        return "human_base_unavailable", None
    source = _source_curve([row["base_xy_model_canvas"]], row["width"], row["height"])[0]
    distance = float(np.linalg.norm(source - np.asarray(plant["base_xy"], dtype=np.float64)))
    return ("base_beyond_0.025D" if distance > 0.025 * float(plant["bbox_diagonal_px"]) else "ok"), distance


def evaluate_plant(method: str, plant: dict, row: dict) -> tuple[dict, list[dict], dict]:
    """Return one unfiltered plant summary, all path outcomes, and provenance."""
    diag = float(plant["bbox_diagonal_px"])
    gt, pred = _gt_traces(plant), _prediction_traces(method, row, diag)
    matched = match_cross_session(gt, pred, diag)
    gmap, pmap = {x["trace_uuid"]: x for x in gt}, {x["trace_uuid"]: x for x in pred}
    path_rows, pair_by_gt = [], {}
    for match in matched:
        g = gmap.get(match.trace_uuid_session_1)
        p = pmap.get(match.trace_uuid_session_2)
        result = {
            "dataset_id": plant["dataset_id"], "method": method, "status": match.status,
            "gt_trace_uuid": match.trace_uuid_session_1, "pred_trace_uuid": match.trace_uuid_session_2,
            "pred_path_id": None if p is None else p["path_id"],
            "matching_cost": match.cost, "tip_norm": match.tip_norm,
            "curve_norm": match.curve_norm, "polar_gap": match.polar_gap,
            "gt_reference_main": None if g is None else g["reference_main_path"],
            "pred_reference_main": None if p is None else p["reference_main_path"],
            "gt_divergence_angle_status": None if g is None else g["divergence_angle_status"],
            "pred_divergence_angle_status": None if p is None else p["divergence_angle_status"],
            "length_error_px": None, "length_error_bbox_norm": None,
            "length_error_mm_derived_from_scanner_metadata": None,
            "length_symmetric_relative_error_pct": None, "angle_absolute_error_deg": None,
            "length_within_dev_mdc95_reference": None, "angle_within_dev_mdc95_reference": None,
            "pred_path_provenance": None if p is None else p["provenance"],
        }
        if match.status == "matched":
            lp, lg = float(p["structural_path_length_px"]), float(g["structural_path_length_px"])
            err = abs(lp - lg)
            length_pct = 100.0 * err / max((lp + lg) / 2.0, EPS)
            result.update({
                "length_error_px": err,
                "length_error_bbox_norm": err / diag,
                "length_error_mm_derived_from_scanner_metadata": err * float(plant["mm_per_output_px_scanner_metadata_derived"]),
                "length_symmetric_relative_error_pct": length_pct,
                "length_within_dev_mdc95_reference": length_pct <= LENGTH_MDC95_PCT,
            })
            if (not g["reference_main_path"] and not p["reference_main_path"]
                and g["divergence_angle_deg"] is not None and p["divergence_angle_deg"] is not None):
                angle = abs(float(p["divergence_angle_deg"]) - float(g["divergence_angle_deg"]))
                result["angle_absolute_error_deg"] = angle
                result["angle_within_dev_mdc95_reference"] = angle <= ANGLE_MDC95_DEG
            pair_by_gt[g["trace_uuid"]] = result
        path_rows.append(result)
    # Existence-only and unresolved structures remain explicit, never paired to curves.
    for item in plant["structures"]:
        if item["visibility_status"] != "measurable":
            path_rows.append({
                "dataset_id": plant["dataset_id"], "method": method,
                "status": f"gt_{item['visibility_status']}",
                "gt_trace_uuid": item["original_trace_uuid"], "pred_trace_uuid": None,
                "pred_path_id": None, "matching_cost": None, "tip_norm": None,
                "curve_norm": None, "polar_gap": None, "gt_reference_main": False,
                "pred_reference_main": None, "gt_divergence_angle_status": "geometry_unavailable",
                "pred_divergence_angle_status": None, "length_error_px": None,
                "length_error_bbox_norm": None, "length_error_mm_derived_from_scanner_metadata": None,
                "length_symmetric_relative_error_pct": None, "angle_absolute_error_deg": None,
                "length_within_dev_mdc95_reference": None, "angle_within_dev_mdc95_reference": None,
                "pred_path_provenance": None,
            })
    G = len(gt)
    V = sum(x["visibility_status"] == "visible_unmeasurable" for x in plant["structures"])
    Q = sum(x["visibility_status"] == "uncertain" for x in plant["structures"])
    E, P = G + V, len(pred)
    M = len(pair_by_gt)
    U = P - M
    if M > G or M > P or U < 0:
        raise ValueError("Invalid one-to-one matching")
    base_status, base_distance = _base_status(plant, row)
    base_ok = base_status == "ok"
    complete_lower = int(E > 0 and Q == V == 0 and M == G == E == P and base_ok)
    complete_upper = int(E > 0 and M == G and E <= P <= E + Q and base_ok)
    length_errors = [p["length_symmetric_relative_error_pct"] for p in pair_by_gt.values()]
    angle_errors = [p["angle_absolute_error_deg"] for p in pair_by_gt.values() if p["angle_absolute_error_deg"] is not None]
    unresolved_gt = sum(
        p["gt_divergence_angle_status"] not in {"ok", "main_path"}
        for p in pair_by_gt.values()
    )
    unresolved_pred = sum(
        p["pred_divergence_angle_status"] not in {"ok", "main_path"}
        for p in pair_by_gt.values()
    )
    metric = {
        "dataset_id": plant["dataset_id"], "source_frame_id": plant["source_frame_id"],
        "method": method, "gt_measurable_count": G, "gt_existence_only_count": V,
        "gt_uncertain_count": Q,
        "gt_non_target_count": sum(x["visibility_status"] == "non_target_structure" for x in plant["structures"]),
        "gt_confirmed_existence_count": E, "prediction_count": P, "matched_count": M,
        "missed_measurable_count": G - M,
        "existence_missed_interval": [E - M - min(U, V), E - M],
        "unmatched_prediction_count": U,
        "possible_extra_interval": [max(0, U - V - Q), U],
        "measurable_recall": M / G if G else None,
        "existence_recall_interval": [M / E, (M + min(U, V)) / E] if E else [None, None],
        "precision_interval": [M / P, (M + min(U, V + Q)) / P] if P else [None, None],
        "path_count_absolute_error": abs(P - E), "path_count_signed_error": P - E,
        "zero_prediction": P == 0, "zero_path_collapse": P == 0 or row["decoder_status"] == "zero_path",
        "base_failure_status": base_status, "base_distance_px": base_distance,
        "base_distance_bbox_norm": None if base_distance is None else base_distance / diag,
        "complete_plant_recovery_interval": [complete_lower, complete_upper],
        "complete_plant_indeterminate": bool(V or Q or E == 0),
        "matched_length_n": len(length_errors),
        "length_error_pct_conditional_mean": float(np.mean(length_errors)) if length_errors else None,
        "length_error_pct_conditional_median": float(np.median(length_errors)) if length_errors else None,
        "matched_angle_n": len(angle_errors),
        "angle_error_deg_conditional_mean": float(np.mean(angle_errors)) if angle_errors else None,
        "matched_gt_angle_unresolved_count": unresolved_gt,
        "matched_pred_angle_unresolved_count": unresolved_pred,
        "angle_identity_conflict_count": sum(
            p["status"] == "matched" and p["gt_reference_main"] != p["pred_reference_main"] for p in path_rows
        ),
        "point_count": row["point_count"], "accepted_node_count": row["accepted_node_count"],
        "rejected_node_count": row["rejected_node_count"], "graph_node_count": row["graph_node_count"],
        "underground_association_count": row["underground_association_count"],
        "graph_status": row["graph_status"], "decoder_status": row["decoder_status"],
        "pairs_by_gt": pair_by_gt,
    }
    provenance = {
        "dataset_id": plant["dataset_id"], "method": method,
        "image_sha256": row["image_sha256"], "source_frame_id": row["source_frame_id"],
        "point_count": row["point_count"], "accepted_node_count": row["accepted_node_count"],
        "rejected_node_count": row["rejected_node_count"], "graph_node_count": row["graph_node_count"],
        "underground_association_count": row["underground_association_count"],
        "graph_status": row["graph_status"], "decoder_status": row["decoder_status"],
        "basal_node_status": row["basal_node_status"], "base_failure_status": base_status,
        "prediction_count": P, "prediction_provenance": row["provenance"],
    }
    return metric, path_rows, provenance


def _paired_rows(all_rows: dict[str, list[dict]]) -> list[dict]:
    by_method = {method: {r["dataset_id"]: r for r in all_rows[method]} for method in METHODS}
    ids = sorted(by_method[METHODS[0]])
    result = []
    for target, baseline, priority in CONTRASTS:
        for ident in ids:
            left, right = by_method[target][ident], by_method[baseline][ident]
            common = sorted(set(left["pairs_by_gt"]) & set(right["pairs_by_gt"]))
            lengths = [
                left["pairs_by_gt"][key]["length_symmetric_relative_error_pct"]
                - right["pairs_by_gt"][key]["length_symmetric_relative_error_pct"]
                for key in common
            ]
            angles = [
                left["pairs_by_gt"][key]["angle_absolute_error_deg"]
                - right["pairs_by_gt"][key]["angle_absolute_error_deg"]
                for key in common
                if left["pairs_by_gt"][key]["angle_absolute_error_deg"] is not None
                and right["pairs_by_gt"][key]["angle_absolute_error_deg"] is not None
            ]
            def difference(field: str) -> float | None:
                a, b = left[field], right[field]
                return None if a is None or b is None else a - b
            result.append({
                "dataset_id": ident, "source_frame_id": left["source_frame_id"],
                "contrast": f"{target} - {baseline}", "priority": priority,
                "shared_gt_trace_uuids": common, "shared_length_path_n": len(lengths),
                "shared_angle_path_n": len(angles),
                "conditional_length_error_diff_pct": float(np.mean(lengths)) if lengths else None,
                "conditional_angle_error_diff_deg": float(np.mean(angles)) if angles else None,
                "measurable_recall_diff": difference("measurable_recall"),
                "existence_recall_lower_diff": left["existence_recall_interval"][0] - right["existence_recall_interval"][0]
                    if None not in (left["existence_recall_interval"][0], right["existence_recall_interval"][0]) else None,
                "existence_recall_upper_diff": left["existence_recall_interval"][1] - right["existence_recall_interval"][1]
                    if None not in (left["existence_recall_interval"][1], right["existence_recall_interval"][1]) else None,
                "precision_lower_diff": left["precision_interval"][0] - right["precision_interval"][0]
                    if None not in (left["precision_interval"][0], right["precision_interval"][0]) else None,
                "precision_upper_diff": left["precision_interval"][1] - right["precision_interval"][1]
                    if None not in (left["precision_interval"][1], right["precision_interval"][1]) else None,
                "possible_extra_lower_diff": left["possible_extra_interval"][0] - right["possible_extra_interval"][0],
                "possible_extra_upper_diff": left["possible_extra_interval"][1] - right["possible_extra_interval"][1],
                "complete_lower_diff": left["complete_plant_recovery_interval"][0] - right["complete_plant_recovery_interval"][0],
                "complete_upper_diff": left["complete_plant_recovery_interval"][1] - right["complete_plant_recovery_interval"][1],
                "prediction_count_diff": difference("prediction_count"),
                "matched_count_diff": difference("matched_count"),
            })
    return result


METHOD_METRICS = {
    "measurable_recall": lambda r: r["measurable_recall"],
    "existence_recall_lower": lambda r: r["existence_recall_interval"][0],
    "existence_recall_upper": lambda r: r["existence_recall_interval"][1],
    "precision_lower": lambda r: r["precision_interval"][0],
    "precision_upper": lambda r: r["precision_interval"][1],
    "possible_extra_lower": lambda r: r["possible_extra_interval"][0],
    "possible_extra_upper": lambda r: r["possible_extra_interval"][1],
    "complete_lower": lambda r: r["complete_plant_recovery_interval"][0],
    "complete_upper": lambda r: r["complete_plant_recovery_interval"][1],
    "conditional_length_error_pct": lambda r: r["length_error_pct_conditional_mean"],
    "conditional_angle_error_deg": lambda r: r["angle_error_deg_conditional_mean"],
    "path_count_absolute_error": lambda r: r["path_count_absolute_error"],
    "path_count_signed_error": lambda r: r["path_count_signed_error"],
    "matched_count": lambda r: r["matched_count"],
    "missed_measurable_count": lambda r: r["missed_measurable_count"],
    "prediction_count": lambda r: r["prediction_count"],
    "base_failure": lambda r: int(r["base_failure_status"] != "ok"),
    "zero_path": lambda r: int(r["zero_path_collapse"]),
    "matched_gt_angle_unresolved_count": lambda r: r["matched_gt_angle_unresolved_count"],
    "matched_pred_angle_unresolved_count": lambda r: r["matched_pred_angle_unresolved_count"],
    "angle_identity_conflict_count": lambda r: r["angle_identity_conflict_count"],
}
PAIRED_METRICS = (
    "measurable_recall_diff", "existence_recall_lower_diff", "existence_recall_upper_diff",
    "precision_lower_diff", "precision_upper_diff", "possible_extra_lower_diff",
    "possible_extra_upper_diff", "complete_lower_diff", "complete_upper_diff",
    "prediction_count_diff", "matched_count_diff",
    "conditional_length_error_diff_pct", "conditional_angle_error_diff_deg",
)


def _draws(ids: list[str], frames: list[str], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    plant_draws = np.random.Generator(np.random.PCG64(PLANT_SEED)).integers(0, len(ids), size=(n, len(ids)))
    frame_names = sorted(set(frames))
    frame_draws = np.random.Generator(np.random.PCG64(FRAME_SEED)).integers(
        0, len(frame_names), size=(n, len(frame_names))
    )
    frame_of_plant = np.asarray([frame_names.index(frame) for frame in frames], dtype=np.int64)
    return plant_draws, frame_draws, frame_of_plant, frame_names


def _ci(samples: np.ndarray) -> dict:
    values = samples[np.isfinite(samples)]
    return {
        "valid_replicates": int(len(values)),
        "total_replicates": int(len(samples)),
        "ci95": [float(np.quantile(values, 0.025, method="linear")),
                 float(np.quantile(values, 0.975, method="linear"))] if len(values) else [None, None],
    }


def _ratio_draws(numer: np.ndarray, denom: np.ndarray, draws: np.ndarray) -> np.ndarray:
    top = numer[draws].sum(axis=1)
    bottom = denom[draws].sum(axis=1)
    return np.divide(top, bottom, out=np.full(len(top), np.nan), where=bottom > 0)


def _frame_ratio_draws(numer: np.ndarray, denom: np.ndarray, frame_draws: np.ndarray, frame_of_plant: np.ndarray) -> np.ndarray:
    count = int(frame_of_plant.max()) + 1
    frame_top = np.bincount(frame_of_plant, weights=numer, minlength=count)
    frame_bottom = np.bincount(frame_of_plant, weights=denom, minlength=count)
    top = frame_top[frame_draws].sum(axis=1)
    bottom = frame_bottom[frame_draws].sum(axis=1)
    return np.divide(top, bottom, out=np.full(len(top), np.nan), where=bottom > 0)


def _summarize(values: list[float | None], plant_draws: np.ndarray, frame_draws: np.ndarray, frame_of_plant: np.ndarray) -> dict:
    array = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    finite = np.isfinite(array)
    usable = array[finite]
    numer = np.where(finite, array, 0.0)
    denom = finite.astype(np.float64)
    return {
        "n_valid_plants": int(finite.sum()),
        "mean": float(usable.mean()) if len(usable) else None,
        "median": float(np.median(usable)) if len(usable) else None,
        "q1": float(np.quantile(usable, 0.25)) if len(usable) else None,
        "q3": float(np.quantile(usable, 0.75)) if len(usable) else None,
        "plant_cluster_bootstrap": _ci(_ratio_draws(numer, denom, plant_draws)),
        "source_frame_cluster_sensitivity": _ci(_frame_ratio_draws(numer, denom, frame_draws, frame_of_plant)),
    }


def _pooled_ratio(
    numer: list[float], denom: list[float], plant_draws: np.ndarray,
    frame_draws: np.ndarray, frame_of_plant: np.ndarray,
) -> dict:
    a, b = np.asarray(numer, dtype=np.float64), np.asarray(denom, dtype=np.float64)
    return {
        "numerator": float(a.sum()), "denominator": float(b.sum()),
        "point_estimate": float(a.sum() / b.sum()) if b.sum() > 0 else None,
        "plant_cluster_bootstrap": _ci(_ratio_draws(a, b, plant_draws)),
        "source_frame_cluster_sensitivity": _ci(_frame_ratio_draws(a, b, frame_draws, frame_of_plant)),
    }


def _statistics(all_rows: dict[str, list[dict]], paired: list[dict], n_bootstrap: int) -> dict:
    ids = [r["dataset_id"] for r in all_rows[METHODS[0]]]
    frames = [r["source_frame_id"] for r in all_rows[METHODS[0]]]
    if any([r["dataset_id"] for r in all_rows[m]] != ids for m in METHODS):
        raise ValueError("Methods not aligned on the same plant order")
    plant_draws, frame_draws, frame_of_plant, frame_names = _draws(ids, frames, n_bootstrap)
    out = {
        "bootstrap": {
            "plant_cluster_replicates": n_bootstrap, "plant_seed": PLANT_SEED,
            "source_frame_cluster_replicates": n_bootstrap, "source_frame_seed": FRAME_SEED,
            "plant_count": len(ids), "source_frame_count": len(frame_names),
            "method_draws_shared": True, "quantile_method": "linear",
        },
        "methods": {},
        "paired_contrasts": {},
    }
    for method in METHODS:
        rows = all_rows[method]
        macro = {
            name: _summarize([getter(r) for r in rows], plant_draws, frame_draws, frame_of_plant)
            for name, getter in METHOD_METRICS.items()
        }
        pooled = {
            "measurable_recall": _pooled_ratio(
                [r["matched_count"] for r in rows], [r["gt_measurable_count"] for r in rows],
                plant_draws, frame_draws, frame_of_plant),
            "existence_recall_lower": _pooled_ratio(
                [r["matched_count"] for r in rows], [r["gt_confirmed_existence_count"] for r in rows],
                plant_draws, frame_draws, frame_of_plant),
            "existence_recall_upper": _pooled_ratio(
                [r["matched_count"] + min(r["unmatched_prediction_count"], r["gt_existence_only_count"]) for r in rows],
                [r["gt_confirmed_existence_count"] for r in rows], plant_draws, frame_draws, frame_of_plant),
            "precision_lower": _pooled_ratio(
                [r["matched_count"] for r in rows], [r["prediction_count"] for r in rows],
                plant_draws, frame_draws, frame_of_plant),
            "precision_upper": _pooled_ratio(
                [r["matched_count"] + min(r["unmatched_prediction_count"], r["gt_existence_only_count"] + r["gt_uncertain_count"]) for r in rows],
                [r["prediction_count"] for r in rows], plant_draws, frame_draws, frame_of_plant),
        }
        length_sum = [
            sum(p["length_symmetric_relative_error_pct"] for p in r["pairs_by_gt"].values())
            for r in rows
        ]
        pooled["matched_path_length_error_pct"] = _pooled_ratio(
            length_sum, [r["matched_length_n"] for r in rows],
            plant_draws, frame_draws, frame_of_plant,
        )
        angle_sum = [
            sum(p["angle_absolute_error_deg"] for p in r["pairs_by_gt"].values() if p["angle_absolute_error_deg"] is not None)
            for r in rows
        ]
        pooled["matched_path_angle_error_deg"] = _pooled_ratio(
            angle_sum, [r["matched_angle_n"] for r in rows],
            plant_draws, frame_draws, frame_of_plant,
        )
        out["methods"][method] = {"plant_macro": macro, "path_pooled_descriptive": pooled}
    for target, baseline, priority in CONTRASTS:
        name = f"{target} - {baseline}"
        rows = [r for r in paired if r["contrast"] == name]
        if [r["dataset_id"] for r in rows] != ids:
            raise ValueError("Paired comparison omitted a plant")
        out["paired_contrasts"][name] = {
            "priority": priority,
            "shared_length_path_n": sum(r["shared_length_path_n"] for r in rows),
            "shared_angle_path_n": sum(r["shared_angle_path_n"] for r in rows),
            "metrics": {
                field: _summarize([r[field] for r in rows], plant_draws, frame_draws, frame_of_plant)
                for field in PAIRED_METRICS
            },
        }
    return out


def _preregistered_figure_rank(all_rows: dict[str, list[dict]]) -> dict:
    """Frozen section-9 Teacher/B ranking; no subjective image replacement."""
    teacher = {row["dataset_id"]: row for row in all_rows["Teacher-direct"]}
    student = {row["dataset_id"]: row for row in all_rows["Student-B"]}
    ranked = []
    for ident in sorted(teacher):
        left, right = teacher[ident], student[ident]
        def recall(row: dict) -> float:
            value = row["measurable_recall"]
            if row["gt_measurable_count"] == 0:
                value = row["existence_recall_interval"][0]
            return -1.0 if value is None else float(value)
        worst_recall = min(recall(left), recall(right))
        extra_lower_sum = left["possible_extra_interval"][0] + right["possible_extra_interval"][0]
        lengths = (left["length_error_pct_conditional_mean"], right["length_error_pct_conditional_mean"])
        length_term = math.inf if any(value is None for value in lengths) else -max(lengths)
        tie_hash = hashlib.sha256(f"V4-test-figure-20260927|{ident}".encode("utf-8")).hexdigest()
        key = (worst_recall, -extra_lower_sum, length_term, tie_hash)
        ranked.append((key, {
            "dataset_id": ident,
            "score": [worst_recall, -extra_lower_sum, None if math.isinf(length_term) else length_term],
            "missing_matched_length_for_ranking": math.isinf(length_term),
            "tie_sha256": tie_hash,
        }))
    ranked.sort(key=lambda item: item[0])
    rows = []
    for rank, (_, row) in enumerate(ranked, start=1):
        rows.append({"rank": rank, **row})
    return {
        "all_40_ranked": rows,
        "preselected_figures": {
            "worst_rank_1": rows[0]["dataset_id"],
            "typical_rank_20": rows[19]["dataset_id"],
            "relatively_high_rank_40": rows[39]["dataset_id"],
        },
        "missing_length_sort_value": "+infinity; JSON null in display score",
    }


def evaluate_synthetic_or_saved(gt: dict, predictions: dict, *, n_bootstrap: int = BOOTSTRAP_N) -> dict:
    """Pure function for synthetic fixtures; formal entry adds pinned file checks."""
    validate_prediction_bundle(predictions, gt)
    plants = sorted(gt["plants"], key=lambda p: p["dataset_id"])
    by_method = {method: {r["dataset_id"]: r for r in predictions["methods"][method]} for method in METHODS}
    all_rows, path_rows, provenance = {}, [], []
    for method in METHODS:
        all_rows[method] = []
        for plant in plants:
            row, paths, source = evaluate_plant(method, plant, by_method[method][plant["dataset_id"]])
            all_rows[method].append(row)
            path_rows.extend(paths)
            provenance.append(source)
    paired = _paired_rows(all_rows)
    stats = _statistics(all_rows, paired, n_bootstrap)
    clean_rows = [
        {field: value for field, value in row.items() if field != "pairs_by_gt"}
        for method in METHODS for row in all_rows[method]
    ]
    return {
        "schema_version": "v4-locked-test-evaluation-v1",
        "scope": "offline frozen saved predictions; no inference in evaluator",
        "gt_protocol": GEOMETRY_PROTOCOL_VERSION,
        "method_order": list(METHODS),
        "primary_contrast": "Student-B - Teacher-direct",
        "secondary_contrasts": ["Student-D - Teacher-direct", "Student-D - Student-B"],
        "plant_level": clean_rows,
        "path_level": path_rows,
        "provenance": provenance,
        "paired_plant_level": paired,
        "statistics": stats,
        "preregistered_figure_rank": _preregistered_figure_rank(all_rows),
        "human_mdc95_practical_reference_only": {
            "length_pct": LENGTH_MDC95_PCT, "angle_deg": ANGLE_MDC95_DEG,
            "equivalence_margin": False,
        },
        "millimeter_status": "derived from scanner metadata; not ruler-calibrated",
    }


def evaluate_files(gt_file: Path, predictions_file: Path, output_dir: Path) -> dict:
    """Formal *post-authorization* offline entry; never called in freeze tests."""
    private_root = (PROTOCOL / "runtime").resolve()
    if not output_dir.resolve().is_relative_to(private_root):
        raise ValueError("Formal evaluation output must stay in Git-ignored private runtime")
    if output_dir.exists():
        raise FileExistsError("Refusing to reread saved test outputs into an existing run directory")
    verify_sources()
    if sha(gt_file) != GT_SHA256:
        raise ValueError("Final frozen GT SHA-256 mismatch")
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    if len(gt["plants"]) != 40 or len({p["source_frame_id"] for p in gt["plants"]}) != 8:
        raise ValueError("Locked test plant/frame membership changed")
    saved = json.loads(predictions_file.read_text(encoding="utf-8"))
    asset_manifest_file = PROTOCOL / "runtime" / "v4_test_single_rater_20260927" / "public" / "measurement_manifest.json"
    if sha(asset_manifest_file) != ASSET_MANIFEST_SHA256:
        raise ValueError("Locked image asset manifest SHA-256 mismatch")
    validate_asset_dimensions(saved, gt, json.loads(asset_manifest_file.read_text(encoding="utf-8")))
    result = evaluate_synthetic_or_saved(gt, saved, n_bootstrap=BOOTSTRAP_N)
    if len(result["plant_level"]) != 40 * len(METHODS):
        raise ValueError("Formal 40 x 3 matrix incomplete")
    # Prepare every byte before creating the output directory: no partial
    # result is presented as completed if schema or statistics fail.
    payloads = {
        "plant_level.json": canonical_json(result["plant_level"]),
        "path_level.json": canonical_json(result["path_level"]),
        "provenance.json": canonical_json(result["provenance"]),
        "paired_plant_level.json": canonical_json(result["paired_plant_level"]),
        "statistics.json": canonical_json(result["statistics"]),
        "preregistered_figure_rank.json": canonical_json(result["preregistered_figure_rank"]),
    }
    ledger = {
        "gt_sha256": GT_SHA256,
        "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
        "saved_predictions_sha256": sha(predictions_file),
        "evaluator_source_sha256": sha(Path(__file__)),
        "geometry_source_sha256": GEOMETRY_SHA256,
        "inverse_letterbox_source_sha256": PILOT_EVALUATOR_SHA256,
        "pipeline_v1_sha256": pipeline_v1_hash(),
        "method_weights_sha256": METHOD_WEIGHTS_SHA256,
        "preregistration_sha256": PREREG_SHA256,
        "output_files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["evaluation_ledger.json"] = canonical_json(ledger)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        (output_dir / name).write_bytes(data)
    return {"output_dir": str(output_dir), "ledger": ledger,
            "plants": len(gt["plants"]), "method_plant_rows": len(result["plant_level"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--saved-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_files(args.gt, args.saved_predictions, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

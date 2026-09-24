"""Freeze a method-blind, provenance-checked phenotype pilot reference.

This script reads only human records, locked image metadata, and the previously
audited automatic-equivalence plan. It never reads model predictions or V4 test.
The private output is intentionally kept under the git-ignored runtime tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from build import digest, read_json
from build_joint_consensus import load_cases
from build_minimal_gt_resolution import AUTO_PAGES, _lookup, _points_sha256, validate

PROTOCOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOCOL))
from phenotype_gt_geometry import (  # noqa: E402
    clockwise_angle_deg,
    divergence_angle,
    polyline_length,
    select_reference_main,
)

VERSION = "phenotype-pilot-final-gt-v1"
AUTO_PLAN_SHA256 = "d52b21be3eb6051379daea4114cf30190acc291d2d4657d62d911ff829f1528c"
PROPOSAL_SHA256 = "25c866356ca34e2dc1d4d9910119a99a19ca0d60c6ba0215c914492507f60749"
ISSUE_KEYS = {5: ("C:T01", "C:T02", "C:T03"), 6: ("B:T01",), 7: ("C:T01", "C:T02")}
ISSUE_COUNTS = {5: 3, 6: 1, 7: 3}
EXPECTED_DECISIONS = {
    5: "cannot_determine",
    6: "one_leaf_questioned_is_artifact",
    7: "two_leaves_exclude_questioned",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _admin_rows(runtime: Path) -> list[dict[str, str]]:
    with (runtime / "admin/blind_id_admin_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _metadata() -> dict[str, dict]:
    rows = read_json(PROTOCOL / "phenotype_pilot_selection.json")
    assert len(rows) == 16 and all(row["split"] == "val" for row in rows)
    return {row["dataset_id"]: row for row in rows}


def _path(source_key: str, points: list[list[float]], bbox_diag: float, mm_per_px: float) -> dict:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (240, 2) or not np.isfinite(array).all():
        raise ValueError(f"Invalid 240-point trace: {source_key}")
    length = polyline_length(array)
    chord = float(np.linalg.norm(array[-1] - array[0]))
    return {
        "source_key": source_key,
        "source_points_sha256": _points_sha256(points),
        "status": "measurable",
        "points_px": points,
        "length_px": length,
        "length_bbox_normalized": length / bbox_diag,
        "length_mm_scanner_metadata_derived": length * mm_per_px,
        "chord_px": chord,
        "tip_clockwise_angle_deg": clockwise_angle_deg(array[0], array[-1]),
        "base_px": points[0],
        "tip_px": points[-1],
    }


def _finish_plant(dataset_id: str, paths: list[dict], count: int, metadata: dict, **extra: object) -> dict:
    row = metadata[dataset_id]
    bbox_diag = math.hypot(row["shoot_bbox_width_px"], row["shoot_bbox_height_px"])
    if len(paths) > count or bbox_diag <= 0:
        raise ValueError(f"Invalid path count/scale for {dataset_id}")
    if len({path["source_points_sha256"] for path in paths}) != len(paths):
        raise ValueError(f"Duplicate paths for {dataset_id}")
    for path in paths:
        points = np.asarray(path["points_px"])
        if (points[:, 0] < 0).any() or (points[:, 0] >= row["standardized_width_px"]).any() or (points[:, 1] < 0).any() or (points[:, 1] >= row["standardized_height_px"]).any():
            raise ValueError(f"Path is outside the locked standardized image: {dataset_id}")
    common = [{**path, "trace_uuid": path["source_points_sha256"], "visibility_status": "measurable", "structural_path_length_px": path["length_px"], "chord_length_px": path["chord_px"]} for path in paths]
    main = select_reference_main(common, bbox_diag)
    main_path = next(path for path in paths if path["source_points_sha256"] == main)
    for path in paths:
        path["reference_main_path"] = path["source_points_sha256"] == main
        if not path["reference_main_path"]:
            result = divergence_angle(np.asarray(main_path["points_px"]), np.asarray(path["points_px"]), bbox_diag)
            path["divergence_angle_status"] = result["status"]
            path["divergence_angle_deg"] = result.get("divergence_angle_deg")
    return {
        "dataset_id": dataset_id,
        "split": "val",
        "pilot_group": row["pilot_group"],
        "operational_independent_path_count": count,
        "measurable_reference_path_count": len(paths),
        "bbox_diagonal_px": bbox_diag,
        "mm_per_output_px_scanner_metadata_derived": row["mm_per_output_px"],
        "paths": paths,
        **extra,
    }


def build(runtime: Path, output: Path, proposal: Path) -> dict:
    if not proposal.is_file() or digest(proposal) != PROPOSAL_SHA256:
        raise ValueError("User-approved operational definition source is missing or has changed")
    source = runtime / "adjudication/semantic_first_20260906"
    minimal = source / "minimal_gt_resolution_20260915_v1"
    export = minimal / "results/minimal_gt_resolution.json"
    validation = validate(minimal, export)
    if validation["submitted_records"] != 3:
        raise ValueError("Three original decisions are required")
    original = read_json(export)
    decisions = {record["original_page"]: record for record in original["records"]}
    if {page: record["decision"] for page, record in decisions.items()} != EXPECTED_DECISIONS:
        raise ValueError("The original decisions changed; refuse to freeze")
    geometry = source / "geometry/results/geometry_decisions.json"
    cases, source_provenance = load_cases(runtime, geometry)
    case_by_page = {case["original_page"]: case for case in cases}
    if len(case_by_page) != 13:
        raise ValueError("Expected 13 adjudicated plants")
    plan_file = runtime / "admin/semantic_first_20260906/minimal_gt_resolution" / f"auto_resolution_plan_{AUTO_PLAN_SHA256}.json"
    if digest(plan_file) != AUTO_PLAN_SHA256:
        raise ValueError("The locked automatic-equivalence plan changed")
    plan = read_json(plan_file)
    planned = {record["original_page"]: record for record in plan["records"]}
    if set(planned) != set(AUTO_PAGES):
        raise ValueError("Unexpected automatic-equivalence pages")
    mapping = read_json(runtime / "admin/semantic_first_20260906/mapping.json")
    pair_by_blind = {item["blind_id"]: item["pair_id"] for item in mapping["items"]}
    admin = _admin_rows(runtime)
    r1_rows = {row["adjudication_pair_id"]: row for row in admin if row["rater_id"] == "R1" and row["measurement_round"] == "1" and row["pilot_group"] in {"core", "diagnostic"}}
    metadata = _metadata()
    plants = []
    for page in range(1, 14):
        case = case_by_page[page]
        dataset_id = r1_rows[pair_by_blind[case["blind_id"]]]["dataset_id"]
        row = metadata[dataset_id]
        diag = math.hypot(row["shoot_bbox_width_px"], row["shoot_bbox_height_px"])
        lookup = _lookup(case)
        if page in planned:
            selected = planned[page]["canonical_paths"]
            keys = [entry["source_key"] for entry in selected]
            for entry in selected:
                if _points_sha256(lookup[entry["source_key"]]["points"]) != entry["points_sha256"]:
                    raise ValueError(f"Locked path checksum mismatch on page {page}")
                if abs(polyline_length(np.asarray(lookup[entry["source_key"]]["points"])) - entry["length_px"]) > 1e-6:
                    raise ValueError(f"Locked path length mismatch on page {page}")
            count = len(keys)
            source_rule = planned[page]["classification"]
        else:
            keys = ISSUE_KEYS[page]
            count = ISSUE_COUNTS[page]
            source_rule = "user_operational_definition_20260924"
        paths = [_path(key, lookup[key]["points"], diag, row["mm_per_output_px"]) for key in keys]
        extra = {"original_adjudication_page": page, "image_sha256": case["image_sha256"], "source_rule": source_rule}
        if page == 5:
            extra["excluded_observation"] = "distal sheath/edge at prior fourth endpoint; no independent terminal path"
        elif page == 6:
            extra["excluded_observation"] = "real immature/cropped shoot tissue may be present; not a complete independently traceable path in standardized image; not labelled artifact"
        elif page == 7:
            extra["path_without_accepted_geometry"] = {
                "status": "independent_structure_present_geometry_unavailable",
                "reason": "two existing short-path curves were explicitly judged geometrically wrong; no invented line or automatic substitute",
                "eligible_for_count_recall": True,
                "eligible_for_length_or_angle": False,
            }
        plants.append(_finish_plant(dataset_id, paths, count, metadata, **extra))

    # The three plants without an adjudication trigger use a prespecified
    # complete human trace, not a newly averaged or model-selected curve.
    r1_export = runtime / "measurements/rater1_round1/revision_after_field_resolution/VPUFUJSD7GCZ_traces.json"
    r1 = read_json(r1_export)
    r1_by_blind = {record["blind_id"]: record for record in r1["records"]}
    remaining = sorted(set(metadata) - {plant["dataset_id"] for plant in plants})
    if len(remaining) != 3:
        raise ValueError(f"Expected exactly three non-adjudicated plants, found {remaining}")
    rows_by_dataset = {row["dataset_id"]: row for row in r1_rows.values()}
    for dataset_id in remaining:
        row = metadata[dataset_id]
        diag = math.hypot(row["shoot_bbox_width_px"], row["shoot_bbox_height_px"])
        record = r1_by_blind[rows_by_dataset[dataset_id]["blind_id"]]
        if not record.get("submitted", True):
            raise ValueError(f"Rater 1 record not submitted: {dataset_id}")
        traces = record["traces"]
        if not traces or any(trace["visibility_status"] != "measurable" or trace["trace_status"] != "complete" for trace in traces):
            raise ValueError(f"Incomplete Rater 1 trace: {dataset_id}")
        paths = []
        for trace in traces:
            points = [[point["x"], point["y"]] for point in trace["submitted_curve"]]
            path = _path("R1_round1:" + trace["trace_uuid"], points, diag, row["mm_per_output_px"])
            if abs(path["length_px"] - trace["structural_path_length_px"]) > 1e-5:
                raise ValueError(f"Human measurement changed: {dataset_id}")
            paths.append(path)
        plants.append(_finish_plant(dataset_id, paths, len(paths), metadata, source_rule="non_adjudicated_R1_round1_complete", original_adjudication_page=None))
    plants.sort(key=lambda plant: plant["dataset_id"])
    if len(plants) != 16 or {p["dataset_id"] for p in plants} != set(metadata):
        raise ValueError("Pilot membership changed")
    reference = {
        "version": VERSION,
        "status": "frozen_with_explicit_geometry_missingness",
        "date": "2026-09-24",
        "scope": "16 locked V4 val pilot plants only; no V4 test",
        "primary_target": "base-to-tip structural path length, not botanical leaf length or rice leaf stage",
        "operational_inclusion": "distinct independent tip and continuous green-organ path to common shoot path/divergence; development stage alone does not exclude",
        "length_units": ["pixel", "shoot_bbox_diagonal_normalized", "millimeter_scanner_metadata_derived_not_ruler_calibrated"],
        "missing_geometry_policy": "count independently visible path; mask its length and angle; do not impute an unaccepted curve",
        "locked_human_error_floor": {"length_symmetric_relative_mdc95_pct": 3.72, "divergence_angle_mdc95_deg": 17.08},
        "source_sha256": {
            "user_operational_proposal": PROPOSAL_SHA256,
            "original_minimal_json": digest(export),
            "original_minimal_csv": digest(export.with_suffix(".csv")),
            "automatic_equivalence_plan": digest(plan_file),
            "rater1_round1_traces": digest(r1_export),
            **source_provenance,
        },
        "plants": plants,
    }
    raw = _json_bytes(reference)
    output.mkdir(parents=True, exist_ok=False)
    (output / "final_gt.json").write_bytes(raw)
    summary = {
        "version": VERSION,
        "status": reference["status"],
        "pilot_plants": len(plants),
        "core_plants": sum(p["pilot_group"] == "core" for p in plants),
        "diagnostic_plants": sum(p["pilot_group"] == "diagnostic" for p in plants),
        "independent_paths": sum(p["operational_independent_path_count"] for p in plants),
        "complete_geometry_paths": sum(p["measurable_reference_path_count"] for p in plants),
        "geometry_missing_paths": sum(p["operational_independent_path_count"] - p["measurable_reference_path_count"] for p in plants),
        "issue_page_counts": {str(page): ISSUE_COUNTS[page] for page in (5, 6, 7)},
        "final_gt_sha256": hashlib.sha256(raw).hexdigest(),
        "method_comparison_ready": True,
        "training_authorized_by_this_file": False,
        "v4_test_opened": False,
    }
    (output / "freeze_manifest.json").write_bytes(_json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=PROTOCOL / "runtime")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proposal", type=Path, default=Path(r"C:\Users\F\Desktop\新建 Text Document.md"))
    args = parser.parse_args()
    output = args.output or args.runtime / "final_gt/phenotype_pilot_20260924_v1"
    print(json.dumps(build(args.runtime, output, args.proposal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

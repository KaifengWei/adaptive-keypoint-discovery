"""Anonymous cross-session matching and human-GT reliability analysis.

This script is evaluation-only. It reads the three frozen human measurement
exports and the private administrator mapping, applies the already frozen
``phenotype-geometry-v1`` Hungarian matcher, and writes anonymous reliability
and adjudication inputs below the git-ignored runtime directory.

It never imports or reads Teacher/Student/model outputs or V4 test images.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from phenotype_gt_geometry import CROSS_SESSION_MATCHING, GEOMETRY_PROTOCOL_VERSION, match_cross_session


PROTOCOL_VERSION = "phenotype-first-pilot-v2"
ANALYSIS_VERSION = "human-gt-reliability-v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260905
LENGTH_ADJUDICATION_THRESHOLD_PCT = 5.0
ANGLE_ADJUDICATION_THRESHOLD_DEG = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def icc_a1(values_1: Iterable[float], values_2: Iterable[float]) -> float:
    """Two-way absolute-agreement, single-measurement ICC(A,1).

    The point-estimate formula is common to the mixed intra-rater and random
    inter-rater forms. Their design labels remain distinct in the output.
    """

    ratings = np.column_stack([
        np.asarray(list(values_1), dtype=np.float64),
        np.asarray(list(values_2), dtype=np.float64),
    ])
    if ratings.ndim != 2 or ratings.shape[0] < 2 or ratings.shape[1] != 2:
        return float("nan")
    n, k = ratings.shape
    grand = float(ratings.mean())
    target_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)
    ms_target = k * float(np.square(target_means - grand).sum()) / (n - 1)
    ms_rater = n * float(np.square(rater_means - grand).sum()) / (k - 1)
    residual = ratings - target_means[:, None] - rater_means[None, :] + grand
    ms_error = float(np.square(residual).sum()) / ((n - 1) * (k - 1))
    denominator = ms_target + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    if abs(denominator) <= 1e-15:
        return float("nan")
    return float((ms_target - ms_error) / denominator)


def difference_statistics(values_1: np.ndarray, values_2: np.ndarray) -> dict[str, float]:
    if len(values_1) != len(values_2) or len(values_1) < 2:
        return {key: float("nan") for key in ("bias", "sd_diff", "sem", "mdc95", "ba_lower", "ba_upper")}
    difference = values_2 - values_1
    bias = float(difference.mean())
    sd_diff = float(difference.std(ddof=1))
    sem = sd_diff / math.sqrt(2.0)
    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "sem": sem,
        "mdc95": 1.96 * sd_diff,
        "ba_lower": bias - 1.96 * sd_diff,
        "ba_upper": bias + 1.96 * sd_diff,
    }


def relative_difference_percent(values_1: np.ndarray, values_2: np.ndarray) -> np.ndarray:
    denominator = 0.5 * (values_1 + values_2)
    result = np.full(len(values_1), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-15
    result[valid] = 100.0 * (values_2[valid] - values_1[valid]) / denominator[valid]
    return result


def percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    valid = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(valid):
        return None, None
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_by_plant(
    rows: list[dict[str, Any]],
    *,
    difference_only: bool,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float | None, float | None]]:
    plant_ids = sorted({str(row["adjudication_pair_id"]) for row in rows})
    if len(plant_ids) < 2:
        return {key: (None, None) for key in ("bias", "sd_diff", "sem", "mdc95", "ba_lower", "ba_upper", "icc")}
    grouped = {plant_id: [row for row in rows if row["adjudication_pair_id"] == plant_id] for plant_id in plant_ids}
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {key: [] for key in ("bias", "sd_diff", "sem", "mdc95", "ba_lower", "ba_upper", "icc")}
    for _ in range(resamples):
        chosen = rng.choice(plant_ids, size=len(plant_ids), replace=True)
        replicate = [row for plant_id in chosen for row in grouped[str(plant_id)]]
        values_1 = np.asarray([float(row["value_1"]) for row in replicate], dtype=np.float64)
        values_2 = np.asarray([float(row["value_2"]) for row in replicate], dtype=np.float64)
        stats = difference_statistics(values_1, values_2)
        for key in ("bias", "sd_diff", "sem", "mdc95", "ba_lower", "ba_upper"):
            sampled[key].append(stats[key])
        sampled["icc"].append(float("nan") if difference_only else icc_a1(values_1, values_2))
    return {key: percentile_interval(values) for key, values in sampled.items()}


def as_match_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_uuid": trace["trace_uuid"],
        "visibility_status": trace["visibility_status"],
        "tip_xy": [float(trace["tip_xy"]["x"]), float(trace["tip_xy"]["y"])],
        "tip_clockwise_angle_deg": float(trace["tip_clockwise_angle_deg"]),
        "resampled_curve": [[float(point["x"]), float(point["y"])] for point in trace["submitted_curve"]],
    }


def load_frozen_export(path: Path, expected_rater: str, expected_round: int) -> dict[str, Any]:
    payload = load_json(path)
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"unexpected protocol in {path}")
    if payload["geometry_protocol_version"] != GEOMETRY_PROTOCOL_VERSION:
        raise ValueError(f"unexpected geometry protocol in {path}")
    if payload["rater_id"] != expected_rater or int(payload["measurement_round"]) != expected_round:
        raise ValueError(f"unexpected rater/round in {path}")
    if len(payload["records"]) != 16 or any(not record["submitted"] for record in payload["records"]):
        raise ValueError(f"incomplete frozen export in {path}")
    return payload


def source_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["blind_id"]: record for record in payload["records"]}


def build_comparison(
    *,
    comparison_type: str,
    first_key: tuple[str, int],
    second_key: tuple[str, int],
    allowed_groups: set[str],
    mapping_by_session: dict[tuple[str, int, str], dict[str, str]],
    records_by_source: dict[tuple[str, int], dict[str, dict[str, Any]]],
    selection_by_dataset: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    match_rows: list[dict[str, Any]] = []
    plant_rows: list[dict[str, Any]] = []
    datasets = sorted({key[2] for key in mapping_by_session if key[:2] == first_key})
    for dataset_id in datasets:
        first_map = mapping_by_session[(first_key[0], first_key[1], dataset_id)]
        if first_map["pilot_group"] not in allowed_groups:
            continue
        second_map = mapping_by_session[(second_key[0], second_key[1], dataset_id)]
        if first_map["adjudication_pair_id"] != second_map["adjudication_pair_id"]:
            raise ValueError(f"anonymous pair ID mismatch for {dataset_id}")
        pair_id = first_map["adjudication_pair_id"]
        first_record = records_by_source[first_key][first_map["blind_id"]]
        second_record = records_by_source[second_key][second_map["blind_id"]]
        first_traces = first_record["traces"]
        second_traces = second_record["traces"]
        bbox_diag = float(first_traces[0]["structural_path_length_px"]) / float(first_traces[0]["structural_path_length_bbox_norm"])
        if second_traces:
            second_diag = float(second_traces[0]["structural_path_length_px"]) / float(second_traces[0]["structural_path_length_bbox_norm"])
            if not math.isclose(bbox_diag, second_diag, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"bbox diagonal mismatch for anonymous pair {pair_id}")
        matches = match_cross_session(
            [as_match_trace(trace) for trace in first_traces],
            [as_match_trace(trace) for trace in second_traces],
            bbox_diag,
        )
        first_by_uuid = {trace["trace_uuid"]: trace for trace in first_traces}
        second_by_uuid = {trace["trace_uuid"]: trace for trace in second_traces}
        reasons: set[str] = set()
        matched_count = 0
        unmatched_first = 0
        unmatched_second = 0
        for match in matches:
            row = {
                "analysis_version": ANALYSIS_VERSION,
                "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
                "cross_session_matching": CROSS_SESSION_MATCHING,
                "comparison_type": comparison_type,
                "pilot_group": first_map["pilot_group"],
                "adjudication_pair_id": pair_id,
                **asdict(match),
            }
            if match.status == "matched":
                matched_count += 1
                first_trace = first_by_uuid[str(match.trace_uuid_session_1)]
                second_trace = second_by_uuid[str(match.trace_uuid_session_2)]
                length_1 = float(first_trace["structural_path_length_px"])
                length_2 = float(second_trace["structural_path_length_px"])
                relative = float(relative_difference_percent(np.asarray([length_1]), np.asarray([length_2]))[0])
                angle_1 = finite_or_none(first_trace.get("divergence_angle_deg"))
                angle_2 = finite_or_none(second_trace.get("divergence_angle_deg"))
                row.update({
                    "gt_leaf_id_session_1": first_trace["gt_leaf_id_postsubmit"],
                    "gt_leaf_id_session_2": second_trace["gt_leaf_id_postsubmit"],
                    "length_px_session_1": length_1,
                    "length_px_session_2": length_2,
                    "length_relative_diff_pct": relative,
                    "length_bbox_norm_session_1": float(first_trace["structural_path_length_bbox_norm"]),
                    "length_bbox_norm_session_2": float(second_trace["structural_path_length_bbox_norm"]),
                    "length_mm_metadata_session_1": length_1 * float(selection_by_dataset[dataset_id]["mm_per_output_px"]),
                    "length_mm_metadata_session_2": length_2 * float(selection_by_dataset[dataset_id]["mm_per_output_px"]),
                    "scale_label": "derived from scanner metadata",
                    "reference_main_session_1": bool(first_trace["reference_main_path"]),
                    "reference_main_session_2": bool(second_trace["reference_main_path"]),
                    "divergence_angle_deg_session_1": angle_1,
                    "divergence_angle_deg_session_2": angle_2,
                })
                if abs(relative) > LENGTH_ADJUDICATION_THRESHOLD_PCT:
                    reasons.add("length_relative_diff_gt_5pct")
                if angle_1 is not None and angle_2 is not None and abs(angle_2 - angle_1) > ANGLE_ADJUDICATION_THRESHOLD_DEG:
                    reasons.add("divergence_angle_diff_gt_5deg")
                if bool(first_trace["reference_main_path"]) != bool(second_trace["reference_main_path"]):
                    reasons.add("reference_main_identity_differs")
            elif match.status == "unmatched_session_1":
                unmatched_first += 1
                reasons.add("leaf_existence_or_identity_differs")
            else:
                unmatched_second += 1
                reasons.add("leaf_existence_or_identity_differs")
            match_rows.append(row)
        plant_rows.append({
            "analysis_version": ANALYSIS_VERSION,
            "comparison_type": comparison_type,
            "pilot_group": first_map["pilot_group"],
            "adjudication_pair_id": pair_id,
            "trace_count_session_1": len(first_traces),
            "trace_count_session_2": len(second_traces),
            "matched_trace_count": matched_count,
            "unmatched_session_1_count": unmatched_first,
            "unmatched_session_2_count": unmatched_second,
            "requires_adjudication": "yes" if reasons else "no",
            "adjudication_reasons": ";".join(sorted(reasons)),
        })
    return match_rows, plant_rows


def endpoint_rows(match_rows: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in match_rows:
        if row["status"] != "matched":
            continue
        if endpoint == "length_px":
            left, right = row.get("length_px_session_1"), row.get("length_px_session_2")
        elif endpoint == "length_bbox_norm":
            left, right = row.get("length_bbox_norm_session_1"), row.get("length_bbox_norm_session_2")
        elif endpoint == "length_mm_metadata":
            left, right = row.get("length_mm_metadata_session_1"), row.get("length_mm_metadata_session_2")
        elif endpoint == "length_relative_pct":
            relative = finite_or_none(row.get("length_relative_diff_pct"))
            if relative is None:
                continue
            left, right = 0.0, relative
        elif endpoint == "divergence_angle_deg":
            left, right = row.get("divergence_angle_deg_session_1"), row.get("divergence_angle_deg_session_2")
        else:
            raise ValueError(endpoint)
        left_value, right_value = finite_or_none(left), finite_or_none(right)
        if left_value is not None and right_value is not None:
            output.append({"adjudication_pair_id": row["adjudication_pair_id"], "value_1": left_value, "value_2": right_value})
    return output


def summarize_endpoint(
    *,
    comparison_type: str,
    match_rows: list[dict[str, Any]],
    plant_rows: list[dict[str, Any]],
    endpoint: str,
    unit: str,
    icc_type: str,
    difference_only: bool = False,
) -> dict[str, Any]:
    rows = endpoint_rows(match_rows, endpoint)
    values_1 = np.asarray([row["value_1"] for row in rows], dtype=np.float64)
    values_2 = np.asarray([row["value_2"] for row in rows], dtype=np.float64)
    stats = difference_statistics(values_1, values_2)
    intervals = bootstrap_by_plant(rows, difference_only=difference_only)
    icc = float("nan") if difference_only else icc_a1(values_1, values_2)
    unmatched = sum(int(row["unmatched_session_1_count"]) + int(row["unmatched_session_2_count"]) for row in plant_rows)
    matched_total = sum(int(row["matched_trace_count"]) for row in plant_rows)
    not_applicable = 0
    role_mismatch = 0
    unresolved = 0
    if endpoint == "divergence_angle_deg":
        for row in match_rows:
            if row["status"] != "matched":
                continue
            main_1 = bool(row.get("reference_main_session_1"))
            main_2 = bool(row.get("reference_main_session_2"))
            if main_1 and main_2:
                not_applicable += 1
            elif main_1 != main_2:
                role_mismatch += 1
            elif finite_or_none(row.get("divergence_angle_deg_session_1")) is None or finite_or_none(row.get("divergence_angle_deg_session_2")) is None:
                unresolved += 1
    missing_endpoint = role_mismatch + unresolved if endpoint == "divergence_angle_deg" else matched_total - len(rows)
    result: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "endpoint": endpoint,
        "unit": unit,
        "comparison_type": comparison_type,
        "n_plants": len({row["adjudication_pair_id"] for row in rows}),
        "n_trace_pairs": len(rows),
        **stats,
        "icc_type": "not_applicable_difference_scale" if difference_only else icc_type,
        "icc": None if not math.isfinite(icc) else icc,
        "bootstrap_cluster": "plant",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "missing_pair_count": unmatched + missing_endpoint,
        "unmatched_trace_count": unmatched,
        "matched_endpoint_missing_count": missing_endpoint,
        "not_applicable_pair_count": not_applicable,
        "endpoint_role_mismatch_count": role_mismatch,
        "endpoint_unresolved_count": unresolved,
    }
    for key, (low, high) in intervals.items():
        result[f"{key}_ci_lower"] = low
        result[f"{key}_ci_upper"] = high
    if difference_only:
        result["icc_ci_lower"] = None
        result["icc_ci_upper"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    protocol_dir = Path(__file__).resolve().parent
    runtime = protocol_dir / "runtime"
    output_dir = args.output_dir or runtime / "reliability" / "20260905"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        ("R1", 1): runtime / "measurements" / "rater1_round1" / "revision_after_field_resolution" / "VPUFUJSD7GCZ_traces.json",
        ("R1", 2): runtime / "measurements" / "rater1_round2" / "raw_export_20260904" / "CHQ474APYAYU_traces.json",
        ("R2", 1): runtime / "measurements" / "rater2_round1" / "revision_after_field_resolution_20260905" / "6GHTS9ZXUD4E_traces.json",
    }
    payloads = {key: load_frozen_export(path, key[0], key[1]) for key, path in paths.items()}
    records_by_source = {key: source_index(payload) for key, payload in payloads.items()}

    mapping_rows = [row for row in read_csv(runtime / "admin" / "blind_id_admin_mapping.csv") if row["pilot_group"] != "practice"]
    mapping_by_session = {(row["rater_id"], int(row["measurement_round"]), row["dataset_id"]): row for row in mapping_rows}
    if len(mapping_by_session) != 48:
        raise ValueError(f"expected 48 formal administrator mappings, found {len(mapping_by_session)}")
    pair_ids_by_dataset: dict[str, set[str]] = {}
    for row in mapping_rows:
        pair_ids_by_dataset.setdefault(row["dataset_id"], set()).add(row["adjudication_pair_id"])
    if len(pair_ids_by_dataset) != 16 or any(len(values) != 1 for values in pair_ids_by_dataset.values()):
        raise ValueError("administrator anonymous pairing is incomplete or inconsistent")

    selection_rows = read_csv(protocol_dir / "phenotype_pilot_selection.csv")
    selection_by_dataset = {row["dataset_id"]: row for row in selection_rows}
    if set(pair_ids_by_dataset) != set(selection_by_dataset):
        raise ValueError("selection metadata and administrator mapping do not contain the same 16 samples")

    comparison_specs = [
        ("intra_rater_R1_round2_minus_round1_all16", ("R1", 1), ("R1", 2), {"core", "diagnostic"}),
        ("inter_rater_R2_minus_R1_round1_core12", ("R1", 1), ("R2", 1), {"core"}),
        ("inter_rater_R2_minus_R1_round1_diagnostic4_descriptive", ("R1", 1), ("R2", 1), {"diagnostic"}),
    ]
    all_match_rows: list[dict[str, Any]] = []
    all_plant_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for comparison_type, first_key, second_key, groups in comparison_specs:
        matches, plants = build_comparison(
            comparison_type=comparison_type,
            first_key=first_key,
            second_key=second_key,
            allowed_groups=groups,
            mapping_by_session=mapping_by_session,
            records_by_source=records_by_source,
            selection_by_dataset=selection_by_dataset,
        )
        all_match_rows.extend(matches)
        all_plant_rows.extend(plants)
        icc_design = "ICC(A,1)_two_way_mixed_absolute_single" if comparison_type.startswith("intra") else "ICC(A,1)_two_way_random_absolute_single"
        for endpoint, unit, difference_only in (
            ("length_px", "px", False),
            ("length_bbox_norm", "bbox_diagonal", False),
            ("length_mm_metadata", "mm_derived_from_scanner_metadata", False),
            ("length_relative_pct", "percent_symmetric_difference", True),
            ("divergence_angle_deg", "degree", False),
        ):
            reliability_rows.append(summarize_endpoint(
                comparison_type=comparison_type,
                match_rows=matches,
                plant_rows=plants,
                endpoint=endpoint,
                unit=unit,
                icc_type=icc_design,
                difference_only=difference_only,
            ))

    match_fields = [
        "analysis_version", "geometry_protocol_version", "cross_session_matching", "comparison_type", "pilot_group",
        "adjudication_pair_id", "trace_uuid_session_1", "trace_uuid_session_2", "status", "cost", "tip_norm",
        "curve_norm", "polar_gap", "gt_leaf_id_session_1", "gt_leaf_id_session_2", "length_px_session_1",
        "length_px_session_2", "length_relative_diff_pct", "length_bbox_norm_session_1", "length_bbox_norm_session_2",
        "length_mm_metadata_session_1", "length_mm_metadata_session_2", "scale_label", "reference_main_session_1",
        "reference_main_session_2", "divergence_angle_deg_session_1", "divergence_angle_deg_session_2",
    ]
    plant_fields = [
        "analysis_version", "comparison_type", "pilot_group", "adjudication_pair_id", "trace_count_session_1",
        "trace_count_session_2", "matched_trace_count", "unmatched_session_1_count", "unmatched_session_2_count",
        "requires_adjudication", "adjudication_reasons",
    ]
    reliability_fields = [
        "analysis_version", "endpoint", "unit", "comparison_type", "n_plants", "n_trace_pairs", "bias", "bias_ci_lower",
        "bias_ci_upper", "sd_diff", "sd_diff_ci_lower", "sd_diff_ci_upper", "sem", "sem_ci_lower", "sem_ci_upper",
        "mdc95", "mdc95_ci_lower", "mdc95_ci_upper", "ba_lower", "ba_lower_ci_lower", "ba_lower_ci_upper",
        "ba_upper", "ba_upper_ci_lower", "ba_upper_ci_upper", "icc_type", "icc", "icc_ci_lower", "icc_ci_upper",
        "bootstrap_cluster", "bootstrap_resamples", "bootstrap_seed", "missing_pair_count", "unmatched_trace_count",
        "matched_endpoint_missing_count", "not_applicable_pair_count", "endpoint_role_mismatch_count", "endpoint_unresolved_count",
    ]
    write_csv(output_dir / "anonymous_trace_matches.csv", all_match_rows, match_fields)
    write_csv(output_dir / "anonymous_plant_adjudication_screen.csv", all_plant_rows, plant_fields)
    write_csv(output_dir / "reliability_summary.csv", reliability_rows, reliability_fields)

    adjudication_rows = [row for row in all_plant_rows if row["requires_adjudication"] == "yes"]
    write_csv(output_dir / "adjudication_candidates.csv", adjudication_rows, plant_fields)
    report = {
        "status": "pass_pending_method_blind_adjudication",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
        "cross_session_matching": CROSS_SESSION_MATCHING,
        "bootstrap": {"cluster": "plant", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "interval": "percentile_2.5_97.5"},
        "adjudication_thresholds": {"absolute_symmetric_length_difference_pct": 5.0, "absolute_divergence_angle_difference_deg": 5.0, "leaf_existence_or_identity_difference": True},
        "input_files_sha256": {str(path.relative_to(protocol_dir)): sha256_file(path) for path in paths.values()},
        "mapping_rows_used": len(mapping_rows),
        "anonymous_pair_count": len(pair_ids_by_dataset),
        "comparison_counts": {
            spec[0]: {
                "plants": len([row for row in all_plant_rows if row["comparison_type"] == spec[0]]),
                "matched_traces": len([row for row in all_match_rows if row["comparison_type"] == spec[0] and row["status"] == "matched"]),
                "unmatched_session_1": len([row for row in all_match_rows if row["comparison_type"] == spec[0] and row["status"] == "unmatched_session_1"]),
                "unmatched_session_2": len([row for row in all_match_rows if row["comparison_type"] == spec[0] and row["status"] == "unmatched_session_2"]),
                "plants_requiring_adjudication": len([row for row in adjudication_rows if row["comparison_type"] == spec[0]]),
            }
            for spec in comparison_specs
        },
        "identity_leakage_detected": False,
        "model_outputs_read": 0,
        "v4_test_read": 0,
        "method_comparison_started": False,
        "next_gate": "method-blind adjudication and final human-GT freeze",
        "reliability": reliability_rows,
    }
    (output_dir / "reliability_summary.json").write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    forbidden = ("v4_val_", "dataset_id", "teacher", "student_b", "student_d", "decoder")
    public_outputs = [
        output_dir / "anonymous_trace_matches.csv",
        output_dir / "anonymous_plant_adjudication_screen.csv",
        output_dir / "adjudication_candidates.csv",
        output_dir / "reliability_summary.csv",
        output_dir / "reliability_summary.json",
    ]
    leakage_hits = []
    for path in public_outputs:
        text = path.read_text(encoding="utf-8-sig").lower()
        for token in forbidden:
            if token in text:
                leakage_hits.append({"file": path.name, "token": token})
    if leakage_hits:
        raise ValueError(f"anonymous output leakage detected: {leakage_hits}")
    print(json.dumps({key: value for key, value in report.items() if key != "reliability"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

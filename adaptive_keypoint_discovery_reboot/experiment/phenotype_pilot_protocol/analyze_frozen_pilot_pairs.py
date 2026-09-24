"""Plant-paired, val-only audit of an already generated frozen pilot comparison.

This script cannot read images, human source traces, model checkpoints or test.
It uses the frozen comparison JSON and keeps Core and Diagnostic separate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


METHOD_PAIRS = (("Student-B", "Teacher-direct"), ("Student-D", "Student-B"))
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPLICATES = 10000


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n_plants": 0, "median": None, "q1": None, "q3": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n_plants": len(array),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def _bootstrap_mean_ci(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[resamples].mean(axis=1)
    return [float(np.quantile(means, q)) for q in (0.025, 0.975)]


def _exact_signflip_p(values: list[float]) -> float | None:
    nonzero = [float(value) for value in values if abs(value) > 1e-12]
    if not nonzero:
        return 1.0 if values else None
    observed = abs(sum(nonzero))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, nonzero))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(nonzero))
    )
    return extreme / (2 ** len(nonzero))


def _plant_pair(left: dict, right: dict) -> dict:
    if left["dataset_id"] != right["dataset_id"] or left["pilot_group"] != right["pilot_group"]:
        raise ValueError("Misaligned pilot plants")
    left_by_gt = {pair["gt_path_sha256"]: pair for pair in left["pairs"]}
    right_by_gt = {pair["gt_path_sha256"]: pair for pair in right["pairs"]}
    common = sorted(set(left_by_gt) & set(right_by_gt))
    length_differences = [
        float(left_by_gt[gt]["length_symmetric_relative_error_pct"])
        - float(right_by_gt[gt]["length_symmetric_relative_error_pct"])
        for gt in common
    ]
    angle_differences = [
        float(left_by_gt[gt]["angle_absolute_error_deg"])
        - float(right_by_gt[gt]["angle_absolute_error_deg"])
        for gt in common
        if left_by_gt[gt]["angle_absolute_error_deg"] is not None
        and right_by_gt[gt]["angle_absolute_error_deg"] is not None
    ]
    return {
        "dataset_id": left["dataset_id"],
        "group": left["pilot_group"],
        "matched_count_difference": left["matched_geometry_count"] - right["matched_geometry_count"],
        "count_absolute_error_difference": left["count_absolute_error"] - right["count_absolute_error"],
        "unmatched_prediction_difference": left["unmatched_prediction_count"] - right["unmatched_prediction_count"],
        "common_matched_gt_paths": len(common),
        "common_angles": len(angle_differences),
        "mean_length_error_difference_pct_points_common_gt": (
            float(np.mean(length_differences)) if length_differences else None
        ),
        "mean_angle_error_difference_deg_common_gt": (
            float(np.mean(angle_differences)) if angle_differences else None
        ),
    }


def _one_pair(comparison: dict, left_name: str, right_name: str) -> dict:
    left_rows = comparison["methods"][left_name]["per_plant"]
    right_rows = comparison["methods"][right_name]["per_plant"]
    if len(left_rows) != 16 or len(right_rows) != 16:
        raise ValueError("Expected 16 paired pilot plants")
    right_by_id = {row["dataset_id"]: row for row in right_rows}
    if len(right_by_id) != 16 or {row["dataset_id"] for row in left_rows} != set(right_by_id):
        raise ValueError("Pilot membership differs between methods")
    paired = [_plant_pair(row, right_by_id[row["dataset_id"]]) for row in left_rows]
    output = {"contrast": f"{left_name} minus {right_name}", "Core": {}, "Diagnostic": {}}
    metrics = (
        "matched_count_difference",
        "count_absolute_error_difference",
        "unmatched_prediction_difference",
        "mean_length_error_difference_pct_points_common_gt",
        "mean_angle_error_difference_deg_common_gt",
    )
    for group_name, group_value in (("Core", "core"), ("Diagnostic", "diagnostic")):
        group_rows = [row for row in paired if row["group"] == group_value]
        if len(group_rows) != (12 if group_name == "Core" else 4):
            raise ValueError("Core/Diagnostic membership changed")
        output[group_name]["per_plant"] = group_rows
        output[group_name]["common_matched_gt_paths"] = sum(row["common_matched_gt_paths"] for row in group_rows)
        output[group_name]["common_angles"] = sum(row["common_angles"] for row in group_rows)
        for index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in group_rows if row[metric] is not None]
            summary = _summary(values)
            if group_name == "Core":
                summary["cluster_bootstrap_mean_95pct_ci"] = _bootstrap_mean_ci(values, BOOTSTRAP_SEED + index)
                summary["exact_signflip_two_sided_p_exploratory"] = _exact_signflip_p(values)
            output[group_name][metric] = summary
    return output


def analyze(source: Path, output: Path) -> dict:
    payload = source.read_bytes()
    comparison = json.loads(payload)
    if comparison["gt_modified"] or comparison["models_modified"] or comparison["test_read"]:
        raise ValueError("Frozen boundary was not maintained")
    if comparison["prediction_coordinate_transform"] != "inverse_518_letterbox_to_standardized_source_pixels_before_matching_and_metrics":
        raise ValueError("Cannot compare uncorrected coordinate systems")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "source_comparison_sha256": hashlib.sha256(payload).hexdigest(),
        "unit_of_resampling": "plant",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "difference_direction": "left_method_minus_right_method; negative error is favorable to left",
        "shared_target_policy": "Length/angle differences only on GT paths matched by both methods; coverage separately includes every plant.",
        "p_value_role": "exact sign-flip descriptive auxiliary, not a decision gate in this 12-plant pilot",
        "contrasts": [_one_pair(comparison, left, right) for left, right in METHOD_PAIRS],
    }
    (output / "paired_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.comparison, args.output)
    for contrast in result["contrasts"]:
        core = contrast["Core"]
        print(contrast["contrast"], "Core common GT", core["common_matched_gt_paths"])
        for metric in ("matched_count_difference", "count_absolute_error_difference", "mean_length_error_difference_pct_points_common_gt", "mean_angle_error_difference_deg_common_gt"):
            print(metric, core[metric])


if __name__ == "__main__":
    main()

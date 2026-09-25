"""Summarize read-only frozen benchmark and perturbation outputs without GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("Teacher-direct", "Student-B")
STAGES = ("point_generation_only", "shared_graph_decoder", "end_to_end_phenotype")
TRANSFORMS = ("flip_horizontal", "rotate_10", "brightness_085")


def percentile(values: list[float] | pd.Series) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return {"n": int(len(values)), "median": float(median), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timings = {method: pd.read_csv(args.input / f"benchmark_{method}.csv") for method in METHODS}
    metadata = {method: json.loads((args.input / f"benchmark_{method}.json").read_text()) for method in METHODS}
    if any(len(frame) != 16 * 10 * 3 for frame in timings.values()):
        raise RuntimeError("Benchmark N is not the preregistered 160 per stage")
    if any(meta["hardware"]["host"] != "neaucs2-OMEN" or "RTX 3090" not in meta["hardware"]["gpu"] for meta in metadata.values()):
        raise RuntimeError("Benchmark was not performed on the preregistered hardware")
    performance = {}
    for stage in STAGES:
        performance[stage] = {}
        for method in METHODS:
            values = timings[method].query("stage == @stage")["milliseconds"]
            performance[stage][method] = {"runtime_ms": percentile(values), "memory": metadata[method]["stages"][stage]}
        paired = timings[METHODS[0]].query("stage == @stage").merge(
            timings[METHODS[1]].query("stage == @stage"),
            on=["stage", "repeat", "dataset_id", "group"], suffixes=("_teacher", "_student"),
            validate="one_to_one",
        )
        performance[stage]["teacher_to_student_paired_speed_ratio"] = percentile(paired.milliseconds_teacher / paired.milliseconds_student)
    robustness = {}
    for method in METHODS:
        payload = json.loads((args.input / f"robustness_{method}.json").read_text())
        frame = pd.DataFrame(payload["rows"])
        if len(frame) != 16 * 3 or set(frame["transform"]) != set(TRANSFORMS):
            raise RuntimeError("Perturbation set or count differs from preregistration")
        robustness[method] = {}
        for group, size in (("core", 12), ("diagnostic", 4)):
            robustness[method][group] = {}
            for transform in TRANSFORMS:
                subset = frame[(frame["group"] == group) & (frame["transform"] == transform)]
                if len(subset) != size:
                    raise RuntimeError(f"Missing robustness cases for {method}/{group}/{transform}")
                lengths = [value for row in subset.length_symmetric_diff_pct for value in row]
                angles = [value for row in subset.branch_angle_abs_diff_deg for value in row]
                total_identity_paths = int(subset.identity_paths.sum())
                total_matched_paths = int(subset.matched_identity_paths.sum())
                robustness[method][group][transform] = {
                    "plants": size,
                    "point_count_absolute_change": percentile(subset.point_count_abs_change),
                    "point_spatial_repeatability_f1": percentile(subset.point_f1),
                    "path_count_exact_n": int(subset.path_count_exact.sum()),
                    "path_count_exact_denominator": size,
                    "matched_identity_paths": total_matched_paths,
                    "identity_paths": total_identity_paths,
                    "path_coverage": None if not total_identity_paths else total_matched_paths / total_identity_paths,
                    "length_symmetric_difference_pct": percentile(lengths),
                    "branch_angle_absolute_difference_deg": percentile(angles),
                }
    result = {"scope": "frozen Core12 + Diagnostic4 V4 val only", "benchmark": performance,
              "robustness": robustness, "hardware": metadata["Teacher-direct"]["hardware"],
              "GT_read": False, "V4_test_read": False, "model_training": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

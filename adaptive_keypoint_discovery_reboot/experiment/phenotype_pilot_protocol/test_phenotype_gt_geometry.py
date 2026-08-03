from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

import phenotype_gt_geometry as geometry


def test_straight_length_and_duplicate_removal() -> None:
    result = geometry.trace_metrics([[0, 0], [0, 0], [50, 0], [100, 0]], 200)
    assert abs(result["structural_path_length_px"] - 100.0) < 1e-8
    assert abs(result["structural_path_length_bbox_norm"] - 0.5) < 1e-8
    assert abs(result["total_turning_angle_deg"]) < 1e-8


def test_reference_main_tie_break() -> None:
    traces = [
        {"trace_uuid": "b", "visibility_status": "measurable", "structural_path_length_px": 100.0, "chord_length_px": 90.0, "tip_clockwise_angle_deg": 20.0},
        {"trace_uuid": "a", "visibility_status": "measurable", "structural_path_length_px": 99.5, "chord_length_px": 90.0, "tip_clockwise_angle_deg": 10.0},
    ]
    assert geometry.select_reference_main(traces, 200.0) == "a"


def test_divergence_angle() -> None:
    main = geometry.resample_trace([[0, 0], [50, 0], [100, 0]])
    branch = geometry.resample_trace([[0, 0], [50, 0], [75, 25], [100, 50]])
    result = geometry.divergence_angle(main, branch, 150.0)
    assert result["status"] == "ok"
    assert 35.0 < result["divergence_angle_deg"] < 55.0


def test_cross_session_matching() -> None:
    def record(uuid: str, points: list[list[float]]) -> dict:
        metrics = geometry.trace_metrics(points, 200.0)
        return {
            "trace_uuid": uuid,
            "visibility_status": "measurable",
            "tip_xy": metrics["resampled_curve"][-1],
            "tip_clockwise_angle_deg": metrics["tip_clockwise_angle_deg"],
            "resampled_curve": metrics["resampled_curve"],
        }
    left = [record("l1", [[0, 0], [50, 0], [100, 0]]), record("l2", [[0, 0], [40, 20], [80, 50]])]
    right = [record("r2", [[1, 0], [41, 21], [81, 51]]), record("r1", [[1, 0], [51, 0], [101, 0]])]
    matches = geometry.match_cross_session(left, right, 200.0)
    pairs = {(m.trace_uuid_session_1, m.trace_uuid_session_2) for m in matches if m.status == "matched"}
    assert pairs == {("l1", "r1"), ("l2", "r2")}


def test_browser_python_parity_if_report_exists() -> None:
    report_path = Path(__file__).resolve().parent / "validation" / "20260803" / "non_pilot_dry_run_report.json"
    if not report_path.exists():
        return
    browser = json.loads(report_path.read_text(encoding="utf-8"))["browser_geometry_fixture"]
    python = geometry.trace_metrics([[0, 0], [40, 20], [80, 10], [120, 50]], 200.0)
    for key in ("structural_path_length_px", "structural_path_length_bbox_norm", "chord_length_px", "total_turning_angle_deg", "mean_abs_curvature_per_px"):
        assert math.isclose(float(browser[key]), float(python[key]), rel_tol=1e-10, abs_tol=1e-10), (key, browser[key], python[key])


if __name__ == "__main__":
    test_straight_length_and_duplicate_removal()
    test_reference_main_tie_break()
    test_divergence_angle()
    test_cross_session_matching()
    test_browser_python_parity_if_report_exists()
    print("phenotype_gt_geometry: 5 tests passed (including browser/Python parity when report is present)")

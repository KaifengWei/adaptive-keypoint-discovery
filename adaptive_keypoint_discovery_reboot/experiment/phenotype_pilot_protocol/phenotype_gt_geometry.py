"""Frozen phenotype geometry and cross-session matching for the human-GT pilot.

This module is evaluation-only.  It does not import or modify any model,
teacher, graph, decoder, threshold, or V4 test resource.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


GEOMETRY_PROTOCOL_VERSION = "phenotype-geometry-v1"
RESAMPLING_METHOD = "pchip_arc_length_240"
ANGLE_PROTOCOL = "shared_prefix_tau_0.01D_local_interval_0.05D"
CROSS_SESSION_MATCHING = "c55_30_15_tip012_total015_hungarian"
EPS = 1e-12


def _deduplicate(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(arr) < 2:
        raise ValueError("a trace needs at least two points")
    keep = np.concatenate([[True], np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-6])
    arr = arr[keep]
    if len(arr) < 2:
        raise ValueError("a trace collapses after duplicate removal")
    return arr


def _edge_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    value = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if value == 0.0 or m0 == 0.0 or math.copysign(1.0, value) != math.copysign(1.0, m0):
        return 0.0
    if math.copysign(1.0, m0) != math.copysign(1.0, m1) and abs(value) > 3.0 * abs(m0):
        return 3.0 * m0
    return value


def _pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    slopes = np.diff(y) / h
    if n == 2:
        return np.asarray([slopes[0], slopes[0]], dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    d[0] = _edge_slope(float(h[0]), float(h[1]), float(slopes[0]), float(slopes[1]))
    d[-1] = _edge_slope(float(h[-1]), float(h[-2]), float(slopes[-1]), float(slopes[-2]))
    for k in range(1, n - 1):
        left, right = float(slopes[k - 1]), float(slopes[k])
        if left == 0.0 or right == 0.0 or math.copysign(1.0, left) != math.copysign(1.0, right):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / left + w2 / right)
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = _pchip_derivatives(x, y)
    result = np.empty_like(target, dtype=np.float64)
    interval = np.searchsorted(x, target, side="right") - 1
    interval = np.clip(interval, 0, len(x) - 2)
    for out_i, k in enumerate(interval):
        h = x[k + 1] - x[k]
        t = (target[out_i] - x[k]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        result[out_i] = h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]
    return result


def resample_trace(points: Iterable[Iterable[float]], samples: int = 240) -> np.ndarray:
    if samples < 2:
        raise ValueError("samples must be at least 2")
    arr = _deduplicate(points)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
    if cumulative[-1] <= EPS:
        raise ValueError("trace has zero arc length")
    target = np.linspace(0.0, float(cumulative[-1]), int(samples))
    return np.column_stack([
        _pchip_eval(cumulative, arr[:, 0], target),
        _pchip_eval(cumulative, arr[:, 1], target),
    ])


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1).sum())


def clockwise_angle_deg(base: np.ndarray, tip: np.ndarray) -> float:
    # Image y increases downward, so atan2(dy, dx) already increases clockwise.
    return float(math.degrees(math.atan2(float(tip[1] - base[1]), float(tip[0] - base[0]))) % 360.0)


def trace_metrics(points: Iterable[Iterable[float]], bbox_diag: float) -> dict[str, Any]:
    curve = resample_trace(points, 240)
    length = polyline_length(curve)
    chord = float(np.linalg.norm(curve[-1] - curve[0]))
    vectors = np.diff(curve, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    total_turn = 0.0
    for left, right, nl, nr in zip(vectors[:-1], vectors[1:], norms[:-1], norms[1:]):
        if nl > EPS and nr > EPS:
            cosine = float(np.clip(np.dot(left, right) / (nl * nr), -1.0, 1.0))
            total_turn += abs(math.acos(cosine))
    return {
        "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
        "resampling_method": RESAMPLING_METHOD,
        "resampled_curve": curve,
        "structural_path_length_px": length,
        "structural_path_length_bbox_norm": length / max(float(bbox_diag), EPS),
        "chord_length_px": chord,
        "tip_clockwise_angle_deg": clockwise_angle_deg(curve[0], curve[-1]),
        "total_turning_angle_deg": math.degrees(total_turn),
        "mean_abs_curvature_per_px": total_turn / max(length, EPS),
    }


def select_reference_main(traces: list[dict[str, Any]], bbox_diag: float) -> str:
    eligible = [trace for trace in traces if trace.get("visibility_status") == "measurable"]
    if not eligible:
        raise ValueError("no measurable trace is available")
    max_length = max(float(trace["structural_path_length_px"]) for trace in eligible)
    candidates = [trace for trace in eligible if (max_length - float(trace["structural_path_length_px"])) / max(max_length, EPS) <= 0.01]
    max_chord = max(float(trace["chord_length_px"]) for trace in candidates)
    chord_tol = max(1.0, 0.001 * float(bbox_diag))
    candidates = [trace for trace in candidates if max_chord - float(trace["chord_length_px"]) <= chord_tol]
    candidates.sort(key=lambda trace: (float(trace["tip_clockwise_angle_deg"]), str(trace["trace_uuid"])))
    return str(candidates[0]["trace_uuid"])


def _cumulative(curve: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])


def _point_at_arc(curve: np.ndarray, cumulative: np.ndarray, target: float) -> np.ndarray:
    target = float(np.clip(target, 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, target, side="right") - 1)
    idx = max(0, min(idx, len(curve) - 2))
    span = cumulative[idx + 1] - cumulative[idx]
    ratio = 0.0 if span <= EPS else (target - cumulative[idx]) / span
    return curve[idx] * (1.0 - ratio) + curve[idx + 1] * ratio


def divergence_angle(main_curve: np.ndarray, branch_curve: np.ndarray, bbox_diag: float) -> dict[str, Any]:
    main = np.asarray(main_curve, dtype=np.float64)
    branch = np.asarray(branch_curve, dtype=np.float64)
    tolerance = max(2.0, 0.01 * float(bbox_diag))
    distances = np.linalg.norm(branch[:, None, :] - main[None, :, :], axis=2)
    nearest_idx = distances.argmin(axis=1)
    nearest_dist = distances[np.arange(len(branch)), nearest_idx]
    consecutive_outside = 0
    shared_end = None
    for index, distance in enumerate(nearest_dist):
        if distance <= tolerance:
            shared_end = index
            consecutive_outside = 0
        else:
            consecutive_outside += 1
            if consecutive_outside >= 3:
                break
    if shared_end is None or shared_end >= len(branch) - 3:
        return {"status": "divergence_unresolved", "divergence_angle_deg": None}
    main_index = int(nearest_idx[shared_end])
    main_s = _cumulative(main)
    branch_s = _cumulative(branch)
    window = 0.05 * float(bbox_diag)
    main_start_s = max(0.0, float(main_s[main_index]) - window)
    branch_end_s = min(float(branch_s[-1]), float(branch_s[shared_end]) + window)
    main_vector = main[main_index] - _point_at_arc(main, main_s, main_start_s)
    branch_vector = _point_at_arc(branch, branch_s, branch_end_s) - branch[shared_end]
    denominator = float(np.linalg.norm(main_vector) * np.linalg.norm(branch_vector))
    if denominator <= EPS:
        return {"status": "divergence_unresolved", "divergence_angle_deg": None}
    cosine = float(np.clip(np.dot(main_vector, branch_vector) / denominator, -1.0, 1.0))
    return {
        "status": "ok",
        "angle_protocol": ANGLE_PROTOCOL,
        "divergence_angle_deg": math.degrees(math.acos(cosine)),
        "divergence_point_x_px": float(branch[shared_end, 0]),
        "divergence_point_y_px": float(branch[shared_end, 1]),
        "main_local_interval_px": float(main_s[main_index] - main_start_s),
        "branch_local_interval_px": float(branch_end_s - branch_s[shared_end]),
    }


def _resample_linear(curve: np.ndarray, samples: int = 64) -> np.ndarray:
    cumulative = _cumulative(curve)
    target = np.linspace(0.0, float(cumulative[-1]), samples)
    return np.column_stack([np.interp(target, cumulative, curve[:, axis]) for axis in range(2)])


def _curve_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return 0.5 * (float(distances.min(axis=1).mean()) + float(distances.min(axis=0).mean()))


@dataclass(frozen=True)
class MatchRecord:
    trace_uuid_session_1: str | None
    trace_uuid_session_2: str | None
    status: str
    cost: float | None
    tip_norm: float | None
    curve_norm: float | None
    polar_gap: float | None


def match_cross_session(session_1: list[dict[str, Any]], session_2: list[dict[str, Any]], bbox_diag: float) -> list[MatchRecord]:
    n1, n2 = len(session_1), len(session_2)
    size = n1 + n2
    dummy_cost = 0.15
    cost = np.full((size, size), dummy_cost, dtype=np.float64)
    details: dict[tuple[int, int], tuple[float, float | None, float]] = {}
    for i, left in enumerate(session_1):
        for j, right in enumerate(session_2):
            tip_norm = float(np.linalg.norm(np.asarray(left["tip_xy"]) - np.asarray(right["tip_xy"]))) / max(float(bbox_diag), EPS)
            angle_gap = abs(float(left["tip_clockwise_angle_deg"]) - float(right["tip_clockwise_angle_deg"])) % 360.0
            polar_gap = min(angle_gap, 360.0 - angle_gap) / 180.0
            curve_norm: float | None = None
            if left.get("visibility_status") == "measurable" and right.get("visibility_status") == "measurable":
                lcurve = _resample_linear(np.asarray(left["resampled_curve"], dtype=np.float64), 64)
                rcurve = _resample_linear(np.asarray(right["resampled_curve"], dtype=np.float64), 64)
                curve_norm = _curve_distance(lcurve, rcurve) / max(float(bbox_diag), EPS)
                value = 0.55 * tip_norm + 0.30 * curve_norm + 0.15 * polar_gap
            else:
                value = 0.75 * tip_norm + 0.25 * polar_gap
            admissible = tip_norm <= 0.12 and value <= 0.15
            if admissible:
                # Stable epsilon makes exact numerical ties deterministic without affecting thresholds.
                rank = sorted((str(left["trace_uuid"]), str(right["trace_uuid"])))
                tie = (sum(ord(char) for char in "|".join(rank)) % 1000) * 1e-15
                cost[i, j] = value + tie
                details[(i, j)] = (tip_norm, curve_norm, polar_gap)
            else:
                cost[i, j] = 1.0
    rows, cols = linear_sum_assignment(cost)
    matched_1: set[int] = set()
    matched_2: set[int] = set()
    result: list[MatchRecord] = []
    for i, j in zip(rows, cols):
        if i < n1 and j < n2 and cost[i, j] <= dummy_cost + EPS and (i, j) in details:
            matched_1.add(i); matched_2.add(j)
            tip_norm, curve_norm, polar_gap = details[(i, j)]
            result.append(MatchRecord(str(session_1[i]["trace_uuid"]), str(session_2[j]["trace_uuid"]), "matched", float(cost[i, j]), tip_norm, curve_norm, polar_gap))
    for i, trace in enumerate(session_1):
        if i not in matched_1:
            result.append(MatchRecord(str(trace["trace_uuid"]), None, "unmatched_session_1", None, None, None, None))
    for j, trace in enumerate(session_2):
        if j not in matched_2:
            result.append(MatchRecord(None, str(trace["trace_uuid"]), "unmatched_session_2", None, None, None, None))
    result.sort(key=lambda item: (item.trace_uuid_session_1 or "~", item.trace_uuid_session_2 or "~"))
    return result

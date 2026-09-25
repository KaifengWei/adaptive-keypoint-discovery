"""Preregistered Pipeline V2 point-to-frozen-skeleton association.

Only the acceptance decision differs from V1. Projection, duplicate resolution,
graph topology, and all downstream code are delegated to the unchanged V1 builder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_prime_phenotype_bridge as bridge  # noqa: E402
from point_conditioned_graph import build_point_conditioned_graph  # noqa: E402


def roi_square_distance(x: float, y: float, roi_xy: np.ndarray) -> float:
    """Exact distance to union of closed, unit pixel squares centered on ROI pixels."""
    if len(roi_xy) == 0:
        raise ValueError("Frozen phenotype ROI is empty")
    gap_x = np.maximum(np.abs(roi_xy[:, 0] - x) - 0.5, 0.0)
    gap_y = np.maximum(np.abs(roi_xy[:, 1] - y) - 0.5, 0.0)
    return float(np.sqrt(np.min(gap_x * gap_x + gap_y * gap_y)))


def association_score(distance: float, roi_distance: float, radius: float, bbox_diag: float) -> tuple[float, float]:
    if not all(math.isfinite(v) and v >= 0 for v in (distance, roi_distance, radius, bbox_diag)):
        raise ValueError("Nonfinite or negative V2 association input")
    if bbox_diag < 1.0:
        raise ValueError("bbox_diag must already use V1 max(1, D) convention")
    sigma0 = 0.0125 * bbox_diag
    sigma = math.hypot(sigma0, min(radius, sigma0))
    return sigma, math.exp(-(distance * distance + roi_distance * roi_distance) / (2.0 * sigma * sigma))


def build_point_conditioned_graph_v2(
    skeleton: np.ndarray,
    learned_points: Sequence[dict[str, Any]],
    bbox_diag: float,
    structural_support: np.ndarray,
    phenotype_roi: np.ndarray,
    basal_transition: np.ndarray,
    seed_base_root: np.ndarray,
    shoot: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return unchanged V1 graph topology and exhaustive V2 association ledger."""
    if not math.isfinite(bbox_diag) or bbox_diag < 1.0:
        raise ValueError("Invalid V1 bbox diagonal")
    shape = skeleton.shape
    if skeleton.ndim != 2 or any(np.asarray(mask).shape != shape for mask in (
        structural_support, phenotype_roi, basal_transition, seed_base_root, shoot
    )):
        raise ValueError("All frozen V2 masks must share the two-dimensional skeleton canvas")
    support = np.asarray(structural_support, dtype=bool)
    roi = np.asarray(phenotype_roi, dtype=bool)
    basal = np.asarray(basal_transition, dtype=bool)
    root = np.asarray(seed_base_root, dtype=bool)
    shoot = np.asarray(shoot, dtype=bool)
    if not roi.any():
        raise ValueError("Frozen phenotype ROI is empty")
    coords, _ = bridge.skeleton_graph(np.asarray(skeleton, dtype=bool))
    if len(coords) == 0:
        graph = build_point_conditioned_graph(skeleton, learned_points, bbox_diag, 0.025)
        return graph, [
            {"input_index": i, "point_id": str(p["point_id"]), "status": "rejected",
             "rejection_reason": "empty_skeleton", "d_px": None, "d_over_D": None,
             "delta_px": None, "r_px": None, "sigma_px": None, "A_i": None,
             "D_px": float(bbox_diag), "graph_node_id": None}
            for i, p in enumerate(learned_points)
        ]
    roi_xy = np.argwhere(roi)[:, ::-1].astype(np.float64)
    edt = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    tree = cKDTree(coords)
    eligible: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for index, point in enumerate(learned_points):
        x, y = float(point["x"]), float(point["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite learned point")
        d, skeleton_index = tree.query(np.asarray([x, y], dtype=np.float64), k=1)
        skeleton_index = int(skeleton_index)
        qx, qy = (int(v) for v in coords[skeleton_index])
        delta = roi_square_distance(x, y, roi_xy)
        radius = min(float(edt[qy, qx]), 0.0125 * bbox_diag)
        sigma, score = association_score(float(d), delta, radius, bbox_diag)
        roi_ok = bool(roi[qy, qx])
        underground_exclusive = bool(root[qy, qx] and not shoot[qy, qx] and not basal[qy, qx])
        reason = ("outside_frozen_roi" if not roi_ok else
                  "root_only_association" if underground_exclusive else
                  "weighted_association_rejection" if score < math.exp(-2.0) else "")
        ledger.append({
            "input_index": index, "point_id": str(point["point_id"]),
            "original_xy": [x, y], "projected_xy": [qx, qy],
            "skeleton_node": skeleton_index, "d_px": float(d),
            "D_px": float(bbox_diag), "d_over_D": float(d) / bbox_diag,
            "delta_px": delta, "r_px": radius, "sigma_px": sigma, "A_i": score,
            "projected_in_roi": roi_ok, "projected_in_underground_exclusive": underground_exclusive,
            "projected_in_root_not_shoot": bool(root[qy, qx] and not shoot[qy, qx]),
            "projected_in_basal_transition": bool(basal[qy, qx]),
            "status": "rejected" if reason else "eligible", "rejection_reason": reason,
            "graph_node_id": None,
        })
        if not reason:
            eligible.append({**dict(point), "v2_original_input_index": index})

    # Every eligible point is within 0.025*sqrt(2)*D, so V1's internal 1.0*D
    # guard is inert. V1 still performs the identical nearest-pixel projection,
    # raw-confidence duplicate merge, MST, and node order.
    graph = build_point_conditioned_graph(skeleton, eligible, bbox_diag, 1.0)
    for node in graph["nodes"]:
        original_index = int(node.pop("v2_original_input_index"))
        node["input_index"] = original_index
        node.update({k: ledger[original_index][k] for k in (
            "d_px", "d_over_D", "delta_px", "r_px", "sigma_px", "A_i",
            "projected_in_roi", "projected_in_underground_exclusive",
        )})
        ledger[original_index]["status"] = "accepted"
        ledger[original_index]["graph_node_id"] = int(node["node_id"])
    for rejected in graph["rejected_points"]:
        original_index = int(rejected["v2_original_input_index"])
        ledger[original_index]["status"] = "rejected"
        ledger[original_index]["rejection_reason"] = rejected["rejection_reason"]
        rejected["input_index"] = original_index
        rejected.pop("v2_original_input_index", None)
    graph["diagnostics"]["v2_weighted_rejection_count"] = sum(
        item["rejection_reason"] == "weighted_association_rejection" for item in ledger
    )
    graph["diagnostics"]["v2_roi_rejection_count"] = sum(
        item["rejection_reason"] == "outside_frozen_roi" for item in ledger
    )
    graph["diagnostics"]["v2_root_rejection_count"] = sum(
        item["rejection_reason"] == "root_only_association" for item in ledger
    )
    graph["diagnostics"]["input_point_count"] = len(learned_points)
    graph["diagnostics"]["rejected_point_count"] = sum(item["status"] == "rejected" and item["rejection_reason"] != "duplicate_projection_merged" for item in ledger)
    assert len(graph["nodes"]) + len(graph["rejected_points"]) + sum(item["rejection_reason"] not in ("", "duplicate_projection_merged") for item in ledger) == len(learned_points)
    return graph, ledger

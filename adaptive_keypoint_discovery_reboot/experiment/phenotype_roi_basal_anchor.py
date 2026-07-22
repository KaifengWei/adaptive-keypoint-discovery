#!/usr/bin/env python
"""Phenotype-focused shoot ROI and shoot-side basal transition proposal.

This module does not define a manual keypoint.  It separates the complete-plant
archival foreground from the region in which learned points are allowed to
support above-ground phenotype paths.  A basal anchor must still be one of the
learned model points; the automatic transition only supplies an organ-domain
constraint and an audit target.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree

from build_stage_clean_v4_fullplant import strict_green_mask


def _bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if not len(xs):
        return float(math.hypot(mask.shape[0], mask.shape[1]))
    return max(1.0, float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min())))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label_id


def estimate_seed_mask(image_rgb: np.ndarray, root_base_mask: np.ndarray) -> np.ndarray:
    """Estimate the attached brown caryopsis for audit/localization only."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    red, green, blue = [image_rgb[..., index].astype(np.float32) for index in range(3)]
    brown = (
        (hue <= 35)
        & (saturation >= 24)
        & (value >= 18)
        & (value <= 245)
        & ((red >= green * 0.98) | (red >= blue * 1.08))
    )
    near_root = cv2.dilate(root_base_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    candidate = cv2.morphologyEx(
        (brown & near_root).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(candidate, dtype=bool)
    minimum = max(8, round(candidate.size * 0.00001))
    eligible = [
        label_id
        for label_id in range(1, count)
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= minimum
    ]
    if not eligible:
        return np.zeros_like(candidate, dtype=bool)
    label_id = max(eligible, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
    return labels == label_id


def _nearest_source_point(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_y, source_x = np.where(source)
    target_y, target_x = np.where(target)
    if not len(source_x) or not len(target_x):
        raise ValueError("source and target masks must both be non-empty")
    tree = cKDTree(np.column_stack([target_x, target_y]).astype(np.float64))
    distances, _ = tree.query(np.column_stack([source_x, source_y]).astype(np.float64), k=1)
    index = int(np.argmin(distances))
    return np.asarray([source_x[index], source_y[index]], dtype=np.float64)


def _shoot_direction(
    shoot_mask: np.ndarray, green_core: np.ndarray, boundary_xy: np.ndarray
) -> np.ndarray:
    evidence = green_core & shoot_mask
    ys, xs = np.where(evidence if int(evidence.sum()) >= 20 else shoot_mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    if len(points) < 2:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    distances = np.linalg.norm(points - boundary_xy[None, :], axis=1)
    threshold = float(np.quantile(distances, 0.82))
    far_points = points[distances >= threshold]
    vector = far_points.mean(axis=0) - boundary_xy
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        covariance = np.cov(points - points.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        vector = vectors[:, int(np.argmax(values))]
        if float(np.dot(vector, points.mean(axis=0) - boundary_xy)) < 0:
            vector = -vector
        norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def _keep_green_linked(mask: np.ndarray, green_seed: np.ndarray) -> np.ndarray:
    if not np.any(mask) or not np.any(green_seed & mask):
        return _largest_component(mask)
    linked = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
    output = np.zeros_like(mask, dtype=bool)
    minimum = max(6, round(mask.size * 0.000005))
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < minimum:
            continue
        component = labels == label_id
        if np.any(component & green_seed):
            output |= component & mask
    return output


def derive_phenotype_roi(
    image_rgb: np.ndarray,
    shoot_mask: np.ndarray,
    root_base_mask: np.ndarray,
    full_mask: np.ndarray,
) -> dict[str, Any]:
    """Derive an auditable shoot-side ROI without using test labels or manual points."""
    shoot = np.asarray(shoot_mask, dtype=bool)
    root_base = np.asarray(root_base_mask, dtype=bool)
    full = np.asarray(full_mask, dtype=bool)
    if not np.any(shoot):
        raise ValueError("shoot mask is empty")
    green_core = strict_green_mask(image_rgb) & full
    seed = estimate_seed_mask(image_rgb, root_base)
    target = seed if np.any(seed) else root_base
    if not np.any(target):
        raise ValueError("neither seed nor root/base localization evidence is available")

    boundary_xy = _nearest_source_point(shoot, target)
    direction = _shoot_direction(shoot, green_core, boundary_xy)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    yy, xx = np.indices(shoot.shape)
    offsets = np.stack([xx - boundary_xy[0], yy - boundary_xy[1]], axis=-1).astype(np.float64)
    longitudinal = offsets @ direction
    transverse = offsets @ perpendicular
    diag = _bbox_diag(full)

    backward_allowance = max(2.0, 0.004 * diag)
    collar_length = max(8.0, 0.035 * diag)
    collar_half_width = max(7.0, 0.035 * diag)
    shoot_side = longitudinal >= -backward_allowance
    collar_geometry = (
        (longitudinal >= -backward_allowance)
        & (longitudinal <= collar_length)
        & (np.abs(transverse) <= collar_half_width)
    )
    seed_margin = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    collar_candidate = full & collar_geometry & ~seed_margin
    preliminary = ((shoot & shoot_side) | collar_candidate) & ~seed_margin
    preliminary = cv2.morphologyEx(
        preliminary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) > 0
    green_seed = green_core & (longitudinal >= max(3.0, 0.012 * diag))
    phenotype_roi = _keep_green_linked(preliminary, green_seed)
    if not np.any(phenotype_roi):
        phenotype_roi = _largest_component(shoot & shoot_side)

    transition = (
        phenotype_roi
        & (longitudinal >= 0.0)
        & (longitudinal <= collar_length)
        & (np.abs(transverse) <= collar_half_width)
    )
    inward_target = boundary_xy + direction * max(4.0, 0.012 * diag)
    roi_y, roi_x = np.where(transition if np.any(transition) else phenotype_roi)
    roi_points = np.column_stack([roi_x, roi_y]).astype(np.float64)
    transition_center_xy = roi_points[
        int(np.argmin(np.linalg.norm(roi_points - inward_target[None, :], axis=1)))
    ]

    shoot_retention = float((phenotype_roi & shoot).sum() / max(1, int(shoot.sum())))
    root_overlap = float((phenotype_roi & root_base).sum() / max(1, int(root_base.sum())))
    return {
        "phenotype_roi": phenotype_roi,
        "basal_transition": transition,
        "seed_mask": seed,
        "boundary_xy": boundary_xy,
        "transition_center_xy": transition_center_xy,
        "shoot_direction_xy": direction,
        "bbox_diag": diag,
        "shoot_retention_ratio": shoot_retention,
        "root_base_overlap_ratio": root_overlap,
        "seed_detected": bool(np.any(seed)),
    }


def select_learned_basal_anchor(
    points: Iterable[dict[str, Any]],
    phenotype_roi: np.ndarray,
    transition_center_xy: np.ndarray,
    bbox_diag: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select a model point on the shoot side; never synthesize an anchor point."""
    points = [dict(point) for point in points]
    radius = max(2, round(0.010 * bbox_diag))
    tolerant_roi = cv2.dilate(
        phenotype_roi.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    ) > 0
    height, width = phenotype_roi.shape
    eligible: list[tuple[float, float, dict[str, Any]]] = []
    audited: list[dict[str, Any]] = []
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        inside = bool(tolerant_roi[iy, ix])
        distance = float(np.linalg.norm(np.asarray([x, y]) - transition_center_xy))
        point.update({"inside_phenotype_roi": inside, "distance_to_transition_px": distance})
        audited.append(point)
        if inside and distance <= max(12.0, 0.080 * bbox_diag):
            eligible.append((distance, -float(point.get("score", 0.0)), point))
    if not eligible:
        return None, audited
    return min(eligible, key=lambda item: item[:2])[2], audited

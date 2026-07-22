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
from pathlib import Path
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


def _valid_base_hint(base_hint_xy: Iterable[float] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if base_hint_xy is None:
        return None
    values = np.asarray(list(base_hint_xy), dtype=np.float64).reshape(-1)
    if len(values) != 2 or not np.all(np.isfinite(values)):
        return None
    height, width = shape
    if not (-0.05 * width <= values[0] <= 1.05 * width and -0.05 * height <= values[1] <= 1.05 * height):
        return None
    return np.asarray(
        [np.clip(values[0], 0, width - 1), np.clip(values[1], 0, height - 1)], dtype=np.float64
    )


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
    base_hint_xy: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Derive an auditable shoot-side ROI without using test labels or manual points."""
    shoot = np.asarray(shoot_mask, dtype=bool)
    root_base = np.asarray(root_base_mask, dtype=bool)
    full = np.asarray(full_mask, dtype=bool)
    if not np.any(shoot):
        raise ValueError("shoot mask is empty")
    green_core = strict_green_mask(image_rgb) & full
    hint = _valid_base_hint(base_hint_xy, shoot.shape)
    # Keep the seed rule identical to the visually accepted val prototype.
    # The automatic base hint is reserved strictly for rows with no seed/root
    # evidence and must not silently redefine an already reviewed transition.
    seed = estimate_seed_mask(image_rgb, root_base)
    target = seed if np.any(seed) else root_base
    if np.any(target):
        boundary_xy = _nearest_source_point(shoot, target)
        localization_source = "estimated_seed" if np.any(seed) else "seed_base_root_mask"
    elif hint is not None:
        target_mask = np.zeros_like(shoot, dtype=bool)
        target_mask[int(round(hint[1])), int(round(hint[0]))] = True
        boundary_xy = _nearest_source_point(shoot, target_mask)
        localization_source = "manifest_automatic_base_hint_fallback"
    else:
        raise ValueError("neither seed/root evidence nor an automatic base hint is available")

    hint_distance = float("nan")
    if hint is not None:
        hint_target = np.zeros_like(shoot, dtype=bool)
        hint_target[int(round(hint[1])), int(round(hint[0]))] = True
        hint_boundary = _nearest_source_point(shoot, hint_target)
        hint_distance = float(np.linalg.norm(boundary_xy - hint_boundary))
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
    shoot_near = cv2.dilate(shoot.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    obvious_root = root_base & ~shoot_near
    collar_candidate = full & collar_geometry & ~seed_margin & ~obvious_root
    preliminary = ((shoot & shoot_side) | collar_candidate) & ~seed_margin
    preliminary = cv2.morphologyEx(
        preliminary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) > 0
    green_seed = green_core & (longitudinal >= max(3.0, 0.012 * diag))
    phenotype_roi = _keep_green_linked(preliminary, green_seed)
    if not np.any(phenotype_roi):
        phenotype_roi = _largest_component(shoot & shoot_side)

    # The saved shoot mask can itself contain root speckles, so repair and the
    # hard gate use strict-green shoot evidence rather than all mask pixels.
    # This preserves disconnected green leaf tissue without reintroducing a
    # mislabeled brown root merely to improve a mask-overlap number.
    retention_before_repair = float((phenotype_roi & shoot).sum() / max(1, int(shoot.sum())))
    green_shoot_reference = green_core & shoot
    green_retention_before_repair = float(
        (phenotype_roi & green_shoot_reference).sum() / max(1, int(green_shoot_reference.sum()))
    )
    retention_repair_used = green_retention_before_repair < 0.90
    if retention_repair_used:
        recovered_green_shoot = cv2.dilate(
            green_shoot_reference.astype(np.uint8), np.ones((5, 5), np.uint8)
        ) > 0
        phenotype_roi |= recovered_green_shoot & shoot & ~seed_margin
    phenotype_roi &= ~obvious_root

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
    green_shoot_retention = float(
        (phenotype_roi & green_shoot_reference).sum() / max(1, int(green_shoot_reference.sum()))
    )
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
        "green_shoot_retention_ratio": green_shoot_retention,
        "root_base_overlap_ratio": root_overlap,
        "seed_detected": bool(np.any(seed)),
        "localization_source": localization_source,
        "automatic_base_hint_distance_px": hint_distance,
        "retention_before_repair": retention_before_repair,
        "green_retention_before_repair": green_retention_before_repair,
        "retention_repair_used": retention_repair_used,
    }


def apply_phenotype_roi(image_rgb: np.ndarray, phenotype_roi: np.ndarray, feather_sigma: float = 0.8) -> np.ndarray:
    """Place only the phenotype ROI on white while preserving soft edges."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    roi = np.asarray(phenotype_roi, dtype=bool)
    if image.shape[:2] != roi.shape:
        raise ValueError("image and phenotype ROI shapes do not match")
    if not np.any(roi):
        raise ValueError("phenotype ROI is empty")
    alpha = roi.astype(np.float32)
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)
    focused = image.astype(np.float32) * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
    return np.clip(np.rint(focused), 0, 255).astype(np.uint8)


def letterbox_rgb_array(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, float]]:
    """Letterbox an in-memory RGB image using a deterministic white fill."""
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(
        rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    pad_x, pad_y = (size - new_width) // 2, (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "source_width": float(width),
        "source_height": float(height),
    }


def mask_to_letterbox(mask: np.ndarray, mapping: dict[str, float], size: int) -> np.ndarray:
    """Map a source-resolution boolean mask to the model canvas."""
    source = np.asarray(mask, dtype=np.uint8)
    new_width = max(1, round(source.shape[1] * float(mapping["scale"])))
    new_height = max(1, round(source.shape[0] * float(mapping["scale"])))
    resized = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0
    canvas = np.zeros((size, size), dtype=bool)
    pad_x, pad_y = int(mapping["pad_x"]), int(mapping["pad_y"])
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas


def load_phenotype_input(
    dataset_root: Path,
    row: dict[str, Any],
    size: int,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, dict[str, Any]]:
    """Load one V4 row and return model canvas, mapping, focused source and ROI metadata."""
    root = Path(dataset_root)

    def portable_path(key: str) -> Path:
        value = str(row[key]).replace("\\", "/")
        return root / Path(value)

    image_bgr = cv2.imread(str(portable_path("relative_path")), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(portable_path("relative_path"))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    masks: dict[str, np.ndarray] = {}
    for name, key in (
        ("shoot", "shoot_mask_relative_path"),
        ("seed_base_root", "seed_base_root_mask_relative_path"),
        ("full_plant", "full_plant_mask_relative_path"),
    ):
        loaded = cv2.imread(str(portable_path(key)), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise FileNotFoundError(portable_path(key))
        masks[name] = loaded > 0

    hint = (row.get("estimated_base_x_work"), row.get("estimated_base_y_work"))
    result = derive_phenotype_roi(
        image_rgb,
        masks["shoot"],
        masks["seed_base_root"],
        masks["full_plant"],
        base_hint_xy=hint,
    )
    result["source_masks"] = masks
    focused_source = apply_phenotype_roi(image_rgb, result["phenotype_roi"])
    canvas, mapping = letterbox_rgb_array(focused_source, size)
    result["phenotype_roi_model"] = mask_to_letterbox(result["phenotype_roi"], mapping, size)
    result["basal_transition_model"] = mask_to_letterbox(result["basal_transition"], mapping, size)
    result["seed_base_root_model"] = mask_to_letterbox(masks["seed_base_root"], mapping, size)
    return canvas, mapping, focused_source, result


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

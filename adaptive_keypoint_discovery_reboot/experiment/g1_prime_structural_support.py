#!/usr/bin/env python
"""G1-prime: label-free structural support + frozen DINOv2 evidence.

This is a feasibility experiment, not a trained keypoint model.  It never reads
legacy keypoint labels and never calls an optimizer/backward pass.  Candidate
number is determined by image-derived topology and evidence thresholds; the
``max_points`` argument is only a safety guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, convolve, label, maximum_filter
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g0_build_dataset as g0  # noqa: E402
import g1_dinov2_feasibility as g1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-root", type=Path, default=HERE / "data_g0")
    parser.add_argument("--output", type=Path, default=HERE / "outputs_g1_prime")
    parser.add_argument("--local-repo", type=Path, default=HERE / "third_party" / "dinov2_git")
    parser.add_argument(
        "--weights",
        type=Path,
        default=HERE / "third_party" / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth",
    )
    parser.add_argument("--model", default="dinov2_vits14_reg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--full-transforms", action="store_true")
    parser.add_argument("--max-points", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank01(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float64).ravel()
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable")
    return order.reshape(array.shape) / max(len(flat) - 1, 1)


def connected_centroids(binary: np.ndarray) -> list[tuple[float, float]]:
    components, count = label(binary)
    result: list[tuple[float, float]] = []
    for index in range(1, count + 1):
        ys, xs = np.where(components == index)
        if len(xs):
            result.append((float(np.mean(xs)), float(np.mean(ys))))
    return result


def prune_components(binary: np.ndarray, min_area: int, relative_to_largest: float = 0.0) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    clean = np.zeros_like(binary, dtype=bool)
    largest = max((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), default=0)
    threshold = max(min_area, round(relative_to_largest * largest))
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= threshold:
            clean[labels == index] = True
    return clean


def high_confidence_vegetation_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Conservative image-only plant support for heterogeneous scan backgrounds.

    The older G0 audit proposal deliberately favored recall and leaked into low-
    saturation gray backgrounds.  G1-prime instead estimates robust background
    statistics from the image border and requires explicit color confidence.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[:, :, index] for index in range(3))
    red, green, blue = (image_rgb[:, :, index].astype(np.int16) for index in range(3))
    excess_green = 2 * green - red - blue
    thickness = max(2, round(min(image_rgb.shape[:2]) * 0.04))
    border_hsv = g0.border_pixels(hsv, thickness)
    border_rgb = g0.border_pixels(image_rgb, thickness).astype(np.int16)
    border_excess = 2 * border_rgb[:, 1] - border_rgb[:, 0] - border_rgb[:, 2]

    # V3 and similar standardized inputs have a nearly uniform background.
    # In that case, colour distance to the observed border is more faithful
    # than a green-only rule: it preserves pale/yellow basal stem pixels that
    # join leaves into one shoot graph.  The shortcut is image-derived and also
    # survives global brightness changes because the border colour is measured
    # anew for each transformed image.
    background_rgb = np.median(border_rgb.astype(np.float32), axis=0)
    border_distance = np.linalg.norm(border_rgb.astype(np.float32) - background_rgb, axis=1)
    border_distance_median = float(np.median(border_distance))
    border_distance_mad = float(np.median(np.abs(border_distance - border_distance_median)))
    color_distance = np.linalg.norm(image_rgb.astype(np.float32) - background_rgb[None, None, :], axis=2)
    uniform_background = border_distance_mad <= 3.0
    if uniform_background:
        distance_threshold = max(9.0, border_distance_median + 6.0 * 1.4826 * border_distance_mad + 5.0)
        mask = color_distance >= distance_threshold
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        # The finalized shoot dataset contains one connected biological target.
        # Keep only its largest component so isolated interpolation specks never
        # become extra endpoints.
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        clean = np.zeros_like(mask, dtype=bool)
        if component_count > 1:
            largest_label = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            if int(component_stats[largest_label, cv2.CC_STAT_AREA]) >= max(20, round(mask.size * 0.00004)):
                clean = component_labels == largest_label
                clean = binary_fill_holes(clean)
        foreground_fraction = float(clean.mean())
        if 0.0001 <= foreground_fraction <= 0.45:
            return clean, {
                "mask_mode_uniform_background": 1.0,
                "background_distance_threshold": distance_threshold,
                "border_color_distance_mad": border_distance_mad,
                "border_saturation_median": float(np.median(border_hsv[:, 1])),
                "border_excess_green_median": float(np.median(border_excess)),
                "saturation_threshold": float("nan"),
                "excess_green_threshold": float("nan"),
            }

    sat_median = float(np.median(border_hsv[:, 1]))
    sat_mad = float(np.median(np.abs(border_hsv[:, 1] - sat_median)))
    ex_median = float(np.median(border_excess))
    ex_mad = float(np.median(np.abs(border_excess - ex_median)))
    sat_threshold = float(np.clip(max(45.0, sat_median + 4.0 * 1.4826 * sat_mad), 45.0, 85.0))
    ex_threshold = float(np.clip(max(15.0, ex_median + 3.0 * 1.4826 * ex_mad), 15.0, 30.0))

    hue_confident = (hue >= 18) & (hue <= 100) & (saturation >= sat_threshold) & (value >= 22)
    excess_confident = (
        (excess_green >= ex_threshold)
        & (saturation >= max(28.0, 0.62 * sat_threshold))
        & (green >= 35)
        & (green >= red - 3)
        & (green >= blue - 3)
    )
    mask = (hue_confident | excess_confident).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    clean = prune_components(mask > 0, max(20, round(mask.size * 0.00004)), relative_to_largest=0.025)
    return clean, {
        "mask_mode_uniform_background": 0.0,
        "background_distance_threshold": float("nan"),
        "border_color_distance_mad": border_distance_mad,
        "border_saturation_median": sat_median,
        "border_excess_green_median": ex_median,
        "saturation_threshold": sat_threshold,
        "excess_green_threshold": ex_threshold,
    }


def automatic_structural_support(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a support from the current image only; no saved audit mask is read."""
    closed, color_diagnostics = high_confidence_vegetation_mask(image_rgb)
    # A half-resolution topology scale suppresses one-pixel color noise without
    # imposing a fixed number of organs or points.
    topology_size = max(64, min(image_rgb.shape[:2]) // 2)
    regularized = cv2.resize(closed.astype(np.uint8), (topology_size, topology_size), interpolation=cv2.INTER_AREA)
    regularized = cv2.morphologyEx((regularized >= 0.30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    regularized = cv2.resize(regularized, (closed.shape[1], closed.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    closed = prune_components(regularized, max(20, round(regularized.size * 0.00004)), relative_to_largest=0.025)
    skel = skeletonize(closed)
    min_skeleton = max(8, round(max(int(skel.sum()), 1) * 0.006))
    skel = prune_components(skel, min_skeleton)
    return closed, skel, {
        "support_fraction": float(closed.mean()),
        "skeleton_pixels": float(skel.sum()),
        **color_diagnostics,
    }


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return (0.0, 0.0, float(mask.shape[1] - 1), float(mask.shape[0] - 1))
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def sample_map(score: np.ndarray, x: float, y: float) -> float:
    h, w = score.shape
    return float(score[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))])


def dinov2_score_maps(
    feature_map: np.ndarray, attention_map: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    contrast = g1.feature_local_contrast(feature_map)
    contrast = cv2.resize(rank01(contrast), (size, size), interpolation=cv2.INTER_CUBIC)
    attention = cv2.resize(rank01(attention_map), (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(contrast, 0, 1), np.clip(attention, 0, 1)


def add_candidate(
    candidates: list[dict[str, Any]], x: float, y: float, kind: str, prior: float,
    support: np.ndarray, contrast: np.ndarray, attention: np.ndarray,
) -> None:
    h, w = support.shape
    xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
    if not support[yi, xi]:
        return
    dino = sample_map(contrast, x, y)
    attn = sample_map(attention, x, y)
    score = prior + 0.24 * dino + 0.08 * attn
    candidates.append({"x": x, "y": y, "kind": kind, "score": score, "dino": dino, "attention": attn})


def persistent_junction_centers(junctions: np.ndarray, skel: np.ndarray, bbox_diag: float) -> list[tuple[float, float]]:
    """Keep only junctions with at least three spatially persistent arms."""
    merge_radius = max(2, round(0.012 * bbox_diag))
    merged = cv2.dilate(
        junctions.astype(np.uint8), np.ones((2 * merge_radius + 1, 2 * merge_radius + 1), np.uint8)
    ) > 0
    centers = connected_centroids(merged)
    ys_all, xs_all = np.where(skel)
    if not len(xs_all):
        return []
    results: list[tuple[float, float]] = []
    cut_radius = max(3, round(0.014 * bbox_diag))
    min_arm = max(8, round(0.035 * bbox_diag))
    yy, xx = np.mgrid[: skel.shape[0], : skel.shape[1]]
    for x, y in centers:
        nearest = int(np.argmin((xs_all - x) ** 2 + (ys_all - y) ** 2))
        x0, y0 = float(xs_all[nearest]), float(ys_all[nearest])
        distance2 = (xx - x0) ** 2 + (yy - y0) ** 2
        cut = skel & (distance2 > cut_radius**2)
        components, count = label(cut)
        arms = 0
        annulus = (distance2 > cut_radius**2) & (distance2 <= (cut_radius + 3) ** 2)
        for index in range(1, count + 1):
            component = components == index
            if int(component.sum()) >= min_arm and np.any(component & annulus):
                arms += 1
        if arms >= 3:
            results.append((x0, y0))
    return results


def merged_endpoint_centers(endpoints: np.ndarray, bbox_diag: float) -> list[tuple[float, float]]:
    """Merge adjacent endpoint pixels without merging distinct nearby branches."""
    ys, xs = np.where(endpoints)
    if not len(xs):
        return []
    radius = max(2, round(0.012 * bbox_diag))
    merged = cv2.dilate(
        endpoints.astype(np.uint8),
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
    ) > 0
    original_xy = np.column_stack([xs, ys]).astype(np.float64)
    centers: list[tuple[float, float]] = []
    for x, y in connected_centroids(merged):
        index = int(np.argmin(np.linalg.norm(original_xy - np.asarray([x, y])[None, :], axis=1)))
        centers.append((float(original_xy[index, 0]), float(original_xy[index, 1])))
    return centers


def structural_candidates(
    image_rgb: np.ndarray,
    feature_map: np.ndarray,
    attention_map: np.ndarray,
    max_points: int,
    evidence_mode: str = "full",
    structure_coverage: bool = False,
    basal_transition_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, float]]:
    size = image_rgb.shape[0]
    support, skel, diagnostics = automatic_structural_support(image_rgb)
    contrast, attention = dinov2_score_maps(feature_map, attention_map, size)
    if evidence_mode not in {"full", "geometry_only"}:
        raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
    if evidence_mode == "geometry_only":
        contrast = np.zeros_like(contrast)
        attention = np.zeros_like(attention)
    if not skel.any():
        return np.empty((0, 2)), [], support, skel, diagnostics

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    degree = convolve(skel.astype(np.uint8), neighbor_kernel, mode="constant", cval=0)
    endpoints = skel & (degree == 1)
    junctions = skel & (degree >= 3)

    bbox = bbox_from_mask(support)
    bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    candidates: list[dict[str, Any]] = []
    endpoint_centers = (
        merged_endpoint_centers(endpoints, bbox_diag)
        if structure_coverage
        else connected_centroids(endpoints)
    )
    for x, y in endpoint_centers:
        add_candidate(candidates, x, y, "endpoint", 0.68, support, contrast, attention)
    for x, y in persistent_junction_centers(junctions, skel, bbox_diag):
        add_candidate(candidates, x, y, "junction", 0.76, support, contrast, attention)

    # The phenotype-focused teacher may explicitly require structural coverage
    # at the automatically derived shoot-side transition.  This is not a
    # manually named/located keypoint: the region is computed independently for
    # each plant and the retained point must still reproduce across transforms.
    basal_candidate_added = False
    if structure_coverage and basal_transition_mask is not None:
        transition = np.asarray(basal_transition_mask, dtype=bool)
        if transition.shape != skel.shape:
            raise ValueError("basal transition mask and structural support must have the same shape")
        transition_radius = max(2, round(0.010 * bbox_diag))
        transition_near = cv2.dilate(
            transition.astype(np.uint8),
            np.ones((2 * transition_radius + 1, 2 * transition_radius + 1), np.uint8),
        ) > 0
        ys, xs = np.where(skel & transition_near)
        transition_y, transition_x = np.where(transition)
        if len(xs) and len(transition_x):
            center = np.asarray([float(np.mean(transition_x)), float(np.mean(transition_y))])
            skeleton_xy = np.column_stack([xs, ys]).astype(np.float64)
            x, y = skeleton_xy[int(np.argmin(np.linalg.norm(skeleton_xy - center[None, :], axis=1)))]
            add_candidate(
                candidates,
                float(x),
                float(y),
                "basal_transition",
                0.80,
                support,
                contrast,
                attention,
            )
            basal_candidate_added = True

    min_distance = max(8, round(0.050 * bbox_diag))

    # Shape corners are proposals only.  Their final acceptance still depends
    # on DINOv2 evidence and non-maximum suppression.
    corner_image = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 1.2)
    corners = cv2.goodFeaturesToTrack(
        corner_image,
        maxCorners=max_points * 3,
        qualityLevel=0.045,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is not None:
        for corner in corners[:, 0, :]:
            add_candidate(candidates, float(corner[0]), float(corner[1]), "shape_corner", 0.32, support, contrast, attention)

    # DINOv2-local maxima may recover a visually distinctive point that is not
    # an endpoint/junction.  It must lie close to the skeleton and be in the top
    # 4 percent of within-support feature contrast.
    skeleton_band = cv2.dilate(skel.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if evidence_mode == "full":
        within = contrast[support]
        threshold = float(np.quantile(within, 0.96)) if len(within) else 1.0
        maxima = (contrast >= maximum_filter(contrast, size=max(5, min_distance // 2))) & (contrast >= threshold)
        for y, x in zip(*np.where(maxima & skeleton_band & support)):
            add_candidate(candidates, float(x), float(y), "dino_distinctive", 0.30, support, contrast, attention)

    # Endpoint/junction evidence is structural; other proposals need a minimum
    # fused score.  Point count therefore follows accepted evidence, not K.
    candidates = [
        item for item in candidates
        if item["kind"] in {"endpoint", "junction", "basal_transition"}
        or (
            item["kind"] == "shape_corner"
            and item["score"] >= (0.32 if evidence_mode == "geometry_only" else 0.56)
        )
        or (item["kind"] == "dino_distinctive" and item["score"] >= 0.57)
    ]
    candidates.sort(key=lambda item: (-item["score"], item["kind"], item["y"], item["x"]))
    selected: list[dict[str, Any]] = []
    if structure_coverage:
        # Endpoints are the only image-derived evidence that a terminal branch
        # exists.  The former global NMS could let a nearby main-axis point
        # suppress a short-leaf endpoint.  Preserve distinct skeleton endpoints
        # and the automatic basal transition first; cross-transform consensus
        # remains the noise rejection gate.
        coverage_kinds = {"endpoint", "basal_transition"}
        coverage_candidates = (
            [item for item in candidates if item["kind"] == "endpoint"]
            + [item for item in candidates if item["kind"] == "basal_transition"]
        )
        other_candidates = [item for item in candidates if item["kind"] not in coverage_kinds]
        for item in coverage_candidates + other_candidates:
            item = dict(item)
            if item["kind"] == "endpoint":
                item["coverage_roles"] = ["terminal"]
            elif item["kind"] == "basal_transition":
                item["coverage_roles"] = ["basal_transition"]
            if item["kind"] in coverage_kinds or all(
                (item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2 >= min_distance**2
                for old in selected
            ):
                # The basal proposal may coincide with a structural endpoint.
                # One target is sufficient and avoids duplicate heatmap peaks.
                duplicate_radius = max(2.0, 0.010 * bbox_diag)
                if item["kind"] == "basal_transition":
                    duplicate_index = next(
                        (
                            index
                            for index, old in enumerate(selected)
                            if (item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2
                            < duplicate_radius**2
                        ),
                        None,
                    )
                    if duplicate_index is not None:
                        roles = list(selected[duplicate_index].get("coverage_roles", []))
                        if "basal_transition" not in roles:
                            roles.append("basal_transition")
                        selected[duplicate_index]["coverage_roles"] = roles
                        continue
                selected.append(item)
            if len(selected) >= max_points:
                break
    else:
        for item in candidates:
            if all((item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2 >= min_distance**2 for old in selected):
                selected.append(item)
            if len(selected) >= max_points:
                break
    points = np.asarray([(item["x"], item["y"]) for item in selected], dtype=np.float64).reshape(-1, 2)
    diagnostics.update(
        {
            "raw_candidates": float(len(candidates)),
            "selected_candidates": float(len(selected)),
            "bbox_diagonal": bbox_diag,
            "safety_cap_hit": float(len(selected) >= max_points),
            "evidence_mode_full": float(evidence_mode == "full"),
            "structure_coverage_enabled": float(structure_coverage),
            "identity_endpoint_count": float(int(endpoints.sum())),
            "merged_terminal_candidate_count": float(len(endpoint_centers)),
            "basal_transition_candidate_added": float(basal_candidate_added),
        }
    )
    return points, selected, support, skel, diagnostics


def support_coverage(points: np.ndarray, skeleton: np.ndarray, radius: float) -> float:
    if not skeleton.any():
        return 1.0 if len(points) == 0 else 0.0
    covered = np.zeros_like(skeleton, dtype=np.uint8)
    for x, y in points:
        cv2.circle(covered, (int(round(x)), int(round(y))), max(1, round(radius)), 1, -1)
    return float((covered.astype(bool) & skeleton).sum() / max(int(skeleton.sum()), 1))


def span_ratio(points: np.ndarray, mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2 or len(points) < 2:
        return 0.0
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pixels.mean(axis=0)
    _, _, vh = np.linalg.svd(pixels - center, full_matrices=False)
    axis = vh[0]
    plant_projection = (pixels - center) @ axis
    point_projection = (points - center) @ axis
    return float(np.clip(np.ptp(point_projection) / max(float(np.ptp(plant_projection)), 1e-9), 0.0, 1.5))


def save_overlay(
    path: Path, image: np.ndarray, support: np.ndarray, skeleton: np.ndarray,
    records: Sequence[dict[str, Any]], title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].imshow(image)
    axes[0].imshow(np.ma.masked_where(~support, support), cmap="Greens", alpha=0.28)
    axes[0].imshow(np.ma.masked_where(~skeleton, skeleton), cmap="magma", alpha=0.9)
    axes[0].set_title("automatic support + skeleton")
    axes[1].imshow(image)
    colors = {
        "endpoint": "#00d4ff",
        "junction": "#ff3b30",
        "shape_corner": "#ffd60a",
        "dino_distinctive": "#bf5af2",
        "basal_transition": "#34c759",
    }
    for index, item in enumerate(records, start=1):
        axes[1].scatter(item["x"], item["y"], s=48, c=colors[item["kind"]], edgecolors="black", linewidths=0.7)
        axes[1].text(item["x"] + 4, item["y"] - 4, str(index), fontsize=7, color="black", bbox={"facecolor": "white", "alpha": 0.65, "pad": 0.5})
    axes[1].set_title(f"adaptive candidates (n={len(records)})")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def summarize(per_image: list[dict[str, Any]], detail: list[dict[str, Any]]) -> dict[str, Any]:
    f1_by_image: dict[str, list[float]] = {}
    error_by_image: dict[str, list[float]] = {}
    photo_by_image: dict[str, list[float]] = {}
    for row in detail:
        f1_by_image.setdefault(row["image_id"], []).append(float(row["f1"]))
        if np.isfinite(float(row["localization_error_bbox_diag"])):
            error_by_image.setdefault(row["image_id"], []).append(float(row["localization_error_bbox_diag"]))
        if row["transform_family"] == "photometric":
            photo_by_image.setdefault(row["image_id"], []).append(float(row["count_difference"]))
    counts = np.asarray([row["candidate_count"] for row in per_image], dtype=float)
    summary = {
        "images": len(per_image),
        "median_candidate_count": float(np.median(counts)) if len(counts) else 0.0,
        "candidate_count_iqr": float(np.quantile(counts, 0.75) - np.quantile(counts, 0.25)) if len(counts) else 0.0,
        "noncollapse_ratio": float(np.mean((counts >= 2) & (counts <= 20))) if len(counts) else 0.0,
        "safety_cap_hit_ratio": float(np.mean([row["safety_cap_hit"] for row in per_image])) if per_image else 0.0,
        "median_g0_audit_mask_agreement": float(np.median([row["g0_audit_mask_agreement"] for row in per_image])) if per_image else 0.0,
        "median_repeatability_f1": float(np.median([np.median(values) for values in f1_by_image.values()])) if f1_by_image else 0.0,
        "median_localization_error_bbox_diag": float(np.median([np.median(values) for values in error_by_image.values()])) if error_by_image else float("nan"),
        "median_photometric_count_difference": float(np.median([np.median(values) for values in photo_by_image.values()])) if photo_by_image else float("nan"),
        "median_skeleton_coverage": float(np.median([row["skeleton_coverage"] for row in per_image])) if per_image else 0.0,
        "median_longitudinal_span_ratio": float(np.median([row["longitudinal_span_ratio"] for row in per_image])) if per_image else 0.0,
    }
    gates = {
        "noncollapse_ratio_gte_0_83": summary["noncollapse_ratio"] >= 0.83,
        "repeatability_f1_gte_0_60": summary["median_repeatability_f1"] >= 0.60,
        "localization_error_lte_0_05": summary["median_localization_error_bbox_diag"] <= 0.05,
        "photometric_count_difference_lte_1": summary["median_photometric_count_difference"] <= 1.0,
        "safety_cap_hit_ratio_eq_0": summary["safety_cap_hit_ratio"] == 0.0,
        "skeleton_coverage_gte_0_55": summary["median_skeleton_coverage"] >= 0.55,
        "longitudinal_span_gte_0_65": summary["median_longitudinal_span_ratio"] >= 0.65,
    }
    return {
        "metrics": summary,
        "machine_gates": gates,
        "pending_independent_gates": {
            "support_accuracy": "not_available; G0 audit mask is not independent and leaked into gray backgrounds",
            "human_structural_meaning": "not_run because a machine gate already failed",
        },
        "decision": "advance_to_calibration20" if all(gates.values()) else "revise_or_stop_before_calibration20",
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    g1.set_deterministic(args.seed)
    if args.size % g1.PATCH_SIZE:
        raise ValueError("Input size must be divisible by DINOv2 patch size 14")
    rows = g1.read_csv(args.g0_root / "pilot80_manifest.csv")
    calibration, _ = g1.partition_pilot(rows, args.seed)
    rows = calibration[: args.limit]
    output = args.output / "smoke6"
    output.mkdir(parents=True, exist_ok=True)

    device = g1.resolve_device(args.device)
    model_args = argparse.Namespace(local_repo=args.local_repo, model=args.model, weights=args.weights)
    model = g1.load_official_model(model_args, device)
    per_image: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    attention_modes: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        base, mapping = g1.letterbox_rgb(Path(row["crop_path"]), args.size)
        audit_mask, audit_bbox = g1.square_plant_mask(row, args.size, mapping)
        transforms = g1.make_transforms(base, args.full_transforms)
        outputs: list[dict[str, Any]] = []
        for transform in transforms:
            reps, attention, attention_mode = g1.extract_representations(model, transform["image"], device)
            attention_modes[attention_mode] += 1
            points, records, support, skel, diagnostics = structural_candidates(
                transform["image"], reps["last4avg"], attention, args.max_points
            )
            outputs.append({"points": points, "records": records, "support": support, "skeleton": skel, "diagnostics": diagnostics})
        reference = outputs[0]
        bbox_diag = max(1.0, math.hypot(audit_bbox[2] - audit_bbox[0], audit_bbox[3] - audit_bbox[1]))
        coverage_radius = 0.08 * bbox_diag
        for item in reference["records"]:
            type_counts[item["kind"]] += 1
        for transform, current in zip(transforms[1:], outputs[1:]):
            mapped = g1.apply_inverse(current["points"], transform["matrix"], args.size)
            matched = g1.match_points(reference["points"], mapped, 0.05 * bbox_diag)
            detail.append(
                {
                    "image_id": row["image_id"],
                    "transform": transform["name"],
                    "transform_family": transform["family"],
                    "base_count": len(reference["points"]),
                    "transformed_count_after_inverse": len(mapped),
                    "count_difference": abs(len(reference["points"]) - len(mapped)),
                    "matches": int(matched["matches"]),
                    "precision": matched["precision"],
                    "recall": matched["recall"],
                    "f1": matched["f1"],
                    "localization_error_bbox_diag": matched["median_error"] / bbox_diag if np.isfinite(matched["median_error"]) else float("nan"),
                }
            )
        per_image.append(
            {
                "image_id": row["image_id"],
                "source_name": row["source_name"],
                "candidate_count": len(reference["points"]),
                "candidate_types": ";".join(item["kind"] for item in reference["records"]),
                "safety_cap_hit": int(reference["diagnostics"].get("safety_cap_hit", 0)),
                "g0_audit_mask_agreement": g1.plant_hit_ratio(reference["points"], audit_mask),
                "support_fraction": reference["diagnostics"].get("support_fraction", 0.0),
                "skeleton_pixels": int(reference["diagnostics"].get("skeleton_pixels", 0)),
                "skeleton_coverage": support_coverage(reference["points"], reference["skeleton"], coverage_radius),
                "longitudinal_span_ratio": span_ratio(reference["points"], audit_mask),
            }
        )
        save_overlay(
            output / "overlays" / f"{row['image_id']}.png",
            base,
            reference["support"],
            reference["skeleton"],
            reference["records"],
            f"{row['image_id']} | G1-prime frozen inference",
        )
        print(f"[{row_index}/{len(rows)}] {row['image_id']} points={len(reference['points'])}", flush=True)

    result = summarize(per_image, detail)
    result.update(
        {
            "method": "automatic vegetation support + skeleton topology + frozen DINOv2 evidence",
            "training_used": False,
            "legacy_keypoint_labels_read": False,
            "device": str(device),
            "input_size": args.size,
            "images": [row["image_id"] for row in rows],
            "transforms": [item["name"] for item in transforms],
            "candidate_type_counts": dict(type_counts),
            "attention_modes": dict(attention_modes),
            "elapsed_seconds": time.time() - started,
            "audit_note": "G0 mask agreement is retained only for traceability. It is not an independent accuracy measure: visual review found that the recall-oriented G0 audit proposal leaks into low-saturation gray backgrounds. The mask is never read by candidate generation.",
        }
    )
    write_csv(output / "per_image_metrics.csv", per_image)
    write_csv(output / "per_transform_metrics.csv", detail)
    (output / "smoke_decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python
"""Connect G1-prime candidates into shoot paths and compute proxy phenotypes.

This stage is label-free.  It does not claim biological correctness: it tests
whether the current image-derived points can be ordered on a skeleton graph,
reconstructed by adaptive splines, and converted into auditable phenotype
measurements.  Pixel and bbox-normalized values are emitted; physical units
require a scale reference that is not present in the current images.
"""

from __future__ import annotations

import argparse
import heapq
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
import pandas as pd
import torch
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v3")
    parser.add_argument("--output", type=Path, default=HERE / "outputs_g1_prime_v3" / "phenotype_smoke6")
    parser.add_argument("--local-repo", type=Path, default=HERE / "third_party" / "dinov2_git")
    parser.add_argument(
        "--weights", type=Path, default=HERE / "third_party" / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth"
    )
    parser.add_argument("--model", default="dinov2_vits14_reg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--splits", nargs="+", default=[])
    parser.add_argument("--max-points", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_smoke(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.copy()
    selected: list[pd.DataFrame] = []
    known_stages = ["lt1_needle", "leaf1_to_2", "gt2"]
    per_stage = max(1, limit // len(known_stages))
    for stage in known_stages:
        subset = frame[(frame["stage_label"] == stage) & (frame["primary_evaluation_eligible"] == 1)]
        subset = subset.sort_values(["split", "dataset_id"])
        selected.append(subset.head(per_stage))
    result = pd.concat(selected, ignore_index=True).drop_duplicates("dataset_id")
    if len(result) < limit:
        remaining = frame[~frame["dataset_id"].isin(result["dataset_id"])].sort_values("dataset_id")
        result = pd.concat([result, remaining.head(limit - len(result))], ignore_index=True)
    return result.head(limit)


def skeleton_graph(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    ys, xs = np.where(skeleton)
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    lookup = np.full(skeleton.shape, -1, dtype=np.int32)
    lookup[ys, xs] = np.arange(len(xs), dtype=np.int32)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    h, w = skeleton.shape
    for index, (x_float, y_float) in enumerate(coords):
        x, y = int(x_float), int(y_float)
        for dx, dy in offsets:
            xx, yy = x + dx, y + dy
            if 0 <= xx < w and 0 <= yy < h:
                other = int(lookup[yy, xx])
                if other >= 0:
                    adjacency[index].append((other, math.sqrt(2.0) if dx and dy else 1.0))
    return coords, adjacency


def dijkstra(adjacency: Sequence[Sequence[tuple[int, float]]], source: int) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    parent = np.full(len(adjacency), -1, dtype=np.int32)
    distance[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        for other, weight in adjacency[node]:
            proposed = current_distance + weight
            if proposed + 1e-12 < distance[other]:
                distance[other] = proposed
                parent[other] = node
                heapq.heappush(queue, (proposed, other))
    return distance, parent


def endpoint_width_score(
    endpoint: int, coords: np.ndarray, adjacency: Sequence[Sequence[tuple[int, float]]], width_map: np.ndarray
) -> float:
    values: list[float] = []
    previous = -1
    current = endpoint
    for _ in range(36):
        x, y = coords[current].astype(int)
        values.append(float(width_map[y, x]))
        choices = [node for node, _ in adjacency[current] if node != previous]
        if len(choices) != 1:
            break
        previous, current = current, choices[0]
    return float(np.median(values)) if values else 0.0


def terminal_edge_length(endpoint: int, adjacency: Sequence[Sequence[tuple[int, float]]]) -> float:
    length = 0.0
    previous = -1
    current = endpoint
    for _ in range(len(adjacency)):
        choices = [(node, weight) for node, weight in adjacency[current] if node != previous]
        if len(choices) != 1:
            break
        next_node, weight = choices[0]
        length += float(weight)
        previous, current = current, next_node
    return length


def choose_base(
    coords: np.ndarray,
    adjacency: Sequence[Sequence[tuple[int, float]]],
    support: np.ndarray,
    image_rgb: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    endpoints = [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    if not endpoints:
        endpoints = [0]
    width_map = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    widths = np.asarray([endpoint_width_score(node, coords, adjacency, width_map) for node in endpoints])
    terminal_lengths = np.asarray([terminal_edge_length(node, adjacency) for node in endpoints], dtype=np.float64)
    basal_colors: list[float] = []
    radius = max(3, round(0.012 * math.hypot(support.shape[0], support.shape[1])))
    for endpoint in endpoints:
        x, y = coords[endpoint].astype(int)
        y0, y1 = max(0, y - radius), min(image_rgb.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image_rgb.shape[1], x + radius + 1)
        patch = image_rgb[y0:y1, x0:x1].astype(np.float32)
        patch_support = support[y0:y1, x0:x1]
        pixels = patch[patch_support] if np.any(patch_support) else patch.reshape(-1, 3)
        chroma = pixels.max(axis=1) - pixels.min(axis=1)
        colored_pixels = pixels[chroma >= 25.0]
        if len(colored_pixels) >= 5:
            pixels = colored_pixels
        red, green, blue = np.median(pixels, axis=0)
        excess_green = 2.0 * green - red - blue
        darkness = 255.0 - float(np.mean([red, green, blue]))
        # Basal/cut stem tissue tends to be less green and slightly darker than
        # leaf tips.  This remains a soft cue and is never a coordinate rule.
        basal_colors.append(float(-excess_green + 0.12 * darkness + 0.10 * (red - green)))
    basal_colors_array = np.asarray(basal_colors)
    eccentricities = []
    for endpoint in endpoints:
        distance, _ = dijkstra(adjacency, endpoint)
        finite = distance[np.isfinite(distance)]
        eccentricities.append(float(finite.max()) if len(finite) else 0.0)
    eccentricities_array = np.asarray(eccentricities)

    def unit(values: np.ndarray) -> np.ndarray:
        spread = float(values.max() - values.min()) if len(values) else 0.0
        return (values - values.min()) / spread if spread > 1e-9 else np.ones_like(values)

    scores = (
        0.55 * unit(basal_colors_array)
        + 0.05 * (1.0 - unit(terminal_lengths))
        + 0.15 * unit(widths)
        + 0.25 * unit(eccentricities_array)
    )
    order = np.argsort(-scores)
    chosen_position = int(order[0])
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
    endpoint_rows = [
        {
            "node": int(node),
            "x": float(coords[node, 0]),
            "y": float(coords[node, 1]),
            "local_half_width": float(widths[position]),
            "terminal_edge_length": float(terminal_lengths[position]),
            "basal_color_score": float(basal_colors_array[position]),
            "eccentricity": float(eccentricities_array[position]),
            "base_score": float(scores[position]),
        }
        for position, node in enumerate(endpoints)
    ]
    return endpoints[chosen_position], {"base_score_margin": margin, "endpoint_scores": endpoint_rows}


def path_from_parent(parent: np.ndarray, node: int, root: int) -> list[int]:
    path = [int(node)]
    seen = {int(node)}
    while path[-1] != root and parent[path[-1]] >= 0:
        next_node = int(parent[path[-1]])
        if next_node in seen:
            break
        seen.add(next_node)
        path.append(next_node)
    path.reverse()
    return path


def smooth_polyline(points: np.ndarray) -> np.ndarray:
    if len(points) < 5:
        return points.copy()
    window = min(11, len(points) if len(points) % 2 else len(points) - 1)
    window = max(3, window)
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    smooth = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    )
    smooth[0], smooth[-1] = points[0], points[-1]
    return smooth


def rdp_indices(points: np.ndarray, epsilon: float) -> list[int]:
    if len(points) <= 2:
        return list(range(len(points)))
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        a, b = points[start], points[end]
        segment = b - a
        denominator = float(np.dot(segment, segment))
        inner = points[start + 1 : end]
        if not len(inner):
            continue
        if denominator <= 1e-12:
            distances = np.linalg.norm(inner - a, axis=1)
        else:
            t = np.clip(((inner - a) @ segment) / denominator, 0.0, 1.0)
            projection = a + t[:, None] * segment
            distances = np.linalg.norm(inner - projection, axis=1)
        offset = int(np.argmax(distances))
        if float(distances[offset]) > epsilon:
            index = start + 1 + offset
            keep.add(index)
            stack.extend([(start, index), (index, end)])
    return sorted(keep)


def adaptive_support(
    path_points: np.ndarray, candidates: Sequence[dict[str, Any]], bbox_diag: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    smooth = smooth_polyline(path_points)
    index_kinds: dict[int, set[str]] = {index: {"rdp_geometry"} for index in rdp_indices(smooth, max(1.5, 0.006 * bbox_diag))}
    if len(smooth):
        tree = cKDTree(smooth)
        for candidate in candidates:
            distance, index = tree.query([candidate["x"], candidate["y"]], k=1)
            if float(distance) <= 0.025 * bbox_diag:
                index_kinds.setdefault(int(index), set()).add(str(candidate["kind"]))
    ordered_indices = sorted(index_kinds)
    # Do not retain two support samples that are only a few skeleton pixels
    # apart unless one is an endpoint.  This is a geometric NMS, not fixed K.
    merged: list[int] = []
    for index in ordered_indices:
        if not merged or index - merged[-1] >= 3 or index in {0, len(smooth) - 1}:
            merged.append(index)
        else:
            index_kinds[merged[-1]].update(index_kinds[index])
    records = [
        {
            "path_index": int(index),
            "x": float(smooth[index, 0]),
            "y": float(smooth[index, 1]),
            "evidence": sorted(index_kinds[index]),
        }
        for index in merged
    ]
    return smooth[np.asarray(merged, dtype=int)], records


def spline_curve(support_points: np.ndarray, samples: int = 240) -> np.ndarray:
    if len(support_points) < 2:
        return support_points.copy()
    differences = np.linalg.norm(np.diff(support_points, axis=0), axis=1)
    keep = np.concatenate([[True], differences > 1e-6])
    points = support_points[keep]
    if len(points) < 2:
        return points
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if cumulative[-1] <= 1e-9:
        return points
    u = cumulative / cumulative[-1]
    try:
        k = min(3, len(points) - 1)
        tck, _ = splprep([points[:, 0], points[:, 1]], u=u, s=max(0.0, 0.12 * len(points)), k=k)
        evaluated = splev(np.linspace(0.0, 1.0, samples), tck)
        return np.column_stack(evaluated)
    except (TypeError, ValueError):
        target = np.linspace(0.0, cumulative[-1], samples)
        return np.column_stack([np.interp(target, cumulative, points[:, axis]) for axis in range(2)])


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) >= 2 else 0.0


def curve_metrics(curve: np.ndarray) -> dict[str, float]:
    length = polyline_length(curve)
    chord = float(np.linalg.norm(curve[-1] - curve[0])) if len(curve) >= 2 else 0.0
    if len(curve) >= 3:
        first = np.diff(curve, axis=0)
        norms = np.linalg.norm(first, axis=1)
        valid = (norms[:-1] > 1e-9) & (norms[1:] > 1e-9)
        cosine = np.ones(len(first) - 1)
        cosine[valid] = np.sum(first[:-1][valid] * first[1:][valid], axis=1) / (norms[:-1][valid] * norms[1:][valid])
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        total_turn = float(np.sum(np.abs(angles)))
    else:
        total_turn = 0.0
    return {
        "spline_length_px": length,
        "chord_length_px": chord,
        "sinuosity": length / max(chord, 1e-9),
        "total_turning_angle_deg": math.degrees(total_turn),
        "mean_abs_curvature_per_px": total_turn / max(length, 1e-9),
    }


def divergence_angle(root_path: np.ndarray, branch_path: np.ndarray) -> float:
    if len(root_path) < 2 or len(branch_path) < 2:
        return float("nan")
    trunk_start = root_path[max(0, len(root_path) - 12)]
    trunk_vector = root_path[-1] - trunk_start
    branch_end = branch_path[min(len(branch_path) - 1, 11)]
    branch_vector = branch_end - branch_path[0]
    denominator = np.linalg.norm(trunk_vector) * np.linalg.norm(branch_vector)
    if denominator <= 1e-9:
        return float("nan")
    cosine = float(np.dot(trunk_vector, branch_vector) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def to_source_xy(point: Sequence[float], mapping: dict[str, float]) -> tuple[float, float]:
    x = (float(point[0]) - mapping["pad_x"]) / mapping["scale"]
    y = (float(point[1]) - mapping["pad_y"]) / mapping["scale"]
    return (
        float(np.clip(x, 0.0, mapping["source_width"] - 1.0)),
        float(np.clip(y, 0.0, mapping["source_height"] - 1.0)),
    )


def extract_paths(
    image_rgb: np.ndarray,
    support: np.ndarray,
    skeleton: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    bbox_diag: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coords, adjacency = skeleton_graph(skeleton)
    if len(coords) < 2:
        return [], {"failure": "skeleton_too_small", "skeleton_nodes": int(len(coords))}
    components = cv2.connectedComponents(skeleton.astype(np.uint8), 8)[0] - 1
    base, base_diagnostics = choose_base(coords, adjacency, support, image_rgb)
    distance, parent = dijkstra(adjacency, base)
    endpoints = [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1 and index != base]
    reachable_endpoints = [index for index in endpoints if np.isfinite(distance[index])]
    if not reachable_endpoints:
        return [], {"failure": "no_reachable_tip", "skeleton_nodes": int(len(coords)), **base_diagnostics}
    main_tip = max(reachable_endpoints, key=lambda index: distance[index])
    junction_nodes = {index for index, neighbors in enumerate(adjacency) if len(neighbors) >= 3}
    paths: list[dict[str, Any]] = []
    minimum_branch_length = max(8.0, 0.04 * bbox_diag)
    for tip in sorted(reachable_endpoints, key=lambda index: distance[index], reverse=True):
        root_nodes = path_from_parent(parent, tip, base)
        if len(root_nodes) < 2:
            continue
        root_points = coords[np.asarray(root_nodes)]
        if tip == main_tip:
            attachment_position = 0
            kind = "main_axis"
        else:
            junction_positions = [position for position, node in enumerate(root_nodes[:-1]) if node in junction_nodes]
            attachment_position = max(junction_positions) if junction_positions else 0
            kind = "lateral_branch"
        branch_points = root_points[attachment_position:]
        branch_length = polyline_length(branch_points)
        if kind == "lateral_branch" and branch_length < minimum_branch_length:
            continue
        support_points, support_records = adaptive_support(branch_points, candidates, bbox_diag)
        curve = spline_curve(support_points)
        reconstruction_error = float(np.median(cKDTree(curve).query(branch_points)[0])) if len(curve) else float("nan")
        metrics = curve_metrics(curve)
        metrics["skeleton_path_length_px"] = branch_length
        metrics["length_normalized_bbox_diag"] = metrics["spline_length_px"] / max(bbox_diag, 1e-9)
        metrics["spline_to_skeleton_median_error_px"] = reconstruction_error
        metrics["divergence_angle_deg"] = 0.0 if kind == "main_axis" else divergence_angle(root_points[: attachment_position + 1], branch_points)
        paths.append(
            {
                "path_id": f"path_{len(paths) + 1:02d}",
                "path_kind": kind,
                "tip_node": int(tip),
                "attachment_xy": branch_points[0].tolist(),
                "tip_xy": branch_points[-1].tolist(),
                "root_path_points": root_points.tolist(),
                "branch_path_points": branch_points.tolist(),
                "support_points": support_points.tolist(),
                "support_records": support_records,
                "spline_curve": curve.tolist(),
                "metrics": metrics,
            }
        )
    diagnostics = {
        "failure": "",
        "skeleton_nodes": int(len(coords)),
        "skeleton_components": int(components),
        "endpoint_count": int(len(endpoints) + 1),
        "junction_pixel_count": int(len(junction_nodes)),
        "base_xy": coords[base].tolist(),
        "main_tip_xy": coords[main_tip].tolist(),
        "path_count": len(paths),
        **base_diagnostics,
    }
    return paths, diagnostics


def save_overlay(
    path: Path,
    image: np.ndarray,
    support: np.ndarray,
    skeleton: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    paths: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
    title: str,
    candidate_label: str = "G1′ candidates",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    axes[0].imshow(image)
    axes[0].imshow(np.ma.masked_where(~support, support), cmap="Greens", alpha=0.22)
    axes[0].imshow(np.ma.masked_where(~skeleton, skeleton), cmap="magma", alpha=0.85)
    for item in candidates:
        axes[0].scatter(item["x"], item["y"], s=34, c="#00b7ff", edgecolors="black", linewidths=0.5)
    axes[0].set_title(f"{candidate_label} n={len(candidates)}")

    axes[1].imshow(image)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(paths))))
    for color, item in zip(colors, paths):
        curve = np.asarray(item["spline_curve"])
        samples = np.asarray(item["support_points"])
        if len(curve):
            axes[1].plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.2)
        if len(samples):
            axes[1].scatter(samples[:, 0], samples[:, 1], s=40, color=color, edgecolors="black", linewidths=0.6)
        attach = np.asarray(item["attachment_xy"])
        axes[1].scatter(attach[0], attach[1], marker="s", s=52, color=color, edgecolors="black")
    if diagnostics.get("base_xy"):
        base = diagnostics["base_xy"]
        axes[1].scatter(base[0], base[1], marker="*", s=130, c="#ff2d55", edgecolors="black")
    axes[1].set_title(f"ordered paths + adaptive spline support n={len(paths)}")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    g1.set_deterministic(args.seed)
    if args.size % g1.PATCH_SIZE:
        raise ValueError("Input size must be divisible by DINOv2 patch size 14")
    frame = pd.read_csv(args.dataset / "manifests" / "all.csv")
    if args.splits:
        frame = frame[frame["split"].isin(args.splits)].copy()
    rows = stratified_smoke(frame, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    device = g1.resolve_device(args.device)
    model_args = argparse.Namespace(local_repo=args.local_repo, model=args.model, weights=args.weights)
    model = g1.load_official_model(model_args, device)

    candidate_rows: list[dict[str, Any]] = []
    phenotype_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows.to_dict("records"), start=1):
        image_path = args.dataset / Path(str(row["relative_path"]).replace("\\", "/"))
        base, mapping = g1.letterbox_rgb(image_path, args.size)
        representations, attention, attention_mode = g1.extract_representations(model, base, device)
        points, candidates, support, skeleton, candidate_diagnostics = gp.structural_candidates(
            base, representations["last4avg"], attention, args.max_points
        )
        bbox = gp.bbox_from_mask(support)
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        paths, graph_diagnostics = extract_paths(base, support, skeleton, candidates, bbox_diag)

        for index, candidate in enumerate(candidates, start=1):
            source_x, source_y = to_source_xy((candidate["x"], candidate["y"]), mapping)
            candidate_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "candidate_id": f"candidate_{index:02d}",
                    "kind": candidate["kind"],
                    "x_model": candidate["x"],
                    "y_model": candidate["y"],
                    "x_source": source_x,
                    "y_source": source_y,
                    "x_source_normalized": source_x / max(mapping["source_width"] - 1.0, 1.0),
                    "y_source_normalized": source_y / max(mapping["source_height"] - 1.0, 1.0),
                    "score": candidate["score"],
                    "dino_local_contrast": candidate["dino"],
                    "dino_cls_attention": candidate["attention"],
                }
            )

        for path_item in paths:
            path_serializable = {"dataset_id": row["dataset_id"], **path_item}
            for key in ("attachment_xy", "tip_xy"):
                source_xy = to_source_xy(path_item[key], mapping)
                path_serializable[f"{key}_source"] = list(source_xy)
            path_rows.append(path_serializable)
            phenotype_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "stage_label": row["stage_label"],
                    "split": row["split"],
                    "path_id": path_item["path_id"],
                    "path_kind": path_item["path_kind"],
                    "adaptive_support_count": len(path_item["support_points"]),
                    **path_item["metrics"],
                    "physical_unit_status": "not_available_no_scale_reference",
                }
            )

        image_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "source_name": row["source_name"],
                "stage_label": row["stage_label"],
                "split": row["split"],
                "candidate_count": len(points),
                "candidate_type_counts": json.dumps(Counter(item["kind"] for item in candidates), ensure_ascii=False),
                "path_count": len(paths),
                "support_fraction": float(support.mean()),
                "skeleton_pixels": int(skeleton.sum()),
                "base_score_margin": graph_diagnostics.get("base_score_margin", float("nan")),
                "endpoint_base_scores": json.dumps(graph_diagnostics.get("endpoint_scores", []), ensure_ascii=False),
                "skeleton_components": graph_diagnostics.get("skeleton_components", 0),
                "graph_failure": graph_diagnostics.get("failure", ""),
                "attention_mode": attention_mode,
                "safety_cap_hit": int(candidate_diagnostics.get("safety_cap_hit", 0)),
            }
        )
        save_overlay(
            args.output / "overlays" / f"{row['dataset_id']}.png",
            base,
            support,
            skeleton,
            candidates,
            paths,
            graph_diagnostics,
            f"{row['dataset_id']} | {row['stage_label']} | phenotype bridge smoke",
        )
        print(
            f"[{row_number}/{len(rows)}] {row['dataset_id']} candidates={len(points)} paths={len(paths)}",
            flush=True,
        )

    pd.DataFrame(candidate_rows).to_csv(args.output / "candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phenotype_rows).to_csv(args.output / "phenotypes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_rows).to_csv(args.output / "per_image.csv", index=False, encoding="utf-8-sig")
    write_jsonl(args.output / "paths.jsonl", path_rows)
    summary = {
        "method": "G1-prime candidates -> rooted skeleton graph -> adaptive RDP support -> spline phenotypes",
        "images": len(image_rows),
        "stages": dict(Counter(row["stage_label"] for row in image_rows)),
        "training_used": False,
        "keypoint_labels_read": False,
        "device": str(device),
        "median_candidate_count": float(np.median([row["candidate_count"] for row in image_rows])) if image_rows else 0.0,
        "median_path_count": float(np.median([row["path_count"] for row in image_rows])) if image_rows else 0.0,
        "connected_graph_success_rate": float(np.mean([not row["graph_failure"] for row in image_rows])) if image_rows else 0.0,
        "safety_cap_hit_rate": float(np.mean([row["safety_cap_hit"] for row in image_rows])) if image_rows else 0.0,
        "physical_units_available": False,
        "interpretation_gate": "visual path correctness and manual phenotype reference are still required",
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

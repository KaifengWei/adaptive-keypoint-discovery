#!/usr/bin/env python
"""Val-only evaluation for keypoint-conditioned structural graph reconstruction.

This script reuses saved learned-point predictions and never runs the detector.
The skeleton supplies geodesic routing only; it cannot create graph nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_phenotype_bridge as bridge  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
from point_conditioned_graph import build_point_conditioned_graph  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402


DEFAULT_RATIOS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v4_fullplant_candidate")
    parser.add_argument(
        "--evaluation", type=Path, default=HERE / "evaluation_outputs" / "core_dinov2_v4_fullplant_val"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "evaluation_outputs" / "point_conditioned_graph_v1_val"
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--render-ratio", type=float, default=-1.0)
    parser.add_argument(
        "--reuse-sweep",
        action="store_true",
        help="Keep an existing full sweep CSV and evaluate only --render-ratio for detailed outputs.",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--input-domain",
        choices=["auto", "full_plant", "phenotype_roi_v1"],
        default="auto",
    )
    return parser.parse_args()


def load_mask_canvas(path: Path, mapping: dict[str, float], size: int) -> np.ndarray:
    source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise FileNotFoundError(path)
    new_w = max(1, round(source.shape[1] * mapping["scale"]))
    new_h = max(1, round(source.shape[0] * mapping["scale"]))
    resized = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    canvas = np.zeros((size, size), dtype=bool)
    pad_x, pad_y = int(mapping["pad_x"]), int(mapping["pad_y"])
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def model_points(point_frame: pd.DataFrame, mapping: dict[str, float]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in point_frame.sort_values("point_id").to_dict("records"):
        records.append(
            {
                "point_id": str(row["point_id"]),
                "x": float(row["x_source"]) * mapping["scale"] + mapping["pad_x"],
                "y": float(row["y_source"]) * mapping["scale"] + mapping["pad_y"],
                "score": float(row["confidence"]),
                "kind": "learned_heatmap",
            }
        )
    return records


def reference_events(skeleton: np.ndarray, bbox_diag: float) -> tuple[np.ndarray, np.ndarray]:
    coords, adjacency = bridge.skeleton_graph(skeleton)
    if len(coords) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    endpoints = coords[np.asarray([len(neighbors) == 1 for neighbors in adjacency], dtype=bool)]
    junction_mask = np.zeros_like(skeleton, dtype=bool)
    for node, neighbors in enumerate(adjacency):
        if len(neighbors) >= 3:
            x, y = coords[node].astype(int)
            junction_mask[y, x] = True
    junctions = np.asarray(gp.persistent_junction_centers(junction_mask, skeleton, bbox_diag), dtype=np.float64)
    if junctions.size == 0:
        junctions = np.empty((0, 2), dtype=np.float64)
    return endpoints.reshape(-1, 2), junctions.reshape(-1, 2)


def event_recall(events: np.ndarray, nodes: list[dict[str, Any]], tolerance: float) -> float:
    if len(events) == 0:
        return float("nan")
    if not nodes:
        return 0.0
    node_xy = np.asarray([node["projected_xy"] for node in nodes], dtype=np.float64)
    return float(np.mean(cKDTree(node_xy).query(events, k=1)[0] <= tolerance))


def region_at(x: float, y: float, masks: dict[str, np.ndarray]) -> str:
    height, width = masks["full_plant"].shape
    xx, yy = int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))
    shoot = bool(masks["shoot"][yy, xx])
    root = bool(masks["seed_base_root"][yy, xx])
    full = bool(masks["full_plant"][yy, xx])
    if shoot and root:
        return "shoot_root_overlap"
    if shoot:
        return "shoot"
    if root:
        return "seed_base_root"
    if full:
        return "full_plant_other"
    return "outside_saved_full_plant"


def annotate_regions(
    graph: dict[str, Any], masks_exact: dict[str, np.ndarray], masks_tolerant: dict[str, np.ndarray]
) -> None:
    for node in graph["nodes"]:
        node["organ_region_exact"] = region_at(*node["projected_xy"], masks_exact)
        node["organ_region_tolerant"] = region_at(*node["projected_xy"], masks_tolerant)
    for edge in graph["edges"]:
        regions = [region_at(float(x), float(y), masks_tolerant) for x, y in edge["path_xy"]]
        counts = Counter(regions)
        edge["dominant_organ_region"] = counts.most_common(1)[0][0] if counts else "none"
        edge["organ_region_fractions"] = {
            key: value / max(len(regions), 1) for key, value in sorted(counts.items())
        }


def graph_coverage(graph: dict[str, Any], skeleton: np.ndarray) -> float:
    denominator = int(skeleton.sum())
    return float(graph["edge_union"].sum() / denominator) if denominator else 0.0


def leave_one_out(
    skeleton: np.ndarray,
    points: list[dict[str, Any]],
    graph: dict[str, Any],
    bbox_diag: float,
    ratio: float,
) -> list[dict[str, Any]]:
    full_coverage = graph_coverage(graph, skeleton)
    rows: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        removed_index = int(node["input_index"])
        reduced = [item for index, item in enumerate(points) if index != removed_index]
        reduced_graph = build_point_conditioned_graph(skeleton, reduced, bbox_diag, ratio)
        reduced_coverage = graph_coverage(reduced_graph, skeleton)
        rows.append(
            {
                "node_id": int(node["node_id"]),
                "point_id": str(node.get("point_id", f"input_{removed_index}")),
                "coverage_with_all": full_coverage,
                "coverage_without_point": reduced_coverage,
                "coverage_drop": full_coverage - reduced_coverage,
                "edge_count_without_point": len(reduced_graph["edges"]),
            }
        )
    return rows


def evaluate_one(
    dataset_id: str,
    manifest_row: dict[str, Any],
    image_points: pd.DataFrame,
    dataset: Path,
    image_size: int,
    ratio: float,
    input_domain: str = "full_plant",
) -> dict[str, Any]:
    image_path = dataset / Path(str(manifest_row["relative_path"]).replace("\\", "/"))
    roi_result: dict[str, Any] | None = None
    if input_domain == "phenotype_roi_v1":
        image, mapping, _, roi_result = load_phenotype_input(dataset, manifest_row, image_size)
    else:
        image, mapping = g1.letterbox_rgb(image_path, image_size)
    support, raw_skeleton, support_diagnostics = gp.automatic_structural_support(image)
    bbox = gp.bbox_from_mask(support)
    bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    topology_skeleton, spur_diagnostics = bridge.prune_short_terminal_spurs(
        raw_skeleton, max(4.0, 0.012 * bbox_diag)
    )
    points = model_points(image_points, mapping)
    graph = build_point_conditioned_graph(topology_skeleton, points, bbox_diag, ratio)
    masks_exact = {
        "shoot": load_mask_canvas(
            dataset / Path(str(manifest_row["shoot_mask_relative_path"]).replace("\\", "/")), mapping, image_size
        ),
        "seed_base_root": load_mask_canvas(
            dataset / Path(str(manifest_row["seed_base_root_mask_relative_path"]).replace("\\", "/")), mapping, image_size
        ),
        "full_plant": load_mask_canvas(
            dataset / Path(str(manifest_row["full_plant_mask_relative_path"]).replace("\\", "/")), mapping, image_size
        ),
    }
    if roi_result is not None:
        masks_exact["phenotype_roi"] = roi_result["phenotype_roi_model"]
        masks_exact["basal_transition"] = roi_result["basal_transition_model"]
    else:
        masks_exact["phenotype_roi"] = masks_exact["shoot"].copy()
        masks_exact["basal_transition"] = np.zeros_like(masks_exact["shoot"], dtype=bool)
    radius = max(2, round(image_size * 0.01))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    masks_tolerant = {
        name: cv2.dilate(mask.astype(np.uint8), kernel) > 0 for name, mask in masks_exact.items()
    }
    annotate_regions(graph, masks_exact, masks_tolerant)
    endpoints, junctions = reference_events(topology_skeleton, bbox_diag)
    tolerance = 0.025 * bbox_diag
    loo = leave_one_out(topology_skeleton, points, graph, bbox_diag, ratio)
    accepted_distances = [float(node["projection_distance_bbox_diag"]) for node in graph["nodes"]]
    positive_drops = [float(item["coverage_drop"]) for item in loo]
    return {
        "dataset_id": dataset_id,
        "image": image,
        "support": support,
        "skeleton": topology_skeleton,
        "masks_exact": masks_exact,
        "masks_tolerant": masks_tolerant,
        "points": points,
        "graph": graph,
        "leave_one_out": loo,
        "metrics": {
            "dataset_id": dataset_id,
            "split": "val",
            "input_domain": input_domain,
            "projection_ratio": ratio,
            "input_point_count": len(points),
            "accepted_node_count": len(graph["nodes"]),
            "accepted_node_ratio": len(graph["nodes"]) / max(len(points), 1),
            "projection_rejected_count": int(graph["diagnostics"]["rejected_point_count"]),
            "merged_duplicate_count": int(graph["diagnostics"]["merged_duplicate_count"]),
            "edge_count": len(graph["edges"]),
            "learned_node_components": int(graph["diagnostics"]["learned_node_components"]),
            "graph_failure": str(graph["diagnostics"]["failure"]),
            "skeleton_pixel_count": int(topology_skeleton.sum()),
            "edge_union_pixel_count": int(graph["edge_union"].sum()),
            "skeleton_coverage_ratio": graph_coverage(graph, topology_skeleton),
            "endpoint_count": len(endpoints),
            "endpoint_node_recall": event_recall(endpoints, graph["nodes"], tolerance),
            "junction_count": len(junctions),
            "junction_node_recall": event_recall(junctions, graph["nodes"], tolerance),
            "median_projection_distance_bbox_diag": float(np.median(accepted_distances)) if accepted_distances else float("nan"),
            "maximum_projection_distance_bbox_diag": float(np.max(accepted_distances)) if accepted_distances else float("nan"),
            "causal_node_fraction": float(np.mean(np.asarray(positive_drops) > 1e-9)) if positive_drops else 0.0,
            "mean_leave_one_out_coverage_drop": float(np.mean(positive_drops)) if positive_drops else 0.0,
            "maximum_leave_one_out_coverage_drop": float(np.max(positive_drops)) if positive_drops else 0.0,
            "zero_point_control_coverage": 0.0,
            "accepted_nodes_shoot_tolerant": sum(
                node["organ_region_tolerant"] == "shoot" for node in graph["nodes"]
            ),
            "accepted_nodes_root_base_tolerant": sum(
                node["organ_region_tolerant"] == "seed_base_root" for node in graph["nodes"]
            ),
            "accepted_nodes_outside_fullplant_tolerant": sum(
                node["organ_region_tolerant"] == "outside_saved_full_plant" for node in graph["nodes"]
            ),
            "support_mask_mode_uniform": float(support_diagnostics.get("mask_mode_uniform_background", 0.0)),
            **spur_diagnostics,
        },
    }


def aggregate(results: list[dict[str, Any]], ratio: float) -> dict[str, Any]:
    rows = pd.DataFrame([item["metrics"] for item in results])

    def median(column: str) -> float | None:
        values = rows[column].dropna()
        return float(values.median()) if len(values) else None

    total_input = int(rows["input_point_count"].sum())
    total_accepted = int(rows["accepted_node_count"].sum())
    return {
        "projection_ratio": ratio,
        "images": len(rows),
        "total_input_points": total_input,
        "total_accepted_nodes": total_accepted,
        "global_accepted_node_ratio": total_accepted / max(total_input, 1),
        "median_accepted_node_count": median("accepted_node_count"),
        "median_projection_distance_bbox_diag": median("median_projection_distance_bbox_diag"),
        "median_skeleton_coverage_ratio": median("skeleton_coverage_ratio"),
        "mean_skeleton_coverage_ratio": float(rows["skeleton_coverage_ratio"].mean()),
        "median_endpoint_node_recall": median("endpoint_node_recall"),
        "median_junction_node_recall": median("junction_node_recall"),
        "images_with_persistent_junction_reference": int((rows["junction_count"] > 0).sum()),
        "median_causal_node_fraction": median("causal_node_fraction"),
        "mean_leave_one_out_coverage_drop": float(rows["mean_leave_one_out_coverage_drop"].mean()),
        "graph_success_rate": float((rows["graph_failure"] == "").mean()),
        "zero_point_control_coverage": 0.0,
    }


def save_overlay(path: Path, result: dict[str, Any]) -> None:
    graph = result["graph"]
    image = result["image"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=135)
    axes[0].imshow(image)
    for point in graph["rejected_points"]:
        if point.get("rejection_reason") == "projection_too_far":
            axes[0].scatter(point["x"], point["y"], c="#ef4444", marker="x", s=58, linewidths=1.8)
    for node in graph["nodes"]:
        original = np.asarray(node["original_xy"])
        projected = np.asarray(node["projected_xy"])
        axes[0].plot([original[0], projected[0]], [original[1], projected[1]], color="#f59e0b", linewidth=1.1)
        axes[0].scatter(original[0], original[1], c="#38bdf8", s=30, edgecolors="black", linewidths=0.5)
        axes[0].scatter(projected[0], projected[1], c="#22c55e", s=35, edgecolors="black", linewidths=0.5)
    axes[0].set_title("learned points → accepted projection")

    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(~result["skeleton"], result["skeleton"]), cmap="gray", alpha=0.45)
    colors = {
        "shoot": "#16a34a",
        "seed_base_root": "#f97316",
        "shoot_root_overlap": "#a855f7",
        "full_plant_other": "#64748b",
        "outside_saved_full_plant": "#ef4444",
    }
    for edge in graph["edges"]:
        xy = np.asarray(edge["path_xy"])
        axes[1].plot(xy[:, 0], xy[:, 1], color="#2563eb", linewidth=2.1)
    for node in graph["nodes"]:
        x, y = node["projected_xy"]
        axes[1].scatter(x, y, c=colors[node["organ_region_tolerant"]], s=42, edgecolors="black", linewidths=0.6)
        axes[1].text(x + 3, y - 3, str(node["node_id"]), fontsize=7, color="black")
    metrics = result["metrics"]
    axes[1].set_title(
        f"point-conditioned edges={metrics['edge_count']} | skeleton coverage={metrics['skeleton_coverage_ratio']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{result['dataset_id']} | ratio={metrics['projection_ratio']:.3f}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def make_contact_sheets(overlays: list[Path], output: Path, columns: int = 4, rows: int = 2) -> None:
    output.mkdir(parents=True, exist_ok=True)
    page_size = columns * rows
    for page, start in enumerate(range(0, len(overlays), page_size), start=1):
        selected = overlays[start : start + page_size]
        thumbs: list[tuple[Image.Image, str]] = []
        for path in selected:
            with Image.open(path) as opened:
                item = opened.convert("RGB")
                item.thumbnail((520, 270))
                thumbs.append((item.copy(), path.stem))
        canvas = Image.new("RGB", (columns * 540, rows * 310), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (thumb, label) in enumerate(thumbs):
            left = (index % columns) * 540 + 10
            top = (index // columns) * 310 + 25
            canvas.paste(thumb, (left, top))
            draw.text((left, 5 + (index // columns) * 310), label, fill="black")
        canvas.save(output / f"point_conditioned_graph_v1_val_{page:02d}.jpg", quality=90)


def flatten_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes, edges, loo = [], [], []
    for result in results:
        dataset_id = result["dataset_id"]
        for node in result["graph"]["nodes"]:
            nodes.append({"dataset_id": dataset_id, **{key: value for key, value in node.items() if key not in {"original_xy", "projected_xy"}}, "original_x": node["original_xy"][0], "original_y": node["original_xy"][1], "projected_x": node["projected_xy"][0], "projected_y": node["projected_xy"][1]})
        for edge_id, edge in enumerate(result["graph"]["edges"], start=1):
            edges.append({"dataset_id": dataset_id, "edge_id": edge_id, "source_node": edge["source_node"], "target_node": edge["target_node"], "geodesic_length_px": edge["geodesic_length_px"], "skeleton_component": edge["skeleton_component"], "dominant_organ_region": edge["dominant_organ_region"], "organ_region_fractions": json.dumps(edge["organ_region_fractions"], ensure_ascii=False), "path_xy": json.dumps(edge["path_xy"], ensure_ascii=False)})
        loo.extend({"dataset_id": dataset_id, **row} for row in result["leave_one_out"])
    return nodes, edges, loo


def run(args: argparse.Namespace) -> None:
    per_image = pd.read_csv(args.evaluation / "per_image.csv")
    if "split" not in per_image or set(per_image["split"].astype(str)) != {"val"}:
        if not args.allow_test:
            raise RuntimeError("Refusing non-val evaluation input without --allow-test")
    points = pd.read_csv(args.evaluation / "points.csv")
    saved_domains = set(per_image.get("input_domain", pd.Series(["full_plant"])).astype(str))
    if len(saved_domains) != 1:
        raise RuntimeError(f"Saved predictions contain mixed input domains: {sorted(saved_domains)}")
    saved_domain = next(iter(saved_domains))
    input_domain = saved_domain if args.input_domain == "auto" else args.input_domain
    if input_domain != saved_domain:
        raise RuntimeError(f"Graph input domain {input_domain} does not match saved predictions {saved_domain}")
    manifest = pd.read_csv(args.dataset / "manifests" / "val.csv", low_memory=False)
    manifest = manifest[manifest["dataset_id"].isin(per_image["dataset_id"])].sort_values("dataset_id")
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    expected = set(manifest["dataset_id"].astype(str))
    if len(expected) != len(manifest) or expected != set(per_image[per_image["dataset_id"].isin(expected)]["dataset_id"].astype(str)):
        raise RuntimeError("Saved val predictions and V4 val manifest do not match")
    args.output.mkdir(parents=True, exist_ok=True)

    cache: dict[float, list[dict[str, Any]]] = {}
    if args.reuse_sweep:
        if args.render_ratio < 0:
            raise RuntimeError("--reuse-sweep requires --render-ratio")
        if not (args.output / "projection_threshold_sweep.csv").exists():
            raise RuntimeError("--reuse-sweep requested but projection_threshold_sweep.csv is missing")
    else:
        sweep_rows: list[dict[str, Any]] = []
        for ratio in sorted(set(float(value) for value in args.ratios)):
            results = []
            for row in manifest.to_dict("records"):
                dataset_id = str(row["dataset_id"])
                image_points = points[points["dataset_id"] == dataset_id]
                results.append(evaluate_one(dataset_id, row, image_points, args.dataset, args.image_size, ratio, input_domain))
            cache[ratio] = results
            sweep_rows.append(aggregate(results, ratio))
            print(json.dumps(sweep_rows[-1], ensure_ascii=False), flush=True)
        pd.DataFrame(sweep_rows).to_csv(args.output / "projection_threshold_sweep.csv", index=False, encoding="utf-8-sig")

    if args.render_ratio < 0:
        return
    ratio = float(args.render_ratio)
    if ratio not in cache:
        results = [
            evaluate_one(str(row["dataset_id"]), row, points[points["dataset_id"] == row["dataset_id"]], args.dataset, args.image_size, ratio, input_domain)
            for row in manifest.to_dict("records")
        ]
    else:
        results = cache[ratio]
    per_image_rows = [item["metrics"] for item in results]
    pd.DataFrame(per_image_rows).to_csv(args.output / "per_image.csv", index=False, encoding="utf-8-sig")
    failure_priority = pd.DataFrame(per_image_rows).sort_values(
        ["skeleton_coverage_ratio", "endpoint_node_recall", "causal_node_fraction"],
        ascending=[True, True, True],
    )
    failure_priority.to_csv(args.output / "failure_priority.csv", index=False, encoding="utf-8-sig")
    manual_review = failure_priority[
        [
            "dataset_id",
            "input_point_count",
            "accepted_node_count",
            "edge_count",
            "skeleton_coverage_ratio",
            "endpoint_node_recall",
            "causal_node_fraction",
        ]
    ].copy()
    manual_review["manual_path_semantics"] = "pending"
    manual_review["manual_missing_organ"] = ""
    manual_review["manual_wrong_connection"] = ""
    manual_review["manual_note"] = ""
    manual_review.to_csv(args.output / "manual_path_review_pending.csv", index=False, encoding="utf-8-sig")
    node_rows, edge_rows, loo_rows = flatten_rows(results)
    pd.DataFrame(node_rows).to_csv(args.output / "nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(edge_rows).to_csv(args.output / "edges.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loo_rows).to_csv(args.output / "leave_one_out.csv", index=False, encoding="utf-8-sig")
    overlays = []
    for result in results:
        path = args.output / "overlays" / f"{result['dataset_id']}.png"
        save_overlay(path, result)
        overlays.append(path)
    make_contact_sheets(overlays, args.output / "contact_sheets")
    summary = {
        **aggregate(results, ratio),
        "dataset": "V4 full-plant candidate",
        "input_domain": input_domain,
        "split": "val",
        "test_images_read": 0,
        "learned_predictions_rerun": False,
        "manual_keypoint_labels_used": False,
        "graph_nodes_source": "accepted projections of learned dynamic-heatmap points only",
        "skeleton_role": "geodesic routing substrate only; no endpoint or junction auto-completion",
        "phenotype_accuracy_status": "pending_manual_path_and_measurement_reference",
        "persistent_junction_reference_status": (
            "available"
            if any(item["metrics"]["junction_count"] > 0 for item in results)
            else "not_evaluable_no_persistent_junctions_in_automatic_val_skeletons"
        ),
        "images_below_0_90_skeleton_coverage": sum(
            item["metrics"]["skeleton_coverage_ratio"] < 0.90 for item in results
        ),
        "minimum_skeleton_coverage_ratio": min(
            item["metrics"]["skeleton_coverage_ratio"] for item in results
        ),
        "positive_leave_one_out_nodes": sum(
            row["coverage_drop"] > 1e-9 for item in results for row in item["leave_one_out"]
        ),
        "zero_leave_one_out_nodes": sum(
            abs(row["coverage_drop"]) <= 1e-9 for item in results for row in item["leave_one_out"]
        ),
        "accepted_node_region_tolerant_counts": dict(
            Counter(
                node["organ_region_tolerant"]
                for item in results
                for node in item["graph"]["nodes"]
            )
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

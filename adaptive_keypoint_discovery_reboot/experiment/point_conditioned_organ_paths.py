#!/usr/bin/env python
"""Decode candidate shoot paths from a learned-point-conditioned graph.

Only learned graph nodes may serve as the basal anchor or terminal tips.  The
edge union supplies ordered geometry after those learned nodes have already
determined which structure exists.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

import g1_prime_phenotype_bridge as bridge


def _interface_mask(shoot: np.ndarray, root_base: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    shoot_near = cv2.dilate(shoot.astype(np.uint8), kernel) > 0
    root_near = cv2.dilate(root_base.astype(np.uint8), kernel) > 0
    return shoot_near & root_near


def _nearest_union_nodes(graph: dict[str, Any], union_coords: np.ndarray) -> dict[int, int]:
    tree = cKDTree(union_coords)
    return {
        int(node["node_id"]): int(tree.query(np.asarray(node["projected_xy"], dtype=np.float64), k=1)[1])
        for node in graph["nodes"]
    }


def _choose_learned_base(
    graph: dict[str, Any], shoot: np.ndarray, root_base: np.ndarray, bbox_diag: float
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    radius = max(2, round(0.012 * bbox_diag))
    interface = _interface_mask(shoot, root_base, radius)
    ys, xs = np.where(interface)
    if not len(xs):
        return None, {
            "failure": "no_shoot_root_interface",
            "interface_pixel_count": 0,
            "interface_radius_px": radius,
        }
    if not graph["nodes"]:
        return None, {
            "failure": "no_learned_nodes",
            "interface_pixel_count": int(len(xs)),
            "interface_radius_px": radius,
        }
    tree = cKDTree(np.column_stack([xs, ys]).astype(np.float64))
    ranked = []
    for node in graph["nodes"]:
        distance = float(tree.query(np.asarray(node["projected_xy"], dtype=np.float64), k=1)[0])
        region = str(node.get("organ_region_tolerant", ""))
        region_priority = 0 if region == "shoot_root_overlap" else 1 if region == "seed_base_root" else 2
        ranked.append((region_priority, distance, -float(node.get("score", 0.0)), node))
    _, distance, _, chosen = min(ranked, key=lambda item: item[:3])
    return chosen, {
        "failure": "",
        "interface_pixel_count": int(len(xs)),
        "interface_radius_px": radius,
        "base_interface_distance_px": float(distance),
        "base_interface_distance_bbox_diag": float(distance) / max(bbox_diag, 1.0),
        "base_selection_rule": "learned node ranked by interface-compatible region then interface distance",
    }


def _path_to_source(parent: np.ndarray, node: int, source: int) -> list[int]:
    path = bridge.path_from_parent(parent, node, source)
    return path if path and path[0] == source else []


def _candidate_record(node: dict[str, Any]) -> dict[str, Any]:
    x, y = node["projected_xy"]
    return {
        "x": float(x),
        "y": float(y),
        "kind": "learned_heatmap_projected",
        "score": float(node.get("score", 0.0)),
    }


def _make_path_record(
    path_id: str,
    path_kind: str,
    base_node_id: int,
    tip_node_id: int,
    attachment_node: int,
    full_path_nodes: list[int],
    branch_nodes: list[int],
    union_coords: np.ndarray,
    graph: dict[str, Any],
    graph_to_union: dict[int, int],
    bbox_diag: float,
) -> dict[str, Any]:
    full_points = union_coords[np.asarray(full_path_nodes, dtype=np.int32)]
    branch_points = union_coords[np.asarray(branch_nodes, dtype=np.int32)]
    branch_node_set = set(branch_nodes)
    candidates = [
        _candidate_record(node)
        for node in graph["nodes"]
        if graph_to_union[int(node["node_id"])] in branch_node_set
    ]
    support_points, support_records = bridge.adaptive_support(branch_points, candidates, bbox_diag)
    curve = bridge.spline_curve(support_points)
    reconstruction_error = (
        float(np.median(cKDTree(curve).query(branch_points, k=1)[0])) if len(curve) else float("nan")
    )
    metrics = bridge.curve_metrics(curve)
    metrics.update(
        {
            "skeleton_path_length_px": bridge.polyline_length(branch_points),
            "length_normalized_bbox_diag": metrics["spline_length_px"] / max(bbox_diag, 1.0),
            "spline_to_skeleton_median_error_px": reconstruction_error,
            "divergence_angle_deg": (
                0.0
                if path_kind == "main_axis"
                else bridge.divergence_angle(
                    full_points[: max(1, full_path_nodes.index(attachment_node) + 1)], branch_points
                )
            ),
        }
    )
    return {
        "path_id": path_id,
        "path_kind": path_kind,
        "base_node_id": int(base_node_id),
        "tip_node_id": int(tip_node_id),
        "attachment_xy": union_coords[attachment_node].tolist(),
        "tip_xy": union_coords[branch_nodes[-1]].tolist(),
        "full_base_to_tip_path": full_points.tolist(),
        "branch_path_points": branch_points.tolist(),
        "support_points": support_points.tolist(),
        "support_records": support_records,
        "spline_curve": curve.tolist(),
        "metrics": metrics,
    }


def decode_candidate_organ_paths(
    graph: dict[str, Any],
    shoot_mask: np.ndarray,
    root_base_mask: np.ndarray,
    bbox_diag: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode a rooted main path and dynamic lateral terminal branches."""
    edge_union = np.asarray(graph["edge_union"], dtype=bool)
    union_coords, union_adjacency = bridge.skeleton_graph(edge_union)
    if len(union_coords) < 2 or not graph["edges"]:
        return [], {
            "failure": "point_conditioned_edge_union_too_small",
            "path_count": 0,
            "edge_union_pixels": int(edge_union.sum()),
        }
    base, base_diagnostics = _choose_learned_base(graph, shoot_mask, root_base_mask, bbox_diag)
    if base is None:
        return [], {**base_diagnostics, "path_count": 0, "edge_union_pixels": int(edge_union.sum())}

    graph_to_union = _nearest_union_nodes(graph, union_coords)
    base_node_id = int(base["node_id"])
    base_union_node = graph_to_union[base_node_id]
    distances, parents = bridge.dijkstra(union_adjacency, base_union_node)
    terminal_nodes = [
        node
        for node in graph["nodes"]
        if int(node["node_id"]) != base_node_id
        # Abstract MST degree is not a spatial terminal test: a shortest edge
        # between two learned points may pass through a third learned tip.  The
        # edge-union endpoint remains learned-point-conditioned and preserves
        # the actual terminal geometry without consulting unused skeleton ends.
        and len(union_adjacency[graph_to_union[int(node["node_id"])]]) == 1
        and str(node.get("organ_region_tolerant", "")) in {"shoot", "shoot_root_overlap"}
        and np.isfinite(distances[graph_to_union[int(node["node_id"])]] )
    ]
    if not terminal_nodes:
        return [], {
            **base_diagnostics,
            "failure": "no_reachable_shoot_terminal_learned_node",
            "base_node_id": base_node_id,
            "terminal_learned_node_count": 0,
            "path_count": 0,
            "edge_union_pixels": int(edge_union.sum()),
        }

    terminal_nodes.sort(
        key=lambda node: float(distances[graph_to_union[int(node["node_id"])]]), reverse=True
    )
    base_paths: dict[int, list[int]] = {}
    for terminal in terminal_nodes:
        node_id = int(terminal["node_id"])
        base_paths[node_id] = _path_to_source(parents, graph_to_union[node_id], base_union_node)
    terminal_nodes = [node for node in terminal_nodes if len(base_paths[int(node["node_id"])]) >= 2]
    if not terminal_nodes:
        return [], {
            **base_diagnostics,
            "failure": "no_valid_base_to_terminal_path",
            "base_node_id": base_node_id,
            "terminal_learned_node_count": 0,
            "path_count": 0,
            "edge_union_pixels": int(edge_union.sum()),
        }

    main = terminal_nodes[0]
    main_id = int(main["node_id"])
    main_nodes = base_paths[main_id]
    paths = [
        _make_path_record(
            "path_01",
            "main_axis",
            base_node_id,
            main_id,
            main_nodes[0],
            main_nodes,
            main_nodes,
            union_coords,
            graph,
            graph_to_union,
            bbox_diag,
        )
    ]
    recovered_tree_nodes = set(main_nodes)
    minimum_branch_length = max(8.0, 0.04 * bbox_diag)
    short_terminal_rejected_count = 0
    for terminal in terminal_nodes[1:]:
        tip_id = int(terminal["node_id"])
        full_nodes = base_paths[tip_id]
        shared_positions = [index for index, node in enumerate(full_nodes[:-1]) if node in recovered_tree_nodes]
        attachment_position = max(shared_positions) if shared_positions else 0
        branch_nodes = full_nodes[attachment_position:]
        if len(branch_nodes) < 2:
            short_terminal_rejected_count += 1
            continue
        branch_points = union_coords[np.asarray(branch_nodes, dtype=np.int32)]
        if bridge.polyline_length(branch_points) < minimum_branch_length:
            short_terminal_rejected_count += 1
            continue
        paths.append(
            _make_path_record(
                f"path_{len(paths) + 1:02d}",
                "lateral_branch",
                base_node_id,
                tip_id,
                branch_nodes[0],
                full_nodes,
                branch_nodes,
                union_coords,
                graph,
                graph_to_union,
                bbox_diag,
            )
        )
        recovered_tree_nodes.update(full_nodes)

    recovered_pixels = np.zeros_like(edge_union, dtype=bool)
    for path in paths:
        xy = np.asarray(path["branch_path_points"], dtype=np.float64).astype(int)
        recovered_pixels[xy[:, 1], xy[:, 0]] = True
    return paths, {
        **base_diagnostics,
        "failure": "",
        "base_node_id": base_node_id,
        "base_xy": base["projected_xy"],
        "terminal_learned_node_count": len(terminal_nodes),
        "decoded_terminal_path_count": len(paths),
        "short_terminal_rejected_count": short_terminal_rejected_count,
        "minimum_lateral_branch_length_px": minimum_branch_length,
        "path_count": len(paths),
        "main_tip_node_id": main_id,
        "edge_union_pixels": int(edge_union.sum()),
        "decoded_branch_union_pixels": int(recovered_pixels.sum()),
        "decoded_to_edge_union_ratio": float(recovered_pixels.sum() / max(int(edge_union.sum()), 1)),
        "unreached_learned_node_count": sum(
            not np.isfinite(distances[graph_to_union[int(node["node_id"])]]) for node in graph["nodes"]
        ),
    }

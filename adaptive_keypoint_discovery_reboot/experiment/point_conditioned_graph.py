#!/usr/bin/env python
"""Build a structural graph whose nodes come only from learned keypoints.

The plant skeleton is used as a routing substrate, not as an independent node
generator.  Consequently, fewer than two accepted learned points produce no
edge, and removing a learned point can change the recovered structure.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

import g1_prime_phenotype_bridge as bridge


def _components(adjacency: Sequence[Sequence[tuple[int, float]]]) -> np.ndarray:
    labels = np.full(len(adjacency), -1, dtype=np.int32)
    component = 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        labels[start] = component
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for other, _ in adjacency[node]:
                if labels[other] < 0:
                    labels[other] = component
                    queue.append(other)
        component += 1
    return labels


def _minimum_spanning_geodesics(
    node_indices: list[int],
    coords: np.ndarray,
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> list[dict[str, Any]]:
    """Return an MST on learned nodes, with edge geometry routed on skeleton."""
    if len(node_indices) < 2:
        return []
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for left_position, left_skeleton_node in enumerate(node_indices[:-1]):
        distances, parents = bridge.dijkstra(adjacency, left_skeleton_node)
        for right_position in range(left_position + 1, len(node_indices)):
            right_skeleton_node = node_indices[right_position]
            distance = float(distances[right_skeleton_node])
            if not np.isfinite(distance):
                continue
            path_nodes = bridge.path_from_parent(parents, right_skeleton_node, left_skeleton_node)
            if len(path_nodes) < 2 or path_nodes[0] != left_skeleton_node:
                continue
            candidates.append((distance, left_position, right_position, np.asarray(path_nodes, dtype=np.int32)))

    parent = list(range(len(node_indices)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    edges: list[dict[str, Any]] = []
    for distance, left, right, path_nodes in sorted(candidates, key=lambda item: item[0]):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        edges.append(
            {
                "source_node": int(left),
                "target_node": int(right),
                "geodesic_length_px": distance,
                "skeleton_node_path": path_nodes.tolist(),
                "path_xy": coords[path_nodes].tolist(),
            }
        )
        if len(edges) == len(node_indices) - 1:
            break
    return edges


def build_point_conditioned_graph(
    skeleton: np.ndarray,
    learned_points: Sequence[dict[str, Any]],
    bbox_diag: float,
    maximum_projection_ratio: float,
) -> dict[str, Any]:
    """Project learned points and connect only the accepted projections.

    Points farther than ``maximum_projection_ratio * bbox_diag`` are rejected.
    Several predictions projected to the same skeleton pixel are merged by
    confidence.  No skeleton endpoint or junction is inserted as a graph node.
    """
    coords, adjacency = bridge.skeleton_graph(skeleton.astype(bool))
    edge_union = np.zeros_like(skeleton, dtype=bool)
    maximum_distance = float(maximum_projection_ratio * max(float(bbox_diag), 1.0))
    if len(coords) == 0:
        return {
            "nodes": [],
            "rejected_points": [dict(point, rejection_reason="empty_skeleton") for point in learned_points],
            "edges": [],
            "edge_union": edge_union,
            "diagnostics": {
                "failure": "empty_skeleton",
                "input_point_count": len(learned_points),
                "accepted_node_count": 0,
                "rejected_point_count": len(learned_points),
                "merged_duplicate_count": 0,
                "edge_count": 0,
                "learned_node_components": 0,
                "maximum_projection_distance_px": maximum_distance,
            },
        }

    component_labels = _components(adjacency)
    tree = cKDTree(coords)
    projected_by_skeleton_node: dict[int, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    merged_duplicate_count = 0
    for input_index, point in enumerate(learned_points):
        xy = np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
        distance, skeleton_node = tree.query(xy, k=1)
        skeleton_node = int(skeleton_node)
        projection = {
            **dict(point),
            "input_index": input_index,
            "original_xy": xy.tolist(),
            "projection_distance_px": float(distance),
            "projection_distance_bbox_diag": float(distance) / max(float(bbox_diag), 1.0),
            "skeleton_node": skeleton_node,
            "projected_xy": coords[skeleton_node].tolist(),
            "skeleton_component": int(component_labels[skeleton_node]),
        }
        if float(distance) > maximum_distance:
            rejected.append({**projection, "rejection_reason": "projection_too_far"})
            continue
        previous = projected_by_skeleton_node.get(skeleton_node)
        if previous is None:
            projected_by_skeleton_node[skeleton_node] = projection
            continue
        merged_duplicate_count += 1
        previous_score = float(previous.get("score", previous.get("confidence", 0.0)))
        current_score = float(projection.get("score", projection.get("confidence", 0.0)))
        if current_score > previous_score:
            rejected.append({**previous, "rejection_reason": "duplicate_projection_merged"})
            projected_by_skeleton_node[skeleton_node] = projection
        else:
            rejected.append({**projection, "rejection_reason": "duplicate_projection_merged"})

    nodes = sorted(projected_by_skeleton_node.values(), key=lambda item: int(item["input_index"]))
    for node_id, node in enumerate(nodes):
        node["node_id"] = node_id

    edges: list[dict[str, Any]] = []
    component_ids = sorted({int(node["skeleton_component"]) for node in nodes})
    for component_id in component_ids:
        local_nodes = [node for node in nodes if int(node["skeleton_component"]) == component_id]
        local_skeleton_nodes = [int(node["skeleton_node"]) for node in local_nodes]
        local_edges = _minimum_spanning_geodesics(local_skeleton_nodes, coords, adjacency)
        for edge in local_edges:
            edge["source_node"] = int(local_nodes[int(edge["source_node"])]["node_id"])
            edge["target_node"] = int(local_nodes[int(edge["target_node"])]["node_id"])
            edge["skeleton_component"] = component_id
            path_nodes = np.asarray(edge["skeleton_node_path"], dtype=np.int32)
            xy = coords[path_nodes].astype(int)
            edge_union[xy[:, 1], xy[:, 0]] = True
            edges.append(edge)

    failure = ""
    if len(nodes) < 2:
        failure = "fewer_than_two_accepted_nodes"
    elif not edges:
        failure = "no_geodesic_edge"
    return {
        "nodes": nodes,
        "rejected_points": rejected,
        "edges": edges,
        "edge_union": edge_union,
        "diagnostics": {
            "failure": failure,
            "input_point_count": len(learned_points),
            "accepted_node_count": len(nodes),
            "rejected_point_count": sum(item["rejection_reason"] == "projection_too_far" for item in rejected),
            "merged_duplicate_count": merged_duplicate_count,
            "edge_count": len(edges),
            "learned_node_components": len(component_ids),
            "maximum_projection_distance_px": maximum_distance,
        },
    }

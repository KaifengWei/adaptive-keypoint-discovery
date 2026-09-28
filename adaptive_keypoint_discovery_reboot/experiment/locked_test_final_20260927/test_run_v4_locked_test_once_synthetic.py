"""Synthetic-only checks for the GT-blind once-only prediction producer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v4_once", HERE / "run_v4_locked_test_once.py")
assert spec is not None and spec.loader is not None
once = importlib.util.module_from_spec(spec)
spec.loader.exec_module(once)


def check_serialization() -> None:
    record = {"dataset_id": "synthetic_01", "output_sha256": "0" * 64, "source_frame_id": "synthetic_frame"}
    sample = {"mapping": {"source_width": 200.0, "source_height": 100.0}}
    point = {"point_id": "p01", "x": 40.0, "y": 40.0, "score": 0.7, "kind": "candidate"}
    node = {"node_id": 0, "projected_xy": [41.0, 39.0], "organ_region_tolerant": "seed_base_root"}
    graph = {"diagnostics": {"failure": ""}, "nodes": [node], "rejected_points": []}
    zero = once.prediction_row("Student-B", record, sample, [point], [], {}, graph)
    assert zero["decoder_status"] == "zero_path"
    assert zero["underground_association_count"] == 1
    assert zero["base_xy_model_canvas"] is None
    assert json.loads(once.json_bytes(zero))["width"] == 200
    path = {
        "path_id": "path_01", "full_base_to_tip_path": [[41.0, 39.0], [80.0, 39.0]],
        "path_kind": "main_axis", "base_node_id": 0, "tip_node_id": 1,
        "support_records": [],
    }
    graph["nodes"].append({"node_id": 1, "projected_xy": [80.0, 39.0], "organ_region_tolerant": "shoot"})
    full = once.prediction_row("Student-B", record, sample, [point], [path], {"base_node_id": 0}, graph)
    assert full["decoder_status"] == "ok"
    assert full["base_xy_model_canvas"] == [41.0, 39.0]
    assert len(full["paths"]) == 1
    once.json_bytes(full)


def check_shared_pipeline_parity() -> None:
    import numpy as np
    sys.path.insert(0, str(once.EXPERIMENT / "method_gate_20260925"))
    import run_frozen_diagnostics as frozen

    image = np.full((518, 518, 3), 255, dtype=np.uint8)
    image[255:263, 60:470] = [75, 130, 45]
    shoot = np.zeros((518, 518), dtype=bool)
    shoot[254:264, 60:470] = True
    base = np.zeros_like(shoot)
    base[253:265, 453:475] = True
    masks = {
        "shoot": shoot, "seed_base_root": base, "full_plant": shoot | base,
        "phenotype_roi": shoot | base, "basal_transition": base,
    }
    points = [
        {"point_id": "p01", "x": 460.0, "y": 259.0, "score": 0.9, "kind": "candidate"},
        {"point_id": "p02", "x": 65.0, "y": 259.0, "score": 0.8, "kind": "candidate"},
    ]
    original_paths, original_decision, original_diag = frozen.graph_decoder(image, masks, points)
    paths, decision, diag, graph = once.graph_decoder_once(image, masks, points, frozen)
    assert once.json_bytes(paths) == once.json_bytes(original_paths)
    assert once.json_bytes(decision) == once.json_bytes(original_decision)
    assert diag == original_diag
    assert len(graph["nodes"]) + len(graph["rejected_points"]) >= 2


if __name__ == "__main__":
    check_serialization()
    check_shared_pipeline_parity()
    print("synthetic producer checks: PASS; no test images or GT read")

#!/usr/bin/env python
"""Re-evaluate topology from saved detector points without re-running a model.

This keeps the learned detector output fixed while testing downstream skeleton
cleanup, path extraction, spline reconstruction, and phenotype proxies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import g1_dinov2_feasibility as g1
import g1_prime_phenotype_bridge as bridge
import g1_prime_structural_support as gp

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v3")
    parser.add_argument(
        "--points",
        type=Path,
        default=HERE / "evaluation_outputs" / "core_dinov2" / "points.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "evaluation_outputs" / "topology_reuse_val_v2",
    )
    parser.add_argument("--splits", nargs="+", default=["val"])
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.dataset / "manifests" / "all.csv")
    frame = manifest[manifest["split"].isin(args.splits)].copy()
    saved_points = pd.read_csv(args.points)
    args.output.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    phenotype_rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        image_path = args.dataset / Path(str(row["relative_path"]).replace("\\", "/"))
        image, mapping = g1.letterbox_rgb(image_path, 518)
        current = saved_points[saved_points["dataset_id"] == row["dataset_id"]]
        records = [
            {
                "x": float(item["x_source"]) * mapping["scale"] + mapping["pad_x"],
                "y": float(item["y_source"]) * mapping["scale"] + mapping["pad_y"],
                "kind": "learned_heatmap",
                "score": float(item["confidence"]),
                "dino": float("nan"),
                "attention": float("nan"),
            }
            for item in current.to_dict("records")
        ]
        support, skeleton, _ = gp.automatic_structural_support(image)
        bbox = gp.bbox_from_mask(support)
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        paths, diagnostics = bridge.extract_paths(image, support, skeleton, records, bbox_diag)

        for item in paths:
            path_rows.append({"dataset_id": row["dataset_id"], **item})
            phenotype_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "stage_label": row["stage_label"],
                    "split": row["split"],
                    "path_id": item["path_id"],
                    "path_kind": item["path_kind"],
                    "adaptive_support_count": len(item["support_points"]),
                    **item["metrics"],
                    "physical_unit_status": "not_available_no_scale_reference",
                }
            )
        base_xy = diagnostics.get("base_xy", [float("nan"), float("nan")])
        image_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "stage_label": row["stage_label"],
                "split": row["split"],
                "point_count": len(records),
                "path_count": len(paths),
                "graph_failure": diagnostics.get("failure", ""),
                "endpoint_count": diagnostics.get("endpoint_count", 0),
                "base_x": base_xy[0],
                "base_y": base_xy[1],
                "base_score_margin": diagnostics.get("base_score_margin", float("nan")),
                "spur_prune_removed_pixels": diagnostics.get("spur_prune_removed_pixels", 0),
            }
        )
        bridge.save_overlay(
            args.output / "overlays" / f"{row['dataset_id']}.png",
            image,
            support,
            skeleton,
            records,
            paths,
            diagnostics,
            f"{row['dataset_id']} | fixed detector points, topology v2",
            candidate_label="saved learned adaptive points",
        )

    image_frame = pd.DataFrame(image_rows)
    phenotype_frame = pd.DataFrame(phenotype_rows)
    image_frame.to_csv(args.output / "per_image.csv", index=False, encoding="utf-8-sig")
    phenotype_frame.to_csv(args.output / "phenotypes.csv", index=False, encoding="utf-8-sig")
    bridge.write_jsonl(args.output / "paths.jsonl", path_rows)
    ratios = (
        phenotype_frame["spline_length_px"] / phenotype_frame["skeleton_path_length_px"]
        if len(phenotype_frame)
        else pd.Series(dtype=float)
    )
    summary = {
        "images": int(len(image_frame)),
        "splits": list(args.splits),
        "detector_predictions_reused": True,
        "median_path_count": float(image_frame["path_count"].median()) if len(image_frame) else 0.0,
        "graph_success_rate": float((image_frame["graph_failure"] == "").mean()) if len(image_frame) else 0.0,
        "total_spur_pixels_removed": int(image_frame["spur_prune_removed_pixels"].sum()) if len(image_frame) else 0,
        "maximum_spline_to_skeleton_length_ratio": float(ratios.max()) if len(ratios) else float("nan"),
        "median_spline_to_skeleton_error_px": (
            float(phenotype_frame["spline_to_skeleton_median_error_px"].median())
            if len(phenotype_frame)
            else float("nan")
        ),
        "manual_keypoint_labels_used": False,
        "phenotype_accuracy_status": "pending_manual_measurement_reference",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

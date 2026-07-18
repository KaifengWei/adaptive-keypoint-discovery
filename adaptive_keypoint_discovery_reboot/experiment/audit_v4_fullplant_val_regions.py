#!/usr/bin/env python
"""Audit learned V4 points against shoot, root/base, and whole-plant masks.

This is a post-hoc diagnostic for the full-plant dataset.  It does not replace
the existing foreground metric and does not run the detector or read keypoint
labels.  The same tolerance used by ``plant_hit_ratio`` is applied to each
saved automatic organ mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v4_fullplant_candidate")
    parser.add_argument("--evaluation", type=Path, default=HERE / "evaluation_outputs" / "core_dinov2_v4_fullplant_val")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def dataset_path(dataset: Path, row: pd.Series, relative_column: str) -> Path:
    value = str(row[relative_column]).replace("\\", "/")
    return dataset / Path(value)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def dilate(mask: np.ndarray) -> np.ndarray:
    # The original metric is measured on a square letterboxed model canvas.
    # Mapping its 1% tolerance back to the source image is approximately 1%
    # of the source long side, not the short side (important for wide scans).
    radius = max(2, round(max(mask.shape) * 0.01))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def point_hits(points: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    if points.empty:
        return np.zeros(0, dtype=bool)
    height, width = mask.shape
    xs = np.clip(np.rint(points["x_source"].to_numpy(float)).astype(int), 0, width - 1)
    ys = np.clip(np.rint(points["y_source"].to_numpy(float)).astype(int), 0, height - 1)
    return mask[ys, xs]


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.dataset / "manifests" / "all.csv", low_memory=False).set_index("dataset_id")
    per_image = pd.read_csv(args.evaluation / "per_image.csv")
    points = pd.read_csv(args.evaluation / "points.csv")
    if "test" in set(per_image["split"].astype(str)) and not args.allow_test:
        raise RuntimeError("Refusing to audit test predictions without --allow-test")

    missing = sorted(set(per_image["dataset_id"]) - set(manifest.index))
    if missing:
        raise RuntimeError(f"Evaluation IDs missing from manifest: {missing[:5]}")

    rows: list[dict[str, object]] = []
    for image_row in per_image.to_dict("records"):
        dataset_id = str(image_row["dataset_id"])
        manifest_row = manifest.loc[dataset_id]
        image_points = points[points["dataset_id"] == dataset_id]
        shoot = dilate(load_mask(dataset_path(args.dataset, manifest_row, "shoot_mask_relative_path")))
        root_base = dilate(load_mask(dataset_path(args.dataset, manifest_row, "seed_base_root_mask_relative_path")))
        full = dilate(load_mask(dataset_path(args.dataset, manifest_row, "full_plant_mask_relative_path")))
        if not (shoot.shape == root_base.shape == full.shape):
            raise RuntimeError(f"Mask shape mismatch for {dataset_id}")

        shoot_hit = point_hits(image_points, shoot)
        root_hit = point_hits(image_points, root_base)
        full_hit = point_hits(image_points, full)
        total = len(image_points)
        root_only = root_hit & ~shoot_hit
        full_other_only = full_hit & ~shoot_hit & ~root_hit
        background = ~full_hit
        rows.append(
            {
                "dataset_id": dataset_id,
                "split": image_row["split"],
                "point_count": total,
                "shoot_hit_count": int(shoot_hit.sum()),
                "root_base_hit_count": int(root_hit.sum()),
                "root_base_only_count": int(root_only.sum()),
                "full_other_only_count": int(full_other_only.sum()),
                "full_plant_hit_count": int(full_hit.sum()),
                "background_miss_count": int(background.sum()),
                "shoot_hit_ratio": ratio(int(shoot_hit.sum()), total),
                "root_base_only_ratio": ratio(int(root_only.sum()), total),
                "full_plant_hit_ratio": ratio(int(full_hit.sum()), total),
                "background_miss_ratio": ratio(int(background.sum()), total),
                "legacy_auto_support_hit_ratio": float(image_row["foreground_hit_ratio"]),
            }
        )

    result = pd.DataFrame(rows).sort_values("dataset_id")
    result.to_csv(args.evaluation / "region_hit_per_image.csv", index=False, encoding="utf-8-sig")
    total_points = int(result["point_count"].sum())
    summary = {
        "images": len(result),
        "splits": sorted(result["split"].astype(str).unique().tolist()),
        "points": total_points,
        "median_legacy_auto_support_hit_ratio": float(result["legacy_auto_support_hit_ratio"].median()),
        "median_shoot_mask_hit_ratio": float(result["shoot_hit_ratio"].median()),
        "median_full_plant_mask_hit_ratio": float(result["full_plant_hit_ratio"].median()),
        "total_root_base_only_ratio": ratio(int(result["root_base_only_count"].sum()), total_points),
        "total_full_plant_hit_ratio": ratio(int(result["full_plant_hit_count"].sum()), total_points),
        "total_background_miss_ratio": ratio(int(result["background_miss_count"].sum()), total_points),
        "images_with_background_miss": int((result["background_miss_count"] > 0).sum()),
        "worst_full_plant_hit_ids": result.sort_values(
            ["full_plant_hit_ratio", "background_miss_count"], ascending=[True, False]
        )["dataset_id"].head(10).tolist(),
        "manual_keypoint_labels_used": False,
        "model_rerun": False,
        "interpretation": "diagnostic organ-region accounting; automatic masks are not manual phenotype ground truth",
    }
    (args.evaluation / "region_hit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

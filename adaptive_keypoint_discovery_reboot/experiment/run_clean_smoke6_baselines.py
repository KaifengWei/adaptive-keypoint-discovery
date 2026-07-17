#!/usr/bin/env python
"""Run the frozen G1 baselines on clean-background smoke6 only."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402


def white_background_plant_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    distance = np.max(255 - image_rgb.astype(np.int16), axis=2)
    mask = (distance >= 22).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    threshold = max(12, round(mask.size * 0.00004))
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= threshold:
            clean[labels == index] = 255
    ys, xs = np.where(clean > 0)
    if not len(xs):
        bbox = (0.0, 0.0, float(mask.shape[1] - 1), float(mask.shape[0] - 1))
    else:
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    return clean, bbox


def overlay(path: Path, image: np.ndarray, panels: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    colors = ["#00d4ff", "#ff453a", "#bf5af2"]
    for axis, (title, points), color in zip(axes, panels, colors):
        axis.imshow(image)
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], s=34, c=color, edgecolors="black", linewidths=0.6)
        axis.set_title(f"{title} | n={len(points)}")
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.time()
    g1.set_deterministic(20260716)
    manifest = g1.read_csv(HERE / "data_clean_core20" / "smoke6_manifest.csv")
    output = HERE / "outputs_clean_smoke6_baselines"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model_args = argparse.Namespace(
        local_repo=HERE / "third_party" / "dinov2_git",
        model="dinov2_vits14_reg",
        weights=HERE / "third_party" / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth",
    )
    model = g1.load_official_model(model_args, device)
    configurations = [
        {"method": "cls_to_patch_attention", "size": 518, "layer": "last4avg", "tau": 0.996},
        {"method": "feature_local_contrast", "size": 518, "layer": "last4avg", "tau": 6.0},
        {"method": "feature_hdbscan_medoid", "size": 728, "layer": "last", "tau": 0.0},
    ]
    per_image: list[dict[str, Any]] = []
    per_transform: list[dict[str, Any]] = []
    overlay_cache: dict[str, dict[str, np.ndarray]] = {}
    for row_index, row in enumerate(manifest, start=1):
        for size in (518, 728):
            base, _ = g1.letterbox_rgb(Path(row["clean_image_path"]), size)
            plant_mask, plant_bbox = white_background_plant_mask(base)
            bbox_diag = max(1.0, math.hypot(plant_bbox[2] - plant_bbox[0], plant_bbox[3] - plant_bbox[1]))
            transforms = g1.make_transforms(base, full=False)
            representations, attention_maps = [], []
            for transform in transforms:
                reps, attention, _ = g1.extract_representations(model, transform["image"], device)
                representations.append(reps)
                attention_maps.append(attention)
            for config in [item for item in configurations if item["size"] == size]:
                candidates = [
                    g1.candidates_for(
                        config["method"],
                        representations[index][config["layer"]],
                        attention_maps[index],
                        config["tau"],
                        size,
                        20260716,
                    )
                    for index in range(len(transforms))
                ]
                reference = candidates[0]
                f1_values, error_values, photo_differences = [], [], []
                for index, transform in enumerate(transforms[1:], start=1):
                    mapped = g1.apply_inverse(candidates[index], transform["matrix"], size)
                    matched = g1.match_points(reference, mapped, 0.05 * bbox_diag)
                    error = matched["median_error"] / bbox_diag if np.isfinite(matched["median_error"]) else float("nan")
                    f1_values.append(matched["f1"])
                    if np.isfinite(error):
                        error_values.append(error)
                    if transform["family"] == "photometric":
                        photo_differences.append(abs(len(mapped) - len(reference)))
                    per_transform.append(
                        {
                            "clean_id": row["clean_id"],
                            "method": config["method"],
                            "transform": transform["name"],
                            "family": transform["family"],
                            "base_count": len(reference),
                            "mapped_count": len(mapped),
                            "f1": matched["f1"],
                            "localization_error_bbox_diag": error,
                        }
                    )
                per_image.append(
                    {
                        "clean_id": row["clean_id"],
                        "source_name": row["source_name"],
                        "method": config["method"],
                        "size": size,
                        "candidate_count": len(reference),
                        "safety_cap_hit": int(len(reference) >= 30),
                        "plant_hit_ratio_white_mask": g1.plant_hit_ratio(reference, plant_mask),
                        "mean_repeatability_f1": float(np.mean(f1_values)),
                        "median_localization_error_bbox_diag": float(np.median(error_values)) if error_values else float("nan"),
                        "photometric_count_difference": float(np.median(photo_differences)) if photo_differences else float("nan"),
                    }
                )
                normalized = reference * (518.0 / size)
                overlay_cache.setdefault(row["clean_id"], {})[config["method"]] = normalized
            if size == 518:
                overlay_cache.setdefault(row["clean_id"], {})["base_image"] = base
        panels = [
            ("attention", overlay_cache[row["clean_id"]]["cls_to_patch_attention"]),
            ("local contrast", overlay_cache[row["clean_id"]]["feature_local_contrast"]),
            ("HDBSCAN", overlay_cache[row["clean_id"]]["feature_hdbscan_medoid"]),
        ]
        overlay(output / "overlays" / f"{row['clean_id']}.png", overlay_cache[row["clean_id"]]["base_image"], panels)
        print(f"[{row_index}/{len(manifest)}] {row['clean_id']}", flush=True)

    g1.write_csv(output / "per_image_metrics.csv", per_image)
    g1.write_csv(output / "per_transform_metrics.csv", per_transform)
    summary = []
    for method in [item["method"] for item in configurations]:
        rows = [row for row in per_image if row["method"] == method]
        counts = np.asarray([row["candidate_count"] for row in rows], dtype=float)
        summary.append(
            {
                "method": method,
                "images": len(rows),
                "median_candidate_count": float(np.median(counts)),
                "candidate_count_iqr": float(np.quantile(counts, 0.75) - np.quantile(counts, 0.25)),
                "safety_cap_hit_rate": float(np.mean([row["safety_cap_hit"] for row in rows])),
                "median_plant_hit_ratio_white_mask": float(np.median([row["plant_hit_ratio_white_mask"] for row in rows])),
                "median_repeatability_f1": float(np.median([row["mean_repeatability_f1"] for row in rows])),
                "median_localization_error_bbox_diag": float(np.nanmedian([row["median_localization_error_bbox_diag"] for row in rows])),
                "median_photometric_count_difference": float(np.nanmedian([row["photometric_count_difference"] for row in rows])),
            }
        )
    g1.write_csv(output / "summary_metrics.csv", summary)
    decision = {
        "training_used": False,
        "keypoint_labels_read": False,
        "device": "cpu",
        "images_run": 6,
        "locked14_run": False,
        "configurations_unchanged_from_g1": True,
        "summary": summary,
        "original_pilot_reference": {
            "cls_to_patch_attention": {"plant_hit": 1.0, "repeatability_f1": 0.7482, "candidate_count": 3.0},
            "feature_local_contrast": {"plant_hit": 0.4410, "repeatability_f1": 0.7291, "candidate_count": 19.0, "safety_cap_hit_rate": 0.20},
            "feature_hdbscan_medoid": {"plant_hit": 0.3333, "repeatability_f1": 0.3679, "candidate_count": 5.0},
        },
        "interpretation_rule": "A clean-background improvement supports a data-quality contribution, but does not prove structural keypoint meaning.",
        "elapsed_seconds": time.time() - started,
    }
    (output / "data_quality_hypothesis_result.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

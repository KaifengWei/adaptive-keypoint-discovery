#!/usr/bin/env python
"""Evaluate the learned variable-count detector on explicitly selected splits."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
import g1_prime_phenotype_bridge as bridge  # noqa: E402
from adaptive_point_model import AdaptivePointDetector  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required when --splits contains test; prevents accidental access to a locked test set.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full-transforms", action="store_true")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--fixed-k", type=int, default=-1)
    parser.add_argument("--dinov2-local-repo", type=Path)
    parser.add_argument("--dinov2-weights", type=Path)
    parser.add_argument(
        "--input-domain",
        choices=["auto", "full_plant", "phenotype_roi_v1"],
        default="auto",
    )
    return parser.parse_args()


def tensor_from_rgb(image: np.ndarray, device: torch.device) -> torch.Tensor:
    normalized = image.astype(np.float32) / 255.0
    normalized = (normalized - g1.IMAGENET_MEAN) / g1.IMAGENET_STD
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)


def point_mask_hit_ratio(points: np.ndarray, mask: np.ndarray) -> float:
    """Exact mask membership without the generic foreground tolerance dilation."""
    if not len(points):
        return 0.0
    height, width = mask.shape
    hits = 0
    for x, y in points:
        xx = int(np.clip(round(x), 0, width - 1))
        yy = int(np.clip(round(y), 0, height - 1))
        hits += int(bool(mask[yy, xx]))
    return hits / len(points)


@torch.no_grad()
def predict(
    model: AdaptivePointDetector,
    image: np.ndarray,
    device: torch.device,
    threshold: float,
    safety_cap: int,
    fixed_k: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    logits = model(tensor_from_rgb(image, device))
    decoded = model.decode(
        logits,
        image.shape[:2],
        threshold=threshold,
        max_points=safety_cap,
        fixed_k=fixed_k,
    )[0]
    points = np.asarray([[item.x, item.y] for item in decoded], dtype=np.float64).reshape(-1, 2)
    records = [
        {
            "x": item.x,
            "y": item.y,
            "kind": "learned_heatmap",
            "score": item.confidence,
            "dino": float("nan"),
            "attention": float("nan"),
        }
        for item in decoded
    ]
    return points, records


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if "test" in args.splits and not args.allow_test:
        raise RuntimeError(
            "Refusing to evaluate the test split without --allow-test. "
            "Use val for development; add --allow-test only after the test gate is explicitly released."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    checkpoint_input_domain = str(config.get("input_domain", "full_plant"))
    input_domain = checkpoint_input_domain if args.input_domain == "auto" else args.input_domain
    if input_domain != checkpoint_input_domain:
        raise RuntimeError(
            f"Evaluation input domain {input_domain} does not match checkpoint domain {checkpoint_input_domain}"
        )
    repo = args.dinov2_local_repo or Path(str(config["dinov2_local_repo"]))
    weights = args.dinov2_weights or Path(str(config["dinov2_weights"]))
    model_args = argparse.Namespace(
        local_repo=repo,
        model=str(config.get("dinov2_model", "dinov2_vits14_reg")),
        weights=weights,
    )
    backbone = g1.load_official_model(model_args, device)
    model = AdaptivePointDetector(
        backbone,
        patch_size=14,
        decoder_dim=int(config.get("decoder_dim", 192)),
        output_stride=int(config.get("output_stride", 4)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        unfreeze_last_blocks=int(config.get("unfreeze_last_blocks", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(config.get("inference_threshold", 0.35)) if args.threshold < 0 else args.threshold
    fixed_k = int(config.get("fixed_k_eval", 0)) if args.fixed_k < 0 else args.fixed_k
    safety_cap = int(config.get("inference_safety_cap", 64))
    image_size = int(config["image_size"])

    frame = pd.read_csv(args.dataset / "manifests" / "all.csv")
    frame = frame[frame["split"].isin(args.splits)].sort_values("dataset_id")
    if args.limit > 0:
        frame = frame.head(args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    phenotype_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(frame.to_dict("records"), start=1):
        image_path = args.dataset / Path(str(row["relative_path"]).replace("\\", "/"))
        roi_result: dict[str, Any] | None = None
        if input_domain == "phenotype_roi_v1":
            base, mapping, focused_source, roi_result = load_phenotype_input(
                args.dataset, row, image_size
            )
            focused_path = args.output / "focused_inputs" / "images" / str(row["split"]) / f"{row['dataset_id']}.png"
            focused_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(focused_path), cv2.cvtColor(focused_source, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write focused evaluation input: {focused_path}")
            for mask_name, mask in (
                ("phenotype_roi", roi_result["phenotype_roi"]),
                ("basal_transition", roi_result["basal_transition"]),
            ):
                mask_path = args.output / "focused_inputs" / "masks" / mask_name / str(row["split"]) / f"{row['dataset_id']}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), np.asarray(mask, dtype=np.uint8) * 255):
                    raise RuntimeError(f"Failed to write focused evaluation mask: {mask_path}")
        else:
            base, mapping = g1.letterbox_rgb(image_path, image_size)
        transforms = g1.make_transforms(base, args.full_transforms)
        outputs = []
        for transform in transforms:
            points, records = predict(model, transform["image"], device, threshold, safety_cap, fixed_k)
            outputs.append({"points": points, "records": records})
        reference = outputs[0]
        support, skeleton, support_diagnostics = gp.automatic_structural_support(base)
        bbox = gp.bbox_from_mask(support)
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        f1_values, localization_values = [], []
        for transform, current in zip(transforms[1:], outputs[1:]):
            mapped = g1.apply_inverse(current["points"], transform["matrix"], image_size)
            matched = g1.match_points(reference["points"], mapped, 0.05 * bbox_diag)
            normalized_error = matched["median_error"] / bbox_diag if np.isfinite(matched["median_error"]) else float("nan")
            f1_values.append(float(matched["f1"]))
            if np.isfinite(normalized_error):
                localization_values.append(float(normalized_error))
            transform_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "transform": transform["name"],
                    "family": transform["family"],
                    "reference_count": len(reference["points"]),
                    "mapped_count": len(mapped),
                    "f1": matched["f1"],
                    "localization_error_bbox_diag": normalized_error,
                }
            )
        paths, graph_diagnostics = bridge.extract_paths(base, support, skeleton, reference["records"], bbox_diag)
        for point_index, item in enumerate(reference["records"], start=1):
            source_x, source_y = bridge.to_source_xy((item["x"], item["y"]), mapping)
            point_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "point_id": f"p{point_index:02d}",
                    "x_source": source_x,
                    "y_source": source_y,
                    "x_normalized": source_x / max(mapping["source_width"] - 1.0, 1.0),
                    "y_normalized": source_y / max(mapping["source_height"] - 1.0, 1.0),
                    "confidence": item["score"],
                }
            )
        for path_item in paths:
            path_rows.append({"dataset_id": row["dataset_id"], **path_item})
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
                "stage_label": row["stage_label"],
                "split": row["split"],
                "point_count": len(reference["points"]),
                "foreground_hit_ratio": g1.plant_hit_ratio(reference["points"], support),
                "phenotype_roi_hit_ratio": (
                    float("nan")
                    if roi_result is None
                    else point_mask_hit_ratio(reference["points"], roi_result["phenotype_roi_model"])
                ),
                "phenotype_roi_hit_ratio_tolerant": (
                    float("nan")
                    if roi_result is None
                    else g1.plant_hit_ratio(reference["points"], roi_result["phenotype_roi_model"])
                ),
                "excluded_seed_root_hit_ratio": (
                    float("nan")
                    if roi_result is None
                    else point_mask_hit_ratio(
                        reference["points"],
                        roi_result["seed_base_root_model"] & ~roi_result["phenotype_roi_model"],
                    )
                ),
                "seed_root_neighborhood_hit_ratio": (
                    float("nan")
                    if roi_result is None
                    else g1.plant_hit_ratio(reference["points"], roi_result["seed_base_root_model"])
                ),
                "roi_shoot_retention_ratio": (
                    float("nan") if roi_result is None else roi_result["shoot_retention_ratio"]
                ),
                "roi_green_shoot_retention_ratio": (
                    float("nan") if roi_result is None else roi_result["green_shoot_retention_ratio"]
                ),
                "roi_localization_source": (
                    "not_applicable" if roi_result is None else roi_result["localization_source"]
                ),
                "input_domain": input_domain,
                "mean_repeatability_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
                "median_localization_error_bbox_diag": float(np.median(localization_values)) if localization_values else float("nan"),
                "path_count": len(paths),
                "graph_failure": graph_diagnostics.get("failure", ""),
                "safety_cap_hit": int(len(reference["points"]) >= safety_cap),
                "support_mask_mode_uniform": support_diagnostics.get("mask_mode_uniform_background", 0.0),
            }
        )
        bridge.save_overlay(
            args.output / "overlays" / f"{row['dataset_id']}.png",
            base,
            support,
            skeleton,
            reference["records"],
            paths,
            graph_diagnostics,
            f"{row['dataset_id']} | learned adaptive points",
            candidate_label="learned adaptive points",
        )
        print(f"[{row_number}/{len(frame)}] {row['dataset_id']} points={len(reference['points'])} paths={len(paths)}", flush=True)

    pd.DataFrame(image_rows).to_csv(args.output / "per_image.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transform_rows).to_csv(args.output / "per_transform.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(point_rows).to_csv(args.output / "points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phenotype_rows).to_csv(args.output / "phenotypes.csv", index=False, encoding="utf-8-sig")
    bridge.write_jsonl(args.output / "paths.jsonl", path_rows)
    summary = {
        "images": len(image_rows),
        "splits": list(args.splits),
        "stage_counts": dict(Counter(row["stage_label"] for row in image_rows)),
        "fixed_k": fixed_k,
        "threshold": threshold,
        "input_domain": input_domain,
        "median_point_count": float(np.median([row["point_count"] for row in image_rows])) if image_rows else 0.0,
        "median_repeatability_f1": float(np.nanmedian([row["mean_repeatability_f1"] for row in image_rows])) if image_rows else float("nan"),
        "median_foreground_hit_ratio": float(np.median([row["foreground_hit_ratio"] for row in image_rows])) if image_rows else 0.0,
        "median_phenotype_roi_hit_ratio": (
            float(np.nanmedian([row["phenotype_roi_hit_ratio"] for row in image_rows]))
            if input_domain == "phenotype_roi_v1" and image_rows
            else float("nan")
        ),
        "median_phenotype_roi_hit_ratio_tolerant": (
            float(np.nanmedian([row["phenotype_roi_hit_ratio_tolerant"] for row in image_rows]))
            if input_domain == "phenotype_roi_v1" and image_rows
            else float("nan")
        ),
        "median_excluded_seed_root_hit_ratio": (
            float(np.nanmedian([row["excluded_seed_root_hit_ratio"] for row in image_rows]))
            if input_domain == "phenotype_roi_v1" and image_rows
            else float("nan")
        ),
        "median_seed_root_neighborhood_hit_ratio": (
            float(np.nanmedian([row["seed_root_neighborhood_hit_ratio"] for row in image_rows]))
            if input_domain == "phenotype_roi_v1" and image_rows
            else float("nan")
        ),
        "graph_success_rate": float(np.mean([not row["graph_failure"] for row in image_rows])) if image_rows else 0.0,
        "safety_cap_hit_rate": float(np.mean([row["safety_cap_hit"] for row in image_rows])) if image_rows else 0.0,
        "manual_keypoint_labels_used": False,
        "phenotype_accuracy_status": "pending_manual_measurement_reference",
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

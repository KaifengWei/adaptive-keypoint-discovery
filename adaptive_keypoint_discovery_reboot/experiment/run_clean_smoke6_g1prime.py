#!/usr/bin/env python
"""Apply the frozen G1-prime heuristic unchanged to clean smoke6."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
from run_clean_smoke6_baselines import white_background_plant_mask  # noqa: E402


def main() -> None:
    started = time.time()
    g1.set_deterministic(20260716)
    rows = g1.read_csv(HERE / "data_clean_core20" / "smoke6_manifest.csv")
    output = HERE / "outputs_clean_smoke6_g1prime"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model_args = argparse.Namespace(
        local_repo=HERE / "third_party" / "dinov2_git",
        model="dinov2_vits14_reg",
        weights=HERE / "third_party" / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth",
    )
    model = g1.load_official_model(model_args, device)
    per_image: list[dict[str, Any]] = []
    per_transform: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        base, _ = g1.letterbox_rgb(Path(row["clean_image_path"]), 518)
        evaluation_mask, evaluation_bbox = white_background_plant_mask(base)
        bbox_diag = max(1.0, math.hypot(evaluation_bbox[2] - evaluation_bbox[0], evaluation_bbox[3] - evaluation_bbox[1]))
        transforms = g1.make_transforms(base, full=False)
        discovered = []
        for transform in transforms:
            reps, attention, _ = g1.extract_representations(model, transform["image"], device)
            points, records, support, skeleton, diagnostics = gp.structural_candidates(
                transform["image"], reps["last4avg"], attention, max_points=20
            )
            discovered.append(
                {
                    "points": points,
                    "records": records,
                    "support": support,
                    "skeleton": skeleton,
                    "diagnostics": diagnostics,
                }
            )
        reference = discovered[0]
        f1_values, errors, photo_differences = [], [], []
        for transform, current in zip(transforms[1:], discovered[1:]):
            mapped = g1.apply_inverse(current["points"], transform["matrix"], 518)
            matched = g1.match_points(reference["points"], mapped, 0.05 * bbox_diag)
            error = matched["median_error"] / bbox_diag if np.isfinite(matched["median_error"]) else float("nan")
            f1_values.append(matched["f1"])
            if np.isfinite(error):
                errors.append(error)
            if transform["family"] == "photometric":
                photo_differences.append(abs(len(mapped) - len(reference["points"])))
            per_transform.append(
                {
                    "clean_id": row["clean_id"],
                    "transform": transform["name"],
                    "family": transform["family"],
                    "base_count": len(reference["points"]),
                    "mapped_count": len(mapped),
                    "f1": matched["f1"],
                    "localization_error_bbox_diag": error,
                }
            )
        per_image.append(
            {
                "clean_id": row["clean_id"],
                "source_name": row["source_name"],
                "candidate_count": len(reference["points"]),
                "candidate_types": ";".join(item["kind"] for item in reference["records"]),
                "safety_cap_hit": int(len(reference["points"]) >= 20),
                "white_mask_hit_ratio": g1.plant_hit_ratio(reference["points"], evaluation_mask),
                "mean_repeatability_f1": float(np.mean(f1_values)),
                "median_localization_error_bbox_diag": float(np.median(errors)) if errors else float("nan"),
                "photometric_count_difference": float(np.median(photo_differences)) if photo_differences else float("nan"),
                "skeleton_coverage": gp.support_coverage(reference["points"], reference["skeleton"], 0.08 * bbox_diag),
                "longitudinal_span_ratio": gp.span_ratio(reference["points"], evaluation_mask),
            }
        )
        gp.save_overlay(
            output / "overlays" / f"{row['clean_id']}.png",
            base,
            reference["support"],
            reference["skeleton"],
            reference["records"],
            f"{row['clean_id']} | unchanged G1-prime on clean background",
        )
        print(f"[{row_index}/{len(rows)}] {row['clean_id']} points={len(reference['points'])}", flush=True)

    g1.write_csv(output / "per_image_metrics.csv", per_image)
    g1.write_csv(output / "per_transform_metrics.csv", per_transform)
    counts = np.asarray([row["candidate_count"] for row in per_image], dtype=float)
    metrics = {
        "images": len(per_image),
        "median_candidate_count": float(np.median(counts)),
        "candidate_count_iqr": float(np.quantile(counts, 0.75) - np.quantile(counts, 0.25)),
        "safety_cap_hit_rate": float(np.mean([row["safety_cap_hit"] for row in per_image])),
        "median_white_mask_hit_ratio": float(np.median([row["white_mask_hit_ratio"] for row in per_image])),
        "median_repeatability_f1": float(np.median([row["mean_repeatability_f1"] for row in per_image])),
        "median_localization_error_bbox_diag": float(np.nanmedian([row["median_localization_error_bbox_diag"] for row in per_image])),
        "median_photometric_count_difference": float(np.nanmedian([row["photometric_count_difference"] for row in per_image])),
        "median_skeleton_coverage": float(np.median([row["skeleton_coverage"] for row in per_image])),
        "median_longitudinal_span_ratio": float(np.median([row["longitudinal_span_ratio"] for row in per_image])),
    }
    gates = {
        "safety_cap_hit_rate_eq_0": metrics["safety_cap_hit_rate"] == 0.0,
        "repeatability_f1_gte_0_60": metrics["median_repeatability_f1"] >= 0.60,
        "photometric_count_difference_lte_1": metrics["median_photometric_count_difference"] <= 1.0,
        "white_mask_hit_ratio_gte_0_80": metrics["median_white_mask_hit_ratio"] >= 0.80,
    }
    result = {
        "method_parameters_changed": False,
        "training_used": False,
        "keypoint_labels_read": False,
        "locked14_run": False,
        "metrics": metrics,
        "machine_gates": gates,
        "decision": "pending_human_structural_meaning_review" if all(gates.values()) else "stop_before_locked14",
        "elapsed_seconds": time.time() - started,
    }
    (output / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

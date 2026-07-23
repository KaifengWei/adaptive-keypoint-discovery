#!/usr/bin/env python
"""Generate variable-count pseudo labels from cross-transform G1-prime consensus.

The outputs are automatic teacher targets for a learnable point detector.  They
are not manual keypoint annotations and are not treated as phenotype ground
truth.  A point is retained only when its inverse-mapped location is reproduced
under multiple geometric/photometric views, unless the explicit no-consistency
ablation is selected.
"""

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v3")
    parser.add_argument("--output", type=Path, default=HERE / "pseudo_labels_g1prime_v3")
    parser.add_argument("--splits", nargs="+", default=["train", "auxiliary"])
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected rows")
    parser.add_argument("--local-repo", type=Path, default=HERE / "third_party" / "dinov2_git")
    parser.add_argument(
        "--weights", type=Path, default=HERE / "third_party" / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth"
    )
    parser.add_argument("--model", default="dinov2_vits14_reg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--max-points", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--full-transforms", action="store_true")
    parser.add_argument("--min-presence", type=float, default=0.75)
    parser.add_argument("--max-localization-error", type=float, default=0.025)
    parser.add_argument("--teacher-variant", choices=["full", "geometry_only"], default="full")
    parser.add_argument(
        "--structure-coverage",
        action="store_true",
        help=(
            "Preserve cross-transform-stable skeleton terminals and an automatically "
            "derived shoot-side basal-transition proposal before generic NMS."
        ),
    )
    parser.add_argument("--no-consistency-filter", action="store_true")
    parser.add_argument(
        "--input-domain",
        choices=["full_plant", "phenotype_roi_v1"],
        default="full_plant",
        help="Model/teacher image domain. phenotype_roi_v1 removes seed and root interference on white.",
    )
    parser.add_argument(
        "--quality-exclusions",
        type=Path,
        help="Optional CSV with dataset_id, exclude_from_teacher and reason; split membership is unchanged.",
    )
    return parser.parse_args()


def load_quality_exclusions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    required = {"dataset_id", "exclude_from_teacher", "reason"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Quality exclusion CSV is missing columns: {sorted(required - set(frame.columns))}")
    selected = frame[frame["exclude_from_teacher"].astype(str).str.lower().isin({"1", "true", "yes"})]
    return {str(row["dataset_id"]): str(row["reason"]) for row in selected.to_dict("records")}


def to_source_xy(point: np.ndarray, mapping: dict[str, float]) -> tuple[float, float]:
    x = (float(point[0]) - mapping["pad_x"]) / mapping["scale"]
    y = (float(point[1]) - mapping["pad_y"]) / mapping["scale"]
    return (
        float(np.clip(x, 0.0, mapping["source_width"] - 1.0)),
        float(np.clip(y, 0.0, mapping["source_height"] - 1.0)),
    )


def assignments(reference: np.ndarray, current: np.ndarray, radius: float) -> dict[int, tuple[int, float]]:
    if not len(reference) or not len(current):
        return {}
    distance = np.linalg.norm(reference[:, None, :] - current[None, :, :], axis=2)
    left, right = linear_sum_assignment(distance)
    return {
        int(i): (int(j), float(distance[i, j]))
        for i, j in zip(left, right)
        if float(distance[i, j]) <= radius
    }


def transformed_mask(mask: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    """Apply the same geometric view transform to an audit-derived model mask."""
    return cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(matrix, dtype=np.float64)[:2],
        (size, size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def inverse_mapped_records(
    points: np.ndarray,
    records: list[dict[str, Any]],
    matrix: np.ndarray,
    size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Inverse-map points while preserving record alignment at image borders."""
    if not len(points):
        return np.empty((0, 2), dtype=np.float64), []
    homogeneous = np.concatenate(
        [np.asarray(points, dtype=np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    mapped = (np.linalg.inv(np.asarray(matrix, dtype=np.float64)) @ homogeneous.T).T[:, :2]
    valid = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < size)
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < size)
    )
    return mapped[valid], [record for record, keep in zip(records, valid) if bool(keep)]


def consensus(
    outputs: list[dict[str, Any]], transforms: list[dict[str, Any]], bbox_diag: float, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_points = outputs[0]["points"]
    reference_records = outputs[0]["records"]
    observations: list[list[dict[str, Any]]] = [
        [{"distance": 0.0, "record": record, "transform": "identity"}] for record in reference_records
    ]
    radius = 0.05 * bbox_diag
    for transform, current in zip(transforms[1:], outputs[1:]):
        mapped, mapped_records = inverse_mapped_records(
            current["points"], current["records"], transform["matrix"], args.size
        )
        matched = assignments(reference_points, mapped, radius)
        for reference_index, (current_index, distance) in matched.items():
            observations[reference_index].append(
                {
                    "distance": distance,
                    "record": mapped_records[current_index],
                    "transform": transform["name"],
                }
            )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, (point, record, point_observations) in enumerate(zip(reference_points, reference_records, observations)):
        presence = len(point_observations) / len(transforms)
        errors = [float(item["distance"]) / max(bbox_diag, 1e-9) for item in point_observations[1:]]
        localization_error = float(np.median(errors)) if errors else 0.0
        teacher_score = float(np.mean([float(item["record"]["score"]) for item in point_observations]))
        observed_coverage_roles = sorted(
            {
                str(role)
                for observation in point_observations
                for role in observation["record"].get("coverage_roles", [])
            }
        )
        # Target semantics come from the identity-view proposal.  Other views
        # prove positional repeatability but must not assign a second basal role
        # through a nearby Hungarian match.
        coverage_roles = sorted(str(role) for role in record.get("coverage_roles", []))
        confidence = float(presence * math.exp(-localization_error / 0.025) * np.clip(teacher_score, 0.0, 1.0))
        keep = args.no_consistency_filter or (
            presence >= args.min_presence
            and localization_error <= args.max_localization_error
            and confidence >= 0.35
        )
        item = {
            "reference_index": index,
            "x_model": float(point[0]),
            "y_model": float(point[1]),
            "kind": record["kind"],
            "coverage_roles": coverage_roles,
            "observed_coverage_roles": observed_coverage_roles,
            "teacher_score": teacher_score,
            "presence_ratio": presence,
            "localization_error_bbox_diag": localization_error,
            "consensus_confidence": confidence,
            "observed_transforms": [str(value["transform"]) for value in point_observations],
            "accepted": int(keep),
        }
        (accepted if keep else rejected).append(item)
    return accepted, rejected


def save_overlay(path: Path, image: np.ndarray, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6), dpi=150)
    axis.imshow(image)
    if rejected:
        points = np.asarray([[row["x_model"], row["y_model"]] for row in rejected])
        axis.scatter(points[:, 0], points[:, 1], s=38, marker="x", c="#ff453a", label=f"rejected {len(rejected)}")
    if accepted:
        points = np.asarray([[row["x_model"], row["y_model"]] for row in accepted])
        confidence = np.asarray([row["consensus_confidence"] for row in accepted])
        axis.scatter(
            points[:, 0], points[:, 1], s=38 + 55 * confidence, c=confidence,
            cmap="viridis", vmin=0.35, vmax=1.0, edgecolors="black", linewidths=0.5,
            label=f"accepted {len(accepted)}",
        )
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if "test" in args.splits:
        raise RuntimeError("Automatic teacher generation is forbidden on the locked test split")
    if args.structure_coverage and args.input_domain != "phenotype_roi_v1":
        raise RuntimeError("Structure coverage requires --input-domain phenotype_roi_v1")
    g1.set_deterministic(args.seed)
    frame = pd.read_csv(args.dataset / "manifests" / "all.csv")
    frame = frame[frame["split"].isin(args.splits)].sort_values("dataset_id")
    selected_before_quality_gate = int(len(frame))
    exclusion_reasons = load_quality_exclusions(args.quality_exclusions)
    excluded_ids = [str(value) for value in frame["dataset_id"] if str(value) in exclusion_reasons]
    if excluded_ids:
        frame = frame[~frame["dataset_id"].astype(str).isin(excluded_ids)]
    if args.limit > 0:
        frame = frame.head(args.limit)
    if frame.empty:
        raise RuntimeError("No images selected from the requested splits")
    args.output.mkdir(parents=True, exist_ok=True)
    device = g1.resolve_device(args.device)
    model = None
    if args.teacher_variant == "full":
        model_args = argparse.Namespace(local_repo=args.local_repo, model=args.model, weights=args.weights)
        model = g1.load_official_model(model_args, device)

    pseudo_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(frame.to_dict("records"), start=1):
        image_relative_path = Path(str(row["relative_path"]).replace("\\", "/"))
        image_path = args.dataset / image_relative_path
        roi_result: dict[str, Any] | None = None
        input_image_relative_path = ""
        if args.input_domain == "phenotype_roi_v1":
            base, mapping, focused_source, roi_result = load_phenotype_input(
                args.dataset, row, args.size
            )
            input_image_relative_path = (
                Path("focused_inputs") / "images" / str(row["split"]) / f"{row['dataset_id']}.png"
            ).as_posix()
            focused_path = args.output / input_image_relative_path
            focused_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(focused_path), cv2.cvtColor(focused_source, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write focused teacher input: {focused_path}")
            for mask_name, mask in (
                ("phenotype_roi", roi_result["phenotype_roi"]),
                ("basal_transition", roi_result["basal_transition"]),
            ):
                mask_path = (
                    args.output
                    / "focused_inputs"
                    / "masks"
                    / mask_name
                    / str(row["split"])
                    / f"{row['dataset_id']}.png"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), np.asarray(mask, dtype=np.uint8) * 255):
                    raise RuntimeError(f"Failed to write focused input mask: {mask_path}")
        else:
            base, mapping = g1.letterbox_rgb(image_path, args.size)
        transforms = g1.make_transforms(base, args.full_transforms)
        outputs: list[dict[str, Any]] = []
        for transform in transforms:
            if model is None:
                grid = args.size // g1.PATCH_SIZE
                feature_map = np.zeros((grid, grid, 2), dtype=np.float32)
                attention = np.zeros((grid, grid), dtype=np.float32)
            else:
                representations, attention, _ = g1.extract_representations(model, transform["image"], device)
                feature_map = representations["last4avg"]
            basal_transition_view = None
            if args.structure_coverage:
                if roi_result is None:
                    raise RuntimeError("Structure coverage requires phenotype ROI metadata")
                basal_transition_view = transformed_mask(
                    roi_result["basal_transition_model"],
                    transform["matrix"],
                    args.size,
                )
            points, records, support, skeleton, diagnostics = gp.structural_candidates(
                transform["image"],
                feature_map,
                attention,
                args.max_points,
                evidence_mode=args.teacher_variant,
                structure_coverage=args.structure_coverage,
                basal_transition_mask=basal_transition_view,
            )
            outputs.append(
                {
                    "points": points,
                    "records": records,
                    "support": support,
                    "skeleton": skeleton,
                    "diagnostics": diagnostics,
                }
            )
        bbox = gp.bbox_from_mask(outputs[0]["support"])
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        accepted, rejected = consensus(outputs, transforms, bbox_diag, args)
        for point_index, item in enumerate(accepted, start=1):
            source_x, source_y = to_source_xy(np.asarray([item["x_model"], item["y_model"]]), mapping)
            item.update(
                {
                    "point_id": f"p{point_index:02d}",
                    "x_source": source_x,
                    "y_source": source_y,
                    "x_normalized": source_x / max(mapping["source_width"] - 1.0, 1.0),
                    "y_normalized": source_y / max(mapping["source_height"] - 1.0, 1.0),
                }
            )
        usable = int(len(accepted) >= 2 and not outputs[0]["diagnostics"].get("safety_cap_hit", 0))
        pseudo_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "image_path_at_generation": str(image_path.resolve()),
                "image_relative_path": image_relative_path.as_posix(),
                "stage_label": row["stage_label"],
                "split": row["split"],
                "input_domain": args.input_domain,
                "input_image_relative_path": input_image_relative_path,
                "teacher_variant": args.teacher_variant,
                "structure_coverage_enabled": args.structure_coverage,
                "consistency_filter_used": not args.no_consistency_filter,
                "transforms": [transform["name"] for transform in transforms],
                "bbox_diagonal_model": bbox_diag,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "training_usable": usable,
                "points": accepted,
                "phenotype_roi_diagnostics": (
                    {}
                    if roi_result is None
                    else {
                        "shoot_retention_ratio": roi_result["shoot_retention_ratio"],
                        "green_shoot_retention_ratio": roi_result["green_shoot_retention_ratio"],
                        "root_base_overlap_ratio": roi_result["root_base_overlap_ratio"],
                        "seed_detected": roi_result["seed_detected"],
                        "localization_source": roi_result["localization_source"],
                        "retention_repair_used": roi_result["retention_repair_used"],
                    }
                ),
            }
        )
        audit_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "stage_label": row["stage_label"],
                "split": row["split"],
                "identity_candidate_count": len(outputs[0]["points"]),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "training_usable": usable,
                "median_confidence": float(np.median([item["consensus_confidence"] for item in accepted])) if accepted else 0.0,
                "median_presence": float(np.median([item["presence_ratio"] for item in accepted])) if accepted else 0.0,
                "median_localization_error": float(np.median([item["localization_error_bbox_diag"] for item in accepted])) if accepted else float("nan"),
                "safety_cap_hit": int(outputs[0]["diagnostics"].get("safety_cap_hit", 0)),
                "identity_endpoint_count": int(
                    outputs[0]["diagnostics"].get("identity_endpoint_count", 0)
                ),
                "basal_transition_candidate_added": int(
                    outputs[0]["diagnostics"].get("basal_transition_candidate_added", 0)
                ),
                "input_domain": args.input_domain,
                "shoot_retention_ratio": (
                    float("nan") if roi_result is None else roi_result["shoot_retention_ratio"]
                ),
                "root_base_overlap_ratio": (
                    float("nan") if roi_result is None else roi_result["root_base_overlap_ratio"]
                ),
                "green_shoot_retention_ratio": (
                    float("nan") if roi_result is None else roi_result["green_shoot_retention_ratio"]
                ),
                "seed_detected": (False if roi_result is None else roi_result["seed_detected"]),
                "roi_localization_source": (
                    "not_applicable" if roi_result is None else roi_result["localization_source"]
                ),
                "retention_repair_used": (
                    False if roi_result is None else roi_result["retention_repair_used"]
                ),
            }
        )
        save_overlay(
            args.output / "overlays" / f"{row['dataset_id']}.png",
            base,
            accepted,
            rejected,
            f"{row['dataset_id']} | consensus pseudo labels",
        )
        print(f"[{row_number}/{len(frame)}] {row['dataset_id']} accepted={len(accepted)} rejected={len(rejected)}", flush=True)

    with (args.output / "pseudo_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in pseudo_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(audit_rows).to_csv(args.output / "pseudo_label_audit.csv", index=False, encoding="utf-8-sig")
    usable_rows = [row for row in audit_rows if row["training_usable"]]
    summary = {
        "images": len(audit_rows),
        "selected_images_before_quality_gate": selected_before_quality_gate,
        "quality_excluded_images": len(excluded_ids),
        "quality_excluded_dataset_ids": excluded_ids,
        "quality_exclusion_reasons": {dataset_id: exclusion_reasons[dataset_id] for dataset_id in excluded_ids},
        "training_usable_images": len(usable_rows),
        "usable_rate": len(usable_rows) / len(audit_rows),
        "teacher_variant": args.teacher_variant,
        "structure_coverage_enabled": args.structure_coverage,
        "input_domain": args.input_domain,
        "consistency_filter_used": not args.no_consistency_filter,
        "keypoint_labels_read": False,
        "manual_keypoint_annotations_created": False,
        "median_accepted_count": float(np.median([row["accepted_count"] for row in audit_rows])),
        "stage_counts": dict(Counter(row["stage_label"] for row in audit_rows)),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "interpretation": "automatic teacher targets for self-training; not phenotype ground truth",
    }
    gate_failure = ""
    if args.input_domain == "phenotype_roi_v1" and audit_rows:
        retention = np.asarray([row["shoot_retention_ratio"] for row in audit_rows], dtype=np.float64)
        green_retention = np.asarray(
            [row["green_shoot_retention_ratio"] for row in audit_rows], dtype=np.float64
        )
        overlap = np.asarray([row["root_base_overlap_ratio"] for row in audit_rows], dtype=np.float64)
        summary.update(
            {
                "median_shoot_retention_ratio": float(np.nanmedian(retention)),
                "minimum_shoot_retention_ratio": float(np.nanmin(retention)),
                "median_green_shoot_retention_ratio": float(np.nanmedian(green_retention)),
                "minimum_green_shoot_retention_ratio": float(np.nanmin(green_retention)),
                "median_root_base_overlap_ratio": float(np.nanmedian(overlap)),
                "maximum_root_base_overlap_ratio": float(np.nanmax(overlap)),
                "retention_repair_images": int(sum(bool(row["retention_repair_used"]) for row in audit_rows)),
            }
        )
        summary["phenotype_input_gate_metric"] = "green_shoot_retention_ratio"
        summary["phenotype_input_gate_passed"] = bool(float(np.nanmin(green_retention)) >= 0.90)
        if not summary["phenotype_input_gate_passed"]:
            gate_failure = "Phenotype input gate failed: at least one included image retains < 90% of strict-green shoot"
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if gate_failure:
        raise RuntimeError(gate_failure)


if __name__ == "__main__":
    run(parse_args())

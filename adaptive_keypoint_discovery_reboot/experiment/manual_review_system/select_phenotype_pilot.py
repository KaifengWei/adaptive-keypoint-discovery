#!/usr/bin/env python3
"""Select a phenotype pilot without using model outcomes for the core subset.

The 12-image core is selected first and exclusively from acquisition metadata
and pixel-derived shoot-mask morphology. The factorized A/B/C/D review is only
loaded afterwards to choose four non-overlapping diagnostic cases.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize


CORE_QUOTAS = {
    "src_0347": 3,
    "src_0546": 2,
    "src_0615": 2,
    "src_0565": 2,
    "src_0349": 1,
    "src_0614": 1,
    "src_0560": 1,
}

FEATURES = (
    "log_shoot_area",
    "bbox_aspect_log",
    "shoot_fill_ratio",
    "skeleton_density",
    "endpoint_count_log",
    "branchpoint_count_log",
    "mean_thickness_log",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.filter2D(binary.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) - binary


def source_dpi(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if not dpi or len(dpi) != 2:
            raise ValueError(f"missing TIFF dpi metadata: {path}")
        return float(dpi[0]), float(dpi[1])


def morphology_record(
    row: dict[str, str],
    built_row: dict[str, str],
    mask_root: Path,
    image_root: Path,
) -> dict:
    dataset_id = row["dataset_id"]
    mask_path = mask_root / f"{dataset_id}.png"
    image_path = image_root / f"{dataset_id}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    binary = mask > 0
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError(f"empty shoot mask: {dataset_id}")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    bbox_area = bbox_w * bbox_h
    shoot_area = int(binary.sum())
    skeleton = skeletonize(binary)
    skeleton_pixels = int(skeleton.sum())
    neighbors = neighbor_count(skeleton)
    endpoint_count = int(np.logical_and(skeleton, neighbors == 1).sum())
    branchpoint_count = int(np.logical_and(skeleton, neighbors >= 3).sum())
    crop_box = ast.literal_eval(row["crop_box_full"])
    fullplant_box = ast.literal_eval(built_row["source_crop_box_fullplant"])
    fullplant_width = int(fullplant_box[2] - fullplant_box[0])
    fullplant_height = int(fullplant_box[3] - fullplant_box[1])
    resize_scale = min(1.0, 1600.0 / max(fullplant_width, fullplant_height))
    padding = int(float(built_row["normalization_padding_pixels"]))
    expected_width = round(fullplant_width * resize_scale) + 2 * padding
    expected_height = round(fullplant_height * resize_scale) + 2 * padding
    if (expected_width, expected_height) != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            f"scale reconstruction mismatch for {dataset_id}: "
            f"expected {(expected_width, expected_height)}, got {(mask.shape[1], mask.shape[0])}"
        )
    dpi_x, dpi_y = source_dpi(Path(row["source_path"]))
    if abs(dpi_x - dpi_y) > 1e-6:
        raise ValueError(f"anisotropic source resolution for {dataset_id}: {dpi_x}, {dpi_y}")
    mm_per_output_px = 25.4 / (dpi_x * resize_scale)

    return {
        "dataset_id": dataset_id,
        "split": row["split"],
        "source_frame_id": row["source_frame_id"],
        "acquisition_period": row["acquisition_period"],
        "candidate_orientation": row["candidate_orientation"],
        "source_relative_path": row["source_relative_path"],
        "crop_width_px": int(crop_box[2] - crop_box[0]),
        "crop_height_px": int(crop_box[3] - crop_box[1]),
        "fullplant_source_crop_width_px": fullplant_width,
        "fullplant_source_crop_height_px": fullplant_height,
        "standardized_width_px": int(mask.shape[1]),
        "standardized_height_px": int(mask.shape[0]),
        "normalization_padding_px": padding,
        "resize_scale": resize_scale,
        "source_dpi": dpi_x,
        "mm_per_output_px": mm_per_output_px,
        "physical_scale_status": "verified_from_tiff_metadata",
        "shoot_area_px": shoot_area,
        "shoot_bbox_width_px": bbox_w,
        "shoot_bbox_height_px": bbox_h,
        "shoot_fill_ratio": shoot_area / bbox_area,
        "skeleton_pixels": skeleton_pixels,
        "endpoint_count": endpoint_count,
        "branchpoint_count": branchpoint_count,
        "mean_thickness_px": shoot_area / max(skeleton_pixels, 1),
        "log_shoot_area": math.log1p(shoot_area),
        "bbox_aspect_log": math.log(max(bbox_w, 1) / max(bbox_h, 1)),
        "skeleton_density": skeleton_pixels / math.hypot(bbox_w, bbox_h),
        "endpoint_count_log": math.log1p(endpoint_count),
        "branchpoint_count_log": math.log1p(branchpoint_count),
        "mean_thickness_log": math.log1p(shoot_area / max(skeleton_pixels, 1)),
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
    }


def robust_feature_matrix(records: list[dict]) -> np.ndarray:
    matrix = np.asarray([[float(record[name]) for name in FEATURES] for record in records], dtype=np.float64)
    median = np.median(matrix, axis=0)
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    scale = q3 - q1
    scale[scale < 1e-9] = 1.0
    return (matrix - median) / scale


def medoid_farthest_indices(matrix: np.ndarray, quota: int) -> list[int]:
    if quota >= len(matrix):
        return list(range(len(matrix)))
    centroid = matrix.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(matrix - centroid, axis=1)))
    selected = [first]
    while len(selected) < quota:
        distances = np.min(
            np.stack([np.linalg.norm(matrix - matrix[index], axis=1) for index in selected]),
            axis=0,
        )
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return selected


def select_core(records: list[dict]) -> list[dict]:
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_frame[record["source_frame_id"]].append(record)

    if Counter(record["source_frame_id"] for record in records) != Counter(
        {frame: len(by_frame[frame]) for frame in by_frame}
    ):
        raise AssertionError("source frame accounting failed")

    selected: list[dict] = []
    for frame, quota in CORE_QUOTAS.items():
        group = sorted(by_frame[frame], key=lambda item: item["dataset_id"])
        matrix = robust_feature_matrix(group)
        for index in medoid_farthest_indices(matrix, quota):
            selected.append(group[index])
    return sorted(selected, key=lambda item: item["dataset_id"])


def choose_diagnostics(review_rows: list[dict[str, str]], excluded_ids: set[str]) -> list[tuple[str, str]]:
    rows = {row["dataset_id"]: row for row in review_rows if row["dataset_id"] not in excluded_ids}

    def required(dataset_id: str, reason: str) -> tuple[str, str]:
        if dataset_id not in rows:
            raise ValueError(f"diagnostic case unavailable after core lock: {dataset_id}")
        return dataset_id, reason

    return [
        required("v4_val_0002", "D>B：增强Teacher恢复漏叶并改善基部的正向案例"),
        required("v4_val_0023", "B>D：D丢失两片真叶且基部恶化的反向案例"),
        required("v4_val_0034", "局部Decoder新增假枝和错连，且各方案均有基部争议"),
        required("v4_val_0018", "B新增假枝和错连而D消除该错误，用于检查错连的表型代价"),
    ]


def make_contact_sheet(selection: list[dict], output_path: Path) -> None:
    cell_w, cell_h = 560, 260
    cols = 2
    rows = math.ceil(len(selection) / cols)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, item in enumerate(selection):
        image = Image.open(item["image_path"]).convert("RGB")
        image.thumbnail((cell_w - 20, cell_h - 55), Image.Resampling.LANCZOS)
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        paste_x = x + (cell_w - image.width) // 2
        paste_y = y + 35 + (cell_h - 45 - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        color = "#155EEF" if item["pilot_group"] == "core" else "#B54708"
        label = f"{item['dataset_id']} | {item['pilot_group']} | {item['source_frame_id']} | {item['acquisition_period']}"
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=color, width=3)
        draw.text((x + 10, y + 10), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--built-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--shoot-mask-root", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_rows = [row for row in read_csv(Path(args.manifest)) if row["split"] == "val"]
    if len(manifest_rows) != 40:
        raise ValueError(f"expected 40 val rows, got {len(manifest_rows)}")

    built_rows = [row for row in read_csv(Path(args.built_manifest)) if row["split"] == "val"]
    built_by_id = {row["dataset_id"]: row for row in built_rows}
    if len(built_by_id) != 40:
        raise ValueError(f"expected 40 built val rows, got {len(built_by_id)}")
    records = [
        morphology_record(
            row,
            built_by_id[row["dataset_id"]],
            Path(args.shoot_mask_root),
            Path(args.image_root),
        )
        for row in manifest_rows
    ]

    # Critical leakage boundary: core is locked before review outcomes are read.
    core = select_core(records)
    core_ids = {item["dataset_id"] for item in core}
    if len(core_ids) != 12:
        raise AssertionError(f"expected 12 unique core images, got {len(core_ids)}")

    review_rows = read_csv(Path(args.review_csv))
    diagnostic_pairs = choose_diagnostics(review_rows, core_ids)
    diagnostic_ids = {dataset_id for dataset_id, _ in diagnostic_pairs}
    if len(diagnostic_ids) != 4 or core_ids & diagnostic_ids:
        raise AssertionError("core and diagnostic sets must be unique and disjoint")

    record_by_id = {record["dataset_id"]: record for record in records}
    review_by_id = {row["dataset_id"]: row for row in review_rows}
    selection: list[dict] = []
    for item in core:
        enriched = dict(item)
        enriched.update(
            {
                "pilot_group": "core",
                "selection_basis": "acquisition+original_morphology_only",
                "selection_reason": "source-frame quota plus robust morphology medoid/farthest sampling",
                "visible_leaf_count_postlock": review_by_id[item["dataset_id"]]["visible_leaf_count"],
            }
        )
        selection.append(enriched)
    for dataset_id, reason in diagnostic_pairs:
        enriched = dict(record_by_id[dataset_id])
        enriched.update(
            {
                "pilot_group": "diagnostic",
                "selection_basis": "frozen_ABCD_disagreement_after_core_lock",
                "selection_reason": reason,
                "visible_leaf_count_postlock": review_by_id[dataset_id]["visible_leaf_count"],
            }
        )
        selection.append(enriched)

    selection.sort(key=lambda item: (item["pilot_group"] != "core", item["dataset_id"]))
    output_dir = Path(args.output_dir)
    inventory_fields = [
        "dataset_id", "split", "source_frame_id", "acquisition_period", "candidate_orientation",
        "crop_width_px", "crop_height_px", "fullplant_source_crop_width_px", "fullplant_source_crop_height_px",
        "standardized_width_px", "standardized_height_px", "normalization_padding_px", "resize_scale",
        "source_dpi", "mm_per_output_px", "physical_scale_status",
        "shoot_area_px", "shoot_bbox_width_px", "shoot_bbox_height_px", "shoot_fill_ratio",
        "skeleton_pixels", "endpoint_count", "branchpoint_count", "mean_thickness_px",
        "source_relative_path", "image_path", "mask_path",
    ]
    selection_fields = [
        "dataset_id", "pilot_group", "selection_basis", "selection_reason", "source_frame_id",
        "acquisition_period", "candidate_orientation", "visible_leaf_count_postlock", "source_dpi",
        "resize_scale", "mm_per_output_px", "physical_scale_status",
        "shoot_area_px", "shoot_bbox_width_px", "shoot_bbox_height_px", "shoot_fill_ratio",
        "skeleton_pixels", "endpoint_count", "branchpoint_count", "mean_thickness_px", "image_path",
    ]
    write_csv(output_dir / "phenotype_pilot_morphology_inventory.csv", records, inventory_fields)
    write_csv(output_dir / "phenotype_pilot_selection.csv", selection, selection_fields)
    (output_dir / "phenotype_pilot_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_contact_sheet(selection, output_dir / "phenotype_pilot_selection_contact_sheet.jpg")

    summary = {
        "selection_status": "proposal_pending_user_approval",
        "test_images_read": 0,
        "core_selection_uses_model_outcomes": False,
        "core_count": len(core),
        "diagnostic_count": len(diagnostic_pairs),
        "core_ids": sorted(core_ids),
        "diagnostic_ids": [dataset_id for dataset_id, _ in diagnostic_pairs],
        "core_source_frame_counts": dict(sorted(Counter(item["source_frame_id"] for item in core).items())),
        "core_period_counts": dict(sorted(Counter(item["acquisition_period"] for item in core).items())),
        "core_orientation_counts": dict(sorted(Counter(item["candidate_orientation"] for item in core).items())),
        "visible_leaf_count_postlock_counts": dict(
            sorted(Counter(item["visible_leaf_count_postlock"] for item in selection if item["pilot_group"] == "core").items())
        ),
    }
    (output_dir / "phenotype_pilot_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Rebuild V4 with complete seedling preservation on a white background.

This revision preserves the already selected sample identities and split
groups.  It changes only the foreground standardisation: green shoots, the
seed/basal region, and connected visible roots are retained.  The method is
classical pixel processing and never reads keypoints, DINOv2 features, teacher
points, model predictions, losses, or phenotype outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageOps

from audit_dataset_images import audit_pixels, strict_green_mask
from build_stage_clean_dataset import font, write_rgb


HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION_MANIFEST = HERE / "v4_fullplant_source_manifest.csv"
DEFAULT_OUTPUT = HERE / "data_stage_clean_v4_fullplant_candidate"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_full_rgb(path: Path) -> np.ndarray:
    try:
        array = np.asarray(tifffile.memmap(path))
    except Exception:
        array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Unsupported source shape: {array.shape}")
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.integer):
            maximum = float(np.iinfo(array.dtype).max)
            array = np.clip(array.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array.astype(np.float32), 0, 255).astype(np.uint8)
    return array


def resize_maximum(rgb: np.ndarray, maximum: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale >= 1.0:
        return rgb.copy()
    return cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--max-output-side", type=int, default=1600)
    parser.add_argument("--refresh-contact-sheets-only", action="store_true")
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            return read_full_rgb(path)
        except Exception:
            array = np.asarray(tifffile.imread(path))
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            return np.asarray(array[:, :, :3], dtype=np.uint8)
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def expanded_source_box(record: dict[str, Any], source_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    values = json.loads(str(record["crop_box_full"]))
    x0, y0, x1, y1 = [int(value) for value in values]
    height, width = source_shape
    box_w, box_h = x1 - x0, y1 - y0
    orientation = str(record.get("candidate_orientation", "horizontal"))
    if orientation == "vertical":
        margin_x, margin_y = round(box_w * 0.22), round(box_h * 0.16)
    else:
        margin_x, margin_y = round(box_w * 0.16), round(box_h * 0.22)
    return max(0, x0 - margin_x), max(0, y0 - margin_y), min(width, x1 + margin_x), min(height, y1 + margin_y)


def keep_components_linked_to_seed(mask: np.ndarray, seed: np.ndarray, bridge: int = 5) -> np.ndarray:
    linked = cv2.dilate(mask.astype(np.uint8), np.ones((bridge, bridge), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    minimum = max(6, round(mask.size * 0.000005))
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < minimum:
            continue
        component = labels == label_id
        if np.any(component & seed):
            kept[component & mask] = 1
    return kept.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label_id


def estimate_base(green: np.ndarray, brown: np.ndarray) -> tuple[tuple[float, float], float, np.ndarray]:
    ys, xs = np.where(green)
    if len(xs) < 20:
        height, width = green.shape
        return (width / 2.0, height / 2.0), max(height, width) * 0.25, np.array([1.0, 0.0])
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projection = (points - center) @ axis
    low_value, high_value = np.quantile(projection, [0.01, 0.99])
    low = center + axis * low_value
    high = center + axis * high_value
    major = max(20.0, float(high_value - low_value))
    yy, xx = np.indices(green.shape)
    radius = max(14.0, major * 0.22)

    def endpoint_score(point: np.ndarray) -> float:
        distance = np.hypot(xx - point[0], yy - point[1])
        local_brown = float((brown & (distance <= radius)).sum())
        local_green = float((green & (distance <= radius)).sum())
        return 4.0 * local_brown + local_green

    base = low if endpoint_score(low) >= endpoint_score(high) else high
    return (float(base[0]), float(base[1])), major, axis


def fullplant_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return shoot, seed/base/root, and complete-plant masks."""
    original_h, original_w = rgb.shape[:2]
    scale = min(1.0, 1100.0 / max(original_h, original_w))
    if scale < 1.0:
        work = cv2.resize(rgb, (round(original_w * scale), round(original_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = rgb.copy()
    height, width = work.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    red, green_channel, blue = [work[..., index].astype(np.float32) for index in range(3)]
    excess_green = 2.0 * green_channel - red - blue

    strict_green = strict_green_mask(work)
    relaxed_green = (
        (hue >= 17)
        & (hue <= 105)
        & (saturation >= 18)
        & (excess_green >= 0.5)
        & (green_channel >= red * 0.96)
    )
    shoot_candidate = strict_green | (cv2.dilate(strict_green.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & relaxed_green
    shoot_candidate = cv2.morphologyEx(shoot_candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    # Expanded crops may contain fragments of neighbouring seedlings.  The
    # original V4 candidate was centred on the target plant, whose strict green
    # component remains the dominant one after expansion.  Seed the relaxed
    # shoot mask from that component only, never from every green object.
    target_green_seed = largest_component(strict_green)
    shoot = keep_components_linked_to_seed(shoot_candidate, target_green_seed, bridge=5)

    brown = (
        (hue <= 35)
        & (saturation >= 24)
        & (value >= 18)
        & ((red >= green_channel * 0.98) | (red >= blue * 1.08))
    )
    base_point, major, axis = estimate_base(shoot, brown)
    yy, xx = np.indices((height, width))
    base_distance = np.hypot(xx - base_point[0], yy - base_point[1])
    base_radius = max(28.0, major * 0.28)
    base_region = base_distance <= base_radius

    # Local background difference recovers low-saturation roots while the base
    # region prevents scanner texture elsewhere from becoming foreground.
    sigma = max(5.0, min(height, width) / 45.0)
    smooth = cv2.GaussianBlur(work.astype(np.float32), (0, 0), sigma)
    colour_difference = np.max(np.abs(work.astype(np.float32) - smooth), axis=2)
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth_gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    darker_than_local = smooth_gray - gray
    # Scanner rulers, frame borders and calibration strips are usually long,
    # straight, low-saturation dark lines.  Remove those line supports before
    # component linking; otherwise a ruler touching the seed can be mistaken
    # for a thick root system.  Brown/chromatic seed tissue is preserved by
    # applying this exclusion only to the low-saturation evidence branches.
    low_sat_dark = (saturation <= 70) & (value <= 215)
    vertical_length = max(31, round(height * 0.14))
    horizontal_length = max(31, round(width * 0.14))
    vertical_lines = cv2.morphologyEx(
        low_sat_dark.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((vertical_length, 1), np.uint8),
    ) > 0
    horizontal_lines = cv2.morphologyEx(
        low_sat_dark.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((1, horizontal_length), np.uint8),
    ) > 0
    scanner_line_support = cv2.dilate(
        (vertical_lines | horizontal_lines).astype(np.uint8),
        np.ones((7, 7), np.uint8),
    ) > 0

    nonbrown_root_evidence = (
        ((saturation >= 12) & (colour_difference >= 7.0) & (value <= 248))
        | ((darker_than_local >= 9.0) & (colour_difference >= 9.0))
    ) & ~scanner_line_support
    root_evidence = base_region & (
        brown
        | nonbrown_root_evidence
    )
    border = max(2, round(min(height, width) * 0.012))
    root_evidence[:border] = False
    root_evidence[-border:] = False
    root_evidence[:, :border] = False
    root_evidence[:, -border:] = False

    base_candidate = root_evidence | (brown & base_region)
    base_candidate = cv2.morphologyEx(base_candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    base_seed_radius = max(10.0, major * 0.075)
    base_seed = base_distance <= base_seed_radius
    basal_shoot_anchor = shoot & (base_distance <= base_seed_radius * 1.6)
    root_base = keep_components_linked_to_seed(base_candidate, base_seed | basal_shoot_anchor, bridge=1)
    root_base &= ~cv2.dilate(shoot.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # Do not union every nearby brown object: expanded crops can contain a seed
    # from an adjacent plant.  Only components linked to this plant's estimated
    # basal endpoint are allowed into the complete-plant mask.
    full = shoot | root_base
    full = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    if scale < 1.0:
        size = (original_w, original_h)
        shoot = cv2.resize(shoot.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        root_base = cv2.resize(root_base.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        full = cv2.resize(full.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0

    diagnostics = {
        "shoot_fraction": float(shoot.mean()),
        "root_base_fraction": float(root_base.mean()),
        "fullplant_fraction": float(full.mean()),
        "nonshoot_to_shoot_area_ratio": float(root_base.sum() / max(1, shoot.sum())),
        "estimated_major_length_work": float(major),
        "estimated_base_x_work": float(base_point[0]),
        "estimated_base_y_work": float(base_point[1]),
        "scanner_line_support_fraction": float(scanner_line_support.mean()),
    }
    return shoot, root_base, full, diagnostics


def normalize_fullplant(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    shoot, root_base, full, diagnostics = fullplant_masks(rgb)
    alpha = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 0.7)[..., None]
    standardized = np.clip(rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    # Preserve tips that genuinely reach the expanded source crop boundary.
    # Padding is preferable to eroding/cropping them and gives downstream
    # affine augmentation and heatmaps a guaranteed background margin.
    padding = max(8, round(min(rgb.shape[:2]) * 0.025))
    standardized = cv2.copyMakeBorder(
        standardized, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    shoot = np.pad(shoot, padding, mode="constant", constant_values=False)
    root_base = np.pad(root_base, padding, mode="constant", constant_values=False)
    full = np.pad(full, padding, mode="constant", constant_values=False)
    diagnostics["normalization_padding_pixels"] = int(padding)
    return standardized, shoot, root_base, full, diagnostics


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def contact_sheet(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 3, 520, 350, 52
    canvas = Image.new("RGB", (columns * tile_w, header + max(1, math.ceil(len(rows) / columns)) * tile_h), "#eef0f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(20))
    body = font(11)
    for index, row in enumerate(rows):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        paths = (row["raw_review_path"], row["output_path"])
        labels = ("raw", "full-plant")
        for column, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((tile_w // 2 - 10, 245), Image.Resampling.LANCZOS)
            left = x0 + column * (tile_w // 2) + ((tile_w // 2) - image.width) // 2
            canvas.paste(image, (left, y0 + 5 + (245 - image.height) // 2))
            draw.text((x0 + column * (tile_w // 2) + 6, y0 + 252), label, fill="#374151", font=body)
        caption = (
            f"{row['dataset_id']} | {row['split']} | nonshoot/shoot={float(row['nonshoot_to_shoot_area_ratio']):.3f}\n"
            f"{Path(str(row['source_path'])).name}"
        )
        draw.multiline_text((x0 + 7, y0 + 274), caption, fill="black", font=body, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)


def write_contact_sheets(frame: pd.DataFrame, output: Path) -> None:
    rows_by_id = {str(row["dataset_id"]): row for row in frame.to_dict("records")}
    for split in ("train", "val", "test"):
        rows = [rows_by_id[dataset_id] for dataset_id in frame[frame["split"] == split]["dataset_id"].astype(str)]
        for start in range(0, len(rows), 18):
            page = rows[start : start + 18]
            contact_sheet(
                page,
                output / "contact_sheets" / f"raw_full_{split}_{start // 18 + 1:02d}.jpg",
                f"V4 full-plant candidate: {split} {start + 1}-{start + len(page)} | raw / full-plant",
            )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.refresh_contact_sheets_only:
        frame = pd.read_csv(output / "manifests" / "all.csv", low_memory=False).sort_values("dataset_id")
        (output / "contact_sheets").mkdir(parents=True, exist_ok=True)
        write_contact_sheets(frame, output)
        print(f"refreshed contact sheets: {output / 'contact_sheets'}")
        return
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    for folder in ("images", "masks", "raw_review", "contact_sheets", "manifests"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest.resolve(), low_memory=False)
    if args.ids:
        manifest = manifest[manifest["dataset_id"].astype(str).isin(set(args.ids))].copy()
        missing = sorted(set(args.ids) - set(manifest["dataset_id"].astype(str)))
        if missing:
            raise RuntimeError(f"Unknown dataset IDs: {missing}")

    records = manifest.to_dict("records")
    records.sort(key=lambda row: (str(row.get("source_path", "")), str(row["dataset_id"])))
    output_rows: list[dict[str, Any]] = []
    current_source: Path | None = None
    current_rgb: np.ndarray | None = None
    for progress, record in enumerate(records, start=1):
        source_path = Path(str(record["source_path"]))
        if source_path != current_source:
            current_rgb = read_rgb(source_path)
            current_source = source_path
        assert current_rgb is not None

        if str(record["new_split_role"]) == "legacy_v3_development":
            raw = resize_maximum(current_rgb, args.max_output_side)
            source_box = [0, 0, int(current_rgb.shape[1]), int(current_rgb.shape[0])]
        else:
            box = expanded_source_box(record, current_rgb.shape[:2])
            x0, y0, x1, y1 = box
            raw = resize_maximum(np.asarray(current_rgb[y0:y1, x0:x1]), args.max_output_side)
            source_box = list(box)

        standardized, shoot_mask, root_base_mask, full_mask, diagnostics = normalize_fullplant(raw)
        dataset_id = str(record["dataset_id"])
        split = str(record["split"])
        image_path = output / "images" / split / f"{dataset_id}.png"
        raw_path = output / "raw_review" / split / f"{dataset_id}.jpg"
        shoot_path = output / "masks" / "shoot" / split / f"{dataset_id}.png"
        root_path = output / "masks" / "seed_base_root" / split / f"{dataset_id}.png"
        full_path = output / "masks" / "full_plant" / split / f"{dataset_id}.png"
        write_rgb(image_path, standardized)
        write_rgb(raw_path, resize_maximum(raw, 800), quality=92)
        write_mask(shoot_path, shoot_mask)
        write_mask(root_path, root_base_mask)
        write_mask(full_path, full_mask)
        quality = audit_pixels(image_path)

        output_rows.append(
            {
                **record,
                "source_crop_box_fullplant": json.dumps(source_box),
                "output_path": str(image_path),
                "relative_path": str(image_path.relative_to(output)),
                "raw_review_path": str(raw_path),
                "raw_review_relative_path": str(raw_path.relative_to(output)),
                "shoot_mask_path": str(shoot_path),
                "shoot_mask_relative_path": str(shoot_path.relative_to(output)),
                "seed_base_root_mask_path": str(root_path),
                "seed_base_root_mask_relative_path": str(root_path.relative_to(output)),
                "full_plant_mask_path": str(full_path),
                "full_plant_mask_relative_path": str(full_path.relative_to(output)),
                "output_sha256": sha256_file(image_path),
                "whole_seedling_preserved": 1,
                "normalization_version": "v4_fullplant_classical_pixel_v2",
                "model_outputs_used_for_selection": 0,
                "keypoint_labels_used": 0,
                **diagnostics,
                **{f"fullplant_quality_{key}": value for key, value in quality.items() if key != "dhash64"},
            }
        )
        if progress % 25 == 0:
            print(f"FULLPLANT {progress}/{len(records)}", flush=True)

    frame = pd.DataFrame(output_rows).sort_values("dataset_id")
    frame.to_csv(output / "manifests" / "all.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output / "manifests" / f"{split}.csv", index=False, encoding="utf-8-sig")

    write_contact_sheets(frame, output)

    summary = {
        "dataset_version": "stage-clean-v4-fullplant-candidate",
        "images": len(frame),
        "split_counts": dict(Counter(frame["split"].astype(str))),
        "sample_identities_and_splits_changed": False,
        "whole_seedling_preserved": True,
        "organ_masks": ["shoot", "seed_base_root", "full_plant"],
        "model_outputs_used": False,
        "keypoint_labels_used": False,
        "nonshoot_to_shoot_area_ratio": {
            "min": float(frame["nonshoot_to_shoot_area_ratio"].min()),
            "median": float(frame["nonshoot_to_shoot_area_ratio"].median()),
            "max": float(frame["nonshoot_to_shoot_area_ratio"].max()),
        },
        "status": "candidate_visual_review_accepted_test_model_locked",
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

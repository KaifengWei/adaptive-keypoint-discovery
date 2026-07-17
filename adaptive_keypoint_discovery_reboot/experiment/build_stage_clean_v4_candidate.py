#!/usr/bin/env python
"""Build the model-blind V4 single-seedling candidate dataset.

The script reads only original image pixels and acquisition paths.  It never
reads DINOv2 features, G1-prime teacher points, keypoint annotations, training
losses, or phenotype outputs.  Existing V3 images are retained as development
training data only; every V4 validation/test image is newly cropped from an
original scanner frame.

Scanner frames contain several seedlings.  Candidate shoots are found with a
strict vegetation mask and orientation-specific morphology, cropped with a
margin, and standardized to white using the auditable classical routine from
``build_stage_clean_dataset.py``.  All crops from one source directory are
assigned to one split, and no crop from an included source frame can appear in
another split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageOps

from audit_dataset_images import audit_pixels, dhash64, load_rgb_thumbnail, strict_green_mask
from build_stage_clean_dataset import contact_sheet, font, normalize_to_white, write_rgb


HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "data_source_reaudit" / "all_images_inventory.csv"
LEGACY_V3 = HERE / "data_stage_clean_v3"
DEFAULT_OUTPUT = HERE / "data_stage_clean_v4_candidate"
SEED = "20260717-stage-clean-v4-model-blind"
VERSION = "stage-clean-v4-candidate"
TARGETS = {"train_new": 122, "val": 40, "test": 40}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail-maximum", type=int, default=1400)
    parser.add_argument("--max-output-side", type=int, default=1600)
    parser.add_argument("--train-new", type=int, default=TARGETS["train_new"])
    parser.add_argument("--val", type=int, default=TARGETS["val"])
    parser.add_argument("--test", type=int, default=TARGETS["test"])
    parser.add_argument("--max-audit-frames", type=int, default=0)
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash64(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.reshape(-1)
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def foreground_crop(rgb: np.ndarray) -> np.ndarray:
    mask = np.any(rgb < 248, axis=2)
    if not np.any(mask):
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_y = max(2, round((y1 - y0) * 0.05))
    margin_x = max(2, round((x1 - x0) * 0.05))
    return rgb[
        max(0, y0 - margin_y) : min(rgb.shape[0], y1 + margin_y),
        max(0, x0 - margin_x) : min(rgb.shape[1], x1 + margin_x),
    ]


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def scanner_metadata(relative_path: str) -> dict[str, str]:
    parts = Path(relative_path).parts
    parents = list(parts[:-1])
    split_group = str(Path(*parents)).lower()
    period = parents[-1] if parents else "unknown"
    period_match = re.search(r"(?:^|[^0-9])((?:04|05|06)\d{2})(?:[^0-9]|$)", period)
    if period_match:
        period = period_match.group(1)
    campaign = parents[1] if len(parents) > 1 else (parents[0] if parents else "unknown")
    return {
        "split_group": f"scanner_directory::{split_group}",
        "acquisition_period": period.lower(),
        "campaign": campaign.lower(),
    }


def detect_shoot_candidates(rgb: np.ndarray) -> list[dict[str, Any]]:
    """Find elongated vegetation structures without any learned feature."""
    raw = strict_green_mask(rgb).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    height, width = raw.shape
    candidates: list[dict[str, Any]] = []
    kernels = (("horizontal", (55, 9)), ("vertical", (9, 55)))
    for orientation, (kernel_w, kernel_h) in kernels:
        linked = cv2.dilate(raw, np.ones((kernel_h, kernel_w), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
        for label_id in range(1, count):
            linked_x = int(stats[label_id, cv2.CC_STAT_LEFT])
            linked_y = int(stats[label_id, cv2.CC_STAT_TOP])
            linked_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            linked_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if int(stats[label_id, cv2.CC_STAT_AREA]) < 100:
                continue
            local_labels = labels[linked_y : linked_y + linked_h, linked_x : linked_x + linked_w]
            local_raw = raw[linked_y : linked_y + linked_h, linked_x : linked_x + linked_w]
            ys, xs = np.where((local_labels == label_id) & (local_raw > 0))
            green_pixels = int(len(xs))
            if green_pixels < 80:
                continue
            xs = xs + linked_x
            ys = ys + linked_y
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            box_w, box_h = x1 - x0, y1 - y0
            major, minor = max(box_w, box_h), min(box_w, box_h)
            elongation = major / max(1, minor)
            if major < max(60, round(max(height, width) * 0.045)):
                continue
            if minor < max(12, round(min(height, width) * 0.010)):
                continue
            if elongation < 1.8:
                continue
            if orientation == "horizontal" and box_w < box_h * 1.55:
                continue
            if orientation == "vertical" and box_h < box_w * 1.55:
                continue
            # Thin scanner-bed edges can be chromatically green but are not
            # compact vegetation structures.
            density = green_pixels / float(max(1, box_w * box_h))
            if density < 0.025:
                continue
            candidates.append(
                {
                    "orientation": orientation,
                    "box": (x0, y0, x1, y1),
                    "green_pixels": green_pixels,
                    "density": density,
                    "elongation": elongation,
                    "score": major + 0.03 * green_pixels,
                }
            )

    # Horizontal and vertical linking may describe the same seedling.  Keep the
    # stronger description and suppress boxes that substantially overlap.
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-float(item["score"]), item["box"])):
        x0, y0, x1, y1 = candidate["box"]
        area = (x1 - x0) * (y1 - y0)
        duplicate = False
        for existing in kept:
            u0, v0, u1, v1 = existing["box"]
            other_area = (u1 - u0) * (v1 - v0)
            intersection = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
            if intersection / max(1, min(area, other_area)) > 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda item: (item["box"][1], item["box"][0], item["orientation"]))
    for index, candidate in enumerate(kept, start=1):
        candidate["candidate_index"] = index
    return kept


def audit_scanner_frames(inventory: pd.DataFrame, thumbnail_maximum: int, limit: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scanner = inventory[
        inventory["source_family"].eq("scanner_collection")
        & inventory["extension"].astype(str).str.lower().isin({".tif", ".tiff"})
    ].copy()
    scanner = scanner.sort_values("relative_path")
    if limit > 0:
        scanner = scanner.head(limit)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for progress, source in enumerate(scanner.to_dict("records"), start=1):
        path = Path(str(source["path"]))
        metadata = scanner_metadata(str(source["relative_path"]))
        try:
            rgb, loaded = load_rgb_thumbnail(path, maximum=thumbnail_maximum)
            candidates = detect_shoot_candidates(rgb)
            rows.append(
                {
                    "candidate_id": source["candidate_id"],
                    "source_path": str(path),
                    "relative_path": source["relative_path"],
                    **metadata,
                    "thumbnail_width": rgb.shape[1],
                    "thumbnail_height": rgb.shape[0],
                    "original_width": loaded["original_width"],
                    "original_height": loaded["original_height"],
                    "detected_shoots": len(candidates),
                    "candidate_boxes_json": json.dumps(candidates, ensure_ascii=False),
                    "pixel_only_audit": 1,
                    "model_outputs_used": 0,
                }
            )
        except Exception as error:
            errors.append({"source_path": str(path), "relative_path": source["relative_path"], "error": repr(error)})
        if progress % 50 == 0:
            print(f"FRAME_AUDIT {progress}/{len(scanner)}", flush=True)
    return rows, errors


def group_capacities(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        if 3 <= int(row["detected_shoots"]) <= 28:
            grouped[str(row["split_group"])].append(row)
    result: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: stable_hash(str(row["relative_path"])))
        periods = Counter(str(row["acquisition_period"]) for row in rows)
        result.append(
            {
                "split_group": group,
                "frames": ordered,
                "estimated_capacity": sum(int(row["detected_shoots"]) for row in ordered),
                "period": periods.most_common(1)[0][0],
                "campaign": Counter(str(row["campaign"]) for row in rows).most_common(1)[0][0],
            }
        )
    return result


def choose_holdout_groups(groups: list[dict[str, Any]], target: int, forbidden: set[str], label: str) -> list[str]:
    eligible = [group for group in groups if group["split_group"] not in forbidden and int(group["estimated_capacity"]) >= 5]
    periods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in eligible:
        periods[str(group["period"])].append(group)
    for period in periods:
        periods[period].sort(key=lambda group: stable_hash(f"{label}|{group['split_group']}"))
    period_order = sorted(periods, key=lambda period: stable_hash(f"{label}|period|{period}"))
    queues = {period: deque(values) for period, values in periods.items()}
    chosen: list[str] = []
    capacity = 0
    chosen_periods: set[str] = set()
    chosen_campaigns: set[str] = set()
    minimum_groups = min(4, len(eligible))
    minimum_periods = min(3, len(periods))
    minimum_campaigns = min(2, len({str(group["campaign"]) for group in eligible}))

    def enough() -> bool:
        return (
            capacity >= math.ceil(target * 1.35)
            and len(chosen) >= minimum_groups
            and len(chosen_periods) >= minimum_periods
            and len(chosen_campaigns) >= minimum_campaigns
        )

    while not enough():
        added = False
        for period in period_order:
            queue = queues[period]
            if not queue:
                continue
            group = queue.popleft()
            chosen.append(str(group["split_group"]))
            capacity += int(group["estimated_capacity"])
            chosen_periods.add(str(group["period"]))
            chosen_campaigns.add(str(group["campaign"]))
            added = True
            if enough():
                break
        if not added:
            break
    if capacity < target:
        raise RuntimeError(f"Insufficient model-blind source capacity for {label}: {capacity} < {target}")
    return chosen


def round_robin_frames(groups: list[dict[str, Any]], chosen: set[str]) -> list[dict[str, Any]]:
    queues: list[deque[dict[str, Any]]] = []
    selected = [group for group in groups if group["split_group"] in chosen]
    selected.sort(key=lambda group: stable_hash(str(group["split_group"])))
    for group in selected:
        queues.append(deque(group["frames"]))
    frames: list[dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                frames.append(queue.popleft())
    return frames


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


def expanded_full_box(frame: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in candidate["box"]]
    box_w, box_h = x1 - x0, y1 - y0
    if candidate["orientation"] == "horizontal":
        margin_x, margin_y = max(14.0, box_w * 0.07), max(16.0, box_h * 0.55)
    else:
        margin_x, margin_y = max(16.0, box_w * 0.55), max(14.0, box_h * 0.07)
    x0, x1 = x0 - margin_x, x1 + margin_x
    y0, y1 = y0 - margin_y, y1 + margin_y
    scale_x = float(frame["original_width"]) / float(frame["thumbnail_width"])
    scale_y = float(frame["original_height"]) / float(frame["thumbnail_height"])
    full_x0 = max(0, math.floor(x0 * scale_x))
    full_y0 = max(0, math.floor(y0 * scale_y))
    full_x1 = min(int(frame["original_width"]), math.ceil(x1 * scale_x))
    full_y1 = min(int(frame["original_height"]), math.ceil(y1 * scale_y))
    return full_x0, full_y0, full_x1, full_y1


def resize_maximum(rgb: np.ndarray, maximum: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale >= 1.0:
        return rgb.copy()
    return cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def green_component_diagnostics(rgb: np.ndarray) -> dict[str, float | int]:
    mask = strict_green_mask(rgb).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum = max(8, round(mask.size * 0.00001))
    areas = sorted(
        [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum],
        reverse=True,
    )
    if not areas:
        return {
            "significant_green_components": 0,
            "second_to_first_green_area_ratio": 1.0,
            "secondary_to_first_green_area_ratio": 1.0,
        }
    second = areas[1] if len(areas) > 1 else 0
    secondary = sum(areas[1:])
    return {
        "significant_green_components": len(areas),
        "second_to_first_green_area_ratio": second / areas[0],
        "secondary_to_first_green_area_ratio": secondary / areas[0],
    }


def acceptable_output(metrics: dict[str, Any]) -> tuple[bool, str]:
    checks = [
        (float(metrics["strict_green_fraction"]) >= 0.003, "green_fraction_too_small"),
        (float(metrics["strict_green_fraction"]) <= 0.38, "green_fraction_too_large"),
        (float(metrics["nonwhite_fraction"]) <= 0.42, "residual_background_too_large"),
        (int(metrics["green_bbox_touches_border"]) == 0, "shoot_touches_crop_border"),
        (float(metrics["blur_laplacian_thumbnail"]) >= 25.0, "blur_below_threshold"),
        (int(metrics["green_components"]) <= 14, "too_many_green_components"),
        (float(metrics["green_bbox_fraction"]) <= 0.78, "foreground_bbox_too_large"),
        (
            float(metrics["secondary_to_first_green_area_ratio"]) <= 0.06,
            "significant_disconnected_shoot_or_second_seedling",
        ),
        (float(metrics["green_preservation_ratio"]) >= 0.40, "shoot_structure_lost_during_standardization"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    return not failures, ";".join(failures)


def legacy_hashes() -> tuple[list[dict[str, str]], pd.DataFrame]:
    frame = pd.read_csv(LEGACY_V3 / "manifests" / "all.csv", low_memory=False)
    hashes: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        path = Path(str(row["output_path"]))
        with Image.open(path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        crop = foreground_crop(rgb)
        hashes.append({"dataset_id": str(row["dataset_id"]), "dhash": dhash64(crop), "phash": phash64(crop)})
    return hashes, frame


def near_legacy_match(dhash_value: str, phash_value: str, legacy: list[dict[str, str]]) -> tuple[bool, str, int, int]:
    best_id, best_dhash, best_phash = "", 65, 65
    for row in legacy:
        d_distance = hamming(dhash_value, row["dhash"])
        p_distance = hamming(phash_value, row["phash"])
        if d_distance + p_distance < best_dhash + best_phash:
            best_id, best_dhash, best_phash = row["dataset_id"], d_distance, p_distance
    return best_dhash <= 3 and best_phash <= 8, best_id, best_dhash, best_phash


def extract_split(
    split: str,
    target: int,
    frames: Iterable[dict[str, Any]],
    output: Path,
    legacy: list[dict[str, str]],
    seen_exact: set[str],
    max_output_side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for frame in frames:
        if len(accepted) >= target:
            break
        source_path = Path(str(frame["source_path"]))
        try:
            full = read_full_rgb(source_path)
        except Exception as error:
            rejected.append({"split": split, "source_path": str(source_path), "reason": "full_read_error", "details": repr(error)})
            continue
        candidates = json.loads(str(frame["candidate_boxes_json"]))
        candidates.sort(key=lambda item: stable_hash(f"{frame['relative_path']}|{item['candidate_index']}"))
        for candidate in candidates:
            if len(accepted) >= target:
                break
            full_box = expanded_full_box(frame, candidate)
            x0, y0, x1, y1 = full_box
            raw = resize_maximum(np.asarray(full[y0:y1, x0:x1]), maximum=max_output_side)
            if min(raw.shape[:2]) < 40:
                rejected.append({"split": split, "source_path": str(source_path), "reason": "crop_too_small", "candidate_index": candidate["candidate_index"]})
                continue
            standardized, diagnostics = normalize_to_white(raw)
            raw_green_pixels = int(strict_green_mask(raw).sum())
            standardized_green_pixels = int(strict_green_mask(standardized).sum())
            temporary = output / "_temporary" / f"{frame['candidate_id']}_{int(candidate['candidate_index']):02d}.png"
            write_rgb(temporary, standardized)
            metrics = audit_pixels(temporary)
            metrics.update(green_component_diagnostics(standardized))
            metrics.update(
                {
                    "raw_green_pixels": raw_green_pixels,
                    "standardized_green_pixels": standardized_green_pixels,
                    "green_preservation_ratio": standardized_green_pixels / max(1, raw_green_pixels),
                }
            )
            passed, reason = acceptable_output(metrics)
            crop = foreground_crop(standardized)
            crop_dhash, crop_phash = dhash64(crop), phash64(crop)
            duplicate_legacy, legacy_id, legacy_dhash_distance, legacy_phash_distance = near_legacy_match(crop_dhash, crop_phash, legacy)
            digest = sha256_file(temporary)
            if digest in seen_exact:
                passed, reason = False, "exact_duplicate_output"
            if split in {"val", "test"} and duplicate_legacy:
                passed, reason = False, f"near_legacy_{legacy_id}"
            base_record = {
                "split": "train" if split == "train_new" else split,
                "new_split_role": split,
                "source_frame_id": frame["candidate_id"],
                "source_path": str(source_path),
                "source_name": source_path.name,
                "source_relative_path": frame["relative_path"],
                "split_group": frame["split_group"],
                "acquisition_period": frame["acquisition_period"],
                "campaign": frame["campaign"],
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_orientation": candidate["orientation"],
                "candidate_box_thumbnail": json.dumps(candidate["box"]),
                "crop_box_full": json.dumps(full_box),
                "normalizer_foreground_fraction": diagnostics.get("foreground_fraction", float("nan")),
                "output_dhash64_foreground_crop": crop_dhash,
                "output_phash64_foreground_crop": crop_phash,
                "nearest_legacy_id": legacy_id,
                "nearest_legacy_dhash_distance": legacy_dhash_distance,
                "nearest_legacy_phash_distance": legacy_phash_distance,
                "keypoint_labels_used": 0,
                "model_outputs_used_for_selection": 0,
                "quality_tier": "V4_scanner_crop_pixel_QC",
                "background_operation": "shoot_only_standardization_scanner_crop",
                **{f"quality_{key}": value for key, value in metrics.items() if key != "dhash64"},
            }
            if not passed:
                temporary.unlink(missing_ok=True)
                rejected.append({**base_record, "reason": reason})
                continue
            dataset_id = f"v4_{split}_{len(accepted) + 1:04d}"
            destination = output / "images" / ("train" if split == "train_new" else split) / f"{dataset_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), str(destination))
            seen_exact.add(digest)
            raw_preview = resize_maximum(raw, maximum=700)
            preview_path = output / "review_raw_crops" / ("train" if split == "train_new" else split) / f"{dataset_id}.jpg"
            write_rgb(preview_path, raw_preview, quality=92)
            accepted.append(
                {
                    "dataset_id": dataset_id,
                    **base_record,
                    "data_role": "new_model_blind_scanner_crop",
                    "stage_label": "unassigned",
                    "stage_label_source": "not_inferred_from_appearance",
                    "output_path": str(destination),
                    "relative_path": str(destination.relative_to(output)),
                    "raw_review_path": str(preview_path),
                    "output_sha256": digest,
                    "manual_quality_review": "pending",
                    "manual_stage_review": "not_required_for_selection",
                    "manual_note": "",
                }
            )
        del full
    if len(accepted) < target:
        raise RuntimeError(f"Accepted only {len(accepted)}/{target} new images for {split}")
    return accepted, rejected


def copy_legacy_training(output: Path, frame: pd.DataFrame, seen_exact: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(frame.to_dict("records"), start=1):
        source_path = Path(str(source["output_path"]))
        dataset_id = f"v4_legacy_{index:04d}"
        destination = output / "images" / "train" / f"{dataset_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        digest = sha256_file(destination)
        if digest in seen_exact:
            destination.unlink(missing_ok=True)
            continue
        seen_exact.add(digest)
        rows.append(
            {
                "dataset_id": dataset_id,
                "split": "train",
                "new_split_role": "legacy_v3_development",
                "source_frame_id": "",
                "source_path": str(source.get("source_path", source_path)),
                "source_name": str(source.get("source_name", source_path.name)),
                "source_relative_path": str(source.get("relative_path", "")),
                "split_group": f"legacy_v3::{source.get('identity_group', source['dataset_id'])}",
                "acquisition_period": "legacy_v3",
                "campaign": "legacy_v3",
                "candidate_index": "",
                "candidate_orientation": "legacy",
                "candidate_box_thumbnail": "",
                "crop_box_full": "",
                "normalizer_foreground_fraction": source.get("foreground_fraction_from_normalizer", float("nan")),
                "output_dhash64_foreground_crop": source.get("output_dhash64_foreground_crop", ""),
                "output_phash64_foreground_crop": "",
                "nearest_legacy_id": source["dataset_id"],
                "nearest_legacy_dhash_distance": 0,
                "nearest_legacy_phash_distance": 0,
                "keypoint_labels_used": 0,
                "model_outputs_used_for_selection": 0,
                "quality_tier": str(source.get("quality_tier", "legacy_v3")),
                "background_operation": str(source.get("background_operation", "legacy_v3_preserved")),
                "data_role": "legacy_v3_development_train_only",
                "stage_label": source.get("stage_label", "unknown"),
                "stage_label_source": source.get("stage_label_source", "legacy_v3"),
                "output_path": str(destination),
                "relative_path": str(destination.relative_to(output)),
                "raw_review_path": "",
                "output_sha256": digest,
                "manual_quality_review": "legacy_v3_previously_reviewed",
                "manual_stage_review": "legacy_v3_metadata_status_preserved",
                "manual_note": "V3 val/test were previously viewed and are therefore training/development only in V4",
            }
        )
    return rows


def raw_standardized_sheet(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 4, 390, 330, 52
    page_rows = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * tile_w, header + page_rows * tile_h), "#eef0f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(20))
    body = font(11)
    for index, row in enumerate(rows):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        with Image.open(str(row["raw_review_path"])) as opened:
            raw = ImageOps.exif_transpose(opened).convert("RGB")
            raw.thumbnail((tile_w // 2 - 12, 238), Image.Resampling.LANCZOS)
        with Image.open(str(row["output_path"])) as opened:
            standardized = ImageOps.exif_transpose(opened).convert("RGB")
            standardized.thumbnail((tile_w // 2 - 12, 238), Image.Resampling.LANCZOS)
        canvas.paste(raw, (x0 + 5 + (tile_w // 2 - 8 - raw.width) // 2, y0 + 3 + (238 - raw.height) // 2))
        canvas.paste(standardized, (x0 + tile_w // 2 + (tile_w // 2 - 8 - standardized.width) // 2, y0 + 3 + (238 - standardized.height) // 2))
        caption = (
            f"{row['dataset_id']} | {row['acquisition_period']} | {row['candidate_orientation']}\n"
            f"raw crop (left) / standardized (right)\n{Path(str(row['source_path'])).name}"
        )
        draw.multiline_text((x0 + 6, y0 + 247), caption, fill="black", font=body, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    for folder in ("images", "manifests", "contact_sheets", "review_raw_crops", "_temporary"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(INVENTORY, low_memory=False)
    frame_rows, frame_errors = audit_scanner_frames(inventory, args.thumbnail_maximum, args.max_audit_frames)
    pd.DataFrame(frame_rows).drop(columns=["candidate_boxes_json"]).to_csv(
        output / "manifests" / "scanner_frame_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(frame_errors).to_csv(output / "manifests" / "scanner_frame_errors.csv", index=False, encoding="utf-8-sig")

    groups = group_capacities(frame_rows)
    test_groups = choose_holdout_groups(groups, args.test, set(), "test")
    val_groups = choose_holdout_groups(groups, args.val, set(test_groups), "val")
    train_groups = [group["split_group"] for group in groups if group["split_group"] not in set(test_groups + val_groups)]
    assignments = []
    for split, selected in (("test", test_groups), ("val", val_groups), ("train_new", train_groups)):
        for group in groups:
            if group["split_group"] in selected:
                assignments.append(
                    {
                        "split_group": group["split_group"],
                        "assigned_split": split,
                        "acquisition_period": group["period"],
                        "campaign": group["campaign"],
                        "estimated_capacity": group["estimated_capacity"],
                    }
                )
    pd.DataFrame(assignments).to_csv(output / "manifests" / "source_group_split_assignment.csv", index=False, encoding="utf-8-sig")

    legacy, legacy_frame = legacy_hashes()
    seen_exact: set[str] = set()
    legacy_rows = copy_legacy_training(output, legacy_frame, seen_exact)
    all_new: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    # Extract locked test first so later development outputs cannot influence it.
    for split, target, selected in (
        ("test", args.test, set(test_groups)),
        ("val", args.val, set(val_groups)),
        ("train_new", args.train_new, set(train_groups)),
    ):
        rows, failures = extract_split(
            split,
            target,
            round_robin_frames(groups, selected),
            output,
            legacy,
            seen_exact,
            args.max_output_side,
        )
        all_new.extend(rows)
        rejected.extend(failures)
        print(f"EXTRACTED {split}={len(rows)} rejected={len(failures)}", flush=True)

    all_rows = legacy_rows + all_new
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "manifests" / "all.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output / "manifests" / f"{split}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected).to_csv(output / "manifests" / "rejected_crops.csv", index=False, encoding="utf-8-sig")
    review = frame[frame["new_split_role"].isin({"val", "test"})][
        [
            "dataset_id",
            "split",
            "source_path",
            "split_group",
            "acquisition_period",
            "candidate_orientation",
            "output_sha256",
            "manual_quality_review",
            "manual_note",
        ]
    ].copy()
    review.to_csv(output / "manifests" / "锁定集人工复核表.csv", index=False, encoding="utf-8-sig")

    for split in ("train_new", "val", "test"):
        rows = [row for row in all_new if row["new_split_role"] == split]
        for start in range(0, len(rows), 20):
            raw_standardized_sheet(
                rows[start : start + 20],
                output / "contact_sheets" / f"raw_vs_standardized_{split}_{start // 20 + 1:02d}.jpg",
                f"{VERSION}: {split} {start + 1}-{start + len(rows[start : start + 20])} | pixels only, no model output",
            )
    # A compact normalized-only overview is useful after the raw/standardized
    # integrity sheets have been checked.
    for split in ("val", "test"):
        rows = [row for row in all_new if row["new_split_role"] == split]
        contact_sheet(rows, output / "contact_sheets" / f"normalized_{split}.jpg", f"{VERSION}: {split} normalized overview")

    shutil.rmtree(output / "_temporary", ignore_errors=True)
    summary = {
        "dataset_version": VERSION,
        "images": len(frame),
        "split_counts": dict(Counter(frame["split"].astype(str))),
        "new_role_counts": dict(Counter(frame["new_split_role"].astype(str))),
        "acquisition_period_counts_new": dict(Counter(str(row["acquisition_period"]) for row in all_new)),
        "legacy_v3_training_only": len(legacy_rows),
        "new_scanner_crops": len(all_new),
        "scanner_frames_audited": len(frame_rows),
        "scanner_frame_errors": len(frame_errors),
        "rejected_new_crops": len(rejected),
        "keypoint_labels_used": False,
        "model_outputs_used_for_selection": False,
        "split_isolation": "source directory groups are disjoint; source frames and sibling crops never cross splits",
        "locked_test_status": "candidate_only_pending_human_raw_image_review; do not run any model before lock",
        "legacy_rule": "all 98 previously viewed V3 images are training/development only",
        "stage_rule": "acquisition period is recorded as metadata and is not presented as an appearance-inferred leaf-stage label",
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

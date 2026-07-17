#!/usr/bin/env python
"""Build a stage-aware, clean-background dataset for the G1-prime programme.

No keypoint label or model output is read.  Stage metadata comes from the
user-provided archive folders or the explicit ``leaf3`` filename.  Images with
non-clean backgrounds are normalized to white and are marked as processed,
never as native clean captures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_dataset_images import audit_pixels, dhash64


HERE = Path(__file__).resolve().parent
STAGE_AUDIT = HERE / "data_stage_source_audit" / "stage_source_audit.csv"
INVENTORY = HERE / "data_source_reaudit" / "all_images_inventory.csv"
G0_AUDIT = HERE / "data_g0" / "audit_manifest.csv"
OUTPUT = HERE / "data_stage_clean_v3_candidate"
SEED = "20260717-stage-clean-v3"
DATASET_VERSION = "stage-clean-v3-candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-blur", type=float, default=50.0)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray, quality: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def border_values(array: np.ndarray, thickness: int) -> np.ndarray:
    top = array[:thickness].reshape(-1, *array.shape[2:])
    bottom = array[-thickness:].reshape(-1, *array.shape[2:])
    left = array[thickness:-thickness, :thickness].reshape(-1, *array.shape[2:])
    right = array[thickness:-thickness, -thickness:].reshape(-1, *array.shape[2:])
    return np.concatenate([top, bottom, left, right], axis=0)


def keep_seed_connected(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    min_area = max(8, round(mask.size * 0.00002))
    for label_id in range(1, count):
        component = labels == label_id
        if int(stats[label_id, cv2.CC_STAT_AREA]) < min_area:
            continue
        if np.any(component & seed):
            kept[component] = 1
    return kept.astype(bool)


def normalize_to_white(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Background normalization using color seeds plus GrabCut.

    The procedure is deliberately classical and auditable.  It is not used as
    biological ground truth; its output is assigned quality tier B.
    """
    original_h, original_w = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(original_h, original_w))
    if scale < 1.0:
        work = cv2.resize(rgb, (round(original_w * scale), round(original_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = rgb.copy()
    h, w = work.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    red, green, blue = [work[..., index].astype(np.float32) for index in range(3)]
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    excess_green = 2.0 * green - red - blue
    thickness = max(3, round(min(h, w) * 0.035))
    border_sat = border_values(saturation[..., None], thickness).reshape(-1)
    border_exg = border_values(excess_green[..., None], thickness).reshape(-1)
    border_value = border_values(value[..., None], thickness).reshape(-1)
    sat_threshold = max(30.0, float(np.quantile(border_sat, 0.98)) + 5.0)
    exg_threshold = max(10.0, float(np.quantile(border_exg, 0.98)) + 4.0)

    green_seed = (
        (excess_green >= exg_threshold)
        & (green >= red * 0.98)
        & (green >= blue * 0.94)
        & ((saturation >= sat_threshold) | (green >= red + 8.0))
    )
    brown_seed = (red >= green * 1.04) & (green >= blue * 1.02) & (saturation >= max(35.0, sat_threshold * 0.75))
    colored_seed = green_seed | brown_seed
    near_colored = cv2.dilate(colored_seed.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    dark_connected = (value <= float(np.median(border_value)) - 28.0) & near_colored
    seed = colored_seed | dark_connected
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    if int(seed.sum()) < max(10, round(seed.size * 0.0002)):
        # Conservative fallback for very low-saturation shoots.
        seed = ((excess_green >= max(5.0, exg_threshold * 0.55)) & (green > red) & (green >= blue)).astype(bool)

    # V3 is explicitly a shoot-phenotype dataset.  Scanner roots, seeds and
    # low-saturation checkerboard remnants are not required for the target
    # traits and were a major source of false skeleton branches in V2.  Build a
    # strict vegetation core, reject small/square chromatic background blobs,
    # and only recover edge colours very close to that core.
    hue = hsv[..., 0].astype(np.float32)
    border_sat_995 = float(np.quantile(border_sat, 0.995))
    color_floor = max(30.0, border_sat_995 + 8.0)
    vegetation_raw = (
        (hue >= 24.0)
        & (hue <= 92.0)
        & (saturation >= color_floor)
        & (excess_green >= max(7.0, exg_threshold * 0.45))
        & (green >= red * 0.98)
        & (green >= blue * 0.92)
    )
    vegetation_raw = cv2.morphologyEx(
        vegetation_raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(vegetation_raw, 8)
    vegetation_core = np.zeros((h, w), dtype=np.uint8)
    min_component = max(18, round(h * w * 0.000025))
    min_span = max(8, round(min(h, w) * 0.018))
    component_rows: list[tuple[int, int]] = []
    for label_id in range(1, component_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        span = max(comp_w, comp_h)
        elongation = span / max(1, min(comp_w, comp_h))
        if area < min_component or span < min_span:
            continue
        if elongation < 1.45 and area < min_component * 7:
            continue
        component_rows.append((area, label_id))
    # A multi-leaf seedling may form several disconnected colour components;
    # keep at most the eight strongest components and reject tiny distant noise.
    for _, label_id in sorted(component_rows, reverse=True)[:8]:
        vegetation_core[labels == label_id] = 1
    if not np.any(vegetation_core):
        vegetation_core = keep_seed_connected(vegetation_raw > 0, green_seed).astype(np.uint8)

    near_core = cv2.dilate(vegetation_core, np.ones((9, 9), np.uint8)) > 0
    relaxed_green = (
        (hue >= 20.0)
        & (hue <= 100.0)
        & (saturation >= max(18.0, color_floor * 0.62))
        & (excess_green >= 1.5)
    )
    yellow_base = (
        (hue >= 12.0)
        & (hue < 30.0)
        & (saturation >= max(32.0, color_floor * 0.75))
        & (green >= blue * 1.02)
    )
    shoot_evidence = (vegetation_core > 0) | (near_core & (relaxed_green | yellow_base))
    # The earlier GrabCut stage is intentionally omitted.  On scanner images it
    # copied checkerboard texture into the probable foreground and did not add
    # useful shoot evidence beyond this chromatic mask.
    mask = shoot_evidence.copy()
    mask = keep_seed_connected(mask, vegetation_core > 0)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    if scale < 1.0:
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST) > 0

    # Slight feathering preserves antialiased leaf edges while producing a
    # genuinely white exterior.
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7)[..., None]
    normalized = np.clip(rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    diagnostics = {
        "foreground_fraction": float(mask.mean()),
        "seed_fraction": float(seed.mean()),
        "vegetation_core_fraction": float((vegetation_core > 0).mean()),
        "saturation_threshold": sat_threshold,
        "excess_green_threshold": exg_threshold,
    }
    return normalized, diagnostics


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def split_groups(rows: list[dict[str, Any]]) -> None:
    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = str(row["stage_label"])
        if stratum == "unknown":
            stratum = f"unknown_{row['complexity_bin']}"
        strata[stratum][str(row["identity_group"])].append(row)
    for stratum, groups in strata.items():
        ordered = sorted(groups, key=lambda key: stable_hash(f"{stratum}|{key}"))
        n = len(ordered)
        n_test = 1 if n >= 3 else 0
        n_val = 1 if n >= 4 else 0
        if n >= 10:
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
        for index, group in enumerate(ordered):
            role = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            for row in groups[group]:
                row["split"] = role


def contact_sheet(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 5, 300, 275, 48
    nrows = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * tile_w, header + nrows * tile_h), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(21))
    body = font(11)
    for index, row in enumerate(rows):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        with Image.open(str(row["output_path"])) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_w - 14, 205), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 3 + (205 - image.height) // 2))
        caption = (
            f"{row['dataset_id']} | {row['stage_label']} | {row['split']}\n"
            f"{row['source_name']}\n{row['quality_tier']} | {row['background_operation']}"
        )
        draw.multiline_text((x0 + 6, y0 + 212), caption, fill="black", font=body, spacing=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=93)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "contact_sheets").mkdir(parents=True, exist_ok=True)

    stage = pd.read_csv(STAGE_AUDIT)
    inventory = pd.read_csv(INVENTORY, low_memory=False)
    inventory = inventory[inventory["source_family"] == "images_400"].copy()
    inventory["source_name"] = inventory["path"].map(lambda value: Path(value).name)
    g0 = pd.read_csv(G0_AUDIT)
    native = inventory.merge(g0, on="source_name", how="inner", suffixes=("_inv", "_g0"))
    strict = native[
        (native["background_class_inv"] == "white_removed_candidate")
        & (native["green_bbox_touches_border"] == 0)
        & (native["border_white_fraction"] >= 0.98)
        & (native["quality_score"] >= 94)
        & (native["auto_review_flags"].fillna("") == "")
    ].copy()

    stage_names = set(stage["source_name"].astype(str))
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stage_counter = 0
    for _, source in stage.sort_values(["stage_label", "source_group", "source_name"]).iterrows():
        source_path = Path(str(source["path"]))
        original_blur = float(source["blur_laplacian_thumbnail"])
        if original_blur < args.min_blur:
            rejected.append({"source_path": str(source_path), "reason": "blur_below_threshold", "value": original_blur})
            continue
        rgb = read_rgb(source_path)
        native_clean = (
            source["background_class"] == "white_removed_candidate"
            and float(source["border_white_fraction"]) >= 0.98
            and float(source["clean_background_score"]) >= 7.0
        )
        # All images receive the same shoot-only standardisation.  Native-white
        # images retain tier A provenance; mixed-background sources remain tier
        # B.  The operation is never presented as an original acquisition.
        normalized, diagnostics = normalize_to_white(rgb)
        if native_clean:
            operation = "shoot_only_standardization_native_white"
            tier = "A_native_white_standardized"
        else:
            operation = "shoot_only_standardization_mixed_background"
            tier = "B_mixed_background_standardized"
        stage_counter += 1
        dataset_id = f"stagev3_{stage_counter:04d}"
        out_path = output / "images" / f"{dataset_id}.png"
        write_rgb(out_path, normalized)
        after = audit_pixels(out_path)
        stage_source = "archive_folder"
        confidence = "user_metadata"
        record = {
            "dataset_id": dataset_id,
            "source_path": str(source_path),
            "source_name": source["source_name"],
            "identity_group": f"stage_archive::{source['stage_label']}::{source['source_group']}",
            "stage_label": source["stage_label"],
            "stage_label_source": stage_source,
            "stage_label_confidence": confidence,
            "quality_tier": tier,
            "background_operation": operation,
            "original_blur": original_blur,
            "output_blur": after["blur_laplacian_thumbnail"],
            "output_background_class": after["background_class"],
            "output_white_fraction": after["white_fraction"],
            "output_border_white_fraction": after["border_white_fraction"],
            "output_green_fraction": after["strict_green_fraction"],
            "foreground_fraction_from_normalizer": diagnostics.get("foreground_fraction", float("nan")),
            "structural_complexity_score": float("nan"),
            "complexity_bin": "stage_metadata",
            "source_sha256": source["sha256"],
            "output_dhash64": dhash64(normalized),
            "output_path": str(out_path),
            "keypoint_labels_used": 0,
            "manual_keep": "",
            "manual_stage_confirm": "",
            "manual_note": "",
        }
        records.append(record)

    # Add independent, native-white images.  A matching stage-archive filename
    # is skipped so the stage-labelled version remains the single canonical row.
    native_pool = strict[~strict["source_name"].isin(stage_names)].copy()
    values = native_pool["structural_complexity_score"].astype(float)
    q1, q2 = values.quantile([0.33, 0.67]).tolist() if len(values) else (0.0, 0.0)
    for _, source in native_pool.sort_values(["structural_complexity_score", "source_name"]).iterrows():
        stage_counter += 1
        dataset_id = f"stagev3_{stage_counter:04d}"
        source_path = Path(str(source["path"]))
        out_path = output / "images" / f"{dataset_id}{source_path.suffix.lower()}"
        source_rgb = read_rgb(source_path)
        standardized, diagnostics = normalize_to_white(source_rgb)
        out_path = out_path.with_suffix(".png")
        write_rgb(out_path, standardized)
        after = audit_pixels(out_path)
        complexity = float(source["structural_complexity_score"])
        complexity_bin = "low" if complexity <= q1 else "mid" if complexity <= q2 else "high"
        if str(source["source_name"]).lower().startswith("leaf3 ("):
            stage_label = "gt2"
            label_source = "filename_leaf3"
            confidence = "name_proxy_needs_confirmation"
        else:
            stage_label = "unknown"
            label_source = "not_assigned"
            confidence = "unknown"
        rgb = read_rgb(out_path)
        records.append(
            {
                "dataset_id": dataset_id,
                "source_path": str(source_path),
                "source_name": source["source_name"],
                "identity_group": f"images400::{Path(str(source['source_name'])).stem}",
                "stage_label": stage_label,
                "stage_label_source": label_source,
                "stage_label_confidence": confidence,
                "quality_tier": "A_native_white_standardized",
                "background_operation": "shoot_only_standardization_native_white",
                "original_blur": float(source["blur_laplacian_thumbnail"]),
                "output_blur": float(source["blur_laplacian_thumbnail"]),
                "output_background_class": after["background_class"],
                "output_white_fraction": after["white_fraction"],
                "output_border_white_fraction": after["border_white_fraction"],
                "output_green_fraction": after["strict_green_fraction"],
                "foreground_fraction_from_normalizer": diagnostics.get("foreground_fraction", float("nan")),
                "structural_complexity_score": complexity,
                "complexity_bin": complexity_bin,
                "source_sha256": source["sha256"],
                "output_dhash64": dhash64(rgb),
                "output_path": str(out_path),
                "keypoint_labels_used": 0,
                "manual_keep": "",
                "manual_stage_confirm": "",
                "manual_note": "",
            }
        )

    # Exact and near-duplicate audit.  Same identity groups are allowed but must
    # remain in one split; cross-group near pairs are written for human review.
    exact_seen: dict[str, str] = {}
    exact_duplicates: list[dict[str, str]] = []
    for row in records:
        digest = hashlib.sha256(Path(row["output_path"]).read_bytes()).hexdigest()
        row["output_sha256"] = digest
        if digest in exact_seen:
            exact_duplicates.append({"dataset_id_a": exact_seen[digest], "dataset_id_b": row["dataset_id"], "reason": "exact_output"})
        else:
            exact_seen[digest] = row["dataset_id"]
    if exact_duplicates:
        raise RuntimeError(f"Exact duplicate outputs found: {exact_duplicates}")

    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        left_hash = int(str(left["output_dhash64"]), 16)
        for right in records[left_index + 1 :]:
            distance = (left_hash ^ int(str(right["output_dhash64"]), 16)).bit_count()
            if distance <= 6:
                near_pairs.append(
                    {
                        "dataset_id_a": left["dataset_id"],
                        "dataset_id_b": right["dataset_id"],
                        "dhash_hamming": distance,
                        "same_identity_group": int(left["identity_group"] == right["identity_group"]),
                        "manual_same_plant": "",
                    }
                )

    split_groups(records)
    # Move output files into the final split folders while keeping IDs stable.
    for row in records:
        old = Path(row["output_path"])
        new = output / "images" / str(row["split"]) / old.name
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        row["output_path"] = str(new)
        row["relative_path"] = str(new.relative_to(output))

    fieldnames = list(records[0])
    all_frame = pd.DataFrame(records)
    all_frame.to_csv(output / "manifests" / "all.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        all_frame[all_frame["split"] == split].to_csv(output / "manifests" / f"{split}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected).to_csv(output / "manifests" / "rejected.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(near_pairs).to_csv(output / "manifests" / "near_duplicate_review.csv", index=False, encoding="utf-8-sig")
    all_frame[["dataset_id", "source_name", "stage_label", "stage_label_source", "split", "manual_keep", "manual_stage_confirm", "manual_note"]].to_csv(
        output / "manifests" / "人工复核表.csv", index=False, encoding="utf-8-sig"
    )

    for stage_label in ("lt1_needle", "leaf1_to_2", "gt2", "unknown"):
        stage_rows = [row for row in records if row["stage_label"] == stage_label]
        contact_sheet(stage_rows, output / "contact_sheets" / f"stage_{stage_label}.jpg", f"{DATASET_VERSION}: {stage_label} | n={len(stage_rows)}")
    for split in ("train", "val", "test"):
        split_rows = [row for row in records if row["split"] == split]
        # Multiple pages prevent unreadably small tiles.
        for page_index in range(0, len(split_rows), 25):
            page = split_rows[page_index : page_index + 25]
            contact_sheet(page, output / "contact_sheets" / f"split_{split}_{page_index // 25 + 1:02d}.jpg", f"{DATASET_VERSION}: {split} {page_index + 1}-{page_index + len(page)}")

    summary = {
        "dataset_version": DATASET_VERSION,
        "created_by": "build_stage_clean_dataset.py",
        "images": len(records),
        "rejected": len(rejected),
        "keypoint_labels_used": False,
        "stage_counts": dict(Counter(str(row["stage_label"]) for row in records)),
        "split_counts": dict(Counter(str(row["split"]) for row in records)),
        "quality_tier_counts": dict(Counter(str(row["quality_tier"]) for row in records)),
        "background_operation_counts": dict(Counter(str(row["background_operation"]) for row in records)),
        "near_duplicate_pairs_dhash_le_6": len(near_pairs),
        "stage_label_caveat": "archive folder labels are user metadata; filename_leaf3 is a name proxy and remains subject to confirmation",
        "target_scope": "rice seedling shoot phenotypes; complete root preservation is intentionally not required",
        "background_caveat": "all outputs are derived shoot-standardized images; tier B mixed-background sources require visual review before inclusion",
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

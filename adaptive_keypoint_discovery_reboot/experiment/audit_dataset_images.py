#!/usr/bin/env python
"""Inventory D:/kp/数据集图片 and preselect genuinely clean-background images.

The script never reads keypoint labels or model outputs.  Existing aug0/1/2
files are inventoried but excluded from independent sample selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_ROOT = Path(r"D:\kp\数据集图片")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data_source_reaudit"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def source_family(path: Path) -> str:
    relative = path.relative_to(DATA_ROOT)
    first = relative.parts[0]
    if first == "kp_dataset":
        return "kp_dataset_augmented"
    if first == "images":
        return "images_400"
    return "scanner_collection"


def identity_group(path: Path, family: str) -> str:
    if family == "kp_dataset_augmented":
        stem = re.sub(r"_aug\d+$", "", path.stem, flags=re.IGNORECASE)
        return f"augmented_source::{stem.lower()}"
    if family == "scanner_collection":
        return "scanner_series::" + str(path.parent.relative_to(DATA_ROOT)).lower()
    return f"images400::{path.stem.lower()}"


def border_pixels(array: np.ndarray, thickness: int) -> np.ndarray:
    top = array[:thickness, :].reshape(-1, array.shape[-1])
    bottom = array[-thickness:, :].reshape(-1, array.shape[-1])
    left = array[thickness:-thickness, :thickness].reshape(-1, array.shape[-1])
    right = array[thickness:-thickness, -thickness:].reshape(-1, array.shape[-1])
    return np.concatenate([top, bottom, left, right], axis=0)


def dhash64(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def strict_green_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[:, :, index] for index in range(3))
    red, green, blue = (rgb[:, :, index].astype(np.float32) for index in range(3))
    normalized_excess = (2 * green - red - blue) / np.maximum(red + green + blue, 20.0)
    mask = (
        ((hue >= 18) & (hue <= 100) & (saturation >= 45) & (value >= 22))
        | ((normalized_excess >= 0.055) & (saturation >= 28) & (green >= 35))
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask > 0


def bbox_touch(mask: np.ndarray) -> tuple[int, float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 1, 1.0
    h, w = mask.shape
    margin = max(2, round(min(h, w) * 0.015))
    touches = int(xs.min() <= margin or ys.min() <= margin or xs.max() >= w - margin or ys.max() >= h - margin)
    fraction = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / float(h * w)
    return touches, float(fraction)


def component_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    threshold = max(5, round(mask.size * 0.00005))
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= threshold for index in range(1, count))


def load_rgb_thumbnail(path: Path, maximum: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        mapped = tifffile.memmap(path)
        array = np.asarray(mapped)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3:
            raise ValueError(f"Unsupported TIFF shape {array.shape}")
        original_height, original_width = array.shape[:2]
        stride = max(1, math.ceil(max(original_height, original_width) / maximum))
        rgb = np.asarray(array[::stride, ::stride, :3], dtype=np.uint8)
        return rgb, {
            "original_mode": "RGB_memmap",
            "original_width": original_width,
            "original_height": original_height,
            "alpha_present": 0,
            "transparent_fraction": 0.0,
        }
    with Image.open(path) as opened:
        original_mode = opened.mode
        original_width, original_height = opened.size
        alpha_present = int("A" in opened.getbands())
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        if alpha_present:
            rgba = np.asarray(image.convert("RGBA"))
            transparent_fraction = float((rgba[:, :, 3] < 250).mean())
            base = Image.new("RGBA", image.size, (255, 255, 255, 255))
            base.alpha_composite(image.convert("RGBA"))
            rgb = np.asarray(base.convert("RGB"))
        else:
            transparent_fraction = 0.0
            rgb = np.asarray(image.convert("RGB"))
    return rgb, {
        "original_mode": original_mode,
        "original_width": original_width,
        "original_height": original_height,
        "alpha_present": alpha_present,
        "transparent_fraction": transparent_fraction,
    }


def audit_pixels(path: Path) -> dict[str, Any]:
    rgb, loaded = load_rgb_thumbnail(path)
    original_mode = loaded["original_mode"]
    original_width = loaded["original_width"]
    original_height = loaded["original_height"]
    alpha_present = loaded["alpha_present"]
    transparent_fraction = loaded["transparent_fraction"]

    h, w = rgb.shape[:2]
    thickness = max(2, round(min(h, w) * 0.04))
    border = border_pixels(rgb, thickness).astype(np.float32)
    white = np.all(rgb >= 245, axis=2)
    border_white = np.all(border >= 245, axis=1)
    black = np.all(rgb <= 10, axis=2)
    nonwhite = np.max(255 - rgb.astype(np.int16), axis=2) >= 22
    green = strict_green_mask(rgb)
    green_fraction = float(green.mean())
    nonwhite_fraction = float(nonwhite.mean())
    plant_to_nonwhite = green_fraction / max(nonwhite_fraction, 1e-8)
    touches, bbox_fraction = bbox_touch(green)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    components = component_count(green)

    white_fraction = float(white.mean())
    border_white_fraction = float(border_white.mean())
    border_std = float(np.mean(np.std(border, axis=0)))
    black_fraction = float(black.mean())
    alpha_support = transparent_fraction

    score = 0.0
    score += 2.5 * border_white_fraction
    score += 2.0 * white_fraction
    score += 2.0 * float(np.clip(plant_to_nonwhite / 0.55, 0, 1))
    score += 1.0 * int(0.004 <= green_fraction <= 0.35)
    score += 0.8 * int(0.005 <= nonwhite_fraction <= 0.35)
    score += 0.7 * int(not touches)
    score += 1.5 * float(np.clip(alpha_support / 0.50, 0, 1))
    score -= 2.5 * int(nonwhite_fraction > 0.48)
    score -= 2.0 * int(plant_to_nonwhite < 0.12 and alpha_support < 0.20)
    score -= 2.0 * float(np.clip(black_fraction / 0.15, 0, 1))
    score -= 0.8 * int(components > 18)

    if alpha_support >= 0.20:
        background_class = "transparent_removed"
    elif border_white_fraction >= 0.94 and white_fraction >= 0.72 and nonwhite_fraction <= 0.35:
        background_class = "white_removed_candidate"
    elif border_white_fraction >= 0.80 and white_fraction >= 0.45:
        background_class = "white_with_large_residual"
    elif black_fraction >= 0.12:
        background_class = "black_or_rotation_border"
    else:
        background_class = "natural_or_mixed"

    return {
        "original_width": original_width,
        "original_height": original_height,
        "mode": original_mode,
        "alpha_present": alpha_present,
        "transparent_fraction": transparent_fraction,
        "white_fraction": white_fraction,
        "border_white_fraction": border_white_fraction,
        "border_rgb_std": border_std,
        "black_fraction": black_fraction,
        "nonwhite_fraction": nonwhite_fraction,
        "strict_green_fraction": green_fraction,
        "green_to_nonwhite_ratio": plant_to_nonwhite,
        "green_components": components,
        "green_bbox_fraction": bbox_fraction,
        "green_bbox_touches_border": touches,
        "blur_laplacian_thumbnail": blur,
        "background_class": background_class,
        "clean_background_score": round(score, 6),
        "dhash64": dhash64(rgb),
    }


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 4, 320, 255, 48
    page_rows = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * tile_w, header + page_rows * tile_h), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(22))
    body = font(13)
    for index, row in enumerate(rows):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        try:
            rgb, _ = load_rgb_thumbnail(Path(row["path"]), maximum=max(tile_w - 16, 188))
            image = Image.fromarray(rgb).convert("RGB")
            image.thumbnail((tile_w - 16, 188), Image.Resampling.LANCZOS)
            x = x0 + (tile_w - image.width) // 2
            y = y0 + 4 + (188 - image.height) // 2
            canvas.paste(image, (x, y))
        except Exception:
            draw.rectangle((x0 + 8, y0 + 8, x0 + tile_w - 8, y0 + 185), fill="#dddddd")
        caption = (
            f"{row['candidate_id']} | {row['source_family']}\n"
            f"score={float(row['clean_background_score']):.2f} white={float(row['white_fraction']):.2f} "
            f"green/nonwhite={float(row['green_to_nonwhite_ratio']):.2f}"
        )
        draw.multiline_text((x0 + 7, y0 + 195), caption, fill="black", font=body, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in DATA_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        family = source_family(path)
        base = {
            "candidate_id": f"src_{index:04d}",
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(DATA_ROOT)),
            "source_family": family,
            "identity_group": identity_group(path, family),
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "independent_selection_eligible": int(family != "kp_dataset_augmented"),
        }
        if family == "kp_dataset_augmented":
            # Metadata only: these are derived samples and must not become new
            # independent observations.
            try:
                with Image.open(path) as opened:
                    base.update(
                        {
                            "original_width": opened.width,
                            "original_height": opened.height,
                            "mode": opened.mode,
                            "background_class": "derived_augmentation_not_a_new_sample",
                        }
                    )
            except Exception as error:
                errors.append({"path": str(path), "error": repr(error)})
            rows.append(base)
            continue
        try:
            base.update(audit_pixels(path))
            rows.append(base)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
        if index % 100 == 0:
            print(f"AUDITED {index}/{len(files)}", flush=True)

    write_csv(OUTPUT / "all_images_inventory.csv", rows)
    if errors:
        write_csv(OUTPUT / "read_errors.csv", errors)

    eligible = [row for row in rows if row.get("independent_selection_eligible") == 1 and "clean_background_score" in row]
    eligible.sort(key=lambda row: (-float(row["clean_background_score"]), row["relative_path"]))
    # One image per scanner series in the first pass; the 400-image source has
    # no reliable plant identity and is therefore retained for manual duplicate review.
    preselection: list[dict[str, Any]] = []
    used_scanner_groups: set[str] = set()
    for row in eligible:
        if row["background_class"] not in {"transparent_removed", "white_removed_candidate"}:
            continue
        if float(row["strict_green_fraction"]) < 0.004 or float(row["nonwhite_fraction"]) > 0.38:
            continue
        if row["source_family"] == "scanner_collection":
            if row["identity_group"] in used_scanner_groups:
                continue
            used_scanner_groups.add(row["identity_group"])
        preselection.append(row)
        if len(preselection) >= 80:
            break
    write_csv(OUTPUT / "clean_background_preselection80.csv", preselection)
    for page in range(math.ceil(len(preselection) / 20)):
        contact_sheet(
            preselection[page * 20 : (page + 1) * 20],
            OUTPUT / f"clean_background_candidates_{page + 1:02d}.jpg",
            f"Clean-background candidates {page * 20 + 1}-{min((page + 1) * 20, len(preselection))} (no model outputs used)",
        )

    summary = {
        "total_images": len(files),
        "inventory_rows": len(rows),
        "read_errors": len(errors),
        "source_family_counts": dict(Counter(row["source_family"] for row in rows)),
        "background_class_counts_eligible": dict(Counter(row.get("background_class", "") for row in eligible)),
        "preselection_count": len(preselection),
        "important_rule": "kp_dataset_augmented is inventoried but excluded from independent sample selection",
    }
    (OUTPUT / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

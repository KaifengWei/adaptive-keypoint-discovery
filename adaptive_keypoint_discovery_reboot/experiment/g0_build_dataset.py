#!/usr/bin/env python3
"""G0 dataset audit and independent split builder for adaptive landmark discovery.

The script never reads legacy keypoint labels or legacy train/val/test roles.
It audits every source image, proposes conservative crops, reports duplicates,
and builds deterministic, stratified splits for the G1 training-free pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_TARGETS = {"development": 280, "validation": 60, "test": 60}
SPLIT_RATIOS = {name: count / 400.0 for name, count in SPLIT_TARGETS.items()}
SEED = 20260716


@dataclass
class AuditRecord:
    image_id: str
    source_path: str
    source_name: str
    sha256: str
    dhash64: str
    width: int
    height: int
    aspect_ratio: float
    orientation: str
    file_size_bytes: int
    blur_laplacian_var: float
    border_luminance_mean: float
    border_luminance_std: float
    border_saturation_mean: float
    background_class: str
    vegetation_fraction: float
    vegetation_components: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    bbox_area_fraction: float
    bbox_border_touch: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    quality_score: float
    auto_review_flags: str
    structural_complexity_score: float = 0.0
    structural_complexity_proxy: str = ""
    exact_duplicate_group: str = ""
    strong_near_duplicate_group: str = ""
    proposed_status: str = ""
    proposed_split: str = ""
    pilot80: int = 0
    crop_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\kp\kp_cc_uns_400\images"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\kp\adaptive_keypoint_discovery_reboot\experiment\data_g0"),
    )
    parser.add_argument("--target-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def enumerate_images(root: Path) -> list[Path]:
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(images, key=lambda p: (str(p.parent).lower(), p.name.lower()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image_rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def border_pixels(array: np.ndarray, thickness: int) -> np.ndarray:
    top = array[:thickness, :].reshape(-1, array.shape[-1])
    bottom = array[-thickness:, :].reshape(-1, array.shape[-1])
    left = array[thickness:-thickness, :thickness].reshape(-1, array.shape[-1])
    right = array[thickness:-thickness, -thickness:].reshape(-1, array.shape[-1])
    return np.concatenate([top, bottom, left, right], axis=0)


def vegetation_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Conservative vegetation mask used only for audit/crop proposals.

    This is deliberately not a phenotype label and is never used as a target.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    r = image_rgb[:, :, 0].astype(np.int16)
    g = image_rgb[:, :, 1].astype(np.int16)
    b = image_rgb[:, :, 2].astype(np.int16)
    hue, sat, val = (hsv[:, :, i] for i in range(3))
    green_hue = (hue >= 18) & (hue <= 100) & (sat >= 24) & (val >= 22)
    green_excess = (2 * g - r - b >= 10) & (g >= 35) & (g >= r - 3) & (g >= b - 3)
    mask = (green_hue | green_excess).astype(np.uint8) * 255
    min_dim = min(image_rgb.shape[:2])
    kernel_size = 3 if min_dim < 700 else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    min_area = max(20, int(mask.size * 0.00004))
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == idx] = 255
    return clean


def mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], float, int]:
    binary = mask > 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = max(0, count - 1)
    if not binary.any():
        return components, (0, 0, mask.shape[1], mask.shape[0]), 1.0, 1
    ys, xs = np.where(binary)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    h, w = mask.shape
    fraction = ((x2 - x1) * (y2 - y1)) / float(w * h)
    margin = max(2, round(min(h, w) * 0.01))
    touches = int(x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin)
    return components, (x1, y1, x2, y2), fraction, touches


def conservative_crop(
    bbox: tuple[int, int, int, int], width: int, height: int, valid_mask: bool
) -> tuple[int, int, int, int]:
    if not valid_mask:
        return 0, 0, width, height
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(0.30 * bw), int(0.05 * width))
    pad_y = max(int(0.30 * bh), int(0.05 * height))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    # Keep at least 60% of each source dimension so non-green organs are not clipped.
    min_w, min_h = int(0.60 * width), int(0.60 * height)
    if cx2 - cx1 < min_w:
        center = (cx1 + cx2) // 2
        cx1, cx2 = center - min_w // 2, center + math.ceil(min_w / 2)
    if cy2 - cy1 < min_h:
        center = (cy1 + cy2) // 2
        cy1, cy2 = center - min_h // 2, center + math.ceil(min_h / 2)
    if cx1 < 0:
        cx2 -= cx1
        cx1 = 0
    if cy1 < 0:
        cy2 -= cy1
        cy1 = 0
    if cx2 > width:
        cx1 -= cx2 - width
        cx2 = width
    if cy2 > height:
        cy1 -= cy2 - height
        cy2 = height
    return max(0, cx1), max(0, cy1), min(width, cx2), min(height, cy2)


def background_classification(lum_mean: float, lum_std: float, sat_mean: float) -> str:
    if lum_mean >= 210 and lum_std <= 35 and sat_mean <= 35:
        return "light_plain"
    if lum_mean <= 65:
        return "dark"
    if lum_std >= 45 or sat_mean >= 45:
        return "mixed"
    return "gray_plain"


def audit_one(path: Path, image_id: str) -> tuple[AuditRecord, Image.Image, np.ndarray]:
    with Image.open(path) as opened:
        pil = opened.convert("RGB")
    rgb = np.asarray(pil)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    thickness = max(2, round(min(width, height) * 0.04))
    border_rgb = border_pixels(rgb, thickness)
    border_hsv = border_pixels(hsv, thickness)
    border_gray = cv2.cvtColor(border_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
    lum_mean, lum_std = float(border_gray.mean()), float(border_gray.std())
    sat_mean = float(border_hsv[:, 1].mean())
    background = background_classification(lum_mean, lum_std, sat_mean)
    mask = vegetation_mask(rgb)
    vegetation_fraction = float((mask > 0).mean())
    components, bbox, bbox_fraction, border_touch = mask_geometry(mask)
    valid_mask = vegetation_fraction >= 0.0015
    crop = conservative_crop(bbox, width, height, valid_mask)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    flags: list[str] = []
    quality = 100.0
    if min(width, height) < 224:
        flags.append("low_resolution")
        quality -= 18
    if blur < 18:
        flags.append("possible_blur")
        quality -= 12
    if vegetation_fraction < 0.0015:
        flags.append("vegetation_mask_failed")
        quality -= 45
    elif vegetation_fraction < 0.004:
        flags.append("very_small_plant_signal")
        quality -= 12
    if vegetation_fraction > 0.50:
        flags.append("foreground_mask_too_large")
        quality -= 15
    if border_touch:
        flags.append("plant_bbox_touches_border")
        quality -= 6
    if background == "mixed":
        flags.append("mixed_background")
        quality -= 4
    orientation = "landscape" if width / height >= 1.15 else "portrait" if height / width >= 1.15 else "square"
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = crop
    record = AuditRecord(
        image_id=image_id,
        source_path=str(path.resolve()),
        source_name=path.name,
        sha256=sha256_file(path),
        dhash64=dhash64(rgb),
        width=width,
        height=height,
        aspect_ratio=round(width / height, 6),
        orientation=orientation,
        file_size_bytes=path.stat().st_size,
        blur_laplacian_var=round(blur, 4),
        border_luminance_mean=round(lum_mean, 4),
        border_luminance_std=round(lum_std, 4),
        border_saturation_mean=round(sat_mean, 4),
        background_class=background,
        vegetation_fraction=round(vegetation_fraction, 7),
        vegetation_components=components,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        bbox_area_fraction=round(bbox_fraction, 7),
        bbox_border_touch=border_touch,
        crop_x1=cx1,
        crop_y1=cy1,
        crop_x2=cx2,
        crop_y2=cy2,
        quality_score=round(max(0.0, quality), 2),
        auto_review_flags=";".join(flags),
    )
    return record, pil, mask


def assign_complexity(records: list[AuditRecord]) -> None:
    scores = []
    for record in records:
        # Audit-only proxy; it must not be interpreted as a growth-stage label.
        score = (
            math.log1p(record.vegetation_components)
            + 1.8 * math.sqrt(max(record.bbox_area_fraction, 0.0))
            + 0.25 * abs(math.log(max(record.aspect_ratio, 1e-6)))
        )
        record.structural_complexity_score = round(score, 6)
        scores.append(score)
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    for record in records:
        if record.structural_complexity_score <= q1:
            record.structural_complexity_proxy = "low"
        elif record.structural_complexity_score <= q2:
            record.structural_complexity_proxy = "medium"
        else:
            record.structural_complexity_proxy = "high"


def duplicate_analysis(records: list[AuditRecord]) -> list[dict[str, object]]:
    by_sha: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        by_sha[record.sha256].append(record)
    for index, group in enumerate((g for g in by_sha.values() if len(g) > 1), start=1):
        label = f"exact_{index:03d}"
        for record in group:
            record.exact_duplicate_group = label

    candidates: list[dict[str, object]] = []
    parent = list(range(len(records)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            distance = hamming_hex(records[left].dhash64, records[right].dhash64)
            if distance <= 5:
                same_size = records[left].width == records[right].width and records[left].height == records[right].height
                strong = distance <= 1 and same_size
                candidates.append(
                    {
                        "image_id_a": records[left].image_id,
                        "source_name_a": records[left].source_name,
                        "image_id_b": records[right].image_id,
                        "source_name_b": records[right].source_name,
                        "dhash_hamming": distance,
                        "same_dimensions": int(same_size),
                        "strong_grouped_for_split": int(strong),
                        "review_decision": "",
                    }
                )
                if strong:
                    union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        groups[find(idx)].append(idx)
    near_index = 0
    for indices in groups.values():
        if len(indices) > 1:
            near_index += 1
            label = f"near_{near_index:03d}"
            for idx in indices:
                records[idx].strong_near_duplicate_group = label
    return candidates


def row_stratum(record: AuditRecord) -> str:
    return "|".join((record.background_class, record.orientation, record.structural_complexity_proxy))


def split_records(records: list[AuditRecord], targets: dict[str, int], seed: int) -> None:
    rng = random.Random(seed)
    groups: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        key = record.exact_duplicate_group or record.strong_near_duplicate_group or record.image_id
        groups[key].append(record)
    items = list(groups.values())
    rng.shuffle(items)
    items.sort(key=lambda group: (-len(group), sum(Counter(row_stratum(r) for r in records)[row_stratum(group[0])] for _ in [0])))
    global_counts = Counter()
    stratum_counts: dict[str, Counter] = defaultdict(Counter)
    total_strata = Counter(row_stratum(record) for record in records)
    for group in items:
        size = len(group)
        group_strata = Counter(row_stratum(record) for record in group)
        choices = []
        for split_name, target in targets.items():
            remaining = target - global_counts[split_name]
            if remaining < size:
                continue
            global_need = remaining / max(target, 1)
            stratum_need = 0.0
            for stratum, amount in group_strata.items():
                desired = total_strata[stratum] * (target / len(records))
                stratum_need += amount * (desired - stratum_counts[stratum][split_name]) / max(desired, 1.0)
            choices.append((global_need + 1.7 * stratum_need, rng.random(), split_name))
        if not choices:
            raise RuntimeError("Could not fit duplicate group into exact split targets.")
        split_name = max(choices)[2]
        for record in group:
            record.proposed_split = split_name
            record.proposed_status = "included_pending_human_audit"
            global_counts[split_name] += 1
            stratum_counts[row_stratum(record)][split_name] += 1
    if dict(global_counts) != targets:
        raise RuntimeError(f"Split count mismatch: {dict(global_counts)} != {targets}")


def choose_pilot(records: list[AuditRecord], count: int, seed: int) -> list[AuditRecord]:
    dev = [record for record in records if record.proposed_split == "development"]
    by_stratum: dict[str, list[AuditRecord]] = defaultdict(list)
    rng = random.Random(seed + 80)
    for record in dev:
        by_stratum[row_stratum(record)].append(record)
    for values in by_stratum.values():
        rng.shuffle(values)
    strata = sorted(by_stratum, key=lambda key: (len(by_stratum[key]), key))
    selected: list[AuditRecord] = []
    # First give every represented stratum one image, then fill proportionally.
    for stratum in strata:
        if len(selected) >= count:
            break
        selected.append(by_stratum[stratum].pop())
    remaining_pool = [record for values in by_stratum.values() for record in values]
    rng.shuffle(remaining_pool)
    while len(selected) < count and remaining_pool:
        current = Counter(row_stratum(record) for record in selected)
        best_index = max(
            range(len(remaining_pool)),
            key=lambda idx: (
                Counter(row_stratum(r) for r in dev)[row_stratum(remaining_pool[idx])] / len(dev)
                - current[row_stratum(remaining_pool[idx])] / max(len(selected), 1),
                rng.random(),
            ),
        )
        selected.append(remaining_pool.pop(best_index))
    for record in selected:
        record.pilot80 = 1
    return selected


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_crop(pil: Image.Image, record: AuditRecord, crop_dir: Path) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop = pil.crop((record.crop_x1, record.crop_y1, record.crop_x2, record.crop_y2))
    output = crop_dir / f"{record.image_id}.jpg"
    crop.save(output, quality=95, subsampling=0)
    record.crop_path = str(output.resolve())


def draw_audit_tile(pil: Image.Image, record: AuditRecord, size: tuple[int, int] = (260, 200)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    picture_box = (0, 0, size[0], 158)
    thumb = pil.copy()
    thumb.thumbnail((picture_box[2], picture_box[3]), Image.Resampling.LANCZOS)
    px = (picture_box[2] - thumb.width) // 2
    py = (picture_box[3] - thumb.height) // 2
    canvas.paste(thumb, (px, py))
    scale_x, scale_y = thumb.width / record.width, thumb.height / record.height
    draw = ImageDraw.Draw(canvas)
    bbox = (
        px + round(record.bbox_x1 * scale_x),
        py + round(record.bbox_y1 * scale_y),
        px + round(record.bbox_x2 * scale_x),
        py + round(record.bbox_y2 * scale_y),
    )
    crop = (
        px + round(record.crop_x1 * scale_x),
        py + round(record.crop_y1 * scale_y),
        px + round(record.crop_x2 * scale_x),
        py + round(record.crop_y2 * scale_y),
    )
    draw.rectangle(crop, outline=(35, 100, 220), width=2)
    draw.rectangle(bbox, outline=(30, 180, 75), width=2)
    status_color = (15, 115, 45) if record.proposed_status.startswith("included") else (190, 70, 20)
    lines = [
        f"{record.image_id}  {record.source_name}",
        f"{record.proposed_split or 'holdout'} | Q={record.quality_score:.0f} | {record.background_class}",
        f"flags: {record.auto_review_flags or 'none'}",
    ]
    draw.rectangle((0, 158, size[0], size[1]), fill=(248, 249, 251))
    draw.text((5, 161), lines[0], fill=(20, 20, 20))
    draw.text((5, 174), lines[1], fill=status_color)
    draw.text((5, 187), lines[2][:44], fill=(80, 80, 80))
    return canvas


def contact_sheets(records: list[AuditRecord], images: dict[str, Image.Image], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols, rows_per_page = 7, 5
    tile_size = (260, 200)
    per_page = cols * rows_per_page
    ordered = sorted(records, key=lambda r: r.image_id)
    for page_start in range(0, len(ordered), per_page):
        page_records = ordered[page_start : page_start + per_page]
        sheet = Image.new("RGB", (cols * tile_size[0], rows_per_page * tile_size[1]), (228, 231, 236))
        for offset, record in enumerate(page_records):
            tile = draw_audit_tile(images[record.image_id], record, tile_size)
            x = (offset % cols) * tile_size[0]
            y = (offset // cols) * tile_size[1]
            sheet.paste(tile, (x, y))
        page_number = page_start // per_page + 1
        sheet.save(output_dir / f"audit_contact_sheet_{page_number:02d}.jpg", quality=92)


def split_summary(records: list[AuditRecord]) -> dict[str, object]:
    included = [record for record in records if record.proposed_split]
    return {
        "total_audited": len(records),
        "included": len(included),
        "review_holdout": len(records) - len(included),
        "split_counts": dict(Counter(record.proposed_split for record in included)),
        "pilot80_count": sum(record.pilot80 for record in included),
        "background_counts": dict(Counter(record.background_class for record in records)),
        "orientation_counts": dict(Counter(record.orientation for record in records)),
        "complexity_proxy_counts": dict(Counter(record.structural_complexity_proxy for record in records)),
        "review_flag_counts": dict(
            Counter(flag for record in records for flag in record.auto_review_flags.split(";") if flag)
        ),
        "exact_duplicate_images": sum(bool(record.exact_duplicate_group) for record in records),
        "strong_near_duplicate_images": sum(bool(record.strong_near_duplicate_group) for record in records),
        "note": "All labels are audit metadata only; no keypoint definition or legacy split is used.",
    }


def write_human_guide(output_root: Path, summary: dict[str, object]) -> None:
    guide = f"""# G0 数据审计：你实际需要做什么

本目录已扫描源图 {summary['total_audited']} 张，并独立提出 {summary['included']} 张实验图的拆分；原图没有被移动或修改，也没有读取任何旧关键点标签。

## 你现在只需要完成两项人工工作

1. 依次打开 `contact_sheets` 中的审计图。绿框是自动估计的植株范围，蓝框是提供给 DINOv2 的保守裁剪范围。只记录明显空图、严重截断、错误裁剪或并非秧苗的图。
2. 打开 `near_duplicate_candidates.csv`，只复核 `dhash_hamming` 较小的图像对是否真的是同一张图。不要因为姿态相似就判成重复。

无需标注关键点、叶尖、叶基、叶片编号或关键点数量。`structural_complexity_proxy` 只是用于抽样均衡的图像统计量，不是生育期标签。

## 文件用途

- `audit_manifest.csv`：401 张图的完整审计账本，可填写 `human_decision` 与 `human_note`。
- `split_manifest.csv`：G1 使用的开发/验证/测试拆分和 80 张先导集标记。
- `crops`：只为模型输入生成的副本；源图保持不变。
- `contact_sheets`：人工快速浏览用。
- `near_duplicate_candidates.csv`：近重复候选，不等同于自动删除清单。
- `g0_summary.json`：统计摘要。
- `split_hashes.json`：锁定本次拆分，避免以后悄悄换图。

完成复核后，只需在 `audit_manifest.csv` 的 `human_decision` 填 `keep`、`exclude` 或 `recrop`，并在 `human_note` 写一句原因。未填写的行保持待复核状态。
"""
    (output_root / "G0_人工复核说明.md").write_text(guide, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = enumerate_images(source)
    if not paths:
        raise SystemExit(f"No supported images found under {source}")
    records: list[AuditRecord] = []
    images: dict[str, Image.Image] = {}
    masks_dir = output / "audit_masks"
    crops_dir = output / "crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        image_id = f"akd_{index:04d}"
        record, pil, mask = audit_one(path, image_id)
        records.append(record)
        images[image_id] = pil
        Image.fromarray(mask).save(masks_dir / f"{image_id}.png")
        if index % 50 == 0 or index == len(paths):
            print(f"AUDIT {index}/{len(paths)}", flush=True)

    assign_complexity(records)
    duplicate_candidates = duplicate_analysis(records)
    ranked = sorted(
        records,
        key=lambda r: (
            r.quality_score,
            -int(bool(r.auto_review_flags)),
            r.vegetation_fraction,
            r.image_id,
        ),
        reverse=True,
    )
    target_count = min(args.target_count, len(ranked))
    selected = ranked[:target_count]
    holdout = ranked[target_count:]
    if target_count == 400:
        split_records(selected, SPLIT_TARGETS, args.seed)
    else:
        dev = round(target_count * 0.70)
        val = round(target_count * 0.15)
        test = target_count - dev - val
        split_records(selected, {"development": dev, "validation": val, "test": test}, args.seed)
    for record in holdout:
        record.proposed_status = "review_holdout_lowest_audit_score"
    pilot = choose_pilot(selected, min(80, len([r for r in selected if r.proposed_split == "development"])), args.seed)

    for record in records:
        save_crop(images[record.image_id], record, crops_dir)

    base_fields = list(AuditRecord.__dataclass_fields__.keys())
    audit_fields = base_fields + ["human_decision", "human_note"]
    audit_rows = []
    for record in records:
        row = vars(record).copy()
        row.update({"human_decision": "", "human_note": ""})
        audit_rows.append(row)
    write_csv(output / "audit_manifest.csv", audit_rows, audit_fields)
    split_rows = [vars(record).copy() for record in records if record.proposed_split]
    write_csv(output / "split_manifest.csv", split_rows, base_fields)
    write_csv(
        output / "pilot80_manifest.csv",
        [vars(record).copy() for record in pilot],
        base_fields,
    )
    duplicate_fields = [
        "image_id_a",
        "source_name_a",
        "image_id_b",
        "source_name_b",
        "dhash_hamming",
        "same_dimensions",
        "strong_grouped_for_split",
        "review_decision",
    ]
    write_csv(output / "near_duplicate_candidates.csv", duplicate_candidates, duplicate_fields)
    contact_sheets(records, images, output / "contact_sheets")
    summary = split_summary(records)
    with (output / "g0_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    hashes = {
        split: [record.sha256 for record in sorted(records, key=lambda r: r.image_id) if record.proposed_split == split]
        for split in ("development", "validation", "test")
    }
    hashes["pilot80"] = [record.sha256 for record in sorted(pilot, key=lambda r: r.image_id)]
    with (output / "split_hashes.json").open("w", encoding="utf-8") as handle:
        json.dump(hashes, handle, ensure_ascii=False, indent=2)
    write_human_guide(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

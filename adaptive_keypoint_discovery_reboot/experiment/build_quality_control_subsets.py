#!/usr/bin/env python
"""Preselect controlled/complex image-quality groups without reading model outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent


def nuisance_score(data: pd.DataFrame) -> pd.Series:
    flags = data["auto_review_flags"].fillna("")
    score = pd.Series(0.0, index=data.index)
    score += 2.5 * data["background_class"].eq("mixed")
    score += 2.0 * flags.str.contains("foreground_mask_too_large")
    score += 2.0 * flags.str.contains("possible_blur")
    score += 0.5 * flags.str.contains("plant_bbox_touches_border")
    score += 1.0 * ((data["vegetation_fraction"] < 0.012) | (data["vegetation_fraction"] > 0.40))
    score += np.abs(np.log(np.clip(data["aspect_ratio"], 1e-6, None))) / 3.0
    for column, weight in (("border_luminance_std", 0.7), ("border_saturation_mean", 0.4)):
        median = float(data[column].median())
        mad = float(np.median(np.abs(data[column] - median)))
        score += weight * np.clip(np.abs(data[column] - median) / max(1.4826 * mad, 1e-6), 0, 3)
    blur_low, blur_high = data["blur_laplacian_var"].quantile([0.08, 0.92])
    score += 0.8 * ((data["blur_laplacian_var"] < blur_low) | (data["blur_laplacian_var"] > blur_high))
    return score


def balanced_select(data: pd.DataFrame, count: int, ascending: bool, blocked: set[str]) -> pd.DataFrame:
    pool = data[~data["image_id"].isin(blocked)].sort_values(
        ["nuisance_score", "image_id"], ascending=[ascending, True]
    )
    targets = {"low": 7, "medium": 7, "high": 6}
    selected = []
    for complexity, target in targets.items():
        selected.append(pool[pool["structural_complexity_proxy"] == complexity].head(target))
    result = pd.concat(selected, ignore_index=False)
    if len(result) < count:
        remaining = pool[~pool["image_id"].isin(result["image_id"])]
        result = pd.concat([result, remaining.head(count - len(result))], ignore_index=False)
    return result.head(count).sort_values(["structural_complexity_proxy", "nuisance_score", "image_id"])


def font(size: int) -> ImageFont.ImageFont:
    candidates = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(data: pd.DataFrame, path: Path, title: str) -> None:
    columns, rows = 4, math.ceil(len(data) / 4)
    tile_w, tile_h, header = 310, 255, 48
    canvas = Image.new("RGB", (columns * tile_w, header + rows * tile_h), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(22))
    body_font = font(14)
    for index, (_, item) in enumerate(data.iterrows()):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        with Image.open(item["crop_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 16, 190))
        x = x0 + (tile_w - image.width) // 2
        y = y0 + 4 + (190 - image.height) // 2
        canvas.paste(image, (x, y))
        flags = str(item["auto_review_flags"]) if pd.notna(item["auto_review_flags"]) else ""
        codes = []
        for phrase, code in (
            ("mixed_background", "MIX"),
            ("foreground_mask_too_large", "MASK"),
            ("possible_blur", "BLUR"),
            ("plant_bbox_touches_border", "EDGE"),
        ):
            if phrase in flags:
                codes.append(code)
        caption = (
            f"{item['image_id']} | {item['background_class']} | {item['structural_complexity_proxy']}\n"
            f"score={item['nuisance_score']:.2f} | flags={','.join(codes) if codes else 'none'}"
        )
        draw.multiline_text((x0 + 8, y0 + 198), caption, fill="black", font=body_font, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    partition = pd.read_csv(HERE / "outputs_g1" / "g1_partition_manifest.csv")
    pilot = partition[partition["g1_role"] == "pilot60"].copy()
    if len(pilot) != 60:
        raise RuntimeError(f"Expected 60 locked pilot images, found {len(pilot)}")
    pilot["nuisance_score"] = nuisance_score(pilot)
    controlled = balanced_select(pilot, 20, ascending=True, blocked=set()).copy()
    complex_group = balanced_select(pilot, 20, ascending=False, blocked=set(controlled["image_id"])).copy()
    controlled["quality_group_preselection"] = "controlled20_pending_human_review"
    complex_group["quality_group_preselection"] = "complex20_pending_human_review"
    used = set(controlled["image_id"]) | set(complex_group["image_id"])
    reserve = pilot[~pilot["image_id"].isin(used)].sort_values(["nuisance_score", "image_id"]).copy()
    reserve["quality_group_preselection"] = "reserve20_for_manual_swaps"
    selected = pd.concat([controlled, complex_group, reserve], ignore_index=True)
    selected["human_keep"] = ""
    selected["human_reason"] = ""
    output = HERE / "data_quality_control"
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "quality_group_preselection", "image_id", "source_name", "crop_path",
        "background_class", "structural_complexity_proxy", "nuisance_score",
        "quality_score", "blur_laplacian_var", "vegetation_fraction",
        "bbox_area_fraction", "aspect_ratio", "auto_review_flags",
        "human_keep", "human_reason",
    ]
    selected[columns].to_csv(output / "quality_control_preselection.csv", index=False, encoding="utf-8-sig")
    contact_sheet(controlled, output / "controlled20_contact_sheet.jpg", "Controlled-quality preselection (20) - review original images only")
    contact_sheet(complex_group, output / "complex20_contact_sheet.jpg", "Complex-quality preselection (20) - review original images only")
    contact_sheet(reserve, output / "reserve20_contact_sheet.jpg", "Reserve images (20) - use for manual swaps")
    print(selected.groupby(["quality_group_preselection", "structural_complexity_proxy"]).size().to_string())


if __name__ == "__main__":
    main()

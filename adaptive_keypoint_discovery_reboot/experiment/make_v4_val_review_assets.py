#!/usr/bin/env python
"""Build readable V4 validation contact sheets and a human path-review table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=Path, default=HERE / "evaluation_outputs" / "core_dinov2_v4_fullplant_val"
    )
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_page(frame: pd.DataFrame, evaluation: Path, output: Path, title: str) -> None:
    columns, rows = 2, 5
    tile_w, tile_h, header = 920, 500, 70
    canvas = Image.new("RGB", (columns * tile_w, header + rows * tile_h), "#eef1f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), title, fill="black", font=font(28))
    caption_font = font(16)
    for local_index, (_, row) in enumerate(frame.iterrows()):
        x0 = (local_index % columns) * tile_w
        y0 = header + (local_index // columns) * tile_h
        overlay_path = evaluation / "overlays" / f"{row['dataset_id']}.png"
        with Image.open(overlay_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_w - 20, 420), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 5))
        caption = (
            f"{row['dataset_id']} | points={int(row['point_count'])} | paths={int(row['path_count'])} | "
            f"repeat={row['mean_repeatability_f1']:.3f}\n"
            f"auto-support hit={row['foreground_hit_ratio']:.3f} | "
            f"saved full-plant hit={row['full_plant_hit_ratio']:.3f} | "
            f"background misses={int(row['background_miss_count'])}"
        )
        draw.multiline_text((x0 + 12, y0 + 430), caption, fill="black", font=caption_font, spacing=2)
    canvas.save(output, quality=94)


def run(args: argparse.Namespace) -> None:
    evaluation = args.evaluation.resolve()
    metrics = pd.read_csv(evaluation / "per_image.csv")
    regions = pd.read_csv(evaluation / "region_hit_per_image.csv")
    frame = metrics.merge(
        regions[["dataset_id", "full_plant_hit_ratio", "background_miss_count"]], on="dataset_id", how="left"
    ).sort_values("dataset_id")
    page_size = 10
    for start in range(0, len(frame), page_size):
        page = frame.iloc[start : start + page_size]
        page_number = start // page_size + 1
        render_page(
            page,
            evaluation,
            evaluation / f"V4_val路径视觉复核_{page_number:02d}.jpg",
            f"V4 full-plant validation: learned points and paths ({page_number}/{math.ceil(len(frame) / page_size)})",
        )

    worst = frame.sort_values(
        ["full_plant_hit_ratio", "background_miss_count", "mean_repeatability_f1"],
        ascending=[True, False, True],
    ).head(10)
    render_page(
        worst,
        evaluation,
        evaluation / "V4_val低命中优先复核.jpg",
        "V4 full-plant validation: lowest mask-hit cases (review first)",
    )

    review = frame[
        [
            "dataset_id",
            "point_count",
            "path_count",
            "mean_repeatability_f1",
            "foreground_hit_ratio",
            "full_plant_hit_ratio",
            "background_miss_count",
        ]
    ].copy()
    review["人工_点是否覆盖主要结构_是或否"] = ""
    review["人工_地上部路径是否正确_是或否"] = ""
    review["人工_根或基部路径是否正确_是或否"] = ""
    review["人工_是否存在漏叶错枝或离体点_说明"] = ""
    review["人工_是否可进入表型误差计算_是或否"] = ""
    review["人工_备注"] = ""
    review.to_csv(evaluation / "V4_val路径视觉复核表_待填写.csv", index=False, encoding="utf-8-sig")
    print(f"wrote 4 sequential pages, 1 worst-case page, and {len(review)} review rows")


if __name__ == "__main__":
    run(parse_args())

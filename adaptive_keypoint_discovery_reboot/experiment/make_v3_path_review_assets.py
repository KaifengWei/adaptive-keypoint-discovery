#!/usr/bin/env python
"""Create a readable contact sheet and human review table for path validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    result = args.result.resolve()
    metrics = pd.read_csv(result / "per_image.csv")
    manifest = pd.read_csv(args.dataset / "manifests" / "all.csv")
    manifest_subset = manifest[["dataset_id", "source_name", "relative_path"]].rename(
        columns={"source_name": "source_name_manifest"}
    )
    frame = metrics.merge(manifest_subset, on="dataset_id", how="left")
    if "source_name" not in frame:
        frame["source_name"] = frame["source_name_manifest"]
    columns, tile_w, tile_h, header = 3, 620, 470, 58
    rows = max(1, math.ceil(len(frame) / columns))
    canvas = Image.new("RGB", (columns * tile_w, header + rows * tile_h), "#eef1f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 13), "G1′ V3 primary val/test: path review (machine output, human decision pending)", fill="black", font=font(22))
    body = font(13)
    for index, row in frame.iterrows():
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        overlay_path = result / "overlays" / f"{row['dataset_id']}.png"
        with Image.open(overlay_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_w - 14, 400), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 3))
        caption = (
            f"{row['dataset_id']} | {row['stage_label']} | {row['split']} | {row['source_name']}\n"
            f"candidates={row['candidate_count']} | paths={row['path_count']} | base margin={row['base_score_margin']:.3f}"
        )
        draw.multiline_text((x0 + 8, y0 + 409), caption, fill="black", font=body, spacing=2)
    canvas.save(result / "路径视觉复核总览.jpg", quality=94)

    review = frame[
        ["dataset_id", "source_name", "stage_label", "split", "relative_path", "candidate_count", "path_count", "base_score_margin"]
    ].copy()
    review["人工_基部是否正确_是或否"] = ""
    review["人工_路径数量是否正确_是或否"] = ""
    review["人工_是否漏叶或串线_说明"] = ""
    review["人工_是否可进入表型误差计算_是或否"] = ""
    review["人工_备注"] = ""
    review.to_csv(result / "路径视觉复核表_待填写.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

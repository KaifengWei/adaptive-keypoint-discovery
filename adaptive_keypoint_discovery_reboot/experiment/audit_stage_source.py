#!/usr/bin/env python
"""Audit the stage-labelled source archive without reading model outputs.

The parent folder supplies only a stage metadata label.  This script does not
create keypoint labels and does not infer a biological stage from appearance.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_dataset_images import audit_pixels


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stage_source_extracted" / "叶龄-秧苗30张"
OUTPUT = HERE / "data_stage_source_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stage_from_parent(path: Path) -> tuple[str, str]:
    parent = path.parent.name
    if parent == "小于1，立针":
        return "lt1_needle", "archive_folder"
    if parent == "1到2":
        return "leaf1_to_2", "archive_folder"
    if parent == "大于2":
        return "gt2", "archive_folder"
    return "unknown", "unknown"


def source_group(path: Path) -> str:
    stem = path.stem
    # Multiple crops from one scanner frame must stay in the same split.
    early_match = re.match(r"^(.+?)\(\d+\)$", stem)
    if early_match:
        return f"scanner_crop::{early_match.group(1)}"
    return f"single::{stem}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def contact_sheet(rows: list[dict[str, object]], path: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 4, 330, 300, 50
    nrows = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, header + nrows * tile_h), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(22))
    body = font(12)
    for index, row in enumerate(rows):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        with Image.open(str(row["path"])) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode == "RGBA":
                white = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(white, image)
            image = image.convert("RGB")
            image.thumbnail((tile_w - 16, 220), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 4 + (220 - image.height) // 2))
        caption = (
            f"{row['audit_id']} | {row['source_name']}\n"
            f"bg={row['background_class']} score={float(row['clean_background_score']):.2f}\n"
            f"white={float(row['white_fraction']):.2f} border={float(row['border_white_fraction']):.2f} "
            f"blur={float(row['blur_laplacian_thumbnail']):.1f}"
        )
        draw.multiline_text((x0 + 7, y0 + 229), caption, fill="black", font=body, spacing=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def main() -> None:
    paths = sorted(path for path in SOURCE.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        stage, label_source = stage_from_parent(path)
        metrics = audit_pixels(path)
        row: dict[str, object] = {
            "audit_id": f"stage_src_{index:03d}",
            "path": str(path),
            "relative_path": str(path.relative_to(SOURCE)),
            "source_name": path.name,
            "stage_label": stage,
            "stage_label_source": label_source,
            "source_group": source_group(path),
            "sha256": sha256(path),
            **metrics,
        }
        rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "stage_source_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for stage in ("lt1_needle", "leaf1_to_2", "gt2"):
        selected = [row for row in rows if row["stage_label"] == stage]
        contact_sheet(selected, OUTPUT / f"contact_{stage}.jpg", f"Stage source audit: {stage} (metadata label only)")
    print(f"audited={len(rows)}")
    for stage in ("lt1_needle", "leaf1_to_2", "gt2"):
        selected = [row for row in rows if row["stage_label"] == stage]
        print(stage, len(selected), {str(row["background_class"]) for row in selected})


if __name__ == "__main__":
    main()

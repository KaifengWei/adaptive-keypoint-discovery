#!/usr/bin/env python
"""Build a clean-background core set without reading any model outputs."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(data: pd.DataFrame, path: Path, title: str) -> None:
    columns, tile_w, tile_h, header = 4, 320, 255, 48
    rows = (len(data) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, header + rows * tile_h), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill="black", font=font(22))
    body = font(13)
    for index, (_, row) in enumerate(data.iterrows()):
        x0, y0 = (index % columns) * tile_w, header + (index // columns) * tile_h
        with Image.open(row["clean_image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_w - 16, 188), Image.Resampling.LANCZOS)
        x = x0 + (tile_w - image.width) // 2
        y = y0 + 4 + (188 - image.height) // 2
        canvas.paste(image, (x, y))
        caption = (
            f"{row['clean_id']} | {row['role']} | {row['orientation']}\n"
            f"{row['source_name']} | score={row['clean_background_score']:.2f}"
        )
        draw.multiline_text((x0 + 7, y0 + 195), caption, fill="black", font=body, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=93)


def main() -> None:
    inventory = pd.read_csv(HERE / "data_source_reaudit" / "all_images_inventory.csv", low_memory=False)
    audit = pd.read_csv(HERE / "data_g0" / "audit_manifest.csv")
    candidates = inventory[
        (inventory["source_family"] == "images_400")
        & (inventory["background_class"] == "white_removed_candidate")
    ].copy()
    candidates["source_name"] = candidates["relative_path"].map(lambda value: Path(value).name)
    data = candidates.merge(
        audit[
            [
                "source_name", "image_id", "sha256", "proposed_split", "pilot80",
                "orientation", "structural_complexity_proxy", "quality_score",
            ]
        ],
        on="source_name",
        how="inner",
    )
    # Strictly avoid every image whose output was already inspected in G1.
    pool = data[(data["proposed_split"] == "development") & (data["pilot80"] == 0)].copy()
    core = pool.sort_values(["clean_background_score", "sha256"], ascending=[False, True]).head(20).copy()
    if len(core) != 20:
        raise RuntimeError(f"Expected 20 clean development images, found {len(core)}")
    if core["sha256"].duplicated().any():
        raise RuntimeError("Exact duplicate detected in clean core")

    near_pairs = []
    for (_, left), (_, right) in itertools.combinations(core.iterrows(), 2):
        distance = (int(str(left["dhash64"]), 16) ^ int(str(right["dhash64"]), 16)).bit_count()
        if distance <= 8:
            near_pairs.append((left["candidate_id"], right["candidate_id"], distance))
    if near_pairs:
        raise RuntimeError(f"Near-duplicate candidates found: {near_pairs}")

    hash_sorted = core.sort_values("sha256")
    portrait_smoke = hash_sorted[hash_sorted["orientation"] == "portrait"].head(3)
    landscape_smoke = hash_sorted[hash_sorted["orientation"] == "landscape"].head(3)
    smoke_ids = set(pd.concat([portrait_smoke, landscape_smoke])["candidate_id"])
    core["role"] = core["candidate_id"].map(lambda value: "smoke6" if value in smoke_ids else "locked14")
    core["clean_id"] = [f"clean_{index:03d}" for index in range(1, len(core) + 1)]
    core["clean_image_path"] = core["path"]
    core["human_keep"] = ""
    core["human_reason"] = ""

    output = HERE / "data_clean_core20"
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "clean_id", "role", "candidate_id", "image_id", "source_name", "clean_image_path",
        "sha256", "dhash64", "orientation", "clean_background_score", "white_fraction",
        "border_white_fraction", "nonwhite_fraction", "strict_green_fraction",
        "green_to_nonwhite_ratio", "green_components", "green_bbox_fraction",
        "green_bbox_touches_border", "proposed_split", "pilot80",
        "human_keep", "human_reason",
    ]
    core[columns].to_csv(output / "clean_core20_manifest.csv", index=False, encoding="utf-8-sig")
    core[core["role"] == "smoke6"][columns].to_csv(output / "smoke6_manifest.csv", index=False, encoding="utf-8-sig")
    core[core["role"] == "locked14"][columns].to_csv(output / "locked14_manifest.csv", index=False, encoding="utf-8-sig")
    contact_sheet(core[core["role"] == "smoke6"], output / "smoke6_contact_sheet.jpg", "Clean core: smoke6 (may be used for method checks)")
    contact_sheet(core[core["role"] == "locked14"], output / "locked14_contact_sheet.jpg", "Clean core: locked14 (do not run before gate)")
    with (output / "near_duplicate_check.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id_a", "candidate_id_b", "dhash_hamming"])
        writer.writerows(near_pairs)
    print(core.groupby(["role", "orientation"]).size().to_string())


if __name__ == "__main__":
    main()

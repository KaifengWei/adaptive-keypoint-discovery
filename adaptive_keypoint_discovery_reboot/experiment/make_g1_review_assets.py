#!/usr/bin/env python3
"""Build compact human-review assets from completed G1 pilot outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"D:\kp\adaptive_keypoint_discovery_reboot\experiment")
PILOT = ROOT / "outputs_g1" / "pilot"
VISUALS = PILOT / "visualizations"
OUTPUT = PILOT / "human_review"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_sheet(items: list[tuple[str, Path]], path: Path, cols: int = 2, rows: int = 3) -> None:
    tile_w, tile_h, label_h = 620, 310, 24
    page = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (225, 228, 233))
    draw = ImageDraw.Draw(page)
    for index, (image_id, source) in enumerate(items[: cols * rows]):
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + label_h)
        px, py = x + (tile_w - image.width) // 2, y + (tile_h - image.height) // 2
        page.paste(image, (px, py))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(248, 249, 251))
        draw.text((x + 6, y + tile_h + 5), image_id, fill=(25, 25, 25))
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(PILOT / "per_image_metrics.csv")
    attention = {
        row["image_id"]: row
        for row in metrics
        if row["method"] == "cls_to_patch_attention"
    }
    local = {
        row["image_id"]: row
        for row in metrics
        if row["method"] == "feature_local_contrast"
    }
    visual_paths = sorted(VISUALS.glob("*_518_last4avg.jpg"))
    visual_by_id = {path.name.split("_518_")[0]: path for path in visual_paths}
    checklist = []
    for image_id in sorted(visual_by_id):
        attn = attention[image_id]
        loc = local[image_id]
        checklist.append(
            {
                "image_id": image_id,
                "visual_path": str(visual_by_id[image_id].resolve()),
                "attention_candidate_count": attn["candidate_count"],
                "attention_repeatability_f1": attn["mean_repeatability_f1"],
                "attention_plant_hit_ratio": attn["plant_hit_ratio"],
                "local_candidate_count": loc["candidate_count"],
                "local_repeatability_f1": loc["mean_repeatability_f1"],
                "reviewer_a_structure_related_0_or_1": "",
                "reviewer_a_coverage_low_medium_high": "",
                "reviewer_a_dominant_region": "",
                "reviewer_a_note": "",
                "reviewer_b_structure_related_0_or_1": "",
                "reviewer_b_coverage_low_medium_high": "",
                "reviewer_b_dominant_region": "",
                "reviewer_b_note": "",
            }
        )
    with (OUTPUT / "双人独立复核表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checklist[0]))
        writer.writeheader()
        writer.writerows(checklist)

    ordered = [(image_id, visual_by_id[image_id]) for image_id in sorted(visual_by_id)]
    for page_index in range(0, len(ordered), 6):
        save_sheet(ordered[page_index : page_index + 6], OUTPUT / f"review_sheet_{page_index // 6 + 1:02d}.jpg")

    worst_attention = sorted(
        visual_by_id,
        key=lambda image_id: float(attention[image_id]["mean_repeatability_f1"]),
    )[:6]
    worst_local = sorted(
        visual_by_id,
        key=lambda image_id: (
            float(local[image_id]["plant_hit_ratio"]),
            float(local[image_id]["mean_repeatability_f1"]),
        ),
    )[:6]
    best_attention = sorted(
        visual_by_id,
        key=lambda image_id: float(attention[image_id]["mean_repeatability_f1"]),
        reverse=True,
    )[:6]
    save_sheet([(image_id, visual_by_id[image_id]) for image_id in worst_attention], OUTPUT / "attention_worst6.jpg")
    save_sheet([(image_id, visual_by_id[image_id]) for image_id in worst_local], OUTPUT / "local_contrast_worst6.jpg")
    save_sheet([(image_id, visual_by_id[image_id]) for image_id in best_attention], OUTPUT / "attention_best6.jpg")

    metadata = {
        "review_images": len(checklist),
        "scope": "Qualitative failure analysis only; automatic G1 gates already failed.",
        "instruction": "Two reviewers work independently. A point set is structure-related only if it follows plant morphology rather than background/cutout texture. Coverage judges whether different visible organs and junctions are represented, not just the base region.",
        "allowed_dominant_region_terms": ["base", "main_axis", "leaf_body", "leaf_tip", "junction", "root", "background", "mixed"],
    }
    (OUTPUT / "复核说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

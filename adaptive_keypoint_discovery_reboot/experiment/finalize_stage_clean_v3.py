#!/usr/bin/env python
"""Finalize the visually audited V3 shoot dataset.

The script copies only accepted candidate images, removes a confirmed duplicate,
uses user-provided stage metadata for the primary train/val/test split, and puts
filename-proxy or unknown-stage images in an auxiliary self-supervised pool.
No keypoint label or model output is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from audit_dataset_images import audit_pixels, dhash64
from build_stage_clean_dataset import contact_sheet, font, write_rgb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "data_stage_clean_v3_candidate"
OUTPUT = HERE / "data_stage_clean_v3"
CONFIRMED_EXCLUSIONS = {
    "stagev3_0084": "duplicate_of_stagev3_0015",
    "stagev3_0045": "detached_true_leaf_after_standardization",
    "stagev3_0090": "detached_true_leaf_after_standardization",
}


def read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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


def keep_largest_shoot_component(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Remove disconnected root/seed/debris without inventing a connection."""
    foreground = np.any(rgb < 248, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return rgb, {"components_before": 0, "second_to_first_area_ratio": 0.0}
    areas = [(int(stats[label_id, cv2.CC_STAT_AREA]), label_id) for label_id in range(1, count)]
    areas.sort(reverse=True)
    keep = labels == areas[0][1]
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cleaned = rgb.copy()
    cleaned[~keep] = 255
    ratio = areas[1][0] / areas[0][0] if len(areas) > 1 and areas[0][0] else 0.0
    return cleaned, {"components_before": len(areas), "second_to_first_area_ratio": ratio}


def pair_sheet(frame: pd.DataFrame, pairs: list[dict[str, Any]], path: Path) -> None:
    tile_w, tile_h, header = 390, 250, 54
    canvas = Image.new("RGB", (tile_w * 2, header + tile_h * max(1, len(pairs))), "#eef0f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), "V3 foreground-crop dHash near-pair audit", fill="black", font=font(20))
    body = font(12)
    lookup = frame.set_index("dataset_id")
    for row_index, pair in enumerate(pairs):
        for column, key in enumerate(("dataset_id_a", "dataset_id_b")):
            dataset_id = str(pair[key])
            row = lookup.loc[dataset_id]
            with Image.open(str(row["output_path"])) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((tile_w - 20, 190), Image.Resampling.LANCZOS)
            x0, y0 = column * tile_w, header + row_index * tile_h
            canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 3 + (190 - image.height) // 2))
            caption = f"{dataset_id} | {row['source_name']}\n{row['stage_label']} | {row['split']}"
            draw.multiline_text((x0 + 8, y0 + 197), caption, fill="black", font=body, spacing=2)
        draw.text(
            (8, header + row_index * tile_h + 232),
            f"d={pair['dhash_hamming']} | review={pair['review_decision']}",
            fill="#1f2937",
            font=body,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def main() -> None:
    if OUTPUT.exists() and any(path.is_file() for path in OUTPUT.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "contact_sheets").mkdir(parents=True, exist_ok=True)

    candidate = pd.read_csv(CANDIDATE / "manifests" / "all.csv")
    excluded_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for record in candidate.to_dict("records"):
        dataset_id = str(record["dataset_id"])
        if dataset_id in CONFIRMED_EXCLUSIONS:
            excluded_rows.append(
                {
                    "dataset_id": dataset_id,
                    "source_name": record["source_name"],
                    "reason": CONFIRMED_EXCLUSIONS[dataset_id],
                    "review_actor": "codex_visual_pair_audit_2026-07-17",
                }
            )
            continue

        # Only explicit user metadata is permitted in the primary independent
        # split.  Proxy and unknown stages remain useful without pretending to
        # be stage-verified evaluation data.
        if str(record["stage_label_confidence"]) == "user_metadata":
            final_split = str(record["split"])
            data_role = "primary_stage_verified"
            primary_evaluation_eligible = 1
        else:
            final_split = "auxiliary"
            data_role = "auxiliary_unverified_stage"
            primary_evaluation_eligible = 0

        source_output = Path(str(record["output_path"]))
        destination = OUTPUT / "images" / final_split / source_output.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_rgb = read_rgb(source_output)
        rgb, component_diagnostics = keep_largest_shoot_component(source_rgb)
        write_rgb(destination, rgb)
        crop_hash = dhash64(foreground_crop(rgb))
        after = audit_pixels(destination)
        record.update(
            {
                "split": final_split,
                "data_role": data_role,
                "training_eligible": 1,
                "primary_evaluation_eligible": primary_evaluation_eligible,
                "visual_review_status": "pass_contact_sheet_shoot_integrity_and_background",
                "visual_review_actor": "codex_contact_sheet_review_2026-07-17",
                "user_stage_confirmation_pending": int(str(record["stage_label_confidence"]) != "user_metadata"),
                "component_cleanup": "largest_connected_shoot_component",
                "components_before_cleanup": component_diagnostics["components_before"],
                "second_to_first_area_ratio_before_cleanup": component_diagnostics["second_to_first_area_ratio"],
                "output_path": str(destination),
                "relative_path": str(destination.relative_to(OUTPUT)),
                "output_dhash64_full_canvas": record.get("output_dhash64", ""),
                "output_dhash64_foreground_crop": crop_hash,
                "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "output_background_class": after["background_class"],
                "output_white_fraction": after["white_fraction"],
                "output_border_white_fraction": after["border_white_fraction"],
            }
        )
        accepted_rows.append(record)

    frame = pd.DataFrame(accepted_rows)
    hashes = frame["output_dhash64_foreground_crop"].astype(str).tolist()
    near_pairs: list[dict[str, Any]] = []
    for left_index in range(len(frame)):
        for right_index in range(left_index + 1, len(frame)):
            distance = (int(hashes[left_index], 16) ^ int(hashes[right_index], 16)).bit_count()
            if distance <= 6:
                near_pairs.append(
                    {
                        "dataset_id_a": frame.iloc[left_index]["dataset_id"],
                        "dataset_id_b": frame.iloc[right_index]["dataset_id"],
                        "dhash_hamming": distance,
                        "cross_split": int(frame.iloc[left_index]["split"] != frame.iloc[right_index]["split"]),
                        "review_decision": "different_visible_structure_not_duplicate",
                    }
                )

    frame.to_csv(OUTPUT / "manifests" / "all.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test", "auxiliary"):
        frame[frame["split"] == split].to_csv(OUTPUT / "manifests" / f"{split}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded_rows).to_csv(OUTPUT / "manifests" / "excluded.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(near_pairs).to_csv(OUTPUT / "manifests" / "near_duplicate_review.csv", index=False, encoding="utf-8-sig")
    frame[
        [
            "dataset_id",
            "source_name",
            "stage_label",
            "stage_label_source",
            "stage_label_confidence",
            "split",
            "visual_review_status",
            "user_stage_confirmation_pending",
        ]
    ].to_csv(OUTPUT / "manifests" / "时期与质量复核表.csv", index=False, encoding="utf-8-sig")

    for stage_label in ("lt1_needle", "leaf1_to_2", "gt2", "unknown"):
        stage_rows = [row for row in accepted_rows if row["stage_label"] == stage_label]
        for start in range(0, len(stage_rows), 25):
            page = stage_rows[start : start + 25]
            contact_sheet(
                page,
                OUTPUT / "contact_sheets" / f"stage_{stage_label}_{start // 25 + 1:02d}.jpg",
                f"stage-clean-v3-final: {stage_label} | {start + 1}-{start + len(page)}",
            )
    pair_sheet(frame, near_pairs, OUTPUT / "contact_sheets" / "near_duplicate_review.jpg")

    summary = {
        "dataset_version": "stage-clean-v3-final",
        "target_scope": "rice seedling shoot phenotypes",
        "images": len(frame),
        "excluded_images": len(excluded_rows),
        "excluded_reason_counts": dict(Counter(str(row["reason"]) for row in excluded_rows)),
        "keypoint_labels_used": False,
        "split_counts": dict(Counter(frame["split"].astype(str))),
        "stage_counts": dict(Counter(frame["stage_label"].astype(str))),
        "primary_stage_verified_images": int(frame["primary_evaluation_eligible"].sum()),
        "auxiliary_unverified_stage_images": int((frame["split"] == "auxiliary").sum()),
        "foreground_crop_near_pairs_dhash_le_6": len(near_pairs),
        "near_pair_review": "contact-sheet reviewed as different visible structures; confirmed duplicate was removed before this audit",
        "training_rule": "train plus auxiliary may be used for unsupervised/self-supervised learning; val/test remain user-stage-metadata only",
        "stage_caveat": "filename-proxy and unknown stages are auxiliary only until user confirmation",
    }
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

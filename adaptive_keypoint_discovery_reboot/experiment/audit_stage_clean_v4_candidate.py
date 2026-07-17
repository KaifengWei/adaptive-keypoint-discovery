#!/usr/bin/env python
"""Audit V4 candidate integrity without reading any model output."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from audit_dataset_images import dhash64
from build_stage_clean_v4_candidate import foreground_crop, hamming, phash64


HERE = Path(__file__).resolve().parent
DATASET = HERE / "data_stage_clean_v4_candidate"
MANUAL_NEAR_PAIR_DECISIONS = {
    frozenset({"v4_val_0014", "v4_train_new_0036"}): "different_visible_structure_not_duplicate_codex_visual_review_20260717"
}


def image_hashes(path: Path) -> tuple[str, str]:
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    crop = foreground_crop(rgb)
    return dhash64(crop), phash64(crop)


def main() -> None:
    manifest = pd.read_csv(DATASET / "manifests" / "all.csv", low_memory=False)
    new = manifest[manifest["new_split_role"].isin({"train_new", "val", "test"})].copy()
    errors: list[str] = []

    expected = {"train": 220, "val": 40, "test": 40}
    counts = Counter(manifest["split"].astype(str))
    if dict(counts) != expected:
        errors.append(f"unexpected split counts: {dict(counts)}")
    if len(manifest) != 300:
        errors.append(f"unexpected image count: {len(manifest)}")
    if manifest["dataset_id"].duplicated().any():
        errors.append("duplicate dataset_id")
    if manifest["output_sha256"].duplicated().any():
        errors.append("exact duplicate output_sha256")
    if int(manifest["keypoint_labels_used"].astype(int).sum()) != 0:
        errors.append("keypoint labels were used")
    if int(manifest["model_outputs_used_for_selection"].astype(int).sum()) != 0:
        errors.append("model outputs were used for selection")

    group_splits = new.groupby("split_group")["split"].nunique()
    frame_splits = new.groupby("source_frame_id")["split"].nunique()
    if (group_splits > 1).any():
        errors.append("source directory group crosses splits")
    if (frame_splits > 1).any():
        errors.append("source frame crosses splits")

    quality_checks = {
        "green_preservation_ratio_min": float(new["quality_green_preservation_ratio"].astype(float).min()),
        "secondary_to_first_green_area_ratio_max": float(
            new["quality_secondary_to_first_green_area_ratio"].astype(float).max()
        ),
        "green_bbox_touches_border_sum": int(new["quality_green_bbox_touches_border"].astype(int).sum()),
        "model_outputs_used_sum": int(new["model_outputs_used_for_selection"].astype(int).sum()),
    }
    if quality_checks["green_preservation_ratio_min"] < 0.40:
        errors.append("green preservation gate violated")
    if quality_checks["secondary_to_first_green_area_ratio_max"] > 0.06:
        errors.append("disconnected component gate violated")
    if quality_checks["green_bbox_touches_border_sum"] != 0:
        errors.append("accepted shoot touches standardized crop border")

    hash_rows: list[dict[str, str]] = []
    for row in manifest.to_dict("records"):
        path = Path(str(row["output_path"]))
        if not path.exists():
            errors.append(f"missing output: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(row["output_sha256"]):
            errors.append(f"sha mismatch: {row['dataset_id']}")
        dhash_value, phash_value = image_hashes(path)
        hash_rows.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "split": str(row["split"]),
                "new_split_role": str(row["new_split_role"]),
                "relative_path": str(row["relative_path"]),
                "sha256": digest,
                "dhash64_foreground_crop": dhash_value,
                "phash64_foreground_crop": phash_value,
            }
        )

    exact = defaultdict(list)
    for row in hash_rows:
        exact[row["sha256"]].append(row)
    exact_cross_split = [values for values in exact.values() if len(values) > 1]
    if exact_cross_split:
        errors.append("exact image duplicate found during live hash audit")

    potential_pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(hash_rows):
        for right in hash_rows[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            d_distance = hamming(left["dhash64_foreground_crop"], right["dhash64_foreground_crop"])
            p_distance = hamming(left["phash64_foreground_crop"], right["phash64_foreground_crop"])
            if d_distance <= 3 and p_distance <= 8:
                pair_key = frozenset({str(left["dataset_id"]), str(right["dataset_id"])})
                potential_pairs.append(
                    {
                        "dataset_id_a": left["dataset_id"],
                        "split_a": left["split"],
                        "dataset_id_b": right["dataset_id"],
                        "split_b": right["split"],
                        "dhash_hamming": d_distance,
                        "phash_hamming": p_distance,
                        "review_status": MANUAL_NEAR_PAIR_DECISIONS.get(pair_key, "pending_visual_pair_review"),
                    }
                )

    hashes = pd.DataFrame(hash_rows)
    hashes.to_csv(DATASET / "manifests" / "hash_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(potential_pairs).to_csv(
        DATASET / "manifests" / "potential_cross_split_near_duplicates.csv", index=False, encoding="utf-8-sig"
    )
    with (DATASET / "manifests" / "candidate_lock.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(hash_rows, key=lambda item: item["relative_path"]):
            handle.write(f"{row['sha256']}  {row['relative_path']}\n")

    summary = {
        "dataset_version": "stage-clean-v4-candidate",
        "audit_status": "pass" if not errors else "fail",
        "errors": errors,
        "image_count": len(manifest),
        "split_counts": dict(counts),
        "new_role_counts": dict(Counter(manifest["new_split_role"].astype(str))),
        "new_source_directory_groups": int(new["split_group"].nunique()),
        "new_source_frames": int(new["source_frame_id"].nunique()),
        "quality_checks": quality_checks,
        "exact_duplicate_outputs": len(exact_cross_split),
        "potential_cross_split_near_duplicate_pairs": len(potential_pairs),
        "pending_cross_split_near_duplicate_reviews": sum(
            row["review_status"] == "pending_visual_pair_review" for row in potential_pairs
        ),
        "keypoint_labels_used": False,
        "model_outputs_used_for_selection": False,
        "lock_status": "candidate hash manifest only; final test remains pending human raw-image review",
    }
    (DATASET / "candidate_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

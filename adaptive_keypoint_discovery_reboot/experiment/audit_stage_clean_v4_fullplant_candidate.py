#!/usr/bin/env python
"""Audit the complete-seedling V4 candidate without using model outputs."""

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
SOURCE = HERE / "data_stage_clean_v4_candidate"
DATASET = HERE / "data_stage_clean_v4_fullplant_candidate"
MANUAL_NEAR_PAIR_DECISIONS = {
    frozenset({"v4_val_0014", "v4_train_new_0036"}): (
        "different_visible_structure_not_duplicate_codex_visual_review_20260717"
    )
}


def read_mask(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot decode mask: {path}")
    return mask > 0


def image_hashes(path: Path) -> tuple[str, str]:
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    crop = foreground_crop(rgb)
    return dhash64(crop), phash64(crop)


def main() -> None:
    source = pd.read_csv(SOURCE / "manifests" / "all.csv", low_memory=False)
    manifest = pd.read_csv(DATASET / "manifests" / "all.csv", low_memory=False)
    errors: list[str] = []
    expected_counts = {"train": 220, "val": 40, "test": 40}
    counts = Counter(manifest["split"].astype(str))

    if len(manifest) != 300:
        errors.append(f"unexpected image count: {len(manifest)}")
    if dict(counts) != expected_counts:
        errors.append(f"unexpected split counts: {dict(counts)}")
    if manifest["dataset_id"].duplicated().any():
        errors.append("duplicate dataset_id")

    identity_columns = ["dataset_id", "split", "new_split_role", "source_path"]
    left = source[identity_columns].sort_values("dataset_id").reset_index(drop=True).astype(str)
    right = manifest[identity_columns].sort_values("dataset_id").reset_index(drop=True).astype(str)
    if not left.equals(right):
        errors.append("sample identities, source paths, roles, or splits changed")

    if int(manifest["keypoint_labels_used"].astype(int).sum()) != 0:
        errors.append("keypoint labels were used")
    if int(manifest["model_outputs_used_for_selection"].astype(int).sum()) != 0:
        errors.append("model outputs were used")
    versions = sorted(set(manifest["normalization_version"].astype(str)))
    if versions != ["v4_fullplant_classical_pixel_v2"]:
        errors.append(f"unexpected normalization versions: {versions}")

    new = manifest[manifest["new_split_role"].isin({"train_new", "val", "test"})].copy()
    if (new.groupby("split_group")["split"].nunique() > 1).any():
        errors.append("source directory group crosses splits")
    if (new.groupby("source_frame_id")["split"].nunique() > 1).any():
        errors.append("source frame crosses splits")

    mask_rows: list[dict[str, object]] = []
    hash_rows: list[dict[str, str]] = []
    for row in manifest.to_dict("records"):
        dataset_id = str(row["dataset_id"])
        image_path = Path(str(row["output_path"]))
        shoot_path = Path(str(row["shoot_mask_path"]))
        root_path = Path(str(row["seed_base_root_mask_path"]))
        full_path = Path(str(row["full_plant_mask_path"]))
        paths = [image_path, shoot_path, root_path, full_path]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            errors.append(f"{dataset_id}: missing files {missing}")
            continue

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != str(row["output_sha256"]):
            errors.append(f"{dataset_id}: sha mismatch")

        shoot, root, full = read_mask(shoot_path), read_mask(root_path), read_mask(full_path)
        with Image.open(image_path) as opened:
            width, height = opened.size
        if shoot.shape != (height, width) or root.shape != shoot.shape or full.shape != shoot.shape:
            errors.append(f"{dataset_id}: image/mask shape mismatch")
            continue
        shoot_outside = int((shoot & ~full).sum())
        root_outside = int((root & ~full).sum())
        if shoot_outside or root_outside:
            errors.append(f"{dataset_id}: full mask does not contain organ masks")
        if not shoot.any():
            errors.append(f"{dataset_id}: empty shoot mask")
        border_pixels = int(full[0].sum() + full[-1].sum() + full[:, 0].sum() + full[:, -1].sum())
        if border_pixels:
            errors.append(f"{dataset_id}: full-plant mask touches crop border")

        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(full.astype(np.uint8), 8)
        component_areas = sorted(
            [int(value) for value in component_stats[1:, cv2.CC_STAT_AREA]], reverse=True
        )
        secondary_area_ratio = float(sum(component_areas[1:]) / max(1, component_areas[0])) if component_areas else 0.0
        mask_rows.append(
            {
                "dataset_id": dataset_id,
                "split": str(row["split"]),
                "new_split_role": str(row["new_split_role"]),
                "shoot_pixels": int(shoot.sum()),
                "seed_base_root_pixels": int(root.sum()),
                "full_plant_pixels": int(full.sum()),
                "nonshoot_to_shoot_area_ratio": float(root.sum() / max(1, shoot.sum())),
                "full_component_count": component_count - 1,
                "secondary_to_main_full_area_ratio": secondary_area_ratio,
                "full_border_pixels": border_pixels,
                "shoot_outside_full_pixels": shoot_outside,
                "root_outside_full_pixels": root_outside,
            }
        )

        dhash_value, phash_value = image_hashes(image_path)
        hash_rows.append(
            {
                "dataset_id": dataset_id,
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
    exact_duplicates = [values for values in exact.values() if len(values) > 1]
    if exact_duplicates:
        errors.append("exact image duplicate found during live hash audit")

    potential_pairs: list[dict[str, object]] = []
    for left_index, left_hash in enumerate(hash_rows):
        for right_hash in hash_rows[left_index + 1 :]:
            if left_hash["split"] == right_hash["split"]:
                continue
            d_distance = hamming(left_hash["dhash64_foreground_crop"], right_hash["dhash64_foreground_crop"])
            p_distance = hamming(left_hash["phash64_foreground_crop"], right_hash["phash64_foreground_crop"])
            if d_distance <= 3 and p_distance <= 8:
                pair_key = frozenset({str(left_hash["dataset_id"]), str(right_hash["dataset_id"])})
                potential_pairs.append(
                    {
                        "dataset_id_a": left_hash["dataset_id"],
                        "split_a": left_hash["split"],
                        "dataset_id_b": right_hash["dataset_id"],
                        "split_b": right_hash["split"],
                        "dhash_hamming": d_distance,
                        "phash_hamming": p_distance,
                        "review_status": MANUAL_NEAR_PAIR_DECISIONS.get(pair_key, "pending_visual_pair_review"),
                    }
                )

    mask_frame = pd.DataFrame(mask_rows)
    hash_frame = pd.DataFrame(hash_rows)
    mask_frame.to_csv(DATASET / "manifests" / "mask_integrity.csv", index=False, encoding="utf-8-sig")
    hash_frame.to_csv(DATASET / "manifests" / "hash_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(potential_pairs).to_csv(
        DATASET / "manifests" / "potential_cross_split_near_duplicates.csv", index=False, encoding="utf-8-sig"
    )
    with (DATASET / "manifests" / "candidate_lock.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(hash_rows, key=lambda item: item["relative_path"]):
            handle.write(f"{row['sha256']}  {row['relative_path']}\n")

    root_ratios = mask_frame["nonshoot_to_shoot_area_ratio"]
    secondary_ratios = mask_frame["secondary_to_main_full_area_ratio"]
    component_counts = Counter(mask_frame["full_component_count"].astype(int))
    pending_pairs = sum(row["review_status"] == "pending_visual_pair_review" for row in potential_pairs)
    if pending_pairs:
        errors.append(f"{pending_pairs} cross-split near-duplicate pairs need visual review")

    summary = {
        "dataset_version": "stage-clean-v4-fullplant-candidate",
        "audit_status": "pass" if not errors else "fail",
        "errors": errors,
        "image_count": len(manifest),
        "split_counts": dict(counts),
        "sample_identities_and_splits_changed": False,
        "organ_masks": ["shoot", "seed_base_root", "full_plant"],
        "mask_integrity": {
            "shoot_outside_full_pixels_sum": int(mask_frame["shoot_outside_full_pixels"].sum()),
            "root_outside_full_pixels_sum": int(mask_frame["root_outside_full_pixels"].sum()),
            "full_border_pixels_sum": int(mask_frame["full_border_pixels"].sum()),
            "full_component_count_histogram": {str(key): value for key, value in sorted(component_counts.items())},
            "secondary_to_main_full_area_ratio_max": float(secondary_ratios.max()),
            "secondary_to_main_full_area_ratio_median": float(secondary_ratios.median()),
            "zero_visible_nonshoot_samples": int((root_ratios == 0).sum()),
            "nonshoot_to_shoot_area_ratio_min": float(root_ratios.min()),
            "nonshoot_to_shoot_area_ratio_median": float(root_ratios.median()),
            "nonshoot_to_shoot_area_ratio_max": float(root_ratios.max()),
        },
        "exact_duplicate_outputs": len(exact_duplicates),
        "potential_cross_split_near_duplicate_pairs": len(potential_pairs),
        "pending_cross_split_near_duplicate_reviews": pending_pairs,
        "keypoint_labels_used": False,
        "model_outputs_used_for_selection": False,
        "lock_status": "candidate only; test remains pending human raw/shoot/full contact-sheet review",
    }
    (DATASET / "candidate_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

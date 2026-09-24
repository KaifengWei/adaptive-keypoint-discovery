"""Adapt frozen G1-prime val-only points to the existing graph/decoder input.

This is a lossless schema adapter. It does not read human GT, alter teacher
points, or select thresholds. Its output is private evaluation input, not
training labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


POINT_COLUMNS = ("dataset_id", "point_id", "x_source", "y_source", "x_normalized", "y_normalized", "confidence")
IMAGE_COLUMNS = ("dataset_id", "split", "input_domain", "point_count", "point_source")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(pseudo_jsonl: Path, val_manifest: Path, output: Path) -> dict:
    with val_manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != 40 or {row["split"] for row in manifest} != {"val"}:
        raise ValueError("Expected the locked 40-image V4 val manifest")
    expected = {row["dataset_id"]: row for row in manifest}
    records = [json.loads(line) for line in pseudo_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 40 or {row["dataset_id"] for row in records} != set(expected):
        raise ValueError("Teacher rows do not exactly match locked V4 val")
    if len({row["dataset_id"] for row in records}) != 40:
        raise ValueError("Duplicate teacher dataset ID")
    images: list[dict] = []
    points: list[dict] = []
    for record in sorted(records, key=lambda row: row["dataset_id"]):
        dataset_id = record["dataset_id"]
        if record["split"] != "val" or record["input_domain"] != "phenotype_roi_v1":
            raise ValueError(f"Wrong split or domain for {dataset_id}")
        if record["teacher_variant"] != "full" or record["structure_coverage_enabled"] or not record["consistency_filter_used"]:
            raise ValueError(f"Unexpected teacher variant for {dataset_id}")
        accepted = record["points"]
        if len(accepted) != record["accepted_count"]:
            raise ValueError(f"Point count mismatch for {dataset_id}")
        if len({point["point_id"] for point in accepted}) != len(accepted):
            raise ValueError(f"Duplicate point ID for {dataset_id}")
        images.append({
            "dataset_id": dataset_id,
            "split": "val",
            "input_domain": "phenotype_roi_v1",
            "point_count": len(accepted),
            "point_source": "frozen_G1prime_RouteB_teacher_direct",
        })
        for point in accepted:
            normalized = (float(point["x_normalized"]), float(point["y_normalized"]))
            if not all(0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(f"Out-of-image point for {dataset_id}")
            points.append({
                "dataset_id": dataset_id,
                "point_id": point["point_id"],
                "x_source": point["x_source"],
                "y_source": point["y_source"],
                "x_normalized": point["x_normalized"],
                "y_normalized": point["y_normalized"],
                "confidence": point["consensus_confidence"],
            })
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "per_image.csv", IMAGE_COLUMNS, images)
    _write_csv(output / "points.csv", POINT_COLUMNS, points)
    audit = {
        "purpose": "Teacher-direct val-only frozen point schema adapter",
        "val_images": 40,
        "total_points": len(points),
        "zero_point_images": [row["dataset_id"] for row in images if row["point_count"] == 0],
        "teacher_source_sha256": _sha(pseudo_jsonl),
        "val_manifest_sha256": _sha(val_manifest),
        "per_image_sha256": _sha(output / "per_image.csv"),
        "points_sha256": _sha(output / "points.csv"),
        "human_gt_read": False,
        "test_read": False,
        "model_or_threshold_modified": False,
    }
    (output / "adapter_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-jsonl", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.pseudo_jsonl, args.val_manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Freeze the submitted single-rater V4 test reference without model inference.

Only frozen human records, their private ID mapping, locked dataset metadata,
and the pre-existing shoot masks needed for bbox normalization are read.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
PROTOCOL = EXPERIMENT / "phenotype_pilot_protocol"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROTOCOL))

from package_tools import sha, require, validate_exports  # noqa: E402
from phenotype_gt_geometry import (  # noqa: E402
    GEOMETRY_PROTOCOL_VERSION,
    divergence_angle,
    select_reference_main,
    trace_metrics,
)

RUNTIME = PROTOCOL / "runtime" / "v4_test_single_rater_20260927"
STAGE = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
OUTPUT = RUNTIME / "final_gt" / "single_rater_20260928_v1"
EXPECTED = {
    "raw_annotations.json": "b3a201f6ca38073b08dabab74d672f3136339177b577a8571cc7238f992c2073",
    "sessions.csv": "0656f3cd5a20a6f2c7e7147e19449b4b167681ba9c963e2323cdd5ba9a6d63bb",
    "traces.csv": "9b1d0432e6c92bfc814f69a18149675dce5bfe529a9092ad9918219c6fe34b1d",
    "SHA256_LEDGER.json": "27e98ff9567f7ec538f12b87c7f29c29ada5be9b3fae40c9f9cec8d3c49d3c91",
    "mapping.json": "042e7f49f88041823f9608a79b2b051d39e2413ec1316a12d5d872fd36fa197f",
    "measurement_manifest.json": "404b89b87b6ec3ea211069762de98718ad7ea47b3f55c54b6b06b45b74c87c45",
    "test.csv": "161f342037f4c982fb5f532bc60b18aadaad6656584ee667c7e0ca828818c5f6",
    "mask_integrity.csv": "a9e9024a530855eb4388918fdd35e07c05bcaa4a413db470fe80890d0df7f95f",
    "phenotype_gt_geometry.py": "c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800",
}
CSV_FIELDS = (
    "dataset_id", "blind_id", "source_frame_id", "image_sha256", "record_sha256",
    "gt_id", "original_trace_uuid", "trace_uuid", "visibility_status", "confidence",
    "occlusion", "interpolation_used", "reference_main_path", "bbox_diagonal_px",
    "length_px", "length_bbox_normalized", "length_mm_derived_from_scanner_metadata",
    "divergence_angle_status", "divergence_angle_deg", "tip_xy_json", "points_px_json",
    "note",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _points_sha256(points: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps(points, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _unique(rows: list[dict], key: str) -> dict[str, dict]:
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key}")
    return result


def _mask_scale(dataset_id: str, row: dict, integrity: dict, width: int, height: int) -> tuple[float, float, str]:
    expected = f"masks/shoot/test/{dataset_id}.png"
    require(row["shoot_mask_relative_path"].replace("\\", "/") == expected, "Unexpected shoot-mask path")
    path = STAGE / expected
    with Image.open(path) as opened:
        require(opened.size == (width, height), "Shoot-mask dimensions mismatch")
        mask = np.asarray(opened.convert("L"))
    require(set(np.unique(mask).tolist()) <= {0, 255}, "Shoot mask is not binary")
    count = int(np.count_nonzero(mask))
    require(count == int(integrity[dataset_id]["shoot_pixels"]) and count > 0, "Shoot-mask count mismatch")
    ys, xs = np.nonzero(mask)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    bbox_diag = math.hypot(bbox_width, bbox_height)
    box = json.loads(row["source_crop_box_fullplant"])
    require(len(box) == 4, "Invalid source crop box")
    crop_w, crop_h = int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
    require(crop_w > 0 and crop_h > 0, "Invalid source crop size")
    scale = min(1.0, 1600.0 / max(crop_w, crop_h))
    pad = int(row["normalization_padding_pixels"])
    require((round(crop_w * scale) + 2 * pad, round(crop_h * scale) + 2 * pad) == (width, height), "Locked resize metadata mismatch")
    return bbox_diag, (25.4 / 600.0) / scale, sha(path)


def _plant(record: dict, meta: dict, mapping: dict, integrity: dict) -> tuple[dict, str]:
    dataset_id = mapping["dataset_id"]
    row = meta[dataset_id]
    require(row["split"] == "test" and record["image_sha256"] == mapping["expected_sha256"] == row["output_sha256"], "Image identity mismatch")
    require(mapping["source_frame_id"] == row["source_frame_id"], "Source-frame identity mismatch")
    width, height = int(record["width"]), int(record["height"])
    bbox_diag, mm_per_px, mask_sha = _mask_scale(dataset_id, row, integrity, width, height)
    structures, paths = [], []
    for item in record["items"]:
        entry = {
            "gt_id": item["gt_id"],
            "original_trace_uuid": item["trace_uuid"],
            "visibility_status": item["visibility_status"],
            "confidence": item["confidence"],
            "occlusion": item["occlusion"],
            "interpolation_used": item["interpolation_used"],
            "tip_xy": item["tip_xy"],
            "note": item["note"],
            "trace_uuid": None,
            "points_px": None,
            "reference_main_path": False,
            "divergence_angle_status": "geometry_unavailable",
            "divergence_angle_deg": None,
            "length_px": None,
            "length_bbox_normalized": None,
            "length_mm_derived_from_scanner_metadata": None,
        }
        if item["visibility_status"] == "measurable":
            metrics = trace_metrics(item["points_px"], bbox_diag)
            curve = metrics["resampled_curve"]
            require(np.isfinite(curve).all(), "Non-finite derived curve")
            require((curve[:, 0] >= 0).all() and (curve[:, 0] < width).all()
                    and (curve[:, 1] >= 0).all() and (curve[:, 1] < height).all(), "Derived curve outside standardized image")
            points = curve.tolist()
            entry.update({
                "source_control_points_sha256": _points_sha256(item["points_px"]),
                "trace_uuid": _points_sha256(points),
                "points_px": points,
                "length_px": metrics["structural_path_length_px"],
                "length_bbox_normalized": metrics["structural_path_length_bbox_norm"],
                "length_mm_derived_from_scanner_metadata": metrics["structural_path_length_px"] * mm_per_px,
                "chord_length_px": metrics["chord_length_px"],
                "tip_clockwise_angle_deg": metrics["tip_clockwise_angle_deg"],
                "structural_path_length_px": metrics["structural_path_length_px"],
                "resampling_method": metrics["resampling_method"],
            })
            paths.append(entry)
        structures.append(entry)
    require(paths, "Each submitted plant must retain its measurable paths")
    require(len({p["trace_uuid"] for p in paths}) == len(paths), "Duplicate derived paths")
    main = select_reference_main(paths, bbox_diag)
    main_curve = next(p["points_px"] for p in paths if p["trace_uuid"] == main)
    for path in paths:
        path["reference_main_path"] = path["trace_uuid"] == main
        if path["reference_main_path"]:
            path["divergence_angle_status"] = "main_path"
        else:
            result = divergence_angle(np.asarray(main_curve), np.asarray(path["points_px"]), bbox_diag)
            path["divergence_angle_status"] = result["status"]
            path["divergence_angle_deg"] = result.get("divergence_angle_deg")
    return ({
        "dataset_id": dataset_id,
        "blind_id": record["blind_id"],
        "split": "test",
        "source_frame_id": mapping["source_frame_id"],
        "image_sha256": record["image_sha256"],
        "session_id": record["session_id"],
        "record_sha256": mapping["record_sha256"],
        "submitted_at": record["submitted_at"],
        "base_xy": record["base_xy"],
        "image_note": record["image_note"],
        "bbox_diagonal_px": bbox_diag,
        "mm_per_output_px_scanner_metadata_derived": mm_per_px,
        "operational_independent_path_count": sum(s["visibility_status"] in {"measurable", "visible_unmeasurable"} for s in structures),
        "measurable_reference_path_count": len(paths),
        "reference_main_trace_uuid": main,
        "structures": structures,
        "paths": paths,
    }, mask_sha)


def _csv_bytes(plants: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for plant in plants:
        for item in plant["structures"]:
            writer.writerow({
                "dataset_id": plant["dataset_id"],
                "blind_id": plant["blind_id"],
                "source_frame_id": plant["source_frame_id"],
                "image_sha256": plant["image_sha256"],
                "record_sha256": plant["record_sha256"],
                "gt_id": item["gt_id"],
                "original_trace_uuid": item["original_trace_uuid"],
                "trace_uuid": item["trace_uuid"] or "",
                "visibility_status": item["visibility_status"],
                "confidence": item["confidence"],
                "occlusion": item["occlusion"],
                "interpolation_used": item["interpolation_used"],
                "reference_main_path": int(item["reference_main_path"]),
                "bbox_diagonal_px": plant["bbox_diagonal_px"],
                "length_px": item["length_px"],
                "length_bbox_normalized": item["length_bbox_normalized"],
                "length_mm_derived_from_scanner_metadata": item["length_mm_derived_from_scanner_metadata"],
                "divergence_angle_status": item["divergence_angle_status"],
                "divergence_angle_deg": item["divergence_angle_deg"],
                "tip_xy_json": json.dumps(item["tip_xy"], separators=(",", ":")),
                "points_px_json": json.dumps(item["points_px"], separators=(",", ":")) if item["points_px"] else "",
                "note": item["note"],
            })
    return stream.getvalue().encode("utf-8")


def build(output: Path = OUTPUT, *, write: bool = False) -> dict:
    results, public, admin = RUNTIME / "results", RUNTIME / "public", RUNTIME / "admin"
    files = {
        **{name: results / name for name in ("raw_annotations.json", "sessions.csv", "traces.csv", "SHA256_LEDGER.json")},
        "mapping.json": admin / "mapping.json",
        "measurement_manifest.json": public / "measurement_manifest.json",
        "test.csv": STAGE / "manifests" / "test.csv",
        "mask_integrity.csv": STAGE / "manifests" / "mask_integrity.csv",
        "phenotype_gt_geometry.py": PROTOCOL / "phenotype_gt_geometry.py",
    }
    for name, path in files.items():
        require(sha(path) == EXPECTED[name], f"Frozen input changed: {name}")
    report = validate_exports(results, public)
    require(report["status"] == "PASS" and report["n"] == 40, "Formal raw export invalid")
    payload = json.loads(json.loads((results / "raw_annotations.json").read_text(encoding="utf-8"))["payload_json"])
    mapping = json.loads((admin / "mapping.json").read_text(encoding="utf-8"))["samples"]
    manifest = json.loads((public / "measurement_manifest.json").read_text(encoding="utf-8"))["samples"]
    by_blind = _unique(mapping, "blind_id")
    by_manifest = _unique(manifest, "blind_id")
    by_dataset = _unique(mapping, "dataset_id")
    meta = _unique(_rows(files["test.csv"]), "dataset_id")
    integrity = _unique([r for r in _rows(files["mask_integrity.csv"]) if r["split"] == "test"], "dataset_id")
    require(len(by_blind) == len(by_dataset) == len(meta) == len(integrity) == 40, "Locked test membership mismatch")
    require(set(by_blind) == set(by_manifest) == {r["blind_id"] for r in payload["samples"]}, "Blind membership mismatch")
    require(set(by_dataset) == set(meta) == set(integrity), "Dataset membership mismatch")
    plants, mask_hashes = [], {}
    for sample in payload["samples"]:
        ident = sample["blind_id"]
        snapshot = sample["history"][-1]
        record = json.loads(snapshot["snapshot_json"])
        require(record["image_sha256"] == by_manifest[ident]["image_sha256"], "Public image identity mismatch")
        mapped = {**by_blind[ident], "record_sha256": snapshot["sha256"]}
        plant, mask_sha = _plant(record, meta, mapped, integrity)
        plants.append(plant)
        mask_hashes[plant["dataset_id"]] = mask_sha
    plants.sort(key=lambda p: p["dataset_id"])
    reference = {
        "version": "v4-test-single-rater-final-gt-v1",
        "status": "frozen_single_rater_model_blind",
        "scope": "40 locked V4 test plants; no model predictions read",
        "reference_name": "single-rater model-blind phenotype reference",
        "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
        "length_name": "base-to-tip structural path length",
        "millimeter_status": "derived from scanner metadata; not ruler-calibrated",
        "source_sha256": EXPECTED,
        "shoot_mask_sha256": mask_hashes,
        "plants": plants,
    }
    raw = _json_bytes(reference)
    csv_data = _csv_bytes(plants)
    summary = {
        "version": reference["version"],
        "status": reference["status"],
        "n_plants": len(plants),
        "n_structures": sum(len(p["structures"]) for p in plants),
        "measurable": sum(p["measurable_reference_path_count"] for p in plants),
        "visible_unmeasurable": sum(s["visibility_status"] == "visible_unmeasurable" for p in plants for s in p["structures"]),
        "uncertain": sum(s["visibility_status"] == "uncertain" for p in plants for s in p["structures"]),
        "non_target_structure": sum(s["visibility_status"] == "non_target_structure" for p in plants for s in p["structures"]),
        "resolved_divergence_angles": sum(s["divergence_angle_status"] == "ok" for p in plants for s in p["paths"]),
        "unresolved_divergence_angles": sum(s["divergence_angle_status"] == "divergence_unresolved" for p in plants for s in p["paths"]),
        "final_gt_json_sha256": hashlib.sha256(raw).hexdigest(),
        "final_gt_csv_sha256": hashlib.sha256(csv_data).hexdigest(),
        "freeze_script_sha256": sha(Path(__file__)),
        "raw_export_sha256": EXPECTED["raw_annotations.json"],
        "admin_mapping_sha256": EXPECTED["mapping.json"],
        "test_model_reads": 0,
        "inference_gate": "CLOSED",
        "test_evaluator": "PENDING",
    }
    if write:
        output.mkdir(parents=True, exist_ok=False)
        (output / "final_gt.json").write_bytes(raw)
        (output / "final_gt_structures.csv").write_bytes(csv_data)
        (output / "freeze_manifest.json").write_bytes(_json_bytes(summary))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create the private immutable freeze directory once")
    args = parser.parse_args()
    print(json.dumps(build(write=args.write), ensure_ascii=False, indent=2))

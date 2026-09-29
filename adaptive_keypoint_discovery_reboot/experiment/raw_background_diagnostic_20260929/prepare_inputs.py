"""Prepare exactly one aligned, uncleaned V4 train crop per source frame.

No model is imported or run. Only the locked train manifest and its sources are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
PROTOCOL = HERE / "PROTOCOL.md"
SEED = "raw-bg-train-20260929"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_train_rows() -> list[dict]:
    with (DATASET / "manifests/train.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 220 or any(row["split"] != "train" for row in rows):
        raise RuntimeError("V4 train manifest identity changed")
    by_frame: dict[str, list[dict]] = {}
    for row in rows:
        frame = row["source_frame_id"].strip()
        if frame:
            by_frame.setdefault(frame, []).append(row)
    if len(by_frame) != 20:
        raise RuntimeError(f"Expected 20 non-legacy source frames, found {len(by_frame)}")
    chosen = []
    for frame in sorted(by_frame):
        group = by_frame[frame]
        chosen.append(min(group, key=lambda item: hashlib.sha256(
            f"{SEED}|{item['dataset_id']}".encode("utf-8")
        ).hexdigest()))
    if len(chosen) != 20 or len({r["dataset_id"] for r in chosen}) != 20:
        raise RuntimeError("Frame-level selection failed")
    return sorted(chosen, key=lambda row: row["dataset_id"])


def reconstruct(row: dict) -> tuple[np.ndarray, dict]:
    # Use the original V4 reader and resize implementation; do not regenerate masks.
    sys.path.insert(0, str(EXPERIMENT))
    from build_stage_clean_v4_fullplant import read_rgb, resize_maximum

    ident = row["dataset_id"]
    source = Path(row["source_path"])
    if not source.is_file() or not row["source_frame_id"].strip():
        raise RuntimeError(f"Original non-legacy source missing: {ident}")
    full = read_rgb(source)
    box = json.loads(row["source_crop_box_fullplant"])
    if len(box) != 4 or any(not isinstance(v, int) for v in box):
        raise RuntimeError(f"Invalid frozen source crop box: {ident}")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= full.shape[1] and 0 <= y0 < y1 <= full.shape[0]):
        raise RuntimeError(f"Source crop outside scanner frame: {ident}")
    cropped = resize_maximum(full[y0:y1, x0:x1], 1600)
    padding = int(float(row["normalization_padding_pixels"]))
    if padding < 1:
        raise RuntimeError(f"Invalid locked padding: {ident}")
    aligned = cv2.copyMakeBorder(cropped, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    clean_path = DATASET / Path(row["relative_path"].replace("\\", "/"))
    if sha(clean_path) != row["output_sha256"]:
        raise RuntimeError(f"Clean train image hash mismatch: {ident}")
    with Image.open(clean_path) as opened:
        clean_size = opened.size
    if (aligned.shape[1], aligned.shape[0]) != clean_size:
        raise RuntimeError(f"Raw/clean coordinate dimensions differ: {ident}")
    return aligned, {"source_path": str(source), "source_sha256": sha(source),
                     "source_crop_box_fullplant": box, "source_crop_size": [int(cropped.shape[1]), int(cropped.shape[0])],
                     "padding_px": padding, "clean_sha256": row["output_sha256"],
                     "width": clean_size[0], "height": clean_size[1]}


def prepare(output: Path) -> None:
    rows = selected_train_rows()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite paired input package: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_aligned").mkdir()
    records = []
    for row in rows:
        ident = row["dataset_id"]
        aligned, evidence = reconstruct(row)
        target = output / "raw_aligned" / f"{ident}.png"
        Image.fromarray(aligned, "RGB").save(target, optimize=True)
        records.append({"dataset_id": ident, "source_frame_id": row["source_frame_id"],
                        "raw_relative_path": f"raw_aligned/{ident}.png", "raw_sha256": sha(target), **evidence})
    manifest = {"kind": "frozen V4 train raw-background paired diagnostic input; no model run",
                "selection_seed": SEED, "split": "train", "count": len(records),
                "train_manifest_sha256": sha(DATASET / "manifests/train.csv"),
                "protocol_sha256": sha(PROTOCOL), "prepare_source_sha256": sha(Path(__file__)),
                "records": records}
    target = output / "input_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(records),
                      "independent_source_frames": len({r['source_frame_id'] for r in records}),
                      "input_manifest_sha256": sha(target)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)

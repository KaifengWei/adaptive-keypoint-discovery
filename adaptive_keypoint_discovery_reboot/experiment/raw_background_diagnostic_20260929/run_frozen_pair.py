"""Run one exploratory frozen clean/raw pair on 20 locked V4 train plants.

Never reads val/test manifests, final GT, or model outputs for parameter choice.
The raw condition changes only pixels inside the clean-derived frozen ROI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
DEFAULT_INPUT = EXPERIMENT / "phenotype_pilot_protocol/runtime/raw_background_train_20260929/input_01"
DEFAULT_OUTPUT = EXPERIMENT / "phenotype_pilot_protocol/runtime/raw_background_train_20260929/run_01"
INPUT_MANIFEST_SHA = "a270ce04284602277aa370ee9c9ad01cd6b9ed2d4476a13fad312a83547706ff"
METHODS = ("Teacher-direct", "Student-B", "Student-D")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(input_root: Path) -> tuple[list[dict], dict[str, dict]]:
    manifest_path = input_root / "input_manifest.json"
    if sha(manifest_path) != INPUT_MANIFEST_SHA:
        raise RuntimeError("Paired raw input manifest hash mismatch")
    package = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = package["records"]
    if package["split"] != "train" or package["count"] != 20 or len(records) != 20:
        raise RuntimeError("This run accepts only the frozen 20-train package")
    if len({r["dataset_id"] for r in records}) != 20 or len({r["source_frame_id"] for r in records}) != 20:
        raise RuntimeError("Frozen train identity/frame uniqueness failed")
    if sha(DATASET / "manifests/train.csv") != package["train_manifest_sha256"]:
        raise RuntimeError("Train manifest SHA mismatch")
    if sha(HERE / "PROTOCOL.md") != package["protocol_sha256"]:
        raise RuntimeError("Preregistered protocol SHA mismatch")
    if sha(HERE / "prepare_inputs.py") != package["prepare_source_sha256"]:
        raise RuntimeError("Input preparation source SHA mismatch")
    with (DATASET / "manifests/train.csv").open(encoding="utf-8-sig", newline="") as stream:
        train = {r["dataset_id"]: r for r in csv.DictReader(stream)}
    for record in records:
        ident = record["dataset_id"]
        row = train.get(ident)
        if row is None or row["split"] != "train" or row["source_frame_id"] != record["source_frame_id"]:
            raise RuntimeError(f"Non-train or changed identity: {ident}")
        raw = input_root / record["raw_relative_path"]
        clean = DATASET / Path(row["relative_path"].replace("\\", "/"))
        if sha(raw) != record["raw_sha256"] or sha(clean) != record["clean_sha256"]:
            raise RuntimeError(f"Aligned raw/clean SHA mismatch: {ident}")
    return records, train


def model_inputs(record: dict, train: dict, input_root: Path, frozen):
    from phenotype_roi_basal_anchor import apply_phenotype_roi, letterbox_rgb_array, load_phenotype_input

    ident = record["dataset_id"]
    row = train[ident]
    clean_canvas, mapping, _, roi = load_phenotype_input(DATASET, row, 518)
    masks = {
        name: frozen.graph_eval.load_mask_canvas(
            DATASET / Path(row[key].replace("\\", "/")), mapping, 518
        )
        for name, key in (
            ("shoot", "shoot_mask_relative_path"),
            ("seed_base_root", "seed_base_root_mask_relative_path"),
            ("full_plant", "full_plant_mask_relative_path"),
        )
    }
    masks["phenotype_roi"] = roi["phenotype_roi_model"]
    masks["basal_transition"] = roi["basal_transition_model"]
    raw_path = input_root / record["raw_relative_path"]
    raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise RuntimeError(f"Could not decode frozen raw crop: {ident}")
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    if (raw_rgb.shape[1], raw_rgb.shape[0]) != (record["width"], record["height"]):
        raise RuntimeError(f"Raw image dimensions changed: {ident}")
    raw_focused = apply_phenotype_roi(raw_rgb, roi["phenotype_roi"])
    raw_canvas, raw_mapping = letterbox_rgb_array(raw_focused, 518)
    if raw_mapping != mapping or raw_canvas.shape != clean_canvas.shape or raw_canvas.dtype != np.uint8:
        raise RuntimeError(f"Raw/clean model-canvas mapping mismatch: {ident}")
    return {"clean": clean_canvas, "raw": raw_canvas}, mapping, masks


def run(method: str, input_root: Path, output_root: Path) -> None:
    if method not in METHODS:
        raise ValueError(method)
    records, train = load_plan(input_root)
    # Import functions, not the once-only test CLI. Neither import reads test pixels.
    sys.path.insert(0, str(EXPERIMENT / "method_gate_20260925"))
    sys.path.insert(0, str(EXPERIMENT / "locked_test_final_20260927"))
    sys.path.insert(0, str(EXPERIMENT))
    import run_frozen_diagnostics as frozen
    import run_v4_locked_test_once as frozen_pipeline
    import torch

    evidence = frozen_pipeline.verify_static()  # frozen source/weight SHA and tracked Git cleanliness
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("Frozen RTX3090 unavailable")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{method}.jsonl"
    if output.exists():
        raise FileExistsError(f"One-pass diagnostic result already exists: {output}")
    model = frozen_pipeline._method_model(method, frozen)
    # The file is exclusive; no execution can silently overwrite or append to a prior run.
    errors = 0
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            ident = record["dataset_id"]
            inputs, mapping, masks = model_inputs(record, train, input_root, frozen)
            for condition in ("clean", "raw"):
                started = time.perf_counter()
                try:
                    canvas = inputs[condition]
                    points = model.points(canvas)
                    paths, decision, _, graph = frozen_pipeline.graph_decoder_once(canvas, masks, points, frozen)
                    torch.cuda.synchronize()
                    row = dict(train[ident])
                    row["output_sha256"] = record[f"{condition}_sha256"]
                    sample = {"mapping": mapping}
                    prediction = frozen_pipeline.prediction_row(method, row, sample, points, paths, decision, graph)
                    event = {"dataset_id": ident, "source_frame_id": record["source_frame_id"],
                             "method": method, "condition": condition, "status": "ok",
                             "input_png_sha256": record[f"{condition}_sha256"], "prediction": prediction,
                             "elapsed_seconds": time.perf_counter() - started}
                except Exception as exc:
                    errors += 1
                    event = {"dataset_id": ident, "source_frame_id": record["source_frame_id"],
                             "method": method, "condition": condition, "status": "error",
                             "input_png_sha256": record[f"{condition}_sha256"],
                             "error_type": type(exc).__name__, "error": str(exc),
                             "elapsed_seconds": time.perf_counter() - started}
                stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps({"method": method, "id": ident, "condition": condition,
                                  "status": event["status"]}, ensure_ascii=False), flush=True)
    ledger = {"kind": "20-train frozen clean/raw paired exploratory diagnostic; no val/test/GT read",
              "method": method, "records": 40, "errors": errors,
              "input_manifest_sha256": INPUT_MANIFEST_SHA,
              "runner_source_sha256": sha(Path(__file__)),
              "git_commit": evidence["git_commit"], "frozen_weights_sha256": evidence["method_weights_sha256"],
              "frozen_sources_sha256": evidence["pinned_source_sha256"],
              "output_sha256": sha(output)}
    ledger_path = output_root / f"{method}_ledger.json"
    with ledger_path.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"method": method, "records": 40, "errors": errors,
                      "output_sha256": ledger["output_sha256"]}, ensure_ascii=False), flush=True)


def preflight(input_root: Path) -> None:
    records, train = load_plan(input_root)
    sys.path.insert(0, str(EXPERIMENT / "method_gate_20260925"))
    sys.path.insert(0, str(EXPERIMENT))
    import run_frozen_diagnostics as frozen

    changed_fractions = []
    changed_within_roi = []
    for record in records:
        inputs, mapping, masks = model_inputs(record, train, input_root, frozen)
        if set(inputs) != {"clean", "raw"} or mapping["source_width"] != record["width"]:
            raise RuntimeError("Paired model input preflight failed")
        if any(mask.shape != (518, 518) for mask in masks.values()):
            raise RuntimeError("Frozen mask canvas shape changed")
        changed = np.any(inputs["clean"] != inputs["raw"], axis=2)
        changed_fractions.append(float(changed.mean()))
        changed_within_roi.append(float(changed[masks["phenotype_roi"]].mean()))
    if min(changed_fractions) <= 0:
        raise RuntimeError("Raw and clean model inputs are identical for at least one plant")
    print(json.dumps({"preflight": "PASS", "split": "train", "plants": len(records),
                      "paired_canvases": 2 * len(records),
                      "changed_pixel_fraction_min": min(changed_fractions),
                      "changed_pixel_fraction_median": float(np.median(changed_fractions)),
                      "changed_pixel_fraction_within_roi_median": float(np.median(changed_within_roi)),
                      "test_model_reads": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.preflight:
        if args.method is not None:
            parser.error("--preflight must not specify --method")
        preflight(args.input)
    else:
        if args.method is None:
            parser.error("--method is required unless --preflight is used")
        run(args.method, args.input, args.output)

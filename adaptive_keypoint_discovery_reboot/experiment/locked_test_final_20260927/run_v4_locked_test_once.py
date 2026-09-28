"""Once-only GT-blind V4 test prediction producer for the frozen methods.

This script never imports, locates, or reads final_gt.json.  It orchestrates
the frozen point generators and the same frozen Pipeline V1 graph/decoder.
The three method commands must be separate processes in Teacher/B/D order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parent.parent
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
RUNTIME = EXPERIMENT / "phenotype_pilot_protocol" / "runtime"
RUN_ROOT = RUNTIME / "v4_locked_test_final_20260929" / "run_01"
METHODS = ("Teacher-direct", "Student-B", "Student-D")
WEIGHTS = {
    "Teacher-direct": ("third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth", "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"),
    "Student-B": ("training_outputs/core_dinov2_v4_phenotype_roi/best.pt", "bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2"),
    "Student-D": ("training_outputs/core_dinov2_v4_structure_coverage/best.pt", "b904eed30832d1a2c6cc20aca97e3d0140cc4444235c6a3b4b4b606bab17ce4a"),
}
PINNED = {
    "locked_test_final_20260927/V4_LOCKED_TEST_PREREGISTRATION.md": "31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9",
    "locked_test_final_20260927/V4_LOCKED_TEST_EVALUATOR.py": "07e6ad7b06dfe58019c85e1956e0e60de84333c6b203109074a2c8231d43e167",
    "phenotype_pilot_protocol/phenotype_gt_geometry.py": "c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800",
    "phenotype_pilot_protocol/evaluate_frozen_phenotype_pilot.py": "b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8",
    "point_conditioned_graph.py": "3a733ee2fb4213938f92a0f91c969bffe2b4c8e23b82356f07fd03ebba9577ca",
    "point_conditioned_organ_paths.py": "f960485eb6265122955ccc3ffa8a944c2fa66127f8c9c5b58683b46d3c3ab357",
    "phenotype_roi_basal_anchor.py": "d4dfedc3b3fa25c7a086d276d3e4f9ab9728ca7c9a28b2130cb7f678041db696",
    "g1_prime_phenotype_bridge.py": "027a364330888b403823c98ff77deb5e00db8792ad35e0831a45e97138292117",
    "method_gate_20260925/run_frozen_diagnostics.py": "1ab3fa19dafe4781e7e25a1fa12c6d02954621501cbc678a5c3080788e6a322f",
}
PIPELINE_SHA = "7e34775b96042436f41dda9794c85068c56af9c199cce26d58d4c2441d480925"
RUN_SEED = 20260925
WARMUP_ID = "v4_legacy_0001"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def exclusive_write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def records(split: str) -> list[dict]:
    with (DATASET / "manifests" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(rows, key=lambda item: item["dataset_id"])


def verify_static() -> dict:
    for relative, expected in PINNED.items():
        path = EXPERIMENT / relative
        if sha(path) != expected:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    for relative, expected in WEIGHTS.values():
        if sha(EXPERIMENT / relative) != expected:
            raise RuntimeError(f"Frozen weight mismatch: {relative}")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    for diff_args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["git", "-C", str(REPO), *diff_args], check=False).returncode != 0:
            raise RuntimeError("Tracked worktree changed before formal run")
    return {"git_commit": head, "pinned_source_sha256": PINNED, "method_weights_sha256": {m: h for m, (_, h) in WEIGHTS.items()}}


def verify_test_assets() -> list[dict]:
    """Byte and header-only audit; no model call and no content QC."""
    from PIL import Image
    rows = records("test")
    if len(rows) != 40 or len({r["dataset_id"] for r in rows}) != 40:
        raise RuntimeError("Test manifest membership changed")
    for row in rows:
        image = DATASET / Path(row["relative_path"].replace("\\", "/"))
        if sha(image) != row["output_sha256"]:
            raise RuntimeError(f"Locked test image SHA mismatch: {row['dataset_id']}")
        with Image.open(image) as header:
            if header.format != "PNG" or min(header.size) <= 0:
                raise RuntimeError(f"Invalid locked PNG header: {row['dataset_id']}")
            row["locked_width"], row["locked_height"] = map(int, header.size)
    return rows


def create_run() -> None:
    evidence = verify_static()
    test_rows = verify_test_assets()
    warmup = next((r for r in records("train") if r["dataset_id"] == WARMUP_ID), None)
    if warmup is None or sha(DATASET / Path(warmup["relative_path"].replace("\\", "/"))) != warmup["output_sha256"]:
        raise RuntimeError("Locked train warmup image mismatch")
    if RUN_ROOT.exists():
        raise FileExistsError("Once-only run directory already exists; never overwrite or restart")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    ledger = {
        "schema_version": "v4-locked-test-run-ledger-v1",
        "git_commit": evidence["git_commit"], "test_model_reads_initial": 0,
        "run_seed": RUN_SEED, "warmup_train_id": WARMUP_ID, "warmup_passes_per_method": 2,
        "method_order": list(METHODS), "planned_method_plant_instances": [
            {"method": method, "dataset_id": row["dataset_id"], "expected_image_sha256": row["output_sha256"]}
            for method in METHODS for row in test_rows
        ],
        "pinned_source_sha256": PINNED,
        "method_weights_sha256": {m: h for m, (_, h) in WEIGHTS.items()},
        "pipeline_v1_sha256": PIPELINE_SHA,
    }
    exclusive_write(RUN_ROOT / "run_ledger.json", json_bytes(ledger))
    print(json.dumps({"run_root": str(RUN_ROOT), "planned_instances": 120, "git_commit": evidence["git_commit"]}, ensure_ascii=False), flush=True)


def _method_model(name: str, frozen):
    if name != "Student-D":
        return frozen.Method(name)
    import torch
    from adaptive_point_model import AdaptivePointDetector
    config = json.loads((EXPERIMENT / "training_outputs/core_dinov2_v4_structure_coverage/resolved_config.json").read_text())
    if (config["input_domain"], config["image_size"], config["decoder_dim"], config["output_stride"],
        config["inference_threshold"], config["inference_safety_cap"], config["fixed_k_eval"]) != (
        "phenotype_roi_v1", 518, 192, 4, 0.35, 64, 0
    ):
        raise RuntimeError("Frozen Student-D inference configuration changed")
    if sha(EXPERIMENT / WEIGHTS[name][0]) != WEIGHTS[name][1]:
        raise RuntimeError("Student-D checkpoint SHA changed before load")
    frozen.g1.set_deterministic(RUN_SEED)
    model_args = argparse.Namespace(local_repo=Path(config["dinov2_local_repo"]), weights=Path(config["dinov2_weights"]), model=config["dinov2_model"])
    device = torch.device("cuda")
    backbone = frozen.g1.load_official_model(model_args, device)
    checkpoint = torch.load(EXPERIMENT / WEIGHTS[name][0], map_location=device, weights_only=False)
    model = AdaptivePointDetector(
        backbone, patch_size=14, decoder_dim=192, output_stride=4,
        freeze_backbone=True, unfreeze_last_blocks=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    class FrozenD:
        name = "Student-D"
        @torch.inference_mode()
        def points(self, image):
            _, found = frozen.student_eval.predict(model, image, device, 0.35, 64, 0)
            return [
                {"point_id": f"p{i:02d}", "x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"]), "kind": p["kind"]}
                for i, p in enumerate(found, 1)
            ]
    return FrozenD()


def _sample(record: dict, frozen) -> dict:
    image, mapping, _, roi = frozen.load_phenotype_input(DATASET, record, 518)
    masks = {
        name: frozen.graph_eval.load_mask_canvas(DATASET / Path(record[key].replace("\\", "/")), mapping, 518)
        for name, key in (
            ("shoot", "shoot_mask_relative_path"),
            ("seed_base_root", "seed_base_root_mask_relative_path"),
            ("full_plant", "full_plant_mask_relative_path"),
        )
    }
    masks["phenotype_roi"] = roi["phenotype_roi_model"]
    masks["basal_transition"] = roi["basal_transition_model"]
    return {"image": image, "mapping": mapping, "masks": masks}


def graph_decoder_once(image, masks, points, frozen):
    """Same calls/constants/order as frozen diagnostics; expose graph provenance."""
    import cv2
    import math
    import numpy as np
    support, raw_skeleton, _ = frozen.gp.automatic_structural_support(image)
    bbox = frozen.gp.bbox_from_mask(support)
    diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    skeleton, _ = frozen.bridge.prune_short_terminal_spurs(raw_skeleton, max(4.0, 0.012 * diag))
    graph = frozen.build_point_conditioned_graph(skeleton, points, diag, 0.025)
    radius = max(2, round(image.shape[0] * 0.01))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    tolerant = {name: cv2.dilate(mask.astype(np.uint8), kernel) > 0 for name, mask in masks.items()}
    frozen.graph_eval.annotate_regions(graph, masks, tolerant)
    paths, decision = frozen.decode_candidate_organ_paths(
        graph, masks["shoot"], masks["seed_base_root"], diag,
        phenotype_roi_mask=masks["phenotype_roi"], basal_transition_mask=masks["basal_transition"],
        branch_pruning_mode="local_learned_support",
    )
    return paths, decision, diag, graph


def prediction_row(method: str, record: dict, sample: dict, points: list[dict], paths: list[dict], decision: dict, graph: dict) -> dict:
    base_id = decision.get("base_node_id")
    graph_failed = bool(graph["diagnostics"]["failure"])
    basal_status = "graph_failure" if graph_failed else ("ok" if base_id is not None else "no_eligible_basal_node")
    base = None
    if base_id is not None:
        base = next(node["projected_xy"] for node in graph["nodes"] if int(node["node_id"]) == int(base_id))
    path_rows = [{
        "path_id": str(path["path_id"]), "full_base_to_tip_path": path["full_base_to_tip_path"],
        "provenance": {
            "path_kind": path["path_kind"], "base_node_id": path["base_node_id"],
            "tip_node_id": path["tip_node_id"], "support_records": path["support_records"],
        },
    } for path in paths]
    return {
        "dataset_id": record["dataset_id"], "image_sha256": record["output_sha256"],
        "width": int(sample["mapping"]["source_width"]), "height": int(sample["mapping"]["source_height"]),
        "source_frame_id": record["source_frame_id"], "paths": path_rows,
        "base_xy_model_canvas": base, "basal_node_status": basal_status,
        "graph_status": "failed" if graph_failed else "ok",
        "decoder_status": "ok" if paths else "zero_path",
        "point_count": len(points), "accepted_node_count": len(graph["nodes"]),
        "rejected_node_count": len(graph["rejected_points"]), "graph_node_count": len(graph["nodes"]),
        "underground_association_count": sum(node.get("organ_region_tolerant", "") == "seed_base_root" for node in graph["nodes"]),
        "provenance": {
            "point_source": method, "point_records": points,
            "graph_diagnostics": graph["diagnostics"],
            "accepted_nodes": graph["nodes"], "rejected_points": graph["rejected_points"],
            "decoder_decision": decision,
        },
    }


def _events() -> list[dict]:
    path = RUN_ROOT / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def run_method(name: str) -> None:
    if name not in METHODS or not (RUN_ROOT / "run_ledger.json").is_file():
        raise RuntimeError("Formal run was not initialized")
    evidence = verify_static()
    ledger = json.loads((RUN_ROOT / "run_ledger.json").read_text())
    if evidence["git_commit"] != ledger["git_commit"]:
        raise RuntimeError("Run git commit changed")
    events = _events()
    if any(e.get("event") == "instance_execution_error" for e in events):
        raise RuntimeError("Prior execution fault: no automatic restart")
    prior = METHODS[:METHODS.index(name)]
    for predecessor in prior:
        if sum(e.get("event") == "instance_completed" and e.get("method") == predecessor for e in events) != 40:
            raise RuntimeError("Teacher/B/D order or previous method completeness violated")
    if any(e.get("method") == name for e in events) or (RUN_ROOT / f"{name}.jsonl").exists():
        raise RuntimeError("Method already started: no repeat")
    test_rows = verify_test_assets()
    planned = [item["dataset_id"] for item in ledger["planned_method_plant_instances"] if item["method"] == name]
    if [item["dataset_id"] for item in test_rows] != planned:
        raise RuntimeError("Locked test run-plan membership changed")
    train = next(r for r in records("train") if r["dataset_id"] == WARMUP_ID)
    sys.path.insert(0, str(EXPERIMENT / "method_gate_20260925"))
    import run_frozen_diagnostics as frozen
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("Locked RTX3090 unavailable")
    model = _method_model(name, frozen)
    warmup_sample = _sample(train, frozen)
    for warmup_index in range(2):
        points = model.points(warmup_sample["image"])
        paths, _, diag, _ = graph_decoder_once(warmup_sample["image"], warmup_sample["masks"], points, frozen)
        frozen.phenotype(paths, diag, warmup_sample["mapping"])
        torch.cuda.synchronize()
        append_jsonl(RUN_ROOT / "events.jsonl", {"event": "non_test_warmup_completed", "method": name, "warmup_index": warmup_index + 1, "train_dataset_id": WARMUP_ID})
    for index, record in enumerate(test_rows, start=1):
        ident = record["dataset_id"]
        append_jsonl(RUN_ROOT / "events.jsonl", {"event": "instance_started", "method": name, "dataset_id": ident, "ordinal": index})
        try:
            io_start = time.perf_counter_ns()
            sample = _sample(record, frozen)
            io_ms = (time.perf_counter_ns() - io_start) / 1e6
            torch.cuda.synchronize()
            baseline_gpu = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            baseline_ram = frozen.rss_bytes()
            rss = frozen.RssSampler()
            rss.start()
            try:
                start = time.perf_counter_ns()
                points = model.points(sample["image"])
                torch.cuda.synchronize()
                point_end = time.perf_counter_ns()
                paths, decision, diag, graph = graph_decoder_once(sample["image"], sample["masks"], points, frozen)
                torch.cuda.synchronize()
                graph_end = time.perf_counter_ns()
                phenotype = frozen.phenotype(paths, diag, sample["mapping"])
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
            finally:
                peak_ram = rss.stop()
            row = prediction_row(name, record, sample, points, paths, decision, graph)
            if (row["width"], row["height"]) != (int(record["locked_width"]), int(record["locked_height"])):
                raise RuntimeError("Locked dimensions changed")
            result = {
                "prediction": row,
                "runtime": {
                    "dataset_id": ident, "method": name, "ordinal": index,
                    "image_io_ms_excluded": io_ms,
                    "point_generation_only_ms": (point_end - start) / 1e6,
                    "shared_graph_decoder_ms": (graph_end - point_end) / 1e6,
                    "phenotype_geometry_ms": (end - graph_end) / 1e6,
                    "end_to_end_phenotype_ms": (end - start) / 1e6,
                    "baseline_gpu_allocated_bytes": int(baseline_gpu),
                    "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "baseline_ram_vmrss_bytes": int(baseline_ram),
                    "observed_peak_ram_vmrss_bytes": int(peak_ram),
                    "vmrss_poll_interval_ms": 5,
                    "phenotype_path_count": len(phenotype),
                },
            }
            append_jsonl(RUN_ROOT / f"{name}.jsonl", result)
            append_jsonl(RUN_ROOT / "events.jsonl", {"event": "instance_completed", "method": name, "dataset_id": ident, "ordinal": index, "prediction_count": len(paths)})
            print(f"{name} {index}/40 {ident} paths={len(paths)}", flush=True)
        except BaseException as exc:
            append_jsonl(RUN_ROOT / "events.jsonl", {"event": "instance_execution_error", "method": name, "dataset_id": ident, "ordinal": index, "error_type": type(exc).__name__, "error": str(exc)})
            raise


def seal() -> None:
    if not (RUN_ROOT / "run_ledger.json").is_file():
        raise RuntimeError("Formal run missing")
    if (RUN_ROOT / "v4-locked-test-saved-predictions-v1.json").exists():
        raise FileExistsError("Prediction package already sealed")
    events = _events()
    if any(e["event"] == "instance_execution_error" for e in events):
        raise RuntimeError("Execution fault recorded; cannot seal incomplete run")
    expected_ids = [r["dataset_id"] for r in records("test")]
    methods, timings = {}, {}
    for name in METHODS:
        lines = [json.loads(line) for line in (RUN_ROOT / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        if len(lines) != 40 or [item["prediction"]["dataset_id"] for item in lines] != expected_ids:
            raise RuntimeError(f"Incomplete or reordered method: {name}")
        complete = [e["dataset_id"] for e in events if e["event"] == "instance_completed" and e.get("method") == name]
        starts = [e["dataset_id"] for e in events if e["event"] == "instance_started" and e.get("method") == name]
        if starts != expected_ids or complete != expected_ids:
            raise RuntimeError(f"Run ledger incomplete: {name}")
        methods[name] = [item["prediction"] for item in lines]
        timings[name] = [item["runtime"] for item in lines]
    bundle = {
        "schema_version": "v4-locked-test-saved-predictions-v1", "split": "test",
        "pipeline_v1_sha256": PIPELINE_SHA,
        "method_weights_sha256": {m: h for m, (_, h) in WEIGHTS.items()},
        "methods": methods,
    }
    package = RUN_ROOT / "v4-locked-test-saved-predictions-v1.json"
    exclusive_write(package, json_bytes(bundle))
    exclusive_write(RUN_ROOT / "runtime_40x3.json", json_bytes({"methods": timings, "hardware": "same RTX3090, batch 1, separate method processes"}))
    manifest = {"prediction_package_sha256": sha(package), "runtime_sha256": sha(RUN_ROOT / "runtime_40x3.json"), "method_rows": {m: len(methods[m]) for m in METHODS}, "run_ledger_sha256": sha(RUN_ROOT / "run_ledger.json"), "event_ledger_sha256": sha(RUN_ROOT / "events.jsonl")}
    exclusive_write(RUN_ROOT / "prediction_freeze_manifest.json", json_bytes(manifest))
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--method", choices=METHODS)
    group.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.create:
        create_run()
    elif args.method:
        run_method(args.method)
    else:
        seal()


if __name__ == "__main__":
    main()

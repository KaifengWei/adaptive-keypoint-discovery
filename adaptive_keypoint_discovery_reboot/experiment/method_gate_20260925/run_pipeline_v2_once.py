"""Frozen val-only Pipeline V2: V1 preflight, then one sealed 40x2 formal run.

The preflight never calls V2 on real images. GT and V4 test have no input route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import evaluate_point_conditioned_graph_v1 as graph_eval  # noqa: E402
import g1_prime_phenotype_bridge as bridge  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
from point_conditioned_graph import build_point_conditioned_graph  # noqa: E402
from point_conditioned_organ_paths import decode_candidate_organ_paths  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402
from point_structure_association_v2 import build_point_conditioned_graph_v2  # noqa: E402

DATA = EXP / "data_stage_clean_v4_fullplant_candidate"
MANIFEST = DATA / "manifests/val.csv"
SOURCES = {
    "Teacher-direct": EXP / "phenotype_pilot_protocol/runtime/method_gate_20260925/sources/teacher_points.csv",
    "Student-B": EXP / "evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv",
}
BASELINES = {
    "Teacher-direct": EXP / "phenotype_pilot_protocol/runtime/method_comparison/teacher_direct_route_b_local_decoder_val_20260924",
    "Student-B": EXP / "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val",
}
FROZEN = {
    "teacher_points": (SOURCES["Teacher-direct"], "1ed017a6d6ba6f360bf216c44b01ffedaef9c2a14f18d46d75b1f640763b1814"),
    "student_points": (SOURCES["Student-B"], "9a11bbcc7283333fca34c0e82a3931edbfa96cef8347ec952f450daecb1aa756"),
    "val_manifest": (MANIFEST, "6931e8b5156edbb8cf0f7274e5da618c894ed8e53075fb92aecfc8e146252d42"),
    "frozen_gt_hash_only": (EXP / "phenotype_pilot_protocol/runtime/final_gt/phenotype_pilot_20260924_v1/final_gt.json", "1b3018c83b2695951dfdf2adff10020f9c3e6e739babfe12ad235a88ae7b53b3"),
    "student_checkpoint": (EXP / "training_outputs/core_dinov2_v4_phenotype_roi/best.pt", "bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2"),
    "preregistration": (HERE / "PIPELINE_V2_PREREGISTRATION.md", "8dfb79d5fc7ec52ee2520bd97413becb1fdcab4d82829c0fddd074fa5772b2eb"),
}
CODE_FILES = [
    HERE / "point_structure_association_v2.py", HERE / "test_point_structure_association_v2.py",
    HERE / "run_pipeline_v2_once.py", HERE / "evaluate_pipeline_v2_once.py",
]
FROZEN_CODE = [
    EXP / name for name in (
        "point_conditioned_graph.py", "point_conditioned_organ_paths.py",
        "phenotype_roi_basal_anchor.py", "g1_prime_structural_support.py",
        "g1_prime_phenotype_bridge.py", "evaluate_point_conditioned_graph_v1.py",
        "phenotype_pilot_protocol/phenotype_gt_geometry.py",
        "phenotype_pilot_protocol/evaluate_frozen_phenotype_pilot.py",
    )
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs() -> dict:
    hashes = {}
    for key, (path, expected) in FROZEN.items():
        observed = sha(path)
        if observed != expected:
            raise ValueError(f"Frozen {key} hash mismatch: {observed}")
        hashes[key] = {"path": str(path), "sha256": observed}
    hashes["frozen_code"] = {str(path.relative_to(ROOT)): sha(path) for path in FROZEN_CODE}
    hashes["v1_baselines"] = {
        f"{method}/{name}": sha(folder / name)
        for method, folder in BASELINES.items()
        for name in ("per_image.csv", "paths.jsonl", "summary.json")
    }
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    hashes["val_image_and_mask_assets"] = {
        f"{row['dataset_id']}/{field}": sha(DATA / Path(str(row[field]).replace("\\", "/")))
        for row in manifest.to_dict("records")
        for field in ("relative_path", "shoot_mask_relative_path",
                      "seed_base_root_mask_relative_path", "full_plant_mask_relative_path")
    }
    return hashes


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_code_hashes() -> dict[str, str]:
    if subprocess.run(["git", "merge-base", "--is-ancestor", "6824bd9", "HEAD"], cwd=ROOT).returncode != 0:
        raise ValueError("Accepted preregistration commit is not an ancestor")
    code_hashes = {}
    for path in CODE_FILES:
        relative = path.relative_to(ROOT).as_posix()
        observed = sha(path)
        try:
            committed = hashlib.sha256(subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"], cwd=ROOT, stderr=subprocess.DEVNULL
            )).hexdigest()
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"V2 code is not committed: {relative}") from exc
        if observed != committed:
            raise ValueError(f"V2 code differs from frozen commit: {relative}")
        code_hashes[str(path.relative_to(ROOT))] = observed
    return code_hashes


def jsonable(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if k != "edge_union"}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def inputs():
    manifest = pd.read_csv(MANIFEST, low_memory=False).sort_values("dataset_id")
    if len(manifest) != 40 or set(manifest["split"].astype(str)) != {"val"} or manifest["dataset_id"].nunique() != 40:
        raise ValueError("Frozen val manifest membership changed")
    point_frames = {method: pd.read_csv(path) for method, path in SOURCES.items()}
    ids = set(manifest["dataset_id"].astype(str))
    for method, frame in point_frames.items():
        if set(frame["dataset_id"].astype(str)) != ids or frame[["dataset_id", "point_id"]].duplicated().any():
            raise ValueError(f"{method} saved-point identity changed")
    return manifest.to_dict("records"), point_frames


def frozen_structure(row: dict) -> dict:
    image, mapping, _, roi_result = load_phenotype_input(DATA, row, 518)
    support, raw_skeleton, _ = gp.automatic_structural_support(image)
    bbox = gp.bbox_from_mask(support)
    diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    skeleton, _ = bridge.prune_short_terminal_spurs(raw_skeleton, max(4.0, .012 * diag))
    masks = {
        name: graph_eval.load_mask_canvas(DATA / Path(str(row[field]).replace("\\", "/")), mapping, 518)
        for name, field in (("shoot", "shoot_mask_relative_path"),
                            ("seed_base_root", "seed_base_root_mask_relative_path"),
                            ("full_plant", "full_plant_mask_relative_path"))
    }
    masks["phenotype_roi"] = roi_result["phenotype_roi_model"]
    masks["basal_transition"] = roi_result["basal_transition_model"]
    if any(mask.shape != (518, 518) for mask in (support, skeleton, *masks.values())):
        raise ValueError("Frozen canvas shape changed")
    return {"image": image, "mapping": mapping, "support": support,
            "skeleton": skeleton, "D": diag, "masks": masks}


def decode(graph: dict, structure: dict):
    masks = structure["masks"]
    radius = max(2, round(518 * .01))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    tolerant = {name: cv2.dilate(mask.astype(np.uint8), kernel) > 0 for name, mask in masks.items()}
    graph_eval.annotate_regions(graph, masks, tolerant)
    return decode_candidate_organ_paths(
        graph, masks["shoot"], masks["seed_base_root"], structure["D"],
        phenotype_roi_mask=masks["phenotype_roi"],
        basal_transition_mask=masks["basal_transition"],
        branch_pruning_mode="local_learned_support",
    )


def per_method_points(frame: pd.DataFrame, dataset_id: str, mapping: dict):
    return graph_eval.model_points(frame[frame["dataset_id"] == dataset_id], mapping)


def preflight(output: Path) -> None:
    hashes = checked_inputs()  # GT is hashed as bytes only; never parsed here.
    rows, frames = inputs()
    baseline_frames = {m: pd.read_csv(BASELINES[m] / "per_image.csv").set_index("dataset_id") for m in SOURCES}
    for method, frame in baseline_frames.items():
        if len(frame) != 40 or set(frame["split"].astype(str)) != {"val"}:
            raise ValueError(f"{method} baseline is not complete val 40")
    matches = []
    for ordinal, row in enumerate(rows, 1):
        dataset_id = str(row["dataset_id"])
        structure = frozen_structure(row)
        for method, frame in frames.items():
            points = per_method_points(frame, dataset_id, structure["mapping"])
            # V1 only. Never call build_point_conditioned_graph_v2 in preflight.
            graph = build_point_conditioned_graph(structure["skeleton"], points, structure["D"], .025)
            paths, diagnostic = decode(graph, structure)
            saved = baseline_frames[method].loc[dataset_id]
            observed = (len(points), len(graph["nodes"]), len(paths))
            expected = (int(saved["input_point_count"]), int(saved["accepted_node_count"]), int(saved["decoded_path_count"]))
            if observed != expected:
                raise ValueError(f"V1 regression mismatch {method}/{dataset_id}: {observed} != {expected}")
            matches.append({"method": method, "dataset_id": dataset_id,
                            "point_count": observed[0], "accepted_node_count": observed[1],
                            "path_count": observed[2], "decode_failure": diagnostic.get("failure", "")})
        print(f"V1 preflight {ordinal}/40", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({"status": "PASS", "test_read": 0,
                                  "v2_real_val_calls": 0, "v1_regression_matches": len(matches),
                                  "frozen_input_hashes": hashes, "rows": matches},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V1 regression 80/80 PASS; wrote {output}", flush=True)


def freeze_implementation(output: Path, preflight_file: Path) -> None:
    hashes = checked_inputs()
    pre = json.loads(preflight_file.read_text(encoding="utf-8"))
    if pre.get("status") != "PASS" or pre.get("v1_regression_matches") != 80 or pre.get("v2_real_val_calls") != 0:
        raise ValueError("Complete V1-only preflight is required")
    if pre["frozen_input_hashes"] != hashes:
        raise ValueError("Frozen input hash changed since preflight")
    head = git_head()
    code_hashes = committed_code_hashes()
    record = {"status": "IMPLEMENTATION_FROZEN_BEFORE_VAL_V2", "git_sha": head,
              "code_sha256": code_hashes, "frozen_input_hashes": hashes,
              "preflight_sha256": sha(preflight_file), "test_read": 0,
              "v2_real_val_calls": 0, "gt_read": 0}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V2 implementation frozen before val: {output}", flush=True)


def formal(output: Path, preflight_file: Path, freeze_file: Path) -> None:
    hashes = checked_inputs()
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if freeze.get("status") != "IMPLEMENTATION_FROZEN_BEFORE_VAL_V2" or freeze.get("v2_real_val_calls") != 0 or freeze.get("gt_read") != 0 or freeze.get("test_read") != 0:
        raise ValueError("Pre-val implementation freeze is missing or invalid")
    head = git_head()
    code_hashes = committed_code_hashes()
    if (freeze["git_sha"] != head or freeze["code_sha256"] != code_hashes or
        freeze["frozen_input_hashes"] != hashes or freeze["preflight_sha256"] != sha(preflight_file)):
        raise ValueError("Implementation or frozen inputs changed after pre-val freeze")
    rows, frames = inputs()
    if output.exists():
        raise FileExistsError(f"Formal V2 output already exists: {output}")
    output.mkdir(parents=True)
    point_rows, image_rows, graph_rows, path_rows, decoder_rows = [], [], [], [], []
    for ordinal, row in enumerate(rows, 1):
        dataset_id = str(row["dataset_id"])
        structure = frozen_structure(row)
        masks = structure["masks"]
        for method, frame in frames.items():
            points = per_method_points(frame, dataset_id, structure["mapping"])
            graph, ledger = build_point_conditioned_graph_v2(
                structure["skeleton"], points, structure["D"], structure["support"],
                masks["phenotype_roi"], masks["basal_transition"],
                masks["seed_base_root"], masks["shoot"],
            )
            paths, diagnostic = decode(graph, structure)
            node_paths = {int(node["node_id"]): [] for node in graph["nodes"]}
            for path in paths:
                path_pixels = {tuple(int(v) for v in xy) for xy in path["full_base_to_tip_path"]}
                for node in graph["nodes"]:
                    if tuple(int(v) for v in node["projected_xy"]) in path_pixels:
                        node_paths[int(node["node_id"])].append(path["path_id"])
                for field in ("base_node_id", "tip_node_id"):
                    if int(path.get(field, -1)) in node_paths:
                        node_paths[int(path[field])].append(path["path_id"])
            for item in ledger:
                item["method"] = method
                item["dataset_id"] = dataset_id
                node_id = item["graph_node_id"]
                item["phenotype_path_ids"] = sorted(set(node_paths.get(node_id, []))) if node_id is not None else []
                item["in_phenotype_path"] = bool(item["phenotype_path_ids"])
                point_rows.append(item)
            root_only_nodes = [n for n in graph["nodes"] if masks["seed_base_root"][int(n["projected_xy"][1]), int(n["projected_xy"][0])] and not masks["shoot"][int(n["projected_xy"][1]), int(n["projected_xy"][0])]]
            root_only_path_pixels = 0
            for path in paths:
                for x, y in path["full_base_to_tip_path"]:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < 518 and 0 <= xi < 518 and masks["seed_base_root"][yi, xi] and not masks["shoot"][yi, xi]:
                        root_only_path_pixels += 1
                path_rows.append({"method": method, "dataset_id": dataset_id, **path})
            image_rows.append({"method": method, "dataset_id": dataset_id,
                               "source_frame_id": str(row["source_frame_id"]),
                               "input_point_count": len(points), "accepted_node_count": len(graph["nodes"]),
                               "projection_rejected_count": graph["diagnostics"]["v2_weighted_rejection_count"],
                               "roi_rejected_count": graph["diagnostics"]["v2_roi_rejection_count"],
                               "root_rejected_count": graph["diagnostics"]["v2_root_rejection_count"],
                               "duplicate_merged_count": graph["diagnostics"]["merged_duplicate_count"],
                               "graph_failure": graph["diagnostics"]["failure"],
                               "base_node_id": diagnostic.get("base_node_id", -1),
                               "base_interface_distance_bbox_diag": diagnostic.get("base_interface_distance_bbox_diag"),
                               "decoded_path_count": len(paths), "decode_failure": diagnostic.get("failure", ""),
                               "root_not_shoot_accepted_nodes": len(root_only_nodes),
                               "root_not_shoot_path_pixels": root_only_path_pixels,
                               "bbox_diag_model_px": structure["D"]})
            graph_rows.append({"method": method, "dataset_id": dataset_id,
                               "nodes": graph["nodes"], "rejected_points": graph["rejected_points"],
                               "edges": graph["edges"], "diagnostics": graph["diagnostics"]})
            decoder_rows.append({"method": method, "dataset_id": dataset_id, "decoder": diagnostic})
        print(f"V2 formal {ordinal}/40", flush=True)
    if len(image_rows) != 80 or {r["dataset_id"] for r in image_rows} != {str(r["dataset_id"]) for r in rows}:
        raise ValueError("Incomplete V2 formal 40x2 output")
    files = {"point_associations.jsonl": point_rows, "per_image.jsonl": image_rows,
             "graphs.jsonl": graph_rows, "paths.jsonl": path_rows,
             "decoder_provenance.jsonl": decoder_rows}
    for name, content in files.items():
        write_jsonl(output / name, content)
    summary = {"status": "SEALED_40x2", "test_read": 0, "gt_read": 0,
               "git_sha": head, "code_sha256": code_hashes, "frozen_input_hashes": hashes,
               "preflight_sha256": sha(preflight_file), "implementation_freeze_sha256": sha(freeze_file),
               "methods": list(SOURCES),
               "images_per_method": 40, "image_records": len(image_rows),
               "point_records": len(point_rows), "path_records": len(path_rows),
               "output_sha256": {name: sha(output / name) for name in files}}
    (output / "SEALED_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SEALED 40x2: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "freeze", "formal"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.output)
    elif args.mode == "freeze":
        if args.preflight is None:
            parser.error("--preflight required for freeze")
        freeze_implementation(args.output, args.preflight)
    else:
        if args.preflight is None or args.freeze is None:
            parser.error("--preflight and --freeze required for formal")
        formal(args.output, args.preflight, args.freeze)

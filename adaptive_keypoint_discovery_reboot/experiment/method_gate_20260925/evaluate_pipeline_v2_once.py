"""Post-seal GT comparison and deterministic preregistered V1/V2 gate.

This program refuses to read phenotype GT until a complete sealed 40x2 V2
batch exists. It does not rerun V2, train, or access locked test files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "phenotype_pilot_protocol"))
import evaluate_frozen_phenotype_pilot as phen  # noqa: E402
from point_conditioned_graph import build_point_conditioned_graph  # noqa: E402
from run_pipeline_v2_once import (BASELINES, CODE_FILES, DATA, FROZEN, SOURCES,  # noqa: E402
                                  checked_inputs, frozen_structure, inputs, jsonable, per_method_points,
                                  sha)

GT = FROZEN["frozen_gt_hash_only"][0]
SELECTION = EXP / "phenotype_pilot_protocol/phenotype_pilot_selection.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_seal(folder: Path) -> tuple[dict, dict, dict, dict]:
    seal = json.loads((folder / "SEALED_SUMMARY.json").read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_40x2" or seal.get("image_records") != 80 or seal.get("images_per_method") != 40 or seal.get("gt_read") != 0 or seal.get("test_read") != 0:
        raise ValueError("Formal V2 output is not a complete pre-GT seal")
    if seal["frozen_input_hashes"] != checked_inputs():
        raise ValueError("Frozen input hash mismatch after formal seal")
    for name, expected in seal["output_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Formal V2 file changed: {name}")
    if {str(path.relative_to(EXP.parent)): sha(path) for path in CODE_FILES} != seal["code_sha256"]:
        raise ValueError("V2 code changed after formal seal")
    images = {(r["method"], r["dataset_id"]): r for r in read_jsonl(folder / "per_image.jsonl")}
    graphs = {(r["method"], r["dataset_id"]): r for r in read_jsonl(folder / "graphs.jsonl")}
    point_rows = defaultdict(list)
    for row in read_jsonl(folder / "point_associations.jsonl"):
        point_rows[(row["method"], row["dataset_id"])].append(row)
    if len(images) != 80 or len(graphs) != 80 or len(point_rows) != 80:
        raise ValueError("Formal V2 provenance incomplete")
    return seal, images, graphs, point_rows


def grouped_paths(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["method"], row["dataset_id"])].append(row)
    return by_key


def root_not_shoot_path_pixels(paths: list[dict], root: np.ndarray, shoot: np.ndarray) -> int:
    count = 0
    for path in paths:
        for x, y in path["full_base_to_tip_path"]:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < 518 and 0 <= xi < 518 and root[yi, xi] and not shoot[yi, xi]:
                count += 1
    return count


def original_v1_interface(rows: list[dict], frames: dict, old_paths: dict) -> dict:
    """V1 node/pixel safety baseline only; does not call V2 or touch GT."""
    result = {}
    for row in rows:
        dataset_id = str(row["dataset_id"])
        structure = frozen_structure(row)
        masks = structure["masks"]
        for method, frame in frames.items():
            points = per_method_points(frame, dataset_id, structure["mapping"])
            graph = build_point_conditioned_graph(structure["skeleton"], points, structure["D"], .025)
            root_nodes = sum(
                masks["seed_base_root"][int(n["projected_xy"][1]), int(n["projected_xy"][0])]
                and not masks["shoot"][int(n["projected_xy"][1]), int(n["projected_xy"][0])]
                for n in graph["nodes"]
            )
            result[(method, dataset_id)] = {
                "accepted_point_ids": [str(n["point_id"]) for n in graph["nodes"]],
                "projection_rejected_count": int(graph["diagnostics"]["rejected_point_count"]),
                "duplicate_merged_count": int(graph["diagnostics"]["merged_duplicate_count"]),
                "root_not_shoot_accepted_nodes": int(root_nodes),
                "root_not_shoot_path_pixels": root_not_shoot_path_pixels(old_paths[(method, dataset_id)], masks["seed_base_root"], masks["shoot"]),
                "D_model": structure["D"],
            }
    return result


def model_base_distance(plant: dict, paths: list[dict], width: int, height: int, D_model: float) -> dict:
    if not paths:
        return {"distance_px": None, "distance_over_D": None, "base_deviation_event": False}
    main = [p for p in plant["paths"] if p["reference_main_path"]]
    if len(main) != 1:
        raise ValueError("No GT common base")
    gt_base = np.asarray(main[0]["points_px"][0], dtype=np.float64)
    scale = min(518 / width, 518 / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    pad = np.asarray([(518 - new_w) // 2, (518 - new_h) // 2], dtype=np.float64)
    gt_model = gt_base * scale + pad
    distances = [float(np.linalg.norm(np.asarray(p["full_base_to_tip_path"][0], dtype=float) - gt_model)) for p in paths]
    d = min(distances)
    return {"distance_px": d, "distance_over_D": d / D_model, "base_deviation_event": d > .025 * D_model}


def confirmed_false_branch_count(row: dict, paths: list[dict]) -> int:
    matched = {p["pred_path_id"] for p in row["pairs"]}
    unmatched_branches = sum(p["path_kind"] == "lateral_branch" and p["path_id"] not in matched for p in paths)
    return max(0, unmatched_branches - row["gt_count_only_no_curve"])


def paired_median_increase(rows_v1: list[dict], rows_v2: list[dict], field: str) -> dict:
    old = {(r["dataset_id"], p["gt_path_sha256"]): p[field]
           for r in rows_v1 for p in r["pairs"] if p[field] is not None}
    new = {(r["dataset_id"], p["gt_path_sha256"]): p[field]
           for r in rows_v2 for p in r["pairs"] if p[field] is not None}
    changes = [new[k] - old[k] for k in sorted(old.keys() & new.keys())]
    return {"n": len(changes), "median_v2_minus_v1": float(np.median(changes)) if changes else None,
            "per_gt": [{"dataset_id": k[0], "gt_path_sha256": k[1], "delta": new[k] - old[k]} for k in sorted(old.keys() & new.keys())]}


def evaluate(folder: Path, output: Path) -> dict:
    # This gate executes before the first read of GT contents.
    seal, v2_images, v2_graphs, v2_points = load_seal(folder)
    rows, frames = inputs()
    source_frame = {str(r["dataset_id"]): str(r["source_frame_id"]) for r in rows}
    val_ids = set(source_frame)
    old_frames = {m: pd.read_csv(BASELINES[m] / "per_image.csv").set_index("dataset_id") for m in SOURCES}
    old_paths = defaultdict(list)
    for method in SOURCES:
        for path in read_jsonl(BASELINES[method] / "paths.jsonl"):
            old_paths[(method, path["dataset_id"])].append(path)
    new_paths = grouped_paths(read_jsonl(folder / "paths.jsonl"))
    v1_safety = original_v1_interface(rows, frames, old_paths)
    changed_points = []
    interface = {}
    safety = []
    benefit_plants = set()
    for method in SOURCES:
        projection_recovered = 0
        into_graph = 0
        into_phenotype_path = 0
        v1_lost = 0
        for dataset_id in sorted(val_ids):
            old = old_frames[method].loc[dataset_id]
            new = v2_images[(method, dataset_id)]
            old_point_count = int(old["input_point_count"])
            if old_point_count != new["input_point_count"]:
                raise ValueError("V1/V2 points differ")
            for point in v2_points[(method, dataset_id)]:
                old_accept = point["point_id"] in v1_safety[(method, dataset_id)]["accepted_point_ids"]
                new_accept = point["status"] == "accepted"
                if old_accept != new_accept:
                    changed_points.append({"method": method, "dataset_id": dataset_id,
                                           "source_frame_id": source_frame[dataset_id], **point,
                                           "v1_accepted_node": old_accept,
                                           "v1_reason": "accepted" if old_accept else
                                           "projection_too_far" if point["d_over_D"] is not None and point["d_over_D"] > .025 else
                                           "duplicate_projection_merged_or_empty_skeleton"})
                if point["d_over_D"] is not None and point["d_over_D"] > .025 and new_accept:
                    projection_recovered += 1
                    into_graph += 1
                    into_phenotype_path += int(point["in_phenotype_path"])
                    if int(old["decoded_path_count"]) == 0 and new["decoded_path_count"] > 0:
                        benefit_plants.add((dataset_id, source_frame[dataset_id], method, "zero_path_recovered"))
                    elif str(old["decode_failure"]).strip() not in ("", "nan") and not new["decode_failure"]:
                        benefit_plants.add((dataset_id, source_frame[dataset_id], method, "base_or_decode_failure_recovered"))
                if old_accept and not new_accept:
                    v1_lost += 1
            prior = v1_safety[(method, dataset_id)]
            safety.append({"method": method, "dataset_id": dataset_id,
                           "v1_accepted_nodes": int(old["accepted_node_count"]),
                           "v2_accepted_nodes": new["accepted_node_count"],
                           "v1_projection_rejections": prior["projection_rejected_count"],
                           "v2_weighted_rejections": new["projection_rejected_count"],
                           "v2_roi_rejections": new["roi_rejected_count"],
                           "v2_root_rejections": new["root_rejected_count"],
                           "v1_duplicate_merges": prior["duplicate_merged_count"],
                           "v2_duplicate_merges": new["duplicate_merged_count"],
                           "v1_paths": int(old["decoded_path_count"]), "v2_paths": new["decoded_path_count"],
                           "new_zero_path": int(old["decoded_path_count"]) > 0 and new["decoded_path_count"] == 0,
                           "v1_root_nodes": prior["root_not_shoot_accepted_nodes"],
                           "v2_root_nodes": new["root_not_shoot_accepted_nodes"],
                           "v1_root_path_pixels": prior["root_not_shoot_path_pixels"],
                           "v2_root_path_pixels": new["root_not_shoot_path_pixels"],
                           "new_root_contamination": (new["root_not_shoot_accepted_nodes"] > prior["root_not_shoot_accepted_nodes"] or
                                                      new["root_not_shoot_path_pixels"] > prior["root_not_shoot_path_pixels"])})
        interface[method] = {"v1_overdistance_to_v2_accepted": projection_recovered,
                             "new_graph_nodes_from_overdistance": into_graph,
                             "new_phenotype_path_nodes_from_overdistance": into_phenotype_path,
                             "v1_distance_eligible_lost": v1_lost}

    # The 80 V2 outputs are sealed above. Only now may the frozen 16 GT be parsed.
    gt = json.loads(GT.read_text(encoding="utf-8"))
    selection = {r["dataset_id"]: r for r in json.loads(SELECTION.read_text(encoding="utf-8"))}
    plants = gt["plants"]
    if len(plants) != 16 or len({p["dataset_id"] for p in plants}) != 16:
        raise ValueError("Frozen pilot GT membership changed")
    pilot_rows = {}
    pair_changes = {}
    warnings = []
    for method in SOURCES:
        v1_rows, v2_rows = [], []
        for plant in plants:
            dataset_id = plant["dataset_id"]
            select = selection[dataset_id]
            args = (int(select["standardized_width_px"]), int(select["standardized_height_px"]), float(select["mm_per_output_px"]))
            p1, p2 = old_paths[(method, dataset_id)], new_paths[(method, dataset_id)]
            r1 = phen._one_plant(f"{method}-V1", plant, p1, *args)
            r2 = phen._one_plant(f"{method}-V2", plant, p2, *args)
            D = v1_safety[(method, dataset_id)]["D_model"]
            base1 = model_base_distance(plant, p1, args[0], args[1], D)
            base2 = model_base_distance(plant, p2, args[0], args[1], D)
            r1["base_check"] = base1
            r2["base_check"] = base2
            r1["confirmed_false_branch_lower_bound"] = confirmed_false_branch_count(r1, p1)
            r2["confirmed_false_branch_lower_bound"] = confirmed_false_branch_count(r2, p2)
            if (base1["base_deviation_event"] and not base2["base_deviation_event"] and
                any(point["d_over_D"] is not None and point["d_over_D"] > .025 and point["status"] == "accepted"
                    for point in v2_points[(method, dataset_id)])):
                benefit_plants.add((dataset_id, source_frame[dataset_id], method, "gt_base_deviation_recovered"))
            v1_rows.append(r1)
            v2_rows.append(r2)
        for group in ("core", "diagnostic"):
            a = [r for r in v1_rows if r["pilot_group"] == group]
            b = [r for r in v2_rows if r["pilot_group"] == group]
            if len(a) != (12 if group == "core" else 4) or [r["dataset_id"] for r in a] != [r["dataset_id"] for r in b]:
                raise ValueError("Frozen Core/Diagnostic split changed")
            key = (method, group)
            pilot_rows[key] = {"V1": phen._aggregate(a), "V2": phen._aggregate(b),
                               "per_plant": [{"dataset_id": x["dataset_id"], "V1": x, "V2": y} for x, y in zip(a, b)]}
            pair_changes[key] = {
                "length_symmetric_relative_error_pct": paired_median_increase(a, b, "length_symmetric_relative_error_pct"),
                "divergence_angle_absolute_error_deg": paired_median_increase(a, b, "angle_absolute_error_deg"),
            }
            if any(y["base_check"]["base_deviation_event"] and not x["base_check"]["base_deviation_event"] for x, y in zip(a, b)):
                warnings.append(f"new_base_deviation:{method}/{group}")
            if any(y["confirmed_false_branch_lower_bound"] > x["confirmed_false_branch_lower_bound"] for x, y in zip(a, b)):
                warnings.append(f"new_confirmed_false_branch:{method}/{group}")
            if any(y["unmatched_prediction_count_possible_true_interval"][0] > x["unmatched_prediction_count_possible_true_interval"][0]
                   for x, y in zip(a, b)):
                warnings.append(f"new_per_plant_unambiguous_extra_path:{method}/{group}")
            if pilot_rows[key]["V2"]["geometry_matched"] < pilot_rows[key]["V1"]["geometry_matched"]:
                warnings.append(f"geometry_match_loss:{method}/{group}")
            if pilot_rows[key]["V2"]["possible_false_prediction_interval"][0] > pilot_rows[key]["V1"]["possible_false_prediction_interval"][0]:
                warnings.append(f"unambiguous_extra_path_increase:{method}/{group}")
    if any(r["new_zero_path"] for r in safety):
        warnings.append("new_zero_path")
    if any(r["new_root_contamination"] for r in safety):
        warnings.append("new_root_node_or_path_contamination")

    # §6: safety first, then conjunctive net-benefit gate; no post-hoc tuning.
    comparable_pairs = [pilot_rows[k] for k in pilot_rows]
    new_gt_match = any(
        any({p["gt_path_sha256"] for p in pair["V2"]["pairs"]} -
            {p["gt_path_sha256"] for p in pair["V1"]["pairs"]}
            for pair in item["per_plant"])
        for item in comparable_pairs
    )
    independent_recovery = len({(x[0], x[1]) for x in benefit_plants}) >= 2 and len({x[1] for x in benefit_plants}) >= 2
    no_group_loss = all(
        item["V2"]["geometry_matched"] >= item["V1"]["geometry_matched"] and
        (item["V2"]["verified_prediction_precision_interval"][0] or 0) >= (item["V1"]["verified_prediction_precision_interval"][0] or 0)
        for item in comparable_pairs
    )
    no_unmatched_upper_increase = all(
        item["V2"]["possible_false_prediction_interval"][1] <= item["V1"]["possible_false_prediction_interval"][1]
        for item in comparable_pairs
    )
    no_large_paired_error = all(
        (value["median_v2_minus_v1"] is None or value["median_v2_minus_v1"] < bound)
        for changes in pair_changes.values()
        for value, bound in ((changes["length_symmetric_relative_error_pct"], phen.LENGTH_MDC95_PCT),
                             (changes["divergence_angle_absolute_error_deg"], phen.ANGLE_MDC95_DEG))
    )
    if warnings:
        result = 3
        label = "degradation -> rollback/freeze V1"
    elif all((independent_recovery, new_gt_match, no_group_loss, no_unmatched_upper_increase, no_large_paired_error)):
        result = 1
        label = "V2 net benefit -> freeze V2"
    else:
        result = 2
        label = "no clear net benefit -> rollback/freeze V1"
    assessment = {
        "preregistration": "PIPELINE_V2_PREREGISTRATION.md §6",
        "result": result, "label": label,
        "frozen_pipeline": "V2" if result == 1 else "V1",
        "test_read": 0, "gt_modified": False, "models_modified": False,
        "formal_seal_sha256": sha(folder / "SEALED_SUMMARY.json"),
        "frozen_gt_sha256": sha(GT), "pilot_selection_sha256": sha(SELECTION),
        "interface": interface, "changed_points": changed_points,
        "safety_40_val": safety,
        "benefit_recovery_cases": [dict(dataset_id=a, source_frame_id=b, method=c, mechanism=d) for a,b,c,d in sorted(benefit_plants)],
        "pilot": {f"{m}/{g}": value for (m,g), value in pilot_rows.items()},
        "paired_error_changes": {f"{m}/{g}": value for (m,g), value in pair_changes.items()},
        "gate": {"safety_warnings": warnings,
                 "two_independent_frames_recovered": independent_recovery,
                 "new_correct_gt_match": new_gt_match,
                 "no_group_match_or_precision_loss": no_group_loss,
                 "no_unmatched_upper_increase": no_unmatched_upper_increase,
                 "no_large_paired_error": no_large_paired_error},
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(assessment), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return assessment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in evaluate(args.sealed_v2, args.output).items() if k in ("result", "label", "gate")}, ensure_ascii=False, indent=2))

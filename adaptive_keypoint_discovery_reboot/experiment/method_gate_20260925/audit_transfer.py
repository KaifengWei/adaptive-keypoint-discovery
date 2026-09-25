"""Read-only audit of frozen Teacher-direct -> Student-B point transfer on the 16 val pilot plants.

Outputs are private because they include deblinded GT-path association. No model is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
import evaluate_point_conditioned_graph_v1 as graph_eval  # noqa: E402
from point_conditioned_organ_paths import decode_candidate_organ_paths  # noqa: E402
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402

FROZEN_COMPARISON_SHA = "e585a628f740a3c5909a78850370bc7de0ee34dd2590420a6234545ba491ff74"
TEACHER_POINTS_SHA = "1ed017a6d6ba6f360bf216c44b01ffedaef9c2a14f18d46d75b1f640763b1814"
STUDENT_POINTS_SHA = "9a11bbcc7283333fca34c0e82a3931edbfa96cef8347ec952f450daecb1aa756"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_by_id(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                result[item["dataset_id"]].append(item)
    return result


def correspondence(left: np.ndarray, right: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    n, m = len(left), len(right)
    if not n or not m:
        return []
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    cost = np.zeros((n + m, n + m), dtype=np.float64)
    cost[:n, :m] = np.where(distances <= radius, distances, 3.0 * radius)
    cost[:n, m:] = radius
    cost[n:, :m] = radius
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, cols) if i < n and j < m and distances[i, j] <= radius]


def path_membership(node: dict, paths: list[dict], radius: float = 2.0) -> list[str]:
    xy = np.asarray(node["projected_xy"], dtype=np.float64)
    memberships = []
    for path in paths:
        raw = np.asarray(path["full_base_to_tip_path"], dtype=np.float64)
        is_endpoint = int(node["node_id"]) in {int(path["base_node_id"]), int(path["tip_node_id"])}
        if is_endpoint or (len(raw) and float(cKDTree(raw).query(xy)[0]) <= radius):
            memberships.append(path["path_id"])
    return memberships


def reconstruct(dataset_id: str, manifest_row: dict, point_frame: pd.DataFrame, dataset: Path) -> tuple[dict, list[dict], dict, dict]:
    result = graph_eval.evaluate_one(
        dataset_id, manifest_row, point_frame, dataset, 518, 0.025, "phenotype_roi_v1"
    )
    bbox = graph_eval.gp.bbox_from_mask(result["support"])
    diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    paths, decision = decode_candidate_organ_paths(
        result["graph"],
        result["masks_exact"]["shoot"],
        result["masks_exact"]["seed_base_root"],
        diag,
        phenotype_roi_mask=result["masks_exact"]["phenotype_roi"],
        basal_transition_mask=result["masks_exact"]["basal_transition"],
        branch_pruning_mode="local_learned_support",
    )
    _, mapping, _, _ = load_phenotype_input(dataset, manifest_row, 518)
    return result, paths, decision, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher_file = args.teacher_points.resolve()
    student_file = EXPERIMENT / "evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv"
    comparison_file = EXPERIMENT / "phenotype_pilot_protocol/runtime/method_comparison/frozen_pilot_20260924_v5_final/frozen_method_comparison.json"
    if (sha(teacher_file), sha(student_file), sha(comparison_file)) != (
        TEACHER_POINTS_SHA, STUDENT_POINTS_SHA, FROZEN_COMPARISON_SHA
    ):
        raise RuntimeError("Frozen source hash mismatch")
    dataset = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
    selection = pd.read_csv(EXPERIMENT / "phenotype_pilot_protocol/phenotype_pilot_selection.csv")
    if len(selection) != 16 or set(selection["pilot_group"]) != {"core", "diagnostic"}:
        raise RuntimeError("Pilot selection changed")
    ids = set(selection["dataset_id"])
    manifest = pd.read_csv(dataset / "manifests/val.csv").set_index("dataset_id", drop=False)
    if not ids <= set(manifest.index):
        raise RuntimeError("Only frozen V4 val IDs are permitted")
    teacher = pd.read_csv(teacher_file)
    student = pd.read_csv(student_file)
    teacher_paths = jsonl_by_id(EXPERIMENT / "phenotype_pilot_protocol/runtime/method_comparison/teacher_direct_route_b_local_decoder_val_20260924/paths.jsonl")
    student_paths = jsonl_by_id(EXPERIMENT / "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/paths.jsonl")
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    gt_by_method = {
        method: {plant["dataset_id"]: plant for plant in comparison["methods"][method]["per_plant"]}
        for method in ("Teacher-direct", "Student-B")
    }
    frozen_summary = {
        "Teacher-direct": pd.read_csv(EXPERIMENT / "phenotype_pilot_protocol/runtime/method_comparison/teacher_direct_route_b_local_decoder_val_20260924/per_image.csv").set_index("dataset_id"),
        "Student-B": pd.read_csv(EXPERIMENT / "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/per_image.csv").set_index("dataset_id"),
    }

    point_rows: list[dict] = []
    image_rows: list[dict] = []
    issues: list[dict] = []
    for selected in selection.to_dict("records"):
        dataset_id = str(selected["dataset_id"])
        record = manifest.loc[dataset_id].to_dict()
        method_data = {}
        for method, frame, paths_by_id in (
            ("Teacher-direct", teacher, teacher_paths),
            ("Student-B", student, student_paths),
        ):
            current = frame[frame["dataset_id"] == dataset_id].sort_values("point_id").reset_index(drop=True)
            graph_result, rebuilt_paths, decision, mapping = reconstruct(dataset_id, record, current, dataset)
            frozen = frozen_summary[method].loc[dataset_id]
            saved_paths = paths_by_id.get(dataset_id, [])
            count_ok = (len(graph_result["graph"]["nodes"]) == int(frozen["accepted_node_count"]) and len(rebuilt_paths) == int(frozen["decoded_path_count"]) == len(saved_paths))
            if not count_ok:
                issues.append({"dataset_id": dataset_id, "method": method, "reason": "graph_or_path_count_reconstruction_mismatch", "rebuilt_nodes": len(graph_result["graph"]["nodes"]), "frozen_nodes": int(frozen["accepted_node_count"]), "rebuilt_paths": len(rebuilt_paths), "frozen_paths": len(saved_paths)})
            accepted = {str(node["point_id"]): node for node in graph_result["graph"]["nodes"]}
            rejection = {str(item.get("point_id")): item for item in graph_result["graph"]["rejected_points"]}
            by_prediction = {pair["pred_path_id"]: pair["gt_path_sha256"] for pair in gt_by_method[method][dataset_id]["pairs"]}
            points = []
            for index, item in current.iterrows():
                point_id = str(item["point_id"])
                node = accepted.get(point_id)
                memberships = path_membership(node, saved_paths) if node is not None else []
                gt_paths = sorted({by_prediction[path_id] for path_id in memberships if path_id in by_prediction})
                unmatched_paths = sorted(path_id for path_id in memberships if path_id not in by_prediction)
                points.append({
                    "index": int(index), "point_id": point_id,
                    "x_source": float(item["x_source"]), "y_source": float(item["y_source"]),
                    "x_model": float(item["x_source"]) * float(mapping["scale"]) + float(mapping["pad_x"]),
                    "y_model": float(item["y_source"]) * float(mapping["scale"]) + float(mapping["pad_y"]),
                    "confidence": float(item["confidence"]), "graph_accepted": node is not None,
                    "node_id": None if node is None else int(node["node_id"]),
                    "rejection_reason": None if point_id not in rejection else rejection[point_id].get("rejection_reason"),
                    "projection_distance_px": None if point_id not in rejection else rejection[point_id].get("projection_distance_px"),
                    "path_ids": memberships, "gt_path_hashes": gt_paths, "unmatched_path_ids": unmatched_paths,
                    "base_or_tip": bool(node is not None and any(int(node["node_id"]) in {int(p["base_node_id"]), int(p["tip_node_id"])} for p in saved_paths)),
                    "labels": [],
                })
            method_data[method] = {"points": points, "graph": graph_result["graph"], "decision": decision, "paths": saved_paths, "frozen": frozen, "gt": gt_by_method[method][dataset_id]}

        left = method_data["Teacher-direct"]["points"]
        right = method_data["Student-B"]["points"]
        diag = max(1.0, math.hypot(*np.subtract(graph_eval.gp.bbox_from_mask(graph_result["support"])[2:], graph_eval.gp.bbox_from_mask(graph_result["support"])[:2])))
        left_xy = np.asarray([[p["x_model"], p["y_model"]] for p in left], dtype=float).reshape(-1, 2)
        right_xy = np.asarray([[p["x_model"], p["y_model"]] for p in right], dtype=float).reshape(-1, 2)
        matches = correspondence(left_xy, right_xy, 0.025 * diag)
        sensitivity = correspondence(left_xy, right_xy, 0.05 * diag)
        left_match = {i: (j, d) for i, j, d in matches}
        right_match = {j: (i, d) for i, j, d in matches}
        teacher_gt = {pair["gt_path_sha256"] for pair in method_data["Teacher-direct"]["gt"]["pairs"]}
        student_gt = {pair["gt_path_sha256"] for pair in method_data["Student-B"]["gt"]["pairs"]}
        count_only_ambiguity = bool(method_data["Teacher-direct"]["gt"]["gt_count_only_no_curve"])

        for i, point in enumerate(left):
            if i in left_match:
                partner = right[left_match[i][0]]
                if set(point["gt_path_hashes"]) & set(partner["gt_path_hashes"]):
                    point["labels"].append("true_point_preservation")
            else:
                lost_gt = set(point["gt_path_hashes"]) & (teacher_gt - student_gt)
                if lost_gt:
                    point["labels"].append("rare_structure_loss")
                if not point["gt_path_hashes"] and not point["path_ids"] and not count_only_ambiguity:
                    point["labels"].append("noise_suppression")
                if point["base_or_tip"] and lost_gt:
                    point["labels"].append("topology_consequence")
        for j, point in enumerate(right):
            if j in right_match:
                partner = left[right_match[j][0]]
                if set(point["gt_path_hashes"]) & set(partner["gt_path_hashes"]):
                    point["labels"].append("true_point_preservation")
            else:
                gained_gt = set(point["gt_path_hashes"]) & (student_gt - teacher_gt)
                if gained_gt:
                    point["labels"].append("new_useful_point")
                if point["unmatched_path_ids"] and not point["gt_path_hashes"] and not count_only_ambiguity:
                    point["labels"].append("new_false_point")
                if point["base_or_tip"] and gained_gt:
                    point["labels"].append("topology_consequence")
        for method, points in (("Teacher-direct", left), ("Student-B", right)):
            for p in points:
                index = p["index"]
                match = left_match.get(index) if method == "Teacher-direct" else right_match.get(index)
                point_rows.append({"dataset_id": dataset_id, "pilot_group": selected["pilot_group"], "method": method, **p, "matched_other_point_id": None if match is None else (right[match[0]]["point_id"] if method == "Teacher-direct" else left[match[0]]["point_id"]), "match_distance_model_px": None if match is None else match[1]})
        image_rows.append({
            "dataset_id": dataset_id, "pilot_group": selected["pilot_group"], "bbox_diagonal_model_px": diag,
            "teacher_point_count": len(left), "student_point_count": len(right),
            "matched_points_0_025D": len(matches), "matched_points_0_05D": len(sensitivity),
            "teacher_removed": len(left) - len(matches), "student_only": len(right) - len(matches),
            "teacher_graph_nodes": sum(p["graph_accepted"] for p in left),
            "student_graph_nodes": sum(p["graph_accepted"] for p in right),
            "teacher_path_participating_points": sum(bool(p["path_ids"]) for p in left),
            "student_path_participating_points": sum(bool(p["path_ids"]) for p in right),
            "teacher_paths": len(method_data["Teacher-direct"]["paths"]),
            "student_paths": len(method_data["Student-B"]["paths"]),
            "teacher_gt_matched": len(teacher_gt), "student_gt_matched": len(student_gt),
            "teacher_decode_failure": method_data["Teacher-direct"]["decision"].get("failure", ""),
            "student_decode_failure": method_data["Student-B"]["decision"].get("failure", ""),
        })
        print(f"[{len(image_rows)}/16] {dataset_id}: T {len(left)}/{len(teacher_gt)} GT, S {len(right)}/{len(student_gt)} GT, matches {len(matches)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(image_rows).to_csv(args.output / "per_plant_private.csv", index=False, encoding="utf-8-sig")
    with (args.output / "per_point_private.jsonl").open("w", encoding="utf-8") as stream:
        for row in point_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {"scope": "16 frozen V4 val pilot plants only", "plants": len(image_rows), "point_rows": len(point_rows), "reconstruction_issues": issues, "source_sha256": {"teacher_points": TEACHER_POINTS_SHA, "student_points": STUDENT_POINTS_SHA, "phenotype_comparison": FROZEN_COMPARISON_SHA}, "test_read": False, "models_run": False, "models_modified": False}
    (args.output / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

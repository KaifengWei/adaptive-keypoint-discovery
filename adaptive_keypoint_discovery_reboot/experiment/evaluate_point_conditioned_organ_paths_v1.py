#!/usr/bin/env python
"""Decode and audit candidate organ paths on V4 val learned-point graphs."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_point_conditioned_graph_v1 as graph_eval  # noqa: E402
import g1_prime_phenotype_bridge as bridge  # noqa: E402
from point_conditioned_organ_paths import decode_candidate_organ_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "data_stage_clean_v4_fullplant_candidate")
    parser.add_argument(
        "--evaluation", type=Path, default=HERE / "evaluation_outputs" / "core_dinov2_v4_fullplant_val"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "evaluation_outputs" / "point_conditioned_organ_paths_v1_val"
    )
    parser.add_argument("--projection-ratio", type=float, default=0.025)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def save_overlay(path: Path, result: dict[str, Any], paths: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
    image = result["image"]
    graph = result["graph"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=135)
    axes[0].imshow(image)
    for edge in graph["edges"]:
        xy = np.asarray(edge["path_xy"], dtype=np.float64)
        axes[0].plot(xy[:, 0], xy[:, 1], color="#60a5fa", linewidth=1.7)
    for node in graph["nodes"]:
        x, y = node["projected_xy"]
        axes[0].scatter(x, y, s=34, c="#22c55e", edgecolors="black", linewidths=0.5)
        axes[0].text(x + 3, y - 3, str(node["node_id"]), fontsize=7)
    if diagnostics.get("base_xy"):
        x, y = diagnostics["base_xy"]
        axes[0].scatter(x, y, marker="*", s=145, c="#ef4444", edgecolors="black")
    axes[0].set_title(f"learned-node graph | base={diagnostics.get('base_node_id', 'fail')}")

    axes[1].imshow(image)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(paths))))
    for color, item in zip(colors, paths):
        curve = np.asarray(item["spline_curve"], dtype=np.float64)
        raw = np.asarray(item["branch_path_points"], dtype=np.float64)
        support = np.asarray(item["support_points"], dtype=np.float64)
        if len(raw):
            axes[1].plot(raw[:, 0], raw[:, 1], color=color, linewidth=1.0, alpha=0.45)
        if len(curve):
            axes[1].plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.3)
        if len(support):
            axes[1].scatter(support[:, 0], support[:, 1], s=24, color=color, edgecolors="black", linewidths=0.4)
        attach = item["attachment_xy"]
        tip = item["tip_xy"]
        axes[1].scatter(attach[0], attach[1], marker="s", s=45, color=color, edgecolors="black")
        axes[1].scatter(tip[0], tip[1], marker="^", s=52, color=color, edgecolors="black")
    axes[1].set_title(
        f"decoded paths={len(paths)} | branch union/graph={diagnostics.get('decoded_to_edge_union_ratio', 0.0):.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(result["dataset_id"])
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def make_review_html(output: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for index, row in enumerate(rows):
        dataset_id = html.escape(str(row["dataset_id"]))
        metrics = html.escape(
            f"points {row['accepted_node_count']}/{row['input_point_count']} | "
            f"paths {row['decoded_path_count']} | coverage {row['skeleton_coverage_ratio']:.3f} | "
            f"base distance {row['base_interface_distance_bbox_diag']:.3f}"
        )
        cards.append(
            f'''<section class="card" data-index="{index}" data-id="{dataset_id}">
<h2>{dataset_id}</h2><p>{metrics}</p>
<img src="overlays/{dataset_id}.png" alt="{dataset_id}">
<div class="fields">
<label>路径语义 <select data-field="manual_path_semantics"><option>pending</option><option>pass</option><option>fail</option><option>uncertain</option></select></label>
<label>漏叶 <select data-field="manual_missing_organ"><option></option><option>no</option><option>yes</option><option>uncertain</option></select></label>
<label>错连 <select data-field="manual_wrong_connection"><option></option><option>no</option><option>yes</option><option>uncertain</option></select></label>
<label>基部 <select data-field="manual_base_selection"><option></option><option>correct</option><option>wrong</option><option>uncertain</option></select></label>
<label class="note">备注 <input data-field="manual_note"></label>
</div></section>'''
        )
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>关键点条件器官路径人工复核</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f3f4f6;color:#111827}}
header{{position:sticky;top:0;background:white;padding:10px 18px;box-shadow:0 2px 8px #0002;z-index:2}}
button{{margin-right:8px;padding:7px 12px}}main{{max-width:1500px;margin:auto;padding:16px}}
.card{{display:none;background:white;border-radius:10px;padding:14px}}.card.active{{display:block}}
img{{width:100%;max-height:72vh;object-fit:contain;background:white}}.fields{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}}
label{{display:flex;gap:8px;align-items:center}}select,input{{flex:1;padding:6px}}.note{{grid-column:1/-1}}
</style></head><body><header><button id="prev">上一张</button><button id="next">下一张</button><button id="export">导出CSV</button><span id="position"></span></header>
<main>{''.join(cards)}</main><script>
const cards=[...document.querySelectorAll('.card')], key='pc-organ-review-v1'; let current=0;
const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
function show(i){{current=(i+cards.length)%cards.length;cards.forEach((c,j)=>c.classList.toggle('active',j===current));document.querySelector('#position').textContent=` ${{current+1}} / ${{cards.length}} · ${{cards[current].dataset.id}}`;}}
cards.forEach(card=>card.querySelectorAll('[data-field]').forEach(input=>{{const id=card.dataset.id,field=input.dataset.field;if(saved[id]?.[field]!==undefined)input.value=saved[id][field];input.onchange=()=>{{saved[id]??={{}};saved[id][field]=input.value;localStorage.setItem(key,JSON.stringify(saved));}};}}));
document.querySelector('#prev').onclick=()=>show(current-1);document.querySelector('#next').onclick=()=>show(current+1);
document.onkeydown=e=>{{if(e.key==='ArrowLeft')show(current-1);if(e.key==='ArrowRight')show(current+1);}};
document.querySelector('#export').onclick=()=>{{const fields=['dataset_id','manual_path_semantics','manual_missing_organ','manual_wrong_connection','manual_base_selection','manual_note'];const lines=[fields.join(',')];cards.forEach(card=>{{const id=card.dataset.id,row=saved[id]||{{}};const vals=[id,...fields.slice(1).map(f=>row[f]||'')].map(v=>'"'+String(v).replaceAll('"','""')+'"');lines.push(vals.join(','));}});const blob=new Blob(['\ufeff'+lines.join('\n')],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_path_review_completed.csv';a.click();}};show(0);
</script></body></html>'''
    (output / "人工路径复核.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    per_image = pd.read_csv(args.evaluation / "per_image.csv")
    if set(per_image["split"].astype(str)) != {"val"} and not args.allow_test:
        raise RuntimeError("Refusing non-val saved predictions without --allow-test")
    points = pd.read_csv(args.evaluation / "points.csv")
    manifest = pd.read_csv(args.dataset / "manifests" / "val.csv", low_memory=False).sort_values("dataset_id")
    manifest = manifest[manifest["dataset_id"].isin(per_image["dataset_id"])]
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    if len(manifest) != (args.limit if 0 < args.limit < 40 else 40):
        raise RuntimeError(f"Unexpected val row count: {len(manifest)}")
    args.output.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    phenotype_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    for number, row in enumerate(manifest.to_dict("records"), start=1):
        dataset_id = str(row["dataset_id"])
        graph_result = graph_eval.evaluate_one(
            dataset_id,
            row,
            points[points["dataset_id"] == dataset_id],
            args.dataset,
            args.image_size,
            args.projection_ratio,
        )
        bbox = graph_eval.gp.bbox_from_mask(graph_result["support"])
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        paths, diagnostics = decode_candidate_organ_paths(
            graph_result["graph"],
            graph_result["masks_exact"]["shoot"],
            graph_result["masks_exact"]["seed_base_root"],
            bbox_diag,
        )
        path_errors = [path["metrics"]["spline_to_skeleton_median_error_px"] for path in paths]
        image_rows.append(
            {
                "dataset_id": dataset_id,
                "split": "val",
                "input_point_count": graph_result["metrics"]["input_point_count"],
                "accepted_node_count": graph_result["metrics"]["accepted_node_count"],
                "skeleton_coverage_ratio": graph_result["metrics"]["skeleton_coverage_ratio"],
                "base_node_id": diagnostics.get("base_node_id", -1),
                "base_interface_distance_bbox_diag": diagnostics.get("base_interface_distance_bbox_diag", float("nan")),
                "shoot_terminal_learned_node_count": diagnostics.get("terminal_learned_node_count", 0),
                "short_terminal_rejected_count": diagnostics.get("short_terminal_rejected_count", 0),
                "minimum_lateral_branch_length_px": diagnostics.get("minimum_lateral_branch_length_px", float("nan")),
                "decoded_path_count": len(paths),
                "main_axis_count": sum(path["path_kind"] == "main_axis" for path in paths),
                "lateral_branch_count": sum(path["path_kind"] == "lateral_branch" for path in paths),
                "decoded_to_edge_union_ratio": diagnostics.get("decoded_to_edge_union_ratio", 0.0),
                "median_spline_to_skeleton_error_px": float(np.median(path_errors)) if path_errors else float("nan"),
                "decode_failure": diagnostics.get("failure", ""),
                "phenotype_accuracy_status": "pending_manual_measurement_reference",
            }
        )
        for path in paths:
            phenotype_rows.append(
                {
                    "dataset_id": dataset_id,
                    "split": "val",
                    "path_id": path["path_id"],
                    "path_kind": path["path_kind"],
                    "base_node_id": path["base_node_id"],
                    "tip_node_id": path["tip_node_id"],
                    "adaptive_support_count": len(path["support_points"]),
                    **path["metrics"],
                    "physical_unit_status": "not_available_no_scale_reference",
                }
            )
            jsonl_rows.append({"dataset_id": dataset_id, **path})
        overlay_path = args.output / "overlays" / f"{dataset_id}.png"
        save_overlay(overlay_path, graph_result, paths, diagnostics)
        overlay_paths.append(overlay_path)
        print(f"[{number}/{len(manifest)}] {dataset_id} paths={len(paths)} failure={diagnostics.get('failure', '')}", flush=True)

    image_frame = pd.DataFrame(image_rows)
    phenotype_frame = pd.DataFrame(phenotype_rows)
    image_frame.to_csv(args.output / "per_image.csv", index=False, encoding="utf-8-sig")
    phenotype_frame.to_csv(args.output / "candidate_phenotypes.csv", index=False, encoding="utf-8-sig")
    bridge.write_jsonl(args.output / "paths.jsonl", jsonl_rows)
    graph_eval.make_contact_sheets(overlay_paths, args.output / "contact_sheets")
    review = image_frame.copy()
    review["auto_review_reasons"] = review.apply(
        lambda row: ";".join(
            reason
            for condition, reason in [
                (bool(row["decode_failure"]), "decode_failure"),
                (row["skeleton_coverage_ratio"] < 0.90, "graph_coverage_below_0.90"),
                (row["base_interface_distance_bbox_diag"] >= 0.03, "base_far_from_mask_interface"),
                (row["decoded_to_edge_union_ratio"] < 0.70, "low_shoot_path_share_of_graph"),
                (row["median_spline_to_skeleton_error_px"] > 2.0, "spline_error_above_2px"),
            ]
            if condition
        ),
        axis=1,
    )
    review["auto_review_priority"] = (review["auto_review_reasons"] != "").astype(int)
    review = review.sort_values(
        ["auto_review_priority", "skeleton_coverage_ratio", "base_interface_distance_bbox_diag"],
        ascending=[False, True, False],
    )
    review["manual_path_semantics"] = "pending"
    review["manual_missing_organ"] = ""
    review["manual_wrong_connection"] = ""
    review["manual_base_selection"] = ""
    review["manual_note"] = ""
    review.to_csv(args.output / "manual_path_review_pending.csv", index=False, encoding="utf-8-sig")
    make_review_html(args.output, review.to_dict("records"))
    phenotype_reference = phenotype_frame[
        [
            "dataset_id",
            "path_id",
            "path_kind",
            "spline_length_px",
            "sinuosity",
            "total_turning_angle_deg",
            "mean_abs_curvature_per_px",
            "divergence_angle_deg",
        ]
    ].copy()
    phenotype_reference = phenotype_reference.rename(
        columns={column: f"auto_{column}" for column in phenotype_reference.columns if column not in {"dataset_id", "path_id", "path_kind"}}
    )
    phenotype_reference["manual_path_valid"] = "pending"
    phenotype_reference["manual_length_px"] = ""
    phenotype_reference["manual_total_turning_angle_deg"] = ""
    phenotype_reference["manual_divergence_angle_deg"] = ""
    phenotype_reference["manual_note"] = ""
    phenotype_reference.to_csv(
        args.output / "manual_phenotype_reference_pending.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "images": len(image_frame),
        "split": "val",
        "test_images_read": 0,
        "projection_ratio": args.projection_ratio,
        "decode_success_rate": float((image_frame["decode_failure"].fillna("") == "").mean()),
        "median_decoded_path_count": float(image_frame["decoded_path_count"].median()),
        "median_lateral_branch_count": float(image_frame["lateral_branch_count"].median()),
        "median_decoded_to_edge_union_ratio": float(image_frame["decoded_to_edge_union_ratio"].median()),
        "median_base_interface_distance_bbox_diag": float(image_frame["base_interface_distance_bbox_diag"].median()),
        "median_spline_to_skeleton_error_px": float(image_frame["median_spline_to_skeleton_error_px"].median()),
        "decoded_path_count_distribution": {
            str(int(key)): int(value) for key, value in image_frame["decoded_path_count"].value_counts().sort_index().items()
        },
        "total_short_terminal_rejections": int(image_frame["short_terminal_rejected_count"].sum()),
        "priority_manual_review_images": int(review["auto_review_priority"].sum()),
        "manual_keypoint_labels_used": False,
        "manual_path_semantics_status": "pending",
        "phenotype_accuracy_status": "pending_manual_measurement_reference",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

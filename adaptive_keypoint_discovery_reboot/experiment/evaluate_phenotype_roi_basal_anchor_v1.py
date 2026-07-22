#!/usr/bin/env python
"""Build a val-only visual audit for phenotype ROI and shoot-side basal anchors."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phenotype_roi_basal_anchor import derive_phenotype_roi, select_learned_basal_anchor


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data_stage_clean_v4_fullplant_candidate"
DEFAULT_POINTS = HERE / "evaluation_outputs" / "core_dinov2_v4_fullplant_val" / "points.csv"
DEFAULT_GRAPH_NODES = HERE / "evaluation_outputs" / "point_conditioned_graph_v1_val" / "nodes.csv"
DEFAULT_PATH_METRICS = HERE / "evaluation_outputs" / "point_conditioned_organ_paths_v1_val" / "per_image.csv"
DEFAULT_MANUAL_AUDIT = HERE / "manual_path_review_completed.csv"
DEFAULT_OUTPUT = HERE / "evaluation_outputs" / "phenotype_roi_basal_anchor_v1_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--graph-nodes", type=Path, default=DEFAULT_GRAPH_NODES)
    parser.add_argument("--path-metrics", type=Path, default=DEFAULT_PATH_METRICS)
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_MANUAL_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot decode mask: {path}")
    return mask > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"Cannot encode mask: {path}")
    encoded.tofile(path)


def dataset_path(dataset: Path, value: Any) -> Path:
    return dataset / Path(str(value).replace("\\", "/"))


def point_records(frame: pd.DataFrame, width: int, height: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        records.append(
            {
                "point_id": str(row["point_id"]),
                "x": float(row["x_normalized"]) * max(1, width - 1),
                "y": float(row["y_normalized"]) * max(1, height - 1),
                "score": float(row["confidence"]),
            }
        )
    return records


def old_base_point_id(dataset_id: str, metrics: pd.DataFrame, nodes: pd.DataFrame) -> str:
    metric = metrics[metrics["dataset_id"].astype(str) == dataset_id]
    if metric.empty or pd.isna(metric.iloc[0].get("base_node_id")):
        return ""
    node_id = int(metric.iloc[0]["base_node_id"])
    match = nodes[
        (nodes["dataset_id"].astype(str) == dataset_id)
        & (pd.to_numeric(nodes["node_id"], errors="coerce") == node_id)
    ]
    return "" if match.empty else str(match.iloc[0]["point_id"])


def _rgba(mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask.astype(np.float32) * alpha
    return overlay


def save_overlay(
    path: Path,
    dataset_id: str,
    image: np.ndarray,
    root_base: np.ndarray,
    result: dict[str, Any],
    audited_points: list[dict[str, Any]],
    proposed_anchor: dict[str, Any] | None,
    old_base_id: str,
) -> None:
    height, width = image.shape[:2]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150)
    for axis in axes:
        axis.imshow(image)
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.axis("off")
    axes[0].set_title(f"{dataset_id} | complete standardized image", fontsize=12)

    axes[1].imshow(_rgba(root_base, (0.88, 0.25, 0.20), 0.22))
    axes[1].imshow(_rgba(result["phenotype_roi"], (0.10, 0.72, 0.35), 0.30))
    axes[1].imshow(_rgba(result["basal_transition"], (1.00, 0.72, 0.05), 0.72))
    axes[1].imshow(_rgba(result["seed_mask"], (0.72, 0.12, 0.80), 0.45))
    bx, by = result["boundary_xy"]
    direction = result["shoot_direction_xy"]
    arrow_length = 0.08 * result["bbox_diag"]
    axes[1].scatter([bx], [by], s=75, marker="o", facecolors="white", edgecolors="black", linewidths=1.4)
    axes[1].arrow(
        bx,
        by,
        direction[0] * arrow_length,
        direction[1] * arrow_length,
        width=max(0.8, 0.002 * result["bbox_diag"]),
        color="#111827",
        length_includes_head=True,
    )
    axes[1].set_title("ROI: green | transition: yellow | seed: purple | root/base: red", fontsize=11)

    for point in audited_points:
        if point["inside_phenotype_roi"]:
            axes[2].scatter(
                [point["x"]], [point["y"]], s=62, marker="o", c="#00c2a8", edgecolors="black", linewidths=0.7
            )
        else:
            axes[2].scatter([point["x"]], [point["y"]], s=72, marker="x", c="#e11d48", linewidths=2.0)
        axes[2].text(point["x"] + 3, point["y"] - 3, point["point_id"], fontsize=7, color="#111827")
        if point["point_id"] == old_base_id:
            axes[2].scatter(
                [point["x"]], [point["y"]], s=150, marker="s", facecolors="none", edgecolors="#111827", linewidths=2.2
            )
    if proposed_anchor is not None:
        axes[2].scatter(
            [proposed_anchor["x"]], [proposed_anchor["y"]], s=240, marker="*", c="#facc15", edgecolors="black", linewidths=1.1
        )
    else:
        axes[2].text(0.02, 0.94, "NO ELIGIBLE LEARNED BASAL ANCHOR", transform=axes[2].transAxes, color="#b91c1c", fontsize=10)
    axes[2].set_title("kept learned points: cyan | excluded: red x | old base: square | proposal: star", fontsize=10)

    figure.tight_layout(pad=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_review_html(output: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for index, row in enumerate(rows):
        dataset_id = html.escape(str(row["dataset_id"]))
        card_class = "card active" if index == 0 else "card"
        manual_note = html.escape(str(row.get("old_manual_note", "")))
        metrics = html.escape(
            f"shoot retention {row['shoot_retention_ratio']:.3f} | root/base overlap {row['root_base_overlap_ratio']:.3f} | "
            f"learned points kept {row['points_inside_roi']}/{row['input_point_count']} | proposed anchor {row['proposed_anchor_point_id'] or 'missing'}"
        )
        old_audit = html.escape(
            f"previous manual audit: path={row.get('old_manual_path_semantics', '') or 'blank'}, "
            f"base={row.get('old_manual_base_selection', '') or 'blank'}"
        )
        cards.append(
            f'''<section class="{card_class}" data-index="{index}" data-id="{dataset_id}">
<h2>{dataset_id}</h2><p>{metrics}</p><p class="old-audit">{old_audit} · {manual_note}</p>
<p class="hint">绿区是拟用于地上部关键点与路径的有效域；黄区是自动定位的shoot侧过渡带。红叉点将被排除，黄色星号只能从现有学习点中选择。</p>
<a href="overlays/{dataset_id}.png" target="_blank" rel="noopener"><img src="overlays/{dataset_id}.png" alt="{dataset_id} phenotype ROI audit"></a>
<div class="fields">
<label>地上部是否完整 <select data-field="manual_roi_completeness"><option>pending</option><option>pass</option><option>fail</option><option>uncertain</option></select></label>
<label>颖果根须是否排除 <select data-field="manual_seed_root_exclusion"><option>pending</option><option>pass</option><option>fail</option><option>uncertain</option></select></label>
<label>黄色过渡区 <select data-field="manual_basal_transition"><option>pending</option><option>correct</option><option>wrong</option><option>uncertain</option></select></label>
<label class="note">备注 <input data-field="manual_note"></label>
</div></section>'''
        )
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>地上部表型有效域与基部过渡区复核</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f3f4f6;color:#111827}}
header{{position:sticky;top:0;background:white;padding:10px 18px;box-shadow:0 2px 8px #0002;z-index:2}}
button{{margin-right:8px;padding:7px 12px}}main{{max-width:1700px;margin:auto;padding:16px}}
.card{{display:none;background:white;border-radius:10px;padding:14px}}.card.active{{display:block}}
.hint{{padding:9px 12px;border-left:4px solid #16a34a;background:#f0fdf4}}.old-audit{{color:#475569}}
img{{width:100%;max-height:70vh;object-fit:contain;background:white;border:1px solid #d1d5db}}
.fields{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}}
label{{display:flex;gap:8px;align-items:center}}select,input{{flex:1;padding:6px}}.note{{grid-column:1/-1}}
@media(max-width:900px){{.fields{{grid-template-columns:1fr}}}}
</style></head><body><header><button id="prev">上一张</button><button id="next">下一张</button><button id="export">导出CSV</button><span id="position"></span></header>
<main>{''.join(cards)}</main><script>
var cards=Array.prototype.slice.call(document.querySelectorAll('.card'));
var key='phenotype-roi-review-v1';var current=0;var saved={{}};
try{{saved=JSON.parse(window.localStorage.getItem(key)||'{{}}')||{{}};}}catch(error){{saved={{}};}}
function persist(){{try{{window.localStorage.setItem(key,JSON.stringify(saved));}}catch(error){{}}}}
function show(i){{if(!cards.length)return;current=(i+cards.length)%cards.length;cards.forEach(function(card,index){{card.classList.toggle('active',index===current);}});document.querySelector('#position').textContent=' '+(current+1)+' / '+cards.length+' · '+cards[current].getAttribute('data-id');}}
cards.forEach(function(card){{Array.prototype.slice.call(card.querySelectorAll('[data-field]')).forEach(function(input){{var id=card.getAttribute('data-id');var field=input.getAttribute('data-field');if(saved[id]&&saved[id][field]!==undefined)input.value=saved[id][field];input.onchange=function(){{if(!saved[id])saved[id]={{}};saved[id][field]=input.value;persist();}};}});}});
document.querySelector('#prev').onclick=function(){{show(current-1);}};document.querySelector('#next').onclick=function(){{show(current+1);}};
document.onkeydown=function(event){{if(event.key==='ArrowLeft')show(current-1);if(event.key==='ArrowRight')show(current+1);}};
document.querySelector('#export').onclick=function(){{var fields=['dataset_id','manual_roi_completeness','manual_seed_root_exclusion','manual_basal_transition','manual_note'];var lines=[fields.join(',')];cards.forEach(function(card){{var id=card.getAttribute('data-id');var row=saved[id]||{{}};var values=[id];fields.slice(1).forEach(function(field){{values.push(row[field]||'');}});values=values.map(function(value){{return '"'+String(value).replace(/"/g,'""')+'"';}});lines.push(values.join(','));}});var blob=new Blob(['\ufeff'+lines.join('\\r\\n')],{{type:'text/csv'}});var link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='manual_phenotype_roi_review_completed.csv';link.click();}};show(0);
</script></body></html>'''
    (output / "地上部有效域复核.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    manifest = pd.read_csv(dataset / "manifests" / "val.csv", low_memory=False).sort_values("dataset_id")
    if len(manifest) != 40 or set(manifest["split"].astype(str)) != {"val"}:
        raise RuntimeError(f"Expected exactly 40 val rows, got {len(manifest)} rows and {set(manifest['split'])}")
    if not manifest["dataset_id"].astype(str).str.startswith("v4_val_").all():
        raise RuntimeError("Refusing non-val dataset IDs")

    points = pd.read_csv(args.points.resolve(), low_memory=False)
    nodes = pd.read_csv(args.graph_nodes.resolve(), low_memory=False)
    path_metrics = pd.read_csv(args.path_metrics.resolve(), low_memory=False)
    manual = pd.read_csv(args.manual_audit.resolve(), low_memory=False).fillna("")
    manual_by_id = {str(row["dataset_id"]): row for row in manual.to_dict("records")}
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    point_audit_rows: list[dict[str, Any]] = []
    for item in manifest.to_dict("records"):
        dataset_id = str(item["dataset_id"])
        image = read_rgb(dataset_path(dataset, item["relative_path"]))
        shoot = read_mask(dataset_path(dataset, item["shoot_mask_relative_path"]))
        root_base = read_mask(dataset_path(dataset, item["seed_base_root_mask_relative_path"]))
        full = read_mask(dataset_path(dataset, item["full_plant_mask_relative_path"]))
        if image.shape[:2] != shoot.shape or shoot.shape != root_base.shape or root_base.shape != full.shape:
            raise RuntimeError(f"Shape mismatch for {dataset_id}")

        result = derive_phenotype_roi(image, shoot, root_base, full)
        image_points = points[points["dataset_id"].astype(str) == dataset_id]
        model_points = point_records(image_points, image.shape[1], image.shape[0])
        proposed_anchor, audited_points = select_learned_basal_anchor(
            model_points,
            result["phenotype_roi"],
            result["transition_center_xy"],
            result["bbox_diag"],
        )
        old_base_id = old_base_point_id(dataset_id, path_metrics, nodes)
        old_base_audit = next((point for point in audited_points if point["point_id"] == old_base_id), None)
        manual_row = manual_by_id.get(dataset_id, {})
        per_image = {
            "dataset_id": dataset_id,
            "split": "val",
            "input_point_count": len(audited_points),
            "points_inside_roi": sum(bool(point["inside_phenotype_roi"]) for point in audited_points),
            "points_excluded_from_roi": sum(not bool(point["inside_phenotype_roi"]) for point in audited_points),
            "shoot_retention_ratio": float(result["shoot_retention_ratio"]),
            "root_base_overlap_ratio": float(result["root_base_overlap_ratio"]),
            "seed_detected": bool(result["seed_detected"]),
            "basal_transition_pixels": int(result["basal_transition"].sum()),
            "old_base_point_id": old_base_id,
            "old_base_inside_roi": bool(old_base_audit and old_base_audit["inside_phenotype_roi"]),
            "proposed_anchor_point_id": "" if proposed_anchor is None else str(proposed_anchor["point_id"]),
            "proposed_anchor_distance_bbox_diag": (
                None
                if proposed_anchor is None
                else float(proposed_anchor["distance_to_transition_px"] / max(result["bbox_diag"], 1.0))
            ),
            "old_manual_path_semantics": str(manual_row.get("manual_path_semantics", "")),
            "old_manual_base_selection": str(manual_row.get("manual_base_selection", "")),
            "old_manual_note": str(manual_row.get("manual_note", "")),
            "test_images_read": 0,
        }
        rows.append(per_image)
        for point in audited_points:
            point_audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    **point,
                    "is_old_base": point["point_id"] == old_base_id,
                    "is_proposed_anchor": proposed_anchor is not None and point["point_id"] == proposed_anchor["point_id"],
                }
            )

        write_mask(output / "masks" / "phenotype_roi" / f"{dataset_id}.png", result["phenotype_roi"])
        write_mask(output / "masks" / "basal_transition" / f"{dataset_id}.png", result["basal_transition"])
        save_overlay(
            output / "overlays" / f"{dataset_id}.png",
            dataset_id,
            image,
            root_base,
            result,
            audited_points,
            proposed_anchor,
            old_base_id,
        )

    frame = pd.DataFrame(rows).sort_values("dataset_id")
    point_frame = pd.DataFrame(point_audit_rows).sort_values(["dataset_id", "point_id"])
    frame.to_csv(output / "per_image.csv", index=False, encoding="utf-8-sig")
    point_frame.to_csv(output / "point_audit.csv", index=False, encoding="utf-8-sig")
    pending = frame[["dataset_id"]].copy()
    pending["manual_roi_completeness"] = "pending"
    pending["manual_seed_root_exclusion"] = "pending"
    pending["manual_basal_transition"] = "pending"
    pending["manual_note"] = ""
    pending.to_csv(output / "manual_phenotype_roi_review_pending.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "phenotype-focused shoot ROI plus shoot-side basal transition v1",
        "purpose": "visual audit before any teacher regeneration or retraining",
        "images": int(len(frame)),
        "split": "val",
        "test_images_read": 0,
        "dataset_rows_or_splits_changed": False,
        "model_rerun": False,
        "teacher_regenerated": False,
        "median_shoot_retention_ratio": float(frame["shoot_retention_ratio"].median()),
        "minimum_shoot_retention_ratio": float(frame["shoot_retention_ratio"].min()),
        "images_below_0_95_shoot_retention": int((frame["shoot_retention_ratio"] < 0.95).sum()),
        "median_root_base_overlap_ratio": float(frame["root_base_overlap_ratio"].median()),
        "maximum_root_base_overlap_ratio": float(frame["root_base_overlap_ratio"].max()),
        "images_above_0_05_root_base_overlap": int((frame["root_base_overlap_ratio"] > 0.05).sum()),
        "images_with_seed_estimate": int(frame["seed_detected"].sum()),
        "total_input_learned_points": int(frame["input_point_count"].sum()),
        "total_points_inside_roi": int(frame["points_inside_roi"].sum()),
        "total_points_excluded_from_roi": int(frame["points_excluded_from_roi"].sum()),
        "global_points_inside_roi_ratio": float(
            frame["points_inside_roi"].sum() / max(1, frame["input_point_count"].sum())
        ),
        "images_with_proposed_learned_anchor": int((frame["proposed_anchor_point_id"] != "").sum()),
        "old_base_inside_roi_images": int(frame["old_base_inside_roi"].sum()),
        "manual_roi_status": "pending",
        "phenotype_accuracy_status": "pending_manual_measurement_reference",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_review_html(output, frame.to_dict("records"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())

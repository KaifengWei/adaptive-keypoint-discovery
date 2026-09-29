"""Audit saved train clean/raw predictions and render an offline paired gallery.

Only saved train outputs are read. This does not invoke a model or inspect GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RUNTIME = EXPERIMENT / "phenotype_pilot_protocol/runtime/raw_background_train_20260929"
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
METHODS = ("Teacher-direct", "Student-B", "Student-D")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved(input_root: Path, run_root: Path):
    package_file = input_root / "input_manifest.json"
    if sha(package_file) != "a270ce04284602277aa370ee9c9ad01cd6b9ed2d4476a13fad312a83547706ff":
        raise RuntimeError("Frozen input identity mismatch")
    package = json.loads(package_file.read_text(encoding="utf-8"))
    ids = [r["dataset_id"] for r in package["records"]]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise RuntimeError("Expected exactly 20 frozen train plants")
    saved = {}
    hashes = {"input_manifest": sha(package_file)}
    for method in METHODS:
        data_file = run_root / f"{method}.jsonl"
        ledger_file = run_root / f"{method}_ledger.json"
        ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
        if sha(data_file) != ledger["output_sha256"] or ledger["records"] != 40 or ledger["errors"] != 0:
            raise RuntimeError(f"Saved run hash/completeness failed: {method}")
        events = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(events) != 40 or any(event["status"] != "ok" or event["method"] != method for event in events):
            raise RuntimeError(f"Saved run event status failed: {method}")
        by_key = {(event["dataset_id"], event["condition"]): event for event in events}
        if set(by_key) != {(ident, condition) for ident in ids for condition in ("clean", "raw")}:
            raise RuntimeError(f"Missing/duplicate train condition: {method}")
        for record in package["records"]:
            for condition in ("clean", "raw"):
                event = by_key[(record["dataset_id"], condition)]
                if event["input_png_sha256"] != record[f"{condition}_sha256"]:
                    raise RuntimeError(f"Saved input image SHA mismatch: {method}")
                if event["prediction"]["image_sha256"] != event["input_png_sha256"]:
                    raise RuntimeError(f"Prediction image identity mismatch: {method}")
        saved[method] = by_key
        hashes[f"{method}_results"] = sha(data_file)
        hashes[f"{method}_ledger"] = sha(ledger_file)
    return package, saved, hashes


def pair_row(record: dict, method: str, events: dict, source_curve) -> dict:
    ident = record["dataset_id"]
    clean = events[(ident, "clean")]["prediction"]
    raw = events[(ident, "raw")]["prediction"]
    bc, br = clean["base_xy_model_canvas"], raw["base_xy_model_canvas"]
    base_move = None
    if bc is not None and br is not None:
        width, height = clean["width"], clean["height"]
        if (width, height) != (raw["width"], raw["height"]):
            raise RuntimeError(f"Paired dimensions differ: {ident}")
        xy_clean = source_curve([bc], width, height)[0]
        xy_raw = source_curve([br], width, height)[0]
        base_move = math.hypot(float(xy_raw[0] - xy_clean[0]), float(xy_raw[1] - xy_clean[1]))
    return {"dataset_id": ident, "source_frame_id": record["source_frame_id"], "method": method,
            "clean_points": clean["point_count"], "raw_points": raw["point_count"],
            "clean_accepted_nodes": clean["accepted_node_count"], "raw_accepted_nodes": raw["accepted_node_count"],
            "clean_paths": len(clean["paths"]), "raw_paths": len(raw["paths"]),
            "clean_base_present": bc is not None, "raw_base_present": br is not None,
            "base_movement_source_px_if_both": base_move,
            "clean_graph_status": clean["graph_status"], "raw_graph_status": raw["graph_status"],
            "clean_decoder_status": clean["decoder_status"], "raw_decoder_status": raw["decoder_status"]}


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        moves = [row["base_movement_source_px_if_both"] for row in group if row["base_movement_source_px_if_both"] is not None]
        result[method] = {
            "plants": len(group),
            "point_count_median_clean_raw": [statistics.median(row["clean_points"] for row in group), statistics.median(row["raw_points"] for row in group)],
            "point_count_changed_plants": sum(row["clean_points"] != row["raw_points"] for row in group),
            "accepted_node_changed_plants": sum(row["clean_accepted_nodes"] != row["raw_accepted_nodes"] for row in group),
            "path_count_median_clean_raw": [statistics.median(row["clean_paths"] for row in group), statistics.median(row["raw_paths"] for row in group)],
            "path_count_increased_raw": sum(row["raw_paths"] > row["clean_paths"] for row in group),
            "path_count_decreased_raw": sum(row["raw_paths"] < row["clean_paths"] for row in group),
            "path_count_unchanged": sum(row["raw_paths"] == row["clean_paths"] for row in group),
            "zero_path_clean_raw": [sum(row["clean_paths"] == 0 for row in group), sum(row["raw_paths"] == 0 for row in group)],
            "zero_path_clean_to_nonzero_raw": sum(row["clean_paths"] == 0 < row["raw_paths"] for row in group),
            "zero_path_nonzero_clean_to_raw": sum(row["clean_paths"] > 0 == row["raw_paths"] for row in group),
            "base_present_clean_raw": [sum(row["clean_base_present"] for row in group), sum(row["raw_base_present"] for row in group)],
            "base_movement_source_px_median_if_both": statistics.median(moves) if moves else None,
            "base_movement_paired_n": len(moves),
            "graph_failures_clean_raw": [sum(row["clean_graph_status"] == "failed" for row in group), sum(row["raw_graph_status"] == "failed" for row in group)],
        }
    return result


def html(samples: list[dict]) -> str:
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>train 原背景/白底冻结方法配对诊断</title><style>
body{margin:0;background:#edf2f8;color:#152944;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:3;background:#17365b;color:#fff;padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:5px 9px;border-radius:5px;border:1px solid #aabbd1;background:#fff;color:#13233a}main{max-width:1900px;margin:auto;padding:14px}
.notice{background:#fff;padding:12px 16px;border-left:4px solid #2563eb;margin-bottom:12px;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.card{background:#fff;border:1px solid #c6d3e1;border-radius:7px;overflow:hidden}.title{padding:7px 12px;background:#e7eff8;font-weight:700}.subtitle{padding:4px 12px;color:#425873;font-size:13px;min-height:22px}
.viewport{height:42vh;overflow:auto;text-align:center;background:#fff}.viewport img{display:block;margin:auto;max-width:100%;max-height:42vh;width:auto;height:auto}.foot{padding:5px 12px;font-size:13px}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
</style></head><body><header><strong>20株 train 原背景/白底配对诊断</strong><button id="prev">上一株</button><span id="counter"></span><button id="next">下一株</button><select id="sample"></select><label>缩放 <select id="zoom"><option value="fit">适合窗口</option><option value="1">原尺寸 1×</option><option value="2">原尺寸 2×</option></select></label></header>
<main><div class="notice">这只是<b>train 上的冻结方法探索性诊断</b>，不是 val/test 评价；没有GT，路径多或少不自动代表更好。左列=现有白底图，右列=同株高分辨率原背景裁剪；三个方法均无训练/调参。
<b>重要：</b>右列显示完整原裁剪供观察，但模型只看了相同 clean-derived ROI 内的原 RGB，ROI 外按冻结管线置白；两列共享冻结掩膜、ROI、graph、decoder。彩线=预测路径，黄点=候选点，灰叉=被图拒绝的点，红框=预测基点。</div><div class="grid" id="grid"></div></main>
<script>const samples=__SAMPLES__,names=["原图 · 白底标准化","原图 · 原背景裁剪","Teacher-direct · 白底","Teacher-direct · 原背景","Student-B · 白底","Student-B · 原背景","Student-D · 白底","Student-D · 原背景"];
const sel=document.getElementById('sample'),grid=document.getElementById('grid'),zoom=document.getElementById('zoom');let index=0,syncing=false;
samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id;sel.appendChild(o)});
function show(i){index=Math.max(0,Math.min(samples.length-1,i));let s=samples[index];sel.value=index;document.getElementById('counter').textContent=(index+1)+' / 20 · '+s.id;grid.textContent='';
s.panels.forEach((p,j)=>{let card=document.createElement('section');card.className='card';let title=document.createElement('div');title.className='title';title.textContent=names[j];let sub=document.createElement('div');sub.className='subtitle';sub.textContent=p.info;let view=document.createElement('div');view.className='viewport';let im=document.createElement('img');im.src=p.file;im.alt=names[j]+' '+s.id;view.appendChild(im);let foot=document.createElement('div');foot.className='foot';let a=document.createElement('a');a.href=p.file;a.target='_blank';a.textContent='打开原尺寸文件';foot.appendChild(a);card.append(title,sub,view,foot);grid.appendChild(card);
view.onscroll=()=>{if(syncing)return;syncing=true;grid.querySelectorAll('.viewport').forEach(v=>{if(v!==view){v.scrollTop=view.scrollTop;v.scrollLeft=view.scrollLeft}});syncing=false}});applyZoom();location.hash=s.id;}
function applyZoom(){grid.querySelectorAll('img').forEach(im=>{if(zoom.value==='fit'){im.style.maxWidth='100%';im.style.maxHeight='42vh';im.style.width='auto';im.style.height='auto'}else{let f=Number(zoom.value);let set=()=>{im.style.maxWidth='none';im.style.maxHeight='none';im.style.width=(im.naturalWidth*f)+'px';im.style.height=(im.naturalHeight*f)+'px'};if(im.complete)set();else im.onload=set}})}
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);sel.onchange=()=>show(Number(sel.value));zoom.onchange=applyZoom;
document.onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')show(index-1);if(e.key==='ArrowRight')show(index+1)};let initial=samples.findIndex(x=>x.id===location.hash.slice(1));show(initial<0?0:initial);
</script></body></html>""".replace("__SAMPLES__", payload)


def analyze(input_root: Path, run_root: Path, output: Path) -> None:
    package, saved, source_hashes = load_saved(input_root, run_root)
    sys.path.insert(0, str(EXPERIMENT / "phenotype_pilot_protocol"))
    sys.path.insert(0, str(EXPERIMENT / "locked_test_final_20260927"))
    from evaluate_frozen_phenotype_pilot import _source_curve
    from render_saved_test_gallery import overlay

    rows = [pair_row(record, method, saved[method], _source_curve)
            for record in package["records"] for method in METHODS]
    summary = aggregate(rows)
    if output.exists():
        raise FileExistsError(f"Never overwrite a saved paired analysis: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    file_hashes, samples = {}, []
    for record in package["records"]:
        ident = record["dataset_id"]
        clean_path = DATASET / "images/train" / f"{ident}.png"
        raw_path = input_root / record["raw_relative_path"]
        if sha(clean_path) != record["clean_sha256"] or sha(raw_path) != record["raw_sha256"]:
            raise RuntimeError(f"Paired image bytes changed: {ident}")
        original = {"clean": Image.open(clean_path).convert("RGB"), "raw": Image.open(raw_path).convert("RGB")}
        panels = []
        for condition in ("clean", "raw"):
            name = f"images/{ident}_{condition}_original.png"
            shutil.copyfile(clean_path if condition == "clean" else raw_path, output / name)
            panels.append({"file": name, "info": f"{record['width']}×{record['height']} px · 同株配准"})
            file_hashes[name] = sha(output / name)
        for method in METHODS:
            for condition in ("clean", "raw"):
                prediction = saved[method][(ident, condition)]["prediction"]
                image = overlay(original[condition], prediction, _source_curve)
                name = f"images/{ident}_{method.replace('-', '_')}_{condition}.png"
                image.save(output / name, optimize=True)
                panels.append({"file": name,
                               "info": f"{prediction['point_count']} 点 · {prediction['accepted_node_count']} 接受节点 · {len(prediction['paths'])} 路径"})
                file_hashes[name] = sha(output / name)
        samples.append({"id": ident, "panels": panels})
    with (output / "per_plant.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({"scope": "20 independent train source frames; fixed ROI/masks; exploratory only",
                                        "test_model_reads": 0, "results": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text(html(samples), encoding="utf-8")
    file_hashes["index.html"] = sha(output / "index.html")
    file_hashes["summary.json"] = sha(summary_path)
    file_hashes["per_plant.csv"] = sha(output / "per_plant.csv")
    ledger = {"scope": "saved train exploratory visualization; no new model inference/GT evaluation",
              "plants": len(samples), "panels": len(samples) * 8, "source_sha256": source_hashes,
              "source_curve_sha256": sha(EXPERIMENT / "phenotype_pilot_protocol/evaluate_frozen_phenotype_pilot.py"),
              "overlay_renderer_sha256": sha(EXPERIMENT / "locked_test_final_20260927/render_saved_test_gallery.py"),
              "file_sha256": file_hashes}
    (output / "gallery_manifest.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "plants": len(samples), "panels": len(samples) * 8,
                      "summary_sha256": file_hashes["summary.json"], "results": summary}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUNTIME / "input_01")
    parser.add_argument("--run", type=Path, default=RUNTIME / "run_01")
    parser.add_argument("--output", type=Path, default=RUNTIME / "analysis_01")
    args = parser.parse_args()
    analyze(args.input, args.run, args.output)

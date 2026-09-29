"""Visualize the frozen 40-plant V4 val predictions without model inference.

This is a retrospective development-set display, not an independent evaluation.
The saved path sources are verified against the frozen pilot comparison ledger.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
COMPARISON = HERE / "runtime/method_comparison/frozen_pilot_20260924_v5_final/frozen_method_comparison.json"
INVERSE_SOURCE = HERE / "evaluate_frozen_phenotype_pilot.py"
INVERSE_SHA = "b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8"
METHODS = {
    "Teacher-direct": {
        "folder": HERE / "runtime/method_comparison/teacher_direct_route_b_local_decoder_val_20260924",
        "points": HERE / "runtime/method_gate_20260925/sources/teacher_points.csv",
    },
    "Student-B": {
        "folder": EXPERIMENT / "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val",
        "points": EXPERIMENT / "evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv",
    },
    "Student-D": {
        "folder": EXPERIMENT / "evaluation_outputs/point_conditioned_organ_paths_v3_structure_coverage_val",
        "points": EXPERIMENT / "evaluation_outputs/core_dinov2_v4_structure_coverage_val/points.csv",
    },
}
COLORS = ("#e11d48", "#0891b2", "#9333ea", "#ea580c", "#0d9488", "#ca8a04", "#be123c", "#2563eb")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_inputs():
    if sha(INVERSE_SOURCE) != INVERSE_SHA:
        raise RuntimeError("Frozen inverse-letterbox source hash mismatch")
    sys.path.insert(0, str(HERE))
    from evaluate_frozen_phenotype_pilot import _source_curve

    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    manifest = sorted(read_csv(DATASET / "manifests/val.csv"), key=lambda row: row["dataset_id"])
    ids = [row["dataset_id"] for row in manifest]
    if len(ids) != 40 or len(set(ids)) != 40 or any(not name.startswith("v4_val_") for name in ids):
        raise RuntimeError("Expected exactly 40 V4 val plants")
    loaded = {}
    source_hashes = {"comparison": sha(COMPARISON), "inverse_letterbox": INVERSE_SHA}
    for method, source in METHODS.items():
        folder, points_path = source["folder"], source["points"]
        expected = comparison["method_source"][method]
        paths_path, summary_path = folder / "paths.jsonl", folder / "summary.json"
        if sha(paths_path) != expected["paths_jsonl_sha256"] or sha(summary_path) != expected["saved_summary_sha256"]:
            raise RuntimeError(f"Frozen saved-path hash mismatch: {method}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(summary.get(key) != expected["frozen_settings"][key] for key in expected["frozen_settings"]):
            raise RuntimeError(f"Frozen method settings mismatch: {method}")
        per_image = {row["dataset_id"]: row for row in read_csv(folder / "per_image.csv")}
        paths = defaultdict(list)
        with paths_path.open(encoding="utf-8") as stream:
            for line in stream:
                path = json.loads(line)
                paths[path["dataset_id"]].append(path)
        points = defaultdict(list)
        for point in read_csv(points_path):
            points[point["dataset_id"]].append(point)
        if set(per_image) != set(ids) or set(paths) - set(ids) or set(points) != set(ids):
            raise RuntimeError(f"Saved val IDs incomplete or unexpected: {method}")
        for ident in ids:
            if len(paths[ident]) != int(per_image[ident]["decoded_path_count"]):
                raise RuntimeError(f"Saved path-count mismatch: {method} {ident}")
            if len(points[ident]) != int(per_image[ident]["input_point_count"]):
                raise RuntimeError(f"Saved point-count mismatch: {method} {ident}")
        loaded[method] = {"per_image": per_image, "paths": paths, "points": points}
        source_hashes[f"{method}_paths"] = sha(paths_path)
        source_hashes[f"{method}_summary"] = sha(summary_path)
        source_hashes[f"{method}_per_image"] = sha(folder / "per_image.csv")
        source_hashes[f"{method}_points"] = sha(points_path)
    return manifest, loaded, _source_curve, source_hashes


def overlay(image: Image.Image, paths: list[dict], points: list[dict], source_curve) -> Image.Image:
    width, height = image.size
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_width = max(2, min(4, round(min(width, height) / 180)))
    radius = max(3, min(5, round(min(width, height) / 135)))
    for index, path in enumerate(paths):
        line = [tuple(map(float, xy)) for xy in source_curve(path["full_base_to_tip_path"], width, height)]
        if len(line) < 2:
            raise RuntimeError("Saved path has fewer than two points")
        color = COLORS[index % len(COLORS)]
        draw.line(line, fill=color + "D0", width=line_width, joint="curve")
        tip_x, tip_y = line[-1]
        draw.ellipse((tip_x - radius, tip_y - radius, tip_x + radius, tip_y + radius),
                     fill="#ffffff", outline=color, width=2)
    for point in points:
        x, y = float(point["x_source"]), float(point["y_source"])
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError("Saved point outside source image")
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill="#facc15", outline="#111827", width=1)
    if paths:
        x, y = (float(v) for v in source_curve(paths[0]["full_base_to_tip_path"][:1], width, height)[0])
        side = radius + 2
        draw.rectangle((x - side, y - side, x + side, y + side), fill="#ffffff", outline="#dc2626", width=2)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def gallery_html(samples: list[dict]) -> str:
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4 val 冻结方法历史效果对照</title><style>
body{margin:0;background:#edf2f8;color:#14243d;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:3;background:#17365b;color:#fff;padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:5px 9px;border-radius:5px;border:1px solid #aabbd1;background:#fff;color:#13233a}
main{max-width:1900px;margin:auto;padding:14px}.notice{background:#fff;padding:12px 16px;border-left:4px solid #2563eb;margin-bottom:12px;border-radius:5px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.card{background:#fff;border:1px solid #c6d3e1;border-radius:7px;overflow:hidden}
.title{padding:8px 12px;background:#e7eff8;font-weight:700}.subtitle{padding:5px 12px;color:#425873;font-size:13px;min-height:22px}
.viewport{height:58vh;overflow:auto;text-align:center;background:#fff}.viewport img{display:block;margin:auto;max-width:100%;max-height:58vh;width:auto;height:auto}
.foot{padding:6px 12px;font-size:13px}@media(max-width:980px){.grid{grid-template-columns:1fr}.viewport{height:53vh}.viewport img{max-height:53vh}}
</style></head><body><header><strong>V4 val 历史效果对照</strong><button id="prev">上一株</button><span id="counter"></span><button id="next">下一株</button>
<select id="sample"></select><label>缩放 <select id="zoom"><option value="fit">适合窗口</option><option value="1">原尺寸 1×</option><option value="2">原尺寸 2×</option></select></label></header>
<main><div class="notice">这是<b>40株开发/验证集（val）的历史存档</b>，曾用于方法开发，<b>不是独立最终 test</b>；页面没有重新运行模型或重新计算效果指标。
左上是无标注标准化原图；彩线是已保存的基部—叶尖预测路径，黄点是已保存的候选关键点（不代表全部被图接受），红框是路径起点；白圈是路径终点。不同模型的路径颜色/编号不表示同一叶片。零路径如实显示。放大后四格同步滚动。</div><div class="grid" id="grid"></div></main>
<script>const samples=__SAMPLES__;const names=["无标注原图","Teacher-direct + Pipeline V1","Student-B epoch53 + Pipeline V1","Student-D + Pipeline V1（次要对照）"];
const sel=document.getElementById('sample'),grid=document.getElementById('grid'),zoom=document.getElementById('zoom');let index=0,syncing=false;
samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id;sel.appendChild(o)});
function show(i){index=Math.max(0,Math.min(samples.length-1,i));let s=samples[index];sel.value=index;document.getElementById('counter').textContent=(index+1)+' / 40 · '+s.id;grid.textContent='';
s.panels.forEach((p,j)=>{let card=document.createElement('section');card.className='card';let title=document.createElement('div');title.className='title';title.textContent=names[j];let sub=document.createElement('div');sub.className='subtitle';sub.textContent=p.info;
let view=document.createElement('div');view.className='viewport';let im=document.createElement('img');im.src=p.file;im.alt=names[j]+' '+s.id;view.appendChild(im);let foot=document.createElement('div');foot.className='foot';let a=document.createElement('a');a.href=p.file;a.target='_blank';a.textContent='打开原尺寸文件';foot.appendChild(a);card.append(title,sub,view,foot);grid.appendChild(card);
view.onscroll=()=>{if(syncing)return;syncing=true;grid.querySelectorAll('.viewport').forEach(v=>{if(v!==view){v.scrollTop=view.scrollTop;v.scrollLeft=view.scrollLeft}});syncing=false}});applyZoom();location.hash=s.id;}
function applyZoom(){grid.querySelectorAll('img').forEach(im=>{if(zoom.value==='fit'){im.style.maxWidth='100%';im.style.maxHeight='58vh';im.style.width='auto';im.style.height='auto'}else{let f=Number(zoom.value);let set=()=>{im.style.maxWidth='none';im.style.maxHeight='none';im.style.width=(im.naturalWidth*f)+'px';im.style.height=(im.naturalHeight*f)+'px'};if(im.complete)set();else im.onload=set}})}
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);sel.onchange=()=>show(Number(sel.value));zoom.onchange=applyZoom;
document.onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')show(index-1);if(e.key==='ArrowRight')show(index+1)};
let initial=samples.findIndex(x=>x.id===location.hash.slice(1));show(initial<0?0:initial);
</script></body></html>""".replace("__SAMPLES__", payload)


def render(output: Path) -> dict:
    manifest, methods, source_curve, source_hashes = load_inputs()
    if output.exists():
        raise FileExistsError(f"Never overwrite an existing gallery: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    gallery_rows, file_hashes = [], {}
    for row in manifest:
        ident = row["dataset_id"]
        source = (DATASET / Path(row["relative_path"].replace("\\", "/"))).resolve()
        if not source.is_relative_to(DATASET.resolve()) or sha(source) != row["output_sha256"]:
            raise RuntimeError(f"Val image identity/hash mismatch: {ident}")
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        if (original.width, original.height) != (int(row["fullplant_quality_original_width"]), int(row["fullplant_quality_original_height"])):
            raise RuntimeError(f"Val image dimensions mismatch: {ident}")
        panels = []
        raw_name = f"images/{ident}_original.png"
        shutil.copyfile(source, output / raw_name)
        panels.append({"file": raw_name, "info": f"标准化原图 {original.width}×{original.height} px"})
        file_hashes[raw_name] = sha(output / raw_name)
        for method, records in methods.items():
            paths, points = records["paths"][ident], records["points"][ident]
            name = f"images/{ident}_{method.replace('-', '_')}.png"
            overlay(original, paths, points, source_curve).save(output / name, optimize=True)
            status = " · 零路径" if not paths else ""
            panels.append({"file": name, "info": f"{len(paths)} 条路径 · {len(points)} 个候选点 · {records['per_image'][ident]['accepted_node_count']} 个接受节点{status}"})
            file_hashes[name] = sha(output / name)
        gallery_rows.append({"id": ident, "panels": panels})
    html = output / "index.html"
    html.write_text(gallery_html(gallery_rows), encoding="utf-8")
    file_hashes["index.html"] = sha(html)
    ledger = {"kind": "historical frozen V4 val visualization; no model inference, no metric selection",
              "plants": len(gallery_rows), "methods": list(METHODS), "source_sha256": source_hashes, "file_sha256": file_hashes}
    (output / "gallery_manifest.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "plants": len(gallery_rows), "panels": len(gallery_rows) * 4,
            "index_sha256": file_hashes["index.html"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.output), ensure_ascii=False))

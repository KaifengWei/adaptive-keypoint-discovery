"""Render the sealed V4 test predictions; never invoke a model or change scores.

The HTML and PNGs are private post-test visualizations, not a new evaluation.
Paths and points are mapped with the frozen inverse-letterbox implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROTOCOL = EXPERIMENT / "phenotype_pilot_protocol"
DATASET = EXPERIMENT / "data_stage_clean_v4_fullplant_candidate"
RUNTIME = PROTOCOL / "runtime" / "v4_locked_test_final_20260929"
PREDICTIONS = RUNTIME / "run_01" / "v4-locked-test-saved-predictions-v1.json"
EVALUATION = RUNTIME / "evaluation_01" / "evaluation_ledger.json"
PREDICTIONS_SHA = "305974bdfdbe9734e9b53f2200f0e6003a3cada01489e1061ac34cfb71267375"
EVALUATION_SHA = "76145183e356a96cd100edde7b3d50b44b1d86e80d32ebc9b57f920e3735ee19"
INVERSE_SOURCE = PROTOCOL / "evaluate_frozen_phenotype_pilot.py"
INVERSE_SHA = "b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8"
METHODS = ("Teacher-direct", "Student-B", "Student-D")
COLORS = ("#e11d48", "#0891b2", "#9333ea", "#ea580c", "#0d9488", "#ca8a04", "#be123c", "#2563eb")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    for path, expected in ((PREDICTIONS, PREDICTIONS_SHA), (EVALUATION, EVALUATION_SHA), (INVERSE_SOURCE, INVERSE_SHA)):
        if sha(path) != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {path}")
    sys.path.insert(0, str(PROTOCOL))
    from evaluate_frozen_phenotype_pilot import _source_curve

    bundle = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "v4-locked-test-saved-predictions-v1" or set(bundle["methods"]) != set(METHODS):
        raise RuntimeError("Frozen saved-prediction schema changed")
    with (DATASET / "manifests" / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = sorted(csv.DictReader(stream), key=lambda item: item["dataset_id"])
    identifiers = [row["dataset_id"] for row in manifest]
    if len(identifiers) != 40 or len(set(identifiers)) != 40:
        raise RuntimeError("Expected exactly 40 locked test plants")
    methods = {}
    for method in METHODS:
        rows = bundle["methods"][method]
        if [row["dataset_id"] for row in rows] != identifiers:
            raise RuntimeError(f"Incomplete/reordered saved predictions: {method}")
        methods[method] = dict(zip(identifiers, rows))
    return manifest, methods, _source_curve


def source_point(xy, width, height, source_curve):
    return tuple(float(v) for v in source_curve([list(xy)], width, height)[0])


def overlay(image: Image.Image, record: dict, source_curve) -> Image.Image:
    width, height = image.size
    if (record["width"], record["height"]) != (width, height):
        raise RuntimeError("Prediction/image dimensions mismatch")
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_width = max(2, min(4, round(min(width, height) / 180)))
    radius = max(3, min(5, round(min(width, height) / 135)))

    for index, path in enumerate(record["paths"]):
        points = [tuple(float(v) for v in xy) for xy in source_curve(path["full_base_to_tip_path"], width, height)]
        color = COLORS[index % len(COLORS)]
        draw.line(points, fill=color + "D0", width=line_width, joint="curve")
        tip_x, tip_y = points[-1]
        draw.ellipse((tip_x - radius, tip_y - radius, tip_x + radius, tip_y + radius),
                     fill="#ffffff", outline=color, width=2)

    rejected = {str(item.get("point_id")) for item in record["provenance"]["rejected_points"]}
    for point in record["provenance"]["point_records"]:
        x, y = source_point((point["x"], point["y"]), width, height, source_curve)
        if str(point["point_id"]) in rejected:
            draw.line(((x - radius, y - radius), (x + radius, y + radius)), fill="#64748b", width=2)
            draw.line(((x - radius, y + radius), (x + radius, y - radius)), fill="#64748b", width=2)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill="#facc15", outline="#111827", width=1)

    if record["base_xy_model_canvas"] is not None:
        x, y = source_point(record["base_xy_model_canvas"], width, height, source_curve)
        side = radius + 2
        draw.rectangle((x - side, y - side, x + side, y + side), fill="#ffffff", outline="#dc2626", width=2)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def quicklook(images: list[Image.Image], titles: list[str], target: Path) -> None:
    panel_w, panel_h, head = 810, 440, 34
    sheet = Image.new("RGB", (panel_w * 2, (panel_h + head) * 2), "#e8edf4")
    draw = ImageDraw.Draw(sheet)
    for index, (source, title) in enumerate(zip(images, titles)):
        col, row = index % 2, index // 2
        left, top = col * panel_w, row * (panel_h + head)
        fitted = source.copy()
        fitted.thumbnail((panel_w - 16, panel_h - 16), Image.Resampling.LANCZOS)
        x = left + (panel_w - fitted.width) // 2
        y = top + head + (panel_h - fitted.height) // 2
        sheet.paste(fitted, (x, y))
        draw.text((left + 12, top + 9), title, fill="#172554")
    sheet.save(target, optimize=True)


def gallery_html(samples: list[dict]) -> str:
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4 test 冻结预测效果对照</title><style>
body{margin:0;background:#edf2f8;color:#14243d;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:3;background:#17365b;color:#fff;padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:5px 9px;border-radius:5px;border:1px solid #aabbd1;background:#fff;color:#13233a}
header strong{margin-right:10px}header span{font-weight:600}main{max-width:1900px;margin:auto;padding:14px}
.notice{background:#fff;padding:12px 16px;border-left:4px solid #2563eb;margin-bottom:12px;border-radius:5px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.card{background:#fff;border:1px solid #c6d3e1;border-radius:7px;overflow:hidden}.title{padding:8px 12px;background:#e7eff8;font-weight:700}
.subtitle{padding:5px 12px;color:#425873;font-size:13px;min-height:22px}.viewport{height:60vh;overflow:auto;text-align:center;background:#fff}
.viewport img{display:block;margin:auto;max-width:100%;max-height:60vh;width:auto;height:auto}.foot{padding:6px 12px;font-size:13px}
@media(max-width:980px){.grid{grid-template-columns:1fr}.viewport{height:55vh}.viewport img{max-height:55vh}}
</style></head><body>
<header><strong>V4 test 冻结预测效果</strong><button id="prev">上一株</button><span id="counter"></span><button id="next">下一株</button>
<select id="sample"></select><label>缩放 <select id="zoom"><option value="fit">适合窗口</option><option value="1">原尺寸 1×</option><option value="2">原尺寸 2×</option></select></label>
<button id="worst">预定 rank 1</button><button id="typical">预定 rank 20</button><button id="best">预定 rank 40</button></header>
<main><div class="notice">仅展示已封存的 test 预测，<b>没有重新运行模型</b>。彩色线=模型解码路径（编号颜色不代表跨模型同一叶片）；黄点=模型点被结构图接受；灰叉=被拒点；红框=预测共同基点；空路径会如实显示。左上是无标注标准化原图，不是 GT。点击下方文件名可查看原尺寸图。四格在放大后同步滚动。</div>
<div class="grid" id="grid"></div></main>
<script>const samples=__SAMPLES__;const names=["无标注原图","Teacher-direct + Pipeline V1","Student-B epoch53 + Pipeline V1","Student-D + Pipeline V1（次要对照）"];
const sel=document.getElementById('sample'),grid=document.getElementById('grid'),zoom=document.getElementById('zoom');let index=0,syncing=false;
samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id;sel.appendChild(o)});
function show(i){index=Math.max(0,Math.min(samples.length-1,i));let s=samples[index];sel.value=index;document.getElementById('counter').textContent=(index+1)+' / 40 · '+s.id;grid.textContent='';
  s.panels.forEach((p,j)=>{let card=document.createElement('section');card.className='card';let title=document.createElement('div');title.className='title';title.textContent=names[j];
    let sub=document.createElement('div');sub.className='subtitle';sub.textContent=p.info;let view=document.createElement('div');view.className='viewport';let im=document.createElement('img');im.src=p.file;im.alt=names[j]+' '+s.id;view.appendChild(im);
    let foot=document.createElement('div');foot.className='foot';let a=document.createElement('a');a.href=p.file;a.target='_blank';a.textContent='打开原尺寸文件';foot.appendChild(a);
    card.append(title,sub,view,foot);grid.appendChild(card);view.onscroll=()=>{if(syncing)return;syncing=true;grid.querySelectorAll('.viewport').forEach(v=>{if(v!==view){v.scrollTop=view.scrollTop;v.scrollLeft=view.scrollLeft}});syncing=false}});
  applyZoom();location.hash=s.id;}
function applyZoom(){grid.querySelectorAll('img').forEach(im=>{if(zoom.value==='fit'){im.style.maxWidth='100%';im.style.maxHeight='60vh';im.style.width='auto';im.style.height='auto'}else{let f=Number(zoom.value);let set=()=>{im.style.maxWidth='none';im.style.maxHeight='none';im.style.width=(im.naturalWidth*f)+'px';im.style.height=(im.naturalHeight*f)+'px'};if(im.complete)set();else im.onload=set}})}
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);sel.onchange=()=>show(Number(sel.value));zoom.onchange=applyZoom;
[['worst','v4_test_0025'],['typical','v4_test_0022'],['best','v4_test_0021']].forEach(([id,key])=>document.getElementById(id).onclick=()=>show(samples.findIndex(x=>x.id===key)));
document.onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')show(index-1);if(e.key==='ArrowRight')show(index+1)};
let initial=samples.findIndex(x=>x.id===location.hash.slice(1));show(initial<0?0:initial);
</script></body></html>""".replace("__SAMPLES__", payload)


def render(output: Path) -> None:
    manifest, methods, source_curve = load_inputs()
    if output.exists():
        raise FileExistsError(f"Never overwrite an existing gallery: {output}")
    output.mkdir(parents=True, exist_ok=False)
    images_dir = output / "images"
    images_dir.mkdir()
    gallery_rows = []
    hashes = {}
    for row in manifest:
        ident = row["dataset_id"]
        source = (DATASET / Path(row["relative_path"].replace("\\", "/"))).resolve()
        if not source.is_relative_to(DATASET.resolve()) or sha(source) != row["output_sha256"]:
            raise RuntimeError(f"Original image hash/path mismatch: {ident}")
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        panels = []
        raw_name = f"images/{ident}_original.png"
        shutil.copyfile(source, output / raw_name)
        panels.append({"file": raw_name, "info": f"原图 {original.width}×{original.height} px · SHA核验通过"})
        rendered = [original]
        for method in METHODS:
            record = methods[method][ident]
            if record["image_sha256"] != row["output_sha256"]:
                raise RuntimeError(f"Prediction/source identity mismatch: {ident} {method}")
            result = overlay(original, record, source_curve)
            method_slug = method.replace("-", "_")
            name = f"images/{ident}_{method_slug}.png"
            result.save(output / name, optimize=True)
            panels.append({"file": name, "info": f"{len(record['paths'])} 条路径 · {record['point_count']} 个模型点 · {record['accepted_node_count']} 个接受节点"})
            rendered.append(result)
        gallery_rows.append({"id": ident, "panels": panels})
        if ident in {"v4_test_0025", "v4_test_0022", "v4_test_0021"}:
            name = f"quicklook_{ident}.png"
            quicklook(rendered, ["Original", "Teacher-direct", "Student-B", "Student-D"], output / name)
            hashes[name] = sha(output / name)
        for panel in panels:
            hashes[panel["file"]] = sha(output / panel["file"])
    html_path = output / "index.html"
    html_path.write_text(gallery_html(gallery_rows), encoding="utf-8")
    hashes["index.html"] = sha(html_path)
    ledger = {
        "kind": "post-test visualization only; no model inference or metric recalculation",
        "prediction_sha256": PREDICTIONS_SHA, "evaluation_ledger_sha256": EVALUATION_SHA,
        "inverse_letterbox_source_sha256": INVERSE_SHA, "plants": len(gallery_rows),
        "methods": list(METHODS), "file_sha256": hashes,
    }
    (output / "gallery_manifest.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "plants": len(gallery_rows), "panels": len(gallery_rows) * 4,
                      "quicklooks": 3, "index_sha256": hashes["index.html"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)

"""Build an offline, method-blind adjudication page from frozen human traces.

Public adjudication assets contain only an opaque pair ID, the standardized
source image, three per-item randomly labelled human candidate sets, and broad
disagreement categories. The private label/source mapping stays under
``runtime/admin``. No model output or V4 test resource is read.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
from typing import Any


BUILD_VERSION = "method-blind-human-adjudication-v1"
PUBLIC_FORBIDDEN = ("v4_val_", "dataset_id", "rater", "measurement_round", "teacher", "student", "decoder")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def public_trace(trace: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "candidate_trace_id": f"T{index:02d}",
        "points": [[round(float(point["x"]), 5), round(float(point["y"]), 5)] for point in trace["submitted_curve"]],
        "main": bool(trace["reference_main_path"]),
    }


def main() -> None:
    protocol_dir = Path(__file__).resolve().parent
    runtime = protocol_dir / "runtime"
    analysis_dir = runtime / "reliability" / "20260905"
    output_dir = runtime / "adjudication" / "method_blind_20260905"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        ("R1", 1): runtime / "measurements" / "rater1_round1" / "revision_after_field_resolution" / "VPUFUJSD7GCZ_traces.json",
        ("R1", 2): runtime / "measurements" / "rater1_round2" / "raw_export_20260904" / "CHQ474APYAYU_traces.json",
        ("R2", 1): runtime / "measurements" / "rater2_round1" / "revision_after_field_resolution_20260905" / "6GHTS9ZXUD4E_traces.json",
    }
    payloads = {key: load_payload(path) for key, path in source_paths.items()}
    records = {key: {record["blind_id"]: record for record in payload["records"]} for key, payload in payloads.items()}
    mapping = [row for row in read_csv(runtime / "admin" / "blind_id_admin_mapping.csv") if row["pilot_group"] != "practice"]
    mapping_by_key = {(row["dataset_id"], row["rater_id"], int(row["measurement_round"])): row for row in mapping}
    candidate_rows = read_csv(analysis_dir / "adjudication_candidates.csv")
    candidate_pair_ids = sorted({row["adjudication_pair_id"] for row in candidate_rows})
    pair_to_dataset: dict[str, str] = {}
    for row in mapping:
        pair_to_dataset.setdefault(row["adjudication_pair_id"], row["dataset_id"])
        if pair_to_dataset[row["adjudication_pair_id"]] != row["dataset_id"]:
            raise ValueError("one opaque pair ID maps to multiple samples")

    admin_mapping_path = runtime / "admin" / "adjudication_candidate_set_mapping_20260905.csv"
    existing_label_rows = read_csv(admin_mapping_path) if admin_mapping_path.exists() else []
    existing_labels_by_pair: dict[str, list[dict[str, str]]] = {}
    for row in existing_label_rows:
        existing_labels_by_pair.setdefault(row["adjudication_pair_id"], []).append(row)

    reason_labels = {
        "length_relative_diff_gt_5pct": "基部—叶尖结构路径长度差异超过预设门槛",
        "divergence_angle_diff_gt_5deg": "分化角差异超过预设门槛",
        "leaf_existence_or_identity_differs": "可见叶数量或对应身份不一致",
        "reference_main_identity_differs": "确定性主路径身份不一致",
    }
    admin_rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    randomizer = secrets.SystemRandom()
    source_keys = list(source_paths)
    r1_package = runtime / "packages" / "rater1_round1"
    for order, pair_id in enumerate(candidate_pair_ids, start=1):
        dataset_id = pair_to_dataset[pair_id]
        existing = existing_labels_by_pair.get(pair_id, [])
        if existing:
            if len(existing) != 3 or {row["candidate_set_label"] for row in existing} != {"A", "B", "C"}:
                raise ValueError(f"invalid persisted candidate labels for {pair_id}")
            labelled = sorted(
                ((row["candidate_set_label"], (row["rater_id"], int(row["measurement_round"]))) for row in existing),
                key=lambda item: item[0],
            )
        else:
            shuffled = list(source_keys)
            randomizer.shuffle(shuffled)
            labelled = list(zip(("A", "B", "C"), shuffled))
        reasons = sorted({reason for row in candidate_rows if row["adjudication_pair_id"] == pair_id for reason in row["adjudication_reasons"].split(";") if reason})
        sets = []
        r1_map = mapping_by_key[(dataset_id, "R1", 1)]
        source_image = r1_package / r1_map["image_alias"]
        destination_image = images_dir / f"{pair_id}.png"
        shutil.copyfile(source_image, destination_image)
        for label, source_key in labelled:
            map_row = mapping_by_key[(dataset_id, source_key[0], source_key[1])]
            record = records[source_key][map_row["blind_id"]]
            sorted_traces = sorted(record["traces"], key=lambda trace: str(trace["gt_leaf_id_postsubmit"]))
            sets.append({
                "label": label,
                "trace_count": len(sorted_traces),
                "traces": [public_trace(trace, index) for index, trace in enumerate(sorted_traces, start=1)],
            })
            admin_rows.append({
                "adjudication_pair_id": pair_id,
                "candidate_set_label": label,
                "dataset_id": dataset_id,
                "rater_id": source_key[0],
                "measurement_round": source_key[1],
                "blind_id": map_row["blind_id"],
                "package_id": map_row["package_id"],
            })
        items.append({
            "order": order,
            "adjudication_pair_id": pair_id,
            "image": f"images/{pair_id}.png",
            "reason_categories": [reason_labels[reason] for reason in reasons],
            "sets": sets,
        })

    write_csv(
        admin_mapping_path,
        admin_rows,
        ["adjudication_pair_id", "candidate_set_label", "dataset_id", "rater_id", "measurement_round", "blind_id", "package_id"],
    )
    data = {
        "build_version": BUILD_VERSION,
        "item_count": len(items),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "instructions": "候选集标签在每张图内独立随机，不代表固定测量者或固定轮次。仅依据原图判断。",
        "items": items,
    }
    (output_dir / "data.js").write_text("window.ADJUDICATION_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    (output_dir / "index.html").write_text(HTML, encoding="utf-8")
    (output_dir / "style.css").write_text(CSS, encoding="utf-8")
    (output_dir / "app.js").write_text(JS, encoding="utf-8")

    public_files = [
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"package_manifest.json", "preview_first_item.png"}
    ]
    leaks = []
    source_blind_ids = {row["blind_id"].lower() for row in mapping}
    for path in public_files:
        relative = str(path.relative_to(output_dir)).replace("\\", "/")
        lower_name = relative.lower()
        for token in PUBLIC_FORBIDDEN:
            if token in lower_name:
                leaks.append({"file": relative, "token": token, "location": "filename"})
        if path.suffix.lower() in {".html", ".js", ".css", ".json", ".csv", ".txt"}:
            text = path.read_text(encoding="utf-8-sig").lower()
            for token in PUBLIC_FORBIDDEN:
                if token in text:
                    leaks.append({"file": relative, "token": token, "location": "content"})
            for blind_id in source_blind_ids:
                if blind_id in text:
                    leaks.append({"file": relative, "token": "source_blind_id", "location": "content"})
    if leaks:
        raise ValueError(f"method-blind package leakage: {leaks[:10]}")
    manifest = {
        "status": "ready_for_method_blind_human_adjudication",
        "build_version": BUILD_VERSION,
        "item_count": len(items),
        "candidate_set_labels_randomized_per_item": True,
        "identity_or_method_leak_hits": 0,
        "human_source_export_sha256": [sha256_file(path) for path in source_paths.values()],
        "public_files": {str(path.relative_to(output_dir)).replace("\\", "/"): sha256_file(path) for path in public_files},
        "model_outputs_read": 0,
        "v4_test_read": 0,
        "decision_options": ["accept_candidate_set_A", "accept_candidate_set_B", "accept_candidate_set_C", "needs_custom_redraw"],
        "next_step": "export completed anonymous adjudication CSV and JSON; preserve all original human measurements",
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"public_files", "human_source_export_sha256"}}, ensure_ascii=False, indent=2))


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>匿名人工轨迹裁决</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <div><h1>匿名人工轨迹裁决</h1><p>只比较三套人工候选轨迹；不含任何模型输出、真实样本编号或测量者身份。</p></div>
    <div class="nav"><button id="prev">上一张</button><strong id="counter"></strong><button id="next">下一张</button><button id="backup">备份进度 JSON</button><button id="export">完成并导出</button></div>
  </header>
  <main>
    <section class="notice"><b>判断原则：</b>先看无标注原图，再依次核对候选集 A/B/C。标签在每张图内重新随机，不对应固定人员或轮次。选择整体最符合真实植株中心结构的一套；若没有任何一套完整正确，选择“需要重新描迹”。鼠标滚轮缩放，按住拖动平移，双击复位。</section>
    <section class="meta"><span>匿名配对：<b id="pair"></b></span><span id="status"></span><div id="reasons"></div></section>
    <section id="panels" class="panels"></section>
    <section class="decision">
      <h2>本图裁决</h2>
      <div class="choices">
        <label><input type="radio" name="decision" value="accept_candidate_set_A">接受候选集 A</label>
        <label><input type="radio" name="decision" value="accept_candidate_set_B">接受候选集 B</label>
        <label><input type="radio" name="decision" value="accept_candidate_set_C">接受候选集 C</label>
        <label><input type="radio" name="decision" value="needs_custom_redraw">三套均不完整，需要重新描迹</label>
      </div>
      <label>裁决信心 <select id="confidence"><option value="">请选择</option><option value="high">高</option><option value="medium">中</option><option value="low">低</option></select></label>
      <label>备注（说明漏叶、错叶、轨迹偏离或需重描的具体位置）<textarea id="notes" rows="3"></textarea></label>
      <div class="actions"><button id="submit">提交本图裁决</button><button id="revise">创建修订</button><button id="resetView">复位全部视图</button></div>
      <p id="message"></p>
    </section>
  </main>
  <script src="data.js"></script><script src="app.js"></script>
</body>
</html>
'''


CSS = r'''
:root{font-family:"Microsoft YaHei",Arial,sans-serif;color:#17324d;background:#eef3f8}*{box-sizing:border-box}body{margin:0}header{position:sticky;top:0;z-index:4;background:#183b63;color:white;display:flex;align-items:center;justify-content:space-between;padding:10px 18px;gap:18px}h1{margin:0;font-size:23px}header p{margin:3px 0 0;font-size:13px;opacity:.9}.nav{display:flex;align-items:center;gap:8px;flex-wrap:wrap}button,select,textarea{font:inherit}button{border:1px solid #98abc0;background:white;border-radius:6px;padding:8px 13px;cursor:pointer}button:hover{background:#e9f2ff}main{max-width:1900px;margin:auto;padding:12px}.notice{background:#e8f2ff;border-left:4px solid #2474d2;padding:10px 14px}.meta{display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:10px 2px}.meta #status{padding:4px 10px;border-radius:12px;background:#ffe9ad}.meta #reasons{color:#6d3900}.panels{display:grid;grid-template-columns:repeat(2,minmax(480px,1fr));gap:10px}.panel{background:white;border:1px solid #b9c7d5;border-radius:8px;overflow:hidden}.panel h3{margin:0;padding:8px 12px;background:#f3f6fa;display:flex;justify-content:space-between}.viewport{height:350px;background:#fff;cursor:grab;touch-action:none}.viewport.dragging{cursor:grabbing}.viewport canvas{width:100%;height:100%;display:block}.decision{margin-top:12px;background:white;border:1px solid #b9c7d5;border-radius:8px;padding:12px}.decision h2{margin:0 0 10px}.choices{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px}.decision>label{display:block;margin:10px 0}.decision textarea{display:block;width:100%;margin-top:5px}.actions{display:flex;gap:10px}.actions #submit{background:#1769aa;color:white}.actions #revise{background:#fff4df}.actions #resetView{margin-left:auto}#message.ok{color:#08752c}#message.error{color:#b00020}@media(max-width:1050px){header{position:static;display:block}.nav{margin-top:10px}.panels{grid-template-columns:1fr}.viewport{height:310px}}
'''


JS = r'''
(()=>{
  const data=window.ADJUDICATION_DATA, key=`adjudication:${data.build_version}`, colors={A:"#e94444",B:"#1678df",C:"#15a764"};
  let index=0, store=JSON.parse(localStorage.getItem(key)||"{}"), renderers=[];
  const $=id=>document.getElementById(id), save=()=>localStorage.setItem(key,JSON.stringify(store));
  function download(name,text,type){const a=document.createElement("a"),u=URL.createObjectURL(new Blob([text],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
  function csvCell(v){const s=v??"";return /[",\r\n]/.test(String(s))?`"${String(s).replaceAll('"','""')}"`:String(s)}
  function renderer(viewport,item,set){const canvas=document.createElement("canvas"),ctx=canvas.getContext("2d"),img=new Image();viewport.appendChild(canvas);let zoom=1,panX=0,panY=0,drag=null;img.src=item.image;
    function draw(){const box=viewport.getBoundingClientRect(),ratio=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(box.width*ratio));canvas.height=Math.max(1,Math.round(box.height*ratio));ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,box.width,box.height);if(!img.complete||!img.naturalWidth)return;const fit=Math.min(box.width/img.naturalWidth,box.height/img.naturalHeight),scale=fit*zoom,ox=(box.width-img.naturalWidth*scale)/2+panX,oy=(box.height-img.naturalHeight*scale)/2+panY;ctx.drawImage(img,ox,oy,img.naturalWidth*scale,img.naturalHeight*scale);if(set){ctx.lineCap="round";ctx.lineJoin="round";for(const t of set.traces){ctx.beginPath();t.points.forEach((p,i)=>{const x=ox+p[0]*scale,y=oy+p[1]*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=colors[set.label];ctx.lineWidth=Math.max(2,2.2*zoom);ctx.stroke();const first=t.points[0],last=t.points[t.points.length-1];ctx.fillStyle=colors[set.label];ctx.beginPath();ctx.arc(ox+first[0]*scale,oy+first[1]*scale,Math.max(3,3.2*zoom),0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(ox+last[0]*scale,oy+last[1]*scale,Math.max(4,4*zoom),0,Math.PI*2);ctx.fill();}}
    }
    img.onload=draw;new ResizeObserver(draw).observe(viewport);viewport.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(1,Math.min(8,zoom*(e.deltaY<0?1.18:1/1.18)));draw()},{passive:false});viewport.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY};viewport.setPointerCapture(e.pointerId);viewport.classList.add("dragging")});viewport.addEventListener("pointermove",e=>{if(!drag)return;panX=drag.px+e.clientX-drag.x;panY=drag.py+e.clientY-drag.y;draw()});viewport.addEventListener("pointerup",()=>{drag=null;viewport.classList.remove("dragging")});viewport.addEventListener("dblclick",()=>{zoom=1;panX=panY=0;draw()});return{reset(){zoom=1;panX=panY=0;draw()}}}
  function panel(title,item,set){const box=document.createElement("article");box.className="panel";box.innerHTML=`<h3><span>${title}</span><small>${set?set.trace_count+" 条候选路径":"无标注"}</small></h3><div class="viewport"></div>`;$("panels").appendChild(box);renderers.push(renderer(box.querySelector(".viewport"),item,set))}
  function render(){const item=data.items[index],record=store[item.adjudication_pair_id]||{};renderers=[];$("panels").innerHTML="";$("counter").textContent=`${index+1} / ${data.item_count}`;$("pair").textContent=item.adjudication_pair_id;$("status").textContent=record.submitted?`已提交 revision ${record.revision}`:"待裁决";$("reasons").textContent=`触发复核：${item.reason_categories.join("；")}`;panel("① 无标注原图",item,null);for(const set of item.sets)panel(`候选集 ${set.label}`,item,set);document.querySelectorAll('input[name="decision"]').forEach(x=>x.checked=x.value===record.decision);$("confidence").value=record.confidence||"";$("notes").value=record.notes||"";document.querySelectorAll('input[name="decision"],#confidence,#notes').forEach(x=>x.disabled=!!record.submitted);$("submit").disabled=!!record.submitted;$("revise").disabled=!record.submitted;$("message").textContent="";}
  function message(text,type){$("message").textContent=text;$("message").className=type}
  $("prev").onclick=()=>{index=Math.max(0,index-1);render()};$("next").onclick=()=>{index=Math.min(data.items.length-1,index+1);render()};$("resetView").onclick=()=>renderers.forEach(x=>x.reset());
  $("submit").onclick=()=>{const item=data.items[index],decision=document.querySelector('input[name="decision"]:checked')?.value,confidence=$("confidence").value,notes=$("notes").value.trim();if(!decision||!confidence)return message("请选择裁决结果和信心。","error");if(decision==="needs_custom_redraw"&&!notes)return message("需要重新描迹时请在备注中说明具体问题。","error");const prior=store[item.adjudication_pair_id];store[item.adjudication_pair_id]={decision,confidence,notes,submitted:true,revision:(prior?.revision||0)+1,submitted_at_utc:new Date().toISOString()};save();render();message("本图已提交。","ok")};
  $("revise").onclick=()=>{const item=data.items[index];if(!confirm("创建新修订？原决定仍保留在导出历史说明中。"))return;store[item.adjudication_pair_id]={...store[item.adjudication_pair_id],submitted:false};save();render()};
  $("backup").onclick=()=>download("anonymous_adjudication_progress.json",JSON.stringify({build_version:data.build_version,records:store},null,2),"application/json");
  $("export").onclick=()=>{const missing=data.items.filter(item=>!store[item.adjudication_pair_id]?.submitted);if(missing.length)return message(`仍有 ${missing.length} 张未提交。`,"error");const rows=data.items.map(item=>({build_version:data.build_version,adjudication_pair_id:item.adjudication_pair_id,decision:store[item.adjudication_pair_id].decision,confidence:store[item.adjudication_pair_id].confidence,revision:store[item.adjudication_pair_id].revision,submitted_at_utc:store[item.adjudication_pair_id].submitted_at_utc,notes:store[item.adjudication_pair_id].notes})),headers=Object.keys(rows[0]),csv="\ufeff"+[headers.join(","),...rows.map(row=>headers.map(h=>csvCell(row[h])).join(","))].join("\r\n")+"\r\n";download("anonymous_adjudication_decisions.csv",csv,"text/csv;charset=utf-8");download("anonymous_adjudication_decisions.json",JSON.stringify({build_version:data.build_version,exported_at_utc:new Date().toISOString(),records:rows},null,2),"application/json");message("已生成 CSV 和 JSON，请将两份文件一起保存。","ok")};
  render();
})();
'''


if __name__ == "__main__":
    main()

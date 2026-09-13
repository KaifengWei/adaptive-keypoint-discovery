"""Build a read-only, administrator-controlled side-by-side consensus review.

This compares two independent human opinions. It never selects a final trace or
freezes GT. All generated files remain below the Git-ignored runtime directory.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from build import PROTOCOL, digest, read_json, validate_csv_companion, validate_geometry


def classify_case(r1_set: dict, r2_record: dict) -> list[str]:
    selected = [leaf["geometry_choice"] for leaf in r2_record["leaves"]
                if ":" in leaf["geometry_choice"]]
    flags = []
    if len(selected) != len(r2_record["leaves"]):
        flags.append("待重描或暂存争议")
    if len(r1_set["traces"]) != len(r2_record["leaves"]):
        flags.append("结构数量不同")
    sources = {choice.split(":", 1)[0] for choice in selected}
    if len(sources) > 1:
        flags.append("第二位测量者混用候选来源")
    if sources and sources != {r1_set["label"]}:
        flags.append("两人候选来源不完全一致")
    return flags


def curve_svg(trace: dict, color: str, label: str) -> str:
    points = trace["points"]
    coords = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in points)
    # The frozen trace samples run from the shared shoot base to the leaf tip.
    tip_x, tip_y = points[-1]
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="2.5" vector-effect="non-scaling-stroke" '
        'stroke-linecap="round" stroke-linejoin="round" />'
        f'<circle cx="{float(tip_x):.2f}" cy="{float(tip_y):.2f}" r="4" '
        f'fill="{color}" vector-effect="non-scaling-stroke" />'
        f'<text x="{float(tip_x) + 7:.2f}" y="{float(tip_y) - 7:.2f}" '
        f'fill="{color}" font-size="15" font-weight="700">{html.escape(label)}</text>'
    )


def panel(item: dict, label: str, overlays: list[str]) -> str:
    width, height = item["width"], item["height"]
    image = html.escape(item["image"], quote=True)
    return (
        '<div class="panel"><h3>' + html.escape(label) + '</h3>'
        '<div class="viewport">'
        f'<svg class="plant" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<image href="{image}" x="0" y="0" width="{width}" height="{height}" />'
        + "".join(overlays) + '</svg></div></div>'
    )


def render_case(number: int, item: dict, r1_set: dict, r2_record: dict, flags: list[str]) -> str:
    colors = ("#0969da", "#d1246a", "#16803c", "#8b5cf6", "#a45b00")
    r1_overlay = [curve_svg(trace, "#d97706", trace["candidate_trace_id"])
                  for trace in r1_set["traces"]]
    lookup = {f'{s["label"]}:{t["candidate_trace_id"]}': t
              for s in item["sets"] for t in s["traces"]}
    r2_overlay = []
    rows = []
    for index, leaf in enumerate(r2_record["leaves"], 1):
        choice = leaf["geometry_choice"]
        color = colors[(index - 1) % len(colors)]
        if choice in lookup:
            r2_overlay.append(curve_svg(lookup[choice], color, str(index)))
        x, y = leaf["point"]
        r2_overlay.append(
            f'<circle cx="{float(x):.2f}" cy="{float(y):.2f}" r="7" '
            f'fill="none" stroke="{color}" stroke-width="2.5" '
            'vector-effect="non-scaling-stroke" />'
        )
        note = leaf.get("geometry_notes", "") or "—"
        rows.append(
            f'<tr><td style="color:{color}">{index}</td>'
            f'<td>{html.escape(leaf["status"])}</td>'
            f'<td>{html.escape(choice)}</td>'
            f'<td>{html.escape(leaf["geometry_confidence"])}</td>'
            f'<td>{html.escape(note)}</td></tr>'
        )
    flag_text = "；".join(flags) if flags else "来源和结构数量初筛一致，仍须人工核查图像与曲线"
    priority = "yes" if flags else "no"
    return (
        f'<article id="sample-{number}" data-priority="{priority}">'
        f'<h2>原第二阶段第 {number} 张 <small>{html.escape(item["blind_id"])}</small></h2>'
        f'<p class="flags">{html.escape(flag_text)}</p>'
        '<div class="panels">'
        + panel(item, "① 未叠线标准化图", [])
        + panel(item, f'② 第一位测量者所选整组（现包 {r1_set["label"]}）', r1_overlay)
        + panel(item, "③ 第二位测量者逐结构建议", r2_overlay)
        + '</div>'
        '<table><thead><tr><th>结构</th><th>第一步语义</th><th>第二步选择</th>'
        '<th>信心</th><th>第二步备注</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
        '<p class="caution">比较的是两人独立意见；左中右图的曲线不自动构成最终GT。'
        '特别核查叶尖是否完整、是否为同一叶、共同基点及多余/漏掉的路径。</p></article>'
    )


def build(runtime: Path, geometry_export: Path, output: Path) -> dict:
    geometry = runtime / "adjudication/semantic_first_20260906/geometry"
    payload = read_json(geometry / "payload.json")
    manifest = read_json(geometry / "package_manifest.json")
    if manifest["stage"] != "geometry" or manifest["item_count"] != 13:
        raise ValueError("Unexpected geometry package")
    for relative, checksum in manifest["files"].items():
        if digest(geometry / relative) != checksum:
            raise ValueError(f"Distributed geometry package changed: {relative}")
    exported = read_json(geometry_export)
    validate_csv_companion(geometry_export, exported)
    validate_geometry(exported, payload)
    if exported["semantic_export_sha256"] != payload["semantic_export_sha256"]:
        raise ValueError("First-stage provenance mismatch")

    admin = runtime / "admin/semantic_first_20260906"
    mapping = read_json(admin / "mapping.json")
    labels = read_json(admin / f'geometry_labels_{payload["semantic_export_sha256"]}.json')
    first = read_json(runtime / "adjudication/opinions/rater1_20260905/anonymous_adjudication_decisions.json")
    if len(first["records"]) != 13:
        raise ValueError("Rater 1 opinion incomplete")
    by_pair = {record["adjudication_pair_id"]: record for record in first["records"]}
    by_blind = {entry["blind_id"]: entry["pair_id"] for entry in mapping["items"]}
    by_r2 = {record["blind_id"]: record for record in exported["records"]}
    if len(by_pair) != 13 or len(by_blind) != 13 or len(by_r2) != 13:
        raise ValueError("Duplicate or missing identities")
    if {item["blind_id"] for item in payload["items"]} != set(by_blind):
        raise ValueError("Geometry image set does not match admin mapping")
    if set(by_blind.values()) != set(by_pair):
        raise ValueError("Rater 1 and Rater 2 image sets differ")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing review: {output}")
    cases = []
    for number, item in enumerate(payload["items"], 1):
        blind = item["blind_id"]
        r1 = by_pair[by_blind[blind]]
        original_label = r1["decision"].removeprefix("accept_candidate_set_")
        if original_label not in {"A", "B", "C"}:
            raise ValueError("Rater 1 did not choose a single candidate set")
        current_label = next((new for new, old in labels[blind].items()
                              if old == original_label), None)
        r1_set = next((candidate for candidate in item["sets"]
                       if candidate["label"] == current_label), None)
        if r1_set is None:
            raise ValueError("Rater 1 selected set missing from geometry package")
        r2 = by_r2[blind]
        flags = classify_case(r1_set, r2)
        cases.append((number, item, r1_set, r2, flags))

    output.mkdir(parents=True)
    (output / "images").mkdir()
    for _, item, _, _, _ in cases:
        source = geometry / item["image"]
        if digest(source) != item["image_sha256"]:
            raise ValueError("Image hash mismatch")
        shutil.copyfile(source, output / item["image"])
    priority = [case for case in cases if case[4]]
    other = [case for case in cases if not case[4]]
    navigation = "".join(
        f'<a href="#sample-{number}">第{number}张</a>'
        for number, _, _, _, _ in priority
    )
    cards = "".join(render_case(*case) for case in priority + other)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>双人路径意见并排复核</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:15px/1.6 Arial,sans-serif}}
header{{position:sticky;top:0;z-index:5;padding:10px 18px;background:#183457;color:white;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
header h1{{margin:0;font-size:18px}}button,select{{padding:5px 9px}}main{{max-width:1750px;margin:auto;padding:16px}}
.intro{{background:white;border:1px solid #d5dfeb;border-radius:8px;padding:12px 16px;margin-bottom:12px}}
.intro a{{margin-right:10px;color:#0754a6}}article{{background:white;border:1px solid #d5dfeb;border-radius:9px;padding:14px;margin:16px 0 28px}}
article h2{{margin:0 0 4px;font-size:20px}}small{{font-size:12px;color:#68768a;font-weight:normal}}
.flags{{background:#fff1dc;padding:6px 9px;border-left:4px solid #d97706}}article[data-priority="no"] .flags{{background:#edf7f1;border-color:#16803c}}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.panel{{border:1px solid #d9e2ec;border-radius:6px;min-width:0}}
.panel h3{{margin:0;padding:6px 9px;font-size:14px;background:#edf3fa}}.viewport{{overflow:auto;height:300px;background:white}}
svg.plant{{display:block;width:100%;height:auto;max-width:none}}table{{border-collapse:collapse;width:100%;margin-top:10px}}
th,td{{border-bottom:1px solid #dce4ee;padding:5px 8px;text-align:left;vertical-align:top}}th{{background:#eff4fa}}
.caution{{color:#59687b;font-size:13px;margin-bottom:0}}body.only-priority article[data-priority="no"]{{display:none}}
@media(max-width:1000px){{.panels{{grid-template-columns:1fr}}.viewport{{height:260px}}}}
</style></head><body><header><h1>双人路径意见并排复核</h1>
<span>13张原锁定图；{len(priority)}张优先讨论，{len(other)}张仍需质量确认</span>
<button id="toggle">显示全部13张</button><label>局部放大 <select id="zoom"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label><span>同一张三栏滚动同步</span></header>
<main><div class="intro"><p>本页只展示独立意见，不替两位测量者裁决。优先讨论不等于其余图已通过。
第一位测量者当时选择的是整组候选；第二位测量者逐结构选择，结构编号仅来自第二位的定位顺序，不能把两边同名路径自动视为同一叶。</p>
<p>优先讨论：{navigation}</p><p>讨论顺序：先确认真实叶片身份与可测性，再核对叶尖、共同基点和路径；候选均错误时保留待重描。任何结果须另行形成双人共识记录并经过最终GT冻结。</p></div>
{cards}</main><script>
document.body.classList.add('only-priority');
const toggle=document.getElementById('toggle');toggle.onclick=()=>{{const showAll=document.body.classList.toggle('only-priority');toggle.textContent=showAll?'显示全部13张':'只看优先讨论';}};
document.getElementById('zoom').onchange=(event)=>{{for(const svg of document.querySelectorAll('svg.plant'))svg.style.width=(Number(event.target.value)*100)+'%';}};
for(const article of document.querySelectorAll('article')){{
  const views=Array.from(article.querySelectorAll('.viewport'));let syncing=false;
  for(const view of views)view.addEventListener('scroll',()=>{{
    if(syncing)return;syncing=true;
    for(const peer of views)if(peer!==view){{peer.scrollLeft=view.scrollLeft;peer.scrollTop=view.scrollTop;}}
    requestAnimationFrame(()=>{{syncing=false;}});
  }});
}}
</script></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    report = {
        "purpose": "read_only_joint_consensus_navigation_not_final_GT",
        "rater2_geometry_json_sha256": digest(geometry_export),
        "rater2_geometry_csv_sha256": digest(geometry_export.with_suffix(".csv")),
        "rater1_opinion_json_sha256": digest(runtime / "adjudication/opinions/rater1_20260905/anonymous_adjudication_decisions.json"),
        "items": 13, "priority_items": len(priority), "quality_check_items": len(other),
        "pending_structures": sum(leaf["geometry_choice"] in {"needs_redraw", "hold_uncertain"}
                                  for _, _, _, record, _ in cases for leaf in record["leaves"]),
        "cases": [{"page": number, "blind_id": item["blind_id"], "flags": flags,
                   "r1_set_path_count": len(r1_set["traces"]),
                   "r2_structure_count": len(r2["leaves"])}
                  for number, item, r1_set, r2, flags in cases],
        "final_GT": False,
    }
    (output / "review_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "priority_items": len(priority),
            "quality_check_items": len(other), "pending_structures": report["pending_structures"], "final_GT": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=PROTOCOL / "runtime")
    parser.add_argument("--geometry-export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export = args.geometry_export or args.runtime / "adjudication/semantic_first_20260906/geometry/results/geometry_decisions.json"
    target = args.output or args.runtime / "adjudication/semantic_first_20260906/consensus_readonly_20260913"
    print(json.dumps(build(args.runtime, export, target), ensure_ascii=False, indent=2))

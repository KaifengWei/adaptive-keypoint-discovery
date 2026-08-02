#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from html.parser import HTMLParser
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROJECT = EXPERIMENT.parent


class ImageReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        values = dict(attrs)
        if values.get("src"):
            self.sources.append(str(values["src"]))


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_files(source: Path, destination: Path) -> None:
    for item in source.rglob("*"):
        if item.is_file():
            copy_file(item, destination / item.relative_to(source))


def blank_review_csv(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for row in rows:
        for field in fieldnames:
            if field != "dataset_id":
                row[field] = ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_html_images(html_file: Path) -> tuple[int, list[str]]:
    parser = ImageReferenceParser()
    parser.feed(html_file.read_text(encoding="utf-8"))
    missing = [
        source
        for source in parser.sources
        if not (html_file.parent / source).resolve().exists()
    ]
    return len(parser.sources), missing


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    manifest = root / "文件完整性清单.csv"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])


def write_factorized_comparison_html(path: Path, dataset_ids: list[str]) -> None:
    variants = [
        ("A：路线B教师 + 全局尺度解码", "a", "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_val/overlays"),
        ("B：路线B教师 + 局部尺度解码", "b", "evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/overlays"),
        ("C：结构覆盖增强教师 + 全局尺度解码", "c", "evaluation_outputs/point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val/overlays"),
        ("D：结构覆盖增强教师 + 局部尺度解码", "d", "evaluation_outputs/point_conditioned_organ_paths_v3_structure_coverage_val/overlays"),
    ]
    count_options = "".join(
        f'<option value="{value}">{value}</option>'
        for value in ["", "0", "1", "2", "3", "4", "5", "6+", "uncertain"]
    )
    connection_options = "".join(
        f'<option value="{value}">{label}</option>'
        for value, label in [("", ""), ("no", "无错连"), ("yes", "有错连"), ("uncertain", "不确定")]
    )
    base_options = "".join(
        f'<option value="{value}">{label}</option>'
        for value, label in [("", ""), ("correct", "正确"), ("wrong", "错误"), ("uncertain", "不确定")]
    )

    def variant_figure(dataset_id: str, label: str, prefix: str, source: str) -> str:
        return f'''<figure class="variant-panel" data-variant="{prefix.upper()}">
<figcaption>{label}</figcaption><a href="{source}" target="_blank"><img src="{source}" alt="{dataset_id} {label}"></a>
<div class="variant-fields">
<label>正确恢复真实器官路径数 <select data-field="{prefix}_true_path_count">{count_options}</select></label>
<label>漏叶数 <select data-field="{prefix}_missing_leaf_count">{count_options}</select></label>
<label>新增假枝数 <select data-field="{prefix}_false_branch_count">{count_options}</select></label>
<label>错连 <select data-field="{prefix}_wrong_connection">{connection_options}</select></label>
<label>基部 <select data-field="{prefix}_base_selection">{base_options}</select></label>
</div></figure>'''

    cards = []
    for index, dataset_id in enumerate(dataset_ids):
        original_source = f"data_stage_clean_v4_fullplant_candidate/images/val/{dataset_id}.png"
        original_figure = (
            f'<figure class="original-panel"><figcaption>原图：先确认真实苗体，再看四种结果</figcaption>'
            f'<a href="{original_source}" target="_blank"><img src="{original_source}" alt="{dataset_id} 原图"></a>'
            f'<label class="visible-count">原图可辨认叶片数（辅助字段） <select data-field="visible_leaf_count">{count_options}</select></label></figure>'
        )
        variant_figures = "".join(
            variant_figure(dataset_id, label, prefix, f"{folder}/{dataset_id}.png")
            for label, prefix, folder in variants
        )
        cards.append(
            f'''<section class="card{' active' if index == 0 else ''}" data-id="{dataset_id}">
<h2>{dataset_id}</h2><div class="panels">{original_figure}{variant_figures}</div>
<div class="comparison-fields"><h3>配对判断</h3>
<label>每图最佳方案 <select data-field="preferred_variant"><option></option><option>A</option><option>B</option><option>C</option><option>D</option><option>tie</option><option>none</option><option>uncertain</option></select></label>
<label>A vs B（只换Decoder） <select data-field="a_vs_b_decoder"><option></option><option value="A_better">A更好</option><option value="B_better">B更好</option><option value="same">相同</option><option value="uncertain">不确定</option></select></label>
<label>A vs C（只换Teacher） <select data-field="a_vs_c_teacher"><option></option><option value="A_better">A更好</option><option value="C_better">C更好</option><option value="same">相同</option><option value="uncertain">不确定</option></select></label>
<label>C vs D（只换Decoder） <select data-field="c_vs_d_decoder"><option></option><option value="C_better">C更好</option><option value="D_better">D更好</option><option value="same">相同</option><option value="uncertain">不确定</option></select></label>
<label>B vs D（只换Teacher） <select data-field="b_vs_d_teacher"><option></option><option value="B_better">B更好</option><option value="D_better">D更好</option><option value="same">相同</option><option value="uncertain">不确定</option></select></label>
<label>D相对A（完整组合变化） <select data-field="d_vs_a"><option></option><option value="improved">改善</option><option value="same">相同</option><option value="worse">变差</option><option value="uncertain">不确定</option></select></label>
<label class="note">备注 <input data-field="comparison_note" placeholder="例如：B补回上方短叶，但新增一条假枝"></label>
</div></section>'''
        )

    export_fields = [
        "dataset_id", "visible_leaf_count",
        *[f"{prefix}_{field}" for prefix in ("a", "b", "c", "d") for field in (
            "true_path_count", "missing_leaf_count", "false_branch_count", "wrong_connection", "base_selection"
        )],
        "preferred_variant", "a_vs_b_decoder", "a_vs_c_teacher", "c_vs_d_decoder",
        "b_vs_d_teacher", "d_vs_a", "comparison_note",
    ]
    storage_key = "adaptive-kp-factorized-review-v2"
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/B/C/D教师×解码器最终配对复核</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f3f4f6;color:#111827}}
header{{position:sticky;top:0;background:white;padding:10px 18px;box-shadow:0 2px 8px #0002;z-index:2}}button{{margin-right:8px;padding:7px 12px}}
main{{max-width:1760px;margin:auto;padding:16px}}.card{{display:none;background:white;border-radius:10px;padding:14px}}.card.active{{display:block}}
.panels{{display:grid;grid-template-columns:repeat(2,minmax(520px,1fr));gap:12px}}figure{{margin:0;border:1px solid #d1d5db;border-radius:8px;padding:8px}}
figcaption{{font-weight:650;margin-bottom:6px}}.original-panel{{grid-column:1/-1}}figure img{{width:100%;height:42vh;min-height:300px;object-fit:contain;background:white}}
.original-panel img{{height:32vh}}.visible-count{{margin-top:8px;max-width:440px}}.variant-fields{{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:8px;margin-top:8px;padding-top:8px;border-top:1px dashed #d1d5db}}
.comparison-fields{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px;padding:14px;background:#f8fafc;border-radius:8px}}.comparison-fields h3{{grid-column:1/-1;margin:0 0 4px}}
label{{display:flex;gap:8px;align-items:center}}select,input{{flex:1;padding:6px;min-width:0}}.note{{grid-column:1/-1}}.warning{{padding:9px 12px;background:#fff7df;border-left:4px solid #e5a000;margin-bottom:12px}}
@media(max-width:1100px){{.panels{{grid-template-columns:1fr}}.original-panel{{grid-column:auto}}.comparison-fields,.variant-fields{{grid-template-columns:1fr}}}}
</style></head><body><header><button id="prev">上一张</button><button id="next">下一张</button><button id="export">导出CSV</button><span id="position"></span><span id="progress"></span></header>
<main><div class="warning"><strong>冻结期最终配对复核：</strong>四个方案没有预设优胜者。先看原图，再分别记录A–D的真实路径、漏叶、假枝、错连和基部，最后填写四组配对判断。人工结果只用于评价，不回流为关键点坐标标签。</div>
{''.join(cards)}</main><script>
var cards=Array.prototype.slice.call(document.querySelectorAll('.card'));var key={json.dumps(storage_key)};var fields={json.dumps(export_fields[1:], ensure_ascii=False)};var current=0;var saved={{}};
try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')||{{}};}}catch(error){{saved={{}};}}function persist(){{try{{localStorage.setItem(key,JSON.stringify(saved));}}catch(error){{}}}}
function updateProgress(){{var required=fields.filter(function(field){{return field!=='comparison_note'&&field!=='visible_leaf_count';}});var complete=cards.filter(function(card){{var row=saved[card.getAttribute('data-id')]||{{}};return required.every(function(field){{return row[field]!==undefined&&row[field]!=='';}});}}).length;document.querySelector('#progress').textContent=' · 核心字段完成 '+complete+' / '+cards.length;}}
function show(i){{if(!cards.length)return;current=(i+cards.length)%cards.length;cards.forEach(function(c,j){{c.classList.toggle('active',j===current);}});document.querySelector('#position').textContent=' '+(current+1)+' / '+cards.length+' · '+cards[current].getAttribute('data-id');updateProgress();}}
cards.forEach(function(card){{Array.prototype.slice.call(card.querySelectorAll('[data-field]')).forEach(function(input){{var id=card.getAttribute('data-id'),field=input.getAttribute('data-field');if(saved[id]&&saved[id][field]!==undefined)input.value=saved[id][field];input.onchange=function(){{if(!saved[id])saved[id]={{}};saved[id][field]=input.value;persist();updateProgress();}};}});}});
document.querySelector('#prev').onclick=function(){{show(current-1);}};document.querySelector('#next').onclick=function(){{show(current+1);}};document.onkeydown=function(event){{if(event.key==='ArrowLeft')show(current-1);if(event.key==='ArrowRight')show(current+1);}};
document.querySelector('#export').onclick=function(){{var lines=[{json.dumps(export_fields, ensure_ascii=False)}.join(',')];cards.forEach(function(card){{var id=card.getAttribute('data-id'),row=saved[id]||{{}},values=[id];fields.forEach(function(field){{values.push(row[field]||'');}});values=values.map(function(value){{return '"'+String(value).replace(/"/g,'""')+'"';}});lines.push(values.join(','));}});var blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv'}});var link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='factorized_method_review_completed.csv';link.click();}};show(0);
</script></body></html>'''
    path.write_text(document, encoding="utf-8")


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    roi_source = EXPERIMENT / "evaluation_outputs" / "phenotype_roi_basal_anchor_v1_val"
    path_source = (
        EXPERIMENT
        / "evaluation_outputs"
        / "point_conditioned_organ_paths_v2_phenotype_roi_val"
    )
    roi_target = output / "evaluation_outputs" / roi_source.name
    path_target = output / "evaluation_outputs" / path_source.name
    route_c_source = (
        EXPERIMENT
        / "evaluation_outputs"
        / "point_conditioned_organ_paths_v3_structure_coverage_val"
    )
    route_c_target = output / "evaluation_outputs" / route_c_source.name

    copy_file(next(roi_source.glob("*.html")), roi_target / next(roi_source.glob("*.html")).name)
    copy_tree_files(roi_source / "overlays", roi_target / "overlays")
    copy_file(next(path_source.glob("*.html")), path_target / next(path_source.glob("*.html")).name)
    copy_tree_files(path_source / "overlays", path_target / "overlays")
    copy_file(next(route_c_source.glob("*.html")), route_c_target / next(route_c_source.glob("*.html")).name)
    copy_tree_files(route_c_source / "overlays", route_c_target / "overlays")
    for diagnostic_name in [
        "point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val",
        "point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val",
    ]:
        source = EXPERIMENT / "evaluation_outputs" / diagnostic_name
        copy_tree_files(
            source / "overlays",
            output / "evaluation_outputs" / diagnostic_name / "overlays",
        )
    copy_tree_files(
        EXPERIMENT / "data_stage_clean_v4_fullplant_candidate" / "images" / "val",
        output / "data_stage_clean_v4_fullplant_candidate" / "images" / "val",
    )

    for file_name in [
        "README.md",
        "人工复核体系与协作说明.md",
        "判定标准速查表.md",
        "下一阶段改进与复核计划.md",
        "同事反馈记录模板.md",
        "打开复核平台.html",
        "analyze_factorized_review.py",
    ]:
        copy_file(HERE / file_name, output / file_name)

    examples = output / "复核表与示例"
    copy_file(
        roi_source / "manual_phenotype_roi_review_completed.csv",
        examples / "有效域复核_用户完成示例.csv",
    )
    copy_file(
        path_source / "manual_path_review_completed.csv",
        examples / "路径复核_用户完成示例.csv",
    )
    blank_review_csv(
        roi_source / "manual_phenotype_roi_review_completed.csv",
        examples / "有效域复核_空白模板.csv",
    )
    blank_review_csv(
        path_source / "manual_path_review_completed.csv",
        examples / "路径复核_空白模板.csv",
    )
    blank_review_csv(
        route_c_source / "manual_path_review_pending.csv",
        examples / "方案C路径复核_空白模板.csv",
    )
    for source_name, target_name in [
        ("manual_path_review_summary.json", "路径复核统计摘要.json"),
        ("threshold_scan_summary_20260723.csv", "阈值扫描摘要.csv"),
    ]:
        copy_file(path_source / source_name, examples / target_name)

    reports = output / "项目说明与阶段报告"
    report_sources = [
        PROJECT / "项目核心概念与论文术语说明.md",
        EXPERIMENT / "地上部表型有效域与基部过渡区_v1验证报告_20260722.md",
        EXPERIMENT / "地上部有效域路线B首轮训练与val对照报告_20260722.md",
        EXPERIMENT / "路线B人工路径复核与下一轮改进建议_20260723.md",
        EXPERIMENT / "结构覆盖增强方案C首轮训练与二维拆分结果_20260723.md",
        EXPERIMENT / "当前研究路线证据审计与下一阶段冻结计划_20260802.md",
    ]
    for source in report_sources:
        copy_file(source, reports / source.name)

    with (route_c_source / "manual_path_review_pending.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        # Neutral dataset-id order avoids placing known failure cases first and
        # priming the final paired review with the earlier priority ranking.
        dataset_ids = sorted(row["dataset_id"] for row in csv.DictReader(handle))
    write_factorized_comparison_html(output / "方法二维拆分对照.html", dataset_ids)

    html_files = sorted(output.rglob("*.html"))
    validation_rows = []
    for html_file in html_files:
        reference_count, missing = validate_html_images(html_file)
        validation_rows.append(
            {
                "html": html_file.relative_to(output).as_posix(),
                "image_references": reference_count,
                "missing_images": len(missing),
            }
        )
        if missing:
            raise RuntimeError(f"{html_file} has missing image references: {missing[:5]}")

    with (output / "HTML完整性检查.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["html", "image_references", "missing_images"]
        )
        writer.writeheader()
        writer.writerows(validation_rows)
    write_manifest(output)


def make_zip(source: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(source.name) / path.relative_to(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.output.resolve())
    if args.zip_path:
        make_zip(args.output.resolve(), args.zip_path.resolve())
    print(f"package={args.output.resolve()}")
    print(f"zip={args.zip_path.resolve() if args.zip_path else 'not_requested'}")


if __name__ == "__main__":
    main()

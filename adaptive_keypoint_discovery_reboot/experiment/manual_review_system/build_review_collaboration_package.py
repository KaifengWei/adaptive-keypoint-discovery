#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
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

    copy_file(next(roi_source.glob("*.html")), roi_target / next(roi_source.glob("*.html")).name)
    copy_tree_files(roi_source / "overlays", roi_target / "overlays")
    copy_file(next(path_source.glob("*.html")), path_target / next(path_source.glob("*.html")).name)
    copy_tree_files(path_source / "overlays", path_target / "overlays")
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
    ]
    for source in report_sources:
        copy_file(source, reports / source.name)

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

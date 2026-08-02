#!/usr/bin/env python
"""Validate and summarize the final A/B/C/D paired human review.

This script only analyzes exported human evaluation fields. It never reads
keypoint coordinates, model checkpoints, or the locked V4 test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = ("a", "b", "c", "d")
COUNT_FIELDS = ("true_path_count", "missing_leaf_count", "false_branch_count")
CATEGORICAL_FIELDS = ("wrong_connection", "base_selection")
PAIR_FIELDS = (
    "a_vs_b_decoder",
    "a_vs_c_teacher",
    "c_vs_d_decoder",
    "b_vs_d_teacher",
    "d_vs_a",
)
REQUIRED_FIELDS = (
    *[f"{variant}_{field}" for variant in VARIANTS for field in (*COUNT_FIELDS, *CATEGORICAL_FIELDS)],
    "preferred_variant",
    *PAIR_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--expected-images", type=int, default=40)
    return parser.parse_args()


def count_values(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "").strip() or "blank" for row in rows).items()))


def numeric_lower_bound(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    exact_total = 0
    lower_bound_total = 0
    non_exact = Counter()
    for row in rows:
        value = row.get(field, "").strip()
        if value.isdigit():
            exact_total += int(value)
            lower_bound_total += int(value)
        elif value.endswith("+") and value[:-1].isdigit():
            lower_bound_total += int(value[:-1])
            non_exact[value] += 1
        else:
            non_exact[value or "blank"] += 1
    return {
        "exact_total_from_exact_rows": exact_total,
        "lower_bound_total_all_countable_rows": lower_bound_total,
        "non_exact_values": dict(sorted(non_exact.items())),
    }


def load_and_validate(path: Path, expected_images: int) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    if len(rows) != expected_images:
        errors.append(f"expected {expected_images} rows, found {len(rows)}")
    ids = [row.get("dataset_id", "").strip() for row in rows]
    blank_ids = sum(not value for value in ids)
    if blank_ids:
        errors.append(f"blank dataset_id rows: {blank_ids}")
    duplicates = sorted(value for value, count in Counter(ids).items() if value and count > 1)
    if duplicates:
        errors.append(f"duplicate dataset_id values: {duplicates}")
    if rows:
        missing_columns = [field for field in ("dataset_id", *REQUIRED_FIELDS) if field not in rows[0]]
        if missing_columns:
            errors.append(f"missing columns: {missing_columns}")
        for field in REQUIRED_FIELDS:
            if field in rows[0]:
                missing = [row.get("dataset_id", "") for row in rows if not row.get(field, "").strip()]
                if missing:
                    errors.append(f"blank required field {field}: {len(missing)} rows")
    return rows, errors


def summarize(rows: list[dict[str, str]], errors: list[str], source: Path) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        variants[variant.upper()] = {
            field: numeric_lower_bound(rows, f"{variant}_{field}") for field in COUNT_FIELDS
        }
        variants[variant.upper()].update(
            {field: count_values(rows, f"{variant}_{field}") for field in CATEGORICAL_FIELDS}
        )
    return {
        "source_csv": str(source.resolve()),
        "images": len(rows),
        "validation_status": "pass" if not errors else "fail",
        "validation_errors": errors,
        "best_method": count_values(rows, "preferred_variant"),
        "paired_comparisons": {field: count_values(rows, field) for field in PAIR_FIELDS},
        "variant_outcomes": variants,
        "manual_keypoint_coordinates_created": False,
        "use": "path-semantics evaluation only; never training supervision",
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A/B/C/D 人工配对复核统计",
        "",
        f"- 图像：{summary['images']}；",
        f"- 完整性：`{summary['validation_status']}`；",
        "- 用途：只作路径语义评价，不作为关键点训练标签。",
        "",
        "## 每图最佳方案",
        "",
        "| 选择 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in summary["best_method"].items())
    lines.extend(["", "## 四组配对比较", ""])
    for field, counts in summary["paired_comparisons"].items():
        lines.append(f"- `{field}`：" + "，".join(f"{key}={value}" for key, value in counts.items()))
    lines.extend(["", "## 各方案错误审计", ""])
    for variant, values in summary["variant_outcomes"].items():
        missing = values["missing_leaf_count"]["lower_bound_total_all_countable_rows"]
        false_branch = values["false_branch_count"]["lower_bound_total_all_countable_rows"]
        wrong = values["wrong_connection"].get("yes", 0)
        base_wrong = values["base_selection"].get("wrong", 0)
        lines.append(
            f"- {variant}：漏叶下界={missing}，假枝下界={false_branch}，错连图={wrong}，基部错误图={base_wrong}。"
        )
    if summary["validation_errors"]:
        lines.extend(["", "## 完整性错误", ""])
        lines.extend(f"- {error}" for error in summary["validation_errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, errors = load_and_validate(args.review_csv, args.expected_images)
    summary = summarize(rows, errors, args.review_csv)
    output_json = args.output_json or args.review_csv.with_name("factorized_method_review_summary.json")
    output_md = args.output_md or args.review_csv.with_name("factorized_method_review_summary.md")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit paired A/B/C/D manual outcomes without touching model outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANT_METRICS = (
    "true_path_count",
    "missing_leaf_count",
    "false_branch_count",
    "wrong_connection",
    "base_selection",
)

PAIRS = (
    ("A", "B", "a_vs_b_decoder"),
    ("A", "C", "a_vs_c_teacher"),
    ("C", "D", "c_vs_d_decoder"),
    ("B", "D", "b_vs_d_teacher"),
)


def _field(variant: str, metric: str) -> str:
    return f"{variant.lower()}_{metric}"


def _integer(row: dict[str, str], variant: str, metric: str) -> int:
    return int(row[_field(variant, metric)])


def analyze_pair(rows: list[dict[str, str]], source: str, target: str, label_field: str) -> dict:
    result = {
        "source": source,
        "target": target,
        "manual_labels": {},
        "true_paths": {"gained_images": [], "lost_images": [], "net": 0},
        "missing_leaves": {"reduced_images": [], "increased_images": [], "net_change": 0},
        "false_branches": {"reduced_images": [], "increased_images": [], "net_change": 0},
        "wrong_connections": {"reduced_images": [], "increased_images": []},
        "base_selection": {"improved_images": [], "worsened_images": []},
        "label_field_disagreements": [],
    }

    for row in rows:
        dataset_id = row["dataset_id"]
        label = row[label_field].strip()
        result["manual_labels"][label] = result["manual_labels"].get(label, 0) + 1

        true_delta = _integer(row, target, "true_path_count") - _integer(row, source, "true_path_count")
        missing_delta = _integer(row, target, "missing_leaf_count") - _integer(row, source, "missing_leaf_count")
        false_delta = _integer(row, target, "false_branch_count") - _integer(row, source, "false_branch_count")
        result["true_paths"]["net"] += true_delta
        result["missing_leaves"]["net_change"] += missing_delta
        result["false_branches"]["net_change"] += false_delta

        if true_delta > 0:
            result["true_paths"]["gained_images"].append([dataset_id, true_delta])
        elif true_delta < 0:
            result["true_paths"]["lost_images"].append([dataset_id, true_delta])

        if missing_delta < 0:
            result["missing_leaves"]["reduced_images"].append([dataset_id, missing_delta])
        elif missing_delta > 0:
            result["missing_leaves"]["increased_images"].append([dataset_id, missing_delta])

        if false_delta < 0:
            result["false_branches"]["reduced_images"].append([dataset_id, false_delta])
        elif false_delta > 0:
            result["false_branches"]["increased_images"].append([dataset_id, false_delta])

        wrong_before = row[_field(source, "wrong_connection")].strip()
        wrong_after = row[_field(target, "wrong_connection")].strip()
        if wrong_before == "yes" and wrong_after == "no":
            result["wrong_connections"]["reduced_images"].append(dataset_id)
        elif wrong_before == "no" and wrong_after == "yes":
            result["wrong_connections"]["increased_images"].append(dataset_id)

        base_before = row[_field(source, "base_selection")].strip()
        base_after = row[_field(target, "base_selection")].strip()
        if base_before == "wrong" and base_after == "correct":
            result["base_selection"]["improved_images"].append(dataset_id)
        elif base_before == "correct" and base_after == "wrong":
            result["base_selection"]["worsened_images"].append(dataset_id)

        fields_equal = all(
            row[_field(source, metric)].strip() == row[_field(target, metric)].strip()
            for metric in VARIANT_METRICS
        )
        if (label == "same" and not fields_equal) or (
            label not in {"same", "uncertain"} and fields_equal
        ):
            result["label_field_disagreements"].append(
                {
                    "dataset_id": dataset_id,
                    "manual_label": label,
                    "fields_equal": fields_equal,
                }
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    source_path = Path(args.review_csv).resolve()
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    report = {
        "source_csv": str(source_path),
        "images": len(rows),
        "pairs": {
            f"{source}_vs_{target}": analyze_pair(rows, source, target, label_field)
            for source, target, label_field in PAIRS
        },
        "comparison_notes": {
            row["dataset_id"]: row["comparison_note"].strip()
            for row in rows
            if row["comparison_note"].strip()
        },
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Reproduce the G1 pilot data-quality attribution analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent


def main() -> None:
    metrics = pd.read_csv(HERE / "outputs_g1" / "pilot" / "per_image_metrics.csv")
    audit = pd.read_csv(HERE / "data_g0" / "pilot80_manifest.csv")
    data = metrics.merge(audit, on="image_id", suffixes=("", "_audit"))
    output = HERE / "outputs_g1" / "pilot" / "data_quality_attribution"
    output.mkdir(parents=True, exist_ok=True)

    group_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    numeric = [
        "blur_laplacian_var",
        "vegetation_fraction",
        "bbox_area_fraction",
        "quality_score",
        "aspect_ratio",
        "border_luminance_std",
        "border_saturation_mean",
    ]
    for method, method_data in data.groupby("method"):
        for background, group in method_data.groupby("background_class"):
            group_rows.append(
                {
                    "method": method,
                    "group_variable": "background_class",
                    "group": background,
                    "n": len(group),
                    "median_repeatability_f1": group["mean_repeatability_f1"].median(),
                    "median_candidate_count": group["candidate_count"].median(),
                    "safety_cap_hit_rate": group["safety_cap_hit"].mean(),
                    "median_g0_mask_agreement": group["plant_hit_ratio"].median(),
                }
            )
        background_groups = [group["mean_repeatability_f1"].to_numpy() for _, group in method_data.groupby("background_class")]
        background_test = kruskal(*background_groups)
        test_rows.append(
            {
                "method": method,
                "test": "kruskal_background_class_vs_repeatability_f1",
                "statistic": background_test.statistic,
                "p_value": background_test.pvalue,
                "interpretation": "association_only_not_causation",
            }
        )
        flagged = method_data["auto_review_flags"].fillna("").ne("")
        flag_test = mannwhitneyu(
            method_data.loc[flagged, "mean_repeatability_f1"],
            method_data.loc[~flagged, "mean_repeatability_f1"],
            alternative="two-sided",
        )
        test_rows.append(
            {
                "method": method,
                "test": "mannwhitney_any_auto_flag_vs_repeatability_f1",
                "statistic": flag_test.statistic,
                "p_value": flag_test.pvalue,
                "interpretation": "audit_flags_are_not_independent_human_quality_labels",
            }
        )
        for variable in numeric:
            result = spearmanr(method_data[variable], method_data["mean_repeatability_f1"], nan_policy="omit")
            test_rows.append(
                {
                    "method": method,
                    "test": f"spearman_{variable}_vs_repeatability_f1",
                    "statistic": result.statistic,
                    "p_value": result.pvalue,
                    "interpretation": "association_only_not_causation",
                }
            )

    pd.DataFrame(group_rows).to_csv(output / "background_group_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(output / "association_tests.csv", index=False, encoding="utf-8-sig")
    summary = {
        "unique_images": int(metrics["image_id"].nunique()),
        "rows": int(len(metrics)),
        "background_counts": audit[audit["image_id"].isin(metrics["image_id"])]["background_class"].value_counts().to_dict(),
        "limits": [
            "G0 quality scores and masks are automatic audit proposals, not independent human ground truth.",
            "The tests identify association, not causality.",
            "Repeatability does not measure structural meaning or organ coverage.",
        ],
    }
    (output / "analysis_scope.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

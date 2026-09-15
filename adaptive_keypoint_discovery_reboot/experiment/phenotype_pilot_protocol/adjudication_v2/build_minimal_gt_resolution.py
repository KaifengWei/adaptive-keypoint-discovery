"""Build and validate the three-case minimal final-GT resolution package.

The package is evaluation-only.  It reduces the previous 13-case consensus
form to the three images with a substantive identity or completeness issue.
It never reads model output and never freezes final GT by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from build import PROTOCOL, digest, read_json
from build_joint_consensus import load_cases, make_package_id

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phenotype_gt_geometry import divergence_angle, polyline_length, resample_trace  # noqa: E402


BUILD_VERSION = "minimal-gt-resolution-v1"
STAGE = "minimal_gt_resolution"
LENGTH_MDC95_PCT = 3.72
ANGLE_MDC95_DEG = 17.08
ISSUE_PAGES = (5, 6, 7)
AUTO_PAGES = (1, 2, 3, 4, 8, 9, 10, 11, 12, 13)
PACKAGE_RE = re.compile(r"^[A-Z2-9]{16}$")

QUESTIONS = {
    5: {
        "title": "只确认一件事：这株苗是3片叶还是4片叶？",
        "detail": "蓝线是双方都认可的3条结构路径，红色虚线是只在一轮描迹中出现的疑似第4片叶。",
        "options": [
            {"code": "three_leaves_exclude_questioned", "label": "3片叶：红色结构不是独立完整叶片"},
            {"code": "four_leaves_use_existing", "label": "4片叶：红色现有路径可以直接使用"},
            {"code": "four_leaves_redraw", "label": "4片叶：但红色路径需要重新描迹"},
            {"code": "cannot_determine", "label": "目前无法确定"},
        ],
    },
    6: {
        "title": "只确认一件事：红色小结构是真实第二片叶，还是叶缘/毛边干扰？",
        "detail": "蓝线是没有争议的主路径，红色虚线仅用于指出争议位置，不代表已被接受。",
        "options": [
            {"code": "one_leaf_questioned_is_artifact", "label": "1片叶：红色结构是叶缘或毛边干扰"},
            {"code": "two_leaves_use_existing", "label": "2片叶：红色现有路径可以直接使用"},
            {"code": "two_leaves_redraw", "label": "2片叶：但红色路径需要重新描迹"},
            {"code": "cannot_determine", "label": "目前无法确定"},
        ],
    },
    7: {
        "title": "只确认一件事：红色短结构是不是一片独立叶片？",
        "detail": "蓝线是已认可路径，红色虚线是身份或完整性仍有争议的短结构。",
        "options": [
            {"code": "two_leaves_exclude_questioned", "label": "2片叶：红色结构不是独立叶片"},
            {"code": "three_leaves_use_existing", "label": "3片叶：红色现有路径可以直接使用"},
            {"code": "three_leaves_redraw", "label": "3片叶：但红色路径需要重新描迹"},
            {"code": "cannot_determine", "label": "目前无法确定"},
        ],
    },
}


def _trace_length(trace: dict[str, Any]) -> float:
    return polyline_length(np.asarray(trace["points"], dtype=np.float64))


def _bbox_diag(case: dict[str, Any]) -> float:
    points = np.asarray(
        [point for candidate in case["sets"] for trace in candidate["traces"] for point in trace["points"]],
        dtype=np.float64,
    )
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _lookup(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f'{candidate["label"]}:{trace["candidate_trace_id"]}': trace
        for candidate in case["sets"]
        for trace in candidate["traces"]
    }


def _r2_selection(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = _lookup(case)
    concrete, unresolved = [], []
    for position, leaf in enumerate(case["r2_record"]["leaves"], 1):
        choice = leaf.get("geometry_choice", "")
        if choice in lookup:
            concrete.append({"position": position, "source_key": choice, "trace": lookup[choice], "leaf": leaf})
        else:
            unresolved.append({"position": position, "choice": choice, "leaf": leaf})
    return concrete, unresolved


def _tip_match(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    cost = np.asarray(
        [[math.dist(a["points"][-1], b["points"][-1]) for b in right] for a in left],
        dtype=np.float64,
    )
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def _symmetric_length_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _trace_length(left), _trace_length(right)
    return 100.0 * abs(a - b) / max((a + b) / 2.0, 1e-12)


def _curve_rms_norm(left: dict[str, Any], right: dict[str, Any], bbox_diag: float) -> float:
    a = np.asarray(left["points"], dtype=np.float64)
    b = np.asarray(right["points"], dtype=np.float64)
    direct = float(np.square(a - b).sum())
    reverse = float(np.square(a - b[::-1]).sum())
    if reverse < direct:
        b = b[::-1]
    rms = float(np.sqrt(np.square(a - b).sum(axis=1).mean()))
    return 100.0 * rms / max(bbox_diag, 1e-12)


def _angle_differences(
    left: list[dict[str, Any]], right: list[dict[str, Any]], bbox_diag: float
) -> list[float]:
    if len(left) < 2 or len(right) < 2:
        return []
    left_main = max(range(len(left)), key=lambda n: _trace_length(left[n]))
    right_main = max(range(len(right)), key=lambda n: _trace_length(right[n]))
    left_branches = [trace for n, trace in enumerate(left) if n != left_main]
    right_branches = [trace for n, trace in enumerate(right) if n != right_main]
    diffs = []
    for li, ri in _tip_match(left_branches, right_branches):
        left_angle = divergence_angle(
            resample_trace(left[left_main]["points"]),
            resample_trace(left_branches[li]["points"]),
            bbox_diag,
        ).get("divergence_angle_deg")
        right_angle = divergence_angle(
            resample_trace(right[right_main]["points"]),
            resample_trace(right_branches[ri]["points"]),
            bbox_diag,
        ).get("divergence_angle_deg")
        if left_angle is not None and right_angle is not None:
            diffs.append(abs(float(left_angle) - float(right_angle)))
    return diffs


def audit_case(case: dict[str, Any]) -> dict[str, Any]:
    r1 = case["r1_set"]["traces"]
    concrete, unresolved = _r2_selection(case)
    r2 = [entry["trace"] for entry in concrete]
    bbox_diag = _bbox_diag(case)
    pairs = _tip_match(r1, r2)
    length_diffs = [_symmetric_length_difference(r1[a], r2[b]) for a, b in pairs]
    rms_diffs = [_curve_rms_norm(r1[a], r2[b], bbox_diag) for a, b in pairs]
    angle_diffs = _angle_differences(r1, r2, bbox_diag)
    exact = (
        len(r1) == len(r2)
        and not unresolved
        and len(pairs) == len(r1)
        and all(np.array_equal(np.asarray(r1[a]["points"]), np.asarray(r2[b]["points"])) for a, b in pairs)
    )
    if len(r1) != len(case["r2_record"]["leaves"]):
        classification, reason = "manual_resolution", "visible_leaf_count_differs"
    elif unresolved:
        classification, reason = "manual_resolution", unresolved[0]["choice"]
    elif len(pairs) != len(r1):
        classification, reason = "manual_resolution", "trace_correspondence_incomplete"
    elif exact:
        classification, reason = "exact_agreement", "same_complete_paths"
    elif max(length_diffs, default=0.0) <= LENGTH_MDC95_PCT and max(angle_diffs, default=0.0) <= ANGLE_MDC95_DEG:
        classification, reason = "phenotype_equivalent", "differences_within_locked_human_measurement_floor"
    else:
        classification, reason = "manual_resolution", "phenotype_difference_exceeds_human_measurement_floor"
    return {
        "original_page": case["original_page"],
        "blind_id": case["blind_id"],
        "classification": classification,
        "reason": reason,
        "rater1_trace_count": len(r1),
        "rater2_structure_count": len(case["r2_record"]["leaves"]),
        "rater2_concrete_path_count": len(r2),
        "max_symmetric_length_difference_pct": max(length_diffs, default=None),
        "max_divergence_angle_difference_deg": max(angle_diffs, default=None),
        "max_curve_rms_pct_bbox_diag": max(rms_diffs, default=None),
        "bbox_diag_px": bbox_diag,
    }


def _curve_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = np.asarray(left["points"], dtype=np.float64)
    b = np.asarray(right["points"], dtype=np.float64)
    direct = np.linalg.norm(a - b, axis=1).mean()
    reverse = np.linalg.norm(a - b[::-1], axis=1).mean()
    return float(min(direct, reverse))


def _candidate_clusters(case: dict[str, Any], anchors: list[dict[str, Any]]) -> list[list[tuple[str, dict[str, Any]]]]:
    clusters: list[list[tuple[str, dict[str, Any]]]] = [[] for _ in anchors]
    for candidate in case["sets"]:
        traces = candidate["traces"]
        for anchor_index, trace_index in _tip_match([entry["trace"] for entry in anchors], traces):
            trace = traces[trace_index]
            clusters[anchor_index].append((f'{candidate["label"]}:{trace["candidate_trace_id"]}', trace))
    return clusters


def _medoid(cluster: list[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    if not cluster:
        raise ValueError("A canonical cluster cannot be empty")
    scored = []
    for key, trace in cluster:
        distance = sum(_curve_distance(trace, other) for _, other in cluster)
        scored.append((distance, key, trace))
    _, key, trace = min(scored, key=lambda item: (round(item[0], 12), item[1]))
    return key, trace


def _points_sha256(points: list[list[float]]) -> str:
    payload = json.dumps(points, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def auto_resolution_plan(case: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    anchors, unresolved = _r2_selection(case)
    if unresolved or audit["classification"] not in {"exact_agreement", "phenotype_equivalent"}:
        raise ValueError("Only automatically resolved cases can receive a canonical plan")
    paths = []
    if audit["classification"] == "exact_agreement":
        chosen = [(entry["source_key"], entry["trace"]) for entry in anchors]
    else:
        chosen = [_medoid(cluster) for cluster in _candidate_clusters(case, anchors)]
    for position, (source_key, trace) in enumerate(chosen, 1):
        paths.append({
            "leaf_position": position,
            "source_key": source_key,
            "points_sha256": _points_sha256(trace["points"]),
            "length_px": _trace_length(trace),
        })
    return {
        "original_page": case["original_page"],
        "blind_id": case["blind_id"],
        "classification": audit["classification"],
        "canonical_rule": "exact_joint_choice_else_existing_whole_trace_medoid",
        "canonical_paths": paths,
        "final_GT": False,
    }


def _unmatched_r1_path(case: dict[str, Any]) -> dict[str, Any] | None:
    concrete, _ = _r2_selection(case)
    r1 = case["r1_set"]["traces"]
    matched = {left for left, _ in _tip_match(r1, [entry["trace"] for entry in concrete])}
    unmatched = [trace for n, trace in enumerate(r1) if n not in matched]
    return unmatched[0] if unmatched else None


def _nearest_path_to_point(case: dict[str, Any], point: list[float]) -> dict[str, Any]:
    candidates = [trace for candidate in case["sets"] for trace in candidate["traces"]]
    return min(candidates, key=lambda trace: math.dist(trace["points"][-1], point))


def make_issue_item(case: dict[str, Any]) -> dict[str, Any]:
    page = case["original_page"]
    concrete, unresolved = _r2_selection(case)
    if page == 5:
        disputed = _unmatched_r1_path(case)
        disputed_point = disputed["points"][-1] if disputed else None
    else:
        if len(unresolved) != 1 or not unresolved[0]["leaf"].get("point"):
            raise ValueError(f"Page {page} does not contain the expected single unresolved structure")
        disputed_point = unresolved[0]["leaf"]["point"]
        disputed = _nearest_path_to_point(case, disputed_point)
    if disputed is None:
        raise ValueError(f"Page {page} has no disputed path to display")
    return {
        "original_page": page,
        "blind_id": case["blind_id"],
        "image": case["image"],
        "image_sha256": case["image_sha256"],
        "width": case["width"],
        "height": case["height"],
        "question": QUESTIONS[page],
        "stable_paths": [entry["trace"]["points"] for entry in concrete],
        "disputed_path": disputed["points"],
        "disputed_point": disputed_point,
    }


def write_admin_plan(runtime: Path, provenance: dict[str, Any], plans: list[dict[str, Any]]) -> tuple[Path, str]:
    content = {
        "build_version": BUILD_VERSION,
        "stage": "auto_equivalence_plan",
        "source_provenance": provenance,
        "thresholds": {
            "length_symmetric_relative_mdc95_pct": LENGTH_MDC95_PCT,
            "divergence_angle_mdc95_deg": ANGLE_MDC95_DEG,
        },
        "canonical_rule": "exact_joint_choice_else_existing_whole_trace_medoid",
        "records": plans,
        "final_GT": False,
    }
    raw = (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    checksum = hashlib.sha256(raw).hexdigest()
    target = runtime / "admin/semantic_first_20260906/minimal_gt_resolution" / f"auto_resolution_plan_{checksum}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("Existing admin plan has unexpected content")
    target.write_bytes(raw)
    return target, checksum


def build(runtime: Path, geometry_export: Path, output: Path, package_id: str | None = None) -> dict[str, Any]:
    cases, provenance = load_cases(runtime, geometry_export)
    audits = [audit_case(case) for case in cases]
    auto_pages = tuple(audit["original_page"] for audit in audits if audit["classification"] != "manual_resolution")
    issue_pages = tuple(audit["original_page"] for audit in audits if audit["classification"] == "manual_resolution")
    if auto_pages != AUTO_PAGES or issue_pages != ISSUE_PAGES:
        raise ValueError(f"Unexpected audit split: auto={auto_pages}, issues={issue_pages}")
    plans = [auto_resolution_plan(case, audit) for case, audit in zip(cases, audits) if audit["original_page"] in AUTO_PAGES]
    _, plan_sha256 = write_admin_plan(runtime, provenance, plans)

    package_id = package_id or make_package_id()
    if not PACKAGE_RE.fullmatch(package_id):
        raise ValueError("Package ID must be a 16-character opaque uppercase ID")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "results").mkdir()
    source_images = runtime / "adjudication/semantic_first_20260906/geometry"
    issue_cases = [case for case in cases if case["original_page"] in ISSUE_PAGES]
    items = [make_issue_item(case) for case in issue_cases]
    for item in items:
        source = source_images / item["image"]
        if digest(source) != item["image_sha256"]:
            raise ValueError("Image hash mismatch")
        shutil.copyfile(source, output / item["image"])

    source_dir = Path(__file__).parent
    shutil.copyfile(source_dir / "minimal_gt_resolution_index.html", output / "index.html")
    shutil.copyfile(source_dir / "minimal_gt_resolution_app.js", output / "app.js")
    shutil.copyfile(source_dir / "minimal_gt_resolution_style.css", output / "style.css")
    public_audits = [{k: v for k, v in audit.items() if k != "blind_id"} for audit in audits]
    public = {
        "build_version": BUILD_VERSION,
        "package_id": package_id,
        "stage": STAGE,
        "source_provenance": {**provenance, "auto_resolution_plan_sha256": plan_sha256},
        "thresholds": {"length_mdc95_pct": LENGTH_MDC95_PCT, "angle_mdc95_deg": ANGLE_MDC95_DEG},
        "auto_resolved_pages": list(AUTO_PAGES),
        "manual_resolution_pages": list(ISSUE_PAGES),
        "audit": public_audits,
        "items": items,
        "final_GT": False,
    }
    (output / "data.js").write_text(
        "window.MINIMAL_GT_DATA=" + json.dumps(public, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (output / "使用说明.txt").write_text(
        "最终GT最小复核（只需3张）\n\n"
        "原13张中，1、2、4、8、12、13完全一致；3、9、10、11的长度和分化角差异均小于已锁定人工误差底线。\n"
        "本页面只保留第5、6、7张真实问题图，不再比较A/B/C，也不要求逐叶打分。\n\n"
        "操作：每张只回答页面上的一个问题，勾选共同确认并提交。完成3张后导出JSON和CSV，一起保存到results。\n"
        "红线只标出争议结构，不表示必须保留。蓝线是当前没有争议的路径。\n"
        "该结果与后台10张自动等效计划合并后，仍须完成必要重描并显式冻结，才成为最终GT。\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    }
    manifest = {
        "build_version": BUILD_VERSION,
        "package_id": package_id,
        "stage": STAGE,
        "item_count": len(items),
        "auto_resolved_count": len(AUTO_PAGES),
        "manual_resolution_count": len(ISSUE_PAGES),
        "source_provenance": public["source_provenance"],
        "files": files,
        "final_GT": False,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "package_id": package_id,
        "auto_resolved_pages": list(AUTO_PAGES),
        "manual_resolution_pages": list(ISSUE_PAGES),
        "auto_resolution_plan_sha256": plan_sha256,
        "initial_records": 0,
        "final_GT": False,
    }


def validate_record(record: dict[str, Any], item: dict[str, Any], require_submitted: bool = True) -> None:
    if record.get("blind_id") != item["blind_id"] or record.get("image_sha256") != item["image_sha256"]:
        raise ValueError("Record identity or image hash mismatch")
    allowed = {option["code"] for option in item["question"]["options"]}
    if record.get("decision") not in allowed:
        raise ValueError("Invalid resolution decision")
    if record["decision"] == "cannot_determine" and not str(record.get("notes", "")).strip():
        raise ValueError("An unresolved decision needs a note")
    if not record.get("jointly_confirmed"):
        raise ValueError("The joint confirmation box is not checked")
    if require_submitted and not record.get("submitted"):
        raise ValueError("Record is not submitted")
    if not isinstance(record.get("history"), list):
        raise ValueError("Revision history must be a list")


def validate_csv_companion(csv_path: Path, payload: dict[str, Any]) -> None:
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    if len(rows) != len(payload["records"]):
        raise ValueError("CSV row count mismatch")
    by_id = {record["blind_id"]: record for record in payload["records"]}
    for row in rows:
        if row.get("blind_id") not in by_id or json.loads(row["record_json"]) != by_id[row["blind_id"]]:
            raise ValueError("CSV and JSON differ")


def validate(package: Path, export: Path | None = None) -> dict[str, Any]:
    manifest = read_json(package / "package_manifest.json")
    if manifest.get("build_version") != BUILD_VERSION or manifest.get("stage") != STAGE or manifest.get("item_count") != 3:
        raise ValueError("Unexpected minimal resolution package")
    for relative, checksum in manifest["files"].items():
        if digest(package / relative) != checksum:
            raise ValueError(f"Package asset changed: {relative}")
    source = (package / "data.js").read_text(encoding="utf-8")
    forbidden = [token for token in ("v4_val_", "dataset_id", "A:T", "B:T", "C:T", "candidate_trace_id") if token in source]
    if forbidden:
        raise ValueError(f"Internal label leak: {forbidden}")
    data = json.loads(source.removeprefix("window.MINIMAL_GT_DATA=").removesuffix(";\n"))
    if data["auto_resolved_pages"] != list(AUTO_PAGES) or data["manual_resolution_pages"] != list(ISSUE_PAGES):
        raise ValueError("Unexpected audit page split")
    report = {
        "package_assets_match": True,
        "internal_label_leak_hits": 0,
        "auto_resolved_count": len(AUTO_PAGES),
        "manual_resolution_count": len(ISSUE_PAGES),
        "initial_final_GT": False,
    }
    if export:
        payload = read_json(export)
        if (
            payload.get("build_version") != BUILD_VERSION
            or payload.get("package_id") != data["package_id"]
            or payload.get("stage") != STAGE
            or payload.get("source_provenance") != data["source_provenance"]
        ):
            raise ValueError("Export does not belong to this package")
        by_id = {item["blind_id"]: item for item in data["items"]}
        records = payload.get("records", [])
        if len(records) != 3 or {record.get("blind_id") for record in records} != set(by_id):
            raise ValueError("Export must contain all three issue images")
        for record in records:
            validate_record(record, by_id[record["blind_id"]])
        validate_csv_companion(export.with_suffix(".csv"), payload)
        report.update({
            "submitted_records": 3,
            "decisions": {code: sum(record["decision"] == code for record in records) for code in sorted({r["decision"] for r in records})},
            "status": "minimal_resolution_complete_pending_redraw_and_final_GT_freeze",
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--runtime", type=Path, default=PROTOCOL / "runtime")
    p_build.add_argument("--geometry-export", type=Path)
    p_build.add_argument("--output", type=Path)
    p_build.add_argument("--package-id")
    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--package", type=Path, required=True)
    p_validate.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        geometry = args.geometry_export or args.runtime / "adjudication/semantic_first_20260906/geometry/results/geometry_decisions.json"
        output = args.output or args.runtime / "adjudication/semantic_first_20260906/minimal_gt_resolution_20260915_v1"
        result = build(args.runtime, geometry, output, args.package_id)
    else:
        result = validate(args.package, args.export)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

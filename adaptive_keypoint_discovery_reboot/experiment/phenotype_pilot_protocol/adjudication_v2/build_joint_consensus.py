"""Build and validate a two-rater joint-consensus capture package.

The package records joint human decisions over frozen human trace candidates.
It neither changes a trace nor promotes the export to final GT.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import secrets
import shutil
from pathlib import Path

from build import PROTOCOL, digest, read_json, validate_csv_companion, validate_geometry
from build_consensus_review import classify_case

BUILD_VERSION = "joint-consensus-capture-v1"
STAGE = "joint_consensus"
PACKAGE_RE = re.compile(r"^[A-Z2-9]{16}$")
DECISIONS = {"consensus_ready", "redraw_required", "third_rater_required"}
SEMANTIC = {"measurable", "visible_unmeasurable", "uncertain"}
SPECIAL_CHOICES = {"needs_redraw", "hold_uncertain", "no_geometry"}
CONFIDENCE = {"high", "medium", "low"}


def make_package_id() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(16))


def load_cases(runtime: Path, geometry_export: Path) -> tuple[list[dict], dict]:
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
    r1_path = runtime / "adjudication/opinions/rater1_20260905/anonymous_adjudication_decisions.json"
    first = read_json(r1_path)
    by_pair = {record["adjudication_pair_id"]: record for record in first["records"]}
    by_blind = {entry["blind_id"]: entry["pair_id"] for entry in mapping["items"]}
    by_r2 = {record["blind_id"]: record for record in exported["records"]}
    if len(by_pair) != 13 or len(by_blind) != 13 or len(by_r2) != 13:
        raise ValueError("Duplicate or missing identities")
    if set(by_blind.values()) != set(by_pair):
        raise ValueError("Rater 1 and Rater 2 image sets differ")

    cases = []
    for number, item in enumerate(payload["items"], 1):
        blind = item["blind_id"]
        r1 = by_pair[by_blind[blind]]
        original_label = r1["decision"].removeprefix("accept_candidate_set_")
        if original_label not in {"A", "B", "C"}:
            raise ValueError("Rater 1 did not choose a single candidate set")
        current_label = next((new for new, old in labels[blind].items() if old == original_label), None)
        r1_set = next((candidate for candidate in item["sets"] if candidate["label"] == current_label), None)
        if r1_set is None:
            raise ValueError("Rater 1 selected set missing from geometry package")
        r2 = by_r2[blind]
        flags = classify_case(r1_set, r2)
        cases.append({
            "original_page": number,
            "blind_id": blind,
            "image": item["image"],
            "image_sha256": item["image_sha256"],
            "width": item["width"],
            "height": item["height"],
            "sets": item["sets"],
            "r1_set": r1_set,
            "r2_record": r2,
            "flags": flags,
            "priority": bool(flags),
        })
    provenance = {
        "geometry_package_id": payload["package_id"],
        "geometry_payload_sha256": digest(geometry / "payload.json"),
        "rater2_geometry_json_sha256": digest(geometry_export),
        "rater2_geometry_csv_sha256": digest(geometry_export.with_suffix(".csv")),
        "rater1_opinion_json_sha256": digest(r1_path),
        "semantic_export_sha256": payload["semantic_export_sha256"],
    }
    return cases, provenance


def build(runtime: Path, geometry_export: Path, output: Path, package_id: str | None = None) -> dict:
    cases, provenance = load_cases(runtime, geometry_export)
    package_id = package_id or make_package_id()
    if not PACKAGE_RE.fullmatch(package_id):
        raise ValueError("Package ID must be a 16-character opaque uppercase ID")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "results").mkdir()
    geometry = runtime / "adjudication/semantic_first_20260906/geometry"
    for item in cases:
        source = geometry / item["image"]
        if digest(source) != item["image_sha256"]:
            raise ValueError("Image hash mismatch")
        shutil.copyfile(source, output / item["image"])

    source_dir = Path(__file__).parent
    shutil.copyfile(source_dir / "joint_consensus_index.html", output / "index.html")
    shutil.copyfile(source_dir / "joint_consensus_app.js", output / "app.js")
    shutil.copyfile(source_dir / "joint_consensus_style.css", output / "style.css")
    public = {
        "build_version": BUILD_VERSION,
        "package_id": package_id,
        "stage": STAGE,
        "source_provenance": provenance,
        "items": cases,
        "final_GT": False,
    }
    (output / "data.js").write_text(
        "window.CONSENSUS_DATA=" + json.dumps(public, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (output / "使用说明.txt").write_text(
        "双人路径共识记录包（13张）\n\n"
        "1. 双击 index.html。建议先完成标为‘优先讨论’的7张，再确认其余6张。\n"
        "2. 两位测量者共同核对真实叶片身份与数量、叶尖、共同基点和共享主路径。\n"
        "3. 每片叶只能选择一条完整候选路径；不得把不同候选的局部线段拼成一条新路径。\n"
        "4. 若没有任何完整候选贴合真实中心线，选择‘需要重新描迹’，并写明原因。\n"
        "5. 每张图由两位测量者分别勾选确认后提交；提交后仍可创建修订。\n"
        "6. 13张全部完成后，导出 joint_consensus_decisions.json 和同名 CSV。\n"
        "7. 请把这两个文件一起保存到本包 results 文件夹。\n\n"
        "本包只记录双人共识和必要重描需求，不修改候选曲线，也不直接冻结最终GT。\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            files[path.relative_to(output).as_posix()] = digest(path)
    manifest = {
        "build_version": BUILD_VERSION,
        "package_id": package_id,
        "stage": STAGE,
        "item_count": len(cases),
        "priority_item_count": sum(item["priority"] for item in cases),
        "source_provenance": provenance,
        "files": files,
        "final_GT": False,
        "instructions": "Two raters complete all 13 records, export JSON and CSV, and save both under results. This package does not freeze final GT.",
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "package_id": package_id,
        "items": len(cases),
        "priority_items": sum(item["priority"] for item in cases),
        "initial_records": 0,
        "final_GT": False,
    }


def validate_record(record: dict, item: dict, require_submitted: bool = True) -> None:
    if record.get("blind_id") != item["blind_id"] or record.get("image_sha256") != item["image_sha256"]:
        raise ValueError("Record identity or image hash mismatch")
    if require_submitted and not record.get("submitted"):
        raise ValueError("Record is not submitted")
    if record.get("decision") not in DECISIONS:
        raise ValueError("Invalid decision")
    count = record.get("agreed_visible_leaf_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Invalid agreed visible leaf count")
    if record.get("overall_confidence") not in CONFIDENCE:
        raise ValueError("Invalid overall confidence")
    if not record.get("rater1_confirmed") or not record.get("rater2_confirmed"):
        raise ValueError("Both raters must confirm")
    checks = record.get("checks", {})
    if set(checks) != {"identity_count", "tip_completeness", "common_base", "shared_path"}:
        raise ValueError("Invalid check fields")
    if not checks["identity_count"]:
        raise ValueError("Identity and count were not jointly checked")
    leaves = record.get("leaves")
    if not isinstance(leaves, list):
        raise ValueError("Leaves must be a list")
    candidate_keys = {
        f'{candidate["label"]}:{trace["candidate_trace_id"]}'
        for candidate in item["sets"] for trace in candidate["traces"]
    }
    uuids, used = set(), set()
    for leaf in leaves:
        leaf_uuid = leaf.get("consensus_leaf_uuid")
        if not isinstance(leaf_uuid, str) or not leaf_uuid or leaf_uuid in uuids:
            raise ValueError("Missing or duplicate consensus leaf UUID")
        uuids.add(leaf_uuid)
        if leaf.get("semantic_status") not in SEMANTIC:
            raise ValueError("Invalid semantic status")
        choice = leaf.get("path_choice")
        if choice not in candidate_keys | SPECIAL_CHOICES:
            raise ValueError("Invalid path choice")
        if choice in candidate_keys:
            if choice in used:
                raise ValueError("One candidate path was assigned twice")
            used.add(choice)
        if leaf.get("confidence") not in CONFIDENCE:
            raise ValueError("Invalid leaf confidence")
        if not leaf.get("source_r2_index") and not str(leaf.get("description", "")).strip():
            raise ValueError("A newly discovered leaf needs a description")
        if leaf["semantic_status"] == "measurable" and choice == "no_geometry":
            raise ValueError("A measurable leaf cannot omit geometry")
        if leaf["semantic_status"] == "visible_unmeasurable" and choice not in {"no_geometry", "hold_uncertain"}:
            raise ValueError("An unmeasurable leaf cannot use complete geometry")
        if choice in {"needs_redraw", "hold_uncertain"} and not str(leaf.get("notes", "")).strip():
            raise ValueError("Redraw or uncertainty needs notes")
        equivalent = leaf.get("equivalent_choices", [])
        if not isinstance(equivalent, list) or not set(equivalent).issubset(candidate_keys):
            raise ValueError("Invalid equivalent candidates")
    if len(leaves) != count:
        raise ValueError("Agreed visible leaf count does not match leaf records")
    decision = record["decision"]
    if decision == "consensus_ready":
        if not all(checks.values()):
            raise ValueError("A ready consensus requires all four checks")
        if any(leaf["semantic_status"] == "uncertain" or leaf["path_choice"] in {"needs_redraw", "hold_uncertain"} for leaf in leaves):
            raise ValueError("Unresolved leaves cannot be marked consensus-ready")
    elif decision == "redraw_required":
        if not any(leaf["path_choice"] == "needs_redraw" for leaf in leaves):
            raise ValueError("Redraw decision has no redraw leaf")
    elif not any(leaf["semantic_status"] == "uncertain" or leaf["path_choice"] == "hold_uncertain" for leaf in leaves) and not str(record.get("notes", "")).strip():
        raise ValueError("Third-rater decision needs an unresolved leaf or notes")
    if not isinstance(record.get("history"), list):
        raise ValueError("Revision history must be a list")


def validate_csv_companion_consensus(csv_path: Path, payload: dict) -> None:
    text = csv_path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != len(payload["records"]):
        raise ValueError("Consensus CSV row count mismatch")
    by_blind = {record["blind_id"]: record for record in payload["records"]}
    if len(by_blind) != len(rows):
        raise ValueError("Duplicate consensus JSON IDs")
    for row in rows:
        blind = row.get("blind_id")
        if blind not in by_blind:
            raise ValueError("Consensus CSV has an unknown ID")
        if json.loads(row["record_json"]) != by_blind[blind]:
            raise ValueError("Consensus CSV record_json differs from JSON")


def validate(package: Path, export: Path | None = None) -> dict:
    package = Path(package)
    manifest = read_json(package / "package_manifest.json")
    if manifest["build_version"] != BUILD_VERSION or manifest["stage"] != STAGE or manifest["item_count"] != 13:
        raise ValueError("Unexpected consensus package")
    for relative, checksum in manifest["files"].items():
        if digest(package / relative) != checksum:
            raise ValueError(f"Consensus package asset changed: {relative}")
    source = (package / "data.js").read_text(encoding="utf-8")
    if "v4_val_" in source or "dataset_id" in source:
        raise ValueError("Dataset identity leak in public consensus data")
    data = json.loads(source.removeprefix("window.CONSENSUS_DATA=").removesuffix(";\n"))
    report = {
        "package_assets_match": True,
        "identity_leak_hits": 0,
        "items": len(data["items"]),
        "priority_items": sum(item["priority"] for item in data["items"]),
        "final_GT": False,
    }
    if export:
        payload = read_json(export)
        if payload.get("build_version") != BUILD_VERSION or payload.get("package_id") != data["package_id"] or payload.get("stage") != STAGE:
            raise ValueError("Export does not belong to this consensus package")
        if payload.get("source_provenance") != data["source_provenance"]:
            raise ValueError("Consensus source provenance mismatch")
        by_item = {item["blind_id"]: item for item in data["items"]}
        records = payload.get("records", [])
        if len(records) != 13 or {record.get("blind_id") for record in records} != set(by_item):
            raise ValueError("Consensus export must contain the locked 13 samples")
        for record in records:
            validate_record(record, by_item[record["blind_id"]])
        validate_csv_companion_consensus(export.with_suffix(".csv"), payload)
        report.update({
            "submitted_records": len(records),
            "consensus_ready": sum(record["decision"] == "consensus_ready" for record in records),
            "redraw_required": sum(record["decision"] == "redraw_required" for record in records),
            "third_rater_required": sum(record["decision"] == "third_rater_required" for record in records),
            "export_sha256": digest(export),
            "status": "joint_consensus_record_complete_pending_final_GT_freeze",
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
        output = args.output or args.runtime / "adjudication/semantic_first_20260906/joint_consensus_20260915_v2"
        result = build(args.runtime, geometry, output, args.package_id)
    else:
        result = validate(args.package, args.export)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

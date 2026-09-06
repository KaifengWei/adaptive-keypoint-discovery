"""Isolated human adjudication packages. Does not import any model/evaluator.

semantic: original images only, fresh IDs/order, no candidate geometry in package.
geometry: requires a validated, archived semantic export; proposals, never final GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import secrets
import shutil
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent
VERSION = "semantic-first-adjudication-v2"
STATES = {"measurable", "visible_unmeasurable", "uncertain", "non_leaf"}
REASONS = {"none", "cropped_tip", "cropped_base", "occlusion", "damage", "identity_uncertain", "artifact", "other"}
FORBIDDEN = ("dataset_id", "v4_val_", "v4_test_", "rater", "measurement_round", "teacher", "student", "decoder")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def put_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def opaque():
    return "".join(secrets.choice("23456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(16))


def archive_opinion(runtime):
    source = runtime / "adjudication/method_blind_20260905/results"
    rows = read_csv(source / "anonymous_adjudication_decisions.csv")
    records = read_json(source / "anonymous_adjudication_decisions.json")["records"]
    if len(rows) != 13 or len(records) != 13 or len({r["adjudication_pair_id"] for r in rows}) != 13:
        raise ValueError("R1 opinion must have 13 unique submitted rows")
    by_id = {r["adjudication_pair_id"]: r for r in records}
    mapping_path = runtime / "admin/adjudication_candidate_set_mapping_20260905.csv"
    mapping = read_csv(mapping_path)
    sources = {}
    for row in rows:
        other = by_id.get(row["adjudication_pair_id"], {})
        for field in ("build_version", "decision", "confidence", "revision", "notes"):
            if str(row[field]) != str(other.get(field)):
                raise ValueError(f"CSV/JSON disagree: {field}")
        if datetime.fromisoformat(row["submitted_at_utc"].replace("Z", "+00:00")) != datetime.fromisoformat(other["submitted_at_utc"].replace("Z", "+00:00")):
            raise ValueError("CSV/JSON timestamp disagreement")
        label = row["decision"].removeprefix("accept_candidate_set_")
        candidates = [m for m in mapping if m["adjudication_pair_id"] == row["adjudication_pair_id"] and m["candidate_set_label"] == label]
        if len(candidates) != 1:
            raise ValueError("Opinion has missing/ambiguous candidate provenance")
        key = f'{candidates[0]["rater_id"]}_round{candidates[0]["measurement_round"]}'
        sources[key] = sources.get(key, 0) + 1
    target = runtime / "adjudication/opinions/rater1_20260905"
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("anonymous_adjudication_decisions.csv", "anonymous_adjudication_decisions.json"):
        src, dst = source / name, target / name
        if dst.exists() and digest(dst) != digest(src):
            raise ValueError("Archived opinion changed; create a separate revision")
        if not dst.exists():
            shutil.copyfile(src, dst)
        hashes[name] = digest(dst)
    manifest = {"status": "rater1_independent_opinion_not_final_GT", "count": 13,
                "files": hashes, "selected_sources": sources, "source_mapping_sha256": digest(mapping_path)}
    put_json(target / "archive_manifest.json", manifest)
    return manifest


def old_data(runtime):
    source = runtime / "adjudication/method_blind_20260905/data.js"
    manifest = read_json(source.parent / "package_manifest.json")
    for relative, checksum in manifest["public_files"].items():
        if digest(source.parent / relative) != checksum:
            raise ValueError(f"Original adjudication package changed: {relative}")
    raw = source.read_text(encoding="utf-8")
    return json.loads(raw.split("=", 1)[1].strip().removesuffix(";")), source


def validate_semantic(payload, data):
    if payload.get("build_version") != VERSION or payload.get("stage") != "semantic" or payload.get("package_id") != data["package_id"]:
        raise ValueError("Wrong semantic package/version/stage")
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) != len(data["items"]):
        raise ValueError("Incomplete semantic export")
    items = {i["blind_id"]: i for i in data["items"]}
    if len({r.get("blind_id") for r in records}) != len(items) or {r.get("blind_id") for r in records} != set(items):
        raise ValueError("Unexpected or duplicate blind IDs")
    for r in records:
        item = items[r["blind_id"]]
        if r.get("submitted") is not True or not r.get("submitted_at_utc") or r.get("review_complete") is not True:
            raise ValueError("Unsubmitted/unreviewed semantic record")
        if r.get("candidates_seen") is not False or r.get("image_sha256") != item["image_sha256"]:
            raise ValueError("Image hash or no-candidate declaration mismatch")
        leaves = r.get("leaves")
        if not isinstance(leaves, list) or (not leaves and not str(r.get("notes", "")).strip()):
            raise ValueError("Zero structures require a reason")
        if len({l.get("leaf_id") for l in leaves}) != len(leaves):
            raise ValueError("Duplicate structure IDs")
        for leaf in leaves:
            point = leaf.get("point")
            if not leaf.get("leaf_id") or leaf.get("status") not in STATES or leaf.get("confidence") not in {"high", "medium", "low"} or leaf.get("reason") not in REASONS:
                raise ValueError("Incomplete structure classification")
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ValueError("Invalid structure location")
            if not (0 <= point[0] < item["width"] and 0 <= point[1] < item["height"]):
                raise ValueError("Structure location outside image")
            if leaf["status"] != "measurable" and (leaf["reason"] == "none" or not str(leaf.get("notes", "")).strip()):
                raise ValueError("Unmeasurable/uncertain/non-leaf requires reason and notes")
            if leaf["status"] == "measurable" and leaf["reason"] not in {"none", "occlusion"}:
                raise ValueError("Truncated/damaged endpoints cannot be complete measurable paths")
    return records


def validate_csv_companion(export_path, payload):
    rows = read_csv(Path(export_path).with_suffix(".csv"))
    records = payload["records"]
    if len(rows) != len(records) or len({r["blind_id"] for r in rows}) != len(records):
        raise ValueError("CSV companion missing/duplicate rows")
    by_id = {r["blind_id"]: r for r in records}
    for row in rows:
        record = by_id.get(row["blind_id"])
        if json.loads(row["record_json"]) != record:
            raise ValueError("CSV/JSON companion content mismatch")
        expected = {"revision": record["revision"], "submitted_at_utc": record["submitted_at_utc"],
                    "structure_count": len(record["leaves"]), "notes": record["notes"]}
        expected.update({f"{state}_count": sum(l["status"] == state for l in record["leaves"]) for state in STATES})
        if any(str(row.get(k)) != str(v) for k, v in expected.items()):
            raise ValueError("CSV summary columns disagree with JSON")


def validate_geometry(payload, data):
    if payload.get("build_version") != VERSION or payload.get("stage") != "geometry" or payload.get("package_id") != data["package_id"] or payload.get("semantic_export_sha256") != data["semantic_export_sha256"]:
        raise ValueError("Wrong geometry package or semantic provenance")
    records = payload.get("records", [])
    items = {i["blind_id"]: i for i in data["items"]}
    if len(records) != len(items) or len({r.get("blind_id") for r in records}) != len(items) or {r.get("blind_id") for r in records} != set(items):
        raise ValueError("Incomplete/duplicate geometry records")
    unresolved = 0
    mixed = 0
    for r in records:
        i = items[r["blind_id"]]
        if r.get("submitted") is not True or r.get("review_complete") is not True or r.get("candidates_seen") is not True or r.get("image_sha256") != i["image_sha256"]:
            raise ValueError("Unsubmitted or invalid geometry record")
        original = {l["leaf_id"]: l for l in i["semantic"]["leaves"]}
        leaves = r.get("leaves", [])
        if len(leaves) != len(original) or {l["leaf_id"] for l in leaves} != set(original):
            raise ValueError("Semantic structures changed after candidate exposure")
        choices = {f'{s["label"]}:{t["candidate_trace_id"]}' for s in i["sets"] for t in s["traces"]}
        selected = set()
        for leaf in leaves:
            for k, v in original[leaf["leaf_id"]].items():
                if leaf.get(k) != v:
                    raise ValueError("Original semantic decision changed after candidate exposure")
            choice = leaf.get("geometry_choice")
            allowed = choices | {"needs_redraw", "hold_uncertain"} if leaf["status"] == "measurable" else {"hold_uncertain", "no_geometry"}
            if choice not in allowed or leaf.get("geometry_confidence") not in {"high", "medium", "low"}:
                raise ValueError("Invalid geometry choice/confidence")
            if choice in {"needs_redraw", "hold_uncertain"} and not str(leaf.get("geometry_notes", "")).strip():
                raise ValueError("Pending geometry decision needs a reason")
            if choice in choices:
                if choice in selected:
                    raise ValueError("Same candidate assigned to multiple structures")
                selected.add(choice)
            if choice in {"needs_redraw", "hold_uncertain"} or leaf["status"] == "uncertain":
                unresolved += 1
        mixed += len({c.split(":")[0] for c in selected}) > 1
    return {"status": "independent_opinion_pending_consensus", "records": len(records), "unresolved_structures": unresolved,
            "mixed_source_plants_require_common_base_review": mixed, "final_GT": False}


def scan_package(out, stage, forbidden_ids=()):
    hits = []
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "package_manifest.json"]
    for p in files:
        if p.suffix in {".html", ".js", ".json", ".css", ".md"}:
            text = p.read_text(encoding="utf-8").lower()
            for token in (*FORBIDDEN, *forbidden_ids):
                if token.lower() in text:
                    hits.append((p.name, token))
    public = read_json(out / "payload.json")
    if stage == "semantic" and any(k in i for i in public["items"] for k in ("sets", "traces", "trace_count", "reason_categories")):
        hits.append(("payload.json", "candidate geometry/count/reasons"))
    if hits:
        raise ValueError(f"Public leakage: {hits[:6]}")
    return {str(p.relative_to(out)).replace("\\", "/"): digest(p) for p in files}


def build(runtime, stage, semantic_export=None, output=None):
    from PIL import Image
    parent = runtime / "adjudication/semantic_first_20260906"
    admin = runtime / "admin/semantic_first_20260906"
    old, old_path = old_data(runtime)
    if len(old["items"]) != 13:
        raise ValueError("Expected the original locked 13 adjudication samples")
    map_file = admin / "mapping.json"
    if map_file.exists():
        mapping = read_json(map_file)
    else:
        pairs = [i["adjudication_pair_id"] for i in old["items"]]
        secrets.SystemRandom().shuffle(pairs)
        mapping = {"package_id": opaque(), "items": [{"blind_id": opaque(), "pair_id": p} for p in pairs]}
        if len({i["blind_id"] for i in mapping["items"]}) != 13:
            raise ValueError("Random ID collision")
        put_json(map_file, mapping)
    if {i["pair_id"] for i in mapping["items"]} != {i["adjudication_pair_id"] for i in old["items"]}:
        raise ValueError("Mapping does not match locked adjudication set")
    out = Path(output) if output else parent / stage
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a distributed package: {out}")
    by_pair = {i["adjudication_pair_id"]: i for i in old["items"]}
    semantic_records = {}
    export_hash = None
    if stage == "geometry":
        if not semantic_export:
            raise ValueError("A complete semantic export is required before geometry release")
        public_semantic = read_json(parent / "semantic/payload.json")
        sem_manifest = read_json(parent / "semantic/package_manifest.json")
        for rel, checksum in sem_manifest["files"].items():
            if digest(parent / "semantic" / rel) != checksum:
                raise ValueError("Distributed semantic package was modified")
        exported = read_json(semantic_export)
        semantic_records = {r["blind_id"]: r for r in validate_semantic(exported, public_semantic)}
        validate_csv_companion(semantic_export, exported)
        export_hash = digest(semantic_export)
        archive = admin / "semantic_exports" / export_hash
        archive.mkdir(parents=True, exist_ok=True)
        dest = archive / "semantic_decisions.json"
        if not dest.exists():
            shutil.copyfile(semantic_export, dest)
            shutil.copyfile(Path(semantic_export).with_suffix(".csv"), archive / "semantic_decisions.csv")
        labels_file = admin / f"geometry_labels_{export_hash}.json"
        if labels_file.exists():
            geometry_labels = read_json(labels_file)
        else:
            geometry_labels = {}
            for m in mapping["items"]:
                labels = ["A", "B", "C"]
                secrets.SystemRandom().shuffle(labels)
                geometry_labels[m["blind_id"]] = dict(zip(("A", "B", "C"), labels))
            put_json(labels_file, geometry_labels)
    out.mkdir(parents=True)
    (out / "images").mkdir()
    (out / "results").mkdir()
    items = []
    for m in mapping["items"]:
        old_item = by_pair[m["pair_id"]]
        src = old_path.parent / old_item["image"]
        img_name = f'images/{m["blind_id"]}.png'
        shutil.copyfile(src, out / img_name)
        with Image.open(src) as im:
            width, height = im.size
        item = {"blind_id": m["blind_id"], "image": img_name, "image_sha256": digest(src), "width": width, "height": height}
        if stage == "geometry":
            item["semantic"] = semantic_records[m["blind_id"]]
            item["sets"] = []
            for label, old_label in geometry_labels[m["blind_id"]].items():
                candidate = next(s for s in old_item["sets"] if s["label"] == old_label)
                item["sets"].append({"label": label, "traces": candidate["traces"]})
        items.append(item)
    data = {"build_version": VERSION, "stage": stage, "package_id": mapping["package_id"], "items": items}
    if export_hash:
        data["semantic_export_sha256"] = export_hash
    put_json(out / "payload.json", data)
    (out / "data.js").write_text("window.REVIEW_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(HERE / name, out / name)
    instructions = "打开 index.html；本包可离线使用。每张提交后自动保存在当前浏览器，也请下载进度备份。完成后分别点击两个下载链接，把 JSON 与 CSV 一起放入 results 文件夹。不要只发送 index.html，要发送此整个文件夹。\n"
    (out / "使用说明.txt").write_text(instructions, encoding="utf-8")
    forbidden_ids = [i["adjudication_pair_id"] for i in old["items"]]
    forbidden_ids += [r["blind_id"] for r in read_csv(runtime / "admin/blind_id_admin_mapping.csv")]
    checksums = scan_package(out, stage, forbidden_ids)
    manifest = {"version": VERSION, "stage": stage, "package_id": mapping["package_id"], "item_count": 13,
                "status": "ready_for_independent_human_opinion", "final_GT": False,
                "candidate_geometry_present": stage == "geometry", "identity_leak_hits": 0,
                "files": checksums, "semantic_export_sha256": export_hash}
    put_json(out / "package_manifest.json", manifest)
    return {"output": str(out), "stage": stage, "items": 13, "final_GT": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["archive-opinion", "semantic", "geometry"])
    parser.add_argument("--runtime", type=Path, default=PROTOCOL / "runtime")
    parser.add_argument("--semantic-export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = archive_opinion(args.runtime) if args.action == "archive-opinion" else build(args.runtime, args.action, args.semantic_export, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))

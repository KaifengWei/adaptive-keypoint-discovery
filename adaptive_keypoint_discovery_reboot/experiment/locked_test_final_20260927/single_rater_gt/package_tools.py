"""Raw-annotation/asset administration only; no model or phenotype computation."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check_locked_test_static import audit_public_package, validate_public_manifest, sha, require


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def code_hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in {".py", ".html", ".js"}}


def inventory(package):
    rows = [{"path": p.relative_to(package).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha(p)} for p in sorted(package.rglob("*")) if p.is_file()]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {"definition": "SHA256(UTF8 sorted-relative-path/bytes/sha256 inventory JSON; sorted keys, compact separators)",
            "package_sha256": digest_text(canonical), "files": rows}


def point_ok(p, s):
    return (isinstance(p, list) and len(p) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in p)
            and 0 <= p[0] <= s["width"] and 0 <= p[1] <= s["height"])


def validate_snapshot(r, s):
    require(r["schema_version"] == "blind-session-v1" and r["rater_id"] == "S1" and r["round"] == 1 and r["session_id"], "Session identity/schema invalid")
    for f in ("blind_id", "image_sha256", "width", "height"):
        require(r[f] == s[f], f"Identity mismatch: {f}")
    require(r["base_xy"] is None or point_ok(r["base_xy"], s), "Invalid base")
    require(r["whole_plant_checked"] is True, "Whole plant not checked")
    require(isinstance(r["items"], list), "Invalid structures")
    require(r["items"] or r["image_note"].strip(), "Empty record needs explanation")
    require(re.fullmatch(r"\d{4}-\d\d-\d\dT.*Z", r["submitted_at"]) is not None, "Invalid submission timestamp")
    ids = set()
    for i, t in enumerate(r["items"], 1):
        require(t["gt_id"] == f"GT{i:02}", "Submitted GT numbering changed")
        require(t["trace_uuid"] and t["trace_uuid"] not in ids, "Duplicate structure UUID")
        ids.add(t["trace_uuid"])
        state = t["visibility_status"]
        require(state in {"measurable", "visible_unmeasurable", "uncertain", "non_target_structure"}, "Invalid state")
        require(t["confidence"] in {"high", "medium", "low"}, "Invalid confidence")
        require(t["occlusion"] in {"none", "minor", "major"} and t["interpolation_used"] in {"yes", "no"}, "Invalid occlusion/interpolation")
        require(t["tip_xy"] is None or point_ok(t["tip_xy"], s), "Invalid tip")
        p = t["points_px"]
        require(isinstance(p, list), "Invalid raw trace")
        if state == "measurable":
            require(r["base_xy"] is not None and len(p) >= 2 and all(point_ok(x, s) for x in p), "Incomplete measurable geometry")
            require(p[0] == r["base_xy"] and p[-1] == t["tip_xy"], "Base/tip mismatch")
            require(any(math.dist(x, p[0]) > 1e-6 for x in p), "Zero-length trace")
            require(t["occlusion"] != "major", "Major occlusion cannot have reliable geometry")
        else:
            require(not p and t["note"].strip(), "Nonmeasurable record contains geometry or lacks reason")
        if t["interpolation_used"] == "yes":
            require(state == "measurable" and t["occlusion"] == "minor", "Invalid interpolation eligibility")


def validate_raw(outer, manifest, manifest_sha):
    require(set(outer) == {"payload_json", "sha256"}, "Unexpected raw envelope")
    require(digest_text(outer["payload_json"]) == outer["sha256"], "Raw envelope hash mismatch")
    payload = json.loads(outer["payload_json"])
    require(set(payload) == {"schema_version", "manifest_sha256", "rater_id", "samples"}, "Unexpected raw fields")
    require(payload["schema_version"] == "blind-raw-v1" and payload["rater_id"] == "S1", "Wrong raw schema/rater")
    require(payload["manifest_sha256"] == manifest_sha, "Wrong package")
    by_id = {s["blind_id"]: s for s in manifest["samples"]}
    require(len(payload["samples"]) == len(by_id), "Raw N mismatch")
    seen = set()
    revisions, counts = 0, dict.fromkeys(["measurable", "visible_unmeasurable", "uncertain", "non_target_structure"], 0)
    for x in payload["samples"]:
        ident = x["blind_id"]
        require(ident in by_id and ident not in seen, "Unknown/duplicate raw sample")
        seen.add(ident)
        previous, first = None, None
        require(x["history"], "Missing submission")
        for n, h in enumerate(x["history"], 1):
            require(set(h) == {"snapshot_json", "sha256"}, "Unexpected snapshot envelope")
            require(digest_text(h["snapshot_json"]) == h["sha256"], "Snapshot hash mismatch")
            r = json.loads(h["snapshot_json"])
            require(r["manifest_sha256"] == manifest_sha, "Snapshot manifest mismatch")
            validate_snapshot(r, by_id[ident])
            require(r["revision"] == n and r["previous_submission_sha256"] == previous, "Revision chain broken")
            if n == 1:
                require(r["technical_revision"] is None, "Initial submission is a revision")
                first = r
            else:
                t = r["technical_revision"]
                require(t and t["kind"] in {"invalid_coordinates", "export_failure", "corrupt_file"} and t["reason"].strip(), "Technical reason missing")
                require(len(r["items"]) == len(first["items"]), "Revision changed count")
                require(r["image_note"] == first["image_note"] and r["whole_plant_checked"] == first["whole_plant_checked"], "Revision changed plant judgment")
                require(r["session_id"] == first["session_id"], "Revision changed session identity")
                for a, b in zip(r["items"], first["items"]):
                    for f in ("trace_uuid", "gt_id", "visibility_status", "occlusion", "interpolation_used", "confidence", "note"):
                        require(a[f] == b[f], f"Revision changed judgment: {f}")
            previous = h["sha256"]
            revisions += 1
        for t in r["items"]:
            counts[t["visibility_status"]] += 1
    return {"status": "PASS", "n": len(seen), "snapshots": revisions, "states": counts,
            "test_model_reads": 0, "inference_gate": "CLOSED"}


def validate_exports(results, package):
    manifest = json.loads((package / "measurement_manifest.json").read_text(encoding="utf-8"))
    manifest_sha = sha(package / "measurement_manifest.json")
    ledger = json.loads((results / "SHA256_LEDGER.json").read_text(encoding="utf-8"))
    require(ledger["manifest_sha256"] == manifest_sha, "Export ledger package mismatch")
    require(set(ledger["files"]) == {"raw_annotations.json", "sessions.csv", "traces.csv"}, "Export files mismatch")
    for name, expected in ledger["files"].items():
        require(sha(results / name) == expected, f"Export file changed: {name}")
    raw = json.loads((results / "raw_annotations.json").read_text(encoding="utf-8"))
    report = validate_raw(raw, manifest, manifest_sha)
    payload = json.loads(raw["payload_json"])
    expected_sessions, expected_traces = [], []
    for x in payload["samples"]:
        h = x["history"][-1]
        r = json.loads(h["snapshot_json"])
        expected_sessions.append((x["blind_id"], str(r["revision"]), h["sha256"], h["snapshot_json"]))
        for t in r["items"]:
            expected_traces.append((x["blind_id"], t, h["sha256"]))
    with (results / "sessions.csv").open(encoding="utf-8-sig", newline="") as f:
        sessions = list(csv.DictReader(f))
    require([(s["blind_id"], s["revision"], s["record_sha256"], s["record_json"]) for s in sessions] == expected_sessions, "Sessions disagree with immutable JSON")
    with (results / "traces.csv").open(encoding="utf-8-sig", newline="") as f:
        traces = list(csv.DictReader(f))
    require(len(traces) == len(expected_traces), "Trace row count mismatch")
    for row, (ident, t, hash_) in zip(traces, expected_traces):
        require(row["blind_id"] == ident and row["record_sha256"] == hash_, "Trace identity mismatch")
        for field in ("gt_id", "trace_uuid", "visibility_status", "occlusion", "interpolation_used", "confidence", "note"):
            require(row[field] == t[field], f"CSV state mismatch: {field}")
        require(json.loads(row["points_px"]) == t["points_px"] and json.loads(row["tip_xy"]) == t["tip_xy"], "CSV coordinates mismatch")
    return report

"""Static preregistration/GT-package checks; never imports or executes a model.

The default audit hashes frozen files and parses manifest metadata only. It
does not open dataset images. A future --public-package audit checks explicit
public assets as bytes (not pixels); it never changes its input or grants an
inference permission. Missing GT/evaluator gates remain closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "V4_LOCKED_TEST_PREREGISTRATION.md"
EXPERIMENT = HERE.parent
IDENTITY_KEYS = (
    "dataset_id", "split", "source_frame_id", "split_group", "crop_box_full",
    "source_crop_box_fullplant", "output_sha256", "relative_path",
    "normalization_version",
)
BLIND_RE = re.compile(r"[A-Z2-7]{16,26}\Z")
SHA_RE = re.compile(r"[a-f0-9]{64}\Z")
LEAK_RE = re.compile(
    r"v4_(?:train|val|test)_\w+|stagev3_\w+|teacher|student|"
    r"\bcore\b|\bdiagnostic\b|visible_leaf_count|expected_leaf_count|"
    r"expected_path_count|dataset_id|checkpoint|\.pt\b|"
    r"source_frame_id|source_relative_path|D:[/\\]kp",
    re.IGNORECASE,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_ledger(path: Path = PROTOCOL) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    block = re.search(
        r"<!-- LOCKED_LEDGER_BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- LOCKED_LEDGER_END -->", text, re.DOTALL,
    )
    require(block is not None, "Frozen ledger block is absent")
    ledger = json.loads(block.group(1))
    require(ledger["test_model_reads"] == 0, "Unexpected recorded model access")
    require(ledger["test_gt"] == "PENDING", "This checker is preregistration-only")
    require(ledger["test_evaluator"] == "PENDING", "Test evaluator not accounted for")
    require(ledger["bootstrap_seed"] == 20260927, "Bootstrap seed changed")
    require(ledger["bootstrap_replicates"] == 10000, "Bootstrap N changed")
    return ledger


def canonical_identity(rows: list[dict]) -> str:
    records = [
        {key: row[key].replace("\\", "/") for key in IDENTITY_KEYS}
        for row in sorted(rows, key=lambda row: row["dataset_id"])
    ]
    data = json.dumps(records, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def audit_splits(experiment: Path, ledger: dict) -> dict:
    reports, ids, groups, frames, image_hashes = {}, {}, {}, {}, {}
    for split, expected in ledger["splits"].items():
        path = experiment / "data_stage_clean_v4_fullplant_candidate/manifests" / f"{split}.csv"
        raw = sha(path)
        require(raw in {expected["local_raw_sha256"], expected["remote_raw_sha256"]},
                f"Unregistered raw manifest bytes: {split}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids[split] = {row["dataset_id"] for row in rows}
        groups[split] = {row["split_group"].replace("\\", "/") for row in rows}
        frames[split] = {row["source_frame_id"] for row in rows if row["source_frame_id"]}
        image_hashes[split] = {row["output_sha256"] for row in rows}
        require(len(rows) == expected["n"], f"Split N changed: {split}")
        require(len(ids[split]) == len(rows), f"Duplicate dataset ID: {split}")
        require(len(image_hashes[split]) == len(rows), f"Duplicate expected image bytes: {split}")
        require(all(row["split"] == split for row in rows), f"Mixed split rows: {split}")
        require(len(frames[split]) == expected["frames"], f"Frame count changed: {split}")
        require(all(row.get("model_outputs_used_for_selection") == "0" for row in rows),
                f"Model-based selection recorded: {split}")
        canon = canonical_identity(rows)
        require(canon == expected["canonical_sha256"], f"Identity manifest changed: {split}")
        reports[split] = {"n": len(rows), "source_frames": len(frames[split]),
                          "raw_sha256": raw, "canonical_sha256": canon}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for label, sets in (("identity", ids), ("group", groups),
                            ("frame", frames), ("image hash", image_hashes)):
            require(not sets[left] & sets[right], f"Cross-split {label}: {left}/{right}")
    return reports


def audit_artifacts(experiment: Path, ledger: dict) -> dict:
    reports = {}
    for relative, expected in ledger["artifacts"].items():
        require(SHA_RE.fullmatch(expected) is not None, f"Invalid ledger hash: {relative}")
        path = experiment / relative
        if not path.is_file() and relative in ledger["remote_only"]:
            reports[relative] = {"status": "REMOTE_VERIFIED_NOT_LOCAL", "sha256": expected}
            continue
        require(path.is_file(), f"Missing frozen source/checkpoint: {relative}")
        actual = sha(path)
        require(actual == expected, f"Frozen artifact changed: {relative}")
        reports[relative] = {"status": "VERIFIED", "sha256": actual}
    return reports


def audit_dinov2_source(experiment: Path, ledger: dict) -> dict:
    repo = experiment / "third_party/dinov2_git"
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    require(commit == ledger["dinov2_source_git"], "DINOv2 source commit changed")
    require(not git("status", "--porcelain"), "DINOv2 source worktree changed")
    return {"status": "VERIFIED", "git_commit": commit}


def validate_public_manifest(payload: dict, expected_n: int = 40) -> list[dict]:
    require(set(payload) == {"schema_version", "samples"}, "Unexpected public root field")
    require(payload["schema_version"] == "v4-test-blind-gt-v1", "Unexpected schema version")
    samples = payload["samples"]
    require(isinstance(samples, list) and len(samples) == expected_n, "Public sample N mismatch")
    allowed = {"blind_id", "image_filename", "image_sha256", "width", "height"}
    identifiers, hashes = set(), set()
    for row in samples:
        require(isinstance(row, dict) and set(row) == allowed, "Unexpected public sample field")
        ident = row["blind_id"]
        require(isinstance(ident, str) and BLIND_RE.fullmatch(ident) is not None,
                "Blind ID not opaque-format")
        require(ident not in identifiers, "Duplicate blind ID")
        identifiers.add(ident)
        name = row["image_filename"]
        require(name in {f"images/{ident}.png", f"images/{ident}.jpg"},
                "Asset filename must be blind-only and local")
        require(isinstance(row["image_sha256"], str) and
                SHA_RE.fullmatch(row["image_sha256"]) is not None, "Invalid asset SHA")
        require(row["image_sha256"] not in hashes, "Duplicate public image hash")
        hashes.add(row["image_sha256"])
        for key in ("width", "height"):
            require(type(row[key]) is int and row[key] > 0, "Invalid image dimensions")
    return samples


def audit_public_package(package: Path, expected_n: int = 40) -> dict:
    package = package.resolve()
    manifest = package / "measurement_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    samples = validate_public_manifest(payload, expected_n)
    require((package / "index.html").is_file(), "Public HTML entry missing")
    allowed_files = {"index.html", "app.js", "style.css", "measurement_manifest.json"}
    allowed_files.update(row["image_filename"] for row in samples)
    for path in package.rglob("*"):
        require(not path.is_symlink(), "Public symlink not allowed")
        require(path.resolve().is_relative_to(package), "Escaping public asset")
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        require(relative in allowed_files, f"Unregistered public file: {relative}")
        if path.suffix.lower() in {".html", ".js", ".css", ".json", ".csv"}:
            require(LEAK_RE.search(path.read_text(encoding="utf-8-sig")) is None,
                    f"Potential public identity/output leak: {relative}")
    for row in samples:
        path = package / row["image_filename"]
        require(path.is_file() and sha(path) == row["image_sha256"], "Public asset hash mismatch")
    return {"status": "STATIC_PASS_NOT_BROWSER_CERTIFICATION", "n": len(samples),
            "measurement_manifest_sha256": sha(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-package", type=Path)
    args = parser.parse_args()
    try:
        ledger = load_ledger()
        result = {
            "static_status": "PASS",
            "preregistration_sha256": sha(PROTOCOL),
            "splits": audit_splits(EXPERIMENT, ledger),
            "artifacts": audit_artifacts(EXPERIMENT, ledger),
            "dinov2_source": audit_dinov2_source(EXPERIMENT, ledger),
            "recorded_test_model_reads": 0,
            "checker_model_reads": 0,
            "checker_dataset_image_reads": 0,
            "inference_gate": "CLOSED_GT_AND_TEST_EVALUATOR_PENDING",
        }
        if args.public_package:
            result["public_package"] = audit_public_package(args.public_package)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.CalledProcessError) as exc:
        print(json.dumps({"static_status": "FAIL", "reason": str(exc),
                          "inference_gate": "CLOSED"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

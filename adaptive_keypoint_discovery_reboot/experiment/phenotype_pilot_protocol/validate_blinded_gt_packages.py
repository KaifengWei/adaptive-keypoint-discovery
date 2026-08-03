"""Static leakage and package-integrity checks for the blinded GT platform."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
RUNTIME = PROTOCOL_DIR / "runtime"
REPORT_DIR = PROTOCOL_DIR / "validation" / "20260803"
FORBIDDEN = (
    "v4_val_", "dataset_id", "pilot_group", "visible_leaf_count", "leaf_exists",
    "method_id", "teacher_direct", "student_b", "student_d", "decoder",
)
TEXT_SUFFIXES = {".html", ".js", ".css", ".json", ".txt", ".csv"}
BLIND_RE = re.compile(r"^[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{14}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    mapping_path = RUNTIME / "admin" / "blind_id_admin_mapping.csv"
    if not mapping_path.exists():
        fail("administrator mapping is missing")
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        mapping = list(csv.DictReader(handle))
    if len(mapping) != 50:
        fail(f"expected 50 anonymous assignments, found {len(mapping)}")
    blind_ids = [row["blind_id"] for row in mapping]
    if len(set(blind_ids)) != 50 or not all(BLIND_RE.fullmatch(value) for value in blind_ids):
        fail("blind IDs are not unique opaque 14-character tokens")

    package_expected = {"practice": 2, "rater1_round1": 16, "rater1_round2": 16, "rater2_round1": 16}
    package_counts: dict[str, int] = {}
    leak_hits: list[dict[str, Any]] = []
    public_images = 0
    for package_name, expected in package_expected.items():
        package = RUNTIME / "packages" / package_name
        public = json.loads((package / "package_manifest_public.json").read_text(encoding="utf-8"))
        if len(public["items"]) != expected:
            fail(f"{package_name} expected {expected} items, found {len(public['items'])}")
        package_counts[package_name] = len(public["items"])
        public_blind = {item["blind_id"] for item in public["items"]}
        mapped_blind = {row["blind_id"] for row in mapping if row["package_id"] == public["package_id"]}
        if public_blind != mapped_blind:
            fail(f"{package_name} public manifest and administrator map disagree")
        for item in public["items"]:
            image_path = package / item["image_alias"]
            if not image_path.exists() or sha256_file(image_path) != item["image_sha256"]:
                fail(f"image hash mismatch in {package_name}: {item['image_alias']}")
            if not BLIND_RE.fullmatch(image_path.stem):
                fail(f"non-opaque image alias in {package_name}: {image_path.name}")
            public_images += 1
        for path in package.rglob("*"):
            relative = str(path.relative_to(package)).replace("\\", "/")
            lower_name = relative.lower()
            for token in FORBIDDEN:
                if token in lower_name:
                    leak_hits.append({"package": package_name, "file": relative, "token": token, "location": "filename"})
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8-sig").lower()
                for token in FORBIDDEN:
                    if token in text:
                        leak_hits.append({"package": package_name, "file": relative, "token": token, "location": "content"})
    if leak_hits:
        fail(f"static leakage scan found {len(leak_hits)} hits: {leak_hits[:5]}")

    formal = [row for row in mapping if row["pilot_group"] != "practice"]
    for package_name in ("rater1_round1", "rater1_round2", "rater2_round1"):
        rows = [row for row in formal if Path(row["image_alias"]).stem in {item["blind_id"] for item in json.loads((RUNTIME / "packages" / package_name / "package_manifest_public.json").read_text(encoding="utf-8"))["items"]}]
        if len(rows) != 16 or {row["pilot_group"] for row in rows} != {"core", "diagnostic"}:
            fail(f"{package_name} does not contain the locked 16-image set")
    r1_first = {row["blind_id"] for row in mapping if row["rater_id"] == "R1" and row["measurement_round"] == "1"}
    r1_second = {row["blind_id"] for row in mapping if row["rater_id"] == "R1" and row["measurement_round"] == "2"}
    if r1_first & r1_second:
        fail("Rater 1 rounds reuse blind IDs")

    for filename in ("phenotype_measurement_session_template.csv", "manual_phenotype_reference_template.csv"):
        lines = (PROTOCOL_DIR / filename).read_text(encoding="utf-8-sig").splitlines()
        if len(lines) != 1:
            fail(f"{filename} is not header-only")
        lower = lines[0].lower()
        if any(token in lower for token in ("dataset_id", "pilot_group", "visible_leaf_count", "leaf_exists", "method_id")):
            fail(f"{filename} contains a forbidden measurement-page field")

    report = {
        "status": "pass",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "phenotype-first-pilot-v2",
        "geometry_protocol_version": "phenotype-geometry-v1",
        "package_counts": package_counts,
        "unique_blind_ids": len(set(blind_ids)),
        "public_image_files_checked": public_images,
        "static_forbidden_hits": 0,
        "rater1_round_blind_id_overlap": 0,
        "measurement_templates_header_only": True,
        "rater2_scope": "all_locked_16",
        "test_images_read": 0,
        "model_outputs_read": 0,
        "admin_mapping_sha256": sha256_file(mapping_path),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "static_leak_check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

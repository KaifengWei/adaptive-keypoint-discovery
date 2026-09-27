"""Create synthetic fixtures first; then one immutable administrative asset package.

No image content QC, pixel analysis, models, masks, or phenotype computation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import secrets
import shutil
import subprocess
from PIL import Image, ImageDraw
from package_tools import HERE, sha, require, code_hashes, inventory, audit_public_package
from check_locked_test_static import canonical_identity, load_ledger

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
FORMAL = HERE.parents[1] / "phenotype_pilot_protocol/runtime/v4_test_single_rater_20260927"
AMENDMENT = HERE.parent / "V4_LOCKED_TEST_GT_SINGLE_RATER_AMENDMENT.md"


def build_public(output, images):
    require(not output.exists(), "Refuse to overwrite or create a second package at existing destination")
    (output / "images").mkdir(parents=True)
    samples, mapping, used = [], [], set()
    order = list(images)
    secrets.SystemRandom().shuffle(order)
    for info in order:
        ident = "".join(secrets.choice(ALPHABET) for _ in range(20))
        while ident in used:
            ident = "".join(secrets.choice(ALPHABET) for _ in range(20))
        used.add(ident)
        source = info["path"]
        name = f"images/{ident}{source.suffix.lower()}"
        require(source.suffix.lower() in {".png", ".jpg"}, "Unsupported standardized asset")
        before = sha(source)
        require(before == info["expected_sha256"], "Standardized asset changed")
        with Image.open(source) as im:
            width, height = im.size
            require(im.mode == "RGB", "Standardized original is not RGB; no conversion permitted")
        shutil.copyfile(source, output / name)
        require(sha(output / name) == before, "Copy changed asset")
        samples.append({"blind_id": ident, "image_filename": name, "image_sha256": before, "width": width, "height": height})
        mapping.append({"blind_id": ident, "display_order": len(samples), **{k: str(v) if isinstance(v, Path) else v for k, v in info.items()}})
    manifest = {"schema_version": "v4-test-blind-gt-v1", "samples": samples}
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path = output / "measurement_manifest.json"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    template = (HERE / "index.html").read_text(encoding="utf-8")
    html = template.replace("__MANIFEST__", json.dumps(manifest, separators=(",", ":"))).replace("__MANIFEST_SHA__", sha(manifest_path))
    (output / "index.html").write_text(html, encoding="utf-8", newline="\n")
    shutil.copyfile(HERE / "app.js", output / "app.js")
    report = audit_public_package(output, len(samples))
    return mapping, report


def fixture(root):
    require(not root.exists(), "Fixture destination exists")
    assets = root / "synthetic_source"
    assets.mkdir(parents=True)
    rows = []
    for n in range(2):
        p = assets / f"synthetic_{n}.png"
        im = Image.new("RGB", (900 + 100 * n, 350), "white")
        d = ImageDraw.Draw(im)
        d.line([(760, 210), (420, 190), (80, 120 + n * 20)], fill=(70, 140, 40), width=10)
        d.line([(760, 210), (420, 190), (270, 250)], fill=(60, 120, 40), width=8)
        im.save(p)
        rows.append({"path": p, "expected_sha256": sha(p), "fixture": True})
    _, audit = build_public(root / "public", rows)
    (root / "fixture_asset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def formal(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["status"] == "PASS" and report["synthetic_only"] is True, "Synthetic/browser acceptance absent")
    require(report["code_sha256"] == code_hashes(), "Code changed after synthetic/browser acceptance")
    require(report["amendment_sha256"] == sha(AMENDMENT), "Amendment changed after acceptance")
    require(not FORMAL.exists(), "Unique formal package already exists; no rebuild authorized")
    git_root = HERE.parents[3]
    ignored = subprocess.run(["git", "-C", str(git_root), "check-ignore", str(FORMAL / "admin/mapping.json")], capture_output=True, text=True)
    require(ignored.returncode == 0, "Private runtime is not Git ignored")
    ledger = load_ledger()
    data = HERE.parents[1] / "data_stage_clean_v4_fullplant_candidate"
    manifest = data / "manifests/test.csv"
    expected = ledger["splits"]["test"]
    require(sha(manifest) == expected["local_raw_sha256"], "Frozen test manifest changed")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 40 and canonical_identity(rows) == expected["canonical_sha256"], "Frozen 40 identities changed")
    require(len({r["dataset_id"] for r in rows}) == 40 and all(r["split"] == "test" for r in rows), "Split identity invalid")
    images = []
    for r in rows:
        p = (data / r["relative_path"].replace("\\", "/")).resolve()
        require(p.is_relative_to((data / "images/test").resolve()), "Non-test asset path")
        images.append({"path": p, "expected_sha256": r["output_sha256"], "dataset_id": r["dataset_id"], "source_frame_id": r["source_frame_id"]})
    # Only exact listed byte hashes and image headers; no pixel-content operations.
    for info in images:
        require(sha(info["path"]) == info["expected_sha256"], "Frozen asset changed")
        with Image.open(info["path"]) as im:
            require(im.mode == "RGB" and min(im.size) > 0, "Image header invalid")
    mapping, audit = build_public(FORMAL / "public", images)
    admin = FORMAL / "admin"
    admin.mkdir()
    (FORMAL / "results").mkdir()
    (admin / "mapping.json").write_text(json.dumps({"schema_version": "private-blind-map-v1", "samples": mapping}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(report_path, admin / "synthetic_browser_acceptance.json")
    asset_ledger = inventory(FORMAL / "public")
    asset_ledger.update({"amendment_sha256": sha(AMENDMENT), "measurement_manifest_sha256": sha(FORMAL / "public/measurement_manifest.json"), "private_mapping_sha256": sha(admin / "mapping.json"), "source_manifest_sha256": sha(manifest), "code_sha256": code_hashes(), "test_model_reads": 0, "inference_gate": "CLOSED", "initial_results_empty": not any((FORMAL / "results").iterdir()), "asset_audit": audit})
    (admin / "SHA256_LEDGER.json").write_text(json.dumps(asset_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"package": str(FORMAL), "package_sha256": asset_ledger["package_sha256"], "n": 40, "blind_unique": len({x["blind_id"] for x in mapping}), "test_model_reads": 0, "inference_gate": "CLOSED"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", type=Path)
    group.add_argument("--formal-after-synthetic-pass", type=Path)
    args = parser.parse_args()
    if args.synthetic:
        fixture(args.synthetic.resolve())
    else:
        formal(args.formal_after_synthetic_pass.resolve())

"""Read-only validation of public package and human CSV/JSON export pair."""
import argparse
import json
from pathlib import Path
from build import digest, read_json, scan_package, validate_semantic, validate_geometry, validate_csv_companion


def validate(package, export=None):
    package = Path(package)
    manifest = read_json(package / "package_manifest.json")
    for rel, checksum in manifest["files"].items():
        if digest(package / rel) != checksum:
            raise ValueError(f"Package asset changed: {rel}")
    data = read_json(package / "payload.json")
    scan_package(package, data["stage"])
    report = {"package_assets_match": True, "stage": data["stage"], "items": len(data["items"]), "identity_leak_hits": 0, "final_GT": False}
    if export:
        payload = read_json(export)
        validate_csv_companion(export, payload)
        if data["stage"] == "semantic":
            report["submitted_records"] = len(validate_semantic(payload, data))
            report["status"] = "semantic_opinion_ready_for_geometry_release"
        else:
            report.update(validate_geometry(payload, data))
        report["export_sha256"] = digest(export)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--export", type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.package, a.export), ensure_ascii=False, indent=2))

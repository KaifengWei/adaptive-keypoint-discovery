"""Build local-only, model-blind human-GT tracing packages.

The generated runtime directory is intentionally ignored by Git because it
contains the administrator identity map.  This script reads only locked V4 val
images and never imports model, teacher, graph, decoder, or test resources.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
from typing import Any

from PIL import Image


PROTOCOL_VERSION = "phenotype-first-pilot-v2"
GEOMETRY_VERSION = "phenotype-geometry-v1"
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
BLIND_LENGTH = 14
PRACTICE_IDS = ("v4_val_0001", "v4_val_0003")
PACKAGE_SPECS = (
    ("practice", "练习与两图dry run", "PRACTICE", 0, True),
    ("rater1_round1", "Rater 1 第一轮正式描迹", "R1", 1, False),
    ("rater1_round2", "Rater 1 第二轮正式描迹", "R1", 2, False),
    ("rater2_round1", "Rater 2 第一轮正式描迹", "R2", 1, False),
)


def random_token(length: int = BLIND_LENGTH) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_locked_records(protocol_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = json.loads((protocol_dir / "phenotype_pilot_selection.json").read_text(encoding="utf-8"))
    if len(selection) != 16:
        raise RuntimeError(f"locked pilot must contain 16 images, found {len(selection)}")
    ids = [row["dataset_id"] for row in selection]
    if len(set(ids)) != 16:
        raise RuntimeError("locked pilot contains duplicate identities")
    if sum(row["pilot_group"] == "core" for row in selection) != 12 or sum(row["pilot_group"] == "diagnostic" for row in selection) != 4:
        raise RuntimeError("locked pilot is not Core12 + Diagnostic4")

    inventory: dict[str, dict[str, str]] = {}
    with (protocol_dir / "phenotype_pilot_morphology_inventory.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            inventory[row["dataset_id"]] = row
    practice: list[dict[str, Any]] = []
    for dataset_id in PRACTICE_IDS:
        if dataset_id in ids:
            raise RuntimeError(f"practice image overlaps locked pilot: {dataset_id}")
        row = dict(inventory[dataset_id])
        row["pilot_group"] = "practice"
        practice.append(row)
    return selection, practice


def normalized_record(row: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    source = Path(str(row.get("image_path", "")))
    if not source.is_absolute():
        source = project_dir / source
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if "\\test\\" in str(source).lower() or "/test/" in str(source).lower():
        raise RuntimeError(f"test path is forbidden: {source}")
    if "\\val\\" not in str(source).lower() and "/val/" not in str(source).lower():
        raise RuntimeError(f"only locked val images are allowed: {source}")
    with Image.open(source) as image:
        width, height = image.size
    bbox_width = float(row["shoot_bbox_width_px"])
    bbox_height = float(row["shoot_bbox_height_px"])
    return {
        "dataset_id": str(row["dataset_id"]),
        "pilot_group": str(row["pilot_group"]),
        "source": source,
        "width": int(width),
        "height": int(height),
        "bbox_diag_px": (bbox_width**2 + bbox_height**2) ** 0.5,
        "source_sha256": sha256_file(source),
    }


def build_new_assignments(pilot: list[dict[str, Any]], practice: list[dict[str, Any]], project_dir: Path) -> list[dict[str, Any]]:
    pilot_rows = [normalized_record(row, project_dir) for row in pilot]
    practice_rows = [normalized_record(row, project_dir) for row in practice]
    used_blind: set[str] = set()
    pair_ids = {row["dataset_id"]: random_token() for row in [*pilot_rows, *practice_rows]}
    assignments: list[dict[str, Any]] = []
    rng = secrets.SystemRandom()
    for package_name, display_name, rater_id, round_id, training_mode in PACKAGE_SPECS:
        package_id = random_token(12)
        source_rows = practice_rows if training_mode else pilot_rows
        ordered = list(source_rows); rng.shuffle(ordered)
        for order, source in enumerate(ordered, 1):
            blind_id = random_token()
            while blind_id in used_blind:
                blind_id = random_token()
            used_blind.add(blind_id)
            assignments.append({
                **source,
                "package_name": package_name,
                "display_name": display_name,
                "package_id": package_id,
                "rater_id": rater_id,
                "measurement_round": round_id,
                "training_mode": training_mode,
                "blind_id": blind_id,
                "image_alias": f"images/{blind_id}.png",
                "randomized_order": order,
                "adjudication_pair_id": pair_ids[source["dataset_id"]],
            })
    return assignments


def write_admin_mapping(runtime: Path, assignments: list[dict[str, Any]]) -> None:
    admin = runtime / "admin"; admin.mkdir(parents=True, exist_ok=True)
    headers = ["blind_id", "dataset_id", "pilot_group", "rater_id", "measurement_round", "image_alias",
               "randomized_order", "package_id", "adjudication_pair_id", "created_at_utc", "mapping_sha256", "mapping_status"]
    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        row = {key: assignment.get(key, "") for key in headers}
        row["created_at_utc"] = created
        row["mapping_status"] = "active_blinded"
        row["mapping_sha256"] = ""
        row["mapping_sha256"] = sha256_text(canonical_json(row))
        rows.append(row)
    mapping_path = admin / "blind_id_admin_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader(); writer.writerows(rows)
    (admin / "ADMIN_ONLY_不要发送给测量者.txt").write_text(
        "该目录含blind ID与真实样本身份映射。人工GT冻结前不得发送给测量者或裁决者，不得提交Git。\n",
        encoding="utf-8",
    )


def write_package(protocol_dir: Path, runtime: Path, package_name: str, assignments: list[dict[str, Any]]) -> None:
    template_dir = protocol_dir / "html_template"
    output = runtime / "packages" / package_name
    images = output / "images"; images.mkdir(parents=True, exist_ok=True)
    rows = sorted([row for row in assignments if row["package_name"] == package_name], key=lambda row: row["randomized_order"])
    if not rows:
        raise RuntimeError(f"no rows for package {package_name}")
    for row in rows:
        shutil.copy2(row["source"], output / row["image_alias"])
    public = {
        "protocol_version": PROTOCOL_VERSION,
        "geometry_protocol_version": GEOMETRY_VERSION,
        "package_id": rows[0]["package_id"],
        "display_name": rows[0]["display_name"],
        "rater_id": rows[0]["rater_id"],
        "measurement_round": rows[0]["measurement_round"],
        "training_mode": bool(rows[0]["training_mode"]),
        "items": [
            {
                "blind_id": row["blind_id"], "image_alias": row["image_alias"], "randomized_order": row["randomized_order"],
                "width": row["width"], "height": row["height"], "bbox_diag_px": row["bbox_diag_px"], "image_sha256": row["source_sha256"],
            }
            for row in rows
        ],
    }
    html = (template_dir / "index.html").read_text(encoding="utf-8").replace("__PACKAGE_DATA_JSON__", json.dumps(public, ensure_ascii=False, separators=(",", ":")))
    (output / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(template_dir / "app.js", output / "app.js")
    shutil.copy2(template_dir / "styles.css", output / "styles.css")
    (output / "package_manifest_public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "使用说明.txt").write_text(
        "使用Edge或Chrome打开index.html。先设置共同地上部基点，再自行判断并逐片添加描迹项目。\n"
        "页面不会提示应有数量。完成全部匿名图片后点击“完成全包并导出”，保存三份文件。\n"
        "Rater 1第二轮应与第一轮间隔3–7天，且不要查看第一轮记录。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent / "runtime")
    parser.add_argument("--refresh", action="store_true", help="refresh HTML/JS/CSS while preserving existing blind IDs and maps")
    args = parser.parse_args()
    protocol_dir = Path(__file__).resolve().parent
    project_dir = protocol_dir.parents[1]
    runtime = args.runtime.resolve()
    if runtime.exists() and args.refresh:
        for package_name, *_ in PACKAGE_SPECS:
            output = runtime / "packages" / package_name
            public = json.loads((output / "package_manifest_public.json").read_text(encoding="utf-8"))
            html = (protocol_dir / "html_template" / "index.html").read_text(encoding="utf-8").replace("__PACKAGE_DATA_JSON__", json.dumps(public, ensure_ascii=False, separators=(",", ":")))
            (output / "index.html").write_text(html, encoding="utf-8")
            shutil.copy2(protocol_dir / "html_template" / "app.js", output / "app.js")
            shutil.copy2(protocol_dir / "html_template" / "styles.css", output / "styles.css")
        print(json.dumps({"status": "refreshed", "blind_ids_preserved": True, "packages": 4}, ensure_ascii=False))
        return
    if runtime.exists():
        raise RuntimeError(f"runtime already exists; preserve the active blind map and do not regenerate: {runtime}")
    pilot, practice = load_locked_records(protocol_dir)
    assignments = build_new_assignments(pilot, practice, project_dir)
    runtime.mkdir(parents=True)
    write_admin_mapping(runtime, assignments)
    for package_name, *_ in PACKAGE_SPECS:
        write_package(protocol_dir, runtime, package_name, assignments)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "geometry_protocol_version": GEOMETRY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_count": 16,
        "practice_count": 2,
        "package_counts": {name: sum(row["package_name"] == name for row in assignments) for name, *_ in PACKAGE_SPECS},
        "unique_blind_ids": len({row["blind_id"] for row in assignments}),
        "mapping_file_sha256": sha256_file(runtime / "admin" / "blind_id_admin_mapping.csv"),
        "test_images_read": 0,
        "model_outputs_read": 0,
    }
    (runtime / "admin" / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

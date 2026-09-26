"""Synthetic-only unit tests: no test pixels, model imports, or GT records."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import check_locked_test_static as checks


def payload() -> dict:
    return {"schema_version": "v4-test-blind-gt-v1", "samples": [{
        "blind_id": "KSPURQWXYZABCD23",
        "image_filename": "images/KSPURQWXYZABCD23.png",
        "image_sha256": hashlib.sha256(b"synthetic asset, not an image").hexdigest(),
        "width": 100, "height": 100,
    }]}


class StaticChecksTest(unittest.TestCase):
    def test_blind_schema(self):
        self.assertEqual(len(checks.validate_public_manifest(payload(), 1)), 1)

    def test_expected_count_forbidden(self):
        data = payload()
        data["samples"][0]["visible_leaf_count"] = 2
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_real_identity_forbidden(self):
        data = payload()
        data["samples"][0]["dataset_id"] = "v4_test_0001"
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_nonopaque_id(self):
        data = payload()
        data["samples"][0]["blind_id"] = "R11_0002"
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_duplicate_blind_id(self):
        data = payload()
        data["samples"].append(copy.deepcopy(data["samples"][0]))
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 2)

    def test_path_escape(self):
        data = payload()
        data["samples"][0]["image_filename"] = "../admin/map.csv"
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_invalid_dimensions(self):
        data = payload()
        data["samples"][0]["width"] = True
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_injected_model_output(self):
        data = payload()
        data["samples"][0]["auto_points"] = [[1, 2]]
        with self.assertRaises(ValueError):
            checks.validate_public_manifest(data, 1)

    def test_manifest_separator_canonicalization(self):
        row = {key: "example" for key in checks.IDENTITY_KEYS}
        row["relative_path"] = "images\\test\\blind.png"
        other = dict(row, relative_path="images/test/blind.png")
        self.assertEqual(checks.canonical_identity([row]), checks.canonical_identity([other]))
        other["output_sha256"] = "changed"
        self.assertNotEqual(checks.canonical_identity([row]), checks.canonical_identity([other]))

    def test_text_leak_patterns(self):
        for text in ("v4_test_0001", "Teacher-direct", "Student-B", "Core 12",
                     "Diagnostic", "visible_leaf_count", "dataset_id", "best.pt"):
            self.assertIsNotNone(checks.LEAK_RE.search(text))
        self.assertIsNone(checks.LEAK_RE.search("请独立判断并自行添加结构路径"))

    def test_public_package_and_admin_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "images").mkdir()
            # Synthetic fixture files are test setup, not project artifacts.
            (folder / payload()["samples"][0]["image_filename"]).write_bytes(b"synthetic asset, not an image")
            (folder / "measurement_manifest.json").write_text(json.dumps(payload()), encoding="utf-8")
            (folder / "index.html").write_text("请独立判断并自行添加结构路径", encoding="utf-8")
            self.assertEqual(checks.audit_public_package(folder, 1)["n"], 1)
            (folder / "admin_mapping.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                checks.audit_public_package(folder, 1)

    def test_missing_or_changed_artifact_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = {"artifacts": {"absent.py": "0" * 64}, "remote_only": []}
            with self.assertRaises(ValueError):
                checks.audit_artifacts(root, ledger)
            (root / "absent.py").write_bytes(b"synthetic code")
            with self.assertRaises(ValueError):
                checks.audit_artifacts(root, ledger)

    def test_split_metadata_and_cross_split_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "data_stage_clean_v4_fullplant_candidate/manifests"
            directory.mkdir(parents=True)
            ledger = {"splits": {}}
            for split in ("train", "val", "test"):
                row = {key: "synthetic" for key in checks.IDENTITY_KEYS}
                row.update(dataset_id=f"{split}_opaque", split=split,
                           source_frame_id=f"frame_{split}", split_group=f"group_{split}",
                           output_sha256=hashlib.sha256(split.encode()).hexdigest(),
                           model_outputs_used_for_selection="0")
                path = directory / f"{split}.csv"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader(); writer.writerow(row)
                ledger["splits"][split] = {
                    "n": 1, "frames": 1, "canonical_sha256": checks.canonical_identity([row]),
                    "local_raw_sha256": checks.sha(path), "remote_raw_sha256": checks.sha(path),
                }
            self.assertEqual(checks.audit_splits(root, ledger)["test"]["n"], 1)
            path = directory / "test.csv"
            text = path.read_text(encoding="utf-8").replace("frame_test", "frame_train")
            path.write_text(text, encoding="utf-8")
            # Updating synthetic hash fixtures must not bypass the disjointness rule.
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            expected = ledger["splits"]["test"]
            expected.update(canonical_sha256=checks.canonical_identity(rows),
                            local_raw_sha256=checks.sha(path), remote_raw_sha256=checks.sha(path))
            with self.assertRaisesRegex(ValueError, "Cross-split frame"):
                checks.audit_splits(root, ledger)


if __name__ == "__main__":
    unittest.main()

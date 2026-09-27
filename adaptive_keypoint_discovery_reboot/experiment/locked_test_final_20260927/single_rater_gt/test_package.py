"""Synthetic raw-schema tests: no real images, models or phenotype GT."""
import copy
import json
import unittest
from package_tools import digest_text, validate_raw, validate_snapshot


class RawSchemaTests(unittest.TestCase):
    def setUp(self):
        self.s = {"blind_id": "A" * 20, "image_sha256": "b" * 64, "width": 900, "height": 350,
                  "image_filename": "images/" + "A" * 20 + ".png"}
        self.r = {k: self.s[k] for k in ("blind_id", "image_sha256", "width", "height")}
        self.r.update(schema_version="blind-session-v1", manifest_sha256="c" * 64, session_id="synthetic-session", rater_id="S1", round=1)
        self.r.update(base_xy=[760, 210], image_note="synthetic only", whole_plant_checked=True,
                      revision=1, technical_revision=None, submitted_at="2026-09-27T00:00:00.000Z",
                      previous_submission_sha256=None, items=[])
        for n, state in enumerate(["measurable", "visible_unmeasurable", "uncertain", "non_target_structure"], 1):
            self.r["items"].append({"gt_id": f"GT{n:02}", "trace_uuid": f"synthetic-{n}",
                "visibility_status": state, "points_px": [[760, 210], [420, 190], [80, 120]] if n == 1 else [],
                "tip_xy": [80, 120] if n == 1 else None, "occlusion": "none", "interpolation_used": "no",
                "confidence": "high", "note": "synthetic reason"})

    def outer(self, records):
        previous, history = None, []
        for n, record in enumerate(records, 1):
            r = copy.deepcopy(record)
            r["revision"], r["previous_submission_sha256"] = n, previous
            text = json.dumps(r, separators=(",", ":"))
            previous = digest_text(text)
            history.append({"snapshot_json": text, "sha256": previous})
        payload = {"schema_version": "blind-raw-v1", "manifest_sha256": "c" * 64, "rater_id": "S1",
                   "samples": [{"blind_id": self.s["blind_id"], "history": history}]}
        text = json.dumps(payload, separators=(",", ":"))
        return {"payload_json": text, "sha256": digest_text(text)}

    def check(self, records):
        return validate_raw(self.outer(records), {"samples": [self.s]}, "c" * 64)

    def test_four_states(self):
        report = self.check([self.r])
        self.assertEqual(report["states"], dict.fromkeys(["measurable", "visible_unmeasurable", "uncertain", "non_target_structure"], 1))

    def test_nonmeasurable_geometry_forbidden(self):
        self.r["items"][1]["points_px"] = [[760, 210], [200, 220]]
        with self.assertRaises(ValueError): self.check([self.r])

    def test_invalid_coordinates(self):
        for value in ([901, 20], [-1, 20], [float("nan"), 20], [True, 20]):
            with self.subTest(value=value):
                r = copy.deepcopy(self.r); r["items"][0]["points_px"][1] = value
                with self.assertRaises(ValueError): self.check([r])

    def test_shared_base_and_tip(self):
        for field in ("base", "tip"):
            r = copy.deepcopy(self.r)
            r["items"][0]["points_px"][0 if field == "base" else -1] = [20, 20]
            with self.assertRaises(ValueError): self.check([r])

    def test_missing_state_and_reason(self):
        for field, index, value in (("visibility_status", 0, ""), ("note", 1, ""), ("confidence", 0, "")):
            r = copy.deepcopy(self.r); r["items"][index][field] = value
            with self.assertRaises(ValueError): self.check([r])

    def test_technical_revision_chain(self):
        r = copy.deepcopy(self.r)
        r["technical_revision"] = {"kind": "invalid_coordinates", "reason": "synthetic click coordinate error"}
        r["items"][0]["points_px"][1] = [430, 190]
        self.assertEqual(self.check([self.r, r])["snapshots"], 2)

    def test_revision_cannot_change_judgment(self):
        for field in ("visibility_status", "note", "confidence", "occlusion", "trace_uuid"):
            r = copy.deepcopy(self.r); r["technical_revision"] = {"kind": "invalid_coordinates", "reason": "test"}
            r["items"][2][field] = "changed"
            with self.assertRaises(ValueError): self.check([self.r, r])

    def test_revision_cannot_change_count(self):
        r = copy.deepcopy(self.r); r["items"].pop()
        r["technical_revision"] = {"kind": "export_failure", "reason": "test"}
        with self.assertRaises(ValueError): self.check([self.r, r])

    def test_hash_corruption(self):
        outer = self.outer([self.r]); outer["payload_json"] += " "
        with self.assertRaises(ValueError): validate_raw(outer, {"samples": [self.s]}, "c" * 64)

    def test_wrong_package(self):
        with self.assertRaises(ValueError): validate_raw(self.outer([self.r]), {"samples": [self.s]}, "d" * 64)

    def test_no_base_for_uncertain_only(self):
        self.r["base_xy"] = None; self.r["items"] = self.r["items"][2:]
        for i,t in enumerate(self.r["items"], 1): t["gt_id"] = f"GT{i:02}"
        self.check([self.r])

    def test_interpolation_not_shared_segment(self):
        self.r["items"][0]["interpolation_used"] = "yes"
        with self.assertRaises(ValueError): self.check([self.r])
        self.r["items"][0]["occlusion"] = "minor"
        self.check([self.r])


if __name__ == "__main__": unittest.main()

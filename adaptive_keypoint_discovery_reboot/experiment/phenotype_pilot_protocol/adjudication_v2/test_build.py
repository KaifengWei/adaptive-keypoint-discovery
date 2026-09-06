import copy
import unittest
from build import VERSION, validate_semantic, validate_geometry


class SemanticGateTests(unittest.TestCase):
    def setUp(self):
        self.leaf = {"leaf_id": "random-leaf", "point": [30, 20], "status": "measurable", "reason": "none", "confidence": "high", "notes": ""}
        self.record = {"blind_id": "random-image", "image_sha256": "abc", "submitted": True, "submitted_at_utc": "2026-09-06T00:00:00Z", "review_complete": True, "candidates_seen": False, "notes": "", "leaves": [self.leaf]}
        self.data = {"package_id": "random-package", "items": [{"blind_id": "random-image", "image_sha256": "abc", "width": 100, "height": 50}]}
        self.payload = {"build_version": VERSION, "package_id": "random-package", "stage": "semantic", "records": [self.record]}

    def test_valid_and_unmeasurable(self):
        self.assertEqual(len(validate_semantic(self.payload, self.data)), 1)
        self.leaf.update(status="visible_unmeasurable", reason="cropped_tip", notes="A visible truncated end")
        self.assertEqual(len(validate_semantic(self.payload, self.data)), 1)

    def test_invalid_gate_inputs(self):
        for field, value in [("submitted", False), ("candidates_seen", True), ("image_sha256", "changed")]:
            with self.subTest(field=field):
                p = copy.deepcopy(self.payload)
                p["records"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_semantic(p, self.data)

    def test_crop_cannot_be_measurable_and_points_are_bounded(self):
        self.leaf["reason"] = "cropped_tip"
        with self.assertRaises(ValueError):
            validate_semantic(self.payload, self.data)
        self.leaf["reason"] = "none"
        for point in ([100, 0], [-1, 0], [float("nan"), 2], [True, 2]):
            self.leaf["point"] = point
            with self.assertRaises(ValueError):
                validate_semantic(self.payload, self.data)

    def test_geometry_preserves_semantics_and_never_freezes(self):
        data = copy.deepcopy(self.data)
        data["semantic_export_sha256"] = "frozen"
        data["items"][0].update(semantic=copy.deepcopy(self.record), sets=[{"label": "A", "traces": [{"candidate_trace_id": "T01"}]}])
        p = copy.deepcopy(self.payload)
        p.update(stage="geometry", semantic_export_sha256="frozen")
        p["records"][0]["candidates_seen"] = True
        p["records"][0]["leaves"][0].update(geometry_choice="A:T01", geometry_confidence="high", geometry_notes="")
        self.assertFalse(validate_geometry(p, data)["final_GT"])
        p["records"][0]["leaves"][0]["status"] = "non_leaf"
        with self.assertRaises(ValueError):
            validate_geometry(p, data)


if __name__ == "__main__":
    unittest.main()

import unittest

from build_minimal_gt_resolution import validate_record


class MinimalResolutionValidationTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "blind_id": "ABCDEFGHIJKLMNOP",
            "image_sha256": "a" * 64,
            "question": {"options": [{"code": "keep"}, {"code": "cannot_determine"}]},
        }
        self.record = {
            "blind_id": "ABCDEFGHIJKLMNOP",
            "image_sha256": "a" * 64,
            "decision": "keep",
            "notes": "",
            "jointly_confirmed": True,
            "submitted": True,
            "history": [],
        }

    def test_valid_record(self):
        validate_record(self.record, self.item)

    def test_confirmation_is_required(self):
        self.record["jointly_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "confirmation"):
            validate_record(self.record, self.item)

    def test_unknown_decision_is_rejected(self):
        self.record["decision"] = "invented"
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_record(self.record, self.item)

    def test_unresolved_decision_needs_note(self):
        self.record["decision"] = "cannot_determine"
        with self.assertRaisesRegex(ValueError, "note"):
            validate_record(self.record, self.item)


if __name__ == "__main__":
    unittest.main()

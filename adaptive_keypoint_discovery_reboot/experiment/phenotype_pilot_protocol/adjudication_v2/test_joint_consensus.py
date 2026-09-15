"""Unit tests for joint-consensus record gates."""

import copy
import unittest

from build_joint_consensus import validate_record


class JointConsensusValidationTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "blind_id": "ABCDEFGHIJKLMNOP",
            "image_sha256": "a" * 64,
            "sets": [
                {"label": "A", "traces": [{"candidate_trace_id": "T01"}]},
                {"label": "B", "traces": [{"candidate_trace_id": "T01"}]},
            ],
        }
        self.record = {
            "blind_id": "ABCDEFGHIJKLMNOP",
            "image_sha256": "a" * 64,
            "submitted": True,
            "revision": 1,
            "history": [],
            "decision": "consensus_ready",
            "agreed_visible_leaf_count": 1,
            "overall_confidence": "high",
            "checks": {"identity_count": True, "tip_completeness": True, "common_base": True, "shared_path": True},
            "leaves": [{
                "consensus_leaf_uuid": "leaf-1", "source_r2_index": 1,
                "semantic_status": "measurable", "path_choice": "A:T01",
                "equivalent_choices": ["B:T01"], "confidence": "high",
                "description": "", "notes": "",
            }],
            "notes": "", "rater1_confirmed": True, "rater2_confirmed": True,
        }

    def test_ready_record_passes(self):
        validate_record(self.record, self.item)

    def test_ready_record_rejects_redraw(self):
        record = copy.deepcopy(self.record)
        record["leaves"][0]["path_choice"] = "needs_redraw"
        record["leaves"][0]["notes"] = "candidate endpoint incomplete"
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            validate_record(record, self.item)

    def test_redraw_requires_redraw_leaf(self):
        record = copy.deepcopy(self.record)
        record["decision"] = "redraw_required"
        with self.assertRaisesRegex(ValueError, "no redraw"):
            validate_record(record, self.item)

    def test_both_raters_must_confirm(self):
        record = copy.deepcopy(self.record)
        record["rater2_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "Both raters"):
            validate_record(record, self.item)

    def test_visible_count_must_match_records(self):
        record = copy.deepcopy(self.record)
        record["agreed_visible_leaf_count"] = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_record(record, self.item)


if __name__ == "__main__":
    unittest.main()

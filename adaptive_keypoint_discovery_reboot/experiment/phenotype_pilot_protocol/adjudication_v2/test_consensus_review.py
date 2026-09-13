"""Tests for navigation flags; no test may promote an opinion to GT."""

import unittest

from build_consensus_review import classify_case


class ConsensusReviewFlagsTests(unittest.TestCase):
    def setUp(self):
        self.r1 = {"label": "A", "traces": [{"candidate_trace_id": "T01"}, {"candidate_trace_id": "T02"}]}

    def test_same_source_is_only_a_quality_check(self):
        r2 = {"leaves": [{"geometry_choice": "A:T01"}, {"geometry_choice": "A:T02"}]}
        self.assertEqual(classify_case(self.r1, r2), [])

    def test_mixed_source_needs_review(self):
        r2 = {"leaves": [{"geometry_choice": "A:T01"}, {"geometry_choice": "B:T02"}]}
        self.assertEqual(classify_case(self.r1, r2), [
            "第二位测量者混用候选来源", "两人候选来源不完全一致"
        ])

    def test_pending_and_count_mismatch_remain_visible(self):
        r2 = {"leaves": [{"geometry_choice": "needs_redraw"}]}
        self.assertEqual(classify_case(self.r1, r2), [
            "待重描或暂存争议", "结构数量不同"
        ])


if __name__ == "__main__":
    unittest.main()

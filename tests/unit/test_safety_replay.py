from __future__ import annotations

import unittest
from typing import Any

from src.twin.decision.safety_policy import load_safety_policy
from src.twin.decision.safety_replay import candidate_from_row, run_gate, with_minimum_relief

POLICY = load_safety_policy()
M_U, M_R = 0.036629233, 0.009170841


def row(name: str, share: float = 0.25, u_src: float = 1.04, u_post: float = 0.9) -> dict[str, Any]:
    return dict(
        attempt_id=name, switch_id="s3", source_controller="c1", target_controller="c2",
        src_utilization=u_src, dst_utilization=0.60, switch_control_load_share=share,
        u_dst_post=u_post, r_dst95_post=0.8,
    )


def predicted(u_dst: float) -> dict[str, list[float]]:
    one = {"u_src_post": 0.8, "u_dst_post": u_dst, "r_src95_post": 0.78,
           "r_dst95_post": 0.79, "r95_post": 0.79, "i_post": 0.05}
    return {name: [value] for name, value in one.items()}


class MarginTest(unittest.TestCase):
    def test_gate_a_accepts_and_gate_b_rejects_a_close_call(self) -> None:
        rows = [row("a", u_post=1.05)]
        a = run_gate(rows, predicted(0.98), POLICY, 0.0, 0.0)
        b = run_gate(rows, predicted(0.98), POLICY, M_U, M_R)
        self.assertEqual((a.metrics.unsafe_accepted, a.metrics.unsafe_rejected), (1, 0))
        self.assertEqual((b.metrics.unsafe_accepted, b.metrics.unsafe_rejected), (0, 1))
        self.assertEqual(b.failed_reasons, {"DST_UTILIZATION_UNSAFE": 1})
        self.assertEqual(a.failed_reasons, {})

    def test_safe_case_is_accepted_by_both(self) -> None:
        rows = [row("a", u_post=0.9)]
        for margins in ((0.0, 0.0), (M_U, M_R)):
            result = run_gate(rows, predicted(0.85), POLICY, *margins)
            self.assertEqual(result.metrics.safe_accepted, 1)


class SkippedAndReliefTest(unittest.TestCase):
    def test_unavailable_checks_are_counted_as_skipped(self) -> None:
        result = run_gate([row("a")], predicted(0.85), POLICY, M_U, M_R)
        self.assertEqual(result.skipped_checks["destination_not_at_risk:NOT_AVAILABLE"], 1)
        self.assertEqual(result.skipped_checks["dst_response:NOT_CONFIGURED"], 1)
        self.assertEqual(result.metrics.rvr_status, "NOT_CONFIGURED")

    def test_minimum_relief_rejects_a_small_migration(self) -> None:
        small = row("a", share=0.1, u_src=0.2)  # relief 0.02
        policy = with_minimum_relief(POLICY, M_U, "test")
        plain = run_gate([small], predicted(0.85), POLICY, M_U, M_R)
        floor = run_gate([small], predicted(0.85), policy, M_U, M_R)
        self.assertEqual(plain.failed_reasons, {})
        self.assertEqual(floor.failed_reasons, {"INSUFFICIENT_IMPROVEMENT": 1})

    def test_relief_helper_changes_nothing_else(self) -> None:
        other = with_minimum_relief(POLICY, None, "off")
        self.assertIsNone(other.minimum_source_relief.value)
        self.assertEqual(other.utilization_safe_limit, POLICY.utilization_safe_limit)


class InputTest(unittest.TestCase):
    def test_candidate_id_is_built_from_the_row(self) -> None:
        self.assertEqual(candidate_from_row(row("att-1")).candidate_id, "att-1:s3:c1->c2")

    def test_series_length_must_match_rows(self) -> None:
        with self.assertRaises(ValueError):
            run_gate([row("a"), row("b")], predicted(0.85), POLICY, M_U, M_R)


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timezone

from src.schemas import decision as d
from src.twin.outcome.predictor import PostMigrationOutcomePrediction

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)


def candidate(**over: object) -> d.MigrationCandidate:
    args: dict = dict(snapshot_id="snap-1", switch_id="s3", source_controller="c1",
                      target_controller="c2", source_utilization=1.04,
                      target_utilization=0.60, switch_control_load_share=0.25,
                      generated_at=NOW)
    args.update(over)
    return d.MigrationCandidate(**args)


def prediction(**over: object) -> PostMigrationOutcomePrediction:
    args: dict = dict(snapshot_id="snap-1", switch_id="s3", source_controller="c1",
                      target_controller="c2", model_name="post-migration-outcome",
                      model_version="post-migration-v1", predicted_src_utilization=0.80,
                      predicted_dst_utilization=0.90, predicted_src_response_p95_ms=0.78,
                      predicted_dst_response_p95_ms=0.79, predicted_worst_response_p95_ms=0.79,
                      predicted_imbalance=0.05, dst_utilization_margin=0.037,
                      response_margin_ms=0.009, inference_time_ms=2.5, status="READY")
    args.update(over)
    return PostMigrationOutcomePrediction(**args)


def decision(**over: object) -> d.SafetyDecision:
    args: dict = dict(candidate_id="snap-1:s3:c1->c2", safe=True,
                      passed_checks=tuple(c for c in d.GATE_CHECKS
                                          if c not in ("dst_response", "minimum_improvement")),
                      failed_checks=(),
                      skipped_checks=(("dst_response", d.NOT_CONFIGURED),
                                      ("minimum_improvement", d.NOT_CONFIGURED)),
                      policy_version="safety-policy-v1",
                      outcome_model_version="post-migration-v1", decision_time_ms=3.0)
    args.update(over)
    return d.SafetyDecision(**args)


class ReasonCodeTest(unittest.TestCase):
    def test_codes_are_unique_and_cover_every_check(self) -> None:
        self.assertEqual(len(d.REASON_CODES), len(set(d.REASON_CODES)))
        self.assertEqual(len(d.GATE_CHECKS), 11)
        self.assertTrue(set(d.CHECK_REASONS.values()) <= set(d.REASON_CODES))


class MigrationCandidateTest(unittest.TestCase):
    def test_id_is_derived_and_overload_is_not_clamped(self) -> None:
        c = candidate()
        self.assertEqual(c.candidate_id, "snap-1:s3:c1->c2")
        self.assertEqual(c.source_utilization, 1.04)
        self.assertAlmostEqual(c.moved_utilization, 0.26)
        self.assertEqual(c.to_dict()["generated_at"], NOW.isoformat())

    def test_invalid_candidates(self) -> None:
        for over in ({"target_controller": "c1"}, {"switch_control_load_share": 1.2},
                     {"source_utilization": -0.1}, {"source_utilization": float("nan")},
                     {"generated_at": datetime(2026, 11, 5)}, {"candidate_id": "other"},
                     {"switch_id": " "}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                candidate(**over)


class EvaluatedCandidateTest(unittest.TestCase):
    def test_from_ready_prediction(self) -> None:
        e = d.EvaluatedCandidate.from_prediction(candidate(), prediction())
        self.assertAlmostEqual(e.dst_upper_bound, 0.937)
        self.assertAlmostEqual(e.response_upper_bound_ms, 0.799)
        self.assertAlmostEqual(e.predicted_src_change, 0.24)
        self.assertEqual(e.outcome_model_version, "post-migration-v1")

    def test_not_ready_or_other_action_is_refused(self) -> None:
        for p in (prediction(status="NOT_READY", reason="INSUFFICIENT_HISTORY"),
                  prediction(switch_id="s4"), prediction(predicted_dst_utilization=None)):
            with self.subTest(p=p), self.assertRaises(ValueError):
                d.EvaluatedCandidate.from_prediction(candidate(), p)


class SafetyDecisionTest(unittest.TestCase):
    def test_safe_with_skipped_checks(self) -> None:
        self.assertTrue(decision().safe)
        self.assertEqual(decision().to_dict()["skipped_checks"][0], ["dst_response",
                                                                     "NOT_CONFIGURED"])

    def test_rejection_may_stop_early(self) -> None:
        r = decision(safe=False, passed_checks=("snapshot_valid",),
                     failed_checks=(d.DST_UTILIZATION_UNSAFE,), skipped_checks=())
        self.assertFalse(r.safe)

    def test_inconsistent_decisions(self) -> None:
        cases = (
            {"failed_checks": (d.DST_UTILIZATION_UNSAFE,)},          # safe but failed
            {"safe": False},                                         # unsafe, no reason
            {"passed_checks": ("snapshot_valid",)},                  # safe, checks missing
            {"failed_checks": ("BAD",), "safe": False},
            {"skipped_checks": (("dst_response", "LATER"),)},
            {"passed_checks": d.GATE_CHECKS},                        # passed and skipped
            {"decision_time_ms": -1.0},
        )
        for over in cases:
            with self.subTest(over=over), self.assertRaises(ValueError):
                decision(**over)


if __name__ == "__main__":
    unittest.main()

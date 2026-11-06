from __future__ import annotations

import dataclasses
import json
import unittest
from datetime import datetime, timezone
from typing import Any

from src.schemas import decision as d
from src.twin.decision import safety_policy as sp
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.outcome.predictor import PostMigrationOutcomePrediction

NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)
POLICY = sp.load_safety_policy()
ALL_CHECKS = set(d.GATE_CHECKS)


def candidate(**over: Any) -> d.MigrationCandidate:
    args: dict[str, Any] = dict(
        snapshot_id="snap-1", switch_id="s3", source_controller="c1", target_controller="c2",
        source_utilization=1.04, target_utilization=0.60, switch_control_load_share=0.25,
        generated_at=NOW,
    )
    args.update(over)
    return d.MigrationCandidate(**args)


def prediction(**over: Any) -> PostMigrationOutcomePrediction:
    args: dict[str, Any] = dict(
        snapshot_id="snap-1", switch_id="s3", source_controller="c1", target_controller="c2",
        model_name="post-migration-outcome", model_version="post-migration-v1",
        predicted_src_utilization=0.80, predicted_dst_utilization=0.90,
        predicted_src_response_p95_ms=0.78, predicted_dst_response_p95_ms=0.79,
        predicted_worst_response_p95_ms=0.79, predicted_imbalance=0.05,
        dst_utilization_margin=0.037, response_margin_ms=0.009, inference_time_ms=2.5,
        status="READY",
    )
    args.update(over)
    return PostMigrationOutcomePrediction(**args)


def not_ready() -> PostMigrationOutcomePrediction:
    return PostMigrationOutcomePrediction(
        snapshot_id="snap-1", switch_id="s3", source_controller="c1", target_controller="c2",
        model_name="post-migration-outcome", model_version="post-migration-v1",
        status="NOT_READY", reason="INSUFFICIENT_HISTORY",
    )


def context(**over: Any) -> GateContext:
    args: dict[str, Any] = dict(
        snapshot_valid=True, destination_connected=True, active_transaction=False,
        destination_forecast_risk=False,
    )
    args.update(over)
    return GateContext(**args)


def policy(**over: Any) -> sp.SafetyPolicy:
    return dataclasses.replace(POLICY, **over)


def limit(value: float) -> sp.Limit:
    return sp.Limit(value, source="test")


def run(pred: Any = None, ctx: GateContext | None = None, pol: sp.SafetyPolicy | None = None,
        cand: d.MigrationCandidate | None = None, **kw: Any) -> d.SafetyDecision:
    gate = SafetyGate(pol or POLICY)
    return gate.evaluate(cand or candidate(), pred if pred is not None else prediction(),
                         ctx or context(), **kw)


class SafeCandidateTest(unittest.TestCase):
    def test_safe_candidate_with_unset_limits_skips_and_reports_them(self) -> None:
        decision = run()
        self.assertTrue(decision.safe)
        self.assertEqual(decision.failed_checks, ())
        self.assertEqual(
            set(decision.skipped_checks),
            {("quality_policy", d.NOT_CONFIGURED), ("dst_response", d.NOT_CONFIGURED),
             ("minimum_improvement", d.NOT_CONFIGURED)},
        )
        self.assertFalse(set(decision.passed_checks) & {c for c, _ in decision.skipped_checks})
        self.assertEqual(
            set(decision.passed_checks) | {c for c, _ in decision.skipped_checks}, ALL_CHECKS)

    def test_passed_checks_keep_the_gate_order(self) -> None:
        order = [c for c in d.GATE_CHECKS if c in run().passed_checks]
        self.assertEqual(list(run().passed_checks), order)

    def test_decision_carries_the_numbers_it_judged(self) -> None:
        decision = run()
        self.assertEqual(decision.candidate_id, "snap-1:s3:c1->c2")
        self.assertAlmostEqual(decision.dst_upper_bound or 0.0, 0.937)
        self.assertAlmostEqual(decision.moved_utilization or 0.0, 0.26)
        self.assertEqual(decision.policy_version, "safety-policy-v1")
        self.assertEqual(decision.outcome_model_version, "post-migration-v1")

    def test_decision_is_deterministic_and_serialisable(self) -> None:
        first = dataclasses.replace(run(), decision_time_ms=0.0)
        second = dataclasses.replace(run(), decision_time_ms=0.0)
        self.assertEqual(first, second)
        self.assertTrue(json.dumps(first.to_dict()))


class StateChecksTest(unittest.TestCase):
    def test_invalid_snapshot(self) -> None:
        self.assertEqual(run(ctx=context(snapshot_valid=False)).failed_checks,
                         (d.INVALID_SNAPSHOT,))

    def test_quality_below_policy_and_boundary(self) -> None:
        pol = policy(minimum_quality_score=limit(0.8))
        self.assertEqual(run(pol=pol, ctx=context(quality_score=0.7)).failed_checks,
                         (d.QUALITY_BELOW_POLICY,))
        self.assertTrue(run(pol=pol, ctx=context(quality_score=0.8)).safe)

    def test_quality_missing_is_skipped_not_passed(self) -> None:
        decision = run(pol=policy(minimum_quality_score=limit(0.8)))
        self.assertIn(("quality_policy", d.NOT_AVAILABLE), decision.skipped_checks)
        self.assertNotIn("quality_policy", decision.passed_checks)

    def test_destination_disconnected(self) -> None:
        self.assertEqual(run(ctx=context(destination_connected=False)).failed_checks,
                         (d.DESTINATION_DISCONNECTED,))

    def test_destination_forecast_risk(self) -> None:
        decision = run(ctx=context(destination_forecast_risk=True))
        self.assertEqual(decision.failed_checks, (d.DESTINATION_ALREADY_AT_RISK,))

    def test_no_forecast_is_skipped_as_not_available(self) -> None:
        decision = run(ctx=context(destination_forecast_risk=None))
        self.assertIn(("destination_not_at_risk", d.NOT_AVAILABLE), decision.skipped_checks)
        self.assertNotIn("destination_not_at_risk", decision.passed_checks)

    def test_active_transaction_conflict(self) -> None:
        self.assertEqual(run(ctx=context(active_transaction=True)).failed_checks,
                         (d.ACTIVE_TRANSACTION_CONFLICT,))

    def test_cooldown_active_boundary_and_first_migration(self) -> None:
        self.assertEqual(run(ctx=context(seconds_since_last_migration=10.0)).failed_checks,
                         (d.COOLDOWN_ACTIVE,))
        self.assertTrue(run(ctx=context(seconds_since_last_migration=37.0)).safe)
        self.assertTrue(run(ctx=context(seconds_since_last_migration=None)).safe)

    def test_unset_cooldown_is_skipped(self) -> None:
        decision = run(pol=policy(cooldown_seconds=sp.Limit(None, reason="test")),
                       ctx=context(seconds_since_last_migration=1.0))
        self.assertIn(("cooldown_clear", d.NOT_CONFIGURED), decision.skipped_checks)

    def test_negative_elapsed_time_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            context(seconds_since_last_migration=-1.0)


class OutcomeChecksTest(unittest.TestCase):
    def test_destination_utilization_boundary_is_inclusive(self) -> None:
        exact = prediction(predicted_dst_utilization=0.96, dst_utilization_margin=0.04)
        self.assertTrue(run(exact).safe)
        over = prediction(predicted_dst_utilization=0.9600011, dst_utilization_margin=0.04)
        self.assertEqual(run(over).failed_checks, (d.DST_UTILIZATION_UNSAFE,))

    def test_margin_is_added_to_the_prediction(self) -> None:
        near = prediction(predicted_dst_utilization=0.95, dst_utilization_margin=0.037)
        self.assertTrue(run(near).safe)
        wide = prediction(predicted_dst_utilization=0.95, dst_utilization_margin=0.06)
        self.assertEqual(run(wide).failed_checks, (d.DST_UTILIZATION_UNSAFE,))

    def test_overload_is_not_clamped(self) -> None:
        decision = run(prediction(predicted_dst_utilization=1.12))
        self.assertEqual(decision.failed_checks, (d.DST_UTILIZATION_UNSAFE,))
        self.assertAlmostEqual(decision.predicted_dst_utilization or 0.0, 1.12)
        self.assertAlmostEqual(decision.dst_upper_bound or 0.0, 1.157)

    def test_response_limit_and_boundary(self) -> None:
        pol = policy(response_p95_max_ms=limit(0.8))
        self.assertTrue(run(prediction(predicted_dst_response_p95_ms=0.791), pol=pol).safe)
        late = prediction(predicted_dst_response_p95_ms=0.795)
        self.assertEqual(run(late, pol=pol).failed_checks, (d.DST_RESPONSE_UNSAFE,))

    def test_unset_response_limit_is_skipped(self) -> None:
        self.assertIn(("dst_response", d.NOT_CONFIGURED), run().skipped_checks)

    def test_source_relief_must_be_positive(self) -> None:
        self.assertEqual(run(cand=candidate(switch_control_load_share=0.0)).failed_checks,
                         (d.NO_SOURCE_RELIEF,))
        off = policy(require_positive_source_relief=False)
        decision = run(pol=off, cand=candidate(switch_control_load_share=0.0))
        self.assertTrue(decision.safe)
        self.assertIn(("source_relief_positive", d.NOT_CONFIGURED), decision.skipped_checks)

    def test_minimum_improvement_and_boundary(self) -> None:
        self.assertEqual(run(pol=policy(minimum_source_relief=limit(0.3))).failed_checks,
                         (d.INSUFFICIENT_IMPROVEMENT,))
        self.assertTrue(run(pol=policy(minimum_source_relief=limit(0.26))).safe)

    def test_rising_source_does_not_reject_a_proactive_migration(self) -> None:
        rising = prediction(predicted_src_utilization=1.10)
        self.assertTrue(run(rising).safe)


class NotReadyTest(unittest.TestCase):
    def test_not_ready_prediction_rejects_without_fallback(self) -> None:
        decision = run(not_ready())
        self.assertFalse(decision.safe)
        self.assertEqual(decision.failed_checks, (d.OUTCOME_PREDICTOR_NOT_READY,))
        self.assertIn(("dst_utilization", d.NOT_AVAILABLE), decision.skipped_checks)
        self.assertIn(("dst_response", d.NOT_CONFIGURED), decision.skipped_checks)
        self.assertIsNone(decision.predicted_dst_utilization)
        self.assertIn("source_relief_positive", decision.passed_checks)

    def test_missing_prediction_rejects(self) -> None:
        gate = SafetyGate(POLICY)
        decision = gate.evaluate(candidate(), None, context())
        self.assertEqual(decision.failed_checks, (d.OUTCOME_PREDICTOR_NOT_READY,))

    def test_prediction_of_another_model_is_refused(self) -> None:
        decision = run(prediction(model_version="analytical-v0"))
        self.assertEqual(decision.failed_checks, (d.OUTCOME_PREDICTOR_NOT_READY,))
        self.assertEqual(decision.outcome_model_version, "analytical-v0")

    def test_prediction_of_another_action_is_refused(self) -> None:
        decision = run(prediction(switch_id="s4"))
        self.assertEqual(decision.failed_checks, (d.OUTCOME_PREDICTOR_NOT_READY,))


class AllReasonsTest(unittest.TestCase):
    def test_every_failing_reason_is_collected_in_gate_order(self) -> None:
        ctx = context(snapshot_valid=False, destination_connected=False,
                      active_transaction=True, seconds_since_last_migration=5.0,
                      destination_forecast_risk=True)
        decision = run(not_ready(), ctx=ctx, cand=candidate(switch_control_load_share=0.0))
        self.assertEqual(
            decision.failed_checks,
            (d.INVALID_SNAPSHOT, d.DESTINATION_DISCONNECTED, d.DESTINATION_ALREADY_AT_RISK,
             d.ACTIVE_TRANSACTION_CONFLICT, d.COOLDOWN_ACTIVE, d.OUTCOME_PREDICTOR_NOT_READY,
             d.NO_SOURCE_RELIEF),
        )
        self.assertFalse(decision.safe)

    def test_stop_at_first_failure_keeps_only_the_first_reason(self) -> None:
        ctx = context(snapshot_valid=False, active_transaction=True)
        decision = run(ctx=ctx, stop_at_first_failure=True)
        self.assertEqual(decision.failed_checks, (d.INVALID_SNAPSHOT,))
        self.assertEqual(decision.passed_checks, ())

    def test_one_failure_among_passes_is_never_compensated(self) -> None:
        best = prediction(predicted_dst_utilization=1.2, predicted_imbalance=0.0,
                          predicted_src_utilization=0.1)
        self.assertFalse(run(best).safe)


if __name__ == "__main__":
    unittest.main()

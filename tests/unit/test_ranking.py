from __future__ import annotations

import dataclasses
import json
import unittest
from datetime import datetime, timezone
from typing import Any

from src.schemas import decision as d
from src.twin.decision import ranking as rk
from src.twin.decision import ranking_policy as rp
from src.twin.decision import safety_policy as sp
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.outcome.predictor import PostMigrationOutcomePrediction

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
SAFETY = dataclasses.replace(
    sp.load_safety_policy(),
    minimum_source_relief=sp.Limit(None, reason="test baseline: limit not configured"),
)
GATE = SafetyGate(SAFETY)
POLICY = rp.load_ranking_policy()
CTX = GateContext(snapshot_valid=True, destination_connected=True, active_transaction=False,
                  destination_forecast_risk=False)
UTIL = {"c1": 1.04, "c2": 0.60}


def option(switch: str, *, src_after: float = 0.80, dst_after: float = 0.90,
           imbalance: float = 0.05, share: float = 0.25, target: str = "c2",
           util: dict[str, float] | None = None,
           ) -> tuple[d.SafetyDecision, d.EvaluatedCandidate | None]:
    u = util or UTIL
    cand = d.MigrationCandidate(
        snapshot_id="snap-1", switch_id=switch, source_controller="c1",
        target_controller=target, source_utilization=u["c1"], target_utilization=u[target],
        switch_control_load_share=share, generated_at=NOW,
    )
    pred = PostMigrationOutcomePrediction(
        snapshot_id="snap-1", switch_id=switch, source_controller="c1",
        target_controller=target, model_name="post-migration-outcome",
        model_version="post-migration-v1", predicted_src_utilization=src_after,
        predicted_dst_utilization=dst_after, predicted_src_response_p95_ms=0.78,
        predicted_dst_response_p95_ms=0.79, predicted_worst_response_p95_ms=0.79,
        predicted_imbalance=imbalance, dst_utilization_margin=0.037,
        response_margin_ms=0.009, inference_time_ms=1.0, status="READY",
    )
    decision = GATE.evaluate(cand, pred, CTX)
    evaluated = d.EvaluatedCandidate.from_prediction(cand, pred)
    return decision, evaluated


def rank(items: list[Any], util: dict[str, float] | None = None,
         policy: rp.RankingPolicy = POLICY, **ctx: Any) -> rk.RankingResult:
    context = rk.RankingContext(controller_utilization=util or UTIL, utilization_limit=1.0, **ctx)
    return rk.rank_candidates(items, context, policy)


class RawMetricsTest(unittest.TestCase):
    def test_raw_values_follow_the_definitions(self) -> None:
        result = rank([option("s3")])
        r = result.ranked[0]
        self.assertAlmostEqual(r.source_relief, 1.04 - 0.80)
        self.assertAlmostEqual(r.destination_headroom, 1.0 - (0.90 + 0.037))
        self.assertAlmostEqual(r.imbalance_gain, abs(1.04 - 0.60) / 2 - 0.05)
        self.assertAlmostEqual(r.max_utilization_gain, 1.04 - 0.90)
        self.assertIsNone(r.response_score)
        self.assertIsNone(r.migration_cost_score)

    def test_one_safe_candidate_scores_the_equal_value(self) -> None:
        result = rank([option("s3")])
        self.assertEqual(result.status, rk.SELECTED)
        self.assertAlmostEqual(result.ranked[0].total_score, rk.EQUAL_NORM)

    def test_minmax(self) -> None:
        self.assertEqual(rk.minmax([]), [])
        self.assertEqual(rk.minmax([2.0, 2.0]), [0.5, 0.5])
        self.assertEqual(rk.minmax([1.0, 3.0, 2.0]), [0.0, 1.0, 0.5])


class SafetyFirstTest(unittest.TestCase):
    def test_unsafe_candidate_is_never_ranked(self) -> None:
        unsafe = option("s1", src_after=0.50, dst_after=0.97, imbalance=0.0)
        self.assertFalse(unsafe[0].safe)
        result = rank([unsafe, option("s3")])
        self.assertEqual(result.n_candidates, 2)
        self.assertEqual(result.n_safe, 1)
        self.assertEqual([r.switch_id for r in result.ranked], ["s3"])
        self.assertEqual(result.selected_candidate_id, "snap-1:s3:c1->c2")

    def test_nothing_safe_is_no_safe_action(self) -> None:
        result = rank([option("s1", dst_after=0.97)])
        self.assertEqual(result.status, rk.NO_SAFE_ACTION)
        self.assertIsNone(result.selected_candidate_id)
        self.assertEqual(rank([]).status, rk.NO_SAFE_ACTION)

    def test_no_lower_bottleneck_is_no_beneficial_action(self) -> None:
        util = {"c1": 0.70, "c2": 0.30, "c3": 0.99}
        result = rank([option("s3", src_after=0.60, dst_after=0.40, util=util)], util=util)
        self.assertEqual(result.status, rk.NO_BENEFICIAL_ACTION)
        self.assertIsNone(result.selected_candidate_id)
        self.assertEqual(len(result.ranked), 1)


class OrderTest(unittest.TestCase):
    def test_better_outcome_ranks_first_and_scores_stay_in_unit_range(self) -> None:
        items = [option("s1", src_after=0.95, dst_after=0.70, imbalance=0.13),
                 option("s2", src_after=0.75, dst_after=0.85, imbalance=0.05),
                 option("s3", src_after=0.85, dst_after=0.80, imbalance=0.03)]
        result = rank(items)
        self.assertEqual([r.rank for r in result.ranked], [1, 2, 3])
        for r in result.ranked:
            self.assertGreaterEqual(r.total_score, 0.0)
            self.assertLessEqual(r.total_score, 1.0)
        scores = [r.total_score for r in result.ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result.selected_candidate_id, result.ranked[0].candidate_id)

    def test_ties_are_broken_by_upper_bound_then_candidate_id(self) -> None:
        tied = rank([option("s3"), option("s10")])
        self.assertEqual([r.switch_id for r in tied.ranked], ["s10", "s3"])  # lexical id
        low = option("s3", dst_after=0.89)[1]
        high = option("s10", dst_after=0.90)[1]
        assert low is not None and high is not None
        keys = sorted([rk.order_key(0.5, high), rk.order_key(0.5 + 1e-14, low)])
        self.assertEqual(keys[0][2], low.candidate.candidate_id)  # equal score: lower bound
        self.assertLess(rk.order_key(0.6, high), rk.order_key(0.5, low))  # score first

    def test_input_order_does_not_change_the_result(self) -> None:
        items = [option("s1", src_after=0.95), option("s2", src_after=0.75),
                 option("s3", src_after=0.85)]
        a = rank(items).to_dict()
        b = rank(list(reversed(items))).to_dict()
        for data in (a, b):
            data.pop("ranking_time_ms")
        self.assertEqual(a, b)
        json.dumps(a)

    def test_cost_when_enabled_prefers_cheaper(self) -> None:
        policy = dataclasses.replace(POLICY, components=tuple(
            dataclasses.replace(c, enabled=True) if c.name == "migration_cost" else c
            for c in POLICY.components))
        items = [option("s1"), option("s2")]
        cost = {"snap-1:s1:c1->c2": 3.0, "snap-1:s2:c1->c2": 1.0}
        result = rank(items, policy=policy, migration_cost=cost)
        self.assertEqual(result.ranked[0].switch_id, "s2")


class ContractTest(unittest.TestCase):
    def test_refuses_mixed_policies_and_missing_inputs(self) -> None:
        decision, evaluated = option("s3")
        other = dataclasses.replace(POLICY, safety_policy_version="safety-policy-v2")
        with self.assertRaises(ValueError):
            rank([(decision, evaluated)], policy=other)
        with self.assertRaises(ValueError):
            rank([(decision, None)])
        with self.assertRaises(ValueError):
            rank([(decision, option("s4")[1])])
        response_on = dataclasses.replace(POLICY, components=tuple(
            dataclasses.replace(c, enabled=True) if c.name == "response" else c
            for c in POLICY.components))
        with self.assertRaises(ValueError):
            rank([(decision, evaluated)], policy=response_on)
        with self.assertRaises(ValueError):
            rank([(decision, evaluated)], util={"c1": 1.04})
        with self.assertRaises(ValueError):
            rk.RankingContext(controller_utilization={}, utilization_limit=1.0)


if __name__ == "__main__":
    unittest.main()

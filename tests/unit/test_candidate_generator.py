from __future__ import annotations

import dataclasses
import unittest
from datetime import timedelta
from typing import Any

from decision_testdata import NOW, make_roles, make_twin
from src.schemas import decision as d
from src.schemas.twin import TwinState
from src.twin.decision.candidate_generator import generate_candidates, natural_key
from src.twin.decision.candidate_precheck import (
    PrecheckContext,
    PrecheckResult,
    precheck_all,
    precheck_candidate,
)


def _ctx(twin: TwinState, **over: Any) -> PrecheckContext:
    base = PrecheckContext(
        at_risk=frozenset({"c1"}),
        role_matrix=make_roles(twin),
        locked_switches=frozenset(),
        last_migration_finished_at=None,
        cooldown_seconds=37.0,
        now=NOW,
    )
    return dataclasses.replace(base, **over)


def _three_controllers() -> TwinState:
    return make_twin(
        utilization={"c1": 0.9, "c2": 0.4, "c3": 0.3},
        capacity={"c1": 933.0, "c2": 933.0, "c3": 933.0},
    )


def _one(twin: TwinState, **over: Any) -> PrecheckResult:
    cand = generate_candidates(twin, {"c1"}, NOW).candidates[0]
    return precheck_candidate(cand, twin, _ctx(twin, **over))


class GenerationTest(unittest.TestCase):
    def test_no_risk_means_no_candidate(self) -> None:
        self.assertEqual(generate_candidates(make_twin(), set(), NOW).candidates, ())

    def test_one_source_four_switches_one_destination_gives_four(self) -> None:
        result = generate_candidates(make_twin(), {"c1"}, NOW)
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual([c.switch_id for c in result.candidates], ["s1", "s2", "s3", "s4"])
        self.assertTrue(all(c.target_controller == "c2" for c in result.candidates))

    def test_source_never_equals_destination(self) -> None:
        twin = _three_controllers()
        for cand in generate_candidates(twin, {"c1", "c2"}, NOW).candidates:
            self.assertNotEqual(cand.source_controller, cand.target_controller)

    def test_ids_are_unique_and_output_is_deterministic(self) -> None:
        twin = _three_controllers()
        first = generate_candidates(twin, {"c2", "c1"}, NOW).candidates
        second = generate_candidates(twin, ["c1", "c2"], NOW).candidates
        self.assertEqual(first, second)
        ids = [c.candidate_id for c in first]
        self.assertEqual(len(ids), len(set(ids)))

    def test_candidate_id_follows_the_shared_schema(self) -> None:
        cand = generate_candidates(make_twin(), {"c1"}, NOW).candidates[0]
        self.assertEqual(cand.candidate_id, d.make_candidate_id("snap-1", "s1", "c1", "c2"))

    def test_candidate_carries_the_snapshot_values(self) -> None:
        cand = generate_candidates(make_twin(), {"c1"}, NOW).candidates[0]
        self.assertEqual(cand.source_utilization, 0.95)
        self.assertEqual(cand.target_utilization, 0.40)
        self.assertEqual(cand.switch_control_load_share, 0.1)
        self.assertEqual(cand.generated_at, NOW)

    def test_unknown_risk_controller_is_ignored(self) -> None:
        self.assertEqual(generate_candidates(make_twin(), {"c9"}, NOW).candidates, ())

    def test_natural_order_puts_s2_before_s10(self) -> None:
        names = ["s10", "s2", "s1"]
        self.assertEqual(sorted(names, key=natural_key), ["s1", "s2", "s10"])

    def test_switch_without_valid_share_is_dropped_and_reported(self) -> None:
        twin = make_twin()
        bad = dataclasses.replace(twin.switches[0], control_load_share=float("nan"))
        twin = dataclasses.replace(twin, switches=[bad, *twin.switches[1:]])
        result = generate_candidates(twin, {"c1"}, NOW)
        self.assertEqual(result.dropped, (("s1", "c1"),))
        self.assertEqual(len(result.candidates), 3)


class PrecheckTest(unittest.TestCase):
    def test_a_healthy_candidate_is_eligible(self) -> None:
        result = _one(make_twin())
        self.assertTrue(result.eligible)
        self.assertEqual(result.reasons, ())

    def test_source_not_at_risk(self) -> None:
        self.assertEqual(_one(make_twin(), at_risk=frozenset()).reasons,
                         (d.SOURCE_NOT_AT_RISK,))

    def test_owner_mismatch(self) -> None:
        twin = make_twin()
        cand = generate_candidates(twin, {"c1"}, NOW).candidates[0]
        moved = make_twin(owners={"s1": "c2", "s2": "c1", "s3": "c1", "s4": "c1", "s5": "c2"})
        result = precheck_candidate(cand, moved, _ctx(moved))
        self.assertIn(d.OWNER_MISMATCH, result.reasons)

    def test_destination_disconnected(self) -> None:
        twin = make_twin()
        roles = dict(make_roles(twin))
        roles[("s1", "c2")] = {"role": "SLAVE", "connected": False}
        self.assertEqual(_one(twin, role_matrix=roles).reasons, (d.DESTINATION_DISCONNECTED,))

    def test_switch_disconnected_from_source(self) -> None:
        twin = make_twin()
        roles = dict(make_roles(twin))
        roles[("s1", "c1")] = {"role": "MASTER", "connected": False}
        self.assertEqual(_one(twin, role_matrix=roles).reasons, (d.SWITCH_DISCONNECTED,))

    def test_role_state_invalid_when_source_is_not_master(self) -> None:
        twin = make_twin()
        roles = dict(make_roles(twin))
        roles[("s1", "c1")] = {"role": "SLAVE", "connected": True}
        self.assertEqual(_one(twin, role_matrix=roles).reasons, (d.ROLE_STATE_INVALID,))

    def test_missing_capacity(self) -> None:
        twin = make_twin(capacity={"c1": 933.0, "c2": 0.0})
        self.assertEqual(_one(twin).reasons, (d.MISSING_CAPACITY,))

    def test_active_transaction_conflict(self) -> None:
        result = _one(make_twin(), locked_switches=frozenset({"s1"}))
        self.assertEqual(result.reasons, (d.ACTIVE_TRANSACTION_CONFLICT,))

    def test_cooldown_active_then_elapsed(self) -> None:
        twin = make_twin()
        recent = NOW - timedelta(seconds=10)
        self.assertEqual(_one(twin, last_migration_finished_at=recent).reasons,
                         (d.COOLDOWN_ACTIVE,))
        old = NOW - timedelta(seconds=37)
        self.assertTrue(_one(twin, last_migration_finished_at=old).eligible)

    def test_cooldown_not_configured_is_skipped_not_passed(self) -> None:
        recent = NOW - timedelta(seconds=1)
        result = _one(make_twin(), cooldown_seconds=None, last_migration_finished_at=recent)
        self.assertTrue(result.eligible)
        self.assertEqual(result.skipped_checks, (("cooldown_clear", d.NOT_CONFIGURED),))

    def test_all_failing_reasons_are_collected(self) -> None:
        recent = NOW - timedelta(seconds=1)
        result = _one(make_twin(), at_risk=frozenset(), locked_switches=frozenset({"s1"}),
                      last_migration_finished_at=recent)
        self.assertEqual(
            result.reasons,
            (d.SOURCE_NOT_AT_RISK, d.ACTIVE_TRANSACTION_CONFLICT, d.COOLDOWN_ACTIVE),
        )

    def test_precheck_all_keeps_candidate_order(self) -> None:
        twin = make_twin()
        cands = generate_candidates(twin, {"c1"}, NOW).candidates
        results = precheck_all(cands, twin, _ctx(twin))
        self.assertEqual([r.candidate_id for r in results], [c.candidate_id for c in cands])

    def test_result_rejects_unknown_reason_codes(self) -> None:
        with self.assertRaises(ValueError):
            PrecheckResult("x", False, ("NOT_A_CODE",))

    def test_to_dict_is_plain_data(self) -> None:
        self.assertEqual(_one(make_twin()).to_dict()["reasons"], [])


if __name__ == "__main__":
    unittest.main()

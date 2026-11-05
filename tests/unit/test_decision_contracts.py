from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from pathlib import Path

from decision_testdata import NOW
from scripts.decision.build_policy_manifest import (
    OUT_DIR,
    build_manifest,
    build_reason_codes,
)
from src.schemas.decision import (
    STRUCTURAL_REASONS,
    EvaluatedCandidate,
    MigrationCandidate,
    ReasonCode,
    SafetyDecision,
)
from src.twin.decision.safety_policy import DEFAULT_POLICY_PATH, load_safety_policy


def _candidate() -> MigrationCandidate:
    return MigrationCandidate("c-1", "snap-1", "s1", "c1", "c2", 0.95, 0.4, 0.1, NOW)


class DecisionContractsTest(unittest.TestCase):
    def test_there_are_fifteen_reason_codes_named_by_their_value(self) -> None:
        self.assertEqual(len(ReasonCode), 15)
        self.assertTrue(all(r.value == r.name for r in ReasonCode))

    def test_structural_reasons_exclude_outcome_based_ones(self) -> None:
        self.assertEqual(len(STRUCTURAL_REASONS), 8)
        outcome_based = {
            ReasonCode.OUTCOME_PREDICTOR_NOT_READY,
            ReasonCode.DST_UTILIZATION_UNSAFE,
            ReasonCode.DST_RESPONSE_UNSAFE,
            ReasonCode.NO_SOURCE_RELIEF,
            ReasonCode.INSUFFICIENT_IMPROVEMENT,
        }
        self.assertFalse(STRUCTURAL_REASONS & outcome_based)

    def test_contracts_are_frozen(self) -> None:
        candidate = _candidate()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(candidate, "switch_id", "s2")

    def test_to_dict_is_json_serialisable(self) -> None:
        candidate = _candidate()
        evaluated = EvaluatedCandidate(candidate, 0.6, 0.7, 1.0, 1.1, 0.1, 0.037, 0.009, "m", 2.5)
        decision = SafetyDecision(
            "c-1", True, ("a",), (), 0.7, 0.037, 0.737, 1.1, 0.009, None, 0.3, 0.9, 1.0,
            "safety-policy-v1", "m", 1.0,
        )
        self.assertTrue(json.dumps(candidate.to_dict()))
        self.assertTrue(json.dumps(evaluated.to_dict()))
        self.assertTrue(json.dumps(decision.to_dict()))
        self.assertEqual(decision.to_dict()["passed_checks"], ["a"])

    def test_manifest_agrees_with_policy_and_artifacts(self) -> None:
        manifest = build_manifest()
        policy = load_safety_policy()
        self.assertEqual(manifest["not_configured"], list(policy.not_configured()))
        thresholds = manifest["thresholds"]
        self.assertEqual(thresholds["utilization_safe_limit"]["value"], policy.utilization_safe_limit)
        self.assertEqual((thresholds["m_U"]["n"], thresholds["m_U"]["rank"]), (60, 58))
        for name in policy.not_configured():
            self.assertEqual(thresholds[name]["status"], "NOT_CONFIGURED")
            self.assertTrue(thresholds[name]["reason"])

    def test_committed_manifest_is_in_sync_with_policy_file(self) -> None:
        manifest = json.loads(Path(OUT_DIR, "policy_manifest.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256(DEFAULT_POLICY_PATH.read_bytes()).hexdigest()
        self.assertEqual(manifest["policy_sha256"], expected)
        codes = json.loads(Path(OUT_DIR, "reason_codes.json").read_text(encoding="utf-8"))
        self.assertEqual(codes, build_reason_codes())


if __name__ == "__main__":
    unittest.main()

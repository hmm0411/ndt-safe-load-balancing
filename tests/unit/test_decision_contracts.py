from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

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
from tests.unit.decision_testdata import NOW


def _candidate() -> MigrationCandidate:
    return MigrationCandidate("c-1", "snap-1", "s1", "c1", "c2", 0.95, 0.4, 0.1, NOW)


def test_there_are_fifteen_reason_codes_named_by_their_value():
    assert len(ReasonCode) == 15
    assert all(r.value == r.name for r in ReasonCode)


def test_structural_reasons_exclude_outcome_based_ones():
    assert len(STRUCTURAL_REASONS) == 8
    outcome_based = {
        ReasonCode.OUTCOME_PREDICTOR_NOT_READY,
        ReasonCode.DST_UTILIZATION_UNSAFE,
        ReasonCode.DST_RESPONSE_UNSAFE,
        ReasonCode.NO_SOURCE_RELIEF,
        ReasonCode.INSUFFICIENT_IMPROVEMENT,
    }
    assert not (STRUCTURAL_REASONS & outcome_based)


def test_contracts_are_frozen():
    candidate = _candidate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.switch_id = "s2"  # type: ignore[misc]


def test_to_dict_is_json_serialisable():
    candidate = _candidate()
    evaluated = EvaluatedCandidate(candidate, 0.6, 0.7, 1.0, 1.1, 0.1, 0.037, 0.009, "m", 2.5)
    decision = SafetyDecision(
        "c-1", True, ("a",), (), 0.7, 0.037, 0.737, 1.1, 0.009, None, 0.3, 0.9, 1.0,
        "safety-policy-v1", "m", 1.0,
    )
    assert json.dumps(candidate.to_dict())
    assert json.dumps(evaluated.to_dict())
    assert json.dumps(decision.to_dict())
    assert decision.to_dict()["passed_checks"] == ["a"]


def test_manifest_agrees_with_policy_and_artifacts():
    manifest = build_manifest()
    policy = load_safety_policy()
    assert manifest["not_configured"] == list(policy.not_configured())
    thresholds = manifest["thresholds"]
    assert thresholds["utilization_safe_limit"]["value"] == policy.utilization_safe_limit
    assert thresholds["m_U"]["n"] == 60 and thresholds["m_U"]["rank"] == 58
    for name in policy.not_configured():
        assert thresholds[name]["status"] == "NOT_CONFIGURED"
        assert thresholds[name]["reason"]


def test_committed_manifest_is_in_sync_with_policy_file():
    manifest = json.loads(Path(OUT_DIR, "policy_manifest.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256(DEFAULT_POLICY_PATH.read_bytes()).hexdigest()
    assert manifest["policy_sha256"] == expected
    codes = json.loads(Path(OUT_DIR, "reason_codes.json").read_text(encoding="utf-8"))
    assert codes == build_reason_codes()

from __future__ import annotations

import copy

import pytest
import yaml

from src.twin.decision.safety_policy import (
    DEFAULT_POLICY_PATH,
    PolicyError,
    load_safety_policy,
    parse_policy,
)


def _raw() -> dict:
    with open(DEFAULT_POLICY_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_default_policy_loads():
    policy = load_safety_policy()
    assert policy.version == "safety-policy-v1"
    assert policy.utilization_safe_limit == 1.0
    assert policy.utilization_margin_source == "post-migration-v1"


def test_unsourced_thresholds_are_null_not_invented():
    policy = load_safety_policy()
    assert set(policy.not_configured()) == {
        "minimum_quality_score",
        "response_p95_max_ms",
        "cooldown_seconds",
        "minimum_source_relief",
    }


def test_missing_section_is_rejected():
    raw = _raw()
    del raw["migration"]
    with pytest.raises(PolicyError):
        parse_policy(raw)


def test_unknown_key_is_rejected():
    raw = copy.deepcopy(_raw())
    raw["destination"]["surprise"] = 1
    with pytest.raises(PolicyError):
        parse_policy(raw)


def test_null_safe_limit_is_rejected():
    raw = _raw()
    raw["destination"]["utilization_safe_limit"] = None
    with pytest.raises(PolicyError):
        parse_policy(raw)


def test_boolean_is_not_accepted_as_number():
    raw = _raw()
    raw["migration"]["cooldown_seconds"] = True
    with pytest.raises(PolicyError):
        parse_policy(raw)

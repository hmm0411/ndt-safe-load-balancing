from __future__ import annotations

import copy
import unittest
from typing import Any

import yaml

from src.twin.decision.safety_policy import (
    DEFAULT_POLICY_PATH,
    PolicyError,
    load_safety_policy,
    parse_policy,
)


def _raw() -> dict[str, Any]:
    with open(DEFAULT_POLICY_PATH, encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


class SafetyPolicyTest(unittest.TestCase):
    def test_default_policy_loads(self) -> None:
        policy = load_safety_policy()
        self.assertEqual(policy.version, "safety-policy-v1")
        self.assertEqual(policy.utilization_safe_limit, 1.0)
        self.assertEqual(policy.utilization_margin_source, "post-migration-v1")

    def test_unsourced_thresholds_are_null_not_invented(self) -> None:
        self.assertEqual(
            set(load_safety_policy().not_configured()),
            {
                "minimum_quality_score",
                "response_p95_max_ms",
                "cooldown_seconds",
                "minimum_source_relief",
            },
        )

    def test_missing_section_is_rejected(self) -> None:
        raw = _raw()
        del raw["migration"]
        with self.assertRaises(PolicyError):
            parse_policy(raw)

    def test_unknown_key_is_rejected(self) -> None:
        raw = copy.deepcopy(_raw())
        raw["destination"]["surprise"] = 1
        with self.assertRaises(PolicyError):
            parse_policy(raw)

    def test_null_safe_limit_is_rejected(self) -> None:
        raw = _raw()
        raw["destination"]["utilization_safe_limit"] = None
        with self.assertRaises(PolicyError):
            parse_policy(raw)

    def test_boolean_is_not_accepted_as_number(self) -> None:
        raw = _raw()
        raw["migration"]["cooldown_seconds"] = True
        with self.assertRaises(PolicyError):
            parse_policy(raw)


if __name__ == "__main__":
    unittest.main()

"""Guards the frozen safety-policy-v1: a change means safety-policy-v2, never an edit of v1."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.twin.decision.safety_policy import load_safety_policy

ROOT = Path(__file__).resolve().parents[2]
MARGINS = ROOT / "data/models/outcome/post-migration-v1/uncertainty_margin.json"


class FrozenPolicyV1Test(unittest.TestCase):
    def test_version_and_limits(self) -> None:
        policy = load_safety_policy()
        self.assertEqual(policy.version, "safety-policy-v1")
        self.assertEqual(policy.utilization_safe_limit.value, 1.0)
        self.assertIsNone(policy.response_p95_max_ms.value)

    def test_minimum_source_relief_is_the_artifact_noise_floor(self) -> None:
        m_u = json.loads(MARGINS.read_text(encoding="utf-8"))["m_U"]
        value = load_safety_policy().minimum_source_relief.value
        assert value is not None
        self.assertAlmostEqual(value, m_u, places=8)


if __name__ == "__main__":
    unittest.main()

"""The 4C safety policy differs from v1 only where it says (version, model, margins)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from src.twin.decision.ranking_policy import load_ranking_policy
from src.twin.decision.safety_policy import (
    POLICY_ENV,
    REPO_ROOT,
    load_margins,
    load_safety_policy,
)

V1 = REPO_ROOT / "configs/decision/safety_policy_v1.yaml"
C4 = REPO_ROOT / "configs/decision/safety_policy_4c_v1.yaml"


def flat(d: object, prefix: str = "") -> dict[str, object]:
    if isinstance(d, dict):
        out: dict[str, object] = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


class Policy4CTest(unittest.TestCase):
    def test_only_version_model_and_sources_change(self) -> None:
        a = flat(yaml.safe_load(V1.read_text(encoding="utf-8")))
        b = flat(yaml.safe_load(C4.read_text(encoding="utf-8")))
        changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        self.assertEqual(changed, {"version", "uncertainty.outcome_model_version",
                                   "uncertainty.margins_file",
                                   "destination.utilization_safe_limit.source",
                                   "improvement.minimum_source_relief.source"})
        self.assertEqual(b["improvement.minimum_source_relief.value"], 0.036629233)

    def test_margins_are_those_of_v2(self) -> None:
        policy = load_safety_policy(C4)
        self.assertEqual(policy.outcome_model_version, "post-migration-v2")
        m_u, m_r = load_margins(policy)
        self.assertAlmostEqual(m_u, 0.13860323509560712)
        self.assertAlmostEqual(m_r, 0.04826759320833318)

    def test_env_selects_the_policy_and_default_stays_v1(self) -> None:
        with mock.patch.dict(os.environ, {POLICY_ENV: str(C4.relative_to(REPO_ROOT))}):
            self.assertEqual(load_safety_policy().version, "safety-policy-4c-v1")
        env = {k: v for k, v in os.environ.items() if k != POLICY_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_safety_policy().version, "safety-policy-v1")
        self.assertTrue(Path(C4).is_file())

    def test_ranking_4c_differs_from_v1_only_in_versions(self) -> None:
        r1 = REPO_ROOT / "configs/decision/ranking_policy_v1.yaml"
        r4 = REPO_ROOT / "configs/decision/ranking_policy_4c_v1.yaml"
        a = flat(yaml.safe_load(r1.read_text(encoding="utf-8")))
        b = flat(yaml.safe_load(r4.read_text(encoding="utf-8")))
        self.assertEqual({k for k in a.keys() | b.keys() if a.get(k) != b.get(k)},
                         {"version", "safety_policy_version"})
        self.assertEqual(load_ranking_policy(r4).safety_policy_version,
                         load_safety_policy(C4).version)

    def test_a_4c_decision_runs_end_to_end(self) -> None:
        """The pilots of 26/11 failed on version pins (gate, then ranking): run the whole
        pipeline with both 4C policies and a v2 prediction."""
        import dataclasses

        from test_decision_plan import make
        from test_decision_service import FakePredictor

        class V2(FakePredictor):
            def predict_batch(self, *args: Any, **kwargs: Any) -> list[Any]:
                return [dataclasses.replace(p, model_version="post-migration-v2")
                        for p in super().predict_batch(*args, **kwargs)]

        r4 = REPO_ROOT / "configs/decision/ranking_policy_4c_v1.yaml"
        service, _ = make(predictor=V2(), policy=load_safety_policy(C4),
                          ranking_policy=load_ranking_policy(r4))
        out = service.plan("burst")
        self.assertEqual(out["status"], "SELECTED")
        self.assertGreater(out["funnel"]["safe"], 0)


if __name__ == "__main__":
    unittest.main()

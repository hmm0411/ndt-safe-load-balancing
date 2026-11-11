from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

import yaml

from src.twin.decision import ranking_policy as rp


def policy_data() -> dict[str, Any]:
    data = yaml.safe_load(rp.POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class RepoRankingPolicyTest(unittest.TestCase):
    def test_policy_loads_with_disabled_components_explained(self) -> None:
        p = rp.load_ranking_policy()
        self.assertEqual(p.version, "ranking-policy-v1")
        self.assertEqual(p.safety_policy_version, "safety-policy-v1")
        self.assertEqual(
            p.enabled,
            ("source_relief", "destination_headroom", "imbalance", "global_improvement"),
        )
        for name in ("response", "migration_cost"):
            self.assertFalse(p.component(name).enabled)
            self.assertTrue(p.component(name).reason)
        self.assertEqual(p.require_positive, ("global_improvement",))

    def test_effective_weights_are_renormalized(self) -> None:
        w = rp.load_ranking_policy().effective_weights()
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)
        self.assertAlmostEqual(w["source_relief"], 0.25 / 0.80, places=12)
        self.assertAlmostEqual(w["destination_headroom"], 0.20 / 0.80, places=12)
        self.assertAlmostEqual(w["global_improvement"], 0.15 / 0.80, places=12)
        self.assertNotIn("response", w)

    def test_sensitivity_sets_keep_enabled_flags(self) -> None:
        p = rp.load_ranking_policy()
        names = [name for name, _ in p.sensitivity]
        self.assertEqual(names, ["base", "relief_heavy", "balance_heavy", "qos_heavy"])
        heavy = p.with_weights("relief_heavy", dict(p.sensitivity)["relief_heavy"])
        self.assertEqual(heavy.enabled, p.enabled)
        self.assertEqual(heavy.version, "ranking-policy-v1+relief_heavy")
        self.assertAlmostEqual(heavy.effective_weights()["source_relief"], 0.30 / 0.80)


class ParseRankingPolicyTest(unittest.TestCase):
    def broken(self, edit: Callable[[dict[str, Any]], object]) -> None:
        data = policy_data()
        edit(data)
        with self.assertRaises(ValueError):
            rp.parse_ranking_policy(data)

    def test_rejects_untraceable_or_inconsistent_policies(self) -> None:
        comps = "components"
        edits: tuple[Callable[[dict[str, Any]], object], ...] = (
            lambda d: d.pop("version"),
            lambda d: d.update(normalization="zscore_on_test"),
            lambda d: d.update(tie_break=["candidate_id_asc"]),
            lambda d: d[comps].pop("migration_cost"),
            lambda d: d[comps]["response"].pop("reason"),
            lambda d: d[comps]["source_relief"].update(weight=0.30),
            lambda d: d[comps]["source_relief"].update(weight=-0.25),
            lambda d: d[comps]["imbalance"].update(enabled="yes"),
            lambda d: d.update(no_migration={"require_positive": ["response"]}),
            lambda d: d["sensitivity"].update(base=[0.30, 0.20, 0.15, 0.15, 0.15, 0.05]),
            lambda d: d["sensitivity"].update(bad=[0.5, 0.5]),
            lambda d: d.pop("sensitivity"),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                self.broken(edit)

    def test_enabled_component_needs_weight(self) -> None:
        data = policy_data()
        data["components"]["imbalance"]["weight"] = 0.0
        data["components"]["source_relief"]["weight"] = 0.45
        data["sensitivity"] = {"base": [0.45, 0.20, 0.0, 0.15, 0.15, 0.05]}
        with self.assertRaises(ValueError):
            rp.parse_ranking_policy(data)


if __name__ == "__main__":
    unittest.main()

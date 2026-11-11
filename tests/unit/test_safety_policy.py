import csv
import json
import math
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import yaml

from src.twin.decision import safety_policy as sp

REPO = sp.REPO_ROOT


def policy_data() -> dict:
    return yaml.safe_load(sp.POLICY_PATH.read_text(encoding="utf-8"))


class RepoPolicyTest(unittest.TestCase):
    def test_policy_loads_with_provenance(self) -> None:
        p = sp.load_safety_policy()
        self.assertEqual(p.version, "safety-policy-v1")
        self.assertEqual(p.utilization_safe_limit.value, 1.0)
        self.assertTrue(p.utilization_safe_limit.source)
        for limit in (p.response_p95_max_ms, p.minimum_quality_score):
            self.assertFalse(limit.configured)
            self.assertTrue(limit.reason)
        self.assertTrue(p.minimum_source_relief.configured)
        self.assertTrue(p.minimum_source_relief.source)

    def test_margins_come_from_the_artifact(self) -> None:
        p = sp.load_safety_policy()
        m_u, m_r = sp.load_margins(p)
        stored = json.loads((REPO / p.margins_file).read_text(encoding="utf-8"))
        self.assertEqual((m_u, m_r), (stored["m_U"], stored["m_R"]))

    def test_cooldown_matches_raw_v2(self) -> None:
        path = REPO / "evidence" / "week12" / "attempt_summary.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r["action_type"] == "MIGRATE" and r["transaction_status"] == "COMMITTED"]
        bound = sp.cooldown_lower_bound([float(r["transaction_s"]) for r in rows],
                                        [float(r["settle_s"]) for r in rows])
        self.assertEqual(len(rows), 300)
        self.assertEqual(sp.load_safety_policy().cooldown_seconds.value, math.ceil(bound))


class ParsePolicyTest(unittest.TestCase):
    def broken(self, edit: Callable[[dict], object]) -> None:
        data = policy_data()
        edit(data)
        with self.assertRaises(ValueError):
            sp.parse_safety_policy(data)

    def test_rejects_unsafe_or_untraceable_policies(self) -> None:
        edits = (
            lambda d: d["destination"]["response_p95_max_ms"].pop("reason"),
            lambda d: d["migration"]["cooldown_seconds"].pop("source"),
            lambda d: d["migration"]["cooldown_seconds"].update(value=True),
            lambda d: d["migration"]["cooldown_seconds"].update(value=-1),
            lambda d: d["state"].update(require_valid_snapshot=False),
            lambda d: d["state"]["minimum_quality_score"].update(value=1.5, source="x"),
            lambda d: d["migration"].update(max_concurrent_transactions=2),
            lambda d: d["migration"].update(cooldown_scope="per_switch"),
            lambda d: d.update(boundary="exclusive"),
            lambda d: d["destination"]["utilization_safe_limit"].update(value=None, reason="x"),
            lambda d: d.pop("improvement"),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                self.broken(edit)

    def test_margin_file_of_another_model_is_refused(self) -> None:
        p = sp.load_safety_policy()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / p.margins_file
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"model_version": "other", "m_U": 0.1, "m_R": 0.1}))
            with self.assertRaises(ValueError):
                sp.load_margins(p, root=Path(tmp))


class BoundaryTest(unittest.TestCase):
    def test_inclusive_boundary(self) -> None:
        self.assertTrue(sp.within_limit(1.0, 0.0, 1.0))
        self.assertTrue(sp.within_limit(0.963370766760003, 0.036629233239997, 1.0))
        self.assertFalse(sp.within_limit(0.97, 0.037, 1.0))
        self.assertFalse(sp.within_limit(1.0, 1e-6, 1.0))
        self.assertTrue(sp.within_limit(1.12 - 0.2, 0.037, 1.0))

    def test_invalid_boundary_inputs(self) -> None:
        for args in ((0.9, -0.01, 1.0), (float("nan"), 0.0, 1.0), (0.9, 0.0, float("inf"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sp.within_limit(*args)

    def test_cooldown_bound(self) -> None:
        tx = [1.0] * 19 + [3.0]
        settle = [4.0] * 19 + [30.0]
        self.assertEqual(sp.cooldown_lower_bound(tx, settle), 33.0)
        with self.assertRaises(ValueError):
            sp.cooldown_lower_bound([1.0], [])


if __name__ == "__main__":
    unittest.main()

"""One-sided margins, coverage and the gate confusion (Day 5)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.twin.outcome import artifact, margins, predictor

REPO_CONFIG = Path("configs/outcome/artifact.yaml")


class MarginTest(unittest.TestCase):
    def test_under_errors_are_one_sided(self) -> None:
        self.assertEqual(margins.under_errors([1.0, 0.5, 0.8], [0.9, 0.7, 0.8]),
                         [1.0 - 0.9, 0.0, 0.0])
        with self.assertRaises(ValueError):
            margins.under_errors([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            margins.under_errors([float("nan")], [1.0])

    def test_margin_uses_the_split_conformal_rank(self) -> None:
        actual = [i / 100 for i in range(1, 61)]  # under-errors 0.01 .. 0.60
        out = margins.one_sided_margin(actual, [0.0] * 60, 0.95)
        self.assertEqual(out["rank"], 58)  # ceil(61 * 0.95)
        self.assertAlmostEqual(out["value"], 0.58)
        self.assertEqual(out["n"], 60)
        self.assertEqual(out["under_predicted"], 60)
        small = margins.one_sided_margin([1.0, 2.0, 3.0], [0.0] * 3, 0.95)
        self.assertEqual(small["rank"], 3)  # clipped to n
        self.assertEqual(small["value"], 3.0)
        over = margins.one_sided_margin([0.5] * 10, [0.9] * 10, 0.95)
        self.assertEqual(over["value"], 0.0)

    def test_coverage_and_groups(self) -> None:
        cov = margins.coverage([1.0, 1.2, 0.5], [0.9, 0.9, 0.9], 0.15)
        self.assertEqual((cov["covered"], cov["n"]), (2, 3))
        rows = [{"u": 1.0, "w": "a"}, {"u": 1.2, "w": "b"}, {"u": 0.5, "w": "a"}]
        table = margins.coverage_by_group(rows, "u", [0.9, 0.9, 0.9], 0.15,
                                          {"workload": lambda r: r["w"]})
        self.assertEqual([(r["group"], r["value"], r["covered"], r["n"]) for r in table],
                         [("all", "all", 2, 3), ("workload", "a", 2, 2),
                          ("workload", "b", 0, 1)])

    def test_gate_confusion(self) -> None:
        g = margins.gate_confusion([1.1, 0.9, 1.05, 0.5], [0.95, 0.97, 0.80, 0.5], 0.05)
        self.assertEqual((g["true_positive"], g["false_alarm"], g["missed_overload"],
                          g["true_negative"]), (1, 1, 1, 1))
        self.assertEqual(g["overload"], 2)
        self.assertAlmostEqual(g["miss_rate"], 0.5)
        none = margins.gate_confusion([0.5], [0.5], 0.0)
        self.assertIsNone(none["miss_rate"])


class ArtifactConfigTest(unittest.TestCase):
    def test_repo_config_matches_the_predictor(self) -> None:
        config = artifact.load_artifact_config(REPO_CONFIG)
        self.assertEqual(config.model_version, predictor.MODEL_VERSION)
        self.assertEqual(config.margins, {"m_U": "u_dst_post", "m_R": "r95_post"})
        self.assertEqual(config.margin_level, 0.95)

    def test_config_is_checked(self) -> None:
        data = yaml.safe_load(REPO_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.yaml"
            for bad in ({**data, "deploy_rule": "lowest_test_S"},
                        {**data, "margin_level": 0.3},
                        {**data, "margins": {"m_U": "u_dst_post"}},
                        {**data, "margins": {"m_U": "latency", "m_R": "r95_post"}}):
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):
                    artifact.load_artifact_config(path)


if __name__ == "__main__":
    unittest.main()

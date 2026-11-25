from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from scripts.analysis import campaign_stats
from src.experiments.final_evaluation import stats

# Hollander & Wolfe depression data (the scipy.stats.wilcoxon example): exact p = 0.0390625
HW = [1.83 - 0.878, 0.50 - 0.647, 1.62 - 0.598, 2.48 - 2.05, 1.68 - 1.06, 1.88 - 1.29,
      1.55 - 1.06, 3.06 - 3.14, 1.30 - 1.29]
TIED = [0.1, 0.1, -0.2, 0.3, 0.3, 0.3, 0.0, 0.4, -0.1, 0.5, 0.2, 0.2]


class WilcoxonTest(unittest.TestCase):
    def test_exact_matches_the_reference(self) -> None:
        r = stats.wilcoxon_signed_rank(HW)
        self.assertEqual(r.method, "exact")
        self.assertAlmostEqual(r.p_value or -1, 0.0390625, places=12)
        self.assertEqual((r.t_plus, r.t_minus), (40.0, 5.0))
        self.assertAlmostEqual(r.rank_biserial or -1, 35 / 45)

    def test_ties_and_zeros_use_the_corrected_normal_approximation(self) -> None:
        wil = stats.wilcoxon_signed_rank(TIED, "wilcox")
        pratt = stats.wilcoxon_signed_rank(TIED, "pratt")
        self.assertEqual((wil.method, wil.n_used, wil.n_zero), ("normal", 11, 1))
        self.assertAlmostEqual(wil.p_value or -1, 0.0225609156578089, places=12)
        self.assertAlmostEqual(pratt.p_value or -1, 0.02217850984439585, places=12)

    def test_all_zero_has_no_p_value(self) -> None:
        r = stats.wilcoxon_signed_rank([0.0] * 10)
        self.assertIsNone(r.p_value)
        self.assertIsNone(r.rank_biserial)

    def test_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            stats.wilcoxon_signed_rank([1.0, float("nan")])
        with self.assertRaises(ValueError):
            stats.wilcoxon_signed_rank([1.0], "zsplit")

    def test_agrees_with_scipy_when_available(self) -> None:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            self.skipTest("scipy not installed")
        rng = random.Random(7)
        for trial in range(150):
            n = rng.randint(5, 60)
            d = [round(rng.gauss(0.3, 1.0), 1 if trial % 2 else 6) for _ in range(n)]
            zm = "pratt" if trial % 3 == 0 else "wilcox"
            if all(x == 0 for x in d):
                continue
            ours = stats.wilcoxon_signed_rank(d, zm)
            method = "exact" if ours.method == "exact" else "approx"
            ref = wilcoxon(d, zero_method=zm, correction=True, method=method).pvalue
            self.assertAlmostEqual(ours.p_value or -1, float(ref), places=10)


class HolmTest(unittest.TestCase):
    def test_step_down_and_monotone(self) -> None:
        out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": None})
        self.assertAlmostEqual(out["a"] or -1, 0.04)
        self.assertAlmostEqual(out["c"] or -1, 0.09)
        self.assertAlmostEqual(out["b"] or -1, 0.09)   # 2 x 0.04 = 0.08, raised to 0.09
        self.assertAlmostEqual(out["d"] or -1, 0.5)
        self.assertIsNone(out["e"])


def episodes(full: Mapping[tuple[str, int], float | None],
             base: Mapping[tuple[str, int], float | None]) -> list[stats.Episode]:
    return ([stats.Episode("Full", w, s, v) for (w, s), v in full.items()]
            + [stats.Episode("Static", w, s, v) for (w, s), v in base.items()])


class CompareTest(unittest.TestCase):
    def test_lower_is_better_sign_and_per_workload(self) -> None:
        full = {("stable", s): 1.0 for s in range(1, 11)}
        full.update({("burst", s): 2.0 for s in range(1, 11)})
        base = {("stable", s): 1.0 + s for s in range(1, 11)}      # Full better on every seed
        base.update({("burst", s): 2.0 for s in range(1, 11)})     # ties
        out = stats.compare(episodes(full, base), "Full", ["Static"], lower_is_better=True)
        c = out["comparisons"]["Static"]
        self.assertEqual((c["test"]["n_pairs"], c["test"]["n_zero"]), (20, 10))
        self.assertEqual(c["test"]["rank_biserial"], 1.0)
        self.assertEqual(c["per_workload"]["stable"]["seeds_won"], 10)
        self.assertEqual(c["per_workload"]["burst"]["seeds_tied"], 10)
        self.assertEqual(c["per_workload"]["stable"]["Static"]["median"], 6.5)
        self.assertTrue(c["significant_holm"])

    def test_missing_values_drop_the_pair(self) -> None:
        full: dict[tuple[str, int], float | None] = {("stable", 1): 1.0, ("stable", 2): None}
        base: dict[tuple[str, int], float | None] = {("stable", 1): 2.0, ("stable", 2): 3.0}
        out = stats.compare(episodes(full, base), "Full", ["Static"], lower_is_better=True)
        self.assertEqual(out["comparisons"]["Static"]["dropped_pairs"], ["stable/2"])

    def test_duplicates_and_unknown_methods_are_errors(self) -> None:
        dup = [stats.Episode("Full", "stable", 1, 1.0), stats.Episode("Full", "stable", 1, 2.0)]
        with self.assertRaises(ValueError):
            stats.compare(dup, "Full", ["Static"], lower_is_better=True)
        with self.assertRaises(ValueError):
            stats.compare(dup[:1], "Full", ["Static"], lower_is_better=True)

    def test_cli_reads_empty_cells_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "ep.csv", Path(tmp) / "out.json"
            lines = ["method,workload,seed,m2"]
            for s in range(9801, 9811):
                lines.append(f"Full,gradual,{s},{s - 9800}")
                lines.append(f"Static,gradual,{s},{'' if s == 9810 else s - 9790}")
            src.write_text("\n".join(lines) + "\n", encoding="utf-8")
            code = campaign_stats.main(["--episodes", str(src), "--metric", "m2",
                                        "--lower-is-better", "--baselines", "Static",
                                        "--out", str(out)])
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        static = data["comparisons"]["Static"]
        self.assertEqual(static["dropped_pairs"], ["gradual/9810"])
        self.assertEqual(static["per_workload"]["gradual"]["seeds_won"], 9)
        self.assertEqual(data["metric"], "m2")


if __name__ == "__main__":
    unittest.main()

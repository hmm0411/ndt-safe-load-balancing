"""Dataset report (Week 11 Day 4-6): manifest, split, quality counters, tables, figures."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.experiments.migration import fixtures as fx
from src.experiments.migration.config import load_outcome_config
from src.experiments.migration.dataset_report import (
    build_dataset_report,
    describe,
    fit_line,
    to_csv,
    write_report,
)
from src.experiments.migration.pilot import validate_pilot_settings
from src.experiments.migration.writer import write_attempt

CONFIG_PATH = Path("configs/experiments/migration_outcome.yaml")
CONFIG = load_outcome_config(CONFIG_PATH)
SETTINGS = validate_pilot_settings({
    "pilot_root": "x", "min_attempts": 1, "collection_budget_hours": 48,
    "budget_utilisation": 0.8, "cleanup_overhead_s": 15, "destination_level_tolerance": 0.10,
    "hot_switch_share_tolerance": 0.10, "workload_drop_ratio": 0.5,
    "delta_ratio_range": [0.3, 3.0], "timeout_headroom": 1.5,
})


def attempt(kind: str, repeat: str, n: int) -> dict[str, Any]:
    prefix = "noop" if kind == "no_op" else "mig"
    attempt_id = f"mig-stable-middle-dst45-c1-c2-{repeat}"
    out = fx.attempt_inputs(kind, attempt_id=attempt_id)
    started = fx.T0 + timedelta(minutes=3 * n)
    out["metadata"] = dataclasses.replace(
        out["metadata"], attempt_id=f"{prefix}-stable-middle-dst45-c1-c2-{repeat}", repeat=repeat,
        seed=CONFIG.repeats[repeat], started_at=started.isoformat(),
        finished_at=(started + timedelta(seconds=129)).isoformat())
    body = out["transaction_raw"].get("body")
    if isinstance(body, dict):
        tx = body.get("transaction") or body.get("detail", {}).get("transaction")
        if isinstance(tx, dict):
            tx["transaction_id"] = f"tx-{n:012d}"
    return out


class DatasetReportTest(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    report: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "pilot"
        plan = [("committed", "r01"), ("restored", "r02"), ("skipped", "r03"),
                ("committed", "r04"), ("committed", "r05"), ("no_op", "r01")]
        for n, (kind, repeat) in enumerate(plan, 1):
            write_attempt(cls.root, **attempt(kind, repeat, n))
        cls.report = build_dataset_report(cls.root, CONFIG, SETTINGS, CONFIG_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        fx.unlock_tree(Path(cls.tmp.name))
        cls.tmp.cleanup()

    def test_quality_report_counters(self) -> None:
        q = self.report["quality_report"]
        self.assertEqual(q["total_attempts"], 6)
        self.assertEqual(q["planned_attempts"], 340)
        self.assertEqual(q["invalid_count"], 0)
        self.assertEqual((q["committed_count"], q["restored_count"], q["failed_count"],
                          q["skipped_count"], q["no_op_count"]), (3, 1, 0, 1, 1))
        self.assertEqual(q["training_eligible_committed_count"], 3)
        self.assertEqual(q["rollback_count"], 1)
        self.assertEqual(q["verification_failure_count"], 1)
        self.assertEqual(q["failure_injected_count"], 1)
        self.assertEqual(q["valid_pre_count"], 6)
        self.assertEqual(q["valid_post_count"], 5)  # SKIPPED has no post window
        self.assertEqual(q["duplicate_transaction_count"], 0)
        self.assertEqual(q["ownership_mismatch_count"], 0)
        self.assertTrue(q["freeze_consistent"])
        self.assertEqual(q["progress_by_repeat"]["r01"]["present"], 2)
        self.assertEqual(q["progress_by_repeat"]["r01"]["planned"], 68)
        for key in ("committed_count", "restored_count", "failed_count", "skipped_count",
                    "valid_pre_count", "valid_post_count", "training_eligible_committed_count",
                    "workload_counts", "phase_counts", "direction_counts",
                    "destination_load_counts", "rollback_count", "verification_failure_count",
                    "missing_field_count", "invalid_quality_count",
                    "duplicate_transaction_count", "ownership_mismatch_count"):
            self.assertIn(key, q)  # every section-36 field

    def test_split_by_repeat_without_leakage(self) -> None:
        split = self.report["split_manifest"]
        self.assertEqual(split["splits"]["train"]["count"], 4)  # r01 x2, r02, r03
        self.assertEqual(split["splits"]["validation"]["attempts"],
                         ["mig-stable-middle-dst45-c1-c2-r04"])
        self.assertEqual(split["splits"]["test"]["count"], 1)
        self.assertEqual(split["splits"]["train"]["no_op"], 1)
        self.assertEqual(split["attempts_in_more_than_one_split"], [])
        self.assertEqual(split["unassigned"], [])

    def test_manifest_has_fingerprints(self) -> None:
        entries = self.report["manifest"]["attempts"]
        self.assertEqual(len(entries), 6)
        self.assertTrue(all(len(e["fingerprint"]) == 64 for e in entries))
        self.assertEqual(len(self.report["manifest"]["config_sha256"]), 64)

    def test_analytic_check_matches_the_fixture(self) -> None:
        # fixture: s1 carries 1/10 of c1 at U = 0.80 -> expected -0.08, observed -0.08
        ratio = self.report["analytic_check"]["all_committed"]["observed_over_expected"]
        self.assertEqual(ratio["n"], 3)
        self.assertAlmostEqual(ratio["median"], 1.0, places=6)

    def test_tables_and_noop_comparison(self) -> None:
        tables = self.report["tables"]
        self.assertEqual(len(tables["attempt_summary"]), 6)
        self.assertEqual(len(tables["committed_summary"]), 3)
        rollback = tables["rollback_summary"]
        self.assertEqual(len(rollback), 1)
        self.assertTrue(rollback[0]["history"].endswith("ROLLING_BACK>RESTORED"))
        cell = tables["noop_vs_migrate"][0]
        self.assertEqual((cell["migrate_n"], cell["no_op_n"]), (3, 1))
        self.assertAlmostEqual(cell["migration_effect_src"], -0.08, places=6)
        self.assertAlmostEqual(tables["workload_distribution"][0]["attempts"], 3)

    def test_latency_report(self) -> None:
        lat = self.report["latency"]
        self.assertEqual(lat["transaction_s"]["n"], 4)  # 3 COMMITTED + 1 RESTORED
        self.assertAlmostEqual(lat["transaction_s"]["median"], 2.0)
        self.assertEqual(lat["attempt_duration_s"]["median"], 129.0)

    def test_write_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = write_report(self.report, Path(tmp))
            names = {p.name for p in written}
            for name in ("migration_outcome_manifest.json", "split_manifest.json",
                         "dataset_quality_report.json", "migration_latency.json",
                         "analytic_check.json", "attempt_summary.csv", "committed_summary.csv",
                         "rollback_summary.csv", "workload_distribution.csv",
                         "noop_vs_migrate.csv"):
                self.assertIn(name, names)
            raw = (Path(tmp) / "attempt_summary.csv").read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(raw.count(b"\n"), 7)
            json.loads((Path(tmp) / "dataset_quality_report.json").read_text())

    def test_cli_with_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "scripts/analysis/migration_dataset_report.py", str(self.root),
                 "--out", tmp], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            figures = sorted(p.name for p in (Path(tmp) / "figures").glob("*.png"))
            self.assertEqual(len(figures), 8, figures)
            self.assertIn("eligible COMMITTED 3", result.stdout)


class HelperTest(unittest.TestCase):
    def test_describe(self) -> None:
        d = describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((d["n"], d["min"], d["median"], d["max"], d["p95"]), (4, 1.0, 2.5, 4.0, 4.0))
        self.assertIsNone(describe([])["mean"])

    def test_fit_line(self) -> None:
        f = fit_line([0.1, 0.2, 0.4], [0.05, 0.1, 0.2])
        self.assertAlmostEqual(float(f["slope"] or 0), 0.5)
        self.assertAlmostEqual(float(f["pearson_r"] or 0), 1.0)
        self.assertAlmostEqual(float(f["slope_through_origin"] or 0), 0.5)
        self.assertIsNone(fit_line([0.1, 0.2], [1.0, 2.0])["slope"])

    def test_csv_lf_and_none(self) -> None:
        text = to_csv([{"a": 1.23456789, "b": None, "c": True}], ["a", "b", "c"])
        self.assertEqual(text, "a,b,c\n1.23457,,true\n")


if __name__ == "__main__":
    unittest.main()

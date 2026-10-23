"""Day 3 pilot report: bug checks, measurements, time estimate (no network)."""
from __future__ import annotations

import copy
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
from src.experiments.migration.pilot import (
    build_pilot_report,
    percentile,
    render_markdown,
    role_mismatches,
    validate_pilot_settings,
)
from src.experiments.migration.schema import StateSnapshotView
from src.experiments.migration.writer import write_attempt

CONFIG = load_outcome_config(Path("configs/experiments/migration_outcome.yaml"))
SETTINGS = {
    "pilot_root": "x", "min_attempts": 1, "collection_budget_hours": 48,
    "budget_utilisation": 0.8, "cleanup_overhead_s": 15, "destination_level_tolerance": 0.10,
    "hot_switch_share_tolerance": 0.10,
    "workload_drop_ratio": 0.5, "delta_ratio_range": [0.3, 3.0], "timeout_headroom": 1.5,
}


def inputs(kind: str = "committed", n: int = 1, *, start_min: int = 0,
           duration_s: int = 60, **meta: Any) -> dict[str, Any]:
    """attempt_inputs with its own attempt id, transaction id and timestamps."""
    attempt_id = f"mig-stable-middle-dst45-c1-c2-r{n:02d}"
    out = fx.attempt_inputs(kind, attempt_id=attempt_id)
    started = fx.T0 + timedelta(minutes=start_min)
    out["metadata"] = dataclasses.replace(
        out["metadata"], started_at=started.isoformat(),
        finished_at=(started + timedelta(seconds=duration_s)).isoformat(), **meta)
    body = out["transaction_raw"].get("body")
    if isinstance(body, dict) and "transaction" in body:
        body["transaction"]["transaction_id"] = f"tx-{n:012d}"
    return out


class PilotTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)
        self.settings = validate_pilot_settings(SETTINGS)

    def report(self) -> dict[str, Any]:
        return build_pilot_report(self.root, CONFIG, self.settings)

    def test_clean_pilot_measures_time_and_settle(self) -> None:
        write_attempt(self.root, **inputs("committed", 1, start_min=0, duration_s=60))
        write_attempt(self.root, **inputs("no_op", 2, start_min=1, duration_s=55))
        report = self.report()
        self.assertEqual(report["dataset_bugs"], [])
        self.assertEqual(report["attempt_bugs"], {})
        self.assertTrue(report["gate"]["no_code_bugs"])
        self.assertTrue(report["gate"]["validator_pass"])
        self.assertFalse(report["gate"]["coverage"])  # only stable, one direction
        row = {r["attempt_id"]: r for r in report["attempt_rows"]}
        committed = row["mig-stable-middle-dst45-c1-c2-r01"]
        # transaction ends at T0+2 s, 5th post snapshot at T0+14 s
        self.assertAlmostEqual(committed["settle_s"], 12.0)
        self.assertAlmostEqual(committed["cycle_s"], 60.0)  # next attempt starts 60 s later
        self.assertAlmostEqual(committed["delta_ratio_src"], 1.0, places=6)
        self.assertAlmostEqual(committed["delta_ratio_dst"], 1.0, places=6)
        timeout = report["decisions"]["post_window_timeout_s"]
        self.assertEqual(timeout["needed"], 20)  # 12 s x 1.5, rounded up to 5 s
        self.assertEqual(timeout["suggested"], 60.0)  # never below the current 60 s
        timing = report["timing"]
        self.assertEqual(timing["planned_attempts"], 340)
        self.assertIn("gradual", timing["extrapolated_workloads"])
        self.assertGreater(timing["estimate_hours"], 0)
        self.assertIn("Gate Day 3", render_markdown(report))

    def test_roles_not_updated_after_commit(self) -> None:
        data = inputs("committed")
        for snap in data["after_raw"]:
            for c in snap["controllers"]:
                c["managed_switch_count"] = 10  # controllers still report the old roles
        write_attempt(self.root, **data)
        bugs = self.report()["attempt_bugs"]["mig-stable-middle-dst45-c1-c2-r01"]
        self.assertTrue(any("roles not updated" in b for b in bugs), bugs)

    def test_post_window_taken_before_transaction_end(self) -> None:
        data = inputs("committed")
        late = (fx.T0 + timedelta(seconds=30)).isoformat()
        data["transaction_raw"]["body"]["transaction"]["finished_at"] = late
        write_attempt(self.root, **data)
        bugs = self.report()["attempt_bugs"]["mig-stable-middle-dst45-c1-c2-r01"]
        self.assertIn("post window starts before the transaction ended", bugs)

    def test_switch_load_lost_after_move(self) -> None:
        data = inputs("committed")
        for snap in data["after_raw"]:
            snap["switches"] = [s for s in snap["switches"] if s["switch_id"] != "s1"]
        write_attempt(self.root, **data)
        bugs = self.report()["attempt_bugs"]["mig-stable-middle-dst45-c1-c2-r01"]
        self.assertIn("s1 load missing from c2 after the move", bugs)

    def test_history_must_end_in_the_status(self) -> None:
        data = inputs("committed")
        data["transaction_raw"]["body"]["transaction"]["history"].pop()
        write_attempt(self.root, **data)
        bugs = self.report()["attempt_bugs"]["mig-stable-middle-dst45-c1-c2-r01"]
        # the schema already refuses it, so the attempt is unreadable and a bug
        self.assertTrue(any("ends in state 'VERIFYING'" in b for b in bugs), bugs)

    def test_freeze_change_and_ownership_reset(self) -> None:
        write_attempt(self.root, **inputs("committed", 1))
        write_attempt(self.root, **inputs("committed", 2, start_min=1))
        other = inputs("restored", 3, start_min=2, git_commit="1" * 40)
        owners = fx.default_ownership()
        owners["s20"] = "c1"
        for snap in other["before_raw"][:1]:
            snap["ownership"] = copy.deepcopy(snap["ownership"])
            for o in snap["ownership"]:
                o["owner_controller_id"] = owners[o["switch_id"]]
        write_attempt(self.root, **other)
        report = self.report()
        self.assertTrue(any(b.startswith("git_commit changed") for b in report["dataset_bugs"]))
        bugs = report["attempt_bugs"]["mig-stable-middle-dst45-c1-c2-r03"]
        self.assertIn("initial ownership differs from the other attempts (not reset)", bugs)

    def test_review_items(self) -> None:
        write_attempt(self.root, **inputs("committed", 1, destination_load_level=0.75))
        write_attempt(self.root, **inputs("skipped", 2, start_min=1))
        review = self.report()["review"]
        self.assertTrue(any("destination U 0.450 vs planned 0.75" in r
                            for r in review["mig-stable-middle-dst45-c1-c2-r01"]))
        self.assertIn("SKIPPED: source_not_master", review["mig-stable-middle-dst45-c1-c2-r02"])

    def test_hot_switch_share_far_from_its_level(self) -> None:
        data = inputs("committed", 1)
        data["metadata"] = dataclasses.replace(
            data["metadata"], attempt_id="mig-hot_switch-peak-dst45-c1-c2-r01",
            workload="hot_switch", phase="peak")
        data["action"] = dataclasses.replace(data["action"], target_hot=True,
                                             selection_rule="hot_switch_alternating")
        write_attempt(self.root, **data)  # s1 carries 0.10 of c1, the peak level is 0.55
        review = self.report()["review"]["mig-hot_switch-peak-dst45-c1-c2-r01"]
        self.assertIn("hot switch share 0.100 vs level 0.55", review)

    def test_injected_rollback_is_not_a_bug(self) -> None:
        write_attempt(self.root, **inputs("restored", 1))
        report = self.report()
        self.assertEqual(report["attempt_bugs"], {})
        self.assertEqual(report["unintended_rollbacks"], 0)


class HelpersTest(unittest.TestCase):
    def test_percentile_nearest_rank(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 4.0)
        self.assertIsNone(percentile([], 95))

    def test_role_mismatch(self) -> None:
        raw = fx.make_twin_state({"c1": 0.5, "c2": 0.5})
        self.assertEqual(role_mismatches(StateSnapshotView.from_twin_state(raw)), [])
        raw["controllers"][0]["managed_switch_count"] = 9
        self.assertEqual(role_mismatches(StateSnapshotView.from_twin_state(raw)),
                         ["c1: managed 9 != owned 10"])

    def test_settings_are_checked(self) -> None:
        for key, value in (("delta_ratio_range", [1.2, 3.0]), ("budget_utilisation", 0),
                           ("workload_drop_ratio", 1.5), ("timeout_headroom", 0.5)):
            with self.assertRaises(ValueError, msg=key):
                validate_pilot_settings({**SETTINGS, key: value})
        with self.assertRaises(ValueError):
            validate_pilot_settings({k: v for k, v in SETTINGS.items() if k != "min_attempts"})


class CliTest(unittest.TestCase):
    def test_exit_code_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            try:
                write_attempt(root, **inputs("committed"))
                out_json, out_md = Path(tmp) / "r.json", Path(tmp) / "r.md"
                result = subprocess.run(
                    [sys.executable, "scripts/experiments/report_migration_pilot.py", str(root),
                     "--out-json", str(out_json), "--out-md", str(out_md)],
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 1, result.stderr)  # coverage incomplete
                self.assertEqual(json.loads(out_json.read_text())["attempts"], 1)
                self.assertIn("Gate Day 3: **FAIL**", out_md.read_text())
            finally:
                fx.unlock_tree(Path(tmp))


if __name__ == "__main__":
    unittest.main()

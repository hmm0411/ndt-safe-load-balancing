from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.twin.decision import safety_report as rep
from src.twin.decision.safety_metrics import ReplayRecord, compute_metrics
from src.twin.decision.safety_replay import ReplayResult


def sample() -> ReplayResult:
    records = (
        ReplayRecord("a", True, 0.90),  # accepted, safe
        ReplayRecord("b", True, 1.10),  # accepted, unsafe
        ReplayRecord("c", False, 1.20),  # rejected, unsafe
        ReplayRecord("d", False, 0.80),  # rejected, safe
        ReplayRecord("e", True, 0.50),  # accepted, safe
    )
    return ReplayResult(
        compute_metrics(records, 1.0), {"dst_utilization": 2},
        {"cooldown_clear:NOT_AVAILABLE": 5}, records,
    )


class TablesTest(unittest.TestCase):
    def test_confusion_counts(self) -> None:
        row = rep.confusion_row("test", "gate_b", sample().metrics)
        self.assertEqual(row, {
            "split": "test", "gate": "gate_b", "records": 5, "unsafe_rejected": 1,
            "unsafe_accepted": 1, "safe_rejected": 1, "safe_accepted": 2,
        })
        self.assertEqual(set(row), set(rep.CONFUSION_COLUMNS))

    def test_ablation_row_has_every_column_and_a_wilson_interval(self) -> None:
        row = rep.ablation_row("stress", "gate_a", 0.0, 0.0, sample().metrics)
        self.assertEqual(set(row), set(rep.ABLATION_COLUMNS))
        self.assertEqual((row["uar_num"], row["uar_den"], row["uar_value"]), (1, 2, 0.5))
        self.assertTrue(0.0 < row["uar_low"] < 0.5 < row["uar_high"] < 1.0)

    def test_reason_rows_keep_failed_and_skipped_apart(self) -> None:
        rows = rep.reason_rows("test", "gate_a", sample())
        self.assertEqual([(r["kind"], r["reason"], r["count"]) for r in rows], [
            ("failed", "dst_utilization", 2), ("skipped", "cooldown_clear:NOT_AVAILABLE", 5),
        ])


class FilesTest(unittest.TestCase):
    def test_csv_writes_blank_for_missing_values_and_returns_the_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            digest = rep.write_csv(path, [{"a": 1, "b": None}], ("a", "b"))
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"a": "1", "b": ""}])

    def test_lock_is_exclusive_and_completes_with_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            rep.open_lock(path, "8c548830861b", "once")
            with self.assertRaises(FileExistsError):
                rep.open_lock(path, "8c548830861b", "once")
            self.assertEqual(json.loads(path.read_text())["status"], "opened")
            rep.complete_lock(path, {"x.csv": "abc"})
            data = json.loads(path.read_text())
            self.assertEqual((data["status"], data["outputs_sha256"]), ("completed", {"x.csv": "abc"}))
            self.assertEqual(data["policy_sha12"], "8c548830861b")


if __name__ == "__main__":
    unittest.main()

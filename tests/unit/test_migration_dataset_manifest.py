"""validate_dataset: counts, duplicate transactions, matrix coverage, no overwrite, _invalid."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.experiments.migration import fixtures as fx
from src.experiments.migration.config import load_outcome_config
from src.experiments.migration.validator import validate_dataset
from src.experiments.migration.writer import INVALID_DIR, write_attempt


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def test_summary_counts(self) -> None:
        write_attempt(self.root, **fx.attempt_inputs("committed"))
        restored = fx.attempt_inputs("restored", attempt_id="mig-stable-middle-dst45-c1-c2-r02")
        restored["transaction_raw"]["body"]["transaction"]["transaction_id"] = "tx-restored0002"
        write_attempt(self.root, **restored)
        write_attempt(self.root, **fx.attempt_inputs("no_op"))
        summary = validate_dataset(self.root)
        self.assertEqual(summary["attempts"], 3)
        self.assertEqual(summary["invalid"], 0, summary["invalid_attempts"])
        self.assertEqual(summary["training_eligible"], 2)
        self.assertEqual(summary["counts"]["transaction_status"],
                         {"COMMITTED": 1, "None": 1, "RESTORED": 1})
        self.assertEqual(summary["counts"]["direction"], {"c1->c2": 3})

    def test_duplicate_transaction_is_flagged(self) -> None:
        write_attempt(self.root, **fx.attempt_inputs("committed"))
        write_attempt(self.root, **fx.attempt_inputs(
            "committed", attempt_id="mig-stable-middle-dst45-c1-c2-r02"))
        summary = validate_dataset(self.root)
        self.assertEqual(summary["invalid"], 2)
        self.assertIn("tx-0123456789ab", summary["duplicate_transactions"])

    def test_matrix_coverage(self) -> None:
        write_attempt(self.root, **fx.attempt_inputs("committed"))
        config = load_outcome_config(Path("configs/experiments/migration_outcome.yaml"))
        summary = validate_dataset(self.root, config)
        self.assertEqual(summary["planned"], 340)
        self.assertEqual(len(summary["missing"]), 339)
        self.assertEqual(summary["unplanned"], [])

    def test_never_overwrite(self) -> None:
        write_attempt(self.root, **fx.attempt_inputs("committed"))
        with self.assertRaises(FileExistsError):
            write_attempt(self.root, **fx.attempt_inputs("committed"))

    def test_crash_while_writing_goes_to_invalid(self) -> None:
        target = "src.experiments.migration.writer.write_checksums"
        with mock.patch(target, side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                write_attempt(self.root, **fx.attempt_inputs("committed"))
        self.assertFalse((self.root / "mig-stable-middle-dst45-c1-c2-r01").exists())
        quarantined = list((self.root / INVALID_DIR).iterdir())
        self.assertEqual(len(quarantined), 1)
        self.assertTrue((quarantined[0] / "INVALID_REASON.txt").is_file())
        self.assertEqual(validate_dataset(self.root)["attempts"], 0)


if __name__ == "__main__":
    unittest.main()

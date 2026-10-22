"""write_attempt + validate_attempt on a full synthetic attempt."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.validator import validate_attempt
from src.experiments.migration.writer import CHECKSUM_FILE, write_attempt


class ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def test_committed_attempt_is_valid_and_eligible(self) -> None:
        written = write_attempt(self.root, **fx.attempt_inputs("committed"))
        self.assertEqual(written.path.name, "mig-stable-middle-dst45-c1-c2-r01")
        self.assertTrue(written.validation["valid"], written.validation["errors"])
        self.assertTrue(written.validation["training_eligible"])
        self.assertEqual(written.validation["transaction_status"], "COMMITTED")
        stored = json.loads((written.path / "validation.json").read_text())
        self.assertEqual(stored["valid"], True)
        again = validate_attempt(written.path)  # with checksums this time
        self.assertTrue(again["valid"], again["errors"])

    def test_directory_is_read_only_with_checksums(self) -> None:
        path = write_attempt(self.root, **fx.attempt_inputs("committed")).path
        self.assertTrue((path / CHECKSUM_FILE).is_file())
        self.assertFalse(os.stat(path / "outcome.json").st_mode & stat.S_IWUSR)
        names = (path / CHECKSUM_FILE).read_text().split()
        self.assertIn("validation.json", names)

    def test_edited_file_breaks_the_checksum(self) -> None:
        path = write_attempt(self.root, **fx.attempt_inputs("committed")).path
        fx.unlock_tree(path)
        target = path / "action.json"
        target.write_text(target.read_text().replace("seeded_random", "manual"))
        report = validate_attempt(path)
        self.assertFalse(report["valid"])
        self.assertIn("checksum mismatch: action.json", report["errors"])

    def test_edited_outcome_is_detected(self) -> None:
        path = write_attempt(self.root, **fx.attempt_inputs("committed")).path
        fx.unlock_tree(path)
        data = json.loads((path / "outcome.json").read_text())
        data["outcome"]["delta_src_u_mean"] = -0.5
        (path / "outcome.json").write_text(json.dumps(data))
        report = validate_attempt(path, check_checksums=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any(e.startswith("outcome.delta_src_u_mean") for e in report["errors"]))

    def test_missing_file_is_invalid(self) -> None:
        path = write_attempt(self.root, **fx.attempt_inputs("committed")).path
        fx.unlock_tree(path)
        (path / "action.json").unlink()
        report = validate_attempt(path)
        self.assertFalse(report["valid"])
        self.assertIn("missing files", report["errors"][0])

    def test_rollback_attempt_is_valid_but_not_eligible(self) -> None:
        report = write_attempt(self.root, **fx.attempt_inputs("restored")).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(report["transaction_status"], "RESTORED")

    def test_no_op_attempt(self) -> None:
        written = write_attempt(self.root, **fx.attempt_inputs("no_op"))
        self.assertTrue(written.path.name.startswith("noop-"))
        self.assertTrue(written.validation["valid"], written.validation["errors"])
        self.assertTrue(written.validation["training_eligible"])
        self.assertIsNone(written.validation["transaction_status"])


if __name__ == "__main__":
    unittest.main()

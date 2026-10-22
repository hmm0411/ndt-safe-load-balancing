"""CLI: exit 0 on a clean dataset, 1 when any attempt is invalid, writes --out."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.writer import write_attempt

SCRIPT = Path("scripts/experiments/validate_migration_outcome.py")


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, check=False)

    def test_clean_dataset(self) -> None:
        write_attempt(self.root / "data", **fx.attempt_inputs("committed"))
        out = self.root / "summary.json"
        result = self.run_cli(str(self.root / "data"), "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(out.read_text())["attempts"], 1)

    def test_invalid_dataset(self) -> None:
        inputs = fx.attempt_inputs("committed")
        inputs["before_raw"] = inputs["before_raw"][:3]
        write_attempt(self.root / "data", **inputs)
        result = self.run_cli(str(self.root / "data"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

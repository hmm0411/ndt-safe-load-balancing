"""VM only: validate the smoke attempts written by the real runner.

Skipped when data/experiment_runs/migration-smoke* does not exist (CI, sandbox).
Run on the VM after Vy's smoke run:
    python3 -m unittest tests.integration.test_migration_smoke_dirs -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.migration.validator import validate_attempt, validate_dataset

ROOTS = sorted(p for p in Path("data/experiment_runs").glob("migration-smoke*") if p.is_dir())


def attempts() -> list[Path]:
    return [d for root in ROOTS for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_"))]


@unittest.skipUnless(ROOTS, "no data/experiment_runs/migration-smoke* on this machine")
class SmokeDirsTest(unittest.TestCase):
    def test_every_attempt_is_valid(self) -> None:
        for root in ROOTS:
            summary = validate_dataset(root)
            self.assertGreater(summary["attempts"], 0, root)
            self.assertEqual(summary["invalid"], 0, summary["invalid_attempts"])

    def test_success_attempt_is_eligible(self) -> None:
        reports = [validate_attempt(d) for d in attempts()]
        committed = [r for r in reports if r["transaction_status"] == "COMMITTED"]
        self.assertTrue(committed, "smoke run has no COMMITTED attempt")
        self.assertTrue(all(r["training_eligible"] for r in committed),
                        [r["errors"] + r["warnings"] for r in committed])

    def test_rollback_attempt_is_valid_not_eligible(self) -> None:
        reports = [validate_attempt(d) for d in attempts()]
        restored = [r for r in reports if r["transaction_status"] == "RESTORED"]
        self.assertTrue(restored, "smoke run has no RESTORED attempt (failure injection)")
        for report in restored:
            self.assertTrue(report["valid"], report["errors"])
            self.assertFalse(report["training_eligible"])


if __name__ == "__main__":
    unittest.main()

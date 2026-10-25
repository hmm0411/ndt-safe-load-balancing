"""Data Card (Week 11 Day 7): every number comes from the evidence files."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.experiments.migration import fixtures as fx
from src.experiments.migration.config import load_outcome_config
from src.experiments.migration.dataset_card import build_dataset_card, load_card_config
from src.experiments.migration.dataset_report import build_dataset_report, write_report
from src.experiments.migration.writer import write_attempt
from test_migration_dataset_report import SETTINGS, attempt

CONFIG_PATH = Path("configs/experiments/migration_outcome.yaml")
CARD_PATH = Path("configs/experiments/migration_dataset_card.yaml")
CONFIG = load_outcome_config(CONFIG_PATH)


class DatasetCardTest(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    evidence: Path
    card: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "data"
        for n, (kind, repeat) in enumerate(
                [("committed", "r01"), ("committed", "r04"), ("no_op", "r01")], 1):
            write_attempt(root, **attempt(kind, repeat, n))
        cls.evidence = Path(cls.tmp.name) / "evidence"
        write_report(build_dataset_report(root, CONFIG, SETTINGS, CONFIG_PATH), cls.evidence)
        cls.card = build_dataset_card(CONFIG, load_card_config(CARD_PATH), cls.evidence,
                                      {"stable": 120.0})

    @classmethod
    def tearDownClass(cls) -> None:
        fx.unlock_tree(Path(cls.tmp.name))
        cls.tmp.cleanup()

    def test_section_37_fields(self) -> None:
        for key in ("purpose", "topology", "action", "workloads", "action_phases",
                    "destination_load_levels", "directions", "repeats", "planned_attempts",
                    "pre_window", "post_window", "quality_gate", "failed_transactions",
                    "raw_imputation", "split"):
            self.assertIn(key, self.card)
        self.assertEqual(self.card["planned_attempts"],
                         {"migrate": 300, "no_op": 40, "total": 340})
        self.assertEqual(self.card["directions"], ["c1->c2", "c2->c1"])
        self.assertEqual(self.card["phases_per_workload"], [3])
        self.assertEqual(self.card["raw_imputation"], "none")

    def test_numbers_come_from_the_evidence(self) -> None:
        actual = self.card["actual_attempts"]
        self.assertEqual((actual["total_attempts"], actual["committed_count"],
                          actual["no_op_count"]), (3, 2, 1))
        self.assertEqual(self.card["split"]["train"]["attempts"], 2)
        self.assertEqual(self.card["split"]["validation"]["attempts"], 1)
        self.assertEqual(self.card["collection"]["workload_duration_s"], [120.0])
        self.assertEqual(self.card["measured_ranges"]["dst_u_after_mean"]["max"], 0.53)
        self.assertEqual(self.card["noop_controls"][0]["migration_effect_dst"], 0.08)

    def test_card_config_is_checked(self) -> None:
        data = yaml.safe_load(CARD_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "card.yaml"
            bad.write_text(yaml.safe_dump({**data, "backup": {"sha256": "abc"}}))
            with self.assertRaises(ValueError):
                load_card_config(bad)
            del data["purpose"]
            bad.write_text(yaml.safe_dump(data))
            with self.assertRaises(ValueError):
                load_card_config(bad)

    def test_missing_evidence_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_dataset_card(CONFIG, load_card_config(CARD_PATH), Path(tmp), {})

    def test_cli_writes_the_card(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/analysis/migration_dataset_card.py",
             "--evidence", str(self.evidence)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        card = json.loads((self.evidence / "dataset_card.json").read_text())
        self.assertEqual(card["dataset_version"], "migration-outcome-raw-v1")


if __name__ == "__main__":
    unittest.main()

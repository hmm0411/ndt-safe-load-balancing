"""configs/experiments/migration_outcome.yaml expands to the planned matrix."""
from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path

from src.experiments.migration.config import (
    attempt_id,
    expand_attempts,
    load_outcome_config,
    validate_outcome_config,
)

CONFIG = Path("configs/experiments/migration_outcome.yaml")
ID = re.compile(r"^(mig|noop)-[a-z0-9_]+-[a-z0-9_]+-dst\d{2}-c\d+-c\d+-r\d{2}$")


class OutcomeConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_outcome_config(CONFIG)
        self.plans = expand_attempts(self.config)

    def test_matrix_size(self) -> None:
        kinds = Counter(p.action_type for p in self.plans)
        self.assertEqual(kinds["MIGRATE"], 5 * 3 * 2 * 2 * 5)
        self.assertEqual(kinds["NO_OP"], 40)

    def test_ids_are_unique_and_parseable(self) -> None:
        ids = [p.attempt_id for p in self.plans]
        self.assertEqual(len(ids), len(set(ids)))
        for item in ids:
            self.assertRegex(item, ID)
        self.assertEqual(
            attempt_id("MIGRATE", "hot_switch", "peak", 0.75, "c2", "c1", "r04"),
            "mig-hot_switch-peak-dst75-c2-c1-r04",
        )

    def test_split_by_repeat(self) -> None:
        for plan in self.plans:
            expected = {"r01": "train", "r02": "train", "r03": "train",
                        "r04": "validation", "r05": "test"}[plan.repeat]
            self.assertEqual(plan.split, expected)
            self.assertEqual(plan.seed, self.config.repeats[plan.repeat])

    def test_no_op_controls_spread_over_repeats(self) -> None:
        per_repeat = Counter(p.repeat for p in self.plans if p.action_type == "NO_OP")
        self.assertEqual(set(per_repeat.values()), {8})
        phases = {(p.workload, p.phase) for p in self.plans if p.action_type == "NO_OP"}
        self.assertIn(("gradual", "u60"), phases)

    def test_bad_configs_are_rejected(self) -> None:
        good = {
            "schema_version": "migration-outcome-v1", "topology": "2C20S",
            "controllers_config": "x", "output_root": "y",
            "pre_window_valid_snapshots": 5, "post_window_valid_snapshots": 5,
            "post_window_timeout_s": 60, "workloads": {"stable": ["early"]},
            "destination_load_levels": [0.45], "directions": [["c1", "c2"]],
            "repeats": {"r01": 1}, "split": {"train": ["r01"]}, "no_op_controls": 0,
        }
        validate_outcome_config(good)
        for key, value in (
            ("schema_version", "v0"),
            ("split", {"train": []}),
            ("directions", [["c1", "c1"]]),
            ("destination_load_levels", [1.2]),
            ("workloads", {"hot-switch": ["peak"]}),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_outcome_config({**good, key: value})


if __name__ == "__main__":
    unittest.main()

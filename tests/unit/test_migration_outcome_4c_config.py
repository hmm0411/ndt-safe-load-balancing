"""The two 4C migration-outcome configs and the seeded destination draw."""
from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from src.experiments.migration.config import (
    draw_destination,
    expand_attempts,
    load_outcome_config,
    validate_outcome_config,
)

TRANSFER = Path("configs/experiments/migration_outcome_4c_transfer_v1.yaml")
NOPROBE = Path("configs/experiments/migration_outcome_4c_noprobe_v1.yaml")


class TransferConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_outcome_config(TRANSFER)
        self.plans = expand_attempts(self.config)

    def test_fifty_attempts_from_c1(self) -> None:
        self.assertEqual(len(self.plans), 50)
        self.assertEqual({p.action_type for p in self.plans}, {"MIGRATE"})
        self.assertEqual({p.source_controller for p in self.plans}, {"c1"})
        self.assertLessEqual({p.target_controller for p in self.plans}, {"c2", "c3", "c4"})

    def test_draw_is_reproducible_and_uses_every_candidate(self) -> None:
        again = expand_attempts(load_outcome_config(TRANSFER))
        self.assertEqual([p.attempt_id for p in self.plans], [p.attempt_id for p in again])
        self.assertEqual(set(Counter(p.target_controller for p in self.plans)), {"c2", "c3", "c4"})
        self.assertEqual(draw_destination(9701, "stable", "middle", 0.45, ("c2", "c3", "c4")),
                         draw_destination(9701, "stable", "middle", 0.45, ("c2", "c3", "c4")))

    def test_directions_and_draw_are_exclusive(self) -> None:
        data = {"schema_version": "migration-outcome-v1", "topology": "4C40S",
                "controllers_config": "x", "output_root": "o", "pre_window_valid_snapshots": 5,
                "post_window_valid_snapshots": 5, "post_window_timeout_s": 60,
                "workloads": {"stable": ["middle"]}, "destination_load_levels": [0.45],
                "destination_draw": {"source": "c1", "candidates": ["c2"]},
                "directions": [["c1", "c2"]], "repeats": {"r01": 1}, "split": {"train": ["r01"]},
                "no_op_controls": 0}
        with self.assertRaises(ValueError):
            validate_outcome_config(data)
        del data["directions"]
        data["destination_draw"] = {"source": "c1", "candidates": ["c1", "c2"]}
        with self.assertRaises(ValueError):
            validate_outcome_config(data)


class NoProbeConfigTest(unittest.TestCase):
    def test_matrix_follows_the_2c_design_with_five_levels(self) -> None:
        plans = expand_attempts(load_outcome_config(NOPROBE))
        kinds = Counter(p.action_type for p in plans)
        self.assertEqual(kinds["MIGRATE"], 5 * 3 * 5 * 2 * 5)
        self.assertEqual(kinds["NO_OP"], 100)
        self.assertEqual({p.destination_load_level for p in plans},
                         {0.45, 0.75, 0.85, 0.92, 0.98})
        self.assertEqual({p.seed for p in plans}, {9901, 9902, 9903, 9904, 9905})
        self.assertEqual(len({p.attempt_id for p in plans}), len(plans))


if __name__ == "__main__":
    unittest.main()

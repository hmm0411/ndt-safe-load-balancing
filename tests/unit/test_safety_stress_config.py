import unittest
from pathlib import Path

from src.experiments.migration.config import expand_attempts, load_outcome_config

ROOT = Path(__file__).resolve().parents[2]


class SafetyStressConfigTest(unittest.TestCase):
    def test_stress_set_is_new_and_evaluation_only(self) -> None:
        stress = load_outcome_config(ROOT / "configs/experiments/safety_stress_v1.yaml")
        raw_v2 = load_outcome_config(ROOT / "configs/experiments/migration_outcome_v2.yaml")
        attempts = expand_attempts(stress)
        self.assertEqual(len(attempts), 60)
        self.assertTrue(all(a.action_type == "MIGRATE" for a in attempts))
        self.assertEqual({a.split for a in attempts}, {"stress"})
        self.assertFalse(set(stress.repeats.values()) & set(raw_v2.repeats.values()))
        self.assertFalse(set(stress.repeats.values()) & {101, 102, 103, 104, 105})
        self.assertEqual(stress.workloads, raw_v2.workloads)
        self.assertNotEqual(stress.output_root, raw_v2.output_root)
        self.assertTrue(all(level > max(raw_v2.destination_load_levels)
                            for level in stress.destination_load_levels))


if __name__ == "__main__":
    unittest.main()

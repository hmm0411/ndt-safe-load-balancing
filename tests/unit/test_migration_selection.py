import random
import unittest

from src.experiments.migration.selection import select_switch

OWNERSHIP = [
    {"switch_id": f"s{i}", "owner_controller_id": "c1" if i <= 10 else "c2"}
    for i in range(1, 21)
]


class SelectSwitchTests(unittest.TestCase):
    def test_random_choice_belongs_to_source(self) -> None:
        for seed in range(50):
            chosen = select_switch(OWNERSHIP, "c1", random.Random(seed))
            self.assertIn(chosen, {f"s{i}" for i in range(1, 11)})

    def test_same_seed_same_switch(self) -> None:
        a = select_switch(OWNERSHIP, "c2", random.Random(9201))
        b = select_switch(OWNERSHIP, "c2", random.Random(9201))
        self.assertEqual(a, b)

    def test_hot_switch_is_chosen_exactly(self) -> None:
        chosen = select_switch(OWNERSHIP, "c1", random.Random(1), hot_switch_id="s3")
        self.assertEqual(chosen, "s3")

    def test_hot_switch_must_belong_to_source(self) -> None:
        with self.assertRaises(ValueError):
            select_switch(OWNERSHIP, "c1", random.Random(1), hot_switch_id="s15")

    def test_source_without_switch_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_switch(OWNERSHIP, "c9", random.Random(1))


if __name__ == "__main__":
    unittest.main()

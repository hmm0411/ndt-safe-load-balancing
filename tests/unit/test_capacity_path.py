from __future__ import annotations

import unittest
from pathlib import Path

from src.twin.capacity.model import resolve_capacity_path

ROOT = Path("/repo")


class ResolveCapacityPathTests(unittest.TestCase):
    def test_default_path_is_unchanged(self) -> None:
        self.assertEqual(
            resolve_capacity_path(ROOT, {}),
            ROOT / "data" / "benchmarks" / "controller_capacity.json",
        )

    def test_empty_value_falls_back_to_default(self) -> None:
        self.assertEqual(
            resolve_capacity_path(ROOT, {"NDT_CAPACITY_FILE": ""}),
            ROOT / "data" / "benchmarks" / "controller_capacity.json",
        )

    def test_absolute_override(self) -> None:
        env = {"NDT_CAPACITY_FILE": "/tmp/cap.json"}
        self.assertEqual(resolve_capacity_path(ROOT, env), Path("/tmp/cap.json"))

    def test_relative_override_is_under_root(self) -> None:
        env = {"NDT_CAPACITY_FILE": "tmp/cap.json"}
        self.assertEqual(resolve_capacity_path(ROOT, env), ROOT / "tmp" / "cap.json")


if __name__ == "__main__":
    unittest.main()

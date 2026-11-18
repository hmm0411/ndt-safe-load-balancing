from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.common.config import load_yaml
from src.experiments.topologies.runtime_4c40s import (
    DEFAULT_CONFIG,
    Runtime4C40STopo,
    controller_specs,
    expected_ownership,
    validate_ownership,
)

ROOT = Path(__file__).resolve().parents[2]


class Runtime4C40STopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topo = Runtime4C40STopo()
        self.links = {frozenset(link) for link in self.topo.links()}

    def test_shape(self) -> None:
        self.assertEqual(len(self.topo.switches()), 40)
        self.assertEqual(len(self.topo.hosts()), 80)

    def test_workload_and_probe_host_on_every_switch(self) -> None:
        for index in range(1, 41):
            self.assertIn(frozenset((f"h{index}", f"s{index}")), self.links)
            self.assertIn(frozenset((f"p{index}", f"s{index}")), self.links)

    def test_switches_form_chain(self) -> None:
        for index in range(1, 40):
            self.assertIn(frozenset((f"s{index}", f"s{index + 1}")), self.links)

    def test_host_addresses_are_unique(self) -> None:
        ips = set()
        macs = set()
        for name in self.topo.hosts():
            info = self.topo.nodeInfo(name)
            ips.add(info["ip"])
            macs.add(info["mac"])
        self.assertEqual(len(ips), 80)
        self.assertEqual(len(macs), 80)


class Runtime4C40SConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_yaml(ROOT / DEFAULT_CONFIG)

    def test_controller_specs(self) -> None:
        specs = controller_specs(self.config)
        self.assertEqual(
            specs["c3"], (6655, "http://127.0.0.1:8083")
        )
        self.assertEqual(
            specs["c4"], (6656, "http://127.0.0.1:8084")
        )
        self.assertEqual(len(specs), 4)

    def test_ownership_matches_config(self) -> None:
        expected = expected_ownership(self.config)
        validate_ownership(dict(expected), expected)

    def test_ownership_mismatch_is_reported(self) -> None:
        expected = expected_ownership(self.config)
        wrong = dict(expected)
        wrong["s21"] = "c1"
        with self.assertRaisesRegex(RuntimeError, "s21"):
            validate_ownership(wrong, expected)

    def test_missing_ownership_is_rejected(self) -> None:
        expected = expected_ownership(self.config)
        with self.assertRaises(RuntimeError):
            validate_ownership(None, expected)


if __name__ == "__main__":
    unittest.main()

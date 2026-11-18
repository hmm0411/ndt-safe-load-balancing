import re
import unittest
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "controllers_4c40s.yaml"
START_PATTERN = re.compile(
    r"--ofp-tcp-listen-port (\d+) --wsapi-port (\d+)"
)


class Controllers4C40SConfigTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(CONFIG.read_text())

    def test_four_controllers_and_forty_switches(self):
        self.assertEqual(list(self.cfg["controllers"]), ["c1", "c2", "c3", "c4"])
        self.assertEqual(len(self.cfg["switches"]), 40)
        self.assertEqual(len(self.cfg["initial_ownership"]), 40)

    def test_ten_switches_per_controller(self):
        counts = Counter(self.cfg["initial_ownership"].values())
        self.assertEqual(dict(counts), {"c1": 10, "c2": 10, "c3": 10, "c4": 10})

    def test_ownership_covers_every_switch(self):
        self.assertEqual(
            set(self.cfg["initial_ownership"]), set(self.cfg["switches"])
        )

    def test_dpids_are_unique(self):
        dpids = [item["dpid"] for item in self.cfg["switches"].values()]
        self.assertEqual(sorted(dpids), list(range(1, 41)))

    def test_ports_are_unique(self):
        openflow = [c["openflow_port"] for c in self.cfg["controllers"].values()]
        rest = [
            int(c["rest_url"].rsplit(":", 1)[1])
            for c in self.cfg["controllers"].values()
        ]
        self.assertEqual(openflow, [6653, 6654, 6655, 6656])
        self.assertEqual(rest, [8081, 8082, 8083, 8084])

    def test_start_scripts_match_config(self):
        for cid, spec in self.cfg["controllers"].items():
            text = (ROOT / "scripts" / f"start_{cid}.sh").read_text()
            self.assertIn(f"CONTROLLER_ID={cid}", text)
            match = START_PATTERN.search(text)
            assert match is not None, cid
            of_port, rest_port = match.groups()
            self.assertEqual(int(of_port), spec["openflow_port"])
            self.assertEqual(
                int(rest_port), int(spec["rest_url"].rsplit(":", 1)[1])
            )


if __name__ == "__main__":
    unittest.main()

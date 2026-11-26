from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.analysis import campaign_episodes as ce
from src.twin.decision.methods import CONFIG_PATH, parse_methods
from test_campaign import T0, record, snap


class CampaignConfigTest(unittest.TestCase):
    def test_config_matches_protocol_and_methods(self) -> None:
        cfg = ce.load_config(ce.CONFIG)
        methods = parse_methods(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
        for names in cfg["scenarios"].values():
            self.assertTrue(set(names) <= set(methods))
        self.assertEqual(cfg["seeds"], list(range(9801, 9811)))
        self.assertEqual(cfg["warmup_s"], 30)
        self.assertEqual(len(cfg["scenarios"]["B"]) * 5 * 10, 250)
        self.assertTrue((ce.ROOT / cfg["capacity"]).is_file())


class CollectTest(unittest.TestCase):
    def test_collect_writes_one_row_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, out = Path(tmp) / "runs", Path(tmp) / "episodes.csv"
            run = runs / "B-full-burst-s9801"
            (run / "decisions" / "d1").mkdir(parents=True)
            meta = {"episode_id": run.name, "method": "full", "workload": "burst",
                    "seed": 9801, "order": 3, "workload_started_at": T0.isoformat()}
            (run / "meta.json").write_text(json.dumps(meta))
            rates = {"c1": 900.0, "c2": 900.0, "c3": 900.0, "c4": 900.0}
            lines = []
            for t in range(60):
                s = snap(t, 450.0)
                s["controllers"] = [{"controller_id": c, "processed_packet_in_rate": r}
                                    for c, r in rates.items()]
                lines.append(json.dumps(s))
            (run / "snapshots.jsonl").write_text("\n".join(lines) + "\n")
            (run / "decisions" / "d1" / "record.json").write_text(
                json.dumps(record(40, "NO_RISK")))
            code = ce.main(["collect", "--runs", str(runs), "--out", str(out)])
            with out.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        # the 4C capacity file: c_safe ~921-927 pps, so 900 pps is U < 1 everywhere
        self.assertEqual(code, 0)
        self.assertEqual([r["episode_id"] for r in rows], ["B-full-burst-s9801"])
        self.assertEqual(float(rows[0]["overload_time_s"]), 0.0)
        self.assertEqual(rows[0]["decisions"], "1")


if __name__ == "__main__":
    unittest.main()

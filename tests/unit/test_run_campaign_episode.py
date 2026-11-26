from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.experiments import run_campaign_episode as rce
from src.experiments.final_evaluation.campaign import PlannedEpisode

CFG = yaml.safe_load(rce.CONFIG.read_text(encoding="utf-8"))
EP = PlannedEpisode(4, "B-threshold-hot-switch-s9803", "threshold", "hot_switch", 9803)


class SetupTest(unittest.TestCase):
    def test_commands_and_environment(self) -> None:
        st = rce.setup(CFG, EP, 240.0, 45.0, Path("/tmp/ep"), "/venv/bin/python3")
        self.assertEqual(st.loop_s, 195.0)
        self.assertEqual(st.orchestrator_env["NDT_DECISION_METHOD"], "threshold")
        self.assertEqual(st.orchestrator_env["NDT_EPISODE_SEED"], "9803")
        self.assertEqual(st.orchestrator_env["RUN_ID"], EP.episode_id)
        self.assertEqual(Path(st.orchestrator_env["NDT_OUTCOME_CONFIG"]).name,
                         "runtime_4c40s.yaml")
        self.assertEqual(st.topology_env["NDT_WORKLOAD_SCALE"], "c1=1.2")
        cmd = st.topology_cmd
        self.assertEqual(cmd[:4], ["sudo", "-E", "env", f"PYTHONPATH={rce.REPO_ROOT}"])
        self.assertEqual(cmd[4], "/venv/bin/python3")
        self.assertEqual(cmd[cmd.index("--forecast") + 1], "hot_switch")
        profiles = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--profile"]
        self.assertEqual(profiles, ["c2=constant:0.6", "c3=constant:0.6", "c4=constant:0.6"])
        self.assertEqual(Path(st.orchestrator_env["NDT_SAFETY_POLICY"]).name,
                         "safety_policy_4c_v1.yaml")
        for path_key in ("capacity", "capacity_runs", "controllers_config", "forecast_config",
                         "outcome_config", "safety_policy"):
            self.assertTrue((rce.REPO_ROOT / CFG[path_key]).is_file(), path_key)

    def test_duration_must_cover_warmup_and_post_window(self) -> None:
        with self.assertRaises(ValueError):
            rce.setup(CFG, EP, 70.0, 45.0, Path("/tmp/ep"))


class PlanAndRecordsTest(unittest.TestCase):
    def test_episode_from_plan_and_records_of(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.csv"
            with plan.open("w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["order", "episode_id", "method", "workload", "seed"])
                w.writerow([4, EP.episode_id, EP.method, EP.workload, EP.seed])
            self.assertEqual(rce.episode_from_plan(plan, EP.episode_id), EP)
            with self.assertRaises(ValueError):
                rce.episode_from_plan(plan, "B-nope")
            root = Path(tmp) / "records"
            rows = ((EP.episode_id, "2026-12-02T01:00:05+00:00"),
                    ("other", "2026-12-02T01:00:05+00:00"),
                    (EP.episode_id, "2026-12-02T01:00:09+00:00"),
                    (EP.episode_id, "2026-12-01T09:00:00+00:00"))   # an older run, same id
            for i, (scenario, decided) in enumerate(rows):
                (root / f"d{i}").mkdir(parents=True)
                (root / f"d{i}" / "record.json").write_text(
                    json.dumps({"scenario": scenario, "decided_at": decided}))
            (root / "bad").mkdir()
            (root / "bad" / "record.json").write_text("{")
            since = datetime(2026, 12, 2, 1, 0, tzinfo=timezone.utc)
            self.assertEqual([p.parent.name for p in rce.records_of(root, EP.episode_id, since)],
                             ["d0", "d2"])


if __name__ == "__main__":
    unittest.main()

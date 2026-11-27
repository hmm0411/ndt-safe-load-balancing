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
        self.assertEqual(Path(st.orchestrator_env["NDT_RANKING_POLICY"]).name,
                         "ranking_policy_4c_v1.yaml")
        self.assertEqual(Path(st.orchestrator_env["NDT_SAFETY_POLICY"]).name,
                         "safety_policy_4c_v1.yaml")
        for path_key in ("capacity", "capacity_runs", "controllers_config", "forecast_config",
                         "outcome_config", "safety_policy", "ranking_policy"):
            self.assertTrue((rce.REPO_ROOT / CFG[path_key]).is_file(), path_key)

    def test_b_episode_has_no_fault(self) -> None:
        st = rce.setup(CFG, EP, 240.0, 45.0, Path("/tmp/ep"))
        self.assertIsNone(st.fault)
        self.assertNotIn("NDT_FAULT_INJECTION", st.orchestrator_env)

    def test_d_episodes(self) -> None:
        d1 = PlannedEpisode(1, "D-full-d1-s9801", "full", "d1", 9801)
        st = rce.setup(CFG, d1, 240.0, 45.0, Path("/tmp/ep"))
        assert st.fault is not None
        self.assertEqual(st.fault.scenario, "d1")
        self.assertEqual(st.orchestrator_env["NDT_FAULT_INJECTION"], "1")
        cmd = st.topology_cmd
        self.assertEqual(cmd[cmd.index("--forecast") + 1], "stable")
        profiles = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--profile"]
        self.assertEqual(profiles, ["c2=constant:0.6", "c3=constant:0.6", "c4=constant:0.6",
                                    "c1=constant:1.0"])
        d7 = PlannedEpisode(2, "D-threshold-d7-u090-s9802", "threshold", "d7-u090", 9802)
        st7 = rce.setup(CFG, d7, 240.0, 45.0, Path("/tmp/ep"))
        cmd7 = st7.topology_cmd
        self.assertIn("c4=constant:0.9", [cmd7[i + 1] for i, x in enumerate(cmd7)
                                          if x == "--profile"])
        self.assertEqual(st7.other_level, 0.9)

    def test_2c20s_episode(self) -> None:
        cfg2 = yaml.safe_load((rce.REPO_ROOT / "configs/experiments/campaign_2c_v1.yaml")
                              .read_text(encoding="utf-8"))
        self.assertEqual(rce.controller_ports(cfg2), {"c1": (6653, 8081), "c2": (6654, 8082)})
        ep = PlannedEpisode(1, "B2-full-hot-switch-s9801", "full", "hot_switch", 9801)
        st = rce.setup(cfg2, ep, 240.0, 45.0, Path("/tmp/ep"))
        cmd = st.topology_cmd
        self.assertIn("src.experiments.topologies.runtime_2c20s", cmd)
        self.assertIn("--forecast-hot-switch", cmd)
        self.assertNotIn("--forecast", cmd)
        self.assertNotIn("--capacity-experiment", cmd)
        self.assertEqual([cmd[i + 1] for i, x in enumerate(cmd) if x == "--profile"],
                         ["c2=constant:0.6"])
        env = st.orchestrator_env
        self.assertEqual(Path(env["NDT_CONTROLLERS_CONFIG"]).name, "controllers_2c20s.yaml")
        self.assertEqual(Path(env["NDT_SAFETY_POLICY"]).name, "safety_policy_v1.yaml")
        self.assertEqual(Path(env["NDT_OUTCOME_CONFIG"]).name, "runtime.yaml")
        for key in ("capacity", "capacity_runs", "controllers_config", "forecast_config",
                    "outcome_config", "safety_policy", "ranking_policy"):
            self.assertTrue((rce.REPO_ROOT / cfg2[key]).is_file(), key)
        with self.assertRaises(ValueError):   # D runs on 4C only
            rce.setup(cfg2, PlannedEpisode(2, "x", "full", "d1", 9801), 240.0, 45.0,
                      Path("/tmp/ep"))

    def test_4c_ports_from_config(self) -> None:
        self.assertEqual(rce.controller_ports(CFG), rce.CONTROLLER_PORTS)

    def test_archive_moves_and_gzips(self) -> None:
        import gzip
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root, out, eid = Path(tmp) / "repo", Path(tmp) / "ep", "B-full-burst-s9801"
            for rel, text in ((f"data/snapshots/{eid}.jsonl", "snap"),
                              (f"data/raw/forecast/{eid}.jsonl", "fc"),
                              (f"data/raw/{eid}/switches.jsonl", "sw"),
                              ("data/decision_runs/dec-1/record.json", "{}"),
                              ("data/decision_runs/dec-1/extra.json", "x"),
                              ("data/snapshots/other.jsonl", "keep")):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            (out / "logs").mkdir(parents=True)
            (out / "logs" / "c1.log").write_text("line\n" * 100)
            moved = rce.archive_episode(root, out, eid,
                                        [root / "data/decision_runs/dec-1/record.json"])
            self.assertEqual(moved, [out / "decisions/dec-1/record.json"])
            self.assertEqual((out / "snapshots.jsonl").read_text(), "snap")
            self.assertEqual((out / "forecast.jsonl").read_text(), "fc")
            self.assertTrue((out / "decisions/dec-1/extra.json").is_file())
            self.assertFalse((root / f"data/snapshots/{eid}.jsonl").exists())   # moved
            self.assertFalse((root / "data/decision_runs/dec-1").exists())
            self.assertTrue((root / "data/snapshots/other.jsonl").exists())     # untouched
            self.assertFalse((out / "logs/c1.log").exists())
            with gzip.open(out / "logs/c1.log.gz", "rt") as handle:
                self.assertEqual(handle.read(), "line\n" * 100)
            with gzip.open(out / "raw/switches.jsonl.gz", "rt") as handle:
                self.assertEqual(handle.read(), "sw")

    def test_signal_controller_without_process(self) -> None:
        self.assertEqual(rce.signal_controller(None, "STOP"), [])

    def test_duration_must_cover_warmup_and_post_window(self) -> None:
        with self.assertRaises(ValueError):
            rce.setup(CFG, EP, 70.0, 45.0, Path("/tmp/ep"))


class PlanAndRecordsTest(unittest.TestCase):
    def test_loop_phases_dry_run_until_the_warmup_ends(self) -> None:
        self.assertEqual(rce.loop_phases(30.0, 195.0, 2.0), (28.0, 165.0))
        self.assertEqual(rce.loop_phases(30.0, 195.0, 40.0), (0.0, 155.0))
        self.assertEqual(rce.loop_phases(30.0, 195.0, 0.0), (30.0, 165.0))
        with self.assertRaises(ValueError):
            rce.loop_phases(30.0, 30.0, 0.0)
        with self.assertRaises(ValueError):
            rce.loop_phases(30.0, 195.0, -1.0)

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

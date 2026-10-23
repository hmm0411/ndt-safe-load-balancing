"""Unit tests of the Week 11 migration runner (no Mininet, no orchestrator)."""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from src.experiments.migration import runner
from src.experiments.migration.config import AttemptPlan
from src.experiments.migration.schema import MigrationTransactionView
from src.experiments.migration.writer import raw_transaction


def raw_state(snapshot_id="snap-1", *, valid=True, owner=None, version=1):
    """A realistic raw GET /api/v1/twin/state payload: s1-s10 on c1, s11-s20 on c2."""
    owner = owner or {}
    ownership = []
    for n in range(1, 21):
        switch = f"s{n}"
        base = "c1" if n <= 10 else "c2"
        ownership.append({
            "switch_id": switch, "owner_controller_id": owner.get(switch, base),
            "source_role": "MASTER", "target_role": "SLAVE",
            "generation_id": 1, "ownership_version": version,
        })
    controllers = []
    for cid in ("c1", "c2"):
        controllers.append({
            "controller_id": cid, "processed_packet_in_rate": 100.0, "safe_capacity_pps": 1000.0,
            "utilization": 0.45, "collection_latency_ms": 5.0, "flow_mod_rate": 1.0,
            "process_cpu_percent": 10.0, "process_memory_rss_mb": 100.0, "response_p95_ms": 3.0,
            "managed_switch_count": sum(1 for o in ownership if o["owner_controller_id"] == cid),
        })
    switches = [{
        "switch_id": o["switch_id"], "controller_id": o["owner_controller_id"],
        "observed_at": "2026-01-01T00:00:00Z", "packet_in_total": 10, "packet_in_rate": 5.0,
        "processed_packet_in_total": 10, "processed_packet_in_rate": 5.0,
        "flow_mod_total": 1, "flow_mod_rate": 0.5, "control_load_share": 0.1,
    } for o in ownership]
    return {
        "snapshot_id": snapshot_id, "created_at": f"t-{snapshot_id}", "ownership_version": version,
        "controllers": controllers, "switches": switches, "ownership": ownership,
        "quality": {"valid": valid, "consistent": True, "consistency_score": 1.0,
                    "quality_score": 1.0, "completeness_ratio": 1.0, "age_of_twin_ms": 100,
                    "synchronization_jitter_ms": 5.0, "twinning_rate": 1.0},
    }


def make_plan(**kw):
    values: dict[str, Any] = dict(
        attempt_id="mig-stable-early-dst45-c1-c2-r01", action_type="MIGRATE", workload="stable",
        phase="early", destination_load_level=0.45, source_controller="c1", target_controller="c2",
        repeat="r01", seed=9201, split="train")
    values.update(kw)
    return AttemptPlan(**values)


def sample(index, raw, valid=True):
    key, view, ok, error = runner.parse_state(raw)
    return runner.Sample(index, "now", raw, key, view, valid and ok, error)


class TinyClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ReadPhaseTest(unittest.TestCase):
    def test_last_complete_row_wins_and_partial_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast_samples.jsonl"
            path.write_text(
                json.dumps({"workload_phase": "early"}) + "\n"
                + json.dumps({"workload_phase": "middle"}) + "\n"
                + '{"workload_phase": "la', encoding="utf-8")
            self.assertEqual(runner.read_last_phase(path), "middle")

    def test_missing_file(self):
        self.assertIsNone(runner.read_last_phase(Path("/nonexistent/forecast_samples.jsonl")))


class WaitForTest(unittest.TestCase):
    def test_returns_value_then_none_on_timeout(self):
        t = TinyClock()
        self.assertEqual(runner.wait_for(lambda: "x", 5, sleep=t.sleep, clock=t.clock), "x")
        self.assertIsNone(runner.wait_for(lambda: None, 1, sleep=t.sleep, clock=t.clock))

    def test_stops_when_not_alive(self):
        t = TinyClock()
        self.assertIsNone(runner.wait_for(lambda: None, 100, sleep=t.sleep, clock=t.clock,
                                          alive=lambda: False))


class ParseStateTest(unittest.TestCase):
    def test_valid(self):
        key, view, valid, error = runner.parse_state(raw_state("a"))
        self.assertTrue(valid)
        self.assertIsNone(error)
        assert view is not None
        self.assertEqual(view.owner_of("s1"), "c1")

    def test_invalid_quality(self):
        self.assertFalse(runner.parse_state(raw_state("a", valid=False))[2])

    def test_not_ready(self):
        key, view, valid, error = runner.parse_state({"status": "NOT_READY"})
        self.assertFalse(valid)
        self.assertIsNone(view)
        self.assertIn("ValueError", error or "")


class PollerTest(unittest.TestCase):
    def test_same_snapshot_is_not_added_twice(self):
        payloads = iter([raw_state("a"), raw_state("a"), raw_state("b")])
        poller = runner.StatePoller(lambda: next(payloads), lambda: {"status": "OK"})
        for _ in range(3):
            poller.poll_once()
        self.assertEqual([(s.snapshot_id or "").split("@")[0] for s in poller.samples], ["a", "b"])
        self.assertEqual([s.index for s in poller.samples], [0, 1])
        self.assertEqual(len(poller.forecasts), 3)

    def test_network_error_is_recorded_not_raised(self):
        def boom():
            raise OSError("down")
        poller = runner.StatePoller(boom)
        poller.poll_once()
        self.assertFalse(poller.samples[0].valid)
        self.assertIn("OSError", poller.samples[0].error or "")


class TrailingWindowTest(unittest.TestCase):
    def setUp(self):
        self.samples = [sample(i, raw_state(f"s{i}")) for i in range(6)]

    def test_enough_valid(self):
        window = runner.trailing_window(self.samples, 0, 5)
        assert window is not None
        self.assertEqual([s.index for s in window], [1, 2, 3, 4, 5])

    def test_start_index_excludes_older(self):
        self.assertIsNone(runner.trailing_window(self.samples, 3, 5))

    def test_one_invalid_breaks_window(self):
        self.samples[4] = sample(4, raw_state("bad", valid=False))
        self.assertIsNone(runner.trailing_window(self.samples, 0, 5))

    def test_accept_filter(self):
        self.assertIsNone(runner.trailing_window(self.samples, 0, 5, lambda v: v.owner_of("s1") == "c2"))


class SwitchChoiceTest(unittest.TestCase):
    def test_random_choice_is_reproducible_and_owned(self):
        plan = make_plan()
        ownership = raw_state()["ownership"]
        first = runner.pick_switch(plan, ownership)
        self.assertEqual(first, runner.pick_switch(plan, ownership))
        self.assertIn(first[0], {f"s{n}" for n in range(1, 11)})
        self.assertEqual(first[1:], ("seeded_random", False))

    def test_target_hot_only_for_hot_switch(self):
        self.assertFalse(runner.is_target_hot(make_plan()))

    def test_hot_switch_cells_alternate(self):
        flags = [runner.is_target_hot(make_plan(
            workload="hot_switch", phase="high", repeat=f"r0{n}")) for n in range(1, 6)]
        self.assertIn(sum(flags), (2, 3))
        self.assertTrue(all(a != b for a, b in zip(flags, flags[1:])))

    def test_hot_switch_plan_picks_hot_host(self):
        plan = make_plan(workload="hot_switch", phase="high")
        for repeat in ("r01", "r02"):
            plan2 = make_plan(workload="hot_switch", phase="high", repeat=repeat)
            switch, rule, hot = runner.pick_switch(plan2, raw_state()["ownership"])
            if hot:
                self.assertEqual(switch, "s1")
                self.assertEqual(rule, "hot_switch_alternating")
            else:
                self.assertEqual(rule, "seeded_random")
        self.assertEqual(plan.source_controller, "c1")


class PreconditionTest(unittest.TestCase):
    def check(self, raw, switch="s3", **kw):
        args = dict(workload_alive=True, capacity_exists=True)
        args.update(kw)
        return runner.check_preconditions(raw, make_plan(), switch, **args)

    def test_ok(self):
        self.assertEqual(self.check(raw_state()), [])

    def test_invalid_twin(self):
        self.assertIn("twin_invalid", self.check(raw_state(valid=False)))

    def test_switch_not_on_source(self):
        self.assertIn("switch_not_owned_by_source", self.check(raw_state(), switch="s15"))

    def test_role_mismatch(self):
        raw = raw_state()
        raw["ownership"][2]["target_role"] = "MASTER"
        self.assertIn("role_mismatch", self.check(raw))

    def test_missing_switches(self):
        raw = raw_state()
        raw["ownership"].pop()
        self.assertTrue(any(r.startswith("switches_not_connected") for r in self.check(raw)))

    def test_dead_workload_and_capacity(self):
        reasons = self.check(raw_state(), workload_alive=False, capacity_exists=False)
        self.assertIn("workload_not_running", reasons)
        self.assertIn("capacity_missing", reasons)


class SendMigrationTest(unittest.TestCase):
    def send(self, status, body):
        with mock.patch.object(runner, "http_post_json", return_value=(status, body)):
            return runner.send_migration("http://x", "s3", "c2", "none")

    def test_200_uses_response(self):
        with mock.patch.object(MigrationTransactionView, "from_response",
                               return_value="VIEW") as parse:
            view, raw = self.send(200, {"status": "COMMITTED"})
        parse.assert_called_once_with({"status": "COMMITTED"})
        self.assertEqual(view, "VIEW")
        self.assertEqual(raw, raw_transaction(200, {"status": "COMMITTED"}))

    def test_500_is_failed(self):
        view, raw = self.send(500, {"detail": "boom"})
        assert view is not None
        self.assertEqual(view.status, "FAILED")
        self.assertEqual(raw, raw_transaction(500, {"detail": "boom"}))

    def test_409_is_skipped(self):
        view, raw = self.send(409, {"detail": "busy"})
        assert view is not None
        self.assertEqual(view.status, "SKIPPED")
        self.assertIn("http_409", view.skip_reason or "")


class WorkloadCommandTest(unittest.TestCase):
    def settings(self):
        root = Path("/repo")
        return runner.RunnerSettings(
            repo_root=root, output_root=root / "out", work_root=root / "work",
            config_path=root / "c.yaml", controllers_config="c.yaml",
            capacity_json=root / "cap.json", capacity_runs=root / "runs.csv", python="python3")

    def test_source_workload_destination_constant(self):
        cmd = runner.build_workload_command(make_plan(), self.settings(), Path("/w"), "rid", 60.0)
        self.assertEqual(cmd[:3], ["sudo", "-E", "env"])
        self.assertIn("--forecast-stable", cmd)
        self.assertIn("c1=stable", cmd)
        self.assertIn("c2=constant:0.45", cmd)
        self.assertEqual(cmd[cmd.index("--run-id") + 1], "rid")
        self.assertNotIn("NDT_HOT_SWITCH_LEVEL=early", cmd)

    def test_hot_switch_level_and_name(self):
        plan = make_plan(workload="hot_switch", phase="high", source_controller="c2",
                         target_controller="c1", destination_load_level=0.75)
        cmd = runner.build_workload_command(plan, self.settings(), Path("/w"), "rid", 60.0)
        self.assertIn("NDT_HOT_SWITCH_LEVEL=high", cmd)
        self.assertIn("c2=hot-switch", cmd)
        self.assertIn("c1=constant:0.75", cmd)
        self.assertIn("--forecast-hot-switch", cmd)
        self.assertIn("NDT_HOT_SWITCH_CHECK_LOAD=0", cmd)

    def test_hot_switch_no_op_keeps_the_load_check(self):
        plan = make_plan(workload="hot_switch", phase="high", action_type="NO_OP")
        cmd = runner.build_workload_command(plan, self.settings(), Path("/w"), "rid", 60.0)
        self.assertIn("NDT_HOT_SWITCH_LEVEL=high", cmd)
        self.assertNotIn("NDT_HOT_SWITCH_CHECK_LOAD=0", cmd)


class AttemptNameTest(unittest.TestCase):
    def test_failure_suffix(self):
        plan = make_plan()
        self.assertEqual(runner.attempt_dir_name(plan, "none"), plan.attempt_id)
        self.assertEqual(runner.attempt_dir_name(plan, "flow_mod"), plan.attempt_id + "-fail-flow_mod")


class PreflightTest(unittest.TestCase):
    def settings(self, cap_exists=True):
        tmp = Path(tempfile.mkdtemp())
        cap = tmp / "cap.json"
        if cap_exists:
            cap.write_text("{}")
        return runner.RunnerSettings(
            repo_root=tmp, output_root=tmp, work_root=tmp, config_path=tmp / "c", controllers_config="x",
            capacity_json=cap, capacity_runs=tmp / "r.csv")

    def test_ok(self):
        runner.preflight(self.settings(), port_open=lambda p: p in (6653, 6654), list_bridges=lambda: "")

    def test_reports_every_problem(self):
        with self.assertRaises(runner.RunnerError) as ctx:
            runner.preflight(self.settings(False), port_open=lambda p: p == 9000,
                             list_bridges=lambda: "s1\ns2\n")
        text = str(ctx.exception)
        for part in ("6653", "6654", "9000", "capacity", "leftover OVS"):
            self.assertIn(part, text)


class ResumeRuleTest(unittest.TestCase):
    def settings(self, root):
        return runner.RunnerSettings(
            repo_root=root, output_root=root / "out", work_root=root / "work",
            config_path=root / "c.yaml", controllers_config="x",
            capacity_json=root / "cap.json", capacity_runs=root / "runs.csv")

    def test_attempt_with_validation_json_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            done = settings.output_root / make_plan().attempt_id
            done.mkdir(parents=True)
            (done / "validation.json").write_text("{}")
            self.assertEqual(runner.run_attempt(make_plan(), settings).status, "exists")

    def test_half_written_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            (settings.output_root / make_plan().attempt_id).mkdir(parents=True)
            with self.assertRaises(runner.RunnerError):
                runner.run_attempt(make_plan(), settings)


class CleanDirTest(unittest.TestCase):
    def test_removes_a_directory_with_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "work"
            (target / "workload").mkdir(parents=True)
            (target / "workload" / "forecast_samples.jsonl").write_text("{}\n")
            runner._clean_dir(target)
            self.assertFalse(target.exists())

    def test_missing_directory_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner._clean_dir(Path(tmp) / "nothing")


class FakeWorld:
    """Scripted twin: the snapshot list grows one entry per sleep."""

    def __init__(self, plan, *, tx_status="COMMITTED", phase="early", switch="s3"):
        self.plan, self.t = plan, TinyClock()
        self.samples, self.forecasts = [], []
        self.phase, self.tx_status, self.switch = phase, tx_status, switch
        self.owner = {}
        self.sent = []
        self.counter = 0
        self.add(3)

    def add(self, n):
        for _ in range(n):
            self.counter += 1
            raw = raw_state(f"s{self.counter}", owner=dict(self.owner))
            self.samples.append(sample(len(self.samples), raw))

    def sleep(self, seconds):
        self.t.sleep(seconds)
        if int(self.t.now * 5) % 5 == 0:
            self.add(1)

    def send(self, switch, target, failure):
        self.sent.append((switch, target, failure))
        if self.tx_status == "COMMITTED":
            self.owner[switch] = target
        view = types.SimpleNamespace(status=self.tx_status)
        return view, {"status": self.tx_status}

    def context(self):
        return runner.Context(
            samples=self.samples, forecasts=self.forecasts, read_phase=lambda: self.phase,
            send=self.send, workload_alive=lambda: True, capacity_exists=True,
            sleep=self.sleep, clock=self.t.clock)


class CollectAttemptTest(unittest.TestCase):
    def collect(self, world, plan=None, **kw):
        plan = plan or world.plan
        return runner.collect_attempt(plan, world.context(), failure="none", duration=60,
                                      window=5, post_timeout=60, **kw)

    def test_committed_waits_for_target_owner(self):
        world = FakeWorld(make_plan())
        got = self.collect(world)
        self.assertEqual(got.transaction.status, "COMMITTED")
        self.assertEqual(len(got.before), 5)
        self.assertEqual(len(got.after), 5)
        self.assertTrue(got.post_complete)
        self.assertTrue(all(s.view.owner_of(got.action.switch_id) == "c2" for s in got.after))
        self.assertTrue(all(s.index < got.after[0].index for s in got.before))
        self.assertEqual(world.sent[0][1:], ("c2", "none"))

    def test_restored_waits_for_source_owner(self):
        world = FakeWorld(make_plan(), tx_status="RESTORED")
        got = self.collect(world)
        self.assertEqual(got.transaction.status, "RESTORED")
        self.assertTrue(all(s.view.owner_of(got.action.switch_id) == "c1" for s in got.after))

    def test_no_op_sends_nothing(self):
        plan = make_plan(action_type="NO_OP", attempt_id="noop-stable-early-dst45-c1-c2-r01")
        world = FakeWorld(plan)
        got = self.collect(world)
        self.assertEqual(world.sent, [])
        self.assertIsNone(got.transaction)
        self.assertEqual(got.transaction_raw, raw_transaction())
        self.assertEqual(len(got.after), 5)

    def test_phase_never_reached_is_skipped(self):
        world = FakeWorld(make_plan(), phase="late")
        got = self.collect(world)
        self.assertEqual(got.transaction.status, "SKIPPED")
        self.assertEqual(world.sent, [])
        self.assertIn("phase_not_reached:early", got.skip_reasons)

    def test_never_ready_raises(self):
        world = FakeWorld(make_plan())
        world.samples.clear()
        ctx = world.context()
        ctx.workload_alive = lambda: False
        with self.assertRaises(runner.RunnerError):
            runner.collect_attempt(world.plan, ctx, failure="none", duration=1,
                                   window=5, post_timeout=1)


class VersionTest(unittest.TestCase):
    def test_quality_version_hashes_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/twin").mkdir(parents=True)
            (root / "src/twin/state_quality.py").write_text("a = 1\n")
            first = runner.quality_version(root)
            (root / "src/twin/state_quality.py").write_text("a = 2\n")
            self.assertNotEqual(first, runner.quality_version(root))

    def test_quality_version_without_match_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runner.RunnerError):
                runner.quality_version(Path(tmp))


if __name__ == "__main__":
    unittest.main()

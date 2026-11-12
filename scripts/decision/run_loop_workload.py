"""Start the orchestrator and ONE workload for the closed loop (Week 13, Day 6; VM only).

The migration runner can only hold the cells of its collection matrix; the closed-loop
scenarios need others (destination at 0.95 without a scripted migration, hot-switch at
peak). This launcher reuses the runner's preflight, process handling and workload
command, and collects nothing: decisions are recorded by the orchestrator itself
(data/decision_runs). Run as the normal user; the workload calls sudo itself.

  python3 scripts/decision/run_loop_workload.py --workload gradual --dst-level 0.45
  python3 scripts/decision/run_loop_workload.py --workload hot_switch --level peak \
      --dst-level 0.95 --duration 150
  python3 scripts/decision/run_loop_workload.py --workload stable --source-level 1.05 \
      --dst-level 0.45 --duration 180      # sustained overload of the source

The forecast workloads were built to keep U below 1 (gradual tops at a 0.95 target,
0.91 measured), so they never reach the risk threshold U + m_f >= 1; --source-level
holds the source at a constant utilization instead (stable workload only).

In a second terminal, once this prints "workload started":
  python3 scripts/decision/run_closed_loop.py --scenario gradual --mode dry_run --duration 110
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.migration.config import AttemptPlan  # noqa: E402
from src.experiments.migration.runner import (  # noqa: E402
    ORCH_START_TIMEOUT_S,
    WORKLOAD_EXIT_MARGIN_S,
    WORKLOAD_FLAGS,
    ProcessGroup,
    RunnerError,
    RunnerSettings,
    _clean_dir,
    _healthy,
    build_workload_command,
    preflight,
    wait_for,
)

WORK_ROOT = "data/experiment_runs/closed-loop-work"
# Closed-loop seed: not 101-105 (forecast), 9201-9205 (raw-v2), 9301 (stress-v1),
# 9401 (calibration-v1) nor 9501 (stress-v2).
DEFAULT_SEED = 9601
HOT_SWITCH_LEVELS = ("moderate", "high", "peak")


def make_plan(workload: str, level: str, dst_level: float, source: str, target: str,
              seed: int) -> AttemptPlan:
    if workload not in WORKLOAD_FLAGS:
        raise ValueError(f"workload must be one of {sorted(WORKLOAD_FLAGS)}")
    if workload == "hot_switch" and level not in HOT_SWITCH_LEVELS:
        raise ValueError(f"hot_switch needs --level in {HOT_SWITCH_LEVELS}")
    if not 0.0 <= dst_level <= 1.5:
        raise ValueError("dst-level must be within [0, 1.5]")
    if source == target:
        raise ValueError("source and target must differ")
    name = f"loop-{workload}-{level}-dst{int(round(dst_level * 100)):02d}-{source}-{target}-s{seed}"
    # action_type MIGRATE: the loop may move the hot switch, so the workload's own
    # "hot switch still on the source" check is turned off (runner.build_workload_command).
    return AttemptPlan(name, "MIGRATE", workload, level, dst_level, source, target, "loop", seed,
                       "closed_loop")


def source_profile(cmd: list[str], plan: AttemptPlan, source_level: float | None) -> list[str]:
    """Replace the source profile of the workload command by constant:<U> if asked."""
    if source_level is None:
        return cmd
    if plan.workload != "stable":
        raise ValueError("--source-level needs --workload stable (constant source load)")
    if not 0.0 <= source_level <= 1.5:
        raise ValueError("source-level must be within [0, 1.5]")
    old = f"{plan.source_controller}={plan.workload}"
    if cmd.count(old) != 1:
        raise ValueError(f"cannot find the source profile {old!r} in the workload command")
    return [f"{plan.source_controller}=constant:{source_level}" if x == old else x for x in cmd]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOAD_FLAGS))
    ap.add_argument("--level", default="run", help="hot_switch: moderate | high | peak")
    ap.add_argument("--dst-level", type=float, required=True,
                    help="constant utilization of the target controller, e.g. 0.45 or 0.95")
    ap.add_argument("--source-level", type=float, default=None,
                    help="hold the source at this constant utilization (stable workload only)")
    ap.add_argument("--source", default="c1")
    ap.add_argument("--target", default="c2")
    ap.add_argument("--duration", type=float, default=120.0, help="workload seconds")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--controllers-config", default="configs/controllers_2c20s.yaml")
    ap.add_argument("--orchestrator-url", default="http://127.0.0.1:9000")
    args = ap.parse_args(argv)
    if os.geteuid() == 0:
        print("run this as the normal user (the workload calls sudo itself)", file=sys.stderr)
        return 2
    os.chdir(REPO_ROOT)
    work_dir = REPO_ROOT / WORK_ROOT / "pending"
    settings = RunnerSettings(
        repo_root=REPO_ROOT, output_root=work_dir, work_root=REPO_ROOT / WORK_ROOT,
        config_path=REPO_ROOT / args.controllers_config,
        controllers_config=args.controllers_config,
        capacity_json=REPO_ROOT / "data/benchmarks/controller_capacity.json",
        capacity_runs=REPO_ROOT / "data/benchmarks/capacity_runs.csv",
        orchestrator_url=args.orchestrator_url,
    )
    try:
        plan = make_plan(args.workload, args.level, args.dst_level, args.source, args.target,
                         args.seed)
        tag = "" if args.source_level is None else f"-src{int(round(args.source_level * 100)):03d}"
        work_dir = REPO_ROOT / WORK_ROOT / (plan.attempt_id + tag)
        command = source_profile(
            build_workload_command(plan, settings, work_dir / "workload", plan.attempt_id,
                                   args.duration), plan, args.source_level)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    url = args.orchestrator_url.rstrip("/")
    orchestrator = workload = None
    try:
        preflight(settings)
        _clean_dir(work_dir)
        work_dir.mkdir(parents=True)
        orchestrator = ProcessGroup(
            ["./scripts/start_orchestrator.sh"], cwd=REPO_ROOT,
            env={**os.environ, "NDT_CONTROLLERS_CONFIG": args.controllers_config,
                 "RUN_ID": plan.attempt_id},
            log_path=work_dir / "orchestrator.log")
        workload = ProcessGroup(
            command, cwd=REPO_ROOT, env=dict(os.environ), log_path=work_dir / "workload.log",
            use_sudo_to_signal=True)
        print(f"[{plan.attempt_id}] starting orchestrator", flush=True)
        orchestrator.start()
        if not wait_for(lambda: _healthy(url), ORCH_START_TIMEOUT_S, alive=orchestrator.alive):
            raise RunnerError("orchestrator did not answer /health (see orchestrator.log)")
        workload.start()
        print(f"[{plan.attempt_id}] workload started ({args.duration:.0f} s), logs in {work_dir}",
              flush=True)
        code = workload.wait(args.duration + WORKLOAD_EXIT_MARGIN_S)
        print(f"[{plan.attempt_id}] workload exit code {code}", flush=True)
        return 0 if code == 0 else 1
    except RunnerError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted, stopping workload and orchestrator", file=sys.stderr)
        return 130
    finally:
        if workload is not None:
            workload.stop()
        if orchestrator is not None:
            orchestrator.stop()


if __name__ == "__main__":
    sys.exit(main())

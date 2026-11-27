#!/usr/bin/env python3
"""One closed-loop campaign episode on 4C40S (protocol v1; VM only, run as the normal user).

Every episode starts from scratch (protocol 5): fresh controllers c1-c4 (empty flow tables),
a fresh orchestrator (ownership, twin, history, margins), a fresh Mininet network. Then:

  1. orchestrator with the episode's method (NDT_DECISION_METHOD) and seed;
  2. runtime_4c40s.py --forecast <workload>: c1 runs the workload scaled by
     NDT_WORKLOAD_SCALE (c1=1.2, decided 26/11), c2-c4 constant 0.60;
  3. once the workload has started, one decision request per second until
     duration - post_window seconds after the workload start (the last migrations still
     get their observed post window inside the workload): dry-run decisions (recorded,
     never executed) during the warm-up, execute after it (decided 26/11, My: every method
     then starts from the initial mapping at the end of the warm-up);
  4. everything is stopped, and the episode keeps: meta.json, the orchestrator's snapshots
     and forecasts of the run, its decision records (scenario label = episode id), logs.

  python3 scripts/experiments/run_campaign_episode.py --plan results/campaign/plan_B.csv \
      --episode-id B-full-burst-s9801 --duration 240
  python3 scripts/experiments/run_campaign_episode.py --pilot full burst 9801 --duration 240

Ports 6653-6656, 8081-8084 and 9000 must be free (no other orchestrator or controller).
Never runs mn -c; leftover links are removed by run_campaign_queue.sh between episodes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import gzip
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decision.run_closed_loop import http_post, run as run_loop  # noqa: E402
from src.experiments.final_evaluation.campaign import PlannedEpisode  # noqa: E402
from src.experiments.final_evaluation.fault_runner import FaultRunner, http_json  # noqa: E402
from src.experiments.final_evaluation.faults import (  # noqa: E402
    FaultSpec,
    check_episode,
    fault_spec,
)

CONFIG = REPO_ROOT / "configs/experiments/campaign_4c_v1.yaml"
OUT_ROOT = REPO_ROOT / "data/experiment_runs/campaign_4c"
CONTROLLER_PORTS = {"c1": (6653, 8081), "c2": (6654, 8082), "c3": (6655, 8083),
                    "c4": (6656, 8084)}
ORCH_PORT = 9000
TOPOLOGY_MODULES = {"4C40S": "src.experiments.topologies.runtime_4c40s",
                    "2C20S": "src.experiments.topologies.runtime_2c20s"}


def controller_ports(cfg: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """{controller: (OpenFlow port, REST port)} of the campaign's controllers_config."""
    data = yaml.safe_load((REPO_ROOT / str(cfg["controllers_config"])).read_text(
        encoding="utf-8"))
    out = {}
    for cid, c in data["controllers"].items():
        rest = int(str(c["rest_url"]).rstrip("/").rsplit(":", 1)[1])
        out[str(cid)] = (int(c["openflow_port"]), rest)
    return out

START_TIMEOUT_S = 40.0
WORKLOAD_START_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class EpisodeSetup:
    episode: PlannedEpisode
    duration_s: float
    loop_s: float
    workload_scale: str
    other_level: float
    orchestrator_env: dict[str, str]
    topology_cmd: list[str]
    topology_env: dict[str, str]
    fault: FaultSpec | None = None   # D episodes (plan workload d1..d6, d7-uNNN)


def setup(cfg: Mapping[str, Any], episode: PlannedEpisode, duration_s: float,
          post_window_s: float, out_dir: Path, python: str = "python3") -> EpisodeSetup:
    """Commands and environments of one episode; pure (tested without a VM)."""
    if duration_s <= post_window_s + float(cfg["warmup_s"]):
        raise ValueError("duration must exceed warm-up + post window")
    load = cfg["load"]
    scale = f"{load['scaled_controller']}={float(load['scale'])}"
    topology = str(cfg.get("topology", "4C40S"))
    if topology not in TOPOLOGY_MODULES:
        raise ValueError(f"unknown topology {topology!r}")
    others = [c for c in controller_ports(cfg) if c != load["scaled_controller"]]
    is_fault = episode.workload[:1] == "d"
    if is_fault and topology != "4C40S":
        raise ValueError("fault scenarios D run on 4C40S only (protocol 4.D)")
    fault = fault_spec(episode.workload, cfg["faults"]) if is_fault else None
    other_level = float(load["other_level"]) if fault is None else fault.dst_level
    workload = episode.workload if fault is None else "stable"
    profiles = {cid: f"constant:{other_level}" for cid in others}
    if fault is not None:  # D: the scaled controller holds a constant overload
        profiles[load["scaled_controller"]] = f"constant:{fault.c1_level}"
    orch_env = {"RUN_ID": episode.episode_id,
                "NDT_CONTROLLERS_CONFIG": str(cfg["controllers_config"]),
                "NDT_CAPACITY_FILE": str(cfg["capacity"]),
                "NDT_FORECAST_CONFIG": str(cfg["forecast_config"]),
                "NDT_OUTCOME_CONFIG": str(cfg["outcome_config"]),
                "NDT_SAFETY_POLICY": str(cfg["safety_policy"]),
                "NDT_RANKING_POLICY": str(cfg["ranking_policy"]),
                "NDT_DECISION_METHOD": episode.method,
                "NDT_EPISODE_SEED": str(episode.seed)}
    if fault is not None:
        orch_env["NDT_FAULT_INJECTION"] = "1"
    # 4C40S: --forecast <workload>; 2C20S: --forecast-<workload> (its own flags)
    chosen = (["--forecast", workload] if topology == "4C40S"
              else [f"--forecast-{workload.replace('_', '-')}"])
    cmd = ["sudo", "-E", "env", f"PYTHONPATH={REPO_ROOT}", python, "-m",
           TOPOLOGY_MODULES[topology], *chosen,
           "--seed", str(episode.seed), "--run-id", episode.episode_id,
           "--output-dir", str(out_dir / "workload"),
           "--capacity-runs", str(cfg["capacity_runs"]),
           "--controller-capacity", str(cfg["capacity"]),
           "--duration", str(duration_s), "--experiment-type", str(cfg["version"])]
    for experiment in cfg.get("capacity_experiments") or []:
        cmd += ["--capacity-experiment", str(experiment)]
    for cid, profile in profiles.items():
        cmd += ["--profile", f"{cid}={profile}"]
    topo_env = {"NDT_WORKLOAD_SCALE": scale, "NDT_HOT_SWITCH_CHECK_LOAD": "0"}
    return EpisodeSetup(episode, duration_s, duration_s - post_window_s, scale,
                        other_level, orch_env, cmd, topo_env, fault)


def loop_phases(warmup_s: float, loop_s: float, since_start_s: float) -> tuple[float, float]:
    """(dry-run seconds, execute seconds) of a decision loop starting ``since_start_s``
    after the workload start; both phases end at fixed times after that start."""
    if loop_s <= warmup_s:
        raise ValueError("the decision loop must outlast the warm-up")
    if since_start_s < 0:
        raise ValueError("the loop cannot start before the workload")
    return (max(0.0, warmup_s - since_start_s),
            max(0.0, loop_s - max(since_start_s, warmup_s)))


def episode_from_plan(plan: Path, episode_id: str) -> PlannedEpisode:
    with plan.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["episode_id"] == episode_id:
                return PlannedEpisode(int(row["order"]), row["episode_id"], row["method"],
                                      row["workload"], int(row["seed"]))
    raise ValueError(f"{episode_id} is not in {plan}")


def records_of(record_root: Path, episode_id: str, since: datetime) -> list[Path]:
    """Decision records of this episode: scenario label = episode id, decided after the
    episode started (a rerun of the same id never picks up an older run's records)."""
    out = []
    for path in sorted(record_root.glob("*/record.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            decided = datetime.fromisoformat(str(rec.get("decided_at")).replace("Z", "+00:00"))
            if rec.get("scenario") == episode_id and decided >= since:
                out.append(path)
        except (OSError, ValueError):
            continue
    return out


def _gzip(path: Path) -> None:
    with path.open("rb") as src, gzip.open(path.with_name(path.name + ".gz"), "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()


def archive_episode(root: Path, out: Path, episode_id: str, records: list[Path]) -> list[Path]:
    """Move (never copy) what the run left under data/ into the episode, then gzip the
    bulky text nobody reads in the analysis (27/11: copies plus ~117 MB of plain logs per
    episode filled the 97 GB disk after 20 episodes). Kept plain: snapshots.jsonl,
    forecast.jsonl and decisions/*/record.json, which collect reads. Returns the moved
    record paths."""
    moves = {root / "data/snapshots" / f"{episode_id}.jsonl": out / "snapshots.jsonl",
             root / "data/raw/forecast" / f"{episode_id}.jsonl": out / "forecast.jsonl"}
    for src, dst in moves.items():
        if src.is_file():
            shutil.move(str(src), str(dst))
    moved = []
    for path in records:
        dest = out / "decisions" / path.parent.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path.parent), str(dest))
        moved.append(dest / path.name)
    raw = root / "data/raw" / episode_id   # per-run raw telemetry (controllers/switches.jsonl)
    if raw.is_dir():
        shutil.move(str(raw), str(out / "raw"))
    for pattern in ("logs/*.log", "raw/*.jsonl"):
        for path in sorted(out.glob(pattern)):
            _gzip(path)
    return moved


def signal_controller(group: Any, name: str) -> list[int]:
    """SIGSTOP / SIGCONT / SIGKILL to a controller's process group (faults D3, D4, D6)."""
    if group is None or group.proc is None:
        return []
    try:
        os.killpg(group.proc.pid, getattr(signal, f"SIG{name}"))
    except ProcessLookupError:
        return []
    return [int(group.proc.pid)]


def start_faults(spec: FaultSpec, url: str, controllers: Mapping[str, Any],
                 events: list[dict[str, Any]], path: Path,
                 workload_start: float) -> threading.Thread:
    """The D episode's actions in a background thread; every event to ``path`` (jsonl)."""
    lock = threading.Lock()

    def log(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")

    runner = FaultRunner(spec, http_json(url),
                         lambda cid, name: signal_controller(controllers.get(cid), name), log)
    thread = threading.Thread(target=runner.run, args=(workload_start,), daemon=True)
    thread.start()
    return thread


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def preflight() -> None:
    busy = [p for c in CONTROLLER_PORTS.values() for p in c if _port_open(p)]
    if _port_open(ORCH_PORT):
        busy.append(ORCH_PORT)
    if busy:
        raise SystemExit(f"PREFLIGHT FAIL: ports in use {busy} (another orchestrator or "
                         "controller is running; the episode needs all of them)")
    if subprocess.run(["pgrep", "-f", "mininet:"], capture_output=True).returncode == 0:
        raise SystemExit("PREFLIGHT FAIL: a Mininet network is up (someone else's run?)")


def wait_until(predicate: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {what}")


def main(argv: Sequence[str] | None = None) -> int:
    from src.experiments.migration.runner import ProcessGroup, _healthy

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--episode-id")
    ap.add_argument("--pilot", nargs=3, metavar=("METHOD", "WORKLOAD", "SEED"))
    ap.add_argument("--duration", type=float, help="seconds (default: config duration_s)")
    ap.add_argument("--post-window", type=float, default=45.0)
    ap.add_argument("--out-root", type=Path,
                    help="default: the config's out_root, else data/experiment_runs/campaign_4c")
    ap.add_argument("--dry-run", action="store_true", help="print the setup, start nothing")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.pilot:
        m, w, s = args.pilot
        episode = PlannedEpisode(0, f"pilot-{m}-{w.replace('_', '-')}-s{s}", m, w, int(s))
    elif args.plan and args.episode_id:
        episode = episode_from_plan(args.plan, args.episode_id)
    else:
        ap.error("--plan with --episode-id, or --pilot METHOD WORKLOAD SEED")
    duration = args.duration or cfg.get("duration_s")
    if not duration:
        ap.error("no duration: pass --duration (the pilot fixes duration_s in the config)")
    out_root = args.out_root or (REPO_ROOT / str(cfg["out_root"]) if cfg.get("out_root")
                                 else OUT_ROOT)
    out = out_root / episode.episode_id
    st = setup(cfg, episode, float(duration), args.post_window, out, sys.executable)
    if args.dry_run:
        print(json.dumps(asdict(st), indent=2, default=str))
        return 0
    if os.geteuid() == 0:
        ap.error("run as the normal user (the topology calls sudo itself)")
    if out.exists():
        print(f"SKIP: {out} exists (never overwritten)")
        return 3
    os.chdir(REPO_ROOT)
    preflight()
    (out / "logs").mkdir(parents=True)
    base_env = dict(os.environ)
    procs: list[Any] = []
    status = "FAILED"
    started_at: str | None = None
    summary: dict[str, Any] = {}
    warmup_summary: dict[str, Any] = {}
    controllers: dict[str, Any] = {}
    fault_events: list[dict[str, Any]] = []
    fault_thread: threading.Thread | None = None
    load_before = os.getloadavg()
    episode_start = datetime.now(timezone.utc)
    try:
        ports = controller_ports(cfg)
        for cid in ports:
            p = ProcessGroup([f"./scripts/start_{cid}.sh"], cwd=REPO_ROOT, env=base_env,
                             log_path=out / "logs" / f"{cid}.log")
            p.start()
            procs.append(p)
            controllers[cid] = p
        wait_until(lambda: all(_port_open(rest) for _, rest in ports.values()),
                   START_TIMEOUT_S, "controllers")
        orch = ProcessGroup(["./scripts/start_orchestrator.sh"], cwd=REPO_ROOT,
                            env={**base_env, **st.orchestrator_env},
                            log_path=out / "logs" / "orchestrator.log")
        orch.start()
        procs.append(orch)
        url = f"http://127.0.0.1:{ORCH_PORT}"
        wait_until(lambda: _healthy(url), START_TIMEOUT_S, "orchestrator /health")
        topo = ProcessGroup(st.topology_cmd, cwd=REPO_ROOT, env={**base_env, **st.topology_env},
                            log_path=out / "logs" / "topology.log", use_sudo_to_signal=True)
        topo.start()
        procs.append(topo)
        meta_path = out / "workload" / episode.episode_id / "metadata.json"
        wait_until(lambda: meta_path.is_file() or not topo.alive(), WORKLOAD_START_TIMEOUT_S,
                   "workload start")
        if not meta_path.is_file():
            raise RuntimeError("topology exited before the workload started (topology.log)")
        started_at = json.loads(meta_path.read_text(encoding="utf-8"))["started_at"]
        print(f"[{episode.episode_id}] workload started {started_at}; loop {st.loop_s:.0f} s",
              flush=True)
        since = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds()
        dry_s, execute_s = loop_phases(float(cfg["warmup_s"]), st.loop_s, max(0.0, since))
        if st.fault is not None:
            fault_thread = start_faults(st.fault, url, controllers, fault_events,
                                        out / "faults.jsonl", time.monotonic() - since)
        if dry_s > 0:
            warmup_summary = run_loop(http_post(url), episode.episode_id, "dry_run", dry_s, 1.0)
        summary = run_loop(http_post(url), episode.episode_id, "execute", execute_s, 1.0)
        code = topo.wait(st.duration_s + 120.0)
        if code != 0:
            raise RuntimeError(f"topology exit code {code} (topology.log)")
        status = "OK"
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f"!! {episode.episode_id}: {exc}", flush=True)
    finally:
        if fault_thread is not None:
            fault_thread.join(timeout=30.0)
        for p in controllers.values():  # a controller left stopped (SIGSTOP) cannot exit
            signal_controller(p, "CONT")
        for p in reversed(procs):
            p.stop()
    records = archive_episode(REPO_ROOT, out, episode.episode_id,
                              records_of(REPO_ROOT / "data/decision_runs", episode.episode_id,
                                         episode_start))
    fault_check = None
    if st.fault is not None and started_at is not None:
        fault_check = check_episode(
            st.fault, fault_events, [json.loads(p.read_text(encoding="utf-8")) for p in records],
            datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=REPO_ROOT).stdout.strip()
    meta = {**asdict(episode), "status": status, "workload_started_at": started_at,
            "duration_s": st.duration_s, "loop_s": st.loop_s,
            "workload_scale": st.workload_scale, "other_level": st.other_level,
            "config": str(args.config), "config_version": cfg["version"], "git_commit": git,
            "decisions_copied": len(records), "loop_summary": summary,
            "warmup_summary": warmup_summary,
            "fault": None if st.fault is None else asdict(st.fault),
            "fault_check": fault_check,
            "load_before": load_before, "load_after": os.getloadavg(),
            "episode_started_at": episode_start.isoformat(),
            "wall_s": round((datetime.now(timezone.utc) - episode_start).total_seconds(), 1),
            "finished_at": datetime.now(timezone.utc).isoformat()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[{episode.episode_id}] {status}: {len(records)} decision records -> {out}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

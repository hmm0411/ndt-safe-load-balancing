"""Scripted-migration runner (Week 11, Day 2).

One attempt = one fresh workload run (like Week 10): orchestrator restart,
Mininet + workload, wait for the planned phase, 5 valid pre snapshots, one
POST /api/v1/migrations (MIGRATE) or nothing (NO_OP), 5 valid post snapshots,
outcome, attempt directory.

The attempt directory follows the runner/validator contract of the Week 11
plan. Nothing here predicts anything (Week 12).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Sequence
from urllib.parse import urlparse

from .config import AttemptPlan, config_sha256
from .metrics import compute_outcome, summarize_quality
from .schema import (
    AttemptMetadata,
    MigrationAction,
    MigrationTransactionView,
    StateSnapshotView,
)
from .selection import select_switch
from .writer import raw_transaction, transaction_from_raw, write_attempt

EXPECTED_SWITCHES = 20
OPENFLOW_PORTS = (6653, 6654)
POLL_INTERVAL_S = 1.0

ORCH_START_TIMEOUT_S = 40.0
READY_TIMEOUT_S = 150.0
PRE_WINDOW_TIMEOUT_S = 40.0
PHASE_TIMEOUT_MARGIN_S = 30.0
MIGRATION_REQUEST_TIMEOUT_S = 120.0
WORKLOAD_EXIT_MARGIN_S = 120.0

WORKLOAD_FLAGS = {
    "stable": "--forecast-stable",
    "gradual": "--forecast-gradual",
    "burst": "--forecast-burst",
    "oscillating": "--forecast-oscillating",
    "hot_switch": "--forecast-hot-switch",
}

# Provisional: each value lets the planned phase start, the 5-snapshot pre
# window, the migration and the post window fit inside the workload. The pilot
# (Day 3) measures the real time and tunes these; override with --duration.
DEFAULT_DURATION_S = {
    "stable": 60.0,
    "gradual": 90.0,
    "burst": 90.0,
    "oscillating": 90.0,
    "hot_switch": 60.0,
}

# Source files whose hash is recorded as quality_version in metadata.json.
QUALITY_SOURCE_GLOBS = ("src/twin/**/*quality*.py",)


class RunnerError(RuntimeError):
    """The attempt cannot produce a usable record (infrastructure or bug)."""


# --------------------------------------------------------------------------
# settings and small helpers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunnerSettings:
    repo_root: Path
    output_root: Path
    work_root: Path
    config_path: Path
    controllers_config: str
    capacity_json: Path
    capacity_runs: Path
    topology: str = "2C20S"
    orchestrator_url: str = "http://127.0.0.1:9000"
    python: str = "python3"  # same interpreter as the Day-1 workload runs under sudo
    required_window: int = 5
    post_window_timeout_s: float = 60.0


@dataclass
class AttemptResult:
    attempt_id: str
    status: str  # "written" | "exists"
    path: Path | None = None
    transaction_status: str | None = None
    training_eligible: bool | None = None
    seconds: float = 0.0
    note: str = ""
    valid: bool | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def attempt_dir_name(plan: AttemptPlan, failure: str) -> str:
    """Directory / attempt id. A failure-injected run gets a suffix so it can
    live next to the clean run of the same matrix cell (smoke rollback)."""
    if failure == "none":
        return plan.attempt_id
    return f"{plan.attempt_id}-fail-{failure}"


def http_get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, body: dict, timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return response.status, (json.loads(text) if text else None)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(text)
        except ValueError:
            return exc.code, text


def wait_for(
    predicate: Callable[[], Any],
    timeout: float,
    *,
    interval: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    alive: Callable[[], bool] | None = None,
) -> Any:
    """Poll until predicate() is truthy; None on timeout or when alive() is False."""
    deadline = clock() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if alive is not None and not alive():
            return None
        if clock() >= deadline:
            return None
        sleep(interval)


# --------------------------------------------------------------------------
# snapshots: poller and windows
# --------------------------------------------------------------------------
@dataclass
class Sample:
    index: int
    observed_at: str
    raw: Any
    snapshot_id: str | None
    view: Any | None
    valid: bool
    error: str | None = None


def parse_state(raw: Any) -> tuple[str | None, Any | None, bool, str | None]:
    """(snapshot_id, view, valid, error) of one GET /api/v1/twin/state payload."""
    snapshot_id = None
    created_at = None
    if isinstance(raw, dict):
        snapshot_id = raw.get("snapshot_id")
        created_at = raw.get("created_at")
    key = None if snapshot_id is None else f"{snapshot_id}@{created_at}"
    try:
        view = StateSnapshotView.from_twin_state(raw)
    except (ValueError, KeyError, TypeError) as exc:
        return key, None, False, f"{type(exc).__name__}: {exc}"
    return key, view, bool(view.quality.valid), None


class StatePoller(threading.Thread):
    """Reads the twin state (and the forecast, reference only) every interval.

    A snapshot already seen (same snapshot_id and created_at) is not added
    twice, so consecutive samples are distinct twin updates.
    """

    def __init__(
        self,
        fetch_state: Callable[[], Any],
        fetch_forecast: Callable[[], Any] | None = None,
        *,
        parse: Callable[[Any], tuple] = parse_state,
        interval: float = POLL_INTERVAL_S,
    ) -> None:
        super().__init__(daemon=True)
        self._fetch_state = fetch_state
        self._fetch_forecast = fetch_forecast
        self._parse = parse
        self._interval = interval
        self._stop_event = threading.Event()
        self.samples: list[Sample] = []
        self.forecasts: list[dict[str, Any]] = []
        self._last_key: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def poll_once(self) -> None:
        stamp = now_iso()
        raw: Any = None
        error: str | None = None
        try:
            raw = self._fetch_state()
        except Exception as exc:  # network error: keep polling, record it
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            key, view, valid, error = self._parse(raw)
        else:
            key, view, valid = None, None, False
        if key is None or key != self._last_key:
            self.samples.append(Sample(len(self.samples), stamp, raw, key, view, valid, error))
            self._last_key = key
        if self._fetch_forecast is not None:
            try:
                self.forecasts.append({"observed_at": stamp, "payload": self._fetch_forecast()})
            except Exception as exc:
                self.forecasts.append({"observed_at": stamp, "error": f"{type(exc).__name__}: {exc}"})

    def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.poll_once()
            self._stop_event.wait(max(0.0, self._interval - (time.monotonic() - started)))


def trailing_window(
    samples: Sequence[Sample],
    start_index: int,
    count: int,
    accept: Callable[[Any], bool] | None = None,
) -> list[Sample] | None:
    """The last `count` samples (index >= start_index) if all are valid and accepted."""
    tail = [s for s in samples if s.index >= start_index]
    if len(tail) < count:
        return None
    window = tail[-count:]
    for item in window:
        if not item.valid:
            return None
        if accept is not None and not accept(item.view):
            return None
    return window


def is_ready_state(view: Any, raw: Any, expected: int = EXPECTED_SWITCHES) -> bool:
    """Twin reports every switch: the testbed and the orchestrator are up."""
    if view is None or not isinstance(raw, dict):
        return False
    managed = sum(int(c.get("managed_switch_count", 0)) for c in raw.get("controllers", []))
    return len(raw.get("ownership", [])) == expected and managed == expected


def read_last_phase(path: Path) -> str | None:
    """workload_phase of the last complete row of the workload's forecast_samples.jsonl."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - 4096))
            tail = handle.read().decode("utf-8", "ignore")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # partial last line
        phase = row.get("workload_phase")
        if phase is not None:
            return str(phase)
    return None


# --------------------------------------------------------------------------
# switch choice and preconditions
# --------------------------------------------------------------------------
def hot_switch_of(source: str) -> str:
    """Switch carrying the hot share when `source` runs the hot-switch workload."""
    from scripts.experiments.run_forecast_hot_switch_2c20s import HOT_HOST

    return f"s{HOT_HOST[source]}"


def is_target_hot(plan: AttemptPlan) -> bool:
    """Half of the hot-switch attempts target the hot switch.

    Within a matrix cell (phase, level, direction) the repeats alternate; a
    hash of the cell sets which parity is hot, so each cell has 2 or 3 of 5.
    """
    if plan.workload != "hot_switch":
        return False
    cell = ":".join(
        (plan.workload, plan.phase, f"{plan.destination_load_level:.2f}",
         plan.source_controller, plan.target_controller)
    )
    offset = int(hashlib.sha256(cell.encode("utf-8")).hexdigest(), 16) % 2
    return (int(plan.repeat[1:]) + offset) % 2 == 0


def pick_switch(plan: AttemptPlan, ownership: Sequence[Any]) -> tuple[str, str, bool]:
    """(switch_id, selection_rule, target_hot), reproducible from the plan."""
    rng = random.Random(f"{plan.seed}:{plan.attempt_id}")
    hot = is_target_hot(plan)
    hot_id = hot_switch_of(plan.source_controller) if hot else None
    switch_id = select_switch(ownership, plan.source_controller, rng, hot_switch_id=hot_id)
    return switch_id, ("hot_switch_alternating" if hot else "seeded_random"), hot


def check_preconditions(
    raw: dict,
    plan: AttemptPlan,
    switch_id: str,
    *,
    workload_alive: bool,
    capacity_exists: bool,
    expected_switches: int = EXPECTED_SWITCHES,
) -> list[str]:
    """Reasons to SKIP the attempt (empty list = go). Never edits any state."""
    reasons: list[str] = []
    quality = raw.get("quality")
    if not isinstance(quality, dict):
        reasons.append("quality_missing")
    elif not quality.get("valid"):
        reasons.append("twin_invalid")
    controllers = {c.get("controller_id"): c for c in raw.get("controllers", [])}
    for controller_id in (plan.source_controller, plan.target_controller):
        if controller_id not in controllers:
            reasons.append(f"controller_missing:{controller_id}")
    managed = sum(int(c.get("managed_switch_count", 0)) for c in controllers.values())
    ownership = raw.get("ownership", [])
    if managed != expected_switches or len(ownership) != expected_switches:
        reasons.append(f"switches_not_connected:{managed}/{len(ownership)}")
    entry = next((o for o in ownership if o.get("switch_id") == switch_id), None)
    if entry is None or entry.get("owner_controller_id") != plan.source_controller:
        reasons.append("switch_not_owned_by_source")
    elif entry.get("source_role") != "MASTER" or entry.get("target_role") != "SLAVE":
        reasons.append("role_mismatch")
    if not capacity_exists:
        reasons.append("capacity_missing")
    if not workload_alive:
        reasons.append("workload_not_running")
    return reasons


def send_migration(
    orchestrator_url: str,
    switch_id: str,
    target_controller: str,
    failure: str,
) -> tuple[MigrationTransactionView | None, dict[str, Any]]:
    """POST /api/v1/migrations -> (transaction view, content of transaction.json)."""
    status, body = http_post_json(
        f"{orchestrator_url.rstrip('/')}/api/v1/migrations",
        {"switch_id": switch_id, "target_controller": target_controller, "simulate_failure": failure},
        timeout=MIGRATION_REQUEST_TIMEOUT_S,
    )
    raw = raw_transaction(status, body)
    return transaction_from_raw(raw), raw


# --------------------------------------------------------------------------
# one attempt: the window protocol (no processes, testable with fakes)
# --------------------------------------------------------------------------
@dataclass
class Context:
    """Everything the protocol needs from the outside world."""

    samples: list[Sample]
    forecasts: list[dict[str, Any]]
    read_phase: Callable[[], str | None]
    send: Callable[[str, str, str], tuple[Any, Any]]  # (switch, target, failure) -> (view, raw)
    workload_alive: Callable[[], bool]
    capacity_exists: bool
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic


@dataclass
class Collected:
    action: MigrationAction
    transaction: Any | None
    transaction_raw: Any
    before: list[Sample]
    transition: list[Sample]
    after: list[Sample]
    post_complete: bool
    skip_reasons: list[str] = field(default_factory=list)


def collect_attempt(
    plan: AttemptPlan,
    ctx: Context,
    *,
    failure: str,
    duration: float,
    window: int,
    post_timeout: float,
) -> Collected:
    """Steps 3-6 of the runner: phase, pre window, precondition, action, post window."""
    def alive() -> bool:
        return ctx.workload_alive()

    common: dict[str, Any] = {"interval": 0.2, "sleep": ctx.sleep, "clock": ctx.clock}
    skip: list[str] = []

    ready = wait_for(
        lambda: next((s for s in ctx.samples if s.valid and is_ready_state(s.view, s.raw)), None),
        READY_TIMEOUT_S, alive=alive, **common,
    )
    if ready is None:
        raise RunnerError("the twin never reported all switches (testbed or orchestrator not ready)")

    seen = wait_for(lambda: ctx.read_phase() == plan.phase, duration + PHASE_TIMEOUT_MARGIN_S,
                    alive=alive, **common)
    if not seen:
        skip.append(f"phase_not_reached:{plan.phase}")
    start_index = len(ctx.samples)

    before = None
    if not skip:
        before = wait_for(lambda: trailing_window(ctx.samples, start_index, window),
                          PRE_WINDOW_TIMEOUT_S, alive=alive, **common)
        if before is None:
            skip.append("pre_window_incomplete")
    if before is None:  # best effort: the last valid samples we have
        valid = [s for s in ctx.samples if s.valid]
        before = valid[-window:]
    if not before:
        raise RunnerError("no valid snapshot at all before the action")

    last = before[-1]
    switch_id, rule, hot = pick_switch(plan, last.raw["ownership"])
    if not skip:
        skip += check_preconditions(
            last.raw, plan, switch_id,
            workload_alive=alive(), capacity_exists=ctx.capacity_exists,
        )

    is_migrate = plan.action_type == "MIGRATE"
    action = MigrationAction(
        action_type=plan.action_type,
        source_controller=plan.source_controller,
        target_controller=plan.target_controller,
        switch_id=switch_id,
        failure_injected=(is_migrate and failure != "none"),
        selection_rule=rule,
        target_hot=hot,
    )

    send_index = len(ctx.samples)
    transaction: Any | None = None
    transaction_raw: Any = raw_transaction()
    if is_migrate:
        if skip:
            reason = "; ".join(skip)
            transaction_raw = raw_transaction(skip_reason=reason)
            transaction = transaction_from_raw(transaction_raw)
        else:
            transaction, transaction_raw = ctx.send(switch_id, plan.target_controller, failure)
    done_index = len(ctx.samples)

    if transaction is not None and transaction.status == "SKIPPED":
        return Collected(action, transaction, transaction_raw, before,
                         ctx.samples[send_index:], [], False, skip)
    if skip:  # NO_OP with a failed precondition: nothing more to wait for
        return Collected(action, None, transaction_raw, before,
                         ctx.samples[send_index:], [], False, skip)

    expected_owner: str | None
    if transaction is not None and transaction.status == "COMMITTED":
        expected_owner = plan.target_controller
    elif transaction is not None and transaction.status == "FAILED":
        expected_owner = None  # ownership unknown after a failed rollback
    else:  # NO_OP, RESTORED
        expected_owner = plan.source_controller

    def accept(view: Any) -> bool:
        return expected_owner is None or view.owner_of(switch_id) == expected_owner

    after = wait_for(lambda: trailing_window(ctx.samples, done_index, window, accept),
                     post_timeout, alive=alive, **common)
    complete = after is not None
    if after is None:
        after = [s for s in ctx.samples[done_index:] if s.valid and accept(s.view)][-window:]
    cut = after[0].index if complete else len(ctx.samples)
    transition = [s for s in ctx.samples if send_index <= s.index < cut]
    return Collected(action, transaction, transaction_raw, before, transition, after, complete, skip)


# --------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------
class ProcessGroup:
    """A child in its own process group, stopped with SIGINT -> SIGTERM -> SIGKILL."""

    def __init__(self, cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path,
                 use_sudo_to_signal: bool = False) -> None:
        self.cmd, self.cwd, self.env, self.log_path = cmd, cwd, env, log_path
        self.use_sudo_to_signal = use_sudo_to_signal
        self.proc: subprocess.Popen | None = None
        self._log: IO[bytes] | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=self.env, stdout=self._log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _signal(self, sig: int) -> None:
        if self.proc is None:
            return
        if self.use_sudo_to_signal:
            subprocess.run(["sudo", "kill", f"-{int(sig)}", "--", f"-{self.proc.pid}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def wait(self, timeout: float) -> int | None:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> None:
        if self.proc is not None and self.alive():
            for sig, grace in ((signal.SIGINT, 20.0), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
                self._signal(sig)
                if self.wait(grace) is not None:
                    break
        if self._log is not None:
            self._log.close()
            self._log = None

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.returncode


def build_workload_command(
    plan: AttemptPlan,
    settings: RunnerSettings,
    workload_out: Path,
    run_id: str,
    duration: float,
) -> list[str]:
    """runtime_2c20s.py with the source on the workload and the target held constant."""
    env_pairs = [f"PYTHONPATH={settings.repo_root}", "PYTHONDONTWRITEBYTECODE=1"]
    if plan.workload == "hot_switch":
        env_pairs.append(f"NDT_HOT_SWITCH_LEVEL={plan.phase}")
    return [
        "sudo", "-E", "env", *env_pairs, settings.python,
        "src/experiments/topologies/runtime_2c20s.py", WORKLOAD_FLAGS[plan.workload],
        "--profile", f"{plan.source_controller}={plan.workload.replace('_', '-')}",
        "--profile", f"{plan.target_controller}=constant:{plan.destination_load_level}",
        "--capacity-runs", str(settings.capacity_runs),
        "--controller-capacity", str(settings.capacity_json),
        "--output-dir", str(workload_out),
        "--duration", str(duration),
        "--run-id", run_id,
        "--seed", str(plan.seed),
    ]


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _list_bridges() -> str | None:
    try:
        out = subprocess.run(["sudo", "ovs-vsctl", "list-br"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def preflight(
    settings: RunnerSettings,
    *,
    port_open: Callable[[int], bool] = _port_open,
    list_bridges: Callable[[], str | None] = _list_bridges,
) -> None:
    """Fail early and loudly; never repair anything silently."""
    problems: list[str] = []
    for port in OPENFLOW_PORTS:
        if not port_open(port):
            problems.append(f"Ryu is not listening on port {port} (tb_up / ctl_restart)")
    orch_port = urlparse(settings.orchestrator_url).port or 9000
    if port_open(orch_port):
        problems.append(f"port {orch_port} is busy: stop the running orchestrator first (orch_stop)")
    if not settings.capacity_json.is_file():
        problems.append(f"capacity artifact missing: {settings.capacity_json}")
    bridges = list_bridges()
    if bridges is None:
        problems.append("cannot run 'sudo ovs-vsctl list-br'")
    elif bridges.strip():
        problems.append(
            "leftover OVS bridges from a previous run: "
            "sudo ovs-vsctl list-br | xargs -r -n1 sudo ovs-vsctl del-br   (do NOT use mn -c: it kills Ryu)"
        )
    if problems:
        raise RunnerError("preflight failed: " + "; ".join(problems))


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------
def git_state(repo_root: Path) -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                              text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def capacity_version(path: Path) -> str:
    return f"{path.name}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def quality_version(repo_root: Path, globs: Sequence[str] = QUALITY_SOURCE_GLOBS) -> str:
    files = sorted({p for pattern in globs for p in repo_root.glob(pattern) if p.is_file()})
    if not files:
        raise RunnerError(f"no source file matches {list(globs)}: set QUALITY_SOURCE_GLOBS in runner.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return f"quality-src@sha256:{digest.hexdigest()[:12]}"


# --------------------------------------------------------------------------
# writing the attempt directory
# --------------------------------------------------------------------------
def _raw(sample: Sample) -> Any:
    if sample.raw is not None:
        return sample.raw
    return {"error": sample.error, "observed_at": sample.observed_at}


def quarantine(settings: RunnerSettings, name: str, work_dir: Path, exc: BaseException) -> Path:
    """Keep the error and the logs of a failed attempt in _invalid/ (never in the dataset)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = settings.output_root / "_invalid" / f"{name}-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "error.txt").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    for log_name in ("orchestrator.log", "workload.log"):
        if (work_dir / log_name).exists():
            shutil.copy2(work_dir / log_name, dest / log_name)
    return dest


# --------------------------------------------------------------------------
# one attempt end to end
# --------------------------------------------------------------------------
def _clean_dir(path: Path) -> None:
    """Remove a leftover work directory; the workload runs under sudo, so its files may belong to root."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        subprocess.run(["sudo", "rm", "-rf", "--", str(path)], check=False)
    if path.exists():
        raise RunnerError(f"cannot remove the leftover work directory {path}")


def run_attempt(
    plan: AttemptPlan,
    settings: RunnerSettings,
    *,
    failure: str = "none",
    duration: float | None = None,
    log: Callable[[str], None] = print,
) -> AttemptResult:
    name = attempt_dir_name(plan, failure)
    final_dir = settings.output_root / name
    if (final_dir / "validation.json").exists():
        return AttemptResult(name, "exists", final_dir)
    if final_dir.exists():
        raise RunnerError(f"{final_dir} exists without validation.json: move it away first")

    started = time.monotonic()
    started_at = now_iso()
    duration = float(duration or DEFAULT_DURATION_S[plan.workload])
    window = settings.required_window

    preflight(settings)
    commit, dirty = git_state(settings.repo_root)
    if dirty:
        log("WARNING: the working tree has uncommitted changes (not allowed for the official collection)")
    cap_version = capacity_version(settings.capacity_json)
    qual_version = quality_version(settings.repo_root)

    work_dir = settings.work_root / name
    _clean_dir(work_dir)
    work_dir.mkdir(parents=True)
    workload_out = work_dir / "workload"
    samples_path = workload_out / name / "forecast_samples.jsonl"

    url = settings.orchestrator_url.rstrip("/")
    orchestrator = ProcessGroup(
        ["./scripts/start_orchestrator.sh"], cwd=settings.repo_root,
        env={**os.environ, "NDT_CONTROLLERS_CONFIG": settings.controllers_config, "RUN_ID": name},
        log_path=work_dir / "orchestrator.log",
    )
    workload = ProcessGroup(
        build_workload_command(plan, settings, workload_out, name, duration),
        cwd=settings.repo_root, env=dict(os.environ),
        log_path=work_dir / "workload.log", use_sudo_to_signal=True,
    )
    poller = StatePoller(lambda: http_get_json(f"{url}/api/v1/twin/state"),
                         lambda: http_get_json(f"{url}/api/v1/twin/forecast"))
    try:
        log(f"[{name}] starting orchestrator")
        orchestrator.start()
        if not wait_for(lambda: _healthy(url), ORCH_START_TIMEOUT_S, alive=orchestrator.alive):
            raise RunnerError("orchestrator did not answer /health (see orchestrator.log)")
        log(f"[{name}] starting workload ({plan.workload}, {duration:.0f} s)")
        workload.start()
        poller.start()

        ctx = Context(
            samples=poller.samples, forecasts=poller.forecasts,
            read_phase=lambda: read_last_phase(samples_path),
            send=lambda switch, target, fail: send_migration(url, switch, target, fail),
            workload_alive=workload.alive,
            capacity_exists=settings.capacity_json.is_file(),
        )
        collected = collect_attempt(plan, ctx, failure=failure, duration=duration,
                                    window=window, post_timeout=settings.post_window_timeout_s)
        poller.stop()
        log(f"[{name}] waiting for the workload to finish")
        code = workload.wait(duration + WORKLOAD_EXIT_MARGIN_S)
        if code is None or code != 0:
            raise RunnerError(f"workload did not exit cleanly (exit code {code}), see workload.log")

        before_views = [s.view for s in collected.before if s.view is not None]
        after_views = [s.view for s in collected.after if s.view is not None]
        outcome = compute_outcome(before_views, after_views, collected.action,
                                  collected.transaction, required_window=window)
        quality = summarize_quality(before_views, after_views, required_window=window)
        metadata = AttemptMetadata(
            attempt_id=name, workload=plan.workload, phase=plan.phase,
            destination_load_level=plan.destination_load_level,
            source_controller=plan.source_controller, target_controller=plan.target_controller,
            repeat=plan.repeat, seed=plan.seed, topology=settings.topology, git_commit=commit,
            config_sha256=config_sha256(settings.config_path), capacity_version=cap_version,
            quality_version=qual_version, started_at=started_at, finished_at=now_iso())
        written = write_attempt(
            settings.output_root, metadata,
            [_raw(s) for s in collected.before], [_raw(s) for s in collected.transition],
            [_raw(s) for s in collected.after], poller.forecasts,
            collected.action, collected.transaction_raw, outcome, quality,
            required_window=window)
        return AttemptResult(
            name, "written", written.path,
            None if collected.transaction is None else collected.transaction.status,
            bool(written.validation.get("training_eligible", outcome.training_eligible)),
            time.monotonic() - started, "; ".join(collected.skip_reasons),
            written.validation.get("valid"))
    except BaseException as exc:
        poller.stop()
        dest = quarantine(settings, name, work_dir, exc)
        log(f"[{name}] FAILED, moved to {dest}")
        raise
    finally:
        poller.stop()
        workload.stop()  # signals only if still alive; always closes the log
        orchestrator.stop()


def _healthy(url: str) -> bool:
    try:
        http_get_json(f"{url}/health", timeout=2.0)
        return True
    except Exception:
        return False

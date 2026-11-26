from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.common.enums import TransactionState
from src.execution.migration import ControllerClient, RoleMigrationExecutor
from src.execution.rollback import RollbackExecutor
from src.execution.verification import MigrationVerifier
from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.decision_service import DecisionService
from src.orchestrator.forecast_service import ForecastService
from src.twin.calibration.online_margin import TrackerConfig
from src.twin.decision.methods import load_method
from src.orchestrator.outcome_service import OutcomeService
from src.orchestrator.ownership_manager import OwnershipManager
from src.orchestrator.fault_injection import FaultInjector
from src.orchestrator.transaction_manager import TransactionConflict, TransactionManager
from src.telemetry.collector import TelemetryCollector
from src.telemetry.snapshot_builder import SnapshotBuilder
from src.telemetry.snapshot_validator import SnapshotValidator
from src.telemetry.writer import JsonlWriter
from src.twin.capacity.model import CapacityModel, resolve_capacity_path
from src.twin.state.builder import TwinStateBuilder
from src.twin.state.setup_probe_store import SetupProbeStore, SetupStats
from src.twin.state.quality import TwinQualityAssessor
from src.twin.state.quality_score import validate_config
from src.twin.state.twinning_rate import TwinningRateTracker, migration_transient

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(env_name: str, default_path: str):
    path = Path(os.getenv(env_name, default_path))
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


controllers_cfg = load_yaml("NDT_CONTROLLERS_CONFIG", "configs/controllers.yaml")
telemetry_cfg = load_yaml("NDT_TELEMETRY_CONFIG", "configs/telemetry.yaml")
migration_cfg = load_yaml("NDT_MIGRATION_CONFIG", "configs/migration.yaml")
forecast_cfg = load_yaml("NDT_FORECAST_CONFIG", "configs/forecasting/runtime.yaml")
outcome_cfg = load_yaml("NDT_OUTCOME_CONFIG", "configs/outcome/runtime.yaml")
quality_score_cfg = validate_config(load_yaml("NDT_QUALITY_CONFIG", "configs/quality.yaml"))

CONTROLLERS = controllers_cfg["controllers"]
SWITCHES = controllers_cfg["switches"]
INITIAL_OWNERSHIP = controllers_cfg["initial_ownership"]
TOPOLOGY_VERSION = int(controllers_cfg.get("topology_version", 1))

controller_urls = {cid: cfg["rest_url"] for cid, cfg in CONTROLLERS.items()}
switch_dpids = {sid: int(cfg["dpid"]) for sid, cfg in SWITCHES.items()}

ownership_manager = OwnershipManager(INITIAL_OWNERSHIP, list(CONTROLLERS))
transaction_manager = TransactionManager()

client = ControllerClient(
    controller_urls,
    request_timeout_seconds=float(migration_cfg["role_request_timeout_seconds"]),
)
migration_executor = RoleMigrationExecutor(
    client,
    role_timeout_seconds=float(migration_cfg["role_request_timeout_seconds"]),
    barrier_timeout_seconds=float(migration_cfg["barrier_timeout_seconds"]),
)
verifier = MigrationVerifier(
    client,
    timeout_seconds=float(migration_cfg["verification_timeout_seconds"]),
)
rollback_executor = RollbackExecutor(
    client,
    role_timeout_seconds=float(migration_cfg["role_request_timeout_seconds"]),
    barrier_timeout_seconds=float(migration_cfg["barrier_timeout_seconds"]),
)

collector = TelemetryCollector(
    controller_urls,
    timeout_seconds=float(telemetry_cfg.get("request_timeout_seconds", 1.0)),
)
validator = SnapshotValidator(
    expected_controller_ids=set(CONTROLLERS),
    expected_switch_ids=set(SWITCHES),
    freshness_max_age_ms=float(telemetry_cfg["freshness_max_age_ms"]),
)
snapshot_builder = SnapshotBuilder(validator)
capacity_model = CapacityModel.from_file(resolve_capacity_path(ROOT))
quality_cfg = telemetry_cfg["twin_quality"]
twinning_tracker = TwinningRateTracker(
    window_seconds=float(quality_cfg["twinning_window_seconds"]),
)
TRANSACTION_GRACE = timedelta(seconds=float(quality_cfg["transaction_grace_seconds"]))
quality_assessor = TwinQualityAssessor(
    expected_controller_ids=set(CONTROLLERS),
    expected_switch_ids=set(SWITCHES),
    max_age_ms=float(telemetry_cfg["freshness_max_age_ms"]),
    min_completeness_ratio=float(quality_cfg["min_completeness_ratio"]),
    min_twinning_rate=float(quality_cfg["min_twinning_rate"]),
    twinning_rate_provider=twinning_tracker.rate,
    quality_config=quality_score_cfg,
)
twin_state_builder = TwinStateBuilder(
    capacity_model=capacity_model,
    quality_provider=quality_assessor.assess,
)
store = CurrentStateStore()
# Fault scenarios D1/D2 only; off (and never consulted) unless NDT_FAULT_INJECTION=1.
fault_injector = FaultInjector(os.environ.get("NDT_FAULT_INJECTION") == "1", CONTROLLERS)
setup_probe_store = SetupProbeStore(float(telemetry_cfg["freshness_max_age_ms"]))
writer = JsonlWriter()
forecast_service = ForecastService.from_config(forecast_cfg, ROOT, store, writer)
outcome_service = OutcomeService.from_config(outcome_cfg, ROOT, store)
# Campaign method (NDT_DECISION_METHOD, configs/decision/methods.yaml); default
# ndt_no_feedback = the decision pipeline as before. The method also sets the m_f mode of
# the forecaster and, for the "no learned part" ablation, the outcome model.
decision_method = load_method()
forecast_service.configure_margins(
    decision_method.margins,
    TrackerConfig(decision_method.feedback.alpha, decision_method.feedback.gamma_forecast))
if decision_method.outcome == "analytical":
    outcome_service.predictor.override = "analytical"
decision_service = DecisionService(
    store, forecast_service.payload, outcome_service.predictor, transaction_manager,
    method=decision_method,
)
print(f"[decision] method={decision_method.name} ({decision_method.version})", flush=True)
# Decision records of plan/execute: one directory per decision (gitignored data).
decision_service.record_root = ROOT / decision_service.loop_config.record_dir
stop_event = threading.Event()


class GenerationIdGenerator:
    def __init__(self):
        self._value = time.time_ns()
        self._lock = threading.Lock()

    def next_generation_id(self):
        with self._lock:
            self._value = max(self._value + 1, time.time_ns())
            return self._value & ((1 << 64) - 1)


generation_ids = GenerationIdGenerator()


class MigrationRequest(BaseModel):
    switch_id: str
    target_controller: str
    simulate_failure: str = "none"


class DecisionRequest(BaseModel):
    scenario: str = "unlabelled"
    simulate_failure: str = "none"


class FaultRequest(BaseModel):
    kind: str
    controller_id: str
    seconds: float


class KillSwitchRequest(BaseModel):
    enabled: bool


class OutcomePredictRequest(BaseModel):
    switch_id: str
    source_controller: str
    target_controller: str


class SetupProbeStatsRequest(BaseModel):
    p95_ms: float
    samples: int
    timeouts: int
    window_end: datetime


class SetupProbeRequest(BaseModel):
    controllers: dict[str, SetupProbeStatsRequest]


def collect_once(run_id: str):
    for controller_id in CONTROLLERS:
        if fault_injector.enabled and fault_injector.skip_poll(controller_id):  # D1/D2 only
            if fault_injector.take_drop(controller_id):
                store.forget_controller(controller_id)
            continue
        try:
            collection = collector.collect_controller(controller_id)
            store.update_collection(
                controller_id,
                collection.controller,
                collection.switches,
                collection.runtime_state,
            )
            writer.append(
                ROOT
                / telemetry_cfg["raw_data_dir"]
                / run_id
                / "controllers.jsonl",
                collection.controller.to_dict(),
            )
            for switch in collection.switches:
                writer.append(
                    ROOT
                    / telemetry_cfg["raw_data_dir"]
                    / run_id
                    / "switches.jsonl",
                    switch.to_dict(),
                )
        except Exception as exc:  # keep polling the other controller
            store.mark_error(controller_id, exc)
            print(f"[telemetry] controller={controller_id} error={exc}", flush=True)


def build_snapshot(run_id: str):
    controllers, switches = store.values()
    snapshot = snapshot_builder.build(
        topology_version=TOPOLOGY_VERSION,
        ownership_version=ownership_manager.version,
        controllers=controllers,
        switches=switches,
        ownership=ownership_manager.states(),
        role_matrix=store.role_matrix(),
    )
    snapshot = setup_probe_store.overlay(snapshot)
    try:
        moving = transaction_manager.switches_in_flight(snapshot.created_at - TRANSACTION_GRACE)
        if migration_transient(snapshot.quality, moving):
            twinning_tracker.skip()
        else:
            twinning_tracker.record(
                success=snapshot.quality.valid,
                observed_at=snapshot.created_at,
            )
        twin_state = twin_state_builder.build(snapshot)
        store.set_snapshot(snapshot)
        store.set_twin_state(twin_state)
    except Exception:
        twinning_tracker.record(success=False)
        raise
    writer.append(
        ROOT / telemetry_cfg["snapshot_data_dir"] / f"{run_id}.jsonl",
        snapshot.to_dict(),
    )
    # Outside the try above: a forecasting error must not count as a failed
    # snapshot (that would lower the twinning rate and the Quality Score).
    forecast_service.on_twin_state(twin_state.to_dict(), run_id)
    outcome_service.on_twin_state(twin_state.to_dict())
    decision_service.on_twin_state(twin_state.to_dict())


def telemetry_loop():
    run_id = os.getenv("RUN_ID", "dev")
    poll_interval = float(telemetry_cfg["poll_interval_seconds"])
    snapshot_interval = float(telemetry_cfg["snapshot_interval_seconds"])

    if poll_interval <= 0.0:
        raise ValueError("poll_interval_seconds must be positive")
    if snapshot_interval <= 0.0:
        raise ValueError("snapshot_interval_seconds must be positive")

    next_poll = time.monotonic()
    next_snapshot = next_poll

    while not stop_event.is_set():
        collect_once(run_id)

        now = time.monotonic()

        if now >= next_snapshot:
            try:
                build_snapshot(run_id)
            except Exception as exc:
                print(f"[snapshot] error={exc}", flush=True)

            # Keep a fixed snapshot cadence instead of scheduling from
            # the completion time of the previous snapshot.
            while next_snapshot <= now:
                next_snapshot += snapshot_interval

        next_poll += poll_interval

        now = time.monotonic()
        while next_poll <= now:
            next_poll += poll_interval

        stop_event.wait(
            max(0.0, next_poll - time.monotonic())
        )


app = FastAPI(title="NDT Safe Load Balancing Orchestrator", version="0.2.0")

# Serve the dashboard from the same origin as the API (http://<host>/dashboard/):
# one `git pull` updates backend and UI together, no CORS or rewrite needed.
app.mount(
    "/dashboard",
    StaticFiles(directory=ROOT / "dashboard", html=True),
    name="dashboard",
)


@app.on_event("startup")
def startup():
    stop_event.clear()
    threading.Thread(
        target=telemetry_loop,
        daemon=True,
        name="telemetry-loop",
    ).start()


@app.on_event("shutdown")
def shutdown():
    stop_event.set()


@app.get("/health")
def health():
    states = {}
    for controller_id in CONTROLLERS:
        try:
            states[controller_id] = {
                "reachable": True,
                "state": client.state(controller_id),
            }
        except Exception as exc:
            states[controller_id] = {"reachable": False, "error": str(exc)}
    return {"status": "UP", "controllers": states}


@app.get("/api/v1/state")
def state():
    return {
        "topology_version": TOPOLOGY_VERSION,
        "ownership_version": ownership_manager.version,
        "ownership": ownership_manager.snapshot_mapping(),
        "locked_switches": transaction_manager.locked_switches(),
        "transactions": transaction_manager.list_all(),
        "telemetry_errors": store.errors(),
        "twinning_skipped": twinning_tracker.skipped,
        "latest_snapshot": store.snapshot_dict(),
    }


@app.get("/api/v1/twin/state")
def twin_state():
    value = store.twin_state_dict()
    if value is None:
        return {"status": "NOT_READY"}
    return value


@app.get("/api/v1/twin/forecast")
def twin_forecast():
    """Latest forecast per controller. Read-only: the planner decides migrations."""
    return forecast_service.payload()


@app.post("/api/v1/twin/outcome/predict")
def twin_outcome_predict(request: OutcomePredictRequest):
    """What-if outcome of moving one switch. No side effects: no transaction, no
    ownership change, no role request. NOT_READY carries a reason and no numbers."""
    try:
        return outcome_service.predict(
            request.switch_id, request.source_controller, request.target_controller
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/twin/decision/evaluate")
def twin_decision_evaluate():
    """Dry run of candidates, pre-check, outcome and Safety Gate on the current snapshot.
    No side effects: no transaction, no ownership change, no role request."""
    return decision_service.evaluate()


@app.post("/api/v1/twin/decision/plan")
def twin_decision_plan(request: DecisionRequest):
    """Dry run of the whole decision: evaluate, rank, select. Nothing is executed; one
    decision record is written. The same snapshot gives the same decision_id."""
    return decision_service.plan(request.scenario)


@app.post("/api/v1/twin/decision/execute")
def twin_decision_execute(request: DecisionRequest):
    """Plan, revalidate on the latest state, then run the selected migration through
    the normal executor (role request, barrier, verification, rollback)."""
    failure_mode = request.simulate_failure.strip().lower()
    if failure_mode not in {"none", "flow_mod"}:
        raise HTTPException(400, "simulate_failure must be 'none' or 'flow_mod'")
    return decision_service.execute(request.scenario, failure_mode)


@app.post("/api/v1/faults")
def inject_fault(request: FaultRequest):
    """Telemetry fault of the D1/D2 episodes; 403 unless NDT_FAULT_INJECTION=1."""
    try:
        return fault_injector.inject(request.kind, request.controller_id, request.seconds)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/faults")
def list_faults():
    return {"enabled": fault_injector.enabled, "active": fault_injector.active()}


@app.post("/api/v1/twin/decision/kill-switch")
def twin_decision_kill_switch(request: KillSwitchRequest):
    """enabled=false stops every plan/execute before any work (status KILL_SWITCH)."""
    return decision_service.set_enabled(request.enabled)


@app.post("/api/v1/twin/setup-probe")
def update_setup_probe(request: SetupProbeRequest):
    """Replace the per-controller flow-setup stats; a controller left out has none."""
    unknown = sorted(set(request.controllers) - set(CONTROLLERS))
    if unknown:
        raise HTTPException(404, f"Unknown controller: {unknown}")
    try:
        setup_probe_store.update(
            {
                cid: SetupStats(x.p95_ms, x.samples, x.timeouts, x.window_end)
                for cid, x in request.controllers.items()
            }
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"accepted": sorted(request.controllers)}


@app.post("/api/v1/init-roles")
def init_roles():
    results = []
    try:
        for switch_id, owner in INITIAL_OWNERSHIP.items():
            dpid = switch_dpids[switch_id]

            for controller_id in CONTROLLERS:
                if controller_id == owner:
                    continue
                generation_id = generation_ids.next_generation_id()
                role = client.set_role(
                    controller_id,
                    dpid,
                    "SLAVE",
                    generation_id,
                    float(migration_cfg["role_request_timeout_seconds"]),
                )
                barrier = client.barrier(
                    controller_id,
                    dpid,
                    float(migration_cfg["barrier_timeout_seconds"]),
                )
                results.append(
                    {
                        "switch_id": switch_id,
                        "controller_id": controller_id,
                        "role": role,
                        "barrier": barrier,
                    }
                )

            generation_id = generation_ids.next_generation_id()
            role = client.set_role(
                owner,
                dpid,
                "MASTER",
                generation_id,
                float(migration_cfg["role_request_timeout_seconds"]),
            )
            barrier = client.barrier(
                owner,
                dpid,
                float(migration_cfg["barrier_timeout_seconds"]),
            )
            results.append(
                {
                    "switch_id": switch_id,
                    "controller_id": owner,
                    "role": role,
                    "barrier": barrier,
                }
            )

        # Start a fresh NDT-quality measurement window only after
        # all initial OpenFlow roles have been established successfully.
        # Startup snapshots collected before Mininet/controller readiness
        # must not contaminate the experiment twinning-rate window.
        twinning_tracker.reset()

        return {
            "status": "INITIALIZED",
            "ownership": ownership_manager.snapshot_mapping(),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/migrations")
def migrate(request: MigrationRequest):
    if request.switch_id not in switch_dpids:
        raise HTTPException(404, f"Unknown switch: {request.switch_id}")
    if request.target_controller not in CONTROLLERS:
        raise HTTPException(404, f"Unknown controller: {request.target_controller}")

    failure_mode = request.simulate_failure.strip().lower()
    if failure_mode not in {"none", "flow_mod"}:
        raise HTTPException(400, "simulate_failure must be 'none' or 'flow_mod'")
    if failure_mode != "none" and not bool(
        migration_cfg.get("allow_simulated_failure", False)
    ):
        raise HTTPException(403, "simulated failure is disabled")

    source = ownership_manager.get_owner(request.switch_id)
    if source == request.target_controller:
        raise HTTPException(400, "Target controller already owns the switch")

    try:
        tx = transaction_manager.create(
            request.switch_id,
            source,
            request.target_controller,
        )
    except TransactionConflict as exc:
        raise HTTPException(409, str(exc)) from exc

    dpid = switch_dpids[request.switch_id]

    try:
        generation_id = generation_ids.next_generation_id()
        tx.generation_id = generation_id
        transaction_manager.transition(tx, TransactionState.ROLE_SWITCHING)
        migration_executor.promote_target(
            dpid,
            request.target_controller,
            generation_id,
        )

        transaction_manager.transition(tx, TransactionState.VERIFYING)
        verification = verifier.verify_migration(
            dpid=dpid,
            source_controller=source,
            target_controller=request.target_controller,
            simulate_flow_mod_failure=(failure_mode == "flow_mod"),
        )

        if not verification.ok:
            raise RuntimeError(
                "Verification failed: " + ",".join(verification.reasons)
            )

        ownership_manager.commit_migration(
            request.switch_id,
            request.target_controller,
            generation_id,
        )
        transaction_manager.transition(
            tx,
            TransactionState.COMMITTED,
            json.dumps(verification.to_dict()),
        )
        transaction_manager.finish(tx)

        return {
            "status": "COMMITTED",
            "verification": verification.to_dict(),
            "transaction": tx.to_dict(),
            "ownership_version": ownership_manager.version,
        }

    except Exception as migration_error:
        transaction_manager.fail(tx, str(migration_error))

        try:
            transaction_manager.transition(tx, TransactionState.ROLLING_BACK)
            rollback_generation_id = generation_ids.next_generation_id()
            rollback_executor.restore_source(
                dpid,
                source,
                rollback_generation_id,
            )

            rollback_verification = verifier.verify_rollback(
                dpid=dpid,
                restored_controller=source,
                other_controller=request.target_controller,
            )
            if not rollback_verification.ok:
                raise RuntimeError(
                    "Rollback verification failed: "
                    + ",".join(rollback_verification.reasons)
                )

            ownership_manager.mark_restored(
                request.switch_id,
                source,
                rollback_generation_id,
            )
            transaction_manager.transition(
                tx,
                TransactionState.RESTORED,
                json.dumps(rollback_verification.to_dict()),
            )
            transaction_manager.finish(tx)

            return {
                "status": "RESTORED",
                "failure_reason": str(migration_error),
                "rollback_verification": rollback_verification.to_dict(),
                "transaction": tx.to_dict(),
                "ownership_version": ownership_manager.version,
            }

        except Exception as rollback_error:
            transaction_manager.fail(tx, str(rollback_error))
            # Release the lock (26/11, fault D6): a transaction left locked kept every later
            # decision DEFERRED for good. Ownership was never committed, so if the switch is
            # not back with the source, the snapshot's role check marks the twin invalid
            # and the Hard Gate stops any decision on it.
            transaction_manager.finish(tx)
            raise HTTPException(
                status_code=500,
                detail={
                    "migration_error": str(migration_error),
                    "rollback_error": str(rollback_error),
                    "transaction": tx.to_dict(),
                },
            ) from rollback_error

def _execute_migration(switch_id: str, target_controller: str, simulate_failure: str):
    """The closed loop executes through the same code path as POST /api/v1/migrations."""
    return migrate(
        MigrationRequest(
            switch_id=switch_id,
            target_controller=target_controller,
            simulate_failure=simulate_failure,
        )
    )


decision_service.executor = _execute_migration

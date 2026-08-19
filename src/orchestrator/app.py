from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common.enums import TransactionState
from src.execution.migration import ControllerClient, RoleMigrationExecutor
from src.execution.rollback import RollbackExecutor
from src.execution.verification import MigrationVerifier
from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.ownership_manager import OwnershipManager
from src.orchestrator.transaction_manager import TransactionConflict, TransactionManager
from src.telemetry.collector import TelemetryCollector
from src.telemetry.snapshot_builder import SnapshotBuilder
from src.telemetry.snapshot_validator import SnapshotValidator
from src.telemetry.writer import JsonlWriter

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
store = CurrentStateStore()
writer = JsonlWriter()
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


def collect_once(run_id: str):
    for controller_id in CONTROLLERS:
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
    store.set_snapshot(snapshot)
    writer.append(
        ROOT / telemetry_cfg["snapshot_data_dir"] / f"{run_id}.jsonl",
        snapshot.to_dict(),
    )


def telemetry_loop():
    run_id = os.getenv("RUN_ID", "dev")
    poll_interval = float(telemetry_cfg["poll_interval_seconds"])
    snapshot_interval = float(telemetry_cfg["snapshot_interval_seconds"])
    next_snapshot = time.monotonic()

    while not stop_event.is_set():
        started = time.monotonic()
        collect_once(run_id)
        if time.monotonic() >= next_snapshot:
            try:
                build_snapshot(run_id)
            except Exception as exc:
                print(f"[snapshot] error={exc}", flush=True)
            next_snapshot = time.monotonic() + snapshot_interval
        stop_event.wait(max(0.0, poll_interval - (time.monotonic() - started)))


app = FastAPI(title="NDT Safe Load Balancing Orchestrator", version="0.2.0")


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
        "latest_snapshot": store.snapshot_dict(),
    }


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
            raise HTTPException(
                status_code=500,
                detail={
                    "migration_error": str(migration_error),
                    "rollback_error": str(rollback_error),
                    "transaction": tx.to_dict(),
                },
            ) from rollback_error
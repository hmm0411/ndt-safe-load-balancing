from __future__ import annotations
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common.enums import TransactionState
from src.execution.migration import ControllerClient, RoleMigrationExecutor
from src.execution.rollback import RollbackExecutor
from src.execution.verification import MigrationVerifier
from src.orchestrator.ownership_manager import OwnershipManager
from src.orchestrator.transaction_manager import TransactionConflict, TransactionManager
from src.schemas.telemetry import ControllerTelemetry, SwitchTelemetry
from src.telemetry.snapshot_builder import SnapshotBuilder
from src.telemetry.snapshot_validator import SnapshotValidator

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

validator = SnapshotValidator(
    expected_controller_ids=set(CONTROLLERS),
    expected_switch_ids=set(SWITCHES),
    freshness_max_age_ms=float(telemetry_cfg["freshness_max_age_ms"]),
)
snapshot_builder = SnapshotBuilder(validator)

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

class TelemetryStore:
    def __init__(self):
        self.controllers = {}
        self.switches = {}
        self.latest_snapshot = None
        self._lock = threading.RLock()

    def update_controller(self, item):
        with self._lock:
            self.controllers[item.controller_id] = item

    def update_switch(self, item):
        with self._lock:
            self.switches[(item.switch_id, item.controller_id)] = item

    def values(self):
        with self._lock:
            return list(self.controllers.values()), list(self.switches.values())

    def set_snapshot(self, snapshot):
        with self._lock:
            self.latest_snapshot = snapshot

    def snapshot_dict(self):
        with self._lock:
            return self.latest_snapshot.to_dict() if self.latest_snapshot else None

store = TelemetryStore()
stop_event = threading.Event()

def parse_dt(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def collect_once(run_id: str):
    ingested_at = datetime.now(timezone.utc)

    for cid in CONTROLLERS:
        payload = client.telemetry(cid)
        source = payload.get("controller")
        if not source:
            continue

        c = ControllerTelemetry(
            controller_id=cid,
            observed_at=parse_dt(source["observed_at"]),
            ingested_at=ingested_at,
            packet_in_total=int(source["packet_in_total"]),
            packet_in_rate=float(source["packet_in_rate"]),
            flow_mod_total=int(source["flow_mod_total"]),
            flow_mod_rate=float(source["flow_mod_rate"]),
            process_cpu_percent=float(source["process_cpu_percent"]),
            process_memory_rss_mb=float(source["process_memory_rss_mb"]),
            response_mean_ms=float(source["response_mean_ms"]),
            response_p95_ms=float(source["response_p95_ms"]),
            managed_switch_count=int(source["managed_switch_count"]),
        )
        store.update_controller(c)
        append_jsonl(
            ROOT / telemetry_cfg["raw_data_dir"] / run_id / "controllers.jsonl",
            c.to_dict(),
        )

        for item in payload.get("switches", []):
            s = SwitchTelemetry(
                switch_id=item["switch_id"],
                controller_id=cid,
                observed_at=parse_dt(item["observed_at"]),
                packet_in_total=int(item["packet_in_total"]),
                packet_in_rate=float(item["packet_in_rate"]),
                flow_mod_total=int(item["flow_mod_total"]),
                flow_mod_rate=float(item["flow_mod_rate"]),
                control_load_share=float(item["control_load_share"]),
            )
            store.update_switch(s)
            append_jsonl(
                ROOT / telemetry_cfg["raw_data_dir"] / run_id / "switches.jsonl",
                s.to_dict(),
            )

def build_snapshot(run_id: str):
    controllers, switches = store.values()
    snapshot = snapshot_builder.build(
        topology_version=TOPOLOGY_VERSION,
        ownership_version=ownership_manager.version,
        controllers=controllers,
        switches=switches,
        ownership=ownership_manager.states(),
    )
    store.set_snapshot(snapshot)
    append_jsonl(
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
        try:
            collect_once(run_id)
            if time.monotonic() >= next_snapshot:
                build_snapshot(run_id)
                next_snapshot = time.monotonic() + snapshot_interval
        except Exception as exc:
            print(f"[telemetry] {exc}", flush=True)

        stop_event.wait(max(0.0, poll_interval - (time.monotonic() - started)))

app = FastAPI(title="NDT Safe Load Balancing Orchestrator", version="0.1.0")

@app.on_event("startup")
def startup():
    threading.Thread(target=telemetry_loop, daemon=True, name="telemetry-loop").start()

@app.on_event("shutdown")
def shutdown():
    stop_event.set()

@app.get("/health")
def health():
    states = {}
    for cid in CONTROLLERS:
        try:
            states[cid] = {"reachable": True, "state": client.state(cid)}
        except Exception as exc:
            states[cid] = {"reachable": False, "error": str(exc)}
    return {"status": "UP", "controllers": states}

@app.get("/api/v1/state")
def state():
    return {
        "topology_version": TOPOLOGY_VERSION,
        "ownership_version": ownership_manager.version,
        "ownership": ownership_manager.snapshot_mapping(),
        "locked_switches": transaction_manager.locked_switches(),
        "transactions": transaction_manager.list_all(),
        "latest_snapshot": store.snapshot_dict(),
    }

@app.post("/api/v1/init-roles")
def init_roles():
    results = []
    try:
        for sid, owner in INITIAL_OWNERSHIP.items():
            dpid = switch_dpids[sid]

            for cid in CONTROLLERS:
                if cid == owner:
                    continue
                gen = generation_ids.next_generation_id()
                role = client.set_role(
                    cid, dpid, "SLAVE", gen,
                    float(migration_cfg["role_request_timeout_seconds"])
                )
                barrier = client.barrier(
                    cid, dpid,
                    float(migration_cfg["barrier_timeout_seconds"])
                )
                results.append({"switch_id": sid, "controller_id": cid,
                                "role": role, "barrier": barrier})

            gen = generation_ids.next_generation_id()
            role = client.set_role(
                owner, dpid, "MASTER", gen,
                float(migration_cfg["role_request_timeout_seconds"])
            )
            barrier = client.barrier(
                owner, dpid,
                float(migration_cfg["barrier_timeout_seconds"])
            )
            results.append({"switch_id": sid, "controller_id": owner,
                            "role": role, "barrier": barrier})

        return {
            "status": "INITIALIZED",
            "ownership": ownership_manager.snapshot_mapping(),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/v1/migrations")
def migrate(request: MigrationRequest):
    if request.switch_id not in switch_dpids:
        raise HTTPException(404, f"Unknown switch: {request.switch_id}")
    if request.target_controller not in CONTROLLERS:
        raise HTTPException(404, f"Unknown controller: {request.target_controller}")

    source = ownership_manager.get_owner(request.switch_id)
    if source == request.target_controller:
        raise HTTPException(400, "Target controller already owns the switch")

    try:
        tx = transaction_manager.create(
            request.switch_id, source, request.target_controller
        )
    except TransactionConflict as exc:
        raise HTTPException(409, str(exc))

    dpid = switch_dpids[request.switch_id]

    try:
        gen = generation_ids.next_generation_id()
        tx.generation_id = gen
        transaction_manager.transition(tx, TransactionState.ROLE_SWITCHING)

        migration_executor.promote_target(dpid, request.target_controller, gen)

        transaction_manager.transition(tx, TransactionState.VERIFYING)

        verification = verifier.verify_migration(
            dpid=dpid,
            source_controller=source,
            target_controller=request.target_controller,
            simulate_flow_mod_failure=(request.simulate_failure == "flow_mod"),
        )
        if not verification.ok:
            raise RuntimeError("Verification failed: " + ",".join(verification.reasons))
        
        ownership_manager.commit_migration(request.switch_id, request.target_controller, gen)
        transaction_manager.transition(
            tx, TransactionState.COMMITTED, json.dumps(verification.to_dict())
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
            rollback_gen = generation_ids.next_generation_id()
            rollback_executor.restore_source(dpid, source, rollback_gen)

            rollback_verification = verifier.verify_roles(
                dpid, request.target_controller, source
            )
            if not rollback_verification.ok:
                raise RuntimeError(
                    "Rollback verification failed: "
                    + ",".join(rollback_verification.reasons)
                )

            ownership_manager.mark_restored(request.switch_id, source, rollback_gen)
            transaction_manager.transition(
                tx, TransactionState.RESTORED,
                json.dumps(rollback_verification.to_dict())
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
            transaction_manager.finish(tx)
            raise HTTPException(
                500,
                {
                    "migration_error": str(migration_error),
                    "rollback_error": str(rollback_error),
                    "transaction": tx.to_dict(),
                },
            )

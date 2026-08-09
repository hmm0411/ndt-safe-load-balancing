# ndt-safe-load-balancing# NDT Safe Load Balancing - Current MVP

Current target:
- Ryu 4.34 + OpenFlow 1.3 reactive forwarding
- 2-controller role-based switch migration
- Role/Barrier verification and rollback
- Shared telemetry schema
- 1-second telemetry polling
- 5-second coherent snapshots
- JSONL raw/snapshot storage

Run instructions are at the end of this README.

## Shared schema

Both developers must import the shared classes from `src/schemas/`:
- `ControllerTelemetry`
- `SwitchTelemetry`
- `OwnershipState`

They use stdlib `dataclasses` so the Ryu virtualenv does not need Pydantic.

## Environments

Ryu:
```bash
source ~/ryu-venv/bin/activate
pip install psutil
```

Orchestrator:
```bash
python3 -m venv ~/ndt-venv
source ~/ndt-venv/bin/activate
pip install -r requirements-orchestrator.txt
```

## Run

Terminal 1:
```bash
./scripts/start_c1.sh
```

Terminal 2:
```bash
./scripts/start_c2.sh
```

Terminal 3:
```bash
./scripts/start_orchestrator.sh
```

Terminal 4:
```bash
sudo python3 src/experiments/topologies/smoke_2c4s.py
```

Then:
```bash
curl -s -X POST http://127.0.0.1:9000/api/v1/init-roles | python3 -m json.tool
```

Return to Mininet:
```text
mininet> pingall
mininet> sh ovs-vsctl --columns=target,is_connected,status list Controller
mininet> sh ovs-ofctl -O OpenFlow13 dump-flows s1
```

Telemetry:
```bash
curl -s http://127.0.0.1:8081/api/v1/telemetry | python3 -m json.tool
curl -s http://127.0.0.1:9000/api/v1/state | python3 -m json.tool
```

Migrate s1 from c1 to c2:
```bash
curl -s -X POST http://127.0.0.1:9000/api/v1/migrations   -H 'Content-Type: application/json'   -d '{"switch_id":"s1","target_controller":"c2"}'   | python3 -m json.tool
```

Rollback test:
```bash
curl -s -X POST http://127.0.0.1:9000/api/v1/migrations   -H 'Content-Type: application/json'   -d '{"switch_id":"s1","target_controller":"c2","simulate_failure":true}'   | python3 -m json.tool
```

Raw telemetry:
```text
data/raw/<RUN_ID>/controllers.jsonl
data/raw/<RUN_ID>/switches.jsonl
```

Snapshots:
```text
data/snapshots/<RUN_ID>.jsonl
```

Default `RUN_ID=dev`.

## Tests

```bash
python3 -m unittest discover -s tests/unit -v
```

`OwnershipState` is intentionally kept exactly in the current 2-controller
contract. Before the 4-controller experiment it should be generalized to an
arbitrary controller-connection list.

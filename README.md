# NDT Safe Load Balancing — Observable Role Migration MVP

This repository is the current Phase-1 implementation of the thesis: multi-controller SDN role migration plus telemetry, state, snapshots, verification and rollback.

## Current milestone

The milestone is complete only when the following chain is observable end to end:

`Ryu C1/C2 -> telemetry every 1 s -> Central Orchestrator -> raw JSONL -> coherent snapshot every 5 s -> migration -> ownership/role change -> post-migration snapshot -> rollback snapshot`

Do **not** start forecasting, XGBoost/Random Forest, safe-capacity modeling, 20/40-switch experiments or dashboard work until this milestone passes.

## Repository responsibilities

- `src/controller/`: Ryu/OpenFlow-facing code: reactive forwarding, role, barrier, in-process telemetry.
- `src/orchestrator/`: global ownership, transactions and latest runtime state.
- `src/telemetry/`: polling, raw storage, snapshot building and validation.
- `src/execution/`: migration, verification and rollback.
- `src/schemas/`: shared contracts. Both developers must use these classes rather than inventing separate JSON shapes.
- `tests/unit/`: fast tests that run on every PR.
- `tests/integration/`: real 2C-4S SDN smoke test for the GCP self-hosted runner.

## Branch workflow for two developers

Protected branches:

- `main`: stable thesis milestones/releases only.
- `dev`: integration branch. No direct feature coding here.

Feature branches:

- Person 1: `feat/role-migration-transaction`
- Person 2: `feat/telemetry-state`
- Later work: one branch per feature, for example `feat/capacity-benchmark`.

Before starting a new feature, update `dev` first:

```bash
git checkout dev
git fetch origin
git pull --ff-only origin dev
git status
git log --oneline -5
git checkout -b feat/<feature-name>
```

**Tool/source:** Git CLI. `fetch` refreshes remote refs; `pull --ff-only` accepts only a fast-forward update and prevents an accidental merge commit on `dev`; `checkout -b` creates an isolated feature branch.

While a feature branch is in progress, sync the latest integration changes without changing `dev`:

```bash
git fetch origin
git merge origin/dev
```

Resolve conflicts on the feature branch, then run tests before pushing. Do not force-push a branch that the other developer is already using unless both people explicitly agree.

Push and open a PR:

```bash
git push -u origin feat/<feature-name>
```

PR target is `dev`. The other developer reviews it. Merge only after CI is green. After a milestone has passed the real SDN integration test, create a PR from `dev` to `main` and tag the milestone.

Recommended repository rules:

- protect `dev` and `main`;
- require pull requests rather than direct pushes;
- require the hosted CI status check before merge;
- require at least one review;
- do not let a feature PR bypass the shared-schema review when `src/schemas/` changes.

A PR checklist template is included at `.github/pull_request_template.md`.

## First clone on a machine

```bash
git clone <REPOSITORY_URL>
cd ndt-safe-load-balancing
git checkout dev
cp .env.example .env
```

**Tool/source:** Git CLI for clone/checkout; standard shell `cp` for the local environment file. Never commit `.env`.

### Ryu environment

Ryu remains separate from the orchestrator virtualenv:

```bash
source ~/ryu-venv/bin/activate
pip install -r requirements-controller-extra.txt
```

**Tool/source:** Python `venv` activation + pip. `psutil` is required to measure the CPU/RSS of the Ryu process itself.

### Orchestrator environment

```bash
python3 -m venv ~/ndt-venv
source ~/ndt-venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-orchestrator.txt
pip install -r requirements-dev.txt
```

**Tool/source:** Python `venv` and pip. The orchestrator environment contains FastAPI/Uvicorn/Requests/PyYAML; dev dependencies provide Ruff and Mypy for CI-equivalent local checks.

## Configuration

`configs/telemetry.yaml`:

```yaml
poll_interval_seconds: 1.0
request_timeout_seconds: 1.0
snapshot_interval_seconds: 5.0
freshness_max_age_ms: 2500
raw_data_dir: "data/raw"
snapshot_data_dir: "data/snapshots"
```

`configs/migration.yaml` contains only migration/verification timeouts. Keep telemetry values out of this file.

## Run order — local development

Always clean an old Mininet run first:

```bash
./scripts/cleanup.sh
```

**Tool/source:** project shell script; internally uses Mininet `mn -c` to remove stale namespaces/OVS state.

Open four terminals from the repository root.

Terminal 1 — C1:

```bash
./scripts/start_c1.sh
```

Terminal 2 — C2:

```bash
./scripts/start_c2.sh
```

Terminal 3 — Central Orchestrator:

```bash
RUN_ID=dev ./scripts/start_orchestrator.sh
```

Terminal 4 — Mininet 2C-4S smoke topology:

```bash
sudo python3 src/experiments/topologies/smoke_2c4s.py
```

Do not generate traffic before roles are initialized.

Initialize roles from another terminal:

```bash
curl -fsS -X POST http://127.0.0.1:9000/api/v1/init-roles | python3 -m json.tool
```

**Tool/source:** curl HTTP CLI + Python stdlib `json.tool`. Expected initial ownership is `s1,s2 -> c1` and `s3,s4 -> c2`, with the owner MASTER and the peer SLAVE.

## Verify controller telemetry before the orchestrator pipeline

```bash
curl -fsS http://127.0.0.1:8081/api/v1/telemetry | python3 -m json.tool
curl -fsS http://127.0.0.1:8082/api/v1/telemetry | python3 -m json.tool
```

With no traffic, rates may be zero, but CPU/RAM and timestamps must exist. The controller payload must include both received and processed counters/rates:

- `packet_in_total`, `packet_in_rate`
- `processed_packet_in_total`, `processed_packet_in_rate`
- `flow_mod_total`, `flow_mod_rate`
- `process_cpu_percent`, `process_memory_rss_mb`
- `response_mean_ms`, `response_p95_ms`

Per-switch telemetry must also contain `processed_packet_in_rate` and `control_load_share`.

## Generate a new Packet-In

Inside Mininet:

```text
mininet> pingall
```

A second ping may hit learned flows and therefore produce no new Packet-In. To force a table miss on `s1` while keeping the priority-0 table-miss rule:

```text
mininet> sh ovs-ofctl -O OpenFlow13 del-flows s1 "priority=10"
mininet> h1 ping -c 1 h4
```

**Tool/source:** Open vSwitch `ovs-ofctl`. `-O OpenFlow13` selects OpenFlow 1.3; deleting only priority-10 rules removes learned reactive flows rather than the table-miss rule.

## Verify raw storage and snapshot

Wait at least 5 seconds, then:

```bash
curl -fsS http://127.0.0.1:9000/api/v1/state | python3 -m json.tool
tail -n 3 data/raw/dev/controllers.jsonl
tail -n 3 data/raw/dev/switches.jsonl
tail -n 2 data/snapshots/dev.jsonl
```

Expected snapshot quality:

```json
{"fresh": true, "complete": true, "consistent": true, "valid": true}
```

`consistent=true` is based on the controller-observed role/connectivity matrix, not merely on the ownership dictionary.

## Migration test: s1 from C1 to C2

Migration verification waits for a **new processed Packet-In** on the target. Use two terminals: one starts the request, the other immediately forces a new table miss/ping in Mininet.

Terminal A:

```bash
curl -fsS -X POST http://127.0.0.1:9000/api/v1/migrations \
  -H 'Content-Type: application/json' \
  -d '{"switch_id":"s1","target_controller":"c2","simulate_failure":"none"}' \
  | python3 -m json.tool
```

Mininet while the request is in `VERIFYING`:

```text
mininet> sh ovs-ofctl -O OpenFlow13 del-flows s1 "priority=10"
mininet> h1 ping -c 1 h4
```

Expected result: `COMMITTED`. Within the next snapshot interval, `ownership.s1` and the snapshot ownership must be `c2`; C2 must process the new Packet-In from `s1`.

## Reverse migration and rollback/fault-injection test

First return `s1` from C2 to C1 successfully. This proves both migration directions before fault injection:

```bash
curl -fsS -X POST http://127.0.0.1:9000/api/v1/migrations \
  -H 'Content-Type: application/json' \
  -d '{"switch_id":"s1","target_controller":"c1","simulate_failure":"none"}' \
  | python3 -m json.tool
```

As with the first migration, force a fresh Packet-In in Mininet during `VERIFYING`. Expected result: `COMMITTED` and the next snapshot has `s1 -> c1`.

Now test rollback from the exact initial ownership direction required by the telemetry milestone. The current fault mode is the string `flow_mod` (not JSON boolean `true`):

```bash
curl -fsS -X POST http://127.0.0.1:9000/api/v1/migrations \
  -H 'Content-Type: application/json' \
  -d '{"switch_id":"s1","target_controller":"c2","simulate_failure":"flow_mod"}' \
  | python3 -m json.tool
```

Again generate a fresh Packet-In while the request is verifying. Expected transaction path is `FAILED -> ROLLING_BACK -> RESTORED`; ownership remains/restores to source controller C1, C1 is MASTER, C2 is SLAVE, `ownership_version` advances according to the transaction contract, and the following snapshot is valid and role-consistent.

## Local test commands before every push

```bash
source ~/ndt-venv/bin/activate
python -m compileall -q src tests
ruff check src/schemas src/telemetry src/orchestrator/app.py src/orchestrator/current_state.py tests/unit
mypy --ignore-missing-imports src/schemas src/telemetry src/orchestrator/current_state.py
python -m unittest discover -s tests/unit -v
```

**Tool/source:** Python stdlib `compileall`/`unittest`, Ruff CLI, and Mypy CLI. These commands mirror the hosted CI job.

## CI/CD model

Convenient local equivalents are also available:

```bash
make ci       # compile + Ruff + Mypy + unit tests
make smoke    # real SDN smoke test; use only on the prepared SDN host
make clean-sdn
```

**Tool/source:** GNU Make wraps the same project commands used by GitHub Actions; it does not replace the underlying Python/Mininet tools.

### CI — every PR to `dev`/`main`

`.github/workflows/ci.yml` runs on a GitHub-hosted runner:

1. checkout;
2. Python setup;
3. compile;
4. Ruff;
5. Mypy on MVP data/state modules;
6. unit tests.

This job intentionally does not run Mininet because it needs kernel/OVS privileges and the thesis-specific Ryu environment.

### Integration — every push to `dev`

`.github/workflows/integration.yml` targets a self-hosted GCP runner with labels `self-hosted`, `linux`, `x64`, `sdn`. That runner must already have Open vSwitch, Mininet, `~/ryu-venv`, `~/ndt-venv`, passwordless sudo for the required Mininet cleanup/test commands, and no other process occupying ports 6653/6654/8081/8082/9000.

One-time runner setup is done from the GitHub repository UI: `Settings -> Actions -> Runners -> New self-hosted runner`. Select Linux/x64, execute the registration commands GitHub generates for that repository, and add the custom label `sdn`. Do not paste a registration token into the repository or README. Keep the runner service online before pushing to `dev`.

The workflow runs the real 2C-4S smoke test: roles -> traffic -> telemetry -> valid snapshot -> C1->C2 committed migration -> C2->C1 committed migration -> injected C1->C2 failure -> restored rollback snapshot.

### CD — manual only during the MVP

`.github/workflows/deploy.yml` is `workflow_dispatch` only. It runs unit tests and the same real SDN smoke test on the GCP runner. At this phase the "deployment" target is the reproducible thesis testbed, not a production service. Do not auto-deploy every feature branch.

## Definition of Done before starting capacity benchmark

- [ ] `/api/v1/telemetry` works on C1 and C2.
- [ ] Controller has observed/ingested timestamps, CPU, RSS, Packet-In total/rate, processed Packet-In total/rate, Flow-Mod total/rate, response mean/p95.
- [ ] Per-switch Packet-In, processed Packet-In, Flow-Mod and `control_load_share` are present.
- [ ] One unavailable controller does not stop polling of the other controller.
- [ ] `controllers.jsonl` and `switches.jsonl` are written under one `RUN_ID`.
- [ ] Snapshot is created every 5 seconds.
- [ ] `fresh`, `complete`, `consistent`, `valid` are computed.
- [ ] Consistency checks actual OpenFlow role/connectivity against ownership.
- [ ] Pre-migration snapshot is valid.
- [ ] C1 -> C2 or C2 -> C1 migration reaches `COMMITTED`.
- [ ] Post-migration snapshot reflects the new owner and `ownership_version`.
- [ ] Fault injection reaches `RESTORED`.
- [ ] Rollback snapshot reflects the restored owner and is consistent.
- [ ] Unit CI is green.
- [ ] Self-hosted 2C-4S integration workflow is green.
- [ ] Both feature PRs are reviewed and merged into `dev`.

Only after every item above passes should the project move to: workload generator -> controller capacity benchmark -> safe-capacity estimation -> forecasting dataset.

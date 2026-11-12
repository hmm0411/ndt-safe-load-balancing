# Closed loop 2C: how to run it (Week 13, Day 5-6)

Runs only on the VM. Nothing here has been run on the testbed yet: the first pass is
the place to find what the unit tests cannot (workload levels, timing).

## Endpoints (orchestrator, port 9000)

| Endpoint | Effect |
|---|---|
| `POST /api/v1/twin/decision/evaluate` | funnel raw -> eligible -> safe, read-only |
| `POST /api/v1/twin/decision/plan` | evaluate + ranking + selected action, nothing executed |
| `POST /api/v1/twin/decision/execute` | plan, revalidation, then the migration executor |
| `POST /api/v1/twin/decision/kill-switch` | `{"enabled": false}` stops every plan/execute |

Body of plan/execute: `{"scenario": "<label>", "simulate_failure": "none"|"flow_mod"}`.
One decision per snapshot: `decision_id = dec-<mode>-<snapshot_id>`; asking again for
the same snapshot replays the stored decision and never executes twice. Every decision
writes `data/decision_runs/<decision_id>/record.json` (gitignored).

## Decision states

`KILL_SWITCH`, `INVALID_SNAPSHOT`, `NOT_READY`, `NO_RISK`, `DEFERRED` (active transaction
or cooldown), `NO_SAFE_ACTION`, `NO_BENEFICIAL_ACTION`, `SELECTED` (dry run),
`ABORT_STALE_DECISION` (revalidation failed), `COMMITTED`, `RESTORED`, `FAILED`.

## One scenario = one workload + one loop

Terminal A (orchestrator + workload). `run_loop_workload.py` runs the same preflight as
the migration runner, starts the orchestrator of this worktree and one workload, and
stops both when the workload ends. Port 9000 must be free; C1 and C2 must be running.

    python3 scripts/decision/run_loop_workload.py --workload <W> [--level <L>] \
        --dst-level <D> [--duration 150]

Terminal B (loop), started when terminal A prints `workload started`:

    python3 scripts/decision/run_closed_loop.py --scenario <S> --mode <M> --duration <T> ...

| Scenario | Terminal A | Loop arguments |
|---|---|---|
| stable | `--workload stable --dst-level 0.45` | `--duration 60` |
| gradual | `--workload gradual --dst-level 0.45` | `--duration 110` |
| hot_switch | `--workload hot_switch --level peak --dst-level 0.45` | `--duration 110` |
| dst_near_capacity | `--workload gradual --dst-level 0.95` | `--duration 110` |
| no_safe_action | `--workload hot_switch --level peak --dst-level 0.95` | `--duration 110` |
| cooldown | as gradual, `--duration 180` | execute: `--after-commit cooldown` |
| conflict | as gradual | `--duration 8 --interval 0.1 --conflict-migration s1:c2`, started when the source is at risk |
| invalid_snapshot | as stable | pause one controller for ~5 s (below) |
| verification_failure | as gradual | execute: `--simulate-failure flow_mod` |

The workload seed is 9601 (not a seed of any training, calibration or stress set).
Dry run all rows first (`--mode dry_run`). Execute (`--mode execute --wait-post 50`,
terminal A with `--duration 180` so the orchestrator outlives the post window) only
gradual, hot_switch and verification_failure, one at a time.

invalid_snapshot: while the loop runs, `pkill -STOP -f 'ryu-manager.*6654'`, wait 5 s,
`pkill -CONT -f 'ryu-manager.*6654'`. Expected: records with status `INVALID_SNAPSHOT`
or `NOT_READY` and no transaction.

## To confirm on the first pass

- A cell reaches the forecast risk (`at_risk` not empty): if `gradual:u95` only gives
  `NO_RISK`, the source never reaches U + m_f >= 1 and another phase is needed.
- `actual_post` appears in the record about 40 s after a COMMITTED decision
  (`post_settle_seconds` + 5 valid snapshots); the orchestrator must stay up meanwhile.
- Each row starts a fresh orchestrator, so ownership, cooldown and transaction history start clean.

## Analysis

    python3 scripts/decision/analyze_closed_loop.py --runs data/decision_runs

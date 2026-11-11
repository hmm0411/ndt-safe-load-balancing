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

Terminal A (workload). The migration runner holds a workload cell without migrating
(`--action NO_OP`); it is only the load generator here, so its output goes to a scratch
root and its own validation result does not matter (in execute mode the loop changes
ownership, which a NO_OP attempt does not expect):

    python3 scripts/experiments/run_migration_outcome.py --action NO_OP \
        --config <CONFIG> --only <CELL> --output-root data/experiment_runs/_loop-scratch

Terminal B (loop), started when terminal A prints `starting workload`:

    python3 scripts/decision/run_closed_loop.py --scenario <S> --mode <M> --duration <D> ...

| Scenario | CONFIG | CELL | Loop arguments |
|---|---|---|---|
| stable | migration_outcome_v2.yaml | stable:middle:dst45:c1-c2:r01 | `--duration 60` |
| gradual | migration_outcome_v2.yaml | gradual:u95:dst45:c1-c2:r01 | `--duration 100` |
| hot_switch | migration_outcome_v2.yaml | hot_switch:peak:dst45:c1-c2:r01 | `--duration 100` |
| dst_near_capacity | safety_stress_v1.yaml | gradual:u95:dst95:c1-c2:r01 | `--duration 100` |
| no_safe_action | safety_stress_v1.yaml | hot_switch:peak:dst95:c1-c2:r01 | `--duration 100` |
| cooldown | as gradual | | execute: `--after-commit cooldown` |
| conflict | as gradual | | `--duration 8 --interval 0.1 --conflict-migration s1:c2` |
| invalid_snapshot | as stable | | pause one controller for ~5 s (below) |
| verification_failure | as gradual | | execute: `--simulate-failure flow_mod` |

Dry run all rows first (`--mode dry_run`). Execute (`--mode execute --wait-post 50`)
only gradual, hot_switch and verification_failure, one at a time.

invalid_snapshot: while the loop runs, `pkill -STOP -f 'ryu-manager.*6654'`, wait 5 s,
`pkill -CONT -f 'ryu-manager.*6654'`. Expected: records with status `INVALID_SNAPSHOT`
or `NOT_READY` and no transaction.

## To confirm on the first pass

- A cell reaches the forecast risk (`at_risk` not empty): if `gradual:u95` only gives
  `NO_RISK`, the source never reaches U + m_f >= 1 and another phase is needed.
- `actual_post` appears in the record about 40 s after a COMMITTED decision
  (`post_settle_seconds` + 5 valid snapshots); the orchestrator must stay up meanwhile.
- The runner's reset between attempts restores the initial ownership before the next row.

## Analysis

    python3 scripts/decision/analyze_closed_loop.py --runs data/decision_runs

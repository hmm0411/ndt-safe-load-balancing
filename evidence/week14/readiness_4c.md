# 4C readiness audit (Week 13-14 plan, Day 7)

Date: 2026-11-13. Audited commit: `2893d94` (branch `w13/day6-closed-loop-runs`, on top
of dev `18d7cce`). Method: reading the code and configs of the repository. Nothing in
this file was run with four controllers; items that need the VM are marked NOT CHECKED.

Verdict: **not ready to run 4C; no blocker in the decision code.** The decision pipeline
(candidates, Outcome Predictor call, Safety Gate, ranking, closed loop) takes controller
ids from the TwinState and the config. What is missing is infrastructure (topology,
start scripts, config, capacity), the workload generators and the models' 2C assumptions.

## Checklist

| # | Item | Status | Evidence / what is missing |
|---|---|---|---|
| 1 | Ryu controller process for c3, c4 | MISSING (small) | Controller id comes from `CONTROLLER_ID` (`reactive_controller.py`), so the code is generic. Only `scripts/start_c1.sh` and `start_c2.sh` exist. Need `start_c3.sh` (OF 6655 / REST 8083) and `start_c4.sh` (6656 / 8084), or one parametrised script. |
| 2 | Controllers config 4C | MISSING | Only `configs/controllers.yaml` and `controllers_2c20s.yaml`. Need `configs/controllers_4c40s.yaml` (4 controllers, 40 switches, initial ownership). The orchestrator reads it through `NDT_CONTROLLERS_CONFIG`, no code change expected. |
| 3 | Topology 4C40S | MISSING | `src/experiments/topologies/` has only 2C topologies; `runtime_2c20s.py` hard-codes c1/c2 and ports 6653/6654 (lines 142-143, 220-230). Need `runtime_4c40s.py`, with one probe host per switch from the start (Huong 3). |
| 4 | Orchestrator: ownership, roles, transactions | READY (by code) | Built from the controllers config; `init-roles` iterates the config. `src/experiments/migration/schema.py` states the 2C20S/4C40S rule (source/target, never a fixed id). NOT CHECKED on the VM. |
| 5 | Telemetry and TwinState for 4 controllers | READY (by code) | One `ControllerTwinState` per configured controller. Collection latency P95 is already 0.99 s with two controllers (closed-loop runs): must be re-measured with four. NOT CHECKED. |
| 6 | Capacity for c3, c4 | MISSING | `controller_capacity.json` (capacity-v4) has c1, c2 only. capacity-v5 must measure all four with the probe running, and report the latency-vs-U curve per controller. `derive_controller_capacity.py` still has the old-VM constants (140/200/240; capacity-v4 needs 90/140/160) and must be fixed first. |
| 7 | CPU for four controllers | NOT CHECKED | VM is e2-standard-8. capacity-v4 was measured with two Ryu processes; four Ryu + orchestrator + Mininet + generators share 8 vCPU. Check `nproc` and whether capacity runs pin CPUs (no `taskset`/cpuset in the repo). c_safe may change again. |
| 8 | Workload generators | MISSING | The five `scripts/experiments/run_forecast_*_2c20s.py` have `CONTROLLER_IDS = ("c1", "c2")`. Need a 4C variant or a parameter. Also: these workloads keep the source below the risk threshold (see item 14). |
| 9 | Forecast model on c3, c4 | PARTLY | Features are per controller (no controller id feature), so the model can run. The margin m_f falls back to the global value for an unknown controller (`RuntimeForecaster.margin_for`). Transfer to 4C and to probe load is untested: transfer test before any retraining. |
| 10 | Outcome model on 4C | PARTLY | Predicts for (source, target) only; `i_post` is the imbalance of the source/target pair (`targets.py`), which is the system imbalance only with two controllers. The other controllers are not in the prediction. Trained on 10 switches per controller; 4C40S keeps 10 per controller. Transfer test (Analytical / Direct / Hybrid) before retraining. |
| 11 | Candidate Generator and pre-check | READY (by code) | Every switch of an at-risk controller x every other controller; no fixed ids. Candidate count grows from 10 to 30 per at-risk source (10 switches x 3 targets); 4C latency was only benchmarked on synthetic states (`benchmark_decision_latency.py`). |
| 12 | Safety Gate, policy | READY (by code) | Per-candidate checks, no controller ids. Policy v2 needs per-controller latency limits (Huong 3). |
| 13 | Ranking | PARTLY | `global_improvement` already uses all controllers. The `imbalance` component compares only source and target (`ranking.py`); with four controllers it should use the system imbalance. To change in ranking-policy-v2, with a test. |
| 14 | Closed loop | PARTLY | `closed_loop.py` and `DecisionService` have no fixed ids. `PostWindow` records `i = |u_src - u_dst| / 2` (2C definition). Findings of the 2C runs that 4C must address are listed below. |
| 15 | Latency probe per controller (Huong 3) | MISSING | `qos_probe.py` runs one 10-hop path at 1 probe/s, only in the capacity pipeline; successes only (a saturated run reports 0.0). Need: one-switch probes, attribution to the owner at send time, timeouts counted, fields `setup_p95_ms`, `setup_samples`, `setup_timeouts` in telemetry and TwinState. Go/no-go 18/11. |
| 16 | Decision records and analysis | READY | Records store the whole TwinState; `analyze_closed_loop.py` reads controller ids from it. |

## Findings of the closed-loop 2C runs to carry into 4C

Source: `evidence/week14/closed_loop_2c_analysis.json` (10 scenarios, 6 committed
migrations, 1 rollback).

1. **Horizon.** Required lead time 3.11 s (collection P95 0.99 s + decision P95 0.03 s +
   committed-migration P95 2.09 s, n = 5) against h = 3 s: `estimate_horizon` gives 4
   steps. Provisional (5 migrations). Decide h before the forecast model is retrained.
2. **Gate v1 accepted a destination at U = 0.91.** Two migrations in `dst_near_capacity`
   ended with the destination at 1.026 and 1.007 (upper bounds 0.999 and 0.994). The
   loop asks every second, so a small per-decision acceptance rate becomes an executed
   migration within tens of seconds.
3. **Ping-pong.** In the same run the second migration went back from c2 to c1 after the
   cooldown. With total load near total capacity no migration helps; nothing in v1
   detects that. With four controllers there are more destinations, and more ways to
   cycle.
4. **Hard Gate after a transaction.** Snapshots stay invalid until the 60 s twinning-rate
   window recovers (`min_twinning_rate` 0.95, reset only at `init-roles`): 7 decisions
   after a commit, 33 after a rollback in the runs. The loop is fail-closed but blind.
5. **Workloads rarely trigger the loop.** gradual peaks at U = 0.914 on the source (risk
   threshold 0.942): 1 migration in 170 s; hot_switch: none. Evaluation needs overload
   scenarios (`run_loop_workload.py --source-level`).

## To check on the VM before the 4C week (read-only)

    nproc; free -g | head -2
    pgrep -af ryu-manager | wc -l
    ss -ltnp | grep -E ':(6655|6656|8083|8084)' | wc -l     # expected 0: ports are free

## Order of work (16-22/11)

start scripts and config (1, 2) -> topology with probe hosts (3) -> probe and telemetry
fields (15) -> go/no-go 18/11 -> capacity-v5 (6, 7) -> generators (8) -> transfer test
(9, 10) -> 4C outcome data.

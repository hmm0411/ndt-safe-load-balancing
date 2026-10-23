# Pilot report: /home/tntvy112_gmail_com/w11-run/data/experiment_runs/migration-pilot-late

Gate Day 3: **FAIL** enough_attempts=NO, coverage=NO, no_code_bugs=ok, validator_pass=ok, time_known=ok

Attempts 2; status {'COMMITTED': 2}; validator valid 2, invalid 0, eligible 2.

## Decisions

- Collection mode: one attempt per run (p95 estimate fits the budget).
- Estimate for 340 attempts: median 12.9 h, p95 13.6 h (budget 38.4 h).
- post_window_timeout_s: current 60, needed (max settle x headroom) 10, suggested 60, incomplete post windows 0.
- NO_OP controls in the matrix: 40.

## Time (s)

| Measure | n | median | p95 | max |
| --- | --- | --- | --- | --- |
| duration_s | 2 | 128.9 | 129.3 | 129.3 |
| cycle_s | 2 | 136.6 | 144.3 | 144.3 |
| settle_s | 2 | 4.5 | 4.9 | 4.9 |
| transition_hard_gate_blocked | 2 | 1.5 | 2.0 | 2.0 |

## Bugs (block the freeze)

- none

## Look at by hand

- none

## Coverage

Missing: {'workload': ['burst', 'gradual', 'hot_switch', 'oscillating']}; NO_OP attempts: 0.

## Freeze (must be one value each)

- git_commit: ['4e2acf604925f6ab81b70f473a843b1441ef26d4']
- config_sha256: ['068c0f1e5648a9c2bd156a2208358e3cde06ba96314e0b0c9f59fafb02a9b8b5']
- capacity_version: ['controller_capacity.json@sha256:789d2ff9a1f1']
- quality_version: ['quality-src@sha256:a77c36a3b362']
- topology: ['2C20S']

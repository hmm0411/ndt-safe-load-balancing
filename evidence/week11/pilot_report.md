# Pilot report: /home/tntvy112_gmail_com/w11-run/data/experiment_runs/migration-pilot-d120

Gate Day 3: **PASS** enough_attempts=ok, coverage=ok, no_code_bugs=ok, validator_pass=ok, time_known=ok

Attempts 17; status {'COMMITTED': 15, 'None': 2}; validator valid 17, invalid 0, eligible 17.

## Decisions

- Collection mode: one attempt per run (p95 estimate fits the budget).
- Estimate for 340 attempts: median 12.5 h, p95 14.8 h (budget 38.4 h).
- post_window_timeout_s: current 60, needed (max settle x headroom) 60, suggested 60, incomplete post windows 0.
- NO_OP controls in the matrix: 40.

## Time (s)

| Measure | n | median | p95 | max |
| --- | --- | --- | --- | --- |
| duration_s | 17 | 129.2 | 131.9 | 131.9 |
| cycle_s | 17 | 129.6 | 264.5 | 264.5 |
| settle_s | 17 | 13.2 | 39.6 | 39.6 |
| transition_hard_gate_blocked | 17 | 10.0 | 36.0 | 36.0 |

## Bugs (block the freeze)

- none

## Look at by hand

- mig-burst-burst_1-dst45-c1-c2-r01: Delta-U src is 5.42x the switch load
- mig-burst-burst_1-dst75-c1-c2-r01: Delta-U src is 4.62x the switch load
- mig-burst-burst_1-dst75-c2-c1-r01: Delta-U src is 4.43x the switch load
- mig-gradual-u60-dst45-c2-c1-r01: Delta-U src is 0.07x the switch load
- mig-gradual-u60-dst75-c1-c2-r01: Delta-U src has the wrong sign (+0.005)
- mig-gradual-u60-dst75-c2-c1-r01: Delta-U src is 0.03x the switch load
- mig-oscillating-peak-dst45-c1-c2-r01: Delta-U src is 3.26x the switch load
- mig-oscillating-peak-dst45-c1-c2-r01: Delta-U dst is 0.10x the switch load
- mig-oscillating-peak-dst45-c2-c1-r01: Delta-U src is 3.30x the switch load
- mig-oscillating-peak-dst45-c2-c1-r01: Delta-U dst is 0.16x the switch load
- mig-oscillating-peak-dst75-c2-c1-r01: Delta-U src is 3.58x the switch load

## Coverage

Missing: none; NO_OP attempts: 2.

## Freeze (must be one value each)

- git_commit: ['3c712708019df5c8f4018489083c7786d178e9e6']
- config_sha256: ['068c0f1e5648a9c2bd156a2208358e3cde06ba96314e0b0c9f59fafb02a9b8b5']
- capacity_version: ['controller_capacity.json@sha256:789d2ff9a1f1']
- quality_version: ['quality-src@sha256:a77c36a3b362']
- topology: ['2C20S']

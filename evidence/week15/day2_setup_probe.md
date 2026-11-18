# Week 15, Day 2: flow-setup probe on 4C40S

Probe: `p<i>` sends a new UDP flow to `h<i>` on the same switch `s<i>`, so each
flow crosses exactly one switch. It reuses `qos_probe` and `qos_echo`.

## Measured on the VM (2026-11-17, probes on s1, s2, s3, all owned by c1)

Probe rate 2 per second per switch, timeout 1 s.

- Run `w15-day2-smoke`: 3 x 80 samples, all successful,
  `flow_setup_latency_ms` from 2.09 to 18.67. P95 for c1 over the 240 samples:
  6.992667 ms, 0 timeouts.
- Run `w15-day2-push2`: `push_setup_probe.py` with a 10 s window and a 1 s
  period. `/api/v1/twin/state` showed c1 `setup_p95_ms` 4.783541,
  `setup_samples` 60, `setup_timeouts` 0; c2, c3, c4 were None. Four seconds
  after the script stopped, all four were None again (freshness_max_age_ms 2500).
- A first attempt ran the script minutes after the probes had stopped. The
  window was empty and the script posted `{}`, not zeros.

The window, period and timeout above were used only to check the path. They are
not choices for the experiments.

## Rules in the code

- The owner of a sample is the owner when it was sent. A committed migration
  changes the owner at its VERIFYING event, when the target is already MASTER.
- A rolled-back role switch leaves its window out; samples in it are counted.
- A controller without fresh stats keeps None in the twin, never 0.

## Not measured

- Any load (Day 3), any migration while probing, the 4C versus 2C collection
  latency.

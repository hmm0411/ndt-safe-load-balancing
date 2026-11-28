# Scenario B: Full vs Static, Threshold, Forecast-only, NDT no feedback

## M2 overload time (s): confirmatory (Wilcoxon signed-rank, zero pairs dropped, Holm over the 4 comparisons)

Improvement = baseline - Full (positive = Full better).

| vs | pairs | zero pairs (dropped) | median improvement | r_rb | p | p (Holm) |
|---|---|---|---|---|---|---|
| Static | 50 | 13 | 8 | 0.94 | 6.9e-07 | 2.8e-06 |
| Threshold | 50 | 14 | 0 | 0.39 | 0.0391 | 0.0783 |
| Forecast-only | 50 | 12 | 0.5 | 0.59 | 0.0014 | 0.0042 |
| NDT no feedback | 50 | 12 | 0 | -0.18 | 0.3124 | 0.3124 |

## Per workload, M2 overload time (s): median [min-max], seeds won by Full

| workload | Full | Static | Threshold | Forecast-only | NDT no feedback |
|---|---|---|---|---|---|
| stable | 0 [0-1] | 0 [0-1] (Full won 1/10, tied 7) | 0 [0-2] (Full won 1/10, tied 5) | 0 [0-1] (Full won 1/10, tied 6) | 0.5 [0-1] (Full won 4/10, tied 3) |
| gradual | 13 [11-14] | 190 [161-191] (Full won 10/10, tied 0) | 13 [2-16] (Full won 5/10, tied 2) | 14.5 [10-34] (Full won 6/10, tied 1) | 13 [4-16] (Full won 5/10, tied 1) |
| burst | 10 [9-11] | 10 [9-10] (Full won 1/10, tied 6) | 10 [8-11] (Full won 3/10, tied 4) | 10 [9-11] (Full won 4/10, tied 2) | 9 [9-10] (Full won 1/10, tied 3) |
| oscillating | 1.5 [1-2] | 42 [39-45] (Full won 10/10, tied 0) | 2 [0-3] (Full won 4/10, tied 3) | 6 [0-14] (Full won 5/10, tied 3) | 2 [1-3] (Full won 4/10, tied 5) |
| hot_switch | 7.5 [3-15] | 18 [14-24] (Full won 10/10, tied 0) | 16.5 [13-25] (Full won 9/10, tied 0) | 17.5 [13-20] (Full won 9/10, tied 0) | 6 [3-11] (Full won 3/10, tied 0) |

## Secondary metrics (descriptive): median of the five per-workload medians

| metric | Full | Static | Threshold | Forecast-only | NDT no feedback |
|---|---|---|---|---|---|
| M2 overload time (s) | 7.5 | 18 | 10 | 10 | 6 |
| Overload exposure (U-s above 1) | 0.373 | 1.72 | 0.867 | 1 | 0.276 |
| Peak U | 1.13 | 1.21 | 1.21 | 1.18 | 1.13 |
| M3 mean imbalance (std of U) | 0.0595 | 0.102 | 0.059 | 0.0724 | 0.0598 |
| Migrations executed | 2.5 | 0 | 2 | 1 | 2 |
| Unsafe migrations (U_dst post > 1) | 0 | 0 | 0 | 0 | 0 |
| Ping-pong migrations | 0 | 0 | 0 | 0 | 0 |
| M8 decision time P95 (ms) | 8.65 | 0 | 0.023 | 7.7 | 8.63 |

Per-workload median [min-max] of every metric: summary.csv.
Percent improvements are not reported without the absolute values above.

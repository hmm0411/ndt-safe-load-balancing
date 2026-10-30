# forecast-v1 under the capacity-v4 scale (Week 12, Day 2)

Script: `scripts/analysis/forecast_scale_check.py` (output: `forecast_scale_check.txt`, `forecast_scale_check.json`).

## Question

`forecast-v1` (Week 10) was trained on U in capacity-v3 units (1433 pps). The orchestrator now
computes U with capacity-v4 (933 pps), 1.54x larger for the same pps. Is `forecast-v1` still usable?

## What was measured

raw-v2 (340 attempts, 2026-10-28/29) was collected while the orchestrator ran `forecast-v1`
live on the capacity-v4 scale, fed directly (no rescaling). `forecast_samples.jsonl` keeps every
prediction. Target of a prediction at t: measured U of the same controller at t + 3 s (horizon of
the model; READY samples are 1.00 s apart). Metric: RMSE of the model / RMSE of persistence, as in
`skill_vs_persistence.json` (below 1 = better than persistence). Samples where the OOD gate fell
back to persistence are reported apart.

## Result

| Subset | n | RMSE ratio | bias (pred - actual) |
|---|---|---|---|
| all READY (incl. OOD fallback) | 9466 | 1.22 | +0.042 |
| model really predicting | 6532 | 1.32 | +0.064 |
| OOD fallback (= persistence) | 2934 | 1.00 | -0.006 |

Model really predicting, by workload (original = XGBoost on the forecast-v1 test split):

| Workload | n | RMSE ratio | original | bias |
|---|---|---|---|---|
| burst | 987 | 0.57 | 0.53 | -0.004 |
| gradual | 1633 | 1.11 | 0.47 | +0.008 |
| oscillating | 72 | 0.97 | 0.11 | -0.057 |
| stable | 3838 | 2.98 | 0.75 | +0.108 |
| hot_switch | 2 | 1.07 | 2.65 | -0.179 |

By U: below 0.6: 1.65 (n=2648, bias +0.130); 0.6-0.9: 1.36 (n=3510); 0.9 and above: 0.37 (n=374).
MIGRATE 1.30 (n=5771), NO_OP 1.47 (n=761).

Other counts: 17132 NOT_READY (INVALID_SNAPSHOT 9204, INSUFFICIENT_HISTORY 5828, HISTORY_GAP 2100);
2934 of 9466 READY samples with a target (31%) fell back to persistence through the OOD gate.

## Reading

- Fed directly with capacity-v4 U, `forecast-v1` is worse than persistence overall (RMSE ratio
  1.32 where it predicts, 1.22 including the OOD fallbacks). The degradation comes mostly from
  `stable` (n=3838, ratio 2.98, bias +0.108): the load is nearly constant, persistence is almost
  exact and the model predicts a systematic rise.
- `burst` keeps the original skill (0.57 against 0.53), so the model still helps where U moves.
- MIGRATE and NO_OP give the same picture, so the migration is not the cause.
- The OOD gate protects part of the samples (31% are persistence), which is a symptom of the
  scale: capacity-v4 U falls outside the ranges learned with capacity-v3.

## Not answered by this check

- The option "convert U to the 1433 scale, predict, convert back" was not tested: forecast_samples
  keeps predictions, not the 11 feature rows per step.
- The cause of the bias is not identified. Two candidates: the U scale, and the CPU of this VM
  (same pps costs about 16-20% more CPU than on the old VM, and the model uses `process_cpu_percent`
  and `cpu_mean_5`). A scale conversion alone would not remove the second one.
- `hot_switch` (n=2 model samples) and `oscillating` (n=72) are too small for a conclusion.
- Samples of one attempt are autocorrelated; n overstates the independent evidence.
- The artifact has `capacity_version: null` in its model card, so the scale mismatch cannot be
  detected by a check at load time.

## For the advisor (question 2)

Direct use of `forecast-v1` on the capacity-v4 scale is not recommended outside the OOD fallback.
Options: (a) collect forecast data on the current VM with capacity-v4 and retrain (also fixes a CPU
shift); (b) rescale U and test again, which needs the feature rows, i.e. new data anyway.
The outcome artifact of Week 12 records `capacity_version` so this cannot happen silently again.

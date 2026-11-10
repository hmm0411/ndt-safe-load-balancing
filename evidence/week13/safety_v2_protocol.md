# safety-policy-v2 protocol (Week 13)

Written and committed BEFORE any safety-calibration-v1 or safety-stress-v2 data is read.
safety-policy-v1 and its Day 4 evidence (decision-safety-v1) stay untouched; v2 is additive.

## Why
On safety-stress-v1 (evaluation only) the outcome model under-predicted u_dst in 11 of 11
unsafe attempts (actual - pred +0.005..+0.097; mean +0.025 over all 60). Gate B (fixed
m_U = 0.0366) let 2 of 11 through (errors +0.093, +0.072) and cost 8 extra false rejections.

## Data
- safety-calibration-v1 (configs/experiments/safety_calibration_v1.yaml, seed 9401, 120
  attempts): diagnose the residual and fit the fix. Never scores the final Gate.
- safety-stress-v2 (configs/experiments/safety_stress_v2.yaml, seed 9501, 90 attempts):
  evaluation only; may be collected earlier, opened only after the fix is committed, read once.
- safety-stress-v1, test and validation are not used to choose the fix.

## Fix candidates (chosen on calibration by cross-validation grouped by attempt)
(a) retrain the outcome model with the calibration attempts added (new artifact, v1 untouched);
(b) keep the model and use a margin that depends on the predicted u_dst (residual quantile by band).
The choice, the bands and the quantile are fixed and committed before stress-v2 is opened.

## Success criteria on stress-v2 (v2 against v1 Gate B on the same attempts)
1. Unsafe accepted (v2) <= unsafe accepted (v1 Gate B).
2. False rejections (v2) <= false rejections (v1 Gate B).
3. Margin coverage of the residual on stress-v2 >= 95% (the Week 12 target).
Report UCRR, UAR, FRR, SAP with Wilson 95% intervals and counts. If any criterion fails, say
so: safety-policy-v1 stays the reference and the failure is reported as a limitation.

## Not allowed
No threshold, margin or model choice after reading stress-v2. A second look at stress-v2 is a
new set (stress-v3), not a retry.

## Addendum 1: selection rule (written after the descriptive residual table, before any cross-validation)
Descriptive facts only: m_U covers 78% of safety-calibration-v1 (worst at dst98, burst, hot_switch)
and the training split holds 11 of 180 rows with destination utilization >= 0.8
(evidence/week13/safety_calibration_residuals.txt).
Candidates, all compared by 5-fold cross-validation on safety-calibration-v1. Folds group the
attempts by workload:phase:destination level (both directions together); the groups are sorted
alphabetically and group number i goes to fold i mod 5.
- v1B: fixed margin m_U = 0.0366 (baseline, nothing fitted).
- b1: one margin, the one-sided 95% conformal residual quantile of the training folds.
- b2: one such margin per band of measured destination utilization before the migration:
  < 0.85, 0.85 to < 0.95, >= 0.95.
- a: outcome model retrained on raw-v2 train plus the training folds, same conformal margin.
Scored on the held-out folds by the destination-utilization check only (accept when
predicted + margin <= 1.0; unsafe = observed > 1.0), the check the margin changes.
Rule: a candidate qualifies if, summed over the held-out folds, unsafe accepted <= v1B's and
false rejections <= v1B's. Among the qualifying ones take the fewest unsafe accepted, then the
fewest false rejections, then the simplest (b1, b2, a). If none qualifies, v2 keeps v1 and
this is reported as a limitation.
Folds, bands, quantile and rule are fixed here and are not changed after results are seen.

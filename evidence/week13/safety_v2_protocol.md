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

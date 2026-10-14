# Forecast artifact contract (forecast-v1)

Agreed in Week 10 (C1) between the model owner (My) and the runtime owner (Vy).
The training/export scripts write files in exactly this shape; the runtime
predictor (`src/twin/forecasting/predictor.py`) reads them in exactly this shape.
Any change to this file needs a PR reviewed by both.

```
Directory: data/models/forecasting/<model_version>/
The runtime predictor reads ONLY the files marked [runtime].

## model_card.json [runtime]
model_version       "forecast-v1"
model_name          "xgboost" | "random_forest" | "linear_regression" | "persistence"
model_file          "xgboost.json" | "random_forest.joblib" | "linear_regression.joblib"
                    (null for persistence)
scaler_file         "scaler.joblib" or null (only linear_regression)
output              "delta" (prediction = utilization + output) | "level"
dataset_version     "forecast-dataset-v2-h3-provisional"
horizon_steps       3
sampling_interval_seconds 1.0
mlflow_run_id, registered_model, registered_version, git_commit
purpose, limitations[] (must include "not authorized to trigger migration directly")

## feature_schema.json [runtime]
feature_names       ordered list, length 27 (same order as training matrix)
history_samples_required 11   (current sample + 10 lags)
target              "future_utilization"

## error_margin.json [runtime]
method              "validation_abs_error_finite_sample_q95"
alpha               0.05
global              float  (utilization units)
controllers         {"c1": float, "c2": float}
n_validation        int
one_sided_q95       float  (quantile of U - U_hat, reported only)

## feature_ranges.json [runtime, optional]
method              "train_min_max"
tolerance_fraction  0.10
consecutive_steps   3
monitor_only_features ["managed_switch_count", "max_switch_control_load_share"]
                    (watched by the gate, NOT model inputs; absent = [])
features            {name: {"min": float, "max": float}}
                    (feature_names + monitor_only_features)

## Reports (not read at runtime)
model_selection.json, inference_latency.json, coverage.json,
ood_degradation.json, validation/test/unseen/per_workload metrics,
hyperparameters.json, training_manifest.json
```

## Notes

- `history_samples_required = 11`. The features use 10 lags
  (`utilization_lag_1..10`), so the predictor needs the current sample plus the
  10 previous ones. `TwinHistoryBuffer` defaults to `history_steps = 10`; using
  the default would leave `utilization_lag_10` empty.
- `feature_names` must equal `src.twin.forecasting.features.feature_names()`
  element by element. The predictor refuses to load otherwise
  (`FEATURE_SCHEMA_MISMATCH`).
- `output = "delta"`: tree models predict `U(t+h) - U(t)`; the forecast is
  `U(t) + model.predict(x)`. Linear Regression predicts the level directly
  (`output = "level"`), after `scaler.transform`.
- `error_margin.controllers[cid]` is the two-sided finite-sample q95 of `|e|` on
  the validation split, `k = ceil((n + 1) * 0.95)`. The runtime uses it as `m_f`
  in `U_hat + m_f >= 1`; `global` is the fallback for an unknown controller id.
- `feature_ranges.json` drives the OOD gate: a sample is out of range when any
  feature leaves `[min - tol * (max - min), max + tol * (max - min)]`; after
  `consecutive_steps` such samples in a row the runtime falls back to
  persistence (`U_hat = U`, `ood_fallback = true`).
- The gate row is the model vector (feature_schema order) followed by the
  `monitor_only_features` values of the same TwinState controller row
  (`ood.monitored_row`; `forecast_rows` adds `max_switch_control_load_share`).
  These columns never vary in training (switch count 10, load share
  0.100-0.107), so the models do not use them; a change in them means the
  model is outside its training conditions. Watching only the 27 model
  features, the gate never fired on hot-switch (D_OOD = 0, load share ~0.54).
  Side effect, intended: after a migration `managed_switch_count` changes and
  the forecast falls back to persistence until a model is trained with
  migrations.
- JSON files are written with sorted keys, so the key order of
  `feature_ranges.features` is alphabetical and means nothing. Build the
  vector in `feature_schema.feature_names` order
  (`FeatureRanges.from_dict(data, feature_names)` does this).
- The forecast never authorizes a migration by itself. Planner and Safety Gate
  (Week 11) decide; the forecast endpoint returns no migration fields.

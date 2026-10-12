#!/usr/bin/env python3
"""Plot actual vs forecast utilization for one test run per workload.

Shows persistence, Linear Regression, Random Forest and XGBoost against the
actual U(t+h). Test runs are the repeat assigned to the test split; hot-switch
uses its first unseen run.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.twin.forecasting.splitter import (
    DEFAULT_SPLIT_POLICY,
    parse_run_id,
)
from src.twin.forecasting.versions import MODEL_VERSION as VERSIONED_MODEL
from src.twin.forecasting.versions import dataset_version

DATASET_VERSION = dataset_version()

DATA_ROOT = Path("data/forecasting/processed") / DATASET_VERSION
MODEL_ROOT = Path("data/models/forecasting") / VERSIONED_MODEL
PLOT_ROOT = MODEL_ROOT / "plots"

WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot-switch")


def plotted_run(df: pd.DataFrame, workload: str) -> str:
    """The test-split run for seen workloads, the first unseen run otherwise."""
    runs = sorted(df.loc[df["workload_type"] == workload, "run_id"].unique())
    if not runs:
        raise RuntimeError(f"no rows for {workload}")

    if workload in DEFAULT_SPLIT_POLICY.unseen_workloads:
        return runs[0]

    test_runs = [
        run_id
        for run_id in runs
        if parse_run_id(run_id).repeat in DEFAULT_SPLIT_POLICY.test_repeats
    ]
    if not test_runs:
        raise RuntimeError(f"no test-split run for {workload}")
    return test_runs[0]


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_ROOT / "forecast_dataset.csv")
    features = json.loads((MODEL_ROOT / "feature_schema.json").read_text())[
        "feature_names"
    ]

    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    linear = joblib.load(MODEL_ROOT / "linear_regression.joblib")
    forest = joblib.load(MODEL_ROOT / "random_forest.joblib")

    from xgboost import XGBRegressor

    booster = XGBRegressor()
    booster.load_model(str(MODEL_ROOT / "xgboost.json"))

    for workload in WORKLOADS:
        run_id = plotted_run(df, workload)
        run_df = df[df["run_id"] == run_id]
        controller_id = sorted(run_df["controller_id"].unique())[0]

        # Keep the temporal row order stored in the processed dataset.
        plot_df = run_df[run_df["controller_id"] == controller_id].reset_index(
            drop=True
        )

        x_matrix = plot_df[features].to_numpy(dtype=float)
        current = plot_df["utilization"].to_numpy(dtype=float)

        # Tree models predict the change in utilization (see tree_models.py).
        series = {
            "Persistence": current,
            "Linear Regression": linear.predict(scaler.transform(x_matrix)),
            "Random Forest": current + forest.predict(x_matrix),
            "XGBoost": current + booster.predict(x_matrix),
        }

        x = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(
            x,
            plot_df["future_utilization"].to_numpy(),
            label="Actual U(t+h)",
            color="black",
            linewidth=2,
        )
        for label, values in series.items():
            ax.plot(x, values, label=label, linewidth=1.2, alpha=0.85)

        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{workload} - {run_id} - {controller_id}")
        ax.set_xlabel("Sample index (1 s)")
        ax.set_ylabel("Controller utilization")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        output = PLOT_ROOT / f"{workload}_prediction.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print("Wrote:", output)


if __name__ == "__main__":
    main()

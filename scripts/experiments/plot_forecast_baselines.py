#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


DATASET_VERSION = (
    "forecast-dataset-v1-h3-provisional"
)

DATA_ROOT = (
    Path("data/forecasting/processed")
    / DATASET_VERSION
)

MODEL_ROOT = Path(
    "data/models/forecasting/baseline-v1"
)

PLOT_ROOT = MODEL_ROOT / "plots"

WORKLOADS = (
    "stable",
    "gradual",
    "burst",
    "oscillating",
)


def main() -> None:
    PLOT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        DATA_ROOT / "forecast_dataset.csv"
    )

    schema = json.loads(
        (
            MODEL_ROOT
            / "feature_schema.json"
        ).read_text()
    )

    features = schema["feature_names"]

    scaler = joblib.load(
        MODEL_ROOT / "scaler.joblib"
    )

    model = joblib.load(
        MODEL_ROOT
        / "linear_regression.joblib"
    )

    for workload in WORKLOADS:
        workload_df = df[
            (df["workload_type"] == workload)
            & (
                df["run_id"].str.endswith(
                    "-r05"
                )
            )
        ].copy()

        if workload_df.empty:
            raise RuntimeError(
                f"no r05 rows for {workload}"
            )

        controller_id = sorted(
            workload_df[
                "controller_id"
            ].unique()
        )[0]

        plot_df = workload_df[
            workload_df["controller_id"]
            == controller_id
        ].copy()

        # Preserve the temporal row order already stored
        # in the processed forecast dataset. Do not sort
        # by a field that is not part of the processed schema.
        plot_df = plot_df.reset_index(drop=True)

        prediction = model.predict(
            scaler.transform(
                plot_df[
                    features
                ].to_numpy(dtype=float)
            )
        )

        x = range(len(plot_df))

        fig, ax = plt.subplots(
            figsize=(10, 4.8)
        )

        ax.plot(
            x,
            plot_df[
                "future_utilization"
            ].to_numpy(),
            label="Actual U(t+h)",
        )

        ax.plot(
            x,
            prediction,
            label="Linear Regression",
        )

        ax.set_title(
            "Forecast baseline - "
            f"{workload} - {controller_id}"
        )

        ax.set_xlabel(
            "Sample index"
        )

        ax.set_ylabel(
            "Controller utilization"
        )

        ax.legend()

        fig.tight_layout()

        output = (
            PLOT_ROOT
            / f"{workload}_prediction.png"
        )

        fig.savefig(
            output,
            dpi=160,
        )

        plt.close(fig)

        print("Wrote:", output)


if __name__ == "__main__":
    main()

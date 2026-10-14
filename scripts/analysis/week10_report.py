#!/usr/bin/env python3
"""Week 10 report: dataset statistics and figures 1-8 (M5).

Inputs (all already produced by earlier steps, nothing is retrained):
  data/forecasting/processed/<dataset>/forecast_dataset.csv (+ quality report)
  data/models/forecasting/<model_version>/  (metrics, selection, margin, coverage)

Outputs in results/week10/, also logged to the training run in MLflow:
  dataset_stats.json
  fig1_utilization_distribution.png      U per workload
  fig2_utilization_series.png            one sample U series per workload
  fig3_test_rmse_p95.png                 RMSE and P95|e| of 5 models on test
  fig4_rmse_ratio_vs_persistence.png     RMSE / RMSE persistence per workload
  fig5_xgboost_learning_curve.png        validation RMSE per boosting round
  fig6_feature_importance.png            top-15 features, RF and XGB
  fig7_residuals_with_margin.png         test residuals of the deployed model, +/- m_f
  fig8_coverage95.png                    Coverage95 per controller/workload, +/- OOD gate

Figure 9 (runtime) is made by Vy from the V4 data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiments.compute_forecast_margin import load_predictor  # noqa: E402
from src.twin.forecasting.splitter import assign_split  # noqa: E402
from src.twin.forecasting.tracking import log_to_pointed_run  # noqa: E402
from src.twin.forecasting.versions import (  # noqa: E402
    MODEL_VERSION,
    MODELS_ROOT,
    PROCESSED_ROOT,
    dataset_version,
)

WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot-switch")
MODELS = ("persistence", "moving_average_w3", "linear_regression", "random_forest", "xgboost")
LABELS = {
    "persistence": "Persistence",
    "moving_average_w3": "MA3",
    "linear_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
}
COLORS = {
    "persistence": "#7f7f7f",
    "moving_average_w3": "#bcbd22",
    "linear_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "xgboost": "#d62728",
}
SPLIT_ORDER = ("train", "validation", "test", "unseen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--output", type=Path, default=Path("results/week10"))
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def save(fig: Any, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def present(order: tuple[str, ...], values: Any) -> list[str]:
    return [name for name in order if name in set(values)]


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def dataset_stats(df: pd.DataFrame, quality: dict[str, Any] | None) -> dict[str, Any]:
    counts = df.groupby(["workload_type", "split"]).size()
    by_workload = {}
    for workload in present(WORKLOADS, df["workload_type"]):
        u = df.loc[df["workload_type"] == workload, "utilization"]
        by_workload[workload] = {
            "samples": int(u.size),
            "samples_per_split": {
                split: int(counts.get((workload, split), 0)) for split in SPLIT_ORDER
            },
            "utilization_mean": float(u.mean()),
            "utilization_std": float(u.std(ddof=0)),
            "utilization_max": float(u.max()),
            "share_utilization_ge_1": float((u >= 1.0).mean()),
            "share_future_utilization_ge_1": float(
                (df.loc[df["workload_type"] == workload, "future_utilization"] >= 1.0).mean()
            ),
        }
    return {
        "dataset_version": dataset_version(),
        "samples": int(len(df)),
        "samples_per_split": {s: int((df["split"] == s).sum()) for s in SPLIT_ORDER},
        "runs": int(df["run_id"].nunique()),
        "by_workload": by_workload,
        "dropped_for_history": None if quality is None else quality.get("dropped_for_history"),
        "dropped_for_future_label": (
            None if quality is None else quality.get("dropped_for_future_label")
        ),
        "raw_rows": None if quality is None else quality.get("raw_rows"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig1_distribution(df: pd.DataFrame, out: Path) -> None:
    workloads = present(WORKLOADS, df["workload_type"])
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.boxplot(
        [df.loc[df["workload_type"] == w, "utilization"] for w in workloads],
        showfliers=False,
        medianprops={"color": "black"},
    )
    ax.set_xticks(range(1, len(workloads) + 1), workloads)
    ax.axhline(1.0, color="#d62728", linewidth=1, linestyle="--", label="U = 1")
    ax.set_ylabel("Utilization U")
    ax.set_title("Utilization per workload (all runs, whiskers 1.5 IQR)")
    ax.legend(loc="upper left")
    save(fig, out / "fig1_utilization_distribution.png")


def fig2_series(df: pd.DataFrame, out: Path) -> None:
    workloads = present(WORKLOADS, df["workload_type"])
    fig, axes = plt.subplots(len(workloads), 1, figsize=(6.0, 1.4 * len(workloads)),
                             sharex=True, constrained_layout=True)
    for ax, workload in zip(np.atleast_1d(axes), workloads, strict=True):
        part = df[df["workload_type"] == workload]
        run_id = sorted(part["run_id"].unique())[0]
        for cid, color in (("c1", "#1f77b4"), ("c2", "#ff7f0e")):
            series = part[(part["run_id"] == run_id) & (part["controller_id"] == cid)]
            if series.empty:
                continue
            t = (series["observed_at"] - series["observed_at"].min()).dt.total_seconds()
            ax.plot(t, series["utilization"], color=color, linewidth=0.9, label=cid)
        ax.axhline(1.0, color="#d62728", linewidth=0.8, linestyle="--")
        ax.set_ylabel("U")
        ax.set_title(f"{workload} ({run_id})", loc="left")
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Time since first sample [s]")
    save(fig, out / "fig2_utilization_series.png")


def fig3_test_errors(test_metrics: dict[str, Any], out: Path) -> None:
    models = present(MODELS, test_metrics)
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, key, title in ((axes[0], "rmse", "RMSE"), (axes[1], "p95_absolute_error", "P95 |e|")):
        values = [test_metrics[m][key] for m in models]
        ax.bar(x, values, color=[COLORS[m] for m in models], width=0.6)
        for xi, value in zip(x, values, strict=True):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [LABELS[m] for m in models])
        ax.set_title(f"{title} on test (repeat 5)")
    axes[0].set_ylabel("Utilization units")
    save(fig, out / "fig3_test_rmse_p95.png")


def fig4_skill(skill: dict[str, Any], out: Path) -> None:
    rows = [(split, w) for split in ("test", "unseen") for w in present(WORKLOADS, skill[split])]
    models = [m for m in MODELS if m != "persistence"
              and any(m in skill[s][w] for s, w in rows)]
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, model in enumerate(models):
        ax.bar(
            [j + i * width for j in range(len(rows))],
            [skill[s][w].get(model, np.nan) for s, w in rows],
            width=width, color=COLORS[model], label=LABELS[model],
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(rows))],
                  [w if s == "test" else f"{w}\n(unseen)" for s, w in rows])
    ax.set_ylabel("RMSE / RMSE persistence")
    ax.set_title("Below 1 = better than persistence", pad=20)
    ax.legend(ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out / "fig4_rmse_ratio_vs_persistence.png")


def fig5_learning_curve(model_dir: Path, validation: list[dict[str, Any]],
                        feature_names: list[str], out: Path) -> None:
    from xgboost import XGBRegressor

    from src.twin.forecasting.linear_model import records_to_matrix

    model = XGBRegressor()
    model.load_model(str(model_dir / "xgboost.json"))
    x = records_to_matrix(validation, feature_names)
    u = np.asarray([float(r["utilization"]) for r in validation])
    y = np.asarray([float(r["future_utilization"]) for r in validation])
    rounds = model.get_booster().num_boosted_rounds()
    steps = sorted(set(np.linspace(1, rounds, num=min(rounds, 60), dtype=int).tolist()))
    rmse = [float(np.sqrt(np.mean((y - u - model.predict(x, iteration_range=(0, s))) ** 2)))
            for s in steps]
    persistence = float(np.sqrt(np.mean((y - u) ** 2)))
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.plot(steps, rmse, color=COLORS["xgboost"], linewidth=1.2, label="XGB (selected config)")
    ax.axhline(persistence, color=COLORS["persistence"], linestyle="--", linewidth=1,
               label="Persistence")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("XGBoost learning curve (validation, repeat 4)")
    ax.legend()
    save(fig, out / "fig5_xgboost_learning_curve.png")


def fig6_importance(importance: dict[str, dict[str, float]], out: Path, top: int = 15) -> None:
    models = [m for m in ("random_forest", "xgboost") if importance.get(m)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6))
    for ax, model in zip(np.atleast_1d(axes), models, strict=True):
        items = sorted(importance[model].items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
        ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS[model])
        ax.set_title(f"{LABELS[model]}: top-{top} importance")
        ax.tick_params(axis="y", labelsize=7)
    save(fig, out / "fig6_feature_importance.png")


def fig7_residuals(residuals: np.ndarray, margin: dict[str, Any], model_name: str,
                   out: Path) -> None:
    m = float(margin["global"])
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.hist(residuals, bins=60, color="#9e9e9e", edgecolor="white", linewidth=0.3)
    for sign in (-1, 1):
        ax.axvline(sign * m, color="#d62728", linewidth=1, linestyle="--")
    inside = float(np.mean(np.abs(residuals) <= m))
    ax.set_xlabel("Residual U(t+h) - U_hat [utilization]")
    ax.set_ylabel("Samples")
    ax.set_title(f"{LABELS.get(model_name, model_name)} on test: m_f = {m:.3f}, "
                 f"{inside:.1%} inside")
    save(fig, out / "fig7_residuals_with_margin.png")


def fig8_coverage(coverage: dict[str, Any], out: Path) -> None:
    splits = coverage["splits"]
    groups: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for split in ("test", "unseen"):
        if split not in splits:
            continue
        for kind in ("by_controller", "by_workload"):
            for label in splits[split]["without_gate"][kind]:
                name = label if split == "test" else f"{label}\n(unseen)"
                groups.append((name, splits[split]["without_gate"][kind][label],
                               splits[split]["with_gate"][kind][label]))
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar(x - 0.2, [g[1]["coverage"] for g in groups], width=0.4, color="#9e9e9e",
           label="without OOD gate")
    ax.bar(x + 0.2, [g[2]["coverage"] for g in groups], width=0.4, color="#1f77b4",
           label="with OOD gate")
    ax.axhline(0.95, color="#d62728", linewidth=1, linestyle="--", label="target 0.95")
    ax.set_xticks(x, [g[0] for g in groups], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage95")
    ax.set_title("Share of samples with |U - U_hat| <= m_f (per controller, per workload)",
                 pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, out / "fig8_coverage95.png")


def main() -> None:
    args = parse_args()
    style()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_ROOT / args.model_version
    data_dir = PROCESSED_ROOT / dataset_version()

    df = pd.read_csv(data_dir / "forecast_dataset.csv")
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    df["split"] = df["run_id"].map(assign_split)
    quality_path = data_dir / "dataset_quality_report.json"
    stats = dataset_stats(df, read_json(quality_path) if quality_path.is_file() else None)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    feature_names = list(read_json(model_dir / "feature_schema.json")["feature_names"])
    records = df.assign(observed_at=df["observed_at"].astype(str)).to_dict("records")
    validation = [r for r in records if r["split"] == "validation"]
    test = [r for r in records if r["split"] == "test"]

    fig1_distribution(df, out)
    fig2_series(df, out)
    fig3_test_errors(read_json(model_dir / "test_metrics.json"), out)
    fig4_skill(read_json(model_dir / "skill_vs_persistence.json"), out)
    fig5_learning_curve(model_dir, validation, feature_names, out)
    fig6_importance(read_json(model_dir / "feature_importance.json"), out)

    margin = read_json(model_dir / "error_margin.json")
    model_name = str(margin["model_name"])
    predictor = load_predictor(model_dir, model_name, feature_names)
    residuals = np.asarray([float(r["future_utilization"]) for r in test]) - predictor(test)
    fig7_residuals(residuals, margin, model_name, out)
    fig8_coverage(read_json(model_dir / "coverage.json"), out)

    for name in ("model_selection.json", "coverage.json", "ood_degradation.json",
                 "error_margin.json"):
        source = model_dir / name
        if source.is_file():
            (out / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    latency = Path("data/benchmarks/forecast_inference_latency.json")
    if latency.is_file():
        (out / latency.name).write_text(latency.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.no_mlflow:
        run_id = log_to_pointed_run(model_dir / "mlflow_run.json", artifacts=[(out, "week10")])
        if run_id:
            print("MLflow run", run_id, "artifacts -> week10/")


if __name__ == "__main__":
    main()

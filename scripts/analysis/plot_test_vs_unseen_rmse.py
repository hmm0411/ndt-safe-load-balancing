#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def find_artifact(filename: str) -> Path:
    candidates = list(ROOT.rglob(filename))

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy {filename} trong repository."
        )

    # Ưu tiên artifact baseline-v1 do MLflow lưu.
    preferred = [
        p for p in candidates
        if "mlruns" in str(p) and "baseline-v1" in str(p)
    ]

    path = preferred[0] if preferred else candidates[0]

    print(f"Using {filename}: {path.relative_to(ROOT)}")
    return path


test_path = find_artifact("test_metrics.json")
unseen_path = find_artifact("unseen_metrics.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


test_data = load_json(test_path)
unseen_data = load_json(unseen_path)


def normalize(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_model(data, possible_names):
    normalized = {
        normalize(k): v
        for k, v in data.items()
    }

    # Exact normalized match.
    for name in possible_names:
        key = normalize(name)
        if key in normalized:
            return normalized[key]

    # Fallback partial match.
    for key, value in normalized.items():
        for name in possible_names:
            if normalize(name) in key:
                return value

    raise KeyError(
        f"Không tìm thấy {possible_names}. "
        f"Available keys: {list(data.keys())}"
    )


def get_rmse(record):
    if isinstance(record, dict):
        if "rmse" in record:
            return float(record["rmse"])

        if "metrics" in record and "rmse" in record["metrics"]:
            return float(record["metrics"]["rmse"])

    raise KeyError(
        f"Không đọc được RMSE từ record: {record}"
    )


model_specs = [
    (
        "Persistence",
        ["persistence"],
    ),
    (
        "Moving Average (w=3)",
        [
            "moving_average_w3",
            "moving_average_3",
            "ma3",
            "moving_average",
        ],
    ),
    (
        "Linear Regression",
        [
            "linear_regression",
            "linear regression",
            "lr",
        ],
    ),
]


models = []
test_rmse = []
unseen_rmse = []

for display_name, aliases in model_specs:
    test_record = find_model(test_data, aliases)
    unseen_record = find_model(unseen_data, aliases)

    models.append(display_name)
    test_rmse.append(get_rmse(test_record))
    unseen_rmse.append(get_rmse(unseen_record))


print("\n=== TEST vs UNSEEN HOT-SWITCH ===")

for model, test_value, unseen_value in zip(
    models,
    test_rmse,
    unseen_rmse,
):
    ratio = unseen_value / test_value

    print(
        f"{model:22s} "
        f"Test={test_value:.6f}  "
        f"Unseen={unseen_value:.6f}  "
        f"Ratio={ratio:.2f}x"
    )


# ============================================================
# PLOT
# ============================================================

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

bars_test = ax.bar(
    x - width / 2,
    test_rmse,
    width,
    label="Test (in-distribution)",
)

bars_unseen = ax.bar(
    x + width / 2,
    unseen_rmse,
    width,
    label="Unseen hot-switch (OOD)",
)

ax.set_title(
    "In-Distribution Test vs Unseen Hot-Switch",
    fontsize=14,
    fontweight="bold",
)

ax.set_ylabel("RMSE (lower is better)")
ax.set_xlabel("Forecasting Model")

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(
    0,
    max(max(test_rmse), max(unseen_rmse)) * 1.18,
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

ax.set_axisbelow(True)
ax.legend()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(unseen_rmse) * 0.018,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


add_labels(bars_test)
add_labels(bars_unseen)


# Highlight LR degradation numerically.
lr_test = test_rmse[2]
lr_unseen = unseen_rmse[2]
lr_ratio = lr_unseen / lr_test

ax.annotate(
    f"LR OOD degradation\n{lr_ratio:.1f}x RMSE",
    xy=(x[2] + width / 2, lr_unseen),
    xytext=(x[1] + 0.45, lr_unseen * 0.78),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2,
    ),
    fontsize=10,
    ha="center",
)


fig.text(
    0.5,
    0.01,
    (
        "Hot-switch is held out from training, validation, "
        "and the normal test split."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])


# ============================================================
# SAVE
# ============================================================

output_dir = ROOT / "evidence" / "week9" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "test_vs_unseen_hot_switch_rmse.png"
pdf_path = output_dir / "test_vs_unseen_hot_switch_rmse.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(fig)

print("\n=== OUTPUT ===")
print(png_path)
print(pdf_path)

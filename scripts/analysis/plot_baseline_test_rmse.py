#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

# Tìm test_metrics.json từ các artifact thực nghiệm hiện có.
candidates = list(ROOT.rglob("test_metrics.json"))

if not candidates:
    raise FileNotFoundError(
        "Không tìm thấy test_metrics.json trong repository."
    )

# Ưu tiên artifact baseline-v1 của MLflow.
preferred = [
    p for p in candidates
    if "baseline-v1" in str(p) and "mlruns" in str(p)
]

metrics_path = preferred[0] if preferred else candidates[0]

print(f"Using metrics: {metrics_path.relative_to(ROOT)}")

with metrics_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("\n=== RAW METRICS ===")
print(json.dumps(data, indent=2))

# Hỗ trợ schema:
# {
#   "persistence": {"rmse": ...},
#   "moving_average_w3": {"rmse": ...},
#   "linear_regression": {"rmse": ...}
# }
#
# hoặc tên model viết hoa/khác nhẹ.

def normalize(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


normalized = {normalize(k): v for k, v in data.items()}


def find_model(possible_names):
    for name in possible_names:
        key = normalize(name)
        if key in normalized:
            return normalized[key]

    for key, value in normalized.items():
        for name in possible_names:
            if normalize(name) in key:
                return value

    raise KeyError(
        f"Không tìm thấy model {possible_names}. "
        f"Available keys: {list(data.keys())}"
    )


persistence = find_model([
    "persistence",
])

ma3 = find_model([
    "moving_average_w3",
    "moving_average_3",
    "ma3",
    "moving_average",
])

lr = find_model([
    "linear_regression",
    "linear regression",
    "lr",
])


def get_rmse(record):
    if isinstance(record, dict):
        if "rmse" in record:
            return float(record["rmse"])

        if "metrics" in record and "rmse" in record["metrics"]:
            return float(record["metrics"]["rmse"])

    raise KeyError(f"Không đọc được RMSE từ record: {record}")


rmse = {
    "Linear Regression": get_rmse(lr),
    "Persistence": get_rmse(persistence),
    "Moving Average (w=3)": get_rmse(ma3),
}

print("\n=== TEST RMSE ===")
for model, value in rmse.items():
    print(f"{model:22s}: {value:.6f}")

# Sort từ RMSE thấp -> cao để biểu đồ dễ đọc.
ordered = sorted(rmse.items(), key=lambda x: x[1])

models = [x[0] for x in ordered]
values = [x[1] for x in ordered]

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(models, values)

ax.set_title(
    "Baseline Forecasting Performance on Test Set",
    fontsize=14,
    fontweight="bold",
)

ax.set_ylabel("RMSE (lower is better)")
ax.set_xlabel("Forecasting Model")
ax.set_ylim(0, max(values) * 1.25)

ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Hiện giá trị chính xác trên đầu bar.
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(values) * 0.025,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Chú thích để tránh hiểu nhầm đây là final model selection.
fig.text(
    0.5,
    0.01,
    "Week 9 baseline evaluation — in-distribution test set",
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

output_dir = ROOT / "evidence" / "week9" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "baseline_test_rmse.png"
pdf_path = output_dir / "baseline_test_rmse.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print("\n=== OUTPUT ===")
print(png_path)
print(pdf_path)

plt.close(fig)

"""Final evaluation of the outcome models (Week 12, Day 4).

Pure functions over dataset rows and prediction tables, used by
``train_outcome_models.py --stage select-and-test``:

* metrics per group (workload, destination level, direction, switch share, saturation);
* saturation sensitivity (all rows, without saturated attempts, saturated only);
* overload cases (every row with ``overload_after``);
* paired bootstrap of RMSE and of the improvements of plan section 58,
  ``improvement(ref, model) = (RMSE_ref - RMSE_model) / RMSE_ref * 100``;
* the persistence base used by the "no switch" ablation.

``Predictions`` maps a model name to target -> one value per row, in row order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.twin.outcome.features import CANDIDATE_FEATURES
from src.twin.outcome.metrics import regression_metrics
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS

Rows = Sequence[Mapping[str, Any]]
Predictions = Mapping[str, Mapping[str, Sequence[float]]]
GROUPABLE = ("workload", "phase", "destination_load_level", "direction", "switch_share_group",
             "saturated_before", "saturated_after", "target_hot")


@dataclass(frozen=True)
class AblationSpec:
    drop: tuple[str, ...]
    persistence_base: bool


@dataclass(frozen=True)
class EvaluationConfig:
    direct_candidates: tuple[str, ...]
    hybrid_candidates: tuple[str, ...]
    group_keys: tuple[str, ...]
    resamples: int
    seed: int
    confidence: float
    ablation: dict[str, AblationSpec]

    def __post_init__(self) -> None:
        if not self.direct_candidates or not all(
                c in ("direct_rf", "direct_xgb") for c in self.direct_candidates):
            raise ValueError("candidates.direct must list direct_rf and/or direct_xgb")
        if not self.hybrid_candidates or not all(
                c in ("hybrid_rf", "hybrid_xgb") for c in self.hybrid_candidates):
            raise ValueError("candidates.hybrid must list hybrid_rf and/or hybrid_xgb")
        bad = [k for k in self.group_keys if k not in GROUPABLE]
        if bad:
            raise ValueError(f"group_keys not groupable: {bad}")
        if self.resamples < 100:
            raise ValueError("bootstrap.resamples must be >= 100")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("bootstrap.confidence must be in (0.5, 1)")
        for name, spec in self.ablation.items():
            unknown = [f for f in spec.drop if f not in CANDIDATE_FEATURES]
            if unknown:
                raise ValueError(f"ablation {name}: unknown features {unknown}")
            if not spec.drop and not spec.persistence_base:
                raise ValueError(f"ablation {name} removes nothing")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        boot = dict(data["bootstrap"])
        return EvaluationConfig(
            direct_candidates=tuple(str(c) for c in data["candidates"]["direct"]),
            hybrid_candidates=tuple(str(c) for c in data["candidates"]["hybrid"]),
            group_keys=tuple(str(k) for k in data["group_keys"]),
            resamples=int(boot["resamples"]),
            seed=int(boot["seed"]),
            confidence=float(boot["confidence"]),
            ablation={str(name): AblationSpec(drop=tuple(str(f) for f in spec.get("drop") or ()),
                                              persistence_base=bool(spec.get("persistence_base")))
                      for name, spec in dict(data.get("ablation") or {}).items()},
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing or malformed key {exc}") from exc


def choose(scores: Mapping[str, float], candidates: Sequence[str]) -> str:
    """Candidate with the lowest score; ties keep the candidate order."""
    missing = [c for c in candidates if c not in scores]
    if missing:
        raise ValueError(f"no score for {missing}")
    best = candidates[0]
    for name in candidates[1:]:
        if scores[name] < scores[best]:
            best = name
    return best


def improvement(reference_rmse: float, model_rmse: float) -> float:
    """Plan section 58, in percent; positive = the model is better than the reference."""
    if reference_rmse <= 0:
        return float("nan")
    return (reference_rmse - model_rmse) / reference_rmse * 100.0


def _metrics(rows: Rows, values: Sequence[float], target: str) -> dict[str, float]:
    m = regression_metrics([float(r[target]) for r in rows], list(values))
    return {k: m[k] for k in ("n", "mae", "rmse", "p95_abs_error", "bias")}


def metrics_rows(split: str, group: str, value: str, rows: Rows, index: Sequence[int],
                 predictions: Predictions) -> list[dict[str, Any]]:
    members = [rows[i] for i in index]
    out = []
    for model, by_target in predictions.items():
        for target in ALL_TARGETS:
            values = [by_target[target][i] for i in index]
            out.append({"split": split, "group": group, "value": value, "model": model,
                        "target": target, **_metrics(members, values, target)})
    return out


def group_metrics(rows: Rows, predictions: Predictions, split: str,
                  group_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Metrics of every model on all rows and on each value of each group key."""
    out = metrics_rows(split, "all", "all", rows, range(len(rows)), predictions)
    for key in group_keys:
        for value in sorted({str(r[key]) for r in rows}):
            index = [i for i, r in enumerate(rows) if str(r[key]) == value]
            out += metrics_rows(split, key, value, rows, index, predictions)
    return out


SATURATION_SUBSETS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "all": lambda r: True,
    "without_saturated": lambda r: not (r["saturated_before"] or r["saturated_after"]),
    "saturated_only": lambda r: bool(r["saturated_before"] or r["saturated_after"]),
}


def saturation_sensitivity(rows: Rows, predictions: Predictions,
                           split: str) -> list[dict[str, Any]]:
    """Metrics with and without the attempts where a controller saturated (CPU p95 >= 95 %)."""
    out = []
    for subset, keep in SATURATION_SUBSETS.items():
        index = [i for i, r in enumerate(rows) if keep(r)]
        if index:
            out += metrics_rows(split, "saturation", subset, rows, index, predictions)
    return out


def overload_cases(rows: Rows, predictions: Predictions, split: str,
                   threshold: float) -> list[dict[str, Any]]:
    """One record per row whose destination ended overloaded (u_dst_post >= threshold)."""
    out = []
    for i, row in enumerate(rows):
        if not row["overload_after"]:
            continue
        record: dict[str, Any] = {
            "split": split, "attempt_id": row["attempt_id"], "workload": row["workload"],
            "phase": row["phase"], "direction": row["direction"],
            "destination_load_level": row["destination_load_level"],
            "switch_share_group": row["switch_share_group"],
            "saturated_before": row["saturated_before"],
            "saturated_after": row["saturated_after"],
            "dst_u_mean_pre": row["dst_u_mean"], "u_dst_post": row["u_dst_post"],
        }
        for model, by_target in predictions.items():
            pred = float(by_target["u_dst_post"][i])
            record[f"pred_{model}"] = pred
            record[f"error_{model}"] = pred - float(row["u_dst_post"])
            record[f"flags_overload_{model}"] = pred >= threshold
        out.append(record)
    return out


def persistence_base(rows: Rows) -> dict[str, list[float]]:
    """Base prediction without the moved switch: U_post = mean U of the pre window,
    response = persistence (as in the Analytical model)."""
    return {
        "u_src_post": [float(r["src_u_mean"]) for r in rows],
        "u_dst_post": [float(r["dst_u_mean"]) for r in rows],
        "r_src95_post": [float(r["analytical_r_src95_post"]) for r in rows],
        "r_dst95_post": [float(r["analytical_r_dst95_post"]) for r in rows],
    }


def with_base(rows: Rows, base: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    """Copies of the rows whose analytical_<base target> columns hold another base."""
    out = []
    for i, row in enumerate(rows):
        copy = dict(row)
        for target in BASE_TARGETS:
            copy[f"analytical_{target}"] = float(base[target][i])
        out.append(copy)
    return out


# ------------------------------------------------------------------ bootstrap
def _percentile_rows(values: np.ndarray, q: float) -> np.ndarray:
    return np.percentile(values, q * 100.0, axis=1)


def _scores(truth: np.ndarray, pred: np.ndarray, weights: Mapping[str, float],
            targets: Sequence[str], eps: float) -> np.ndarray:
    """Selection score per resample; truth / pred have shape (targets, resamples, n)."""
    score = np.zeros(truth.shape[1])
    for k, target in enumerate(targets):
        if target not in weights:
            continue
        rmse = np.sqrt(np.mean((pred[k] - truth[k]) ** 2, axis=1))
        spread = _percentile_rows(truth[k], 0.95) - _percentile_rows(truth[k], 0.05)
        score += weights[target] * rmse / (spread + eps)
    return score


def paired_bootstrap(
    rows: Rows,
    predictions: Predictions,
    comparisons: Sequence[tuple[str, str]],
    weights: Mapping[str, float],
    eps: float,
    resamples: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """RMSE of each model and improvement(reference, model) for each (reference, model),
    over the same resamples of rows (paired). Target ``S`` is the selection score."""
    n = len(rows)
    if n < 2:
        raise ValueError("bootstrap needs at least 2 rows")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(resamples, n))
    targets = list(ALL_TARGETS)
    full_truth = np.asarray([[float(r[t]) for r in rows] for t in targets])
    truth = full_truth[:, index]
    stats: dict[str, dict[str, np.ndarray]] = {}
    point: dict[str, dict[str, float]] = {}
    for model, by_target in predictions.items():
        pred_full = np.asarray([[float(v) for v in by_target[t]] for t in targets])
        if pred_full.shape != full_truth.shape:
            raise ValueError(f"{model}: predictions do not match the rows")
        pred = pred_full[:, index]
        stats[model] = {t: np.sqrt(np.mean((pred[k] - truth[k]) ** 2, axis=1))
                        for k, t in enumerate(targets)}
        stats[model]["S"] = _scores(truth, pred, weights, targets, eps)
        point[model] = {t: float(np.sqrt(np.mean((pred_full[k] - full_truth[k]) ** 2)))
                        for k, t in enumerate(targets)}
        point[model]["S"] = float(_scores(full_truth[:, None, :], pred_full[:, None, :],
                                          weights, targets, eps)[0])
    low_q, high_q = (1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0
    out: list[dict[str, Any]] = []
    for model in predictions:
        for target in targets + ["S"]:
            values = stats[model][target]
            out.append({"kind": "rmse" if target != "S" else "score", "model": model,
                        "reference": "", "target": target, "estimate": point[model][target],
                        "ci_low": float(np.quantile(values, low_q)),
                        "ci_high": float(np.quantile(values, high_q)), "share_better": ""})
    for reference, model in comparisons:
        for target in targets + ["S"]:
            ref, mod = stats[reference][target], stats[model][target]
            with np.errstate(divide="ignore", invalid="ignore"):
                imp = np.where(ref > 0, (ref - mod) / ref * 100.0, np.nan)
            finite = imp[np.isfinite(imp)]
            out.append({
                "kind": "improvement", "model": model, "reference": reference, "target": target,
                "estimate": improvement(point[reference][target], point[model][target]),
                "ci_low": float(np.quantile(finite, low_q)) if finite.size else math.nan,
                "ci_high": float(np.quantile(finite, high_q)) if finite.size else math.nan,
                "share_better": float(np.mean(mod < ref)),
            })
    return out

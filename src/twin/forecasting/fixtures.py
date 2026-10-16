"""Synthetic TwinState dicts and a persistence artifact for forecaster tests.

Shared by the predictor, service and replay tests (and handy for a smoke run
without a testbed). Not used by the runtime itself.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.twin.forecasting.artifact import build_feature_schema, build_model_card
from src.twin.forecasting.features import feature_names
from src.twin.forecasting.ood import FeatureRanges, monitored_names

T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)
MARGINS = {"c1": 0.048, "c2": 0.055}


def twin_state(
    second: float,
    u: tuple[float, float] = (0.6, 0.5),
    *,
    valid: bool = True,
    share: float = 0.1,
    switch_count: int = 10,
    capacity: float = 1433.0,
) -> dict[str, Any]:
    controllers = []
    switches = []
    ownership = []
    for index, (cid, util) in enumerate(zip(("c1", "c2"), u, strict=True)):
        controllers.append({
            "controller_id": cid,
            "processed_packet_in_rate": util * capacity,
            "safe_capacity_pps": capacity,
            "utilization": util,
            "collection_latency_ms": 40.0,
            "flow_mod_rate": 50.0,
            "process_cpu_percent": 40.0,
            "process_memory_rss_mb": 120.0,
            "response_p95_ms": 3.0,
            "managed_switch_count": switch_count,
        })
        for k in range(switch_count):
            sid = f"s{index * switch_count + k + 1}"
            switches.append({"switch_id": sid, "controller_id": cid,
                             "control_load_share": share if k == 0 else 0.1})
            ownership.append({"switch_id": sid, "owner_controller_id": cid})
    return {
        "snapshot_id": f"snap-{second}",
        "created_at": (T0 + timedelta(seconds=second)).isoformat(),
        "controllers": controllers,
        "switches": switches,
        "ownership": ownership,
        "quality": {"valid": valid, "quality_score": 0.97, "consistency_score": 1.0},
    }


def write_persistence_artifact(directory: Path) -> None:
    """A contract-C1 artifact whose model is persistence: U_hat = U."""
    names = list(feature_names())
    card = build_model_card(
        model_name="persistence", model_version=directory.name,
        dataset_version="test", horizon_steps=3, sampling_interval_seconds=1.0,
        mlflow_run_id=None, registered_model=None, registered_version=None,
        git_commit="test", dataset_sha256="0" * 64, selection_score=None,
    )
    margin = {
        "method": "validation_abs_error_finite_sample_q95", "alpha": 0.05,
        "global": 0.05, "controllers": MARGINS, "n_validation": 100,
        "one_sided_q95": 0.03, "model_name": "persistence",
    }
    watched = list(monitored_names(names))
    wide = {n: (-1e9, 1e9) for n in watched}
    wide["max_switch_control_load_share"] = (0.100, 0.107)
    wide["managed_switch_count"] = (10.0, 10.0)
    ranges = FeatureRanges(
        tuple(watched), tuple(wide[n][0] for n in watched), tuple(wide[n][1] for n in watched),
        0.10, 3, tuple(watched[len(names):]),
    )
    for name, data in (
        ("model_card.json", card),
        ("feature_schema.json", build_feature_schema(names)),
        ("error_margin.json", margin),
        ("feature_ranges.json", ranges.to_dict()),
    ):
        (directory / name).write_text(json.dumps(data), encoding="utf-8")

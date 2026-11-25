"""Feedback for the two margins the decision uses (thesis loop step 6).

* ForecastFeedback: per controller, the forecast made h steps ago is compared with the
  utilization measured now (error = U(t) - U_hat(t | t-h)); the controller's m_f is then
  updated (online_margin.MarginTracker). A gap (a NOT_READY step) drops the pending
  forecasts of that controller, so an error is never matched to the wrong step.
* OutcomeMargin: what the Safety Gate sees as m_U. ``fixed`` = the artifact's margin,
  ``zero`` = 0 (ablation), ``adaptive`` = updated after each committed migration from
  actual U_dst - predicted U_dst.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.twin.calibration.online_margin import MarginTracker, OnlineMargin, TrackerConfig

FIXED, ADAPTIVE, ZERO = "fixed", "adaptive", "zero"
MODES = (FIXED, ADAPTIVE, ZERO)


class ForecastFeedback:
    def __init__(self, initial: Mapping[str, float], default: float, horizon_steps: int,
                 mode: str, config: TrackerConfig) -> None:
        if mode not in MODES:
            raise ValueError(f"margin mode {mode!r} not in {MODES}")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.mode = mode
        self.horizon_steps = horizon_steps
        self.tracker = MarginTracker(initial, default, config, enabled=mode == ADAPTIVE)
        self._pending: dict[str, deque[float]] = {}

    def margin(self, controller_id: str) -> float:
        return 0.0 if self.mode == ZERO else self.tracker.value(controller_id)

    def observe(self, predictions: Iterable[Any]) -> None:
        """Feed the predictions of one snapshot (objects with controller_id, status,
        current_utilization, predicted_utilization); call after every forecast update."""
        for p in predictions:
            cid = str(p.controller_id)
            if p.status != "READY" or p.current_utilization is None \
                    or p.predicted_utilization is None:
                self._pending.pop(cid, None)
                continue
            queue = self._pending.setdefault(cid, deque())
            if len(queue) == self.horizon_steps:
                self.tracker.update(cid, float(p.current_utilization) - queue.popleft())
            queue.append(float(p.predicted_utilization))

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "horizon_steps": self.horizon_steps,
                "margins": self.tracker.to_dict()}


class OutcomeMargin:
    """Wraps the outcome predictor: READY predictions carry the margin of this mode."""

    def __init__(self, predictor: Any, mode: str, config: TrackerConfig,
                 initial: float | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"margin mode {mode!r} not in {MODES}")
        self.predictor = predictor
        self.mode = mode
        self.config = config
        self._initial = initial
        self.margin: OnlineMargin | None = None

    def _ensure(self, artifact_margin: float) -> OnlineMargin:
        if self.margin is None:
            start = artifact_margin if self._initial is None else self._initial
            self.margin = OnlineMargin(start, self.config, enabled=self.mode == ADAPTIVE)
        return self.margin

    def _adjust(self, prediction: Any) -> Any:
        if getattr(prediction, "status", None) != "READY" or self.mode == FIXED:
            return prediction
        if self.mode == ZERO:
            return dataclasses.replace(prediction, dst_utilization_margin=0.0,
                                       response_margin_ms=0.0)
        tracked = self._ensure(float(prediction.dst_utilization_margin))
        return dataclasses.replace(prediction, dst_utilization_margin=tracked.value)

    def predict_batch(self, twin_state: Mapping[str, Any], actions: Sequence[Any],
                      observe: bool = True) -> list[Any]:
        return [self._adjust(p) for p in
                self.predictor.predict_batch(twin_state, actions, observe=observe)]

    def observe_outcome(self, predicted_dst: float | None,
                        actual_dst: float | None) -> bool | None:
        """One committed migration's observed U_dst (adaptive mode); None when it cannot be
        scored or no READY prediction has set the starting margin yet."""
        if self.mode != ADAPTIVE or self.margin is None or predicted_dst is None \
                or actual_dst is None:
            return None
        return self.margin.update(float(actual_dst) - float(predicted_dst))

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode,
                "margin": None if self.margin is None else self.margin.to_dict()}

    def __getattr__(self, name: str) -> Any:   # everything else is the wrapped predictor's
        return getattr(self.predictor, name)

"""Runtime post-migration what-if service inside the orchestrator (Week 12, Day 4).

build_snapshot() hands every new TwinState to OutcomeService.on_twin_state(), which only
feeds the 5-snapshot history. POST /api/v1/twin/outcome/predict returns
OutcomeService.predict(): a pure what-if. It never creates a transaction, never changes
ownership and never sends a role request; the Safety Gate decides in Week 13.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.experiments.migration.schema import MigrationAction
from src.orchestrator.current_state import CurrentStateStore
from src.twin.outcome.predictor import PostMigrationOutcomePredictor


def capacity_version(path: Path) -> str:
    """Same format as the dataset and the training manifest: <file>@sha256:<12 hex>."""
    return f"{path.name}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


class OutcomeService:
    def __init__(self, predictor: PostMigrationOutcomePredictor, store: CurrentStateStore) -> None:
        self.predictor = predictor
        self.store = store
        self.last_error: str | None = None

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], root: Path, store: CurrentStateStore
    ) -> OutcomeService:
        running: str | None = None
        try:
            running = capacity_version(root / str(config["capacity_file"]))
        except (OSError, KeyError):
            pass  # unknown capacity version: every prediction stays NOT_READY
        artifact = root / str(config["artifact_dir"]) if config.get("artifact_dir") else None
        return cls(PostMigrationOutcomePredictor(artifact, running), store)

    def on_twin_state(self, twin_state: Mapping[str, Any]) -> None:
        """Feed the history. A problem here is logged and kept out of the telemetry path."""
        try:
            self.predictor.observe(twin_state)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[outcome] error={exc}", flush=True)

    def predict(
        self, switch_id: str, source_controller: str, target_controller: str
    ) -> dict[str, Any]:
        """Body of POST /api/v1/twin/outcome/predict. ValueError for a malformed action."""
        action = MigrationAction(
            action_type="MIGRATE",
            source_controller=source_controller,
            target_controller=target_controller,
            switch_id=switch_id,
        )
        twin_state = self.store.twin_state_dict() or {}  # {} -> NOT_READY / INVALID_SNAPSHOT
        prediction = self.predictor.predict(twin_state, action, observe=False)
        return dataclasses.asdict(prediction)

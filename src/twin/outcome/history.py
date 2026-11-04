"""Last pre-window snapshots of the runtime what-if predictor (Week 12, Day 3).

``build_features`` needs whole snapshots (controllers, switches, ownership, quality),
while ``TwinHistoryBuffer`` keeps one row per controller. The buffer is reused as it
is, with ONE key for the whole snapshot, so the continuity rules stay identical to the
forecaster and to the dataset: an invalid snapshot is never stored, a duplicate is
rejected, a gap longer than the buffer limit or a backwards timestamp clears the
history. Nothing is stitched across a hole.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

from src.experiments.migration.schema import StateSnapshotView
from src.twin.forecasting.history import AppendStatus, TwinHistoryBuffer

WINDOW_SNAPSHOTS = 5
_KEY = "twin"


class OutcomeHistory:
    def __init__(self, window: int = WINDOW_SNAPSHOTS, safety_buffer: int = 5) -> None:
        self.window_size = window
        self._buffer = TwinHistoryBuffer(history_steps=window, safety_buffer=safety_buffer)

    def observe(self, twin_state: Mapping[str, Any]) -> AppendStatus | None:
        """Store one TwinState (dict of GET /api/v1/twin/state).

        None when the payload holds no snapshot (``{"status": "NOT_READY"}``). Raises
        ValueError when the snapshot is malformed: it is not stored.
        """
        if not isinstance(twin_state.get("controllers"), list):
            return None
        created_at = twin_state.get("created_at")
        snapshot_id = str(twin_state.get("snapshot_id", ""))
        if created_at is None or not snapshot_id:
            raise ValueError("twin_state needs created_at and snapshot_id")
        quality = twin_state.get("quality")
        valid = isinstance(quality, Mapping) and quality.get("valid") is True
        if not valid:
            return self._buffer.append(_KEY, created_at, snapshot_id, {}, snapshot_valid=False)
        view = StateSnapshotView.from_dict(twin_state)
        return self._buffer.append(_KEY, created_at, snapshot_id, {"view": view})

    def size(self) -> int:
        return self._buffer.size(_KEY)

    def ready(self) -> bool:
        return self._buffer.ready(_KEY)

    def window(self) -> list[StateSnapshotView]:
        """The latest ``window_size`` consecutive snapshots, oldest first."""
        entries = self._buffer.get_history(_KEY)
        return [cast(StateSnapshotView, entry.row["view"]) for entry in entries]

    def clear(self) -> None:
        self._buffer.clear(_KEY)

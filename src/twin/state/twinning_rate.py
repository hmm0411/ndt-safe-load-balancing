from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any


class TwinningRateTracker:
    def __init__(
        self,
        window_seconds: float,
    ) -> None:

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be > 0"
            )

        self.window_seconds = float(
            window_seconds
        )

        self._attempts: deque[tuple[datetime, bool]] = deque()
        # Snapshots left out of the rate because a migration was changing their roles
        # (migration_transient); counted for the audit, never a success or a failure.
        self.skipped = 0

    def skip(self) -> None:
        """A snapshot that neither succeeds nor fails (see migration_transient)."""
        self.skipped += 1

    def record(
        self,
        success: bool,
        observed_at: datetime | None = None,
    ) -> None:

        now = (
            observed_at
            or datetime.now(timezone.utc)
        )

        self._attempts.append(
            (
                now,
                bool(success),
            )
        )

        self._prune(now)

    def _prune(
        self,
        now: datetime,
    ) -> None:

        threshold = (
            now
            - timedelta(
                seconds=self.window_seconds
            )
        )

        while (
            self._attempts
            and self._attempts[0][0]
            < threshold
        ):
            self._attempts.popleft()

    def reset(self) -> None:
        """Clear observations from the current twinning-rate window."""
        self._attempts.clear()
        self.skipped = 0

    def rate(
        self,
        now: datetime | None = None,
    ) -> float:

        current = (
            now
            or datetime.now(timezone.utc)
        )

        self._prune(current)

        if not self._attempts:
            return 0.0

        # Count the number of successes in the current window
        successes = sum(
            1
            for _, success
            in self._attempts
            if success
        )

        return (
            successes
            / len(self._attempts)
        )

# Conflicts a role change produces by itself: while (and just after) a transaction moves a
# switch, the controllers' observed roles and the committed ownership disagree for a poll or
# two. "disconnected", "invalid_owner" or missing role state stay failures (faults D3/D6).
TRANSIENT_CONFLICTS = ("role_mismatch",)


def migration_transient(quality: Any, switches: Iterable[str]) -> bool:
    """True when an invalid snapshot's only problems are role mismatches on switches that a
    migration transaction is moving (decided 26/11, My: such a snapshot stays invalid for
    the Hard Gate but is left out of the twinning rate, so one migration no longer blocks
    the twin for the ~40 snapshots the 0.95 rate needs to recover)."""
    moving = set(switches)
    conflicts = list(getattr(quality, "conflicts", []) or [])
    if (getattr(quality, "valid", True) or not moving or not conflicts
            or getattr(quality, "stale_sources", None) or getattr(quality, "missing_fields", None)):
        return False
    for conflict in conflicts:
        kind, _, rest = str(conflict).partition(":")
        if kind not in TRANSIENT_CONFLICTS or rest.split(":", 1)[0] not in moving:
            return False
    return True

from __future__ import annotations

from src.schemas.snapshot import (
    NetworkSnapshot,
)
from src.schemas.twin import (
    TwinQuality,
)
from src.twin.state.quality_score import quality_components, quality_score


class TwinQualityAssessor:
    def __init__(
        self,
        expected_controller_ids: set[str],
        expected_switch_ids: set[str],
        max_age_ms: float,
        min_completeness_ratio: float,
        min_twinning_rate: float,
        twinning_rate_provider,
        quality_config=None,
    ) -> None:
        # Validated configs/quality.yaml (quality_score.load_quality_config);
        # None = no Quality Score, the Hard Gate works as before.
        self.quality_config = quality_config

        self.expected_controller_ids = (
            set(expected_controller_ids)
        )

        self.expected_switch_ids = (
            set(expected_switch_ids)
        )

        self.max_age_ms = float(
            max_age_ms
        )

        self.min_completeness_ratio = (
            float(
                min_completeness_ratio
            )
        )

        self.min_twinning_rate = float(
            min_twinning_rate
        )

        self.twinning_rate_provider = (
            twinning_rate_provider
        )

    def _age_of_twin_ms(
        self,
        snapshot: NetworkSnapshot,
    ) -> float:

        if not snapshot.controllers:
            return float("inf")

        return max(
            max(
                0.0,
                (
                    snapshot.created_at
                    - item.observed_at
                ).total_seconds()
                * 1000.0,
            )
            for item
            in snapshot.controllers
        )

    def _sync_jitter_ms(
        self,
        snapshot: NetworkSnapshot,
    ) -> float:

        timestamps = [
            item.observed_at
            for item
            in snapshot.controllers
        ]

        if len(timestamps) < 2:
            return 0.0

        return (
            max(timestamps)
            - min(timestamps)
        ).total_seconds() * 1000.0

    def _completeness(
        self,
        snapshot: NetworkSnapshot,
    ) -> float:

        controller_ids = {
            item.controller_id
            for item
            in snapshot.controllers
        }

        switch_ids = {
            item.switch_id
            for item
            in snapshot.switches
        }

        ownership_ids = {
            item.switch_id
            for item
            in snapshot.ownership
        }

        controllers_present = len(
            controller_ids
            & self.expected_controller_ids
        )

        switches_present = len(
            switch_ids
            & self.expected_switch_ids
        )

        ownership_present = len(
            ownership_ids
            & self.expected_switch_ids
        )

        expected_total = (
            len(
                self.expected_controller_ids
            )
            + 2
            * len(
                self.expected_switch_ids
            )
        )

        if expected_total == 0:
            return 0.0

        return (
            controllers_present
            + switches_present
            + ownership_present
        ) / expected_total

    def assess(
        self,
        snapshot: NetworkSnapshot,
    ) -> TwinQuality:

        age = self._age_of_twin_ms(
            snapshot
        )

        completeness = (
            self._completeness(
                snapshot
            )
        )

        jitter = self._sync_jitter_ms(
            snapshot
        )

        twinning_rate = float(
            self.twinning_rate_provider()
        )

        consistent = (
            snapshot.quality.consistent
        )

        valid = (
            snapshot.quality.valid
            and age <= self.max_age_ms
            and completeness
            >= self.min_completeness_ratio
            and twinning_rate
            >= self.min_twinning_rate
        )

        consistency_score = float(
            getattr(snapshot.quality, "consistency_score", 1.0)
        )
        score = None
        components: dict[str, float] = {}
        if self.quality_config is not None:
            components = quality_components(
                consistency_score=consistency_score,
                completeness_ratio=completeness,
                age_ms=age,
                jitter_ms=jitter,
                collection_ms=self._collection_ms(snapshot),
                twinning_rate=twinning_rate,
                thresholds=self.quality_config["thresholds"],
            )
            score = quality_score(components, self.quality_config["weights"])

        return TwinQuality(
            age_of_twin_ms=age,
            twinning_rate=twinning_rate,
            completeness_ratio=(
                completeness
            ),
            synchronization_jitter_ms=(
                jitter
            ),
            consistent=consistent,
            valid=valid,
            consistency_score=consistency_score,
            quality_score=score,
            quality_components=components,
        )

    def _collection_ms(
        self,
        snapshot: NetworkSnapshot,
    ) -> float:
        """Slowest controller collection: ingested_at - observed_at."""
        if not snapshot.controllers:
            return float("inf")
        return max(
            max(0.0, (item.ingested_at - item.observed_at).total_seconds() * 1000.0)
            for item in snapshot.controllers
        )
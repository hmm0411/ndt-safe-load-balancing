from __future__ import annotations

from typing import Callable

from src.schemas.snapshot import (
    NetworkSnapshot,
)
from src.schemas.twin import (
    ControllerTwinState,
    TwinQuality,
    TwinState,
)
from src.twin.capacity.model import (
    CapacityModel,
)

# QualityProvider is a callable that takes a NetworkSnapshot and returns a TwinQuality.
# it is dependency-injected into the TwinStateBuilder to allow for different implementations of quality calculation.

QualityProvider = Callable[
    [NetworkSnapshot],
    TwinQuality,
]


class TwinStateBuilder:
    def __init__(
        self,
        capacity_model: CapacityModel,
        quality_provider: QualityProvider,
    ) -> None:
        self.capacity_model = (
            capacity_model
        )

        self.quality_provider = (
            quality_provider
        )

    def build(
        self,
        snapshot: NetworkSnapshot,
    ) -> TwinState:

        controllers = []

        for sample in snapshot.controllers:
            capacity = (
                self.capacity_model
                .safe_capacity(
                    sample.controller_id
                )
            )

            utilization = (
                self.capacity_model
                .utilization(
                    sample.controller_id,
                    sample
                    .processed_packet_in_rate,
                )
            )

            collection_latency_ms = (
                sample.ingested_at
                - sample.observed_at
            ).total_seconds() * 1000.0

            controllers.append(
                ControllerTwinState(
                    controller_id=(
                        sample.controller_id
                    ),

                    processed_packet_in_rate=(
                        sample
                        .processed_packet_in_rate
                    ),

                    safe_capacity_pps=(
                        capacity
                    ),

                    utilization=(
                        utilization
                    ),

                    collection_latency_ms=max(
                        0.0,
                        collection_latency_ms,
                    ),

                    flow_mod_rate=(
                        sample.flow_mod_rate
                    ),

                    process_cpu_percent=(
                        sample
                        .process_cpu_percent
                    ),

                    process_memory_rss_mb=(
                        sample
                        .process_memory_rss_mb
                    ),

                    response_p95_ms=(
                        sample.response_p95_ms
                    ),

                    managed_switch_count=(
                        sample
                        .managed_switch_count
                    ),

                    setup_p95_ms=sample.setup_p95_ms,

                    setup_samples=sample.setup_samples,

                    setup_timeouts=sample.setup_timeouts,
                )
            )

        return TwinState(
            snapshot_id=(
                snapshot.snapshot_id
            ),

            created_at=(
                snapshot.created_at
            ),

            topology_version=(
                snapshot.topology_version
            ),

            ownership_version=(
                snapshot.ownership_version
            ),

            controllers=controllers,

            switches=list(
                snapshot.switches
            ),

            ownership=list(
                snapshot.ownership
            ),

            quality=(
                self.quality_provider(
                    snapshot
                )
            ),
        )
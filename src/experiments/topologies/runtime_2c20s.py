#!/usr/bin/env python3

import argparse
from pathlib import Path

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

from src.experiments.topologies.capacity_2c20s import (
    Capacity2C20STopo,
)
from src.experiments.topologies.forecast_hot_switch_2c20s import (
    ForecastHotSwitch2C20STopo,
)
from src.experiments.topologies.capacity_smoke_2c20s import (
    post_json,
    validate_initialized_ownership,
    wait_for_controller_connections,
)
from src.experiments.workloads.profile_spec import parse_profile_specs
from scripts.experiments.run_forecast_stable_2c20s import (
    run_forecast_stable,
)
from scripts.experiments.run_forecast_gradual_2c20s import (
    run_forecast_gradual,
)
from scripts.experiments.run_forecast_burst_2c20s import (
    run_forecast_burst,
)
from scripts.experiments.run_forecast_oscillating_2c20s import (
    run_forecast_oscillating,
)
from scripts.experiments.run_forecast_hot_switch_2c20s import (
    run_forecast_hot_switch,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-hot-switch",
        action="store_true",
        help=(
            "run the hot-switch forecast workload "
            "instead of opening the Mininet CLI"
        ),
    )

    parser.add_argument(
        "--forecast-oscillating",
        action="store_true",
        help=(
            "run the oscillating forecast workload "
            "instead of opening the Mininet CLI"
        ),
    )

    parser.add_argument(
        "--forecast-burst",
        action="store_true",
        help=(
            "run the burst forecast workload instead "
            "of opening the Mininet CLI"
        ),
    )

    parser.add_argument(
        "--forecast-gradual",
        action="store_true",
        help=(
            "run the gradual forecast workload instead "
            "of opening the Mininet CLI"
        ),
    )

    parser.add_argument(
        "--forecast-stable",
        action="store_true",
        help="run the stable forecast smoke instead of opening the Mininet CLI",
    )
    parser.add_argument(
        "--capacity-runs",
        type=Path,
        help="benchmark CSV required by --forecast-stable",
    )
    parser.add_argument(
        "--controller-capacity",
        type=Path,
        default=Path("data/benchmarks/controller_capacity.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiment_runs/forecasting"),
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "deterministic experiment run ID; "
            "if omitted, the workload runner generates a smoke UUID"
        ),
    )
    parser.add_argument("--experiment-type", default=None)
    parser.add_argument("--topology-name", default=None)
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--capacity-artifact", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--target-utilization", type=float, default=0.60)
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        metavar="CONTROLLER=PROFILE",
        help="per-controller load profile, e.g. --profile c1=stable --profile c2=constant:0.45",
    )
    parser.add_argument(
        "--orchestrator-url",
        default="http://127.0.0.1:9000",
    )
    return parser



def initialize_forecast_runtime(
    *,
    orchestrator_url: str,
) -> None:
    """Prepare the 2C20S control plane for any forecast workload."""
    print(
        "*** Waiting for all 20 switch-controller connections"
    )

    wait_for_controller_connections(
        controller_urls={
            "c1": "http://127.0.0.1:8081",
            "c2": "http://127.0.0.1:8082",
        },
        expected_switch_count=20,
    )

    print("CONTROLLER_CONNECTIONS: PASS")

    base_url = orchestrator_url.rstrip("/")

    print("*** Initializing OpenFlow roles")

    roles = post_json(
        f"{base_url}/api/v1/init-roles"
    )

    validate_initialized_ownership(
        roles.get("ownership")
    )

    print(
        "*** INIT_ROLES:",
        roles.get("ownership"),
        flush=True,
    )

    print("*** Roles initialized successfully")

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    profile_modes = (
        args.forecast_stable,
        args.forecast_gradual,
        args.forecast_burst,
        args.forecast_oscillating,
    )
    if args.profile and not (any(profile_modes) or args.forecast_hot_switch):
        parser.error("--profile is not supported for this workload yet")
    forecast_modes = (
        args.forecast_stable,
        args.forecast_gradual,
        args.forecast_burst,
        args.forecast_oscillating,
        args.forecast_hot_switch,
    )

    if (
        any(forecast_modes)
        and args.capacity_runs is None
    ):
        raise SystemExit(
            "--capacity-runs is required with forecast workloads"
        )

    if sum(bool(mode) for mode in forecast_modes) > 1:
        raise SystemExit(
            "choose only one forecast workload"
        )

    setLogLevel("info")

    topology = (
        ForecastHotSwitch2C20STopo()
        if args.forecast_hot_switch
        else Capacity2C20STopo()
    )

    net = Mininet(
        topo=topology,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        build=False,
    )

    # External Ryu controllers.
    net.addController(
        "c1",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653,
    )

    net.addController(
        "c2",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6654,
    )

    try:
        print("*** Building 2C20S topology")
        net.build()

        print("*** Starting network")
        net.start()

        print()
        print("*** 2C20S runtime topology READY")
        print("*** Controllers:")
        print("***   C1 -> 127.0.0.1:6653")
        print("***   C2 -> 127.0.0.1:6654")
        print()

        if (
            args.forecast_stable
            or args.forecast_gradual
            or args.forecast_burst
            or args.forecast_oscillating
            or args.forecast_hot_switch
        ):
            initialize_forecast_runtime(
                orchestrator_url=args.orchestrator_url,
            )

        if args.forecast_hot_switch:
            print(
                "*** Starting hot-switch "
                "forecast workload"
            )

            run_forecast_hot_switch(
                net,
                capacity_runs_path=(
                    args.capacity_runs
                ),
                controller_capacity_path=(
                    args.controller_capacity
                ),
                output_dir=args.output_dir,
                duration_seconds=args.duration,
                sampling_interval_seconds=(
                    args.sample_interval
                ),
                orchestrator_url=(
                    args.orchestrator_url
                ),
                run_id=args.run_id,
                experiment_type=args.experiment_type,
                topology=args.topology_name,
                repeat_index=args.repeat_index,
                seed=args.seed,
                capacity_artifact=args.capacity_artifact,
                git_commit=args.git_commit,
                profiles=parse_profile_specs(args.profile),
            )

        elif args.forecast_oscillating:
            print(
                "*** Starting oscillating "
                "forecast workload"
            )

            run_forecast_oscillating(
                net,
                profiles=parse_profile_specs(args.profile),
                capacity_runs_path=(
                    args.capacity_runs
                ),
                controller_capacity_path=(
                    args.controller_capacity
                ),
                output_dir=args.output_dir,
                duration_seconds=args.duration,
                sampling_interval_seconds=(
                    args.sample_interval
                ),
                orchestrator_url=(
                    args.orchestrator_url
                ),
                run_id=args.run_id,
                experiment_type=args.experiment_type,
                topology=args.topology_name,
                repeat_index=args.repeat_index,
                seed=args.seed,
                capacity_artifact=args.capacity_artifact,
                git_commit=args.git_commit,
            )

        elif args.forecast_burst:
            print(
                "*** Starting burst forecast workload"
            )

            run_forecast_burst(
                net,
                profiles=parse_profile_specs(args.profile),
                capacity_runs_path=args.capacity_runs,
                controller_capacity_path=(
                    args.controller_capacity
                ),
                output_dir=args.output_dir,
                duration_seconds=args.duration,
                sampling_interval_seconds=(
                    args.sample_interval
                ),
                orchestrator_url=(
                    args.orchestrator_url
                ),
                run_id=args.run_id,
                experiment_type=args.experiment_type,
                topology=args.topology_name,
                repeat_index=args.repeat_index,
                seed=args.seed,
                capacity_artifact=args.capacity_artifact,
                git_commit=args.git_commit,
            )

        elif args.forecast_gradual:
            print(
                "*** Starting gradual forecast workload"
            )

            run_forecast_gradual(
                net,
                profiles=parse_profile_specs(args.profile),
                capacity_runs_path=args.capacity_runs,
                controller_capacity_path=(
                    args.controller_capacity
                ),
                output_dir=args.output_dir,
                duration_seconds=args.duration,
                sampling_interval_seconds=(
                    args.sample_interval
                ),
                orchestrator_url=(
                    args.orchestrator_url
                ),
                run_id=args.run_id,
                experiment_type=args.experiment_type,
                topology=args.topology_name,
                repeat_index=args.repeat_index,
                seed=args.seed,
                capacity_artifact=args.capacity_artifact,
                git_commit=args.git_commit,
            )

        elif args.forecast_stable:
            print(
                "*** Starting stable forecast workload"
            )

            run_forecast_stable(
                net,
                capacity_runs_path=args.capacity_runs,
                controller_capacity_path=(
                    args.controller_capacity
                ),
                output_dir=args.output_dir,
                duration_seconds=args.duration,
                sampling_interval_seconds=(
                    args.sample_interval
                ),
                target_utilization=(
                    args.target_utilization
                ),
                profiles=parse_profile_specs(args.profile),
                orchestrator_url=(
                    args.orchestrator_url
                ),
                run_id=args.run_id,
                experiment_type=args.experiment_type,
                topology=args.topology_name,
                repeat_index=args.repeat_index,
                seed=args.seed,
                capacity_artifact=args.capacity_artifact,
                git_commit=args.git_commit,
            )

        else:
            CLI(net)

    finally:
        print("*** Stopping network")
        net.stop()


if __name__ == "__main__":
    main()
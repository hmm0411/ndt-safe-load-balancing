from __future__ import annotations
import argparse
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from src.schemas.workload import WorkloadSample

DEFAULT_SOURCE_PORT_START = 12000
DEFAULT_SOURCE_PORT_END = 65535
NANOSECONDS_PER_SECOND = 1_000_000_000
LATE_EVENT_TOLERANCE_NS = 1_000_000

class PortAllocator:
    def __init__(self, start: int, end: int) -> None:
        if not 1 <= start <= 65535:
            raise ValueError(f"invalid source port start: {start}")
        if not 1 <= end <= 65535:
            raise ValueError(f"invalid source port end: {end}")
        if start > end:
            raise ValueError("source port start must be <= source port end")

        self.start = start
        self.end = end
        self.current = start

    def next_port(self) -> int:
        port = self.current
        self.current = self.start if self.current == self.end else self.current + 1
        return port

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP new-flow workload generator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-ip", required=True)
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--source-port-start", type=int, default=DEFAULT_SOURCE_PORT_START)
    parser.add_argument("--source-port-end", type=int, default=DEFAULT_SOURCE_PORT_END)
    parser.add_argument("--output", type=Path, required=True)
    return parser

def validate_config(args: argparse.Namespace) -> None:
    if args.rate <= 0:
        raise ValueError("rate must be > 0")
    if args.duration <= 0:
        raise ValueError("duration must be > 0")
    if not 1 <= args.target_port <= 65535:
        raise ValueError(f"invalid target port: {args.target_port}")
    if not 1 <= args.source_port_start <= 65535:
        raise ValueError(f"invalid source port start: {args.source_port_start}")
    if not 1 <= args.source_port_end <= 65535:
        raise ValueError(f"invalid source port end: {args.source_port_end}")
    if args.source_port_start > args.source_port_end:
        raise ValueError("source port start must be <= source port end")

def sleep_until_ns(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        time.sleep(remaining_ns / NANOSECONDS_PER_SECOND)

def send_udp_flow(source_ip: str, source_port: int, target_ip: str, target_port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((source_ip, source_port))
        sock.sendto(b"x", (target_ip, target_port))

def run_generator(args: argparse.Namespace) -> WorkloadSample:
    validate_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    allocator = PortAllocator(args.source_port_start, args.source_port_end)
    interval_ns = NANOSECONDS_PER_SECOND / args.rate
    event_count = max(1, math.ceil(args.rate * args.duration))

    start_ns = time.monotonic_ns()
    end_ns = start_ns + int(args.duration * NANOSECONDS_PER_SECOND)
    attempted_flows = 0
    emitted_flows = 0
    send_errors = 0
    late_events = 0
    max_schedule_lag_ns = 0
    first_source_port = None
    last_source_port = None

    with args.output.open("w", encoding="utf-8") as output_file:
        for index in range(event_count):
            deadline_ns = start_ns + int(index * interval_ns)
            if deadline_ns >= end_ns:
                break
            sleep_until_ns(deadline_ns)
            actual_start_ns = time.monotonic_ns()
            schedule_lag_ns = max(0, actual_start_ns - deadline_ns)

            if schedule_lag_ns > LATE_EVENT_TOLERANCE_NS:
                late_events += 1
            max_schedule_lag_ns = max(max_schedule_lag_ns, schedule_lag_ns)
            source_port = allocator.next_port()
            attempted_flows += 1
            if first_source_port is None:
                first_source_port = source_port
            last_source_port = source_port

            flow_record = {
                "run_id": args.run_id,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "source_host": args.source_host,
                "source_ip": args.source_ip,
                "source_port": source_port,
                "target_ip": args.target_ip,
                "target_port": args.target_port,
                "scheduled_at_ns": deadline_ns,
                "started_at_ns": actual_start_ns,
            }

            try:
                send_udp_flow(args.source_ip, source_port, args.target_ip, args.target_port)
                emitted_flows += 1
                flow_record["status"] = "emitted"
                flow_record["emitted_at_ns"] = time.monotonic_ns()
            except OSError as exc:
                send_errors += 1
                flow_record["status"] = "send_error"
                flow_record["error"] = str(exc)
                flow_record["emitted_at_ns"] = None

            output_file.write(json.dumps(flow_record) + "\n")
            output_file.flush()

    actual_elapsed_seconds = (time.monotonic_ns() - start_ns) / NANOSECONDS_PER_SECOND
    elapsed_seconds = max(args.duration, actual_elapsed_seconds)
    sample = WorkloadSample(
        run_id=args.run_id,
        observed_at=datetime.now(timezone.utc),
        source_host=args.source_host,
        source_ip=args.source_ip,
        target_host=args.target_ip,
        target_ip=args.target_ip,
        target_port=args.target_port,
        protocol="UDP",
        pattern="new-flow",
        target_new_flow_rate=args.rate,
        emitted_new_flow_rate=emitted_flows / elapsed_seconds,
        interval_seconds=elapsed_seconds,
        attempted_flows=attempted_flows,
        emitted_flows=emitted_flows,
        send_errors=send_errors,
        late_events=late_events,
        max_schedule_lag_ms=max_schedule_lag_ns / 1_000_000.0,
        cumulative_attempted_flows=attempted_flows,
        cumulative_emitted_flows=emitted_flows,
        cumulative_send_errors=send_errors,
        first_source_port=first_source_port,
        last_source_port=last_source_port,
    )
    with args.output.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(sample.to_dict()) + "\n")

    return sample

def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        sample = run_generator(args)
        print(json.dumps(sample.to_dict()))
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(f"UDP new-flow generator failed: {exc}") from exc

if __name__ == "__main__":
    main()
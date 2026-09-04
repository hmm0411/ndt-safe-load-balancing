from __future__ import annotations
import argparse
import socket

DEFAULT_PORT = 9000
RECV_BUFFER_SIZE = 65535

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP sink for workload experiments")
    parser.add_argument("--bind-ip", required=True, help="IP address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port to bind")
    return parser

def run_sink(bind_ip: str, port: int) -> int:
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid UDP port: {port}")
    packet_count = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind((bind_ip, port))
        print(f"UDP sink listening on {bind_ip}:{port}", flush=True)

        while True:
            _data, _addr = sock.recvfrom(RECV_BUFFER_SIZE)
            packet_count += 1
    except KeyboardInterrupt:
        print(f"UDP sink stopped; received_packets={packet_count}")
        return packet_count
    finally:
        sock.close()

def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_sink(args.bind_ip, args.port)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"UDP sink failed: {exc}") from exc

if __name__ == "__main__":
    main()
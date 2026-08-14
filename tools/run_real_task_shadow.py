from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.real_task_shadow import (
    capture_packet,
    probe_placeholder,
    read_canonical_json,
    replay_packet,
    replay_registry,
    verify_packet_against_snapshot,
    verify_result_against_packets,
    verify_result_against_registry,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the result-free real-task shadow protocol")
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe", help="run the unregistered placeholder round-trip")
    probe.add_argument("--fixture", type=Path, required=True)

    capture = commands.add_parser("capture", help="append one packet to a local registry")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--registry-dir", type=Path, required=True)

    verify_packet = commands.add_parser("verify-packet", help="verify a packet and snapshot")
    verify_packet.add_argument("--packet", type=Path, required=True)
    verify_packet.add_argument("--snapshot", type=Path, required=True)

    replay = commands.add_parser("replay", help="cumulatively replay packets into both shadow arms")
    replay_input = replay.add_mutually_exclusive_group(required=True)
    replay_input.add_argument("--packet", type=Path)
    replay_input.add_argument("--registry-dir", type=Path)
    replay.add_argument("--snapshot", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    verify_result = commands.add_parser("verify-result", help="recompute and verify an exact replay result")
    verify_result.add_argument("--result", type=Path, required=True)
    verify_input = verify_result.add_mutually_exclusive_group(required=True)
    verify_input.add_argument("--packet", type=Path)
    verify_input.add_argument("--registry-dir", type=Path)
    verify_result.add_argument("--snapshot", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "probe":
        print(json.dumps(probe_placeholder(args.fixture), ensure_ascii=False, sort_keys=True))
    elif args.command == "capture":
        print(capture_packet(read_canonical_json(args.input), args.registry_dir))
    elif args.command == "verify-packet":
        verify_packet_against_snapshot(read_canonical_json(args.packet), args.snapshot)
        print("packet verified")
    elif args.command == "replay":
        result = (
            replay_packet(read_canonical_json(args.packet), args.snapshot)
            if args.packet is not None
            else replay_registry(args.registry_dir, args.snapshot)
        )
        write_json_exclusive(args.output, result)
        print(args.output)
    else:
        result = read_canonical_json(args.result)
        if args.packet is not None:
            verify_result_against_packets(
                result, [read_canonical_json(args.packet)], args.snapshot
            )
        else:
            verify_result_against_registry(result, args.registry_dir, args.snapshot)
        print("result verified")


if __name__ == "__main__":
    main()

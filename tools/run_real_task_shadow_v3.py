from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.real_task_shadow import read_canonical_json
from neuron_graph_rag.real_task_shadow_v3 import (
    audit_repository_lifecycle,
    capture_packet,
    probe_placeholder,
    replay_packet,
    replay_registry,
    verify_packet_against_snapshot,
    verify_result_against_packets,
    verify_result_against_registry,
    write_final_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lifecycle-safe result-free real-task shadow v3 protocol"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser(
        "probe", help="run the unregistered runtime-shaped placeholder round-trip"
    )
    probe.add_argument("--fixture", type=Path, required=True)
    capture = commands.add_parser("capture", help="append one v3 packet to a registry")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--registry-dir", type=Path, required=True)
    verify_packet = commands.add_parser(
        "verify-packet", help="verify a v3 packet against its explicit snapshot"
    )
    verify_packet.add_argument("--packet", type=Path, required=True)
    verify_packet.add_argument("--snapshot", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="cumulatively replay v3 packets into both shadow arms"
    )
    replay_input = replay.add_mutually_exclusive_group(required=True)
    replay_input.add_argument("--packet", type=Path)
    replay_input.add_argument("--registry-dir", type=Path)
    replay.add_argument("--snapshot", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    verify_result = commands.add_parser(
        "verify-result", help="recompute and verify an exact v3 result"
    )
    verify_result.add_argument("--result", type=Path, required=True)
    verify_input = verify_result.add_mutually_exclusive_group(required=True)
    verify_input.add_argument("--packet", type=Path)
    verify_input.add_argument("--registry-dir", type=Path)
    verify_result.add_argument("--snapshot", type=Path, required=True)
    audit = commands.add_parser(
        "audit-lifecycle",
        help="verify frozen hashes and repository lifecycle without a live snapshot",
    )
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--repository-root", type=Path, required=True)
    audit.add_argument("--registered-root", type=Path)

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
            if args.packet
            else replay_registry(args.registry_dir, args.snapshot)
        )
        write_final_aggregate(result, args.output)
        print(args.output)
    elif args.command == "verify-result":
        result = read_canonical_json(args.result)
        if args.packet:
            verify_result_against_packets(
                result, [read_canonical_json(args.packet)], args.snapshot
            )
        else:
            verify_result_against_registry(result, args.registry_dir, args.snapshot)
        print("result verified")
    else:
        print(
            json.dumps(
                audit_repository_lifecycle(
                    args.manifest,
                    repository_root=args.repository_root,
                    registered_root=args.registered_root,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

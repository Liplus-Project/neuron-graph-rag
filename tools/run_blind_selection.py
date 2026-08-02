from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.blind_selection import (
    aggregate_blind_results,
    audit_result_free_freeze,
    capture_judge_response,
    generate_blind_packet,
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen blind channel-selection protocol step."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("audit-freeze")
    freeze.add_argument("--manifest", type=Path, required=True)

    packet = subparsers.add_parser("generate-packet")
    packet.add_argument("split", choices=("development", "holdout"))
    packet.add_argument("--manifest", type=Path, required=True)
    packet.add_argument("--development-result", type=Path)
    packet.add_argument("--output", type=Path, required=True)

    capture = subparsers.add_parser("capture-response")
    capture.add_argument("--packet", type=Path, required=True)
    capture.add_argument("--raw-response", type=Path, required=True)
    capture.add_argument("--judge-id", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--agent-type", required=True)
    capture.add_argument("--executed-at")
    capture.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--packet", type=Path, required=True)
    aggregate.add_argument(
        "--response", type=Path, action="append", required=True
    )
    aggregate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "audit-freeze":
        audit_result_free_freeze(args.manifest)
        return 0
    if args.command == "generate-packet":
        payload = generate_blind_packet(
            args.manifest,
            args.split,
            development_result_path=args.development_result,
        )
    elif args.command == "capture-response":
        payload = capture_judge_response(
            args.packet,
            args.raw_response.read_text(encoding="utf-8"),
            judge_id=args.judge_id,
            model=args.model,
            agent_type=args.agent_type,
            executed_at=args.executed_at,
        )
    else:
        payload = aggregate_blind_results(
            args.manifest, args.packet, args.response
        )
    write_json_exclusive(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

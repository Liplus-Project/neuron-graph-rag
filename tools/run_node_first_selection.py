from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.node_first_selection import (
    aggregate_node_first_results,
    audit_node_first_result_freeze,
    capture_single_response,
    generate_node_first_stage,
    write_json_exclusive,
    write_node_first_stage_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen node-first blind-selection protocol step."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("audit-freeze")
    freeze.add_argument("--manifest", type=Path, required=True)

    stage = subparsers.add_parser("generate-stage")
    stage.add_argument("split", choices=("development", "holdout"))
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--development-result", type=Path)

    capture = subparsers.add_parser("capture-response")
    capture.add_argument("--case-packet", type=Path, required=True)
    capture.add_argument("--raw-response", type=Path, required=True)
    capture.add_argument("--judge-id", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--agent-type", required=True)
    capture.add_argument("--executed-at")
    capture.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--stage-packet", type=Path, required=True)
    aggregate.add_argument("--response", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "audit-freeze":
        audit_node_first_result_freeze(args.manifest)
        return 0
    if args.command == "generate-stage":
        stage_packet, case_artifacts = generate_node_first_stage(
            args.manifest,
            args.split,
            development_result_path=args.development_result,
        )
        write_node_first_stage_exclusive(
            args.manifest, args.split, stage_packet, case_artifacts
        )
        return 0
    if args.command == "capture-response":
        payload = capture_single_response(
            args.case_packet,
            args.raw_response.read_text(encoding="utf-8"),
            judge_id=args.judge_id,
            model=args.model,
            agent_type=args.agent_type,
            executed_at=args.executed_at,
        )
    else:
        payload = aggregate_node_first_results(
            args.manifest, args.stage_packet, args.response
        )
    write_json_exclusive(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

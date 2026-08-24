"""Audit and execute the frozen GitHub RAG versus NGR parity protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from neuron_graph_rag.github_retrieval_parity import (
    load_protocol,
    prove_writer_verifier_round_trip,
    register_capture,
    run_registered_stage,
    verify_frozen_artifacts,
    verify_registered_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--probe", action="store_true")
    actions.add_argument("--register-capture", choices=("development", "holdout"))
    actions.add_argument("--stage", choices=("development", "holdout"))
    actions.add_argument("--verify", choices=("development", "holdout"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--protocol-commit")
    args = parser.parse_args()

    if args.audit:
        protocol = load_protocol()
        verify_frozen_artifacts(protocol)
        print(
            json.dumps(
                {
                    "protocol_id": protocol["manifest"]["protocol_id"],
                    "result_free": True,
                },
                indent=2,
            )
        )
        return
    if args.probe:
        with TemporaryDirectory() as directory:
            prove_writer_verifier_round_trip(Path(directory) / "placeholder.json")
        print(
            json.dumps(
                {"placeholder_round_trip": True, "registered_queries_executed": False},
                indent=2,
            )
        )
        return
    if args.register_capture:
        if args.input is None or args.protocol_commit is None:
            parser.error("--register-capture requires --input and --protocol-commit")
        print(register_capture(args.register_capture, args.input, args.protocol_commit))
        return
    if args.stage:
        if args.protocol_commit is None:
            parser.error("--stage requires --protocol-commit")
        print(run_registered_stage(args.stage, args.protocol_commit))
        return
    verify_registered_result(args.verify)
    print(f"{args.verify} verification passed")


if __name__ == "__main__":
    main()

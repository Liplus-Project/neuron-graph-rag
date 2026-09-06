"""Audit or execute the frozen GitHub retrieval parity v2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.github_retrieval_parity_v2 import (
    audit_repository_lifecycle,
    audit_result_free,
    register_capture,
    run_registered_stage,
    verify_registered_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--lifecycle", action="store_true")
    actions.add_argument("--register-capture", choices=("development", "holdout"))
    actions.add_argument("--stage", choices=("development", "holdout"))
    actions.add_argument("--verify", choices=("development", "holdout"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--protocol-commit")
    args = parser.parse_args()

    if args.audit:
        print(json.dumps(audit_result_free(), ensure_ascii=False, indent=2))
        return
    if args.lifecycle:
        print(json.dumps(audit_repository_lifecycle(), ensure_ascii=False, indent=2))
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
    verify_registered_result(str(args.verify))
    print(f"{args.verify} verification passed")


if __name__ == "__main__":
    main()

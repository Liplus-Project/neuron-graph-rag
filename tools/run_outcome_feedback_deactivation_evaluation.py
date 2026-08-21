from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from neuron_graph_rag.outcome_feedback_deactivation_evaluation import (
    acquire_transactional_snapshot,
    preflight_snapshot,
    prove_writer_verifier_round_trip,
    run_registered_stage,
    verify_registered_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--acquire", action="store_true")
    actions.add_argument("--probe", action="store_true")
    actions.add_argument("--stage", choices=("development", "holdout"))
    actions.add_argument("--verify", choices=("development", "holdout"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.acquire:
        if args.source is None or args.snapshot is None:
            parser.error("--acquire requires --source and --snapshot")
        print(json.dumps(acquire_transactional_snapshot(args.source, args.snapshot), indent=2))
    elif args.probe:
        if args.snapshot is None:
            parser.error("--probe requires --snapshot")
        with TemporaryDirectory() as directory:
            prove_writer_verifier_round_trip(Path(directory) / "placeholder.json")
        print(json.dumps(preflight_snapshot(args.snapshot), indent=2))
    elif args.stage:
        if args.snapshot is None:
            parser.error("--stage requires --snapshot")
        print(run_registered_stage(args.stage, args.snapshot))
    else:
        verify_registered_result(args.verify)
        print(f"{args.verify} verification passed")


if __name__ == "__main__":
    main()

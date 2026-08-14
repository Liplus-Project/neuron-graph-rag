from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from neuron_graph_rag.feedback_policy_comparison_evaluation import (
    prove_writer_verifier_round_trip,
    run_registered_stage,
    verify_registered_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "holdout"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--verify", choices=("development", "holdout"))
    args = parser.parse_args()
    selected = sum(value is not None and value is not False for value in (args.stage, args.probe, args.verify))
    if selected != 1:
        parser.error("select exactly one of --stage, --probe, or --verify")
    if args.probe:
        with TemporaryDirectory() as directory:
            prove_writer_verifier_round_trip(Path(directory) / "placeholder.json")
        print("placeholder round-trip passed")
    elif args.verify:
        verify_registered_result(args.verify)
        print(f"{args.verify} verification passed")
    else:
        print(run_registered_stage(args.stage))


if __name__ == "__main__":
    main()

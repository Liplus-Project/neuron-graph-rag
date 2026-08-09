from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.engine_feedback_trajectory import (
    read_engine_feedback_trajectory_manifest,
    run_engine_feedback_trajectory_development,
    run_engine_feedback_trajectory_holdout,
    write_engine_feedback_trajectory_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen engine-backed feedback trajectory stage."
    )
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_engine_feedback_trajectory_manifest(args.manifest)
    registered_output = (
        args.manifest.parent / manifest["result_paths"][args.stage]
    ).resolve()
    if args.output.resolve() != registered_output:
        raise SystemExit("--output must match the frozen exclusive result path")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an observed result: {args.output}")

    if args.stage == "development":
        if args.development_result is not None:
            raise SystemExit("--development-result is only valid for holdout")
        result = run_engine_feedback_trajectory_development(args.manifest)
    else:
        if args.development_result is None:
            raise SystemExit("holdout requires --development-result")
        result = run_engine_feedback_trajectory_holdout(
            args.manifest, args.development_result
        )
    write_engine_feedback_trajectory_result(args.output, result)
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

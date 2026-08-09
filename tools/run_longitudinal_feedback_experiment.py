from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.longitudinal_feedback_experiment import (
    run_longitudinal_feedback_development,
    run_longitudinal_feedback_holdout,
    write_longitudinal_feedback_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one registered frozen longitudinal feedback stage.")
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    expected = manifest.parent / ("d1_liplus_longitudinal_feedback_experiment.development.result.json" if args.stage == "development" else "d1_liplus_longitudinal_feedback_experiment.holdout.result.json")
    if args.output.resolve() != expected.resolve():
        raise SystemExit(f"Refusing unregistered output path: {args.output}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an observed result: {args.output}")
    if args.stage == "development":
        if args.development_result is not None:
            raise SystemExit("--development-result is only valid for holdout")
        result = run_longitudinal_feedback_development(manifest)
    else:
        if args.development_result is None:
            raise SystemExit("holdout requires --development-result")
        result = run_longitudinal_feedback_holdout(manifest, args.development_result)
    write_longitudinal_feedback_result(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

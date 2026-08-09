from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.feedback_adaptation_reproduction import (
    run_feedback_adaptation_reproduction_development,
    run_feedback_adaptation_reproduction_holdout,
    write_feedback_adaptation_reproduction_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen feedback adaptation reproduction stage.")
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an observed result: {args.output}")
    if args.stage == "development":
        if args.development_result is not None:
            raise SystemExit("--development-result is only valid for holdout")
        result = run_feedback_adaptation_reproduction_development(args.manifest)
    else:
        if args.development_result is None:
            raise SystemExit("holdout requires --development-result")
        result = run_feedback_adaptation_reproduction_holdout(args.manifest, args.development_result)
    write_feedback_adaptation_reproduction_result(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

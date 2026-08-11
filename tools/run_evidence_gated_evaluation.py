from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.evidence_gated_evaluation import run_and_write_stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one registered evidence-gated feedback evaluation stage."
    )
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    result = run_and_write_stage(args.repo_root.resolve(), args.stage)
    print(json.dumps({"stage": args.stage, "all_pass": result["all_pass"]}))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

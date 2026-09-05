from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.source_grounded_relation_observation_v3 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen source-grounded relation v3 stage."
    )
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--shared-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_stage(
        args.stage,
        args.protocol_commit,
        args.shared_database,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

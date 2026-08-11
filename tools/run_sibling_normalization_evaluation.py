from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.sibling_normalization_evaluation import run_registered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the registered sibling normalization controlled evaluation."
    )
    parser.add_argument("stage", choices=("development", "holdout"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = run_registered(repo_root, args.stage)
    print(output.relative_to(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

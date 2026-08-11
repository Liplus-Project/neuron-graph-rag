from __future__ import annotations

import argparse
from pathlib import Path

from neuron_graph_rag.rank_elasticity import (
    run_rank_elasticity,
    write_rank_elasticity_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure feedback rank elasticity on isolated SQLite clones."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(
            f"Refusing to overwrite a rank elasticity result: {arguments.output}"
        )
    result = run_rank_elasticity(arguments.database, arguments.schedule)
    write_rank_elasticity_result(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

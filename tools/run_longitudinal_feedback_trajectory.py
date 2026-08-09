from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuron_graph_rag.longitudinal_feedback_trajectory import ProtocolError, run_registered_split


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one create-only longitudinal feedback trajectory split."
    )
    parser.add_argument("split", choices=("development", "holdout"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = run_registered_split(root, args.split)
    except ProtocolError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

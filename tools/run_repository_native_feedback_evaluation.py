"""Run one immutable stage of the repository-native feedback evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuron_graph_rag.repository_native_feedback_evaluation import (
    ProtocolStop,
    assert_valid_development_result,
    build_stage_result,
    write_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="repository-native feedback evaluator")
    parser.add_argument("stage", choices=("development", "holdout"))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    output = (
        repository_root
        / "artifacts"
        / "repository-native-feedback-v2"
        / f"{args.stage}.result.json"
    )
    try:
        allow_holdout = False
        if args.stage == "holdout":
            development_result = output.with_name("development.result.json")
            if not development_result.is_file():
                raise ProtocolStop("holdout requires an existing development result")
            assert_valid_development_result(development_result)
            allow_holdout = True
        result = build_stage_result(args.stage, repository_root, allow_holdout=allow_holdout)
        write_exclusive(output, result)
    except ProtocolStop as error:
        print(f"停止: {error}", file=sys.stderr)
        return 2
    print(f"完了: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

from neuron_graph_rag.canonical_gate_evaluation import run_registered_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "holdout"), required=True)
    args = parser.parse_args()
    output = run_registered_stage(args.stage)
    print(output)


if __name__ == "__main__":
    main()

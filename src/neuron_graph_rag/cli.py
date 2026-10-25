from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .engine import EngineConfig, NeuronGraphRAG
from .benchmark import run_benchmark, write_benchmark_result
from .database_home import resolve_database
from .evaluation import evaluate
from .sample import load_sample_corpus


def run_demo(database: str | Path = ":memory:") -> dict[str, Any]:
    query = "The payment retry policy changed from two attempts to five attempts."
    with NeuronGraphRAG(
        database,
        config=EngineConfig(seed_count=1, max_hops=2),
    ) as engine:
        load_sample_corpus(engine)
        edge_before = engine.store.edge(
            "decision", "implementation", "implemented_by"
        ).weight
        first = engine.search(query, limit=5, now=1_000.0)
        first_target = next(
            hit for hit in first.hits if hit.node.node_id == "implementation"
        )
        receipt = engine.record_success(
            first.trace_id, ["implementation"], now=1_001.0
        )
        edge_after = engine.store.edge(
            "decision", "implementation", "implemented_by"
        ).weight
        second = engine.search(query, limit=5, now=1_002.0)
        second_target = next(
            hit for hit in second.hits if hit.node.node_id == "implementation"
        )
        return {
            "query": query,
            "trace_before": first.trace_id,
            "trace_after": second.trace_id,
            "retrieval_count": engine.store.count_retrievals(),
            "feedback_count": engine.store.count_feedback(),
            "target": "implementation",
            "before": {
                "rank": _rank(first, "implementation"),
                "graph_activation": first_target.graph_activation,
                "explanation": first_target.explain(),
            },
            "success_feedback": {
                "feedback_id": receipt.feedback_id,
                "used_node_ids": list(receipt.used_node_ids),
                "reinforced_edges": [
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "edge_type": edge.edge_type,
                        "old_weight": edge.old_weight,
                        "new_weight": edge.new_weight,
                    }
                    for edge in receipt.reinforced_edges
                ],
            },
            "after": {
                "rank": _rank(second, "implementation"),
                "graph_activation": second_target.graph_activation,
                "explanation": second_target.explain(),
            },
            "implemented_by_weight": {
                "before": edge_before,
                "after": edge_after,
            },
        }


def expire_nodes(
    database: str | Path | None,
    older_than_days: float,
    *,
    dry_run: bool = False,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> dict[str, Any]:
    resolution = resolve_database(
        database, environ=os.environ if environ is None else environ, home=home
    )
    if not resolution.path.is_file():
        raise FileNotFoundError(f"database does not exist: {resolution.path}")
    with NeuronGraphRAG(resolution.path) as engine:
        result = engine.expire_unused_nodes(older_than_days, now=now, dry_run=dry_run)
    result["database"] = str(resolution.path)
    result["database_source"] = resolution.source
    return result


def _positive_days(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number of days") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of days")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuron-graph-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="Run the vertical slice demo")
    demo_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite path. The default uses an in-memory database.",
    )
    subparsers.add_parser("eval", help="Compare hybrid and graph retrieval")
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run a frozen real-corpus benchmark"
    )
    benchmark_parser.add_argument("--fixture", type=Path, required=True)
    benchmark_parser.add_argument("--gold", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, default=None)
    expire_parser = subparsers.add_parser(
        "expire-nodes",
        help="Delete nodes not successfully used within N days (opt-in, explicit only)",
    )
    expire_parser.add_argument(
        "--database",
        default=None,
        help=(
            "Existing SQLite database to expire "
            "(default: NGR_DATABASE or the shared database home)"
        ),
    )
    expire_parser.add_argument(
        "--older-than-days",
        type=_positive_days,
        required=True,
        help="Expire nodes whose last successful use (or creation, if never used) is older",
    )
    expire_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the nodes that would expire without deleting anything",
    )
    args = parser.parse_args(argv)

    if args.command == "demo":
        result = run_demo(args.db or ":memory:")
    elif args.command == "eval":
        result = evaluate()
    elif args.command == "expire-nodes":
        try:
            result = expire_nodes(
                args.database, args.older_than_days, dry_run=args.dry_run
            )
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    else:
        result = run_benchmark(args.fixture, args.gold)
        if args.output is not None:
            write_benchmark_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _rank(trace: object, node_id: str) -> int:
    hits = getattr(trace, "hits")
    return next(
        index
        for index, hit in enumerate(hits, start=1)
        if hit.node.node_id == node_id
    )

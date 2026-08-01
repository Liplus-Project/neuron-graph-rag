from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .d1_fixture import load_fixture, read_fixture
from .engine import EngineConfig, NeuronGraphRAG


BENCHMARK_SCHEMA_VERSION = 1
COHORTS = ("direct_lookup", "relation", "negative_control")


def read_gold(path: str | Path) -> dict[str, Any]:
    gold_path = Path(path)
    with gold_path.open(encoding="utf-8") as stream:
        gold = json.load(stream)
    if gold.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark schema version: {gold.get('schema_version')!r}"
        )
    cases = gold.get("cases")
    if not isinstance(cases, list) or len(cases) < 12:
        raise ValueError("Benchmark gold must contain at least 12 cases")
    case_ids = [str(case.get("id", "")) for case in cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Benchmark case ids must be non-empty and unique")
    counts = Counter(str(case.get("cohort", "")) for case in cases)
    if any(counts[cohort] == 0 for cohort in COHORTS):
        raise ValueError(f"Benchmark must cover every cohort: {COHORTS!r}")
    for case in cases:
        if not str(case.get("query", "")).strip():
            raise ValueError(f"Benchmark case {case['id']} has an empty query")
        if not str(case.get("expected_node_id", "")):
            raise ValueError(f"Benchmark case {case['id']} lacks expected_node_id")
        if int(case.get("acceptable_rank", 0)) < 1:
            raise ValueError(f"Benchmark case {case['id']} has invalid acceptable_rank")
        source_url = str(case.get("source_url", ""))
        if not source_url.startswith("https://github.com/"):
            raise ValueError(f"Benchmark case {case['id']} lacks a GitHub source URL")
        if case["cohort"] == "relation":
            expected_path = case.get("expected_path")
            if not isinstance(expected_path, list) or not expected_path:
                raise ValueError(
                    f"Relation case {case['id']} must declare expected_path"
                )
            if len(expected_path) not in (1, 2):
                raise ValueError(
                    f"Relation case {case['id']} must use a one- or two-hop path"
                )
            for step in expected_path:
                if set(step) != {"source_id", "target_id", "edge_type"}:
                    raise ValueError(
                        f"Relation case {case['id']} has an invalid path step"
                    )
    return gold


def run_benchmark(
    fixture_path: str | Path,
    gold_path: str | Path,
) -> dict[str, Any]:
    fixture_path = Path(fixture_path)
    gold_path = Path(gold_path)
    fixture = read_fixture(fixture_path)
    gold = read_gold(gold_path)
    node_ids = {str(node["node_id"]) for node in fixture["nodes"]}
    for case in gold["cases"]:
        if case["expected_node_id"] not in node_ids:
            raise ValueError(
                f"Benchmark case {case['id']} targets a node outside the fixture"
            )

    settings = gold["benchmark"]
    limit = int(settings["limit"])
    if limit < len(node_ids):
        raise ValueError("Benchmark limit must cover the complete compact fixture")
    baseline_config = EngineConfig(**settings["baseline_config"])
    graph_config = EngineConfig(**settings["graph_config"])
    baseline_results, _ = _run_cases(
        fixture_path, gold["cases"], baseline_config, limit
    )
    graph_results, _ = _run_cases(
        fixture_path, gold["cases"], graph_config, limit
    )

    comparisons = []
    graph_by_id = {result["id"]: result for result in graph_results}
    for baseline in baseline_results:
        graph = graph_by_id[baseline["id"]]
        rank_delta = baseline["rank"] - graph["rank"]
        comparisons.append(
            {
                "id": baseline["id"],
                "cohort": baseline["cohort"],
                "acceptable_rank": baseline["acceptable_rank"],
                "baseline_rank": baseline["rank"],
                "graph_rank": graph["rank"],
                "rank_delta": rank_delta,
                "outcome": (
                    "improved"
                    if rank_delta > 0
                    else "worsened" if rank_delta < 0 else "equal"
                ),
            }
        )

    explanations = [
        {
            "id": result["id"],
            "matched": result["path_matched"],
            "expected_path": result["expected_path"],
            "observed_paths": result["observed_paths"],
        }
        for result in graph_results
        if result["cohort"] == "relation"
    ]
    feedback = _run_feedback(
        fixture_path,
        gold["cases"],
        graph_config,
        limit,
        str(settings["feedback_case_id"]),
    )
    metrics = {
        "baseline_hybrid": _metrics(baseline_results),
        "graph_integrated": _metrics(graph_results),
        "comparison": _comparison_metrics(comparisons),
    }
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "inputs": {
            "fixture_sha256": _sha256(fixture_path),
            "gold_sha256": _sha256(gold_path),
        },
        "corpus": {
            "node_count": len(fixture["nodes"]),
            "edge_count": len(fixture["edges"]),
            "edge_types": sorted(
                {str(edge["edge_type"]) for edge in fixture["edges"]}
            ),
            "source": fixture.get("source", {}),
        },
        "case_count": len(gold["cases"]),
        "cohort_counts": dict(sorted(Counter(case["cohort"] for case in gold["cases"]).items())),
        "benchmark": settings,
        "metrics": metrics,
        "cases": comparisons,
        "explanations": explanations,
        "feedback": feedback,
    }
    result["hypotheses"] = _judge_hypotheses(result)
    return result


def write_benchmark_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_cases(
    fixture_path: Path,
    cases: list[dict[str, Any]],
    config: EngineConfig,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        for case in cases:
            trace = engine.search(case["query"], limit=limit, now=1_000.0)
            traces[case["id"]] = trace
            hit = next(
                hit for hit in trace.hits if hit.node.node_id == case["expected_node_id"]
            )
            rank = next(
                index
                for index, candidate in enumerate(trace.hits, start=1)
                if candidate.node.node_id == case["expected_node_id"]
            )
            expected_path = case.get("expected_path", [])
            observed_paths = [_simple_path(path) for path in hit.explain()["paths"]]
            results.append(
                {
                    "id": case["id"],
                    "cohort": case["cohort"],
                    "acceptable_rank": int(case["acceptable_rank"]),
                    "rank": rank,
                    "expected_path": expected_path,
                    "observed_paths": observed_paths,
                    "path_matched": (
                        any(path["steps"] == expected_path for path in observed_paths)
                        if expected_path
                        else None
                    ),
                }
            )
    return results, traces


def _run_feedback(
    fixture_path: Path,
    cases: list[dict[str, Any]],
    config: EngineConfig,
    limit: int,
    feedback_case_id: str,
) -> dict[str, Any]:
    case_by_id = {str(case["id"]): case for case in cases}
    if feedback_case_id not in case_by_id:
        raise ValueError("feedback_case_id does not identify a gold case")
    feedback_case = case_by_id[feedback_case_id]
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before_results: dict[str, int] = {}
        feedback_trace = None
        for case in cases:
            trace = engine.search(case["query"], limit=limit, now=2_000.0)
            before_results[case["id"]] = _rank(trace, case["expected_node_id"])
            if case["id"] == feedback_case_id:
                feedback_trace = trace
        if feedback_trace is None:
            raise ValueError("Feedback case trace was not produced")

        before_edges = {_edge_key(edge): edge.weight for edge in engine.store.list_edges()}
        receipt = engine.record_success(
            feedback_trace.trace_id,
            [feedback_case["expected_node_id"]],
            now=2_001.0,
        )
        after_edges = {_edge_key(edge): edge.weight for edge in engine.store.list_edges()}
        changed_edges = [
            {
                "source_id": key[0],
                "target_id": key[1],
                "edge_type": key[2],
                "old_weight": before_edges[key],
                "new_weight": after_edges[key],
            }
            for key in sorted(before_edges)
            if before_edges[key] != after_edges[key]
        ]
        credited_edges = [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "old_weight": edge.old_weight,
                "new_weight": edge.new_weight,
            }
            for edge in receipt.reinforced_edges
        ]
        credited_keys = {
            (edge["source_id"], edge["target_id"], edge["edge_type"])
            for edge in credited_edges
        }
        uncredited_changes = [
            edge
            for edge in changed_edges
            if (edge["source_id"], edge["target_id"], edge["edge_type"])
            not in credited_keys
        ]
        after_results: dict[str, int] = {}
        for case in cases:
            trace = engine.search(case["query"], limit=limit, now=2_002.0)
            after_results[case["id"]] = _rank(trace, case["expected_node_id"])
        non_target_rank_changes = [
            {
                "id": case_id,
                "before_rank": before_results[case_id],
                "after_rank": after_results[case_id],
            }
            for case_id in sorted(before_results)
            if case_id != feedback_case_id
            and before_results[case_id] != after_results[case_id]
        ]
        return {
            "case_id": feedback_case_id,
            "target_node_id": feedback_case["expected_node_id"],
            "target_rank_before": before_results[feedback_case_id],
            "target_rank_after": after_results[feedback_case_id],
            "credited_edges": credited_edges,
            "changed_edges": changed_edges,
            "uncredited_edge_changes": uncredited_changes,
            "non_target_rank_changes": non_target_rank_changes,
        }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _cohort_metrics(results),
        "cohorts": {
            cohort: _cohort_metrics(
                [result for result in results if result["cohort"] == cohort]
            )
            for cohort in COHORTS
        },
    }


def _cohort_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(result["rank"]) for result in results]
    return {
        "cases": len(ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "hit_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "ranks": ranks,
    }


def _comparison_metrics(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["outcome"] for row in rows)
        return {
            "improved": counts["improved"],
            "equal": counts["equal"],
            "worsened": counts["worsened"],
            "rank_delta_sum": sum(int(row["rank_delta"]) for row in rows),
        }

    return {
        "overall": summarize(comparisons),
        "cohorts": {
            cohort: summarize(
                [row for row in comparisons if row["cohort"] == cohort]
            )
            for cohort in COHORTS
        },
    }


def _judge_hypotheses(result: dict[str, Any]) -> list[dict[str, str]]:
    relation_baseline = result["metrics"]["baseline_hybrid"]["cohorts"]["relation"]
    relation_graph = result["metrics"]["graph_integrated"]["cohorts"]["relation"]
    relation_comparison = result["metrics"]["comparison"]["cohorts"]["relation"]
    if (
        relation_graph["mean_reciprocal_rank"]
        > relation_baseline["mean_reciprocal_rank"]
        and relation_comparison["improved"] > relation_comparison["worsened"]
    ):
        h1 = "supported"
    elif (
        relation_graph["mean_reciprocal_rank"]
        <= relation_baseline["mean_reciprocal_rank"]
        and relation_comparison["improved"] <= relation_comparison["worsened"]
    ):
        h1 = "unsupported"
    else:
        h1 = "inconclusive"

    controls = [
        case
        for case in result["cases"]
        if case["cohort"] in {"direct_lookup", "negative_control"}
    ]
    excessive = any(
        case["graph_rank"] > case["acceptable_rank"] or case["rank_delta"] < -1
        for case in controls
    )
    if excessive:
        h2 = "unsupported"
    elif all(case["rank_delta"] >= 0 for case in controls):
        h2 = "supported"
    else:
        h2 = "inconclusive"

    explanations = result["explanations"]
    h3 = "supported" if all(item["matched"] for item in explanations) else "unsupported"

    feedback = result["feedback"]
    if feedback["uncredited_edge_changes"] or feedback["non_target_rank_changes"]:
        h4 = "unsupported"
    elif feedback["changed_edges"]:
        h4 = "supported"
    else:
        h4 = "inconclusive"
    return [
        {"id": "H1", "status": h1},
        {"id": "H2", "status": h2},
        {"id": "H3", "status": h3},
        {"id": "H4", "status": h4},
    ]


def _simple_path(path: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_id": path["seed_id"],
        "steps": [
            {
                "source_id": step["source_id"],
                "target_id": step["target_id"],
                "edge_type": step["edge_type"],
            }
            for step in path["steps"]
        ],
    }


def _rank(trace: Any, node_id: str) -> int:
    return next(
        index
        for index, hit in enumerate(trace.hits, start=1)
        if hit.node.node_id == node_id
    )


def _edge_key(edge: Any) -> tuple[str, str, str]:
    return edge.source_id, edge.target_id, edge.edge_type


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .engine import EngineConfig, NeuronGraphRAG

SCHEMA_VERSION = "ngr.rank-elasticity/v1"
CONTROL_ROLES = {
    "direct_control",
    "lexical_control",
    "directional_negative_control",
}
CASE_ROLES = CONTROL_ROLES | {"relation_target"}
COUNT_TABLES = (
    "nodes",
    "edges",
    "retrievals",
    "success_feedback",
    "source_use_state",
    "delayed_outcomes",
)


def read_rank_elasticity_schedule(path: str | Path) -> dict[str, Any]:
    schedule = _read_json(Path(path))
    _validate_schedule(schedule)
    return schedule


def run_rank_elasticity(
    database: str | Path, schedule_path: str | Path
) -> dict[str, Any]:
    source = Path(database).resolve()
    if not source.is_file():
        raise ValueError(f"Source database does not exist: {source}")
    schedule = read_rank_elasticity_schedule(schedule_path)
    before = _source_fingerprint(source)
    config = EngineConfig(**dict(schedule.get("config", {})))
    scenarios = [
        _run_scenario(source, schedule, scenario, config)
        for scenario in schedule["scenarios"]
    ]
    after = _source_fingerprint(source)
    if after != before:
        raise RuntimeError("Source database changed during rank elasticity simulation")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_database": {
            "sha256": before["sha256"],
            "counts": before["counts"],
            "edges": before["edges"],
            "unchanged_after_simulation": True,
        },
        "schedule": {
            "checkpoints": list(schedule["checkpoints"]),
            "limit": int(schedule["limit"]),
            "feedback_timestamp": float(schedule["feedback_timestamp"]),
            "score_timestamp": float(schedule["score_timestamp"]),
        },
        "config": asdict(config),
        "scenarios": scenarios,
    }


def write_rank_elasticity_result(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError(
            f"Refusing to overwrite a rank elasticity result: {output}"
        ) from error


def _run_scenario(
    source: Path,
    schedule: dict[str, Any],
    scenario: dict[str, Any],
    config: EngineConfig,
) -> dict[str, Any]:
    checkpoints = tuple(int(value) for value in schedule["checkpoints"])
    checkpoint_runs = [
        _run_checkpoint(
            source,
            scenario,
            config,
            feedback_count,
            int(schedule["limit"]),
            float(schedule["feedback_timestamp"]),
            float(schedule["score_timestamp"]),
        )
        for feedback_count in checkpoints
    ]
    cases = _case_trajectories(scenario, checkpoint_runs)
    target = next(
        case for case in cases if case["case_id"] == scenario["target_case_id"]
    )
    diagnosis = _diagnose(target, checkpoint_runs)
    return {
        "scenario_id": scenario["scenario_id"],
        "feedback": dict(scenario["feedback"]),
        "target_case_id": scenario["target_case_id"],
        "diagnosis": diagnosis,
        "cases": cases,
    }


def _run_checkpoint(
    source: Path,
    scenario: dict[str, Any],
    config: EngineConfig,
    feedback_count: int,
    limit: int,
    feedback_timestamp: float,
    score_timestamp: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ngr-rank-elasticity-") as directory:
        clone = Path(directory) / "simulation.sqlite"
        _clone_database(source, clone)
        with NeuronGraphRAG(clone, config=config) as engine:
            initial_edges = _edge_snapshot(engine)
            feedback = scenario["feedback"]
            for index in range(feedback_count):
                trace = _feedback_trace(
                    engine,
                    str(feedback["query"]),
                    str(feedback["channel"]),
                    limit,
                    feedback_timestamp,
                )
                used_node_id = str(feedback["used_node_id"])
                if used_node_id not in {hit.node.node_id for hit in trace.hits}:
                    raise ValueError(
                        f"Feedback node is not in trace: {scenario['scenario_id']}"
                    )
                engine.record_success(
                    trace.trace_id,
                    [used_node_id],
                    now=feedback_timestamp,
                )
            final_edges = _edge_snapshot(engine)
            measurements = {
                str(case["case_id"]): _measure_case(
                    engine, case, limit, score_timestamp
                )
                for case in scenario["cases"]
            }
        return {
            "feedback_count": feedback_count,
            "changed_edges": _edge_changes(initial_edges, final_edges),
            "measurements": measurements,
        }


def _feedback_trace(
    engine: NeuronGraphRAG,
    query: str,
    channel: str,
    limit: int,
    timestamp: float,
) -> Any:
    if channel == "search":
        return engine.search(query, limit=limit, now=timestamp)
    channels = engine.search_channels(query, limit=limit, now=timestamp)
    return channels.relation if channel == "relation" else channels.lexical


def _measure_case(
    engine: NeuronGraphRAG,
    case: dict[str, Any],
    limit: int,
    timestamp: float,
) -> dict[str, Any]:
    trace = engine.search(str(case["query"]), limit=limit, now=timestamp)
    hits = list(trace.hits)
    node_id = str(case["node_id"])
    rank_by_node = {
        hit.node.node_id: rank for rank, hit in enumerate(hits, start=1)
    }
    if node_id not in rank_by_node:
        raise ValueError(f"Measured node is outside top-k: {case['case_id']}")
    rank = rank_by_node[node_id]
    hit = hits[rank - 1]
    return {
        "rank": rank,
        "scores": {
            "entry": hit.entry_score,
            "graph_raw": hit.graph_activation,
            "graph_normalized": hit.normalized_graph_activation,
            "final": hit.final_score,
        },
        "rank_by_node": rank_by_node,
        "final_score_by_node": {
            candidate.node.node_id: candidate.final_score for candidate in hits
        },
    }


def _case_trajectories(
    scenario: dict[str, Any], checkpoint_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline = checkpoint_runs[0]
    trajectories: list[dict[str, Any]] = []
    for case in scenario["cases"]:
        case_id = str(case["case_id"])
        node_id = str(case["node_id"])
        baseline_measurement = baseline["measurements"][case_id]
        baseline_ranks = baseline_measurement["rank_by_node"]
        baseline_rank = int(baseline_measurement["rank"])
        adjacent_node_id = (
            next(
                candidate
                for candidate, rank in baseline_ranks.items()
                if rank == baseline_rank - 1
            )
            if baseline_rank > 1
            else None
        )
        records: list[dict[str, Any]] = []
        for checkpoint in checkpoint_runs:
            measurement = checkpoint["measurements"][case_id]
            current_ranks = measurement.pop("rank_by_node")
            current_scores = measurement.pop("final_score_by_node")
            rank_deltas = _rank_deltas(baseline_ranks, current_ranks)
            non_target = [
                delta
                for delta in rank_deltas
                if delta["node_id"] != node_id and delta["delta"] != 0
            ]
            adjacent_margin = None
            if adjacent_node_id is not None and adjacent_node_id in current_scores:
                adjacent_margin = (
                    float(measurement["scores"]["final"])
                    - float(current_scores[adjacent_node_id])
                )
            records.append(
                {
                    "feedback_count": checkpoint["feedback_count"],
                    "rank": measurement["rank"],
                    "scores": measurement["scores"],
                    "adjacent_node_id": adjacent_node_id,
                    "adjacent_margin": adjacent_margin,
                    "changed_edges": checkpoint["changed_edges"],
                    "top_k_rank_delta": rank_deltas,
                    "non_target_churn": {
                        "changed_node_count": len(non_target),
                        "total_absolute_rank_delta": sum(
                            abs(int(delta["delta"])) for delta in non_target
                        ),
                        "node_ids": [delta["node_id"] for delta in non_target],
                    },
                }
            )
        trajectories.append(
            {
                "case_id": case_id,
                "role": case["role"],
                "node_id": node_id,
                "query": case["query"],
                "rank_stable_through_schedule": all(
                    record["rank"] == records[0]["rank"] for record in records
                ),
                "checkpoints": records,
            }
        )
    return trajectories


def _diagnose(
    target: dict[str, Any], checkpoint_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    records = target["checkpoints"]
    baseline_rank = int(records[0]["rank"])
    first_flip = next(
        (
            int(record["feedback_count"])
            for record in records[1:]
            if int(record["rank"]) < baseline_rank
        ),
        None,
    )
    first_regression = next(
        (
            int(record["feedback_count"])
            for record in records[1:]
            if int(record["rank"]) > baseline_rank
        ),
        None,
    )
    edge_changed = any(run["changed_edges"] for run in checkpoint_runs[1:])
    rank_stable = all(int(record["rank"]) == baseline_rank for record in records)
    normalized_stable = all(
        math.isclose(
            float(record["scores"]["graph_normalized"]),
            float(records[0]["scores"]["graph_normalized"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for record in records
    )
    final_stable = all(
        math.isclose(
            float(record["scores"]["final"]),
            float(records[0]["scores"]["final"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for record in records
    )
    if first_regression is not None:
        classification = "rank_regression"
    elif first_flip is not None:
        classification = "rank_flip_threshold"
    elif edge_changed and rank_stable:
        classification = "edge_changed_but_rank_unchanged"
    else:
        classification = "rank_stable_through_schedule"
    return {
        "classification": classification,
        "rank_flip_threshold": first_flip,
        "rank_regression_first_checkpoint": first_regression,
        "rank_stable_through_schedule": rank_stable,
        "edge_changed_but_rank_unchanged": edge_changed and rank_stable,
        "fusion_side_ceiling": (
            edge_changed and rank_stable and normalized_stable and final_stable
        ),
    }


def _rank_deltas(
    baseline: dict[str, int], current: dict[str, int]
) -> list[dict[str, Any]]:
    outside_top_k_rank = max(len(baseline), len(current)) + 1
    return [
        {
            "node_id": node_id,
            "baseline_rank": baseline.get(node_id),
            "current_rank": current.get(node_id),
            "delta": baseline.get(node_id, outside_top_k_rank)
            - current.get(node_id, outside_top_k_rank),
            "top_k_status": (
                "entered"
                if node_id not in baseline
                else "left"
                if node_id not in current
                else "retained"
            ),
        }
        for node_id in sorted(set(baseline) | set(current))
    ]


def _clone_database(source: Path, destination: Path) -> None:
    uri = source.as_uri() + "?mode=ro"
    with (
        closing(sqlite3.connect(uri, uri=True)) as source_connection,
        closing(sqlite3.connect(destination)) as destination_connection,
    ):
        source_connection.execute("PRAGMA query_only = ON")
        source_connection.backup(destination_connection)


def _source_fingerprint(path: Path) -> dict[str, Any]:
    uri = path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        counts = {
            table: (
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                if table in tables
                else 0
            )
            for table in COUNT_TABLES
        }
        edges = []
        if "edges" in tables:
            edges = [
                {
                    "source_id": str(row["source_id"]),
                    "target_id": str(row["target_id"]),
                    "edge_type": str(row["edge_type"]),
                    "weight": float(row["weight"]),
                    "reinforced_count": int(row["reinforced_count"]),
                }
                for row in connection.execute(
                    "SELECT * FROM edges ORDER BY source_id, target_id, edge_type"
                )
            ]
    return {
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": counts,
        "edges": edges,
    }


def _edge_snapshot(
    engine: NeuronGraphRAG,
) -> dict[tuple[str, str, str], tuple[float, int]]:
    return {
        (edge.source_id, edge.target_id, edge.edge_type): (
            edge.weight,
            edge.reinforced_count,
        )
        for edge in engine.store.list_edges()
    }


def _edge_changes(
    before: dict[tuple[str, str, str], tuple[float, int]],
    after: dict[tuple[str, str, str], tuple[float, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": key[0],
            "target_id": key[1],
            "edge_type": key[2],
            "before_weight": before[key][0],
            "after_weight": after[key][0],
            "before_reinforced_count": before[key][1],
            "after_reinforced_count": after[key][1],
        }
        for key in sorted(before)
        if before[key] != after[key]
    ]


def _validate_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Rank elasticity schema version is unsupported")
    checkpoints = schedule.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or checkpoints[0] != 0
        or any(isinstance(value, bool) or not isinstance(value, int) for value in checkpoints)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] > 100
    ):
        raise ValueError("checkpoints must be unique increasing integers from 0 through 100")
    limit = schedule.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 through 100")
    for field in ("feedback_timestamp", "score_timestamp"):
        value = schedule.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    config = schedule.get("config", {})
    allowed_config = {field.name for field in fields(EngineConfig)}
    if not isinstance(config, dict) or set(config) - allowed_config:
        raise ValueError("config contains an unknown EngineConfig field")
    EngineConfig(**config)
    scenarios = schedule.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("At least one scenario is required")
    scenario_ids: set[str] = set()
    case_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise TypeError("Each scenario must be an object")
        scenario_id = str(scenario.get("scenario_id", ""))
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError("scenario_id must be non-empty and unique")
        scenario_ids.add(scenario_id)
        feedback = scenario.get("feedback")
        if not isinstance(feedback, dict) or set(feedback) != {
            "query", "used_node_id", "channel"
        }:
            raise ValueError(f"Feedback definition is invalid: {scenario_id}")
        if feedback["channel"] not in {"search", "lexical", "relation"}:
            raise ValueError(f"Feedback channel is invalid: {scenario_id}")
        if not str(feedback["query"]).strip() or not str(feedback["used_node_id"]):
            raise ValueError(f"Feedback query or node is empty: {scenario_id}")
        cases = scenario.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Scenario has no cases: {scenario_id}")
        roles = [case.get("role") for case in cases if isinstance(case, dict)]
        if len(cases) != len(CASE_ROLES) or set(roles) != CASE_ROLES:
            raise ValueError(
                f"Scenario must contain one target and every control role: {scenario_id}"
            )
        local_case_ids = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {
                "case_id", "role", "query", "node_id"
            }:
                raise ValueError(f"Case definition is invalid: {scenario_id}")
            case_id = str(case["case_id"])
            if not case_id or case_id in case_ids:
                raise ValueError("case_id must be non-empty and globally unique")
            if case["role"] not in CASE_ROLES:
                raise ValueError(f"Case role is invalid: {case_id}")
            if not str(case["query"]).strip() or not str(case["node_id"]):
                raise ValueError(f"Case query or node is empty: {case_id}")
            case_ids.add(case_id)
            local_case_ids.add(case_id)
        if scenario.get("target_case_id") not in local_case_ids:
            raise ValueError(f"target_case_id is outside scenario: {scenario_id}")
        target = next(case for case in cases if case["case_id"] == scenario["target_case_id"])
        if target["role"] != "relation_target":
            raise ValueError(f"Target case must have relation_target role: {scenario_id}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .d1_fixture import load_fixture, read_fixture
from .engine import EngineConfig, NeuronGraphRAG
from .models import SearchChannelsResult


CHANNEL_EXPERIMENT_SCHEMA_VERSION = 1


def read_channel_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CHANNEL_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported channel experiment schema version")
    if manifest.get("candidate_id") != "dual-lane":
        raise ValueError("Channel experiment must freeze the dual-lane candidate")
    base = manifest_path.parent
    for split_name in ("development", "holdout"):
        split = manifest[split_name]
        for field_name in ("fixture", "gold", "provenance"):
            if _canonical_sha256(base / split[field_name]) != split[
                f"{field_name}_sha256"
            ]:
                raise ValueError(
                    f"Frozen {split_name} {field_name} hash mismatch"
                )
        _validate_gold(base / split["fixture"], base / split["gold"])
    audit = manifest["contamination_audit"]
    audit_path = base / audit["artifact"]
    if _canonical_sha256(audit_path) != audit["artifact_sha256"]:
        raise ValueError("Frozen channel contamination audit hash mismatch")
    audit_payload = _read_json(audit_path)
    if not audit_payload.get("passed"):
        raise ValueError("Frozen channel contamination audit did not pass")
    for split_name in ("development", "holdout"):
        for field_name in ("fixture", "gold"):
            if audit_payload["inputs"][f"{split_name}_{field_name}_sha256"] != (
                manifest[split_name][f"{field_name}_sha256"]
            ):
                raise ValueError("Audit input differs from channel manifest")
    development_paths = _doc_paths(base / manifest["development"]["fixture"])
    holdout_paths = _doc_paths(base / manifest["holdout"]["fixture"])
    if development_paths & holdout_paths:
        raise ValueError("Channel development and holdout paths overlap")
    return manifest


def run_channel_development(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = read_channel_manifest(manifest_path)
    result = _evaluate_split(manifest_path, manifest, "development")
    gate_passed = all(result["gate"].values())
    return {
        "schema_version": CHANNEL_EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "development",
        "manifest_sha256": _canonical_sha256(manifest_path),
        "inputs": result["inputs"],
        "candidate_id": "dual-lane",
        "cases": result["cases"],
        "metrics": result["metrics"],
        "feedback": result["feedback"],
        "gate": result["gate"],
        "gate_passed": gate_passed,
        "selection": {
            "selected_candidate_id": "dual-lane" if gate_passed else "current",
            "reason": (
                "all_frozen_channel_gates_passed"
                if gate_passed
                else "channel_candidate_failed_frozen_gate"
            ),
        },
        "holdout_status": (
            "not_opened_candidate_selected"
            if gate_passed
            else "not_opened_no_candidate"
        ),
    }


def run_channel_holdout(
    manifest_path: str | Path, development_result_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    development_result_path = Path(development_result_path)
    manifest = read_channel_manifest(manifest_path)
    development = _read_json(development_result_path)
    if development.get("manifest_sha256") != _canonical_sha256(manifest_path):
        raise ValueError("Development result does not match frozen channel manifest")
    if not development.get("gate_passed"):
        raise ValueError("Stop rule forbids opening channel holdout")
    result = _evaluate_split(manifest_path, manifest, "holdout")
    adopted = all(result["gate"].values())
    return {
        "schema_version": CHANNEL_EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "holdout",
        "manifest_sha256": _canonical_sha256(manifest_path),
        "development_result_sha256": _canonical_sha256(
            development_result_path
        ),
        "holdout_open_count": 1,
        "inputs": result["inputs"],
        "candidate_id": "dual-lane",
        "cases": result["cases"],
        "metrics": result["metrics"],
        "feedback": result["feedback"],
        "gate": result["gate"],
        "decision": {
            "adopted": adopted,
            "default_api": "search_channels" if adopted else "search",
            "reason": (
                "all_frozen_holdout_gates_passed"
                if adopted
                else "channel_candidate_failed_frozen_holdout_gate"
            ),
        },
    }


def write_channel_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evaluate_split(
    manifest_path: Path, manifest: dict[str, Any], split_name: str
) -> dict[str, Any]:
    base = manifest_path.parent
    split = manifest[split_name]
    fixture_path = base / split["fixture"]
    gold = _read_json(base / split["gold"])
    config = _channel_config(manifest)
    limit = int(manifest["shared_config"]["limit"])
    cases = [
        _evaluate_case(fixture_path, case, config, limit)
        for case in gold["cases"]
    ]
    feedback = _evaluate_feedback(fixture_path, gold["cases"], config)
    relation_cases = [case for case in cases if case["cohort"] == "relation"]
    control_cases = [
        case
        for case in cases
        if case["cohort"] in {"direct_lookup", "directional_negative"}
    ]
    intended_ranks = [int(case["intended_rank"]) for case in cases]
    relation_mrr = _mrr([int(case["relation_rank"]) for case in relation_cases])
    relation_bm25_mrr = _mrr(
        [int(case["bm25_rank"]) for case in relation_cases]
    )
    metrics = {
        "intended_lane_mrr": _mrr(intended_ranks),
        "intended_lane_hit_at_1": sum(rank == 1 for rank in intended_ranks)
        / len(intended_ranks),
        "relation_mrr": relation_mrr,
        "relation_bm25_mrr": relation_bm25_mrr,
        "union_coverage": sum(bool(case["union_contains_expected"]) for case in cases)
        / len(cases),
        "agreement_rate": sum(bool(case["expected_in_agreement"]) for case in cases)
        / len(cases),
    }
    gate = {
        "lexical_lane_matches_isolated_bm25": all(
            case["lexical_parity"] for case in cases
        ),
        "lexical_controls_do_not_regress": all(
            int(case["lexical_rank"]) <= int(case["bm25_rank"])
            and int(case["lexical_rank"]) <= int(case["acceptable_rank"])
            for case in control_cases
        ),
        "relation_lane_matches_isolated_anchored_graph": all(
            case["relation_parity"] for case in cases
        ),
        "relation_strictly_improves_bm25": relation_mrr > relation_bm25_mrr
        and any(
            int(case["relation_rank"]) < int(case["bm25_rank"])
            for case in relation_cases
        ),
        "relation_paths_match_and_exclude_zero_hop": all(
            case["path_matched"] for case in relation_cases
        ),
        "no_cross_lane_combined_result": all(
            case["no_combined_result"] for case in cases
        ),
        "independent_trace_ids_share_query": all(
            case["independent_trace_ids_share_query"] for case in cases
        ),
        "search_does_not_change_edges": all(
            case["search_edge_unchanged"] for case in cases
        ),
        "lexical_success_does_not_change_edges": bool(
            feedback["lexical_success_isolated"]
        ),
        "relation_success_changes_only_credited_edges": bool(
            feedback["relation_success_isolated"]
        ),
        "cross_lane_trace_misuse_rejected": bool(
            feedback["cross_lane_misuse_rejected"]
        ),
        "deterministic_lane_results": all(case["deterministic"] for case in cases),
    }
    return {
        "inputs": {
            "fixture_sha256": split["fixture_sha256"],
            "gold_sha256": split["gold_sha256"],
            "provenance_sha256": split["provenance_sha256"],
        },
        "cases": cases,
        "metrics": metrics,
        "feedback": feedback,
        "gate": gate,
    }


def _evaluate_case(
    fixture_path: Path,
    case: dict[str, Any],
    config: EngineConfig,
    limit: int,
) -> dict[str, Any]:
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before_edges = _edge_snapshot(engine)
        channels = engine.search_channels(case["query"], limit=limit, now=1_000.0)
        after_edges = _edge_snapshot(engine)
    repeated = _run_channels(fixture_path, case["query"], config, limit)
    bm25 = _run_isolated_bm25(fixture_path, case["query"], config, limit)
    relation = _run_isolated_relation(fixture_path, case["query"], config, limit)
    lexical_hits = _channel_hits(channels.lexical.hits)
    relation_hits = _channel_hits(channels.relation.hits)
    expected = str(case["expected_node_id"])
    lexical_rank = _rank(lexical_hits, expected, limit)
    relation_rank = _rank(relation_hits, expected, limit)
    bm25_rank = _rank(bm25, expected, limit)
    expected_path = case.get("expected_path", [])
    observed_paths = [
        path["steps"]
        for hit in relation_hits
        if hit["node_id"] == expected
        for path in hit["paths"]
    ]
    intended_rank = (
        relation_rank
        if case["intended_channel"] == "relation"
        else lexical_rank
    )
    relation_empty_expected = bool(case.get("expected_relation_empty", False))
    path_matched = (
        any(path == expected_path for path in observed_paths)
        and all(path for path in observed_paths)
        if expected_path
        else relation_empty_expected and not relation_hits
    )
    return {
        "id": case["id"],
        "cohort": case["cohort"],
        "query": case["query"],
        "expected_node_id": expected,
        "intended_channel": case["intended_channel"],
        "acceptable_rank": int(case["acceptable_rank"]),
        "lexical_trace_id": channels.lexical.trace_id,
        "relation_trace_id": channels.relation.trace_id,
        "lexical_hits": lexical_hits,
        "relation_hits": relation_hits,
        "agreement_node_ids": list(channels.agreement_node_ids),
        "isolated_bm25_hits": bm25,
        "isolated_relation_hits": relation,
        "lexical_rank": lexical_rank,
        "relation_rank": relation_rank,
        "bm25_rank": bm25_rank,
        "intended_rank": intended_rank,
        "lexical_parity": _same_hits(lexical_hits, bm25),
        "relation_parity": _same_hits(relation_hits, relation),
        "path_matched": path_matched,
        "expected_path": expected_path,
        "observed_paths": observed_paths,
        "union_contains_expected": expected
        in {hit["node_id"] for hit in [*lexical_hits, *relation_hits]},
        "expected_in_agreement": expected in channels.agreement_node_ids,
        "no_combined_result": _no_combined_result(channels),
        "independent_trace_ids_share_query": (
            channels.lexical.trace_id != channels.relation.trace_id
            and channels.lexical.query == channels.relation.query == case["query"]
        ),
        "search_edge_unchanged": before_edges == after_edges,
        "deterministic": _same_channels(channels, repeated),
        "diagnostics": {
            "lexical": channels.lexical.diagnostics,
            "relation": channels.relation.diagnostics,
        },
    }


def _evaluate_feedback(
    fixture_path: Path,
    cases: list[dict[str, Any]],
    config: EngineConfig,
) -> dict[str, Any]:
    lexical_case = next(
        case for case in cases if case["cohort"] == "direct_lookup"
    )
    relation_case = next(case for case in cases if case["cohort"] == "relation")
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        channels = engine.search_channels(lexical_case["query"], limit=2, now=2_000.0)
        before = _edge_snapshot(engine)
        receipt = engine.record_success(
            channels.lexical.trace_id,
            [lexical_case["expected_node_id"]],
            now=2_001.0,
        )
        after = _edge_snapshot(engine)
        lexical_feedback = {
            "trace_id": channels.lexical.trace_id,
            "channel": receipt.channel,
            "reinforced_edges": [_reinforced(edge) for edge in receipt.reinforced_edges],
            "edge_changes": _edge_changes(before, after),
        }
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        channels = engine.search_channels(relation_case["query"], limit=2, now=3_000.0)
        before = _edge_snapshot(engine)
        receipt = engine.record_success(
            channels.relation.trace_id,
            [relation_case["expected_node_id"]],
            now=3_001.0,
        )
        after = _edge_snapshot(engine)
        changed = _edge_changes(before, after)
        credited = [_reinforced(edge) for edge in receipt.reinforced_edges]
        credited_keys = {
            (edge["source_id"], edge["target_id"], edge["edge_type"])
            for edge in credited
        }
        uncredited = [
            edge
            for edge in changed
            if (edge["source_id"], edge["target_id"], edge["edge_type"])
            not in credited_keys
        ]
        relation_feedback = {
            "trace_id": channels.relation.trace_id,
            "channel": receipt.channel,
            "credited_edges": credited,
            "edge_changes": changed,
            "uncredited_edge_changes": uncredited,
        }
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        channels = engine.search_channels(relation_case["query"], limit=1, now=4_000.0)
        before = _edge_snapshot(engine)
        rejected = False
        try:
            engine.record_success(
                channels.lexical.trace_id,
                [relation_case["expected_node_id"]],
                now=4_001.0,
            )
        except ValueError:
            rejected = True
        after = _edge_snapshot(engine)
        cross_lane = {
            "lexical_trace_id": channels.lexical.trace_id,
            "relation_trace_id": channels.relation.trace_id,
            "rejected": rejected,
            "edge_changes": _edge_changes(before, after),
            "feedback_count": engine.store.count_feedback(),
        }
    return {
        "lexical": lexical_feedback,
        "relation": relation_feedback,
        "cross_lane": cross_lane,
        "lexical_success_isolated": (
            lexical_feedback["channel"] == "lexical"
            and not lexical_feedback["reinforced_edges"]
            and not lexical_feedback["edge_changes"]
        ),
        "relation_success_isolated": (
            relation_feedback["channel"] == "relation"
            and bool(relation_feedback["credited_edges"])
            and not relation_feedback["uncredited_edge_changes"]
        ),
        "cross_lane_misuse_rejected": (
            cross_lane["rejected"]
            and not cross_lane["edge_changes"]
            and cross_lane["feedback_count"] == 0
        ),
    }


def _run_channels(
    fixture_path: Path, query: str, config: EngineConfig, limit: int
) -> SearchChannelsResult:
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        return engine.search_channels(query, limit=limit, now=1_000.0)


def _run_isolated_bm25(
    fixture_path: Path, query: str, config: EngineConfig, limit: int
) -> list[dict[str, Any]]:
    values = config.__dict__.copy() if hasattr(config, "__dict__") else {
        field.name: getattr(config, field.name) for field in fields(config)
    }
    values.update(
        {
            "sparse_weight": 1.0,
            "dense_weight": 0.0,
            "entry_weight": 1.0,
            "graph_weight": 0.0,
            "use_dense_retrieval": False,
            "use_graph_propagation": False,
        }
    )
    with NeuronGraphRAG(config=EngineConfig(**values)) as engine:
        load_fixture(engine, fixture_path)
        trace = engine.search(query, limit=limit, now=1_000.0)
        return [
            {
                "node_id": hit.node.node_id,
                "rank": rank,
                "channel_score": hit.sparse_raw_score,
                "bm25": hit.sparse_score,
                "bm25_raw": hit.sparse_raw_score,
                "dense": 0.0,
                "dense_raw": 0.0,
                "entry": hit.sparse_score,
                "graph_activation": 0.0,
                "paths": [],
            }
            for rank, hit in enumerate(trace.hits, start=1)
        ]


def _run_isolated_relation(
    fixture_path: Path, query: str, config: EngineConfig, limit: int
) -> list[dict[str, Any]]:
    values = {field.name: getattr(config, field.name) for field in fields(config)}
    values.update(
        {
            "activation_strategy": "anchored_local_competition",
            "entry_weight": 0.0,
            "graph_weight": 1.0,
            "graph_normalization": "none",
        }
    )
    with NeuronGraphRAG(config=EngineConfig(**values)) as engine:
        load_fixture(engine, fixture_path)
        trace = engine.search(query, limit=limit, now=1_000.0)
        hits = [
            hit
            for hit in trace.hits
            if hit.graph_activation > 0.0 and any(path.steps for path in hit.paths)
        ]
        return [
            {
                "node_id": hit.node.node_id,
                "rank": rank,
                "channel_score": hit.graph_activation,
                "bm25": hit.sparse_score,
                "bm25_raw": hit.sparse_raw_score,
                "dense": hit.dense_score,
                "dense_raw": hit.dense_raw_score,
                "entry": hit.entry_score,
                "graph_activation": hit.graph_activation,
                "paths": hit.explain()["paths"],
            }
            for rank, hit in enumerate(hits, start=1)
        ]


def _channel_hits(hits: Any) -> list[dict[str, Any]]:
    return [
        {
            "node_id": hit.node.node_id,
            "rank": hit.rank,
            "channel_score": hit.channel_score,
            "bm25": hit.sparse_score,
            "bm25_raw": hit.sparse_raw_score,
            "dense": hit.dense_score,
            "dense_raw": hit.dense_raw_score,
            "entry": hit.entry_score,
            "graph_activation": hit.graph_activation,
            "paths": hit.explain()["paths"],
        }
        for hit in hits
    ]


def _same_hits(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return left == right


def _same_channels(left: SearchChannelsResult, right: SearchChannelsResult) -> bool:
    return (
        left.query == right.query
        and _channel_hits(left.lexical.hits) == _channel_hits(right.lexical.hits)
        and _channel_hits(left.relation.hits) == _channel_hits(right.relation.hits)
        and left.agreement_node_ids == right.agreement_node_ids
        and left.lexical.diagnostics == right.lexical.diagnostics
        and left.relation.diagnostics == right.relation.diagnostics
    )


def _no_combined_result(result: SearchChannelsResult) -> bool:
    names = {field.name for field in fields(result)}
    forbidden = {"hits", "final_score", "combined_hits", "combined_rank", "winner"}
    return not names & forbidden and all(
        "final" not in hit.explain()["scores"]
        for trace in (result.lexical, result.relation)
        for hit in trace.hits
    )


def _rank(hits: list[dict[str, Any]], node_id: str, limit: int) -> int:
    return next(
        (int(hit["rank"]) for hit in hits if hit["node_id"] == node_id),
        limit + 1,
    )


def _mrr(ranks: list[int]) -> float:
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _edge_snapshot(engine: NeuronGraphRAG) -> dict[tuple[str, str, str], tuple[float, int]]:
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


def _reinforced(edge: Any) -> dict[str, Any]:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "edge_type": edge.edge_type,
        "old_weight": edge.old_weight,
        "new_weight": edge.new_weight,
    }


def _channel_config(manifest: dict[str, Any]) -> EngineConfig:
    values = {
        key: value
        for key, value in manifest["shared_config"].items()
        if key != "limit"
    }
    return EngineConfig(**values)


def _validate_gold(fixture_path: Path, gold_path: Path) -> None:
    fixture = read_fixture(fixture_path)
    gold = _read_json(gold_path)
    cases = gold.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("Channel gold must freeze exactly four cases")
    cohorts = [str(case.get("cohort")) for case in cases]
    if cohorts.count("direct_lookup") != 2 or cohorts.count("relation") != 1 or cohorts.count("directional_negative") != 1:
        raise ValueError("Channel gold must freeze the four hard-gate roles")
    node_ids = {str(node["node_id"]) for node in fixture["nodes"]}
    edges = {
        (str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"]))
        for edge in fixture["edges"]
    }
    for case in cases:
        if str(case["expected_node_id"]) not in node_ids:
            raise ValueError(f"Gold target outside fixture: {case['id']}")
        if case.get("intended_channel") not in {"lexical", "relation"}:
            raise ValueError(f"Unknown intended channel: {case['id']}")
        if not str(case.get("source_url", "")).startswith("https://github.com/"):
            raise ValueError(f"Gold case lacks public source URL: {case['id']}")
        for step in case.get("expected_path", []):
            key = (str(step["source_id"]), str(step["target_id"]), str(step["edge_type"]))
            if key not in edges:
                raise ValueError(f"Gold path outside fixture: {case['id']}")


def _doc_paths(path: Path) -> set[str]:
    fixture = read_fixture(path)
    return {
        str(node.get("metadata", {}).get("doc_path", ""))
        for node in fixture["nodes"]
        if str(node.get("metadata", {}).get("doc_path", ""))
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_sha256(path: Path) -> str:
    value = _read_json(path)
    canonical = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()

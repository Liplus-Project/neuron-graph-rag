from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .engine import EngineConfig, NeuronGraphRAG


PROTOCOL_NAME = "sibling-normalization-controlled-v1"
MANIFEST_PATH = Path(
    "tests/fixtures/sibling_normalization_controlled_v1.manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_protocol(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / MANIFEST_PATH
    manifest = load_json(manifest_path)
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise ValueError("Unexpected sibling normalization protocol")

    registered = manifest.get("registered_artifacts")
    if not isinstance(registered, dict) or not registered:
        raise ValueError("Manifest has no registered artifacts")
    actual_hashes: dict[str, str] = {}
    for relative, expected in sorted(registered.items()):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing registered artifact: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Registered artifact hash mismatch: {relative}: {actual} != {expected}"
            )
        actual_hashes[relative] = actual

    source = manifest.get("evaluated_source")
    if not isinstance(source, dict):
        raise ValueError("Manifest has no evaluated source")
    source_path = repo_root / str(source["path"])
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != source.get("sha256"):
        raise ValueError("Evaluated engine source differs from the frozen source hash")
    source_commit = str(source["commit"])
    _git(repo_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    _git(repo_root, "merge-base", "--is-ancestor", source_commit, "HEAD")

    fixture = load_json(repo_root / str(manifest["paths"]["fixture"]))
    gold = load_json(repo_root / str(manifest["paths"]["gold"]))
    schedule = load_json(repo_root / str(manifest["paths"]["schedule"]))
    gate = load_json(repo_root / str(manifest["paths"]["gate"]))
    _validate_split_identity(fixture, gold)
    _validate_schedule(schedule)
    _validate_gate(gate)
    return {
        "manifest": manifest,
        "artifact_hashes": actual_hashes,
        "source_hash": actual_source_hash,
    }


def _validate_split_identity(
    fixture: dict[str, Any], gold: dict[str, Any]
) -> None:
    fixture_splits = fixture.get("splits")
    gold_splits = gold.get("splits")
    if not isinstance(fixture_splits, dict) or set(fixture_splits) != {
        "development",
        "holdout",
    }:
        raise ValueError("Fixture must contain development and holdout splits")
    if not isinstance(gold_splits, dict) or set(gold_splits) != set(fixture_splits):
        raise ValueError("Gold split names do not match fixture split names")

    identities: dict[str, set[str]] = {}
    nodes: dict[str, set[str]] = {}
    for stage, split in fixture_splits.items():
        if not isinstance(split, dict):
            raise ValueError(f"Invalid fixture split: {stage}")
        cases = split.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Fixture split has no cases: {stage}")
        stage_clusters: set[str] = set()
        stage_nodes: set[str] = set()
        for case in cases:
            cluster_id = str(case["cluster_id"])
            if cluster_id in stage_clusters:
                raise ValueError(f"Duplicate cluster identity: {cluster_id}")
            stage_clusters.add(cluster_id)
            case_nodes = {str(node["node_id"]) for node in case["documents"]}
            if stage_nodes & case_nodes:
                raise ValueError(f"Node identity reused within {stage}")
            stage_nodes.update(case_nodes)
        identities[stage] = stage_clusters
        nodes[stage] = stage_nodes
        gold_ids = {str(case["case_id"]) for case in gold_splits[stage]["cases"]}
        fixture_ids = {str(case["case_id"]) for case in cases}
        if gold_ids != fixture_ids:
            raise ValueError(f"Gold case identities do not match {stage}")
    if identities["development"] & identities["holdout"]:
        raise ValueError("Development and holdout cluster identities overlap")
    if nodes["development"] & nodes["holdout"]:
        raise ValueError("Development and holdout node identities overlap")

    probe = fixture.get("atomicity_probe")
    if not isinstance(probe, dict):
        raise ValueError("Fixture has no atomicity probe")
    probe_nodes = {str(node["node_id"]) for node in probe["documents"]}
    if probe_nodes & (nodes["development"] | nodes["holdout"]):
        raise ValueError("Atomicity probe identities overlap an evaluation split")


def _validate_schedule(schedule: dict[str, Any]) -> None:
    normalization = schedule.get("normalization")
    if normalization != {"baseline": 0.0, "treatment": 1.0}:
        raise ValueError("The registered normalization pair must be 0.0 and 1.0")
    if schedule.get("feedback_events") != 1:
        raise ValueError("The registered schedule must contain one relation event")
    checkpoints = schedule.get("checkpoints")
    if checkpoints != [0, 1]:
        raise ValueError("The registered checkpoints must be [0, 1]")
    timestamps = schedule.get("timestamps")
    required = {
        "primary_before",
        "direct_before",
        "negative_before",
        "lexical_feedback",
        "zero_hop_feedback",
        "relation_feedback",
        "primary_after",
        "direct_after",
        "negative_after",
    }
    if not isinstance(timestamps, dict) or set(timestamps) != required:
        raise ValueError("Registered timestamp schedule is incomplete")


def _validate_gate(gate: dict[str, Any]) -> None:
    expected = {
        "relation_headroom_strict_improvement",
        "relation_ceiling_non_regression",
        "direct_non_regression",
        "lexical_non_regression",
        "directional_negative_non_regression",
        "credited_path_match",
        "mutation_scope",
        "failed_transaction_atomicity",
        "deterministic_replay",
    }
    gates = gate.get("gates")
    if not isinstance(gates, list):
        raise ValueError("Gate artifact has no gates")
    actual = {str(item["id"]) for item in gates}
    if actual != expected:
        raise ValueError("Gate set differs from the registered hard gates")


def _engine_config(normalization: float) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        entry_weight=1.0,
        graph_weight=0.0,
        seed_count=1,
        max_hops=1,
        hop_decay=0.7,
        feedback_learning_rate=0.2,
        sibling_feedback_normalization=normalization,
        activation_budget=10.0,
        recurrent_steps=1,
    )


def _populate(engine: NeuronGraphRAG, case: dict[str, Any]) -> None:
    for node in case["documents"]:
        engine.add_document(
            str(node["node_id"]),
            str(node["text"]),
            metadata={"cluster_id": case["cluster_id"]},
        )
    for edge in case["edges"]:
        engine.add_edge(
            str(edge["source_id"]),
            str(edge["target_id"]),
            str(edge["edge_type"]),
            weight=float(edge["weight"]),
        )


def _edge_key(source_id: str, target_id: str, edge_type: str) -> str:
    return "|".join((source_id, target_id, edge_type))


def _edge_snapshot(engine: NeuronGraphRAG) -> dict[str, float]:
    return {
        _edge_key(edge.source_id, edge.target_id, edge.edge_type): edge.weight
        for edge in engine.store.list_edges()
    }


def _rank(hits: Any, node_id: str) -> int | None:
    for hit in hits:
        if hit.node.node_id == node_id:
            return int(hit.rank)
    return None


def _project_path(hit: Any) -> list[dict[str, str]]:
    if not hit.paths:
        return []
    selected = max(
        hit.paths,
        key=lambda path: (path.contribution, path.seed_id),
    )
    return [
        {
            "source_id": step.source_id,
            "target_id": step.target_id,
            "edge_type": step.edge_type,
        }
        for step in selected.steps
    ]


def _changed_edges(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, dict[str, float]]:
    return {
        key: {"before": before[key], "after": after[key]}
        for key in sorted(before)
        if before[key] != after[key]
    }


def _evaluate_condition(
    case: dict[str, Any],
    case_gold: dict[str, Any],
    schedule: dict[str, Any],
    normalization: float,
) -> dict[str, Any]:
    expected_node = str(case_gold["expected_relation_node"])
    source_node = str(case_gold["expected_source_node"])
    used_nodes = [str(value) for value in case_gold["used_node_ids"]]
    timestamps = schedule["timestamps"]
    with NeuronGraphRAG(config=_engine_config(normalization)) as engine:
        _populate(engine, case)
        initial_edges = _edge_snapshot(engine)
        primary_before = engine.search_channels(
            str(case["primary_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["primary_before"]),
        )
        direct_before = engine.search(
            str(case["primary_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["direct_before"]),
        )
        negative_before = engine.search_channels(
            str(case["directional_negative_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["negative_before"]),
        )

        lexical_receipt = engine.record_success(
            primary_before.lexical.trace_id,
            [source_node],
            now=float(timestamps["lexical_feedback"]),
        )
        after_lexical = _edge_snapshot(engine)
        zero_hop_receipt = engine.record_success(
            direct_before.trace_id,
            [source_node],
            now=float(timestamps["zero_hop_feedback"]),
        )
        after_zero_hop = _edge_snapshot(engine)
        relation_receipt = engine.record_success(
            primary_before.relation.trace_id,
            used_nodes,
            now=float(timestamps["relation_feedback"]),
        )
        after_relation = _edge_snapshot(engine)

        primary_after = engine.search_channels(
            str(case["primary_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["primary_after"]),
        )
        direct_after = engine.search(
            str(case["primary_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["direct_after"]),
        )
        negative_after = engine.search_channels(
            str(case["directional_negative_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["negative_after"]),
        )

        path_by_node: dict[str, list[dict[str, str]]] = {}
        for node_id in used_nodes:
            hit = next(
                hit
                for hit in primary_before.relation.hits
                if hit.node.node_id == node_id
            )
            path_by_node[node_id] = _project_path(hit)

        return {
            "normalization": normalization,
            "trace_provenance": {
                "lexical_trace_id": primary_before.lexical.trace_id,
                "relation_trace_id": primary_before.relation.trace_id,
                "trace_ids_distinct": (
                    primary_before.lexical.trace_id
                    != primary_before.relation.trace_id
                ),
                "relation_feedback_trace_id": relation_receipt.trace_id,
                "relation_feedback_used_same_trace": (
                    relation_receipt.trace_id == primary_before.relation.trace_id
                ),
                "relation_feedback_channel": relation_receipt.channel,
            },
            "relation": {
                "expected_node": expected_node,
                "rank_before": _rank(primary_before.relation.hits, expected_node),
                "rank_after": _rank(primary_after.relation.hits, expected_node),
                "order_before": [hit.node.node_id for hit in primary_before.relation.hits],
                "order_after": [hit.node.node_id for hit in primary_after.relation.hits],
                "credited_paths": path_by_node,
            },
            "direct": {
                "expected_node": source_node,
                "rank_before": _rank(direct_before.hits, source_node),
                "rank_after": _rank(direct_after.hits, source_node),
            },
            "lexical": {
                "expected_node": source_node,
                "rank_before": _rank(primary_before.lexical.hits, source_node),
                "rank_after": _rank(primary_after.lexical.hits, source_node),
            },
            "directional_negative": {
                "forbidden_node": source_node,
                "order_before": [hit.node.node_id for hit in negative_before.relation.hits],
                "order_after": [hit.node.node_id for hit in negative_after.relation.hits],
            },
            "mutation": {
                "initial": initial_edges,
                "after_lexical": after_lexical,
                "after_zero_hop": after_zero_hop,
                "after_relation": after_relation,
                "lexical_changed": _changed_edges(initial_edges, after_lexical),
                "zero_hop_changed": _changed_edges(after_lexical, after_zero_hop),
                "relation_changed": _changed_edges(after_zero_hop, after_relation),
                "reinforced_edges": [
                    _edge_key(edge.source_id, edge.target_id, edge.edge_type)
                    for edge in relation_receipt.reinforced_edges
                ],
                "normalized_sibling_edges": [
                    _edge_key(edge.source_id, edge.target_id, edge.edge_type)
                    for edge in relation_receipt.normalized_sibling_edges
                ],
                "lexical_receipt_edge_counts": [
                    len(lexical_receipt.reinforced_edges),
                    len(lexical_receipt.normalized_sibling_edges),
                ],
                "zero_hop_receipt_edge_counts": [
                    len(zero_hop_receipt.reinforced_edges),
                    len(zero_hop_receipt.normalized_sibling_edges),
                ],
            },
        }


def _stable_condition(value: dict[str, Any]) -> dict[str, Any]:
    stable = dict(value)
    stable.pop("trace_provenance")
    return stable


def _evaluate_atomicity_probe(
    probe: dict[str, Any], schedule: dict[str, Any]
) -> dict[str, Any]:
    timestamps = schedule["timestamps"]
    with NeuronGraphRAG(config=_engine_config(1.0)) as engine:
        _populate(engine, probe)
        result = engine.search_channels(
            str(probe["primary_query"]),
            limit=int(schedule["limit"]),
            now=float(timestamps["primary_before"]),
        )
        removed = probe["removed_edge_before_feedback"]
        engine.store.connection.execute(
            "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
            (
                removed["source_id"],
                removed["target_id"],
                removed["edge_type"],
            ),
        )
        engine.store.connection.commit()
        before = _edge_snapshot(engine)
        feedback_before = engine.store.count_feedback()
        error_type: str | None = None
        try:
            engine.record_success(
                result.relation.trace_id,
                [str(value) for value in probe["used_node_ids"]],
                now=float(timestamps["relation_feedback"]),
            )
        except Exception as error:  # the registered gate checks the exact type
            error_type = type(error).__name__
        after = _edge_snapshot(engine)
        return {
            "expected_error": probe["expected_error"],
            "actual_error": error_type,
            "edges_before": before,
            "edges_after": after,
            "feedback_before": feedback_before,
            "feedback_after": engine.store.count_feedback(),
        }


def _rank_non_regression(before: int | None, after: int | None) -> bool:
    return before is not None and after is not None and after <= before


def _evaluate_gates(
    cases: list[dict[str, Any]],
    gold_cases: dict[str, dict[str, Any]],
    atomicity: dict[str, Any],
    replay_equal: bool,
) -> dict[str, bool]:
    headroom = next(case for case in cases if case["role"] == "headroom")
    ceiling = next(case for case in cases if case["role"] == "ceiling")
    headroom_pass = (
        headroom["treatment"]["relation"]["rank_after"]
        < headroom["baseline"]["relation"]["rank_after"]
    )
    ceiling_pass = _rank_non_regression(
        ceiling["treatment"]["relation"]["rank_before"],
        ceiling["treatment"]["relation"]["rank_after"],
    )
    direct_pass = all(
        _rank_non_regression(
            condition["direct"]["rank_before"],
            condition["direct"]["rank_after"],
        )
        for case in cases
        for condition in (case["baseline"], case["treatment"])
    )
    lexical_pass = all(
        _rank_non_regression(
            condition["lexical"]["rank_before"],
            condition["lexical"]["rank_after"],
        )
        and condition["mutation"]["lexical_changed"] == {}
        and condition["mutation"]["zero_hop_changed"] == {}
        for case in cases
        for condition in (case["baseline"], case["treatment"])
    )
    negative_pass = all(
        condition["directional_negative"]["forbidden_node"]
        not in condition["directional_negative"]["order_before"]
        and condition["directional_negative"]["forbidden_node"]
        not in condition["directional_negative"]["order_after"]
        and condition["directional_negative"]["order_after"]
        == condition["directional_negative"]["order_before"]
        for case in cases
        for condition in (case["baseline"], case["treatment"])
    )
    path_pass = all(
        condition["relation"]["credited_paths"] == gold_cases[case["case_id"]]["paths"]
        and condition["trace_provenance"]["trace_ids_distinct"]
        and condition["trace_provenance"]["relation_feedback_used_same_trace"]
        and condition["trace_provenance"]["relation_feedback_channel"] == "relation"
        for case in cases
        for condition in (case["baseline"], case["treatment"])
    )

    mutation_pass = True
    for case in cases:
        gold = gold_cases[case["case_id"]]
        credited = set(gold["credited_edges"])
        sibling = set(gold["uncredited_sibling_edges"])
        unrelated = set(gold["unrelated_edges"])
        baseline = case["baseline"]["mutation"]
        treatment = case["treatment"]["mutation"]
        if set(baseline["relation_changed"]) != credited:
            mutation_pass = False
        if set(treatment["relation_changed"]) != credited | sibling:
            mutation_pass = False
        if set(baseline["reinforced_edges"]) != credited:
            mutation_pass = False
        if set(treatment["reinforced_edges"]) != credited:
            mutation_pass = False
        if set(baseline["normalized_sibling_edges"]):
            mutation_pass = False
        if set(treatment["normalized_sibling_edges"]) != sibling:
            mutation_pass = False
        for key in credited:
            baseline_delta = (
                baseline["after_relation"][key] - baseline["after_zero_hop"][key]
            )
            treatment_delta = (
                treatment["after_relation"][key] - treatment["after_zero_hop"][key]
            )
            if abs(baseline_delta - treatment_delta) > 1e-12:
                mutation_pass = False
        for key in sibling:
            if baseline["after_relation"][key] != baseline["after_zero_hop"][key]:
                mutation_pass = False
            if treatment["after_relation"][key] >= treatment["after_zero_hop"][key]:
                mutation_pass = False
        for key in unrelated:
            if baseline["after_relation"][key] != baseline["after_zero_hop"][key]:
                mutation_pass = False
            if treatment["after_relation"][key] != treatment["after_zero_hop"][key]:
                mutation_pass = False

    atomicity_pass = (
        atomicity["actual_error"] == atomicity["expected_error"]
        and atomicity["edges_after"] == atomicity["edges_before"]
        and atomicity["feedback_after"] == atomicity["feedback_before"]
    )
    return {
        "relation_headroom_strict_improvement": headroom_pass,
        "relation_ceiling_non_regression": ceiling_pass,
        "direct_non_regression": direct_pass,
        "lexical_non_regression": lexical_pass,
        "directional_negative_non_regression": negative_pass,
        "credited_path_match": path_pass,
        "mutation_scope": mutation_pass,
        "failed_transaction_atomicity": atomicity_pass,
        "deterministic_replay": replay_equal,
    }


def evaluate_stage(repo_root: Path, stage: str) -> dict[str, Any]:
    validated = validate_protocol(repo_root)
    manifest = validated["manifest"]
    if stage not in {"development", "holdout"}:
        raise ValueError("stage must be development or holdout")
    fixture = load_json(repo_root / str(manifest["paths"]["fixture"]))
    gold = load_json(repo_root / str(manifest["paths"]["gold"]))
    schedule = load_json(repo_root / str(manifest["paths"]["schedule"]))
    cases_by_id = {
        str(case["case_id"]): case for case in fixture["splits"][stage]["cases"]
    }
    gold_cases = {
        str(case["case_id"]): case for case in gold["splits"][stage]["cases"]
    }
    cases: list[dict[str, Any]] = []
    replay_equal = True
    for case_id in sorted(cases_by_id):
        case = cases_by_id[case_id]
        case_gold = gold_cases[case_id]
        conditions: dict[str, Any] = {}
        for condition_name, normalization in schedule["normalization"].items():
            first = _evaluate_condition(case, case_gold, schedule, float(normalization))
            replay = _evaluate_condition(case, case_gold, schedule, float(normalization))
            replay_equal = replay_equal and (
                _stable_condition(first) == _stable_condition(replay)
            )
            conditions[condition_name] = first
        cases.append(
            {
                "case_id": case_id,
                "cluster_id": case["cluster_id"],
                "role": case_gold["role"],
                **conditions,
            }
        )
    atomicity = _evaluate_atomicity_probe(fixture["atomicity_probe"], schedule)
    gates = _evaluate_gates(cases, gold_cases, atomicity, replay_equal)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "stage": stage,
        "evaluated_source": manifest["evaluated_source"],
        "freeze_commit": _git(repo_root, "rev-parse", "HEAD"),
        "registered_artifact_hashes": validated["artifact_hashes"],
        "schedule": schedule,
        "cases": cases,
        "atomicity_probe": atomicity,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "claim_boundary": manifest["claim_boundary"],
    }


def run_registered(repo_root: Path, stage: str) -> Path:
    validated = validate_protocol(repo_root)
    manifest = validated["manifest"]
    outputs = manifest["exclusive_outputs"]
    output_path = repo_root / str(outputs[stage])
    if output_path.exists():
        raise FileExistsError(f"Registered output already exists: {output_path}")
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("Registered evaluation requires a clean worktree")
    head = _git(repo_root, "rev-parse", "HEAD")
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError("Registered evaluation requires the frozen commit to be pushed")
    if stage == "development" and (repo_root / str(outputs["holdout"])).exists():
        raise RuntimeError("Holdout output cannot predate development")
    if stage == "holdout":
        development_path = repo_root / str(outputs["development"])
        if not development_path.is_file():
            raise RuntimeError("Development output is required before holdout")
        development = load_json(development_path)
        if not development.get("all_gates_passed"):
            raise RuntimeError("Development gates did not authorize holdout")
        if development.get("registered_artifact_hashes") != validated["artifact_hashes"]:
            raise RuntimeError("Protocol artifacts changed after development")

    result = evaluate_stage(repo_root, stage)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def validate_observed_outputs(repo_root: Path) -> None:
    validated = validate_protocol(repo_root)
    manifest = validated["manifest"]
    outputs = manifest["exclusive_outputs"]
    development_path = repo_root / str(outputs["development"])
    holdout_path = repo_root / str(outputs["holdout"])
    if holdout_path.exists() and not development_path.exists():
        raise ValueError("Holdout output exists without development output")
    for stage, path in (
        ("development", development_path),
        ("holdout", holdout_path),
    ):
        if not path.exists():
            continue
        result = load_json(path)
        if result.get("protocol") != PROTOCOL_NAME or result.get("stage") != stage:
            raise ValueError(f"Unexpected observed output identity: {path}")
        if result.get("registered_artifact_hashes") != validated["artifact_hashes"]:
            raise ValueError(f"Observed output protocol hash mismatch: {path}")
    if holdout_path.exists():
        development = load_json(development_path)
        if not development.get("all_gates_passed"):
            raise ValueError("Holdout was opened without a passing development gate")

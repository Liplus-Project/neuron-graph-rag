"""One-shot evaluation for the fresh repository-native feedback protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus_integrity import verify_source_sha256
from .engine import EngineConfig, NeuronGraphRAG


FRESH_EVALUATION_SCHEMA_VERSION = 1
PATH_IDENTITY_FIELDS = ("source_id", "target_id", "edge_type")


def read_fresh_native_feedback_manifest(path: str | Path) -> dict[str, Any]:
    """Read and verify every result-free artifact before opening a split."""
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != FRESH_EVALUATION_SCHEMA_VERSION:
        raise ValueError("Unsupported fresh native feedback schema version")
    if manifest.get("candidate_id") != "fresh-native-credited-feedback-v2":
        raise ValueError("Fresh native feedback candidate is not frozen")

    artifacts = _read_frozen_artifacts(manifest_path, manifest)
    _validate_result_free_audit(artifacts["fixture"], artifacts["audit"])
    _validate_gold_and_schedule(artifacts["fixture"], artifacts["gold"], artifacts["schedule"])
    _validate_gate(artifacts["gate"])
    return manifest


def run_fresh_native_feedback_development(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_fresh_native_feedback_manifest(manifest_path)
    artifacts = _read_frozen_artifacts(manifest_path, manifest)
    integrity = _verify_source_integrity(manifest_path, artifacts["fixture"])
    result = _evaluate_split(manifest_path, manifest, artifacts, "development")
    gate = _gate_result(result, integrity, artifacts["audit"], artifacts["gate"], "development")
    passed = all(gate.values())
    return {
        "schema_version": FRESH_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": manifest["evaluation_id"],
        "stage": "development",
        "manifest_sha256": _raw_sha256(manifest_path),
        "source_integrity": integrity,
        **result,
        "gate": gate,
        "gate_passed": passed,
        "holdout_status": "opened_by_development_gate" if passed else "not_opened_development_gate_failed",
    }


def run_fresh_native_feedback_holdout(
    path: str | Path, development_result_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_fresh_native_feedback_manifest(manifest_path)
    development_path = Path(development_result_path)
    development = _read_json(development_path)
    if development.get("stage") != "development":
        raise ValueError("Holdout requires a development result")
    if development.get("manifest_sha256") != _raw_sha256(manifest_path):
        raise ValueError("Development result does not match frozen manifest")
    if not development.get("gate_passed"):
        raise ValueError("Stop rule forbids opening fresh native feedback holdout")

    artifacts = _read_frozen_artifacts(manifest_path, manifest)
    integrity = _verify_source_integrity(manifest_path, artifacts["fixture"])
    result = _evaluate_split(manifest_path, manifest, artifacts, "holdout")
    gate = _gate_result(result, integrity, artifacts["audit"], artifacts["gate"], "holdout")
    passed = all(gate.values())
    return {
        "schema_version": FRESH_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": manifest["evaluation_id"],
        "stage": "holdout",
        "manifest_sha256": _raw_sha256(manifest_path),
        "development_result_sha256": _raw_sha256(development_path),
        "holdout_open_count": 1,
        "source_integrity": integrity,
        **result,
        "gate": gate,
        "gate_passed": passed,
        "decision": {
            "adopted": passed,
            "reason": "全ての固定 holdout gate を通過した" if passed else "固定 holdout gate が失敗したため採用しない",
        },
    }


def write_fresh_native_feedback_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_frozen_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("fixture", "gold", "schedule", "gate", "audit"):
        record = manifest["artifacts"].get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Missing frozen {name} record")
        artifact = manifest_path.parent / str(record["path"])
        if _raw_sha256(artifact) != record["sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
        loaded[name] = _read_json(artifact)
    return loaded


def _validate_result_free_audit(fixture: dict[str, Any], audit: dict[str, Any]) -> None:
    if audit.get("result_free") is not True or audit.get("passed") is not True:
        raise ValueError("Result-free audit did not pass")
    fixture_splits = fixture.get("splits")
    audit_splits = audit.get("splits")
    if not isinstance(fixture_splits, dict) or not isinstance(audit_splits, dict):
        raise ValueError("Split audit is missing")
    observed: dict[str, dict[str, set[tuple[str, ...] | str]]] = {}
    for stage in ("development", "holdout"):
        split = fixture_splits.get(stage)
        if not isinstance(split, dict):
            raise ValueError(f"Fixture split is missing: {stage}")
        nodes = split.get("nodes")
        edges = split.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError(f"Fixture split arrays are missing: {stage}")
        identity = {
            "node_ids": {str(node["node_id"]) for node in nodes},
            "document_paths": {str(node["document_path"]) for node in nodes},
            "source_urls": {str(node["source_url"]) for node in nodes},
            "credited_edges": {
                tuple(str(edge[field]) for field in PATH_IDENTITY_FIELDS) for edge in edges
            },
        }
        if any(len(values) != len(nodes if name != "credited_edges" else edges) for name, values in identity.items()):
            raise ValueError(f"Split contains duplicate identity: {stage}")
        expected = audit_splits.get(stage)
        expected_identity = {} if not isinstance(expected, dict) else {
            "node_ids": set(map(str, expected.get("node_ids", []))),
            "document_paths": set(map(str, expected.get("document_paths", []))),
            "source_urls": set(map(str, expected.get("source_urls", []))),
            "credited_edges": {
                tuple(map(str, edge)) for edge in expected.get("credited_edges", [])
            },
        }
        if expected_identity != identity:
            raise ValueError(f"Audit identity does not match fixture: {stage}")
        observed[stage] = identity
    for name in ("node_ids", "document_paths", "source_urls", "credited_edges"):
        if observed["development"][name] & observed["holdout"][name]:
            raise ValueError(f"Development and holdout overlap: {name}")


def _validate_gold_and_schedule(
    fixture: dict[str, Any], gold: dict[str, Any], schedule: dict[str, Any]
) -> None:
    if gold.get("schema_version") != FRESH_EVALUATION_SCHEMA_VERSION or schedule.get("schema_version") != FRESH_EVALUATION_SCHEMA_VERSION:
        raise ValueError("Gold or schedule schema version is unsupported")
    for stage in ("development", "holdout"):
        nodes = {str(node["node_id"]) for node in fixture["splits"][stage]["nodes"]}
        edges = {
            tuple(str(edge[field]) for field in PATH_IDENTITY_FIELDS)
            for edge in fixture["splits"][stage]["edges"]
        }
        cases = gold.get("splits", {}).get(stage, {}).get("cases")
        event = schedule.get("splits", {}).get(stage)
        if not isinstance(cases, list) or {case.get("role") for case in cases} != {"headroom", "control", "ceiling"}:
            raise ValueError(f"Gold must define headroom, control, and ceiling: {stage}")
        if not isinstance(event, dict) or event.get("channel") != "relation":
            raise ValueError(f"Feedback schedule must use relation channel: {stage}")
        if str(event.get("used_node_id")) not in nodes:
            raise ValueError(f"Feedback target is outside fixture: {stage}")
        for case in cases:
            if str(case.get("expected_node_id")) not in nodes or not str(case.get("query", "")).strip():
                raise ValueError(f"Gold case is invalid: {stage}")
            expected_path = case.get("expected_path")
            if not isinstance(expected_path, list) or any(
                set(step) != set(PATH_IDENTITY_FIELDS) for step in expected_path
            ):
                raise ValueError(f"Gold path identity is malformed: {stage}")
            for step in expected_path:
                if tuple(str(step[field]) for field in PATH_IDENTITY_FIELDS) not in edges:
                    raise ValueError(f"Gold path is outside fixture: {stage}")
        if event.get("query") in {case["query"] for case in cases}:
            raise ValueError(f"Feedback query must be independent of scoring queries: {stage}")
        credited = {
            tuple(str(edge[field]) for field in PATH_IDENTITY_FIELDS)
            for edge in event.get("credited_edges", [])
        }
        if not credited or not credited <= edges:
            raise ValueError(f"Credited edges are invalid: {stage}")


def _validate_gate(gate: dict[str, Any]) -> None:
    required = {
        "source_integrity",
        "split_disjointness",
        "headroom_relation_mrr",
        "control_relation_mrr",
        "ceiling_relation_mrr",
        "control_edge_mutation",
        "treatment_credited_edges",
    }
    if gate.get("schema_version") != FRESH_EVALUATION_SCHEMA_VERSION:
        raise ValueError("Gate schema version is unsupported")
    if any(set(gate.get("splits", {}).get(stage, {})) != required for stage in ("development", "holdout")):
        raise ValueError("Gate does not freeze all required checks")


def _verify_source_integrity(manifest_path: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    root = (manifest_path.parent / str(fixture["repository_root"])).resolve()
    checks: list[dict[str, str | bool | None]] = []
    for stage in ("development", "holdout"):
        for node in fixture["splits"][stage]["nodes"]:
            verification = verify_source_sha256(
                root / str(node["document_path"]), str(node["source_sha256"])
            )
            checks.append(
                {
                    "stage": stage,
                    "node_id": str(node["node_id"]),
                    "document_path": str(node["document_path"]),
                    "accepted": verification.accepted,
                    "decision": verification.decision,
                    "reason": verification.reason,
                    "expected_sha256": verification.expected_sha256,
                    "raw_sha256": verification.raw_sha256,
                    "alternate_sha256": verification.alternate_sha256,
                }
            )
    return {"passed": all(bool(item["accepted"]) for item in checks), "checks": checks}


def _evaluate_split(
    manifest_path: Path, manifest: dict[str, Any], artifacts: dict[str, dict[str, Any]], stage: str
) -> dict[str, Any]:
    fixture = artifacts["fixture"]
    gold = artifacts["gold"]["splits"][stage]
    schedule = artifacts["schedule"]["splits"][stage]
    config_values = dict(manifest["shared_config"])
    limit = int(config_values.pop("limit"))
    config = EngineConfig(**config_values)
    control = _run_group(manifest_path, fixture, gold, schedule, config, limit, mutate=False)
    treatment = _run_group(manifest_path, fixture, gold, schedule, config, limit, mutate=True)
    return {
        "inputs": {name: manifest["artifacts"][name]["sha256"] for name in manifest["artifacts"]},
        "registered_runs": manifest["registered_runs"][stage],
        "control": control,
        "treatment": treatment,
        "metrics": {
            "control_headroom_relation_mrr": _role_mrr(control["cases"], "headroom"),
            "treatment_headroom_relation_mrr": _role_mrr(treatment["cases"], "headroom"),
            "control_control_relation_mrr": _role_mrr(control["cases"], "control"),
            "treatment_control_relation_mrr": _role_mrr(treatment["cases"], "control"),
            "control_ceiling_relation_mrr": _role_mrr(control["cases"], "ceiling"),
            "treatment_ceiling_relation_mrr": _role_mrr(treatment["cases"], "ceiling"),
        },
    }


def _run_group(
    manifest_path: Path,
    fixture: dict[str, Any],
    gold: dict[str, Any],
    schedule: dict[str, Any],
    config: EngineConfig,
    limit: int,
    *,
    mutate: bool,
) -> dict[str, Any]:
    with NeuronGraphRAG(config=config) as engine:
        _load_source_backed_split(engine, manifest_path, fixture, schedule["stage"])
        before = _edge_snapshot(engine)
        trace = engine.search_channels(schedule["query"], limit=limit, now=float(schedule["now"]))
        if trace.relation.trace_id != trace.relation.trace_id or schedule["used_node_id"] not in {hit.node.node_id for hit in trace.relation.hits}:
            raise ValueError("Frozen feedback target was not retrieved by relation trace")
        if mutate:
            receipt = engine.record_success(trace.relation.trace_id, [schedule["used_node_id"]], now=float(schedule["now"]) + 1.0)
            credited = [_reinforced(edge) for edge in receipt.reinforced_edges]
            if receipt.channel != "relation":
                raise ValueError("Treatment feedback must be attributed to relation trace")
        else:
            engine.store.apply_success_feedback(
                "fresh-native-control", trace.relation.trace_id, float(schedule["now"]) + 1.0,
                [schedule["used_node_id"]], (),
            )
            credited = []
        after = _edge_snapshot(engine)
        cases = [_score_case(engine, case, limit) for case in gold["cases"]]
    changed = _edge_changes(before, after)
    credited_keys = {tuple(edge[field] for field in PATH_IDENTITY_FIELDS) for edge in credited}
    expected_keys = {
        tuple(str(edge[field]) for field in PATH_IDENTITY_FIELDS)
        for edge in schedule["credited_edges"]
    }
    changed_keys = {tuple(edge[field] for field in PATH_IDENTITY_FIELDS) for edge in changed}
    return {
        "cases": cases,
        "feedback": {
            "recorded": True,
            "feedback_count": 1,
            "credited_edges": credited,
            "edge_changes": changed,
            "no_edge_mutation": not changed,
            "credited_only": bool(credited) and credited_keys == expected_keys and changed_keys == expected_keys,
        },
    }


def _load_source_backed_split(engine: NeuronGraphRAG, manifest_path: Path, fixture: dict[str, Any], stage: str) -> None:
    root = (manifest_path.parent / str(fixture["repository_root"])).resolve()
    split = fixture["splits"][stage]
    for node in split["nodes"]:
        engine.add_document(
            str(node["node_id"]),
            (root / str(node["document_path"])).read_text(encoding="utf-8"),
            metadata={"source_url": node["source_url"], "document_path": node["document_path"]},
        )
    for edge in split["edges"]:
        engine.add_edge(
            str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"]),
            weight=float(edge.get("weight", 1.0)), factuality=float(edge.get("factuality", 1.0)),
        )


def _score_case(engine: NeuronGraphRAG, case: dict[str, Any], limit: int) -> dict[str, Any]:
    trace = engine.search_channels(case["query"], limit=limit, now=float(case["now"]))
    expected = str(case["expected_node_id"])
    hit = next((item for item in trace.relation.hits if item.node.node_id == expected), None)
    paths = [] if hit is None else [
        [{field: str(step[field]) for field in PATH_IDENTITY_FIELDS} for step in path["steps"]]
        for path in hit.explain()["paths"]
    ]
    expected_path = case["expected_path"]
    return {
        "id": case["id"],
        "role": case["role"],
        "expected_node_id": expected,
        "relation_rank": limit + 1 if hit is None else hit.rank,
        "relation_found": hit is not None,
        "expected_path_matched": expected_path in paths,
        "observed_paths": paths,
    }


def _gate_result(result: dict[str, Any], integrity: dict[str, Any], audit: dict[str, Any], gate: dict[str, Any], stage: str) -> dict[str, bool]:
    metrics = result["metrics"]
    return {
        "source_integrity": bool(integrity["passed"]),
        "split_disjointness": bool(audit["passed"]),
        "headroom_relation_mrr": metrics["treatment_headroom_relation_mrr"] > metrics["control_headroom_relation_mrr"],
        "control_relation_mrr": metrics["treatment_control_relation_mrr"] >= metrics["control_control_relation_mrr"],
        "ceiling_relation_mrr": metrics["treatment_ceiling_relation_mrr"] >= metrics["control_ceiling_relation_mrr"],
        "control_edge_mutation": result["control"]["feedback"]["no_edge_mutation"],
        "treatment_credited_edges": result["treatment"]["feedback"]["credited_only"],
    }


def _role_mrr(cases: list[dict[str, Any]], role: str) -> float:
    ranks = [int(case["relation_rank"]) for case in cases if case["role"] == role]
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _edge_snapshot(engine: NeuronGraphRAG) -> dict[tuple[str, str, str], tuple[float, int]]:
    return {(edge.source_id, edge.target_id, edge.edge_type): (edge.weight, edge.reinforced_count) for edge in engine.store.list_edges()}


def _edge_changes(before: dict[tuple[str, str, str], tuple[float, int]], after: dict[tuple[str, str, str], tuple[float, int]]) -> list[dict[str, Any]]:
    return [
        {"source_id": key[0], "target_id": key[1], "edge_type": key[2], "before_weight": before[key][0], "after_weight": after[key][0], "before_reinforced_count": before[key][1], "after_reinforced_count": after[key][1]}
        for key in sorted(before) if before[key] != after[key]
    ]


def _reinforced(edge: Any) -> dict[str, Any]:
    return {"source_id": edge.source_id, "target_id": edge.target_id, "edge_type": edge.edge_type, "old_weight": edge.old_weight, "new_weight": edge.new_weight}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _raw_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

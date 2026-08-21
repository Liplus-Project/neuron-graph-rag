"""Frozen engine-backed longitudinal feedback trajectory evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .corpus_integrity import (
    HistoricalSourceSnapshot,
    verify_historical_source_hashes,
    verify_manifest_source_hashes,
    verify_source_bytes,
)
from .engine import EngineConfig, NeuronGraphRAG


SCHEMA_VERSION = 1
CANDIDATE_ID = "repository-native-controlled-v3-engine-trajectory"
SOURCE_COMMIT = "94c8bc250b7352e3009eeee1b353c3aec677bfb7"
PATH_FIELDS = ("source_id", "target_id", "edge_type")
STAGES = ("development", "holdout")
ROLES = ("headroom", "control", "ceiling")
REQUIRED_GATES = {
    "source_integrity",
    "split_cluster_identity",
    "manifest_artifact_hashes",
    "registered_run_count",
    "relation_channel_only",
    "expected_paths",
    "feedback_checkpoint_counts",
    "headroom_strict_improvement_0_to_10",
    "headroom_intermediate_non_regression",
    "control_case_non_regression",
    "ceiling_case_non_regression",
    "control_arm_non_regression",
    "control_feedback_recorded_without_mutation",
    "treatment_record_success_relation_trace",
    "treatment_credited_edges_only",
    "exclusive_output",
}
RELATIVE_LINK = re.compile(r"\[[^]]+\]\(([^)]+\.md)\)")


def read_engine_feedback_trajectory_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate the complete result-free registration."""
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported engine feedback trajectory schema version")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("Engine feedback trajectory candidate is not frozen")
    if manifest.get("source_corpus_commit") != SOURCE_COMMIT:
        raise ValueError("Source corpus commit is not frozen")
    if manifest.get("checkpoints") != [0, 1, 3, 10]:
        raise ValueError("Feedback checkpoints are not frozen")

    artifacts = _read_frozen_artifacts(manifest_path, manifest)
    _validate_fixture_and_audit(artifacts["fixture"], artifacts["audit"])
    _validate_gold_and_schedule(
        artifacts["fixture"], artifacts["gold"], artifacts["schedule"]
    )
    _validate_gate(artifacts["gate"])
    _validate_registration(manifest)
    return manifest


def run_engine_feedback_trajectory_development(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_engine_feedback_trajectory_manifest(manifest_path)
    _require_output_absence(manifest_path, manifest, "development")
    if _result_path(manifest_path, manifest, "holdout").exists():
        raise ValueError("Holdout output exists before the registered development run")
    return _run_stage(manifest_path, manifest, "development")


def run_engine_feedback_trajectory_holdout(
    path: str | Path, development_result_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_engine_feedback_trajectory_manifest(manifest_path)
    expected_development_path = _result_path(manifest_path, manifest, "development")
    development_path = Path(development_result_path)
    if development_path.resolve() != expected_development_path.resolve():
        raise ValueError("Holdout requires the registered development output")
    development = _read_json(development_path)
    if development.get("stage") != "development":
        raise ValueError("Holdout requires a development result")
    if development.get("manifest_sha256") != _raw_sha256(manifest_path):
        raise ValueError("Development result does not match the frozen manifest")
    if not development.get("gate_passed"):
        raise ValueError("Stop rule forbids opening engine feedback trajectory holdout")
    _require_output_absence(manifest_path, manifest, "holdout")
    result = _run_stage(manifest_path, manifest, "holdout")
    result["development_result_sha256"] = _raw_sha256(development_path)
    result["holdout_open_count"] = 1
    result["decision"] = {
        "adopted": bool(result["gate_passed"]),
        "reason": (
            "固定した全 holdout gate を通過した"
            if result["gate_passed"]
            else "固定した holdout gate が失敗したため採用しない"
        ),
    }
    return result


def write_engine_feedback_trajectory_result(
    path: str | Path, result: dict[str, Any]
) -> None:
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to overwrite an observed result: {output}")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_stage(
    manifest_path: Path, manifest: dict[str, Any], stage: str
) -> dict[str, Any]:
    artifacts = _read_frozen_artifacts(manifest_path, manifest)
    integrity = _verify_source_integrity(manifest_path, artifacts["fixture"], stage)
    control = _run_arm(
        manifest_path, manifest, artifacts, stage, arm="control"
    )
    treatment = _run_arm(
        manifest_path, manifest, artifacts, stage, arm="treatment"
    )
    gates = _gate_result(
        manifest_path,
        manifest,
        artifacts,
        stage,
        integrity,
        control,
        treatment,
    )
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": manifest["evaluation_id"],
        "stage": stage,
        "manifest_sha256": _raw_sha256(manifest_path),
        "source_corpus_commit": manifest["source_corpus_commit"],
        "inputs": {
            name: manifest["artifacts"][name]["sha256"]
            for name in manifest["artifacts"]
        },
        "registered_runs": manifest["registered_runs"][stage],
        "source_integrity": integrity,
        "control": control,
        "treatment": treatment,
        "gate": gates,
        "gate_passed": passed,
        "claim_boundary": manifest["claim_boundary"],
    }
    if stage == "development":
        result["holdout_status"] = (
            "opened_by_development_gate"
            if passed
            else "not_opened_development_gate_failed"
        )
    return result


def _run_arm(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    stage: str,
    *,
    arm: str,
) -> dict[str, Any]:
    fixture = artifacts["fixture"]["splits"][stage]
    cases = artifacts["gold"]["splits"][stage]["cases"]
    events = artifacts["schedule"]["splits"][stage]["events"]
    checkpoints = list(artifacts["schedule"]["checkpoints"])
    config_values = dict(manifest["shared_config"])
    limit_by_stage = config_values.pop("limit_by_stage")
    limit = int(limit_by_stage[stage])
    config = EngineConfig(**config_values)

    with NeuronGraphRAG(config=config) as engine:
        _load_split(engine, manifest_path, artifacts["fixture"], stage)
        initial_edges = _edge_snapshot(engine)
        feedback_records: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        feedback_counts = {str(event["cluster"]): 0 for event in events}
        current_count = 0

        for checkpoint in checkpoints:
            while current_count < checkpoint:
                current_count += 1
                for event_index, event in enumerate(events):
                    feedback_records.append(
                        _apply_feedback_event(
                            engine,
                            event,
                            stage,
                            arm,
                            current_count,
                            event_index,
                            limit,
                        )
                    )
                    feedback_counts[str(event["cluster"])] += 1
            checkpoint_records.append(
                _score_checkpoint(engine, cases, checkpoint, limit)
            )

        final_edges = _edge_snapshot(engine)

    return {
        "arm": arm,
        "checkpoints": checkpoint_records,
        "metrics": {
            str(record["feedback_count"]): record["role_mrr"]
            for record in checkpoint_records
        },
        "feedback": {
            "per_cluster_count": feedback_counts,
            "total_event_count": len(feedback_records),
            "records": feedback_records,
        },
        "initial_edges": _edge_rows(initial_edges),
        "final_edges": _edge_rows(final_edges),
        "edge_changes": _edge_changes(initial_edges, final_edges),
    }


def _apply_feedback_event(
    engine: NeuronGraphRAG,
    event: dict[str, Any],
    stage: str,
    arm: str,
    feedback_count: int,
    event_index: int,
    limit: int,
) -> dict[str, Any]:
    now = float(event["start_now"]) + feedback_count
    trace = engine.search_channels(str(event["query"]), limit=limit, now=now)
    used_node_id = str(event["used_node_id"])
    used_hit = next(
        (hit for hit in trace.relation.hits if hit.node.node_id == used_node_id),
        None,
    )
    if used_hit is None:
        raise ValueError("Frozen feedback target was not returned by the relation trace")
    raw_paths, projected_paths = _paths_from_hit(used_hit)
    expected_path = _project_path(event["credited_path"])
    if expected_path not in projected_paths:
        raise ValueError("Frozen credited path was not returned by the relation trace")

    before = _edge_snapshot(engine)
    if arm == "treatment":
        receipt = engine.record_success(
            trace.relation.trace_id, [used_node_id], now=now + 0.25
        )
        credited = [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "old_weight": edge.old_weight,
                "new_weight": edge.new_weight,
            }
            for edge in receipt.reinforced_edges
        ]
        feedback_id = receipt.feedback_id
        channel = receipt.channel
    else:
        feedback_id = (
            f"trajectory-control-{stage}-{event_index}-{feedback_count}"
        )
        engine.store.apply_success_feedback(
            feedback_id,
            trace.relation.trace_id,
            now + 0.25,
            [used_node_id],
            (),
        )
        credited = []
        channel = engine.store.retrieval_channel(trace.relation.trace_id)
    after = _edge_snapshot(engine)
    return {
        "cluster": event["cluster"],
        "feedback_count": feedback_count,
        "feedback_id": feedback_id,
        "trace_id": trace.relation.trace_id,
        "channel": channel,
        "used_node_id": used_node_id,
        "used_node_rank": used_hit.rank,
        "raw_paths": raw_paths,
        "projected_paths": projected_paths,
        "expected_path_matched": expected_path in projected_paths,
        "credited_edges": credited,
        "edge_changes": _edge_changes(before, after),
    }


def _score_checkpoint(
    engine: NeuronGraphRAG,
    cases: list[dict[str, Any]],
    feedback_count: int,
    limit: int,
) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    for offset, case in enumerate(cases):
        trace = engine.search_channels(
            str(case["query"]),
            limit=limit,
            now=float(case["now"]) + feedback_count * 100.0 + offset,
        )
        expected_node_id = str(case["expected_node_id"])
        hit = next(
            (
                candidate
                for candidate in trace.relation.hits
                if candidate.node.node_id == expected_node_id
            ),
            None,
        )
        raw_paths, projected_paths = ([], []) if hit is None else _paths_from_hit(hit)
        expected_path = _project_path(case["expected_path"])
        observed.append(
            {
                "id": case["id"],
                "cluster": case["cluster"],
                "role": case["role"],
                "expected_node_id": expected_node_id,
                "relation_trace_id": trace.relation.trace_id,
                "channel": trace.relation.channel,
                "relation_found": hit is not None,
                "relation_rank": limit + 1 if hit is None else hit.rank,
                "raw_paths": raw_paths,
                "projected_paths": projected_paths,
                "expected_path_matched": expected_path in projected_paths,
            }
        )
    return {
        "feedback_count": feedback_count,
        "cases": observed,
        "role_mrr": {
            role: _role_mrr(observed, role)
            for role in ROLES
        },
    }


def _gate_result(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    stage: str,
    integrity: dict[str, Any],
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, bool]:
    treatment_metrics = treatment["metrics"]
    control_metrics = control["metrics"]
    headroom = [float(treatment_metrics[str(point)]["headroom"]) for point in (0, 1, 3, 10)]
    control_case = [float(treatment_metrics[str(point)]["control"]) for point in (0, 1, 3, 10)]
    ceiling = [float(treatment_metrics[str(point)]["ceiling"]) for point in (0, 1, 3, 10)]
    control_arm = [
        float(control_metrics[str(point)][role])
        for role in ROLES
        for point in (0, 1, 3, 10)
    ]
    control_role_sequences = [
        [float(control_metrics[str(point)][role]) for point in (0, 1, 3, 10)]
        for role in ROLES
    ]
    expected_event_count = 10 * len(artifacts["schedule"]["splits"][stage]["events"])
    expected_changed_edges = {
        _path_key(event["credited_path"][0])
        for event in artifacts["schedule"]["splits"][stage]["events"]
    }
    treatment_changed_edges = {
        _path_key(change) for change in treatment["edge_changes"]
    }
    all_score_cases = [
        case
        for arm in (control, treatment)
        for checkpoint in arm["checkpoints"]
        for case in checkpoint["cases"]
    ]
    all_feedback = control["feedback"]["records"] + treatment["feedback"]["records"]
    treatment_feedback = treatment["feedback"]["records"]
    control_feedback = control["feedback"]["records"]
    artifact_hashes_match = all(
        _raw_sha256(manifest_path.parent / record["path"]) == record["sha256"]
        for record in manifest["artifacts"].values()
    )
    registered = manifest["registered_runs"][stage]
    expected_counts = {
        str(event["cluster"]): int(event["feedback_count"])
        for event in artifacts["schedule"]["splits"][stage]["events"]
    }
    return {
        "source_integrity": bool(integrity["passed"]),
        "split_cluster_identity": bool(artifacts["audit"]["passed"]),
        "manifest_artifact_hashes": artifact_hashes_match,
        "registered_run_count": registered.get("control") == 1 and registered.get("treatment") == 1,
        "relation_channel_only": all(case["channel"] == "relation" for case in all_score_cases) and all(record["channel"] == "relation" for record in all_feedback),
        "expected_paths": all(case["expected_path_matched"] for case in all_score_cases) and all(record["expected_path_matched"] for record in all_feedback),
        "feedback_checkpoint_counts": control["feedback"]["per_cluster_count"] == expected_counts and treatment["feedback"]["per_cluster_count"] == expected_counts and control["feedback"]["total_event_count"] == expected_event_count and treatment["feedback"]["total_event_count"] == expected_event_count,
        "headroom_strict_improvement_0_to_10": headroom[-1] > headroom[0],
        "headroom_intermediate_non_regression": _nondecreasing(headroom),
        "control_case_non_regression": _nondecreasing(control_case),
        "ceiling_case_non_regression": _nondecreasing(ceiling),
        "control_arm_non_regression": bool(control_arm) and all(_nondecreasing(sequence) for sequence in control_role_sequences),
        "control_feedback_recorded_without_mutation": len(control_feedback) == expected_event_count and not control["edge_changes"] and all(not record["edge_changes"] and not record["credited_edges"] for record in control_feedback),
        "treatment_record_success_relation_trace": len(treatment_feedback) == expected_event_count and all(record["channel"] == "relation" and record["credited_edges"] for record in treatment_feedback),
        "treatment_credited_edges_only": treatment_changed_edges == expected_changed_edges and all({_path_key(edge) for edge in record["credited_edges"]} == {_path_key(next(event["credited_path"][0] for event in artifacts["schedule"]["splits"][stage]["events"] if event["cluster"] == record["cluster"]))} and {_path_key(change) for change in record["edge_changes"]} <= expected_changed_edges for record in treatment_feedback),
        "exclusive_output": not _result_path(manifest_path, manifest, stage).exists(),
    }


def _read_frozen_artifacts(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    root = manifest_path.resolve().parents[2]
    records: dict[str, tuple[str, str]] = {}
    for name in ("fixture", "gold", "schedule", "gate", "audit"):
        record = manifest.get("artifacts", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Missing frozen artifact: {name}")
        artifact_path = manifest_path.parent / str(record.get("path", ""))
        relative = artifact_path.resolve().relative_to(root).as_posix()
        records[name] = (relative, str(record.get("sha256", "")))
    registered = verify_manifest_source_hashes(
        root,
        manifest_path,
        {relative: expected for relative, expected in records.values()},
    )
    loaded: dict[str, dict[str, Any]] = {}
    for name, (relative, _) in records.items():
        loaded[name] = _read_json_bytes(
            registered.artifact_bytes[relative],
            f"{registered.source_commit}:{relative}",
        )
    return loaded


def _validate_fixture_and_audit(
    fixture: dict[str, Any], audit: dict[str, Any]
) -> None:
    if fixture.get("schema_version") != SCHEMA_VERSION or audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Fixture or audit schema version is unsupported")
    if fixture.get("source_commit") != SOURCE_COMMIT or audit.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Fixture or audit source commit does not match")
    if audit.get("result_free") is not True or audit.get("passed") is not True:
        raise ValueError("Result-free audit did not pass")
    if any(record.get("absent") is not True for record in audit.get("exclusive_outputs", [])):
        raise ValueError("Exclusive outputs were not absent at freeze")

    identities: dict[str, dict[str, set[Any]]] = {}
    for stage in STAGES:
        split = fixture.get("splits", {}).get(stage)
        recorded = audit.get("splits", {}).get(stage)
        if not isinstance(split, dict) or not isinstance(recorded, dict):
            raise ValueError(f"Missing frozen split identity: {stage}")
        nodes = split.get("nodes", [])
        edges = split.get("edges", [])
        identity = {
            "clusters": set(map(str, split.get("clusters", []))),
            "node_ids": {str(node["node_id"]) for node in nodes},
            "document_paths": {str(node["document_path"]) for node in nodes},
            "credited_edges": {_path_key(edge) for edge in edges},
        }
        expected = {
            "clusters": set(map(str, recorded.get("clusters", []))),
            "node_ids": set(map(str, recorded.get("node_ids", []))),
            "document_paths": set(map(str, recorded.get("document_paths", []))),
            "credited_edges": {tuple(map(str, edge)) for edge in recorded.get("credited_edges", [])},
        }
        if identity != expected:
            raise ValueError(f"Audit identity does not match fixture: {stage}")
        if len(identity["node_ids"]) != len(nodes) or len(identity["credited_edges"]) != len(edges):
            raise ValueError(f"Duplicate fixture identity: {stage}")
        if {str(node["cluster"]) for node in nodes} != identity["clusters"]:
            raise ValueError(f"Node cluster identity mismatch: {stage}")
        if {str(edge["cluster"]) for edge in edges} != identity["clusters"]:
            raise ValueError(f"Edge cluster identity mismatch: {stage}")
        identities[stage] = identity
    for field in ("clusters", "node_ids", "document_paths", "credited_edges"):
        if identities["development"][field] & identities["holdout"][field]:
            raise ValueError(f"Development and holdout overlap: {field}")


def _validate_gold_and_schedule(
    fixture: dict[str, Any], gold: dict[str, Any], schedule: dict[str, Any]
) -> None:
    if gold.get("schema_version") != SCHEMA_VERSION or schedule.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Gold or schedule schema version is unsupported")
    if schedule.get("checkpoints") != [0, 1, 3, 10] or schedule.get("channel") != "relation":
        raise ValueError("Schedule checkpoints or channel are not frozen")
    for stage in STAGES:
        split = fixture["splits"][stage]
        nodes = {str(node["node_id"]) for node in split["nodes"]}
        edges = {_path_key(edge) for edge in split["edges"]}
        clusters = set(map(str, split["clusters"]))
        cases = gold.get("splits", {}).get(stage, {}).get("cases", [])
        events = schedule.get("splits", {}).get(stage, {}).get("events", [])
        if {str(case.get("role")) for case in cases} != set(ROLES):
            raise ValueError(f"Gold roles are incomplete: {stage}")
        if {str(event.get("cluster")) for event in events} != clusters:
            raise ValueError(f"Schedule does not cover every cluster: {stage}")
        score_queries = {str(case.get("query")) for case in cases}
        for case in cases:
            if str(case.get("cluster")) not in clusters or str(case.get("expected_node_id")) not in nodes:
                raise ValueError(f"Gold case is outside fixture: {stage}")
            expected_path = _project_path(case.get("expected_path"))
            if any(step not in edges for step in expected_path):
                raise ValueError(f"Gold path is outside fixture: {stage}")
        for event in events:
            if str(event.get("query")) in score_queries or int(event.get("feedback_count", -1)) != 10:
                raise ValueError(f"Feedback schedule is invalid: {stage}")
            if str(event.get("used_node_id")) not in nodes:
                raise ValueError(f"Feedback node is outside fixture: {stage}")
            credited_path = _project_path(event.get("credited_path"))
            if any(step not in edges for step in credited_path):
                raise ValueError(f"Feedback path is outside fixture: {stage}")


def _validate_gate(gate: dict[str, Any]) -> None:
    if gate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Gate schema version is unsupported")
    for stage in STAGES:
        stage_gate = gate.get("splits", {}).get(stage, {})
        if set(stage_gate) != REQUIRED_GATES or not all(stage_gate.values()):
            raise ValueError(f"Gate registration is incomplete: {stage}")


def _validate_registration(manifest: dict[str, Any]) -> None:
    result_paths = manifest.get("result_paths", {})
    if set(result_paths) != set(STAGES) or len(set(result_paths.values())) != len(STAGES):
        raise ValueError("Exclusive result paths are not frozen")
    for stage in STAGES:
        registration = manifest.get("registered_runs", {}).get(stage, {})
        if registration.get("control") != 1 or registration.get("treatment") != 1:
            raise ValueError(f"Registered run count is invalid: {stage}")
    if manifest["registered_runs"]["holdout"].get("conditional_on_development_gate") is not True:
        raise ValueError("Holdout stop rule is not frozen")


def _verify_source_integrity(
    manifest_path: Path, fixture: dict[str, Any], stage: str
) -> dict[str, Any]:
    root = (manifest_path.parent / str(fixture["repository_root"])).resolve()
    source = _registered_source_snapshot(root, fixture, stage)
    checks: list[dict[str, Any]] = []
    for node in fixture["splits"][stage]["nodes"]:
        relative = str(node["document_path"])
        verification = verify_source_bytes(
            source.artifact_bytes[relative],
            str(node["source_sha256"]),
            allow_text_newline_alternate=True,
        )
        checks.append(
            {
                "node_id": node["node_id"],
                "document_path": node["document_path"],
                "accepted": verification.accepted,
                "decision": verification.decision,
                "reason": verification.reason,
                "expected_sha256": verification.expected_sha256,
                "raw_sha256": verification.raw_sha256,
                "alternate_sha256": verification.alternate_sha256,
            }
        )
    explicit_links = _verify_explicit_links(
        fixture["splits"][stage], source.artifact_bytes
    )
    return {
        "passed": all(bool(check["accepted"]) for check in checks) and explicit_links["passed"],
        "source_commit": fixture["source_commit"],
        "checks": checks,
        "explicit_links": explicit_links,
    }


def _verify_explicit_links(
    split: dict[str, Any], source_bytes: dict[str, bytes]
) -> dict[str, Any]:
    node_by_path = {
        Path(str(node["document_path"])).name: str(node["node_id"])
        for node in split["nodes"]
    }
    expected = {_path_key(edge) for edge in split["edges"]}
    observed: set[tuple[str, str, str]] = set()
    for cluster in split["clusters"]:
        overview_name = f"{cluster}-overview.md"
        overview_node = node_by_path[overview_name]
        overview_relative = next(
            str(node["document_path"])
            for node in split["nodes"]
            if Path(str(node["document_path"])).name == overview_name
        )
        overview_text = source_bytes[overview_relative].decode("utf-8", errors="strict")
        for target_name in RELATIVE_LINK.findall(overview_text):
            observed.add((overview_node, node_by_path[target_name], "explicit_link"))
    return {
        "passed": observed == expected,
        "observed": [list(edge) for edge in sorted(observed)],
        "expected": [list(edge) for edge in sorted(expected)],
    }


def _load_split(
    engine: NeuronGraphRAG,
    manifest_path: Path,
    fixture: dict[str, Any],
    stage: str,
) -> None:
    root = (manifest_path.parent / str(fixture["repository_root"])).resolve()
    split = fixture["splits"][stage]
    source = _registered_source_snapshot(root, fixture, stage)
    for node in split["nodes"]:
        relative = str(node["document_path"])
        engine.add_document(
            str(node["node_id"]),
            source.artifact_bytes[relative].decode("utf-8", errors="strict"),
            metadata={
                "cluster": node["cluster"],
                "source_url": node["source_url"],
                "document_path": node["document_path"],
                "source_commit": fixture["source_commit"],
            },
        )
    for edge in split["edges"]:
        engine.add_edge(
            str(edge["source_id"]),
            str(edge["target_id"]),
            str(edge["edge_type"]),
            weight=float(edge["weight"]),
            factuality=float(edge["factuality"]),
        )


def _registered_source_snapshot(
    root: Path, fixture: dict[str, Any], stage: str
) -> HistoricalSourceSnapshot:
    return verify_historical_source_hashes(
        root,
        str(fixture["source_commit"]),
        {
            str(node["document_path"]): str(node["source_sha256"])
            for node in fixture["splits"][stage]["nodes"]
        },
        allow_text_newline_alternate=True,
    )
def _paths_from_hit(hit: Any) -> tuple[list[dict[str, Any]], list[tuple[tuple[str, str, str], ...]]]:
    raw = list(hit.explain()["paths"])
    projected = [_project_path(path["steps"]) for path in raw]
    return raw, projected


def _project_path(steps: Any) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(steps, list) or not steps:
        raise ValueError("Expected a non-empty relation path")
    return tuple(_path_key(step) for step in steps)


def _path_key(step: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(step[field]) for field in PATH_FIELDS)  # type: ignore[return-value]


def _role_mrr(cases: list[dict[str, Any]], role: str) -> float:
    ranks = [int(case["relation_rank"]) for case in cases if case["role"] == role]
    if not ranks:
        raise ValueError(f"No cases registered for role: {role}")
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _nondecreasing(values: list[float]) -> bool:
    return all(later + 1e-12 >= earlier for earlier, later in zip(values, values[1:]))


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


def _edge_rows(
    snapshot: dict[tuple[str, str, str], tuple[float, int]]
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": key[0],
            "target_id": key[1],
            "edge_type": key[2],
            "weight": value[0],
            "reinforced_count": value[1],
        }
        for key, value in sorted(snapshot.items())
    ]


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


def _require_output_absence(
    manifest_path: Path, manifest: dict[str, Any], stage: str
) -> None:
    output = _result_path(manifest_path, manifest, stage)
    if output.exists():
        raise ValueError(f"Refusing to overwrite an observed result: {output}")


def _result_path(
    manifest_path: Path, manifest: dict[str, Any], stage: str
) -> Path:
    return manifest_path.parent / str(manifest["result_paths"][stage])


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path.read_bytes(), str(path))


def _read_json_bytes(raw: bytes, source: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _raw_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

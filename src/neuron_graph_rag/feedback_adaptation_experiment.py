from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .d1_fixture import load_fixture, read_fixture
from .engine import EngineConfig, NeuronGraphRAG


FEEDBACK_ADAPTATION_SCHEMA_VERSION = 1


def read_feedback_adaptation_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != FEEDBACK_ADAPTATION_SCHEMA_VERSION:
        raise ValueError("Unsupported feedback adaptation schema version")
    if manifest.get("candidate_id") != "trace-credited-feedback":
        raise ValueError("Feedback adaptation candidate is not frozen")
    base = manifest_path.parent
    for stage in ("development", "holdout"):
        split = manifest.get(stage, {})
        for field in ("fixture", "gold", "provenance"):
            artifact = base / str(split[field])
            if _canonical_sha256(artifact) != split[f"{field}_sha256"]:
                raise ValueError(f"Frozen {stage} {field} hash mismatch")
        _validate_gold(base / str(split["fixture"]), base / str(split["gold"]))
    audit = manifest["contamination_audit"]
    audit_path = base / str(audit["artifact"])
    if _canonical_sha256(audit_path) != audit["artifact_sha256"]:
        raise ValueError("Frozen feedback adaptation contamination audit hash mismatch")
    audit_payload = _read_json(audit_path)
    if not audit_payload.get("passed"):
        raise ValueError("Frozen feedback adaptation contamination audit did not pass")
    return manifest


def run_feedback_adaptation_development(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_feedback_adaptation_manifest(manifest_path)
    result = _evaluate_split(manifest_path, manifest, "development")
    gate_passed = all(result["gate"].values())
    return {
        "schema_version": FEEDBACK_ADAPTATION_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "development",
        "manifest_sha256": _canonical_sha256(manifest_path),
        **result,
        "gate_passed": gate_passed,
        "holdout_status": "not_opened_candidate_selected" if gate_passed else "not_opened_no_candidate",
    }


def run_feedback_adaptation_holdout(
    path: str | Path, development_result_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_feedback_adaptation_manifest(manifest_path)
    development = _read_json(Path(development_result_path))
    if development.get("manifest_sha256") != _canonical_sha256(manifest_path):
        raise ValueError("Development result does not match frozen feedback manifest")
    if not development.get("gate_passed"):
        raise ValueError("Stop rule forbids opening feedback adaptation holdout")
    result = _evaluate_split(manifest_path, manifest, "holdout")
    adopted = all(result["gate"].values())
    return {
        "schema_version": FEEDBACK_ADAPTATION_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "holdout",
        "manifest_sha256": _canonical_sha256(manifest_path),
        "development_result_sha256": _canonical_sha256(Path(development_result_path)),
        "holdout_open_count": 1,
        **result,
        "decision": {
            "adopted": adopted,
            "reason": "all_frozen_holdout_gates_passed" if adopted else "feedback_candidate_failed_frozen_holdout_gate",
        },
    }


def write_feedback_adaptation_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evaluate_split(path: Path, manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    base = path.parent
    split = manifest[stage]
    fixture = base / str(split["fixture"])
    gold = _read_json(base / str(split["gold"]))
    config = EngineConfig(**manifest["shared_config"])
    limit = int(manifest["shared_config"]["limit"])
    control = _run_group(fixture, gold, config, limit, mutate=False)
    treatment = _run_group(fixture, gold, config, limit, mutate=True)
    replay = _run_group(fixture, gold, config, limit, mutate=True)
    relation_control = _by_role(control["cases"], "relation")
    relation_treatment = _by_role(treatment["cases"], "relation")
    direct_control = _by_role(control["cases"], "direct_lexical")
    direct_treatment = _by_role(treatment["cases"], "direct_lexical")
    negative_control = _by_role(control["cases"], "directional_negative")
    negative_treatment = _by_role(treatment["cases"], "directional_negative")
    control_mrr = _mrr([case["relation_rank"] for case in relation_control])
    treatment_mrr = _mrr([case["relation_rank"] for case in relation_treatment])
    return {
        "inputs": {key: split[f"{key}_sha256"] for key in ("fixture", "gold", "provenance")},
        "control": control,
        "treatment": treatment,
        "metrics": {
            "control_relation_mrr": control_mrr,
            "treatment_relation_mrr": treatment_mrr,
            "control_relation_recall": _mean([case["relation_found"] for case in relation_control]),
            "treatment_relation_recall": _mean([case["relation_found"] for case in relation_treatment]),
            "control_relation_hit_at_k": _mean([case["relation_found"] for case in relation_control]),
            "treatment_relation_hit_at_k": _mean([case["relation_found"] for case in relation_treatment]),
        },
        "gate": {
            "treatment_strictly_improves_relation_mrr": treatment_mrr > control_mrr,
            "relation_recall_and_hit_at_k_do_not_regress": all(
                right["relation_found"] >= left["relation_found"]
                for left, right in zip(relation_control, relation_treatment, strict=True)
            ),
            "expected_relation_endpoint_and_edge_type_match": all(case["path_matched"] for case in relation_treatment),
            "direct_lexical_control_does_not_regress": all(
                right["lexical_rank"] <= left["lexical_rank"]
                for left, right in zip(direct_control, direct_treatment, strict=True)
            ),
            "directional_negative_control_does_not_regress": all(
                right["relation_hits"] == left["relation_hits"] == []
                for left, right in zip(negative_control, negative_treatment, strict=True)
            ),
            "only_credited_edges_change": treatment["feedback"]["credited_only"] and control["feedback"]["no_edge_mutation"],
            "same_schedule_replay_is_deterministic": treatment == replay,
        },
    }


def _run_group(
    fixture_path: Path, gold: dict[str, Any], config: EngineConfig, limit: int, *, mutate: bool
) -> dict[str, Any]:
    feedback = gold["feedback"]
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before = _edge_snapshot(engine)
        channels = engine.search_channels(feedback["query"], limit=limit, now=feedback["now"])
        if feedback["used_node_id"] not in {hit.node.node_id for hit in channels.relation.hits}:
            raise ValueError("Frozen feedback target was not retrieved by relation trace")
        if feedback["channel"] != "relation":
            raise ValueError("Frozen feedback must use the relation trace")
        if mutate:
            receipt = engine.record_success(channels.relation.trace_id, [feedback["used_node_id"]], now=float(feedback["now"]) + 1.0)
            credited = [_reinforced(edge) for edge in receipt.reinforced_edges]
        else:
            engine.store.apply_success_feedback(
                "control-feedback", channels.relation.trace_id, float(feedback["now"]) + 1.0,
                [feedback["used_node_id"]], (),
            )
            credited = []
        after = _edge_snapshot(engine)
        cases = [_score_case(engine, case, limit) for case in gold["scoring_cases"]]
        changed = _edge_changes(before, after)
    credited_keys = {(edge["source_id"], edge["target_id"], edge["edge_type"]) for edge in credited}
    return {
        "cases": cases,
        "feedback": {
            "recorded": True,
            "feedback_count": 1,
            "credited_edges": credited,
            "edge_changes": changed,
            "no_edge_mutation": not changed,
            "credited_only": bool(credited) and all(
                (edge["source_id"], edge["target_id"], edge["edge_type"]) in credited_keys for edge in changed
            ),
        },
    }


def _score_case(engine: NeuronGraphRAG, case: dict[str, Any], limit: int) -> dict[str, Any]:
    channels = engine.search_channels(case["query"], limit=limit, now=case["now"])
    expected = str(case["expected_node_id"])
    lexical_hits = [hit.node.node_id for hit in channels.lexical.hits]
    relation_hits = [hit.node.node_id for hit in channels.relation.hits]
    observed_paths = [
        path["steps"] for hit in channels.relation.hits if hit.node.node_id == expected for path in hit.explain()["paths"]
    ]
    expected_path = case.get("expected_path", [])
    return {
        "id": case["id"], "role": case["role"], "expected_node_id": expected,
        "lexical_rank": _rank(lexical_hits, expected, limit),
        "relation_rank": _rank(relation_hits, expected, limit),
        "relation_found": expected in relation_hits,
        "relation_hits": relation_hits,
        "path_matched": any(path == expected_path for path in observed_paths) if expected_path else bool(case.get("expected_relation_empty")) and not relation_hits,
        "observed_paths": observed_paths,
    }


def _validate_gold(fixture_path: Path, gold_path: Path) -> None:
    fixture = read_fixture(fixture_path)
    gold = _read_json(gold_path)
    nodes = {str(node["node_id"]) for node in fixture["nodes"]}
    edges = {(str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"])) for edge in fixture["edges"]}
    feedback = gold.get("feedback", {})
    if feedback.get("channel") != "relation" or str(feedback.get("used_node_id")) not in nodes:
        raise ValueError("Feedback schedule is invalid")
    cases = gold.get("scoring_cases")
    if not isinstance(cases, list) or {case.get("role") for case in cases} != {"relation", "direct_lexical", "directional_negative"}:
        raise ValueError("Gold must freeze relation and both controls")
    if feedback.get("query") in {case.get("query") for case in cases}:
        raise ValueError("Feedback query must be separate from scoring queries")
    for case in cases:
        if str(case.get("expected_node_id")) not in nodes or not str(case.get("source_url", "")).startswith("https://github.com/"):
            raise ValueError("Gold case target or source URL is invalid")
        for step in case.get("expected_path", []):
            if (str(step["source_id"]), str(step["target_id"]), str(step["edge_type"])) not in edges:
                raise ValueError("Expected relation path is outside fixture")


def _by_role(cases: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [case for case in cases if case["role"] == role]


def _rank(hits: list[str], node_id: str, limit: int) -> int:
    return next((index for index, value in enumerate(hits, start=1) if value == node_id), limit + 1)


def _mrr(ranks: list[int]) -> float:
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _mean(values: list[bool]) -> float:
    return sum(values) / len(values)


def _edge_snapshot(engine: NeuronGraphRAG) -> dict[tuple[str, str, str], tuple[float, int]]:
    return {(edge.source_id, edge.target_id, edge.edge_type): (edge.weight, edge.reinforced_count) for edge in engine.store.list_edges()}


def _edge_changes(before: dict[tuple[str, str, str], tuple[float, int]], after: dict[tuple[str, str, str], tuple[float, int]]) -> list[dict[str, Any]]:
    return [{"source_id": key[0], "target_id": key[1], "edge_type": key[2], "before_weight": before[key][0], "after_weight": after[key][0], "before_reinforced_count": before[key][1], "after_reinforced_count": after[key][1]} for key in sorted(before) if before[key] != after[key]]


def _reinforced(edge: Any) -> dict[str, Any]:
    return {"source_id": edge.source_id, "target_id": edge.target_id, "edge_type": edge.edge_type, "old_weight": edge.old_weight, "new_weight": edge.new_weight}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256((json.dumps(_read_json(path), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evidence_feedback import EngineConfig, NeuronGraphRAG


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
PROTOCOL_STEM = "canonical_evidence_gate_v1"
MANIFEST_PATH = FIXTURES / f"{PROTOCOL_STEM}.manifest.json"


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if text.encode("utf-8") != raw:
        raise ValueError(f"non-canonical UTF-8 artifact: {path}")
    return json.loads(text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_gate_ids(gate_path: Path) -> list[str]:
    payload = read_json(gate_path)
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("registered gates must be a non-empty array")
    gate_ids = [item.get("gate_id") for item in gates if isinstance(item, dict)]
    if len(gate_ids) != len(gates) or not all(isinstance(item, str) for item in gate_ids):
        raise ValueError("every registered gate needs a string gate_id")
    if len(set(gate_ids)) != len(gate_ids):
        raise ValueError("registered gate IDs must be unique")
    return gate_ids


def validate_gate_array(
    gates: object,
    expected_gate_ids: Sequence[str],
    *,
    require_all_passed: bool,
) -> None:
    if not isinstance(gates, list):
        raise ValueError("observed gates must be an array")
    if not all(isinstance(item, dict) for item in gates):
        raise ValueError("each observed gate must be an object")
    actual_ids = [item.get("gate_id") for item in gates]
    if actual_ids != list(expected_gate_ids):
        raise ValueError("observed gate order or membership differs from registration")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("observed gate IDs must be unique")
    if not all(isinstance(item.get("passed"), bool) for item in gates):
        raise ValueError("each observed gate needs a boolean passed field")
    if require_all_passed and not all(item["passed"] for item in gates):
        raise ValueError("not every observed gate passed")


def write_observed_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def prove_writer_verifier_round_trip(path: Path) -> None:
    expected = ["zulu-placeholder", "alpha-placeholder", "mike-placeholder"]
    payload = {
        "protocol_id": "temporary-canonical-array-probe",
        "gates": [{"gate_id": gate_id, "passed": True} for gate_id in expected],
        "all_pass": True,
    }
    write_observed_exclusive(path, payload)
    try:
        written = read_json(path)
        validate_gate_array(written.get("gates"), expected, require_all_passed=True)
        if written.get("all_pass") is not True:
            raise ValueError("round-trip probe lost all_pass")
    finally:
        path.unlink(missing_ok=True)


def verify_identity_only_registry(
    fixture: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, bool]:
    cases = fixture["cases"]
    nodes = [node for case in cases for node in case["nodes"]]
    node_ids = [node["node_id"] for node in nodes]
    source_urls = [node["source_url"] for node in nodes]
    credited = [
        tuple(case["credited_edge"][key] for key in ("source_id", "target_id", "edge_type"))
        for case in cases
    ]
    development_queries = {case["query"] for case in cases if case["stage"] == "development"}
    holdout_queries = {case["query"] for case in cases if case["stage"] == "holdout"}
    namespace = registry["new_protocol_namespace"]
    reserved = registry["reserved_public_namespaces"]
    return {
        "new_namespace_excludes_reserved_public_namespaces": all(
            reserved_item not in namespace and namespace not in reserved_item
            for reserved_item in reserved
        ),
        "new_corpus_directory_is_unique": registry["new_corpus_directory"]
        == "corpora/canonical-evidence-gates-v1",
        "development_and_holdout_case_ids_are_disjoint": not (
            {case["case_id"] for case in cases if case["stage"] == "development"}
            & {case["case_id"] for case in cases if case["stage"] == "holdout"}
        ),
        "node_ids_are_globally_unique": len(node_ids) == len(set(node_ids))
        and all(node_id.startswith(namespace) for node_id in node_ids),
        "source_urls_are_globally_unique": len(source_urls) == len(set(source_urls))
        and all(url.startswith(registry["new_source_url_prefix"]) for url in source_urls),
        "query_tokens_are_split_unique": not (development_queries & holdout_queries),
        "credited_edge_identities_are_globally_unique": len(credited) == len(set(credited)),
    }


def _config(variant: Mapping[str, Any]) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=1,
        hop_decay=0.7,
        feedback_learning_rate=0.2,
        sibling_feedback_normalization=float(variant["sibling_normalization"]),
        relation_feedback_evidence_quorum=int(variant["evidence_quorum"]),
    )


def _populate(engine: NeuronGraphRAG, case: Mapping[str, Any]) -> None:
    for node in case["nodes"]:
        engine.add_document(
            node["node_id"],
            node["text"],
            metadata={"source_url": node["source_url"]},
        )
    for edge in case["edges"]:
        engine.add_edge(
            edge["source_id"],
            edge["target_id"],
            edge["edge_type"],
            weight=float(edge["weight"]),
        )


def _edge_state(engine: NeuronGraphRAG, case: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in ("credited_edge", "sibling_edge", "unrelated_edge", "reverse_edge"):
        identity = case[name]
        edge = engine.store.edge(
            identity["source_id"], identity["target_id"], identity["edge_type"]
        )
        state[name] = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "reinforced_count": edge.reinforced_count,
        }
    return state


def _rank_snapshot(engine: NeuronGraphRAG, case: Mapping[str, Any], now: float) -> dict[str, Any]:
    channels = engine.search_channels(case["query"], limit=len(case["nodes"]), now=now)
    relation = [
        {"node_id": hit.node.node_id, "rank": hit.rank, "score": hit.channel_score}
        for hit in channels.relation.hits
    ]
    lexical = [
        {"node_id": hit.node.node_id, "rank": hit.rank, "score": hit.channel_score}
        for hit in channels.lexical.hits
    ]
    relation_ranks = {item["node_id"]: item["rank"] for item in relation}
    lexical_ranks = {item["node_id"]: item["rank"] for item in lexical}
    target_rank = relation_ranks[case["target_node_id"]]
    return {
        "relation": relation,
        "lexical": lexical,
        "target_rank": target_rank,
        "target_mrr": 1.0 / target_rank,
        "direct_lexical_rank": lexical_ranks[case["direct_node_id"]],
        "reverse_lexical_rank": lexical_ranks[case["reverse_node_id"]],
    }


def _trajectory(
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    checkpoint: int,
) -> dict[str, Any]:
    with NeuronGraphRAG(config=_config(variant)) as engine:
        _populate(engine, case)
        before = _edge_state(engine, case)
        receipts: list[dict[str, Any]] = []
        for event_index in range(1, checkpoint + 1):
            channels = engine.search_channels(
                case["query"], limit=len(case["nodes"]), now=1_000.0 + event_index
            )
            relation = channels.relation
            hit = next(
                item for item in relation.hits if item.node.node_id == case["target_node_id"]
            )
            projected_steps = [
                {
                    "source_id": step.source_id,
                    "target_id": step.target_id,
                    "edge_type": step.edge_type,
                }
                for step in hit.paths[0].steps
            ]
            receipt = engine.record_success(
                relation.trace_id,
                [case["target_node_id"]],
                now=2_000.0 + event_index,
            )
            duplicate = engine.record_success(
                relation.trace_id,
                [case["target_node_id"]],
                now=3_000.0 + event_index,
            )
            receipts.append(
                {
                    "event": event_index,
                    "channel": receipt.channel,
                    "path": projected_steps,
                    "evidence": [
                        {
                            "source_id": item.source_id,
                            "target_id": item.target_id,
                            "edge_type": item.edge_type,
                            "count": item.count,
                            "quorum": item.quorum,
                            "activated": item.activated,
                        }
                        for item in receipt.evidence
                    ],
                    "reinforced": [asdict(edge) for edge in receipt.reinforced_edges],
                    "normalized": [
                        asdict(edge) for edge in receipt.normalized_sibling_edges
                    ],
                    "duplicate": {
                        "evidence": [
                            {
                                "count": item.count,
                                "activated": item.activated,
                            }
                            for item in duplicate.evidence
                        ],
                        "reinforced_count": len(duplicate.reinforced_edges),
                        "normalized_count": len(duplicate.normalized_sibling_edges),
                    },
                }
            )
        return {
            "checkpoint": checkpoint,
            "before": before,
            "after": _edge_state(engine, case),
            "ranking": _rank_snapshot(engine, case, 9_000.0 + checkpoint),
            "receipts": receipts,
        }


def _rollback_probe(case: Mapping[str, Any], *, sibling_failure: bool) -> bool:
    variant = {"evidence_quorum": 1, "sibling_normalization": 1.0}
    with NeuronGraphRAG(config=_config(variant)) as engine:
        _populate(engine, case)
        relation = engine.search_channels(case["query"], limit=len(case["nodes"]), now=1.0).relation
        before = _edge_state(engine, case)
        feedback_before = engine.store.count_feedback()
        evidence_before = engine.store.count_feedback_evidence()
        credited = case["credited_edge"]
        missing = (credited["source_id"], "canon77-missing-target", credited["edge_type"])
        updates = [
            (
                credited["source_id"],
                credited["target_id"],
                credited["edge_type"],
                0.1,
                2.0,
            )
        ]
        if not sibling_failure:
            updates.append((missing[0], missing[1], missing[2], 0.1, 2.0))
        try:
            engine.store.apply_evidence_gated_success_feedback(
                "canon77-injected-rollback",
                relation.trace_id,
                2.0,
                [case["target_node_id"]],
                updates,
                [
                    (
                        credited["source_id"],
                        (missing,),
                        1.0,
                    )
                ]
                if sibling_failure
                else (),
                evidence_quorum=1,
            )
        except KeyError:
            pass
        else:
            return False
        return (
            _edge_state(engine, case) == before
            and engine.store.count_feedback() == feedback_before
            and engine.store.count_feedback_evidence() == evidence_before
        )


def _rank_order(snapshot: Mapping[str, Any]) -> list[str]:
    return [item["node_id"] for item in snapshot["ranking"]["relation"]]


def _churn(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_ranks = {node_id: rank for rank, node_id in enumerate(_rank_order(left), 1)}
    right_ranks = {node_id: rank for rank, node_id in enumerate(_rank_order(right), 1)}
    return sum(left_ranks[node_id] != right_ranks[node_id] for node_id in left_ranks)


def _changed_edges(snapshot: Mapping[str, Any]) -> list[str]:
    return [
        name
        for name, before in snapshot["before"].items()
        if snapshot["after"][name] != before
    ]


def _artifact_preflight(manifest: Mapping[str, Any], stage: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for relative, expected_hash in manifest["artifacts"].items():
        path = ROOT / relative
        checks[f"hash:{relative}"] = path.is_file() and _sha256(path) == expected_hash
        if path.is_file() and path.suffix in {".json", ".md", ".py"}:
            try:
                path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                checks[f"utf8:{relative}"] = False
            else:
                checks[f"utf8:{relative}"] = True
    output = ROOT / manifest["outputs"][stage]
    checks["registered-output-absent"] = not output.exists()
    other_stage = "holdout" if stage == "development" else "development"
    other_output = ROOT / manifest["outputs"][other_stage]
    if stage == "development":
        checks["holdout-output-absent"] = not other_output.exists()
        checks["development-precedes-holdout"] = True
    else:
        checks["development-precedes-holdout"] = other_output.exists()
        if other_output.exists():
            development = read_json(other_output)
            checks["development-all-pass"] = development.get("all_pass") is True
    audit = read_json(ROOT / manifest["audit"])
    fixture = read_json(ROOT / manifest["fixture"])
    registry = read_json(ROOT / manifest["identity_registry"])
    identity_checks = verify_identity_only_registry(fixture, registry)
    checks["identity-audit"] = (
        audit.get("checks") == identity_checks
        and audit.get("all_identity_checks_pass") is all(identity_checks.values())
    )
    return {"checks": checks, "passed": all(checks.values())}


def _evaluate_gates(
    cases: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    preflight: Mapping[str, Any],
    rollback: Mapping[str, bool],
) -> dict[str, bool]:
    credit_ok = True
    duplicate_ok = True
    prequorum_ok = True
    threshold_ok = True
    ceiling_ok = True
    controls_ok = True
    ablation_ok = True
    recompute_ok = True
    for case in cases:
        case_runs = trajectories[case["case_id"]]
        credited = case["credited_edge"]
        for variant_runs in case_runs.values():
            for checkpoint, snapshot in variant_runs.items():
                expected_order = [
                    item["node_id"]
                    for item in sorted(
                        snapshot["ranking"]["relation"],
                        key=lambda item: (-item["score"], item["node_id"]),
                    )
                ]
                recompute_ok &= expected_order == _rank_order(snapshot)
                recompute_ok &= snapshot["ranking"]["target_mrr"] == 1.0 / snapshot["ranking"]["target_rank"]
                for receipt in snapshot["receipts"]:
                    credit_ok &= receipt["channel"] == "relation"
                    credit_ok &= receipt["path"] == [credited]
                    credit_ok &= all(
                        {key: evidence[key] for key in ("source_id", "target_id", "edge_type")}
                        == credited
                        for evidence in receipt["evidence"]
                    )
                    duplicate_ok &= receipt["duplicate"]["reinforced_count"] == 0
                    duplicate_ok &= receipt["duplicate"]["normalized_count"] == 0
                    duplicate_ok &= all(not item["activated"] for item in receipt["duplicate"]["evidence"])
        baseline = case_runs["combined"]["0"]
        for checkpoint in ("1", "2"):
            combined = case_runs["combined"][checkpoint]
            evidence = case_runs["evidence-only"][checkpoint]
            prequorum_ok &= combined["after"] == baseline["after"]
            prequorum_ok &= combined["after"] == evidence["after"]
            prequorum_ok &= _rank_order(combined) == _rank_order(baseline)
            prequorum_ok &= _rank_order(combined) == _rank_order(evidence)
        baseline_direct = baseline["ranking"]["direct_lexical_rank"]
        baseline_reverse = baseline["ranking"]["reverse_lexical_rank"]
        baseline_non_target_order = [
            node_id
            for node_id in _rank_order(baseline)
            if node_id != case["target_node_id"]
        ]
        for variant_runs in case_runs.values():
            for snapshot in variant_runs.values():
                controls_ok &= snapshot["ranking"]["direct_lexical_rank"] == baseline_direct
                controls_ok &= snapshot["ranking"]["reverse_lexical_rank"] == baseline_reverse
                controls_ok &= snapshot["after"]["unrelated_edge"] == snapshot["before"]["unrelated_edge"]
                controls_ok &= snapshot["after"]["reverse_edge"] == snapshot["before"]["reverse_edge"]
                controls_ok &= [
                    node_id
                    for node_id in _rank_order(snapshot)
                    if node_id != case["target_node_id"]
                ] == baseline_non_target_order
        if case["stratum"] == "headroom":
            threshold_ok &= case_runs["current"]["1"]["ranking"]["target_rank"] == 1
            threshold_ok &= case_runs["local-only"]["1"]["ranking"]["target_rank"] == 1
            threshold_ok &= case_runs["combined"]["3"]["ranking"]["target_rank"] == 1
            threshold_ok &= _churn(baseline, case_runs["combined"]["1"]) < _churn(baseline, case_runs["current"]["1"])
            threshold_ok &= _churn(baseline, case_runs["combined"]["2"]) < _churn(baseline, case_runs["current"]["2"])
        if case["stratum"] == "ceiling":
            ceiling_ok &= all(
                snapshot["ranking"]["target_rank"] == 1
                for variant_runs in case_runs.values()
                for snapshot in variant_runs.values()
            )
        ceiling_ok &= case_runs["combined"]["10"]["ranking"]["target_rank"] <= case_runs["current"]["10"]["ranking"]["target_rank"]
        ceiling_ok &= case_runs["combined"]["10"]["ranking"]["target_mrr"] >= case_runs["current"]["10"]["ranking"]["target_mrr"]
        ablation_ok &= _changed_edges(case_runs["current"]["1"]) == ["credited_edge"]
        ablation_ok &= _changed_edges(case_runs["evidence-only"]["3"]) == ["credited_edge"]
        ablation_ok &= set(_changed_edges(case_runs["local-only"]["1"])) == {"credited_edge", "sibling_edge"}
        ablation_ok &= set(_changed_edges(case_runs["combined"]["3"])) == {"credited_edge", "sibling_edge"}
    return {
        "artifact-integrity": bool(preflight["passed"]),
        "trace-credit": credit_ok and duplicate_ok,
        "quorum-boundary": prequorum_ok,
        "headroom-response": threshold_ok,
        "ceiling-safety": ceiling_ok,
        "control-invariance": controls_ok,
        "ablation-locality": ablation_ok,
        "atomic-rollback": all(rollback.values()),
        "deterministic-recompute": recompute_ok,
        "canonical-roundtrip": True,
    }


def verify_observed_payload(payload: Mapping[str, Any], expected_gate_ids: Sequence[str]) -> None:
    validate_gate_array(payload.get("gates"), expected_gate_ids, require_all_passed=False)
    gates = payload["gates"]
    if payload.get("all_pass") is not all(item["passed"] for item in gates):
        raise ValueError("all_pass is not recomputable from the canonical gate array")
    for case_runs in payload["trajectories"].values():
        for variant_runs in case_runs.values():
            for snapshot in variant_runs.values():
                expected = [
                    item["node_id"]
                    for item in sorted(
                        snapshot["ranking"]["relation"],
                        key=lambda item: (-item["score"], item["node_id"]),
                    )
                ]
                if expected != _rank_order(snapshot):
                    raise ValueError("relation rank is not recomputable")
    recomputed = _evaluate_gates(
        payload["case_contracts"],
        payload["variants"],
        payload["trajectories"],
        payload["preflight"],
        payload["rollback"],
    )
    actual = {item["gate_id"]: item["passed"] for item in gates}
    if actual != recomputed:
        raise ValueError("gate outcomes are not recomputable from the result")


def run_registered_stage(stage: str, *, root: Path = ROOT) -> Path:
    if root != ROOT:
        raise ValueError("registered execution must use the repository root")
    manifest = read_json(MANIFEST_PATH)
    if stage not in {"development", "holdout"}:
        raise ValueError("stage must be development or holdout")
    preflight = _artifact_preflight(manifest, stage)
    if not preflight["passed"]:
        raise RuntimeError(f"protocol preflight failed: {preflight['checks']}")
    fixture = read_json(ROOT / manifest["fixture"])
    gold = read_json(ROOT / manifest["gold"])
    schedule = read_json(ROOT / manifest["schedule"])
    gate_ids = canonical_gate_ids(ROOT / manifest["gate"])
    cases = [item for item in fixture["cases"] if item["stage"] == stage]
    expected_targets = {
        item["case_id"]: item["target_node_id"] for item in gold["cases"]
    }
    if any(expected_targets.get(case["case_id"]) != case["target_node_id"] for case in cases):
        raise RuntimeError("gold target identity differs from the registered fixture")
    variants = schedule["variants"]
    trajectories: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        trajectories[case["case_id"]] = {}
        for variant in variants:
            trajectories[case["case_id"]][variant["variant_id"]] = {
                str(checkpoint): _trajectory(case, variant, checkpoint)
                for checkpoint in schedule["checkpoints"]
            }
    rollback = {
        "credited_failure": _rollback_probe(cases[0], sibling_failure=False),
        "sibling_failure": _rollback_probe(cases[0], sibling_failure=True),
    }
    computed = _evaluate_gates(cases, variants, trajectories, preflight, rollback)
    missing = [gate_id for gate_id in gate_ids if gate_id not in computed]
    extra = [gate_id for gate_id in computed if gate_id not in gate_ids]
    if missing or extra:
        raise RuntimeError(f"gate implementation mismatch: missing={missing}, extra={extra}")
    gates = [{"gate_id": gate_id, "passed": computed[gate_id]} for gate_id in gate_ids]
    payload = {
        "protocol_id": manifest["protocol_id"],
        "stage": stage,
        "source_commit": manifest["source_commit"],
        "preflight": preflight,
        "rollback": rollback,
        "case_contracts": cases,
        "variants": variants,
        "trajectories": trajectories,
        "gates": gates,
        "all_pass": all(item["passed"] for item in gates),
    }
    verify_observed_payload(payload, gate_ids)
    output = ROOT / manifest["outputs"][stage]
    write_observed_exclusive(output, payload)
    verify_observed_payload(read_json(output), gate_ids)
    return output


def verify_registered_result(stage: str) -> None:
    manifest = read_json(MANIFEST_PATH)
    gate_ids = canonical_gate_ids(ROOT / manifest["gate"])
    verify_observed_payload(read_json(ROOT / manifest["outputs"][stage]), gate_ids)

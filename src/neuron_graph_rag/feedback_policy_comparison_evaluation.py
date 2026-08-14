from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evidence_feedback import EngineConfig, NeuronGraphRAG
from .feedback import FeedbackLedger
from .models import SourceUseEvent
from .retrieval import normalize_scores


ROOT = Path(__file__).resolve().parents[2]
STEM = "feedback_policy_comparison_v1"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
EDGE_FIELDS = ("source_id", "target_id", "edge_type")
CHECKPOINTS = (0, 1, 3, 10)


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    if text.encode("utf-8") != raw:
        raise ValueError(f"non-canonical UTF-8 artifact: {path}")
    payload = json.loads(text)
    if text != json.dumps(payload, ensure_ascii=False, indent=2) + "\n":
        raise ValueError(f"non-canonical JSON artifact: {path}")
    return payload


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def write_observed_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def prove_writer_verifier_round_trip(path: Path) -> None:
    payload = {
        "probe_id": "temporary-policy-comparison-placeholder",
        "identities": ["zulu-placeholder", "alpha-placeholder", "mike-placeholder"],
        "semantic_round_trip": True,
    }
    write_observed_exclusive(path, payload)
    try:
        observed = read_json(path)
        if list(observed) != ["probe_id", "identities", "semantic_round_trip"]:
            raise ValueError("placeholder field order changed")
        if observed != payload:
            raise ValueError("placeholder semantics changed")
    finally:
        path.unlink(missing_ok=True)


def _git_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _load_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_json(MANIFEST_PATH)
    for relative, expected in manifest["artifact_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {relative}")
    artifacts: dict[str, dict[str, Any]] = {}
    for name, relative in manifest["protocol_artifacts"].items():
        path = ROOT / relative
        artifacts[name] = read_json(path)
    return manifest, artifacts


def _source_documents(
    manifest: Mapping[str, Any], fixture: Mapping[str, Any]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    commit = str(manifest["source_commit"])
    corpus_path = str(manifest["source_manifest"])
    corpus_raw = _git_bytes(commit, corpus_path)
    if _sha256_bytes(corpus_raw) != manifest["source_manifest_sha256"]:
        raise RuntimeError("source corpus manifest hash mismatch")
    corpus = json.loads(corpus_raw.decode("utf-8", errors="strict"))
    registered = {
        item["node_id"]: item
        for split in corpus["splits"].values()
        for item in split["documents"]
    }
    texts: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    for split in fixture["splits"].values():
        for node in split["nodes"]:
            node_id = str(node["node_id"])
            source = registered.get(node_id)
            if (
                source is None
                or source["path"] != node["path"]
                or source["source_url"] != node["source_url"]
            ):
                raise RuntimeError(f"source identity is not in corpus manifest: {node_id}")
            raw = _git_bytes(commit, str(node["path"]))
            actual = f"sha256:{_sha256_bytes(raw)}"
            accepted = actual == source["raw_sha256"] == node["raw_sha256"]
            checks.append(
                {
                    "node_id": node_id,
                    "path": node["path"],
                    "raw_sha256": actual,
                    "accepted": accepted,
                }
            )
            if not accepted:
                raise RuntimeError(f"source document hash mismatch: {node_id}")
            texts[node_id] = raw.decode("utf-8", errors="strict")
    return texts, checks


def _config(base: Mapping[str, Any], arm: Mapping[str, Any]) -> EngineConfig:
    values = dict(base)
    values.update(arm["engine_overrides"])
    return EngineConfig(**values)


def _populate(
    engine: NeuronGraphRAG,
    split: Mapping[str, Any],
    texts: Mapping[str, str],
) -> None:
    for node in split["nodes"]:
        engine.add_document(
            str(node["node_id"]),
            texts[str(node["node_id"])],
            metadata={"source_url": node["source_url"], "document_path": node["path"]},
        )
    for edge in split["edges"]:
        engine.add_edge(
            str(edge["source_id"]),
            str(edge["target_id"]),
            str(edge["edge_type"]),
            weight=float(edge["weight"]),
        )


def _edge_key(edge: Mapping[str, Any]) -> str:
    return "|".join(str(edge[field]) for field in EDGE_FIELDS)


def _edge_snapshot(engine: NeuronGraphRAG, split: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for identity in sorted(split["edges"], key=_edge_key):
        edge = engine.store.edge(*(str(identity[field]) for field in EDGE_FIELDS))
        confirmation = engine.store.connection.execute(
            """
            SELECT confirmation_count FROM confirmed_edge_state
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            tuple(str(identity[field]) for field in EDGE_FIELDS),
        ).fetchone()
        rows.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "reinforced_count": edge.reinforced_count,
                "confirmation_count": 0 if confirmation is None else int(confirmation["confirmation_count"]),
                "evidence_count": engine.store.feedback_evidence_count(
                    edge.source_id, edge.target_id, edge.edge_type
                ),
            }
        )
    return rows


def _rank(items: Sequence[Mapping[str, Any]], node_id: str, limit: int) -> int:
    return next(
        (int(item["rank"]) for item in items if item["node_id"] == node_id),
        limit + 1,
    )


def _query_snapshot(
    engine: NeuronGraphRAG,
    case: Mapping[str, Any],
    limit: int,
    now: float,
) -> dict[str, Any]:
    channels = engine.search_channels(str(case["relation_query"]), limit=limit, now=now)
    relation = [
        {
            "node_id": hit.node.node_id,
            "rank": hit.rank,
            "raw_graph_score": hit.graph_activation,
        }
        for hit in channels.relation.hits
    ]
    normalized = normalize_scores(
        {item["node_id"]: float(item["raw_graph_score"]) for item in relation}
    )
    for item in relation:
        item["normalized_graph_score"] = normalized[item["node_id"]]

    hybrid = engine.search(str(case["relation_query"]), limit=limit, now=now + 0.01)
    hybrid_items = [
        {
            "node_id": hit.node.node_id,
            "rank": index,
            "final_score": hit.final_score,
            "raw_graph_score": hit.graph_activation,
            "normalized_graph_score": hit.normalized_graph_activation,
        }
        for index, hit in enumerate(hybrid.hits, start=1)
    ]
    target_id = str(case["expected_terminal_node_id"])
    corrected_id = str(case["used_node_id"])
    target_rank = _rank(relation, target_id, limit)
    target_hybrid = next(item for item in hybrid_items if item["node_id"] == target_id)
    competitor = max(
        (item["final_score"] for item in hybrid_items if item["node_id"] != target_id),
        default=0.0,
    )

    direct = engine.search_channels(str(case["direct_query"]), limit=limit, now=now + 0.02)
    reverse = engine.search_channels(str(case["reverse_query"]), limit=limit, now=now + 0.03)
    direct_items = [
        {"node_id": hit.node.node_id, "rank": hit.rank}
        for hit in direct.lexical.hits
    ]
    reverse_items = [
        {"node_id": hit.node.node_id, "rank": hit.rank}
        for hit in reverse.relation.hits
    ]
    return {
        "relation": relation,
        "hybrid": hybrid_items,
        "target": {
            "node_id": target_id,
            "rank": target_rank,
            "mrr": 1.0 / target_rank,
            "hit_at_3": target_rank <= 3,
            "raw_graph_score": next(
                (item["raw_graph_score"] for item in relation if item["node_id"] == target_id),
                0.0,
            ),
            "normalized_graph_score": next(
                (item["normalized_graph_score"] for item in relation if item["node_id"] == target_id),
                0.0,
            ),
            "final_score": target_hybrid["final_score"],
            "final_score_margin": target_hybrid["final_score"] - competitor,
        },
        "corrected_sibling_rank": _rank(relation, corrected_id, limit),
        "non_target_sibling_rank": _rank(
            relation, str(case["non_target_sibling_node_id"]), limit
        ),
        "unrelated_rank": _rank(relation, str(case["unrelated_node_id"]), limit),
        "direct_lookup_rank": _rank(
            direct_items, str(case["direct_node_id"]), limit
        ),
        "reverse_direction_rank": _rank(
            reverse_items, str(case["reverse_expected_node_id"]), limit
        ),
    }


def _feedback_semantics(feedback: Any) -> dict[str, Any] | None:
    if feedback is None:
        return None
    return {
        "channel": feedback.channel,
        "reinforced": [asdict(item) for item in feedback.reinforced_edges],
        "normalized": [asdict(item) for item in feedback.normalized_sibling_edges],
        "evidence": [asdict(item) for item in feedback.evidence],
    }


def _outcome_semantics(receipt: Any) -> dict[str, Any]:
    return {
        "outcome": receipt.outcome,
        "reinforcement_applied": receipt.reinforcement_applied,
        "confirmations": [asdict(item) for item in receipt.confirmations],
        "credited_paths": [asdict(item) for item in receipt.credited_paths],
        "normalized": [asdict(item) for item in receipt.normalized_sibling_edges],
    }


def _record_event(
    engine: NeuronGraphRAG,
    arm: Mapping[str, Any],
    case: Mapping[str, Any],
    event_index: int,
    now: float,
) -> dict[str, Any]:
    ledger = FeedbackLedger(engine)
    relation = engine.search_channels(
        str(case["relation_query"]), limit=int(case["limit"]), now=now
    ).relation
    used_node_id = str(case["used_node_id"])
    if used_node_id not in {hit.node.node_id for hit in relation.hits}:
        raise RuntimeError(f"used node is absent from fresh relation trace: {used_node_id}")
    events = tuple(
        SourceUseEvent(used_node_id, stage) for stage in ("selected", "validated", "used")
    )
    use_key = f"{arm['arm_id']}-{case['case_id']}-use-{event_index}"
    used = ledger.record_source_use(relation.trace_id, events, idempotency_key=use_key, now=now + 0.1)
    used_replay = ledger.record_source_use(
        relation.trace_id, events, idempotency_key=use_key, now=now + 0.1
    )

    outcome = str(case["outcome"])
    summary = str(case["outcome_summary"])
    outcome_key = f"{arm['arm_id']}-{case['case_id']}-outcome-{event_index}"
    if arm["policy"] == "audit-only":
        payload_json = json.dumps(
            {
                "trace_id": relation.trace_id,
                "node_ids": [used_node_id],
                "outcome": outcome,
                "summary": summary,
                "external_ref": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        stored = engine.store.record_outcome(
            idempotency_key=outcome_key,
            payload_json=payload_json,
            outcome_id=f"control-{case['case_id']}-{event_index}",
            trace_id=relation.trace_id,
            node_ids=(used_node_id,),
            outcome=outcome,
            summary=summary,
            external_ref=None,
            recorded_at=now + 0.2,
        )
        replay = engine.store.record_outcome(
            idempotency_key=outcome_key,
            payload_json=payload_json,
            outcome_id=f"ignored-{case['case_id']}-{event_index}",
            trace_id=relation.trace_id,
            node_ids=(used_node_id,),
            outcome=outcome,
            summary=summary,
            external_ref=None,
            recorded_at=now + 0.2,
        )
        outcome_semantics = {
            "outcome": stored["outcome"],
            "reinforcement_applied": stored["reinforcement_applied"],
            "confirmations": [],
            "credited_paths": [],
            "normalized": [],
        }
        outcome_replay_equal = replay == stored
    else:
        receipt = ledger.record_outcome(
            relation.trace_id,
            [used_node_id],
            outcome,
            summary,
            idempotency_key=outcome_key,
            now=now + 0.2,
        )
        replay = ledger.record_outcome(
            relation.trace_id,
            [used_node_id],
            outcome,
            summary,
            idempotency_key=outcome_key,
            now=now + 0.2,
        )
        outcome_semantics = _outcome_semantics(receipt)
        outcome_replay_equal = replay == receipt
    return {
        "event": event_index,
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "fresh_relation_trace": True,
        "source_use": {
            "stages": [item.stage for item in used.events],
            "changed": [item.changed for item in used.events],
            "feedback": _feedback_semantics(used.feedback),
            "idempotency_replay_equal": used_replay == used,
        },
        "outcome": outcome_semantics,
        "outcome_idempotency_replay_equal": outcome_replay_equal,
    }


def _checkpoint(
    engine: NeuronGraphRAG,
    split: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    checkpoint: int,
    initial: Mapping[str, Any] | None,
) -> dict[str, Any]:
    queries = {
        str(case["case_id"]): _query_snapshot(
            engine, case, int(case["limit"]), 50_000.0 + checkpoint * 100 + index
        )
        for index, case in enumerate(cases)
    }
    edges = _edge_snapshot(engine, split)
    changed = []
    actual_delta_cumulative = 0.0
    if initial is not None:
        before = {_edge_key(item): item for item in initial["edges"]}
        mutation_fields = ("weight", "reinforced_count", "confirmation_count")
        changed = [
            _edge_key(item)
            for item in edges
            if any(
                item[field] != before[_edge_key(item)][field]
                for field in mutation_fields
            )
        ]
        actual_delta_cumulative = sum(
            max(0.0, float(item["weight"]) - float(before[_edge_key(item)]["weight"]))
            for item in edges
        )
    top_k_changes: dict[str, dict[str, list[str]]] = {}
    non_target_churn = 0
    if initial is not None:
        for case in cases:
            case_id = str(case["case_id"])
            current_order = [item["node_id"] for item in queries[case_id]["relation"]]
            baseline_order = [
                item["node_id"] for item in initial["queries"][case_id]["relation"]
            ]
            current_top = set(current_order[:2])
            baseline_top = set(baseline_order[:2])
            top_k_changes[case_id] = {
                "entry": sorted(current_top - baseline_top),
                "exit": sorted(baseline_top - current_top),
            }
            excluded = {
                str(case["expected_terminal_node_id"]), str(case["used_node_id"])
            }
            baseline_ranks = {node_id: rank for rank, node_id in enumerate(baseline_order, 1)}
            current_ranks = {node_id: rank for rank, node_id in enumerate(current_order, 1)}
            non_target_churn += sum(
                baseline_ranks.get(node_id) != current_ranks.get(node_id)
                for node_id in set(baseline_ranks) | set(current_ranks)
                if node_id not in excluded
            )
    return {
        "checkpoint": checkpoint,
        "queries": queries,
        "edges": edges,
        "mutation_count": len(changed),
        "actual_delta_cumulative": actual_delta_cumulative,
        "changed_edges": sorted(changed),
        "top_k_changes": top_k_changes,
        "non_target_churn": non_target_churn,
    }


def _run_arm(
    arm: Mapping[str, Any],
    base_config: Mapping[str, Any],
    split: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    texts: Mapping[str, str],
) -> dict[str, Any]:
    with NeuronGraphRAG(config=_config(base_config, arm)) as engine:
        _populate(engine, split, texts)
        checkpoints: list[dict[str, Any]] = []
        initial = _checkpoint(engine, split, cases, 0, None)
        checkpoints.append(initial)
        receipts: list[dict[str, Any]] = []
        for event_index in range(1, 11):
            for case_index, case in enumerate(cases):
                receipts.append(
                    _record_event(
                        engine,
                        arm,
                        case,
                        event_index,
                        10_000.0 + event_index * 100 + case_index,
                    )
                )
            if event_index in CHECKPOINTS:
                checkpoints.append(_checkpoint(engine, split, cases, event_index, initial))
        return {
            "arm_id": arm["arm_id"],
            "policy": arm["policy"],
            "config": asdict(_config(base_config, arm)),
            "checkpoints": checkpoints,
            "receipts": receipts,
        }


def _checkpoint_by_id(arm: Mapping[str, Any], checkpoint: int) -> Mapping[str, Any]:
    return next(item for item in arm["checkpoints"] if item["checkpoint"] == checkpoint)


def _edge_map(checkpoint: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {_edge_key(item): item for item in checkpoint["edges"]}


def _case_by_cohort(cases: Sequence[Mapping[str, Any]], cohort: str) -> Mapping[str, Any]:
    return next(case for case in cases if case["cohort"] == cohort)


def _evaluate_gates(
    preflight: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    control = arms["control"]
    used = arms["used_q3_s1"]
    confirmed = arms["confirmed_r05_s1"]
    confirmed_case = _case_by_cohort(cases, "confirmed-use")
    corrected_case = _case_by_cohort(cases, "corrected-use")
    confirmed_id = str(confirmed_case["case_id"])
    corrected_id = str(corrected_case["case_id"])

    protocol_integrity = bool(preflight["passed"]) and all(
        arm["fresh_clone_replay"]
        and all(
            receipt["source_use"]["idempotency_replay_equal"]
            and receipt["outcome_idempotency_replay_equal"]
            and receipt["fresh_relation_trace"]
            for receipt in arm["receipts"]
        )
        for arm in arms.values()
    )

    confirmation_deltas = [
        confirmation["actual_delta"]
        for receipt in confirmed["receipts"]
        if receipt["case_id"] == confirmed_id
        for confirmation in receipt["outcome"]["confirmations"]
    ]
    grouped = [confirmation_deltas[index : index + 2] for index in range(0, len(confirmation_deltas), 2)]
    confirmed_diminishing = (
        len(grouped) == 10
        and all(grouped[0])
        and max(grouped[0]) <= float(confirmed["config"]["feedback_learning_rate"])
        and all(
            max(current) <= min(previous)
            for previous, current in zip(grouped, grouped[1:])
        )
    )

    used_0 = _checkpoint_by_id(used, 0)
    used_1 = _checkpoint_by_id(used, 1)
    used_3 = _checkpoint_by_id(used, 3)
    serving_fields = ("weight", "reinforced_count", "confirmation_count")
    serving_0 = {
        _edge_key(edge): tuple(edge[field] for field in serving_fields)
        for edge in used_0["edges"]
    }
    serving_1 = {
        _edge_key(edge): tuple(edge[field] for field in serving_fields)
        for edge in used_1["edges"]
    }
    serving_3 = {
        _edge_key(edge): tuple(edge[field] for field in serving_fields)
        for edge in used_3["edges"]
    }
    used_quorum = serving_1 == serving_0 and serving_3 != serving_0

    corrected_isolation = all(
        _checkpoint_by_id(confirmed, checkpoint)["queries"][corrected_id]
        == _checkpoint_by_id(control, checkpoint)["queries"][corrected_id]
        and all(
            edge == _edge_map(_checkpoint_by_id(control, checkpoint))[_edge_key(edge)]
            for edge in _checkpoint_by_id(confirmed, checkpoint)["edges"]
            if edge["source_id"] in set(corrected_case["cluster_node_ids"])
        )
        for checkpoint in CHECKPOINTS
    )

    baseline = _checkpoint_by_id(control, 0)["queries"][confirmed_id]["target"]["mrr"]
    confirmed_one = _checkpoint_by_id(confirmed, 1)["queries"][confirmed_id]["target"]["mrr"]
    control_one = _checkpoint_by_id(control, 1)["queries"][confirmed_id]["target"]["mrr"]
    confirmed_headroom = confirmed_one > control_one if baseline < 1.0 else confirmed_one >= control_one

    confirmed_ten = _checkpoint_by_id(confirmed, 10)
    used_ten = _checkpoint_by_id(used, 10)
    control_ten = _checkpoint_by_id(control, 10)
    checkpoint_ten = (
        confirmed_ten["queries"][confirmed_id]["target"]["mrr"]
        >= used_ten["queries"][confirmed_id]["target"]["mrr"]
        and all(
            confirmed_ten["queries"][case_id][metric]
            <= control_ten["queries"][case_id][metric]
            for case_id in (confirmed_id, corrected_id)
            for metric in ("direct_lookup_rank", "reverse_direction_rank", "unrelated_rank")
        )
    )

    allowed_by_arm = {
        "control": set(),
        "confirmed_r05_s1": set(confirmed_case["allowed_mutation_edges"]),
        "used_q3_s1": set(confirmed_case["allowed_mutation_edges"])
        | set(corrected_case["allowed_mutation_edges"]),
    }
    mutation_locality = all(
        set(_checkpoint_by_id(arm, 10)["changed_edges"]) <= allowed_by_arm[arm_id]
        for arm_id, arm in arms.items()
    ) and not any(
        receipt["outcome"]["reinforcement_applied"]
        for receipt in confirmed["receipts"]
        if receipt["case_id"] == corrected_id
    )
    return {
        "protocol-integrity": protocol_integrity,
        "confirmed-diminishing": confirmed_diminishing,
        "used-quorum-boundary": used_quorum,
        "corrected-isolation": corrected_isolation,
        "confirmed-headroom": confirmed_headroom,
        "checkpoint-10-safety": checkpoint_ten,
        "mutation-locality": mutation_locality,
    }


def _preflight(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    stage: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    fixture = artifacts["fixture"]
    texts, source_checks = _source_documents(manifest, fixture)
    output = ROOT / manifest["outputs"][stage]
    other = "holdout" if stage == "development" else "development"
    checks = {
        "source_commit": all(item["accepted"] for item in source_checks),
        "split_identity": set(fixture["splits"]["development"]["identity_set"]).isdisjoint(
            fixture["splits"]["holdout"]["identity_set"]
        ),
        "schedule": tuple(artifacts["schedule"]["checkpoints"]) == CHECKPOINTS,
        "registered_output_absent": not output.exists(),
        "conditional_holdout": (
            not (ROOT / manifest["outputs"][other]).exists()
            if stage == "development"
            else (ROOT / manifest["outputs"]["development"]).exists()
            and read_json(ROOT / manifest["outputs"]["development"])["all_pass"] is True
        ),
        "result_schema": artifacts["result_schema"]["top_level_fields"]
        == [
            "protocol_id",
            "stage",
            "source_commit",
            "protocol_hashes",
            "preflight",
            "arms",
            "gates",
            "all_pass",
            "interpretation_ja",
        ],
        "result_free_audit": artifacts["audit"]["result_free"] is True
        and artifacts["audit"]["placeholder_round_trip_passed"] is True,
    }
    return {
        "checks": checks,
        "source_checks": source_checks,
        "passed": all(checks.values()),
    }, texts


def _ordered_gate_ids(gate: Mapping[str, Any]) -> list[str]:
    ids = [str(item["gate_id"]) for item in gate["gates"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("gate IDs must be non-empty and unique")
    return ids


def _semantic_hash(arm: Mapping[str, Any]) -> str:
    return _sha256_bytes(_encoded(arm))


def verify_observed_payload(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    if list(payload) != artifacts["result_schema"]["top_level_fields"]:
        raise ValueError("observed top-level field order differs from frozen schema")
    if payload["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("observed protocol identity differs")
    gate_ids = _ordered_gate_ids(artifacts["gate"])
    actual_ids = [item.get("gate_id") for item in payload["gates"]]
    if actual_ids != gate_ids or not all(isinstance(item.get("passed"), bool) for item in payload["gates"]):
        raise ValueError("observed gate array differs from registration")
    arms = {item["arm_id"]: item for item in payload["arms"]}
    if [item["arm_id"] for item in payload["arms"]] != [
        item["arm_id"] for item in artifacts["schedule"]["arms"]
    ]:
        raise ValueError("observed arm order differs from registration")
    for arm in payload["arms"]:
        if list(arm) != artifacts["result_schema"]["arm_fields"]:
            raise ValueError("observed arm field order differs from frozen schema")
        if any(
            list(checkpoint) != artifacts["result_schema"]["checkpoint_fields"]
            for checkpoint in arm["checkpoints"]
        ):
            raise ValueError("observed checkpoint field order differs from frozen schema")
    computed = _evaluate_gates(payload["preflight"], artifacts["gold"]["splits"][payload["stage"]], arms)
    if {item["gate_id"]: item["passed"] for item in payload["gates"]} != computed:
        raise ValueError("gate results are not recomputable")
    if payload["all_pass"] is not all(item["passed"] for item in payload["gates"]):
        raise ValueError("all_pass is not recomputable")
    expected_decision = "支持" if payload["all_pass"] else "不支持"
    if payload["interpretation_ja"]["decision"] != expected_decision:
        raise ValueError("human-readable decision differs from gates")


def run_registered_stage(stage: str) -> Path:
    if stage not in {"development", "holdout"}:
        raise ValueError("stage must be development or holdout")
    manifest, artifacts = _load_protocol()
    preflight, texts = _preflight(manifest, artifacts, stage)
    if not preflight["passed"]:
        raise RuntimeError(f"protocol preflight failed: {preflight['checks']}")
    split = artifacts["fixture"]["splits"][stage]
    cases = artifacts["gold"]["splits"][stage]
    observed_arms: list[dict[str, Any]] = []
    for arm in artifacts["schedule"]["arms"]:
        primary = _run_arm(arm, artifacts["schedule"]["base_engine_config"], split, cases, texts)
        replay = _run_arm(arm, artifacts["schedule"]["base_engine_config"], split, cases, texts)
        semantic_sha256 = _semantic_hash(primary)
        primary["semantic_sha256"] = semantic_sha256
        primary["fresh_clone_replay"] = replay == {key: primary[key] for key in ("arm_id", "policy", "config", "checkpoints", "receipts")}
        observed_arms.append(primary)
    arms = {item["arm_id"]: item for item in observed_arms}
    computed = _evaluate_gates(preflight, cases, arms)
    gate_ids = _ordered_gate_ids(artifacts["gate"])
    if set(computed) != set(gate_ids):
        raise RuntimeError("gate implementation differs from frozen gate registry")
    gates = [{"gate_id": gate_id, "passed": computed[gate_id]} for gate_id in gate_ids]
    all_pass = all(item["passed"] for item in gates)
    decision = "支持" if all_pass else "不支持"
    payload = {
        "protocol_id": manifest["protocol_id"],
        "stage": stage,
        "source_commit": manifest["source_commit"],
        "protocol_hashes": preflight and {
            relative: manifest["artifact_sha256"][relative]
            for relative in manifest["artifact_sha256"]
        },
        "preflight": preflight,
        "arms": observed_arms,
        "gates": gates,
        "all_pass": all_pass,
        "interpretation_ja": {
            "decision": decision,
            "summary": (
                "固定した controlled corpus と gate の範囲で confirmed+diminishing candidate を支持する。"
                if all_pass
                else "固定した controlled corpus と gate の範囲で comparative advantage は成立しなかった。"
            ),
            "scope": "external corpus、Agent end-to-end、production quality、default adoption へ一般化しない。",
        },
    }
    verify_observed_payload(payload, manifest, artifacts)
    output = ROOT / manifest["outputs"][stage]
    write_observed_exclusive(output, payload)
    verify_observed_payload(read_json(output), manifest, artifacts)
    return output


def verify_registered_result(stage: str) -> None:
    manifest, artifacts = _load_protocol()
    verify_observed_payload(read_json(ROOT / manifest["outputs"][stage]), manifest, artifacts)

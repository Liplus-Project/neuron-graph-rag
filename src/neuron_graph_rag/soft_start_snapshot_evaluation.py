from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .config_provenance import effective_config_provenance
from .corpus_integrity import verify_manifest_source_hashes
from .evidence_feedback import EngineConfig, NeuronGraphRAG
from .feedback import FeedbackLedger
from .models import FeedbackReceipt, SourceUseEvent


ROOT = Path(__file__).resolve().parents[2]
STEM = "soft_start_snapshot_v1"
MANIFEST_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.manifest.json"
EDGE_FIELDS = ("source_id", "target_id", "edge_type")
STAGES = ("development", "holdout")
POLICIES = (
    "control",
    "used_q3_s1",
    "confirmed_r05_s1",
    "soft_start_r025_r05_s1",
)
_ABSOLUTE_PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/Users/|/home/|\\\\[^\\]+\\[^\\]+)"
)
_CREDENTIAL = re.compile(
    r"(?i)(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?:token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,}]+)"
)


def _encoded(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    payload = json.loads(text)
    if text != json.dumps(payload, ensure_ascii=False, indent=2) + "\n":
        raise ValueError(f"non-canonical JSON artifact: {path}")
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
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


def assert_public_payload(payload: Any) -> None:
    """Reject private paths, database text, and common credential shapes."""

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                if name.lower() in {
                    "node_text",
                    "document_text",
                    "raw_text",
                    "private_path",
                    "source_path",
                    "snapshot_path",
                }:
                    raise ValueError(f"private field is forbidden: {'.'.join((*path, name))}")
                walk(item, (*path, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))
        elif isinstance(value, str):
            if value.startswith(("/", "\\")) or _ABSOLUTE_PRIVATE_PATH.search(value):
                raise ValueError(f"absolute private path is forbidden: {'.'.join(path)}")
            if _CREDENTIAL.search(value):
                raise ValueError(f"credential-shaped value is forbidden: {'.'.join(path)}")

    walk(payload, ())


def _schema_identity(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = [
        {"name": str(row[0]), "sql": str(row[1])}
        for row in connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    ]
    return {
        "schema_sha256": _sha256_bytes(_encoded(rows)),
        "table_names": [row["name"] for row in rows],
    }


def acquire_transactional_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Create one exclusive SQLite backup without exposing either local path."""
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("source and snapshot must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(reservation)
    source_before = _sha256(source)
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        source_connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.execute("PRAGMA query_only = ON")
            source_connection.backup(destination_connection)
            destination_connection.commit()
            if destination_connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("snapshot integrity check failed")
            schema = _schema_identity(destination_connection)
            counts = {
                name: int(destination_connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
                for name in ("nodes", "edges", "retrievals", "success_feedback", "delayed_outcomes")
            }
        finally:
            destination_connection.close()
            source_connection.close()
        source_after = _sha256(source)
        if source_after != source_before:
            raise RuntimeError("source database container changed during snapshot acquisition")
        provenance = {
            "source_locator": "local_codex_ngr_database",
            "source_access": "sqlite-uri-mode-ro-query-only",
            "capture_method": "sqlite-backup-api",
            "captured_at": captured_at,
            "source_container_sha256_before": source_before,
            "source_container_sha256_after": source_after,
            "snapshot_sha256": _sha256(destination),
            "snapshot_size": destination.stat().st_size,
            **schema,
            "row_counts": counts,
        }
        assert_public_payload(provenance)
        return provenance
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def prove_writer_verifier_round_trip(path: Path) -> None:
    payload = {
        "probe_id": "temporary-soft-start-snapshot-placeholder",
        "identities": ["zulu-placeholder", "alpha-placeholder", "mike-placeholder"],
        "semantic_round_trip": True,
    }
    write_json_exclusive(path, payload)
    try:
        if read_json(path) != payload:
            raise ValueError("placeholder semantics changed")
    finally:
        path.unlink(missing_ok=True)


def _load_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_json(MANIFEST_PATH)
    registered = verify_manifest_source_hashes(
        ROOT, MANIFEST_PATH, manifest["artifact_sha256"]
    )
    artifacts = {
        name: json.loads(registered.artifact_bytes[relative].decode("utf-8", errors="strict"))
        for name, relative in manifest["protocol_artifacts"].items()
    }
    assert_public_payload(manifest)
    assert_public_payload(artifacts)
    return manifest, artifacts


def _load_current_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load current bytes for the result-free pre-commit probe only."""
    manifest = read_json(MANIFEST_PATH)
    for relative, expected in manifest["artifact_sha256"].items():
        path = ROOT / relative
        if _sha256(path) != expected:
            raise ValueError(f"current artifact hash mismatch: {relative}")
    artifacts = {
        name: read_json(ROOT / relative)
        for name, relative in manifest["protocol_artifacts"].items()
    }
    assert_public_payload(manifest)
    assert_public_payload(artifacts)
    return manifest, artifacts


def _config(
    arm: Mapping[str, Any], base_engine_config: Mapping[str, Any] | None = None
) -> EngineConfig:
    values = dict(base_engine_config or {})
    values.update(arm["engine_config"])
    return EngineConfig(**values)


def _edge_key(value: Mapping[str, Any] | Sequence[str]) -> str:
    if isinstance(value, Mapping):
        return "|".join(str(value[field]) for field in EDGE_FIELDS)
    return "|".join(str(item) for item in value)


def _edge_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for edge in engine.store.list_edges():
        identity = (edge.source_id, edge.target_id, edge.edge_type)
        confirmed = engine.store.connection.execute(
            """
            SELECT confirmation_count, base_increment FROM confirmed_edge_state
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            identity,
        ).fetchone()
        soft = engine.store.connection.execute(
            """
            SELECT confirmation_count, base_increment, soft_start_ratio
            FROM soft_start_edge_state
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            identity,
        ).fetchone()
        states.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "reinforced_count": edge.reinforced_count,
                "evidence_count": engine.store.feedback_evidence_count(*identity),
                "confirmation_count": (
                    int(soft["confirmation_count"])
                    if soft is not None
                    else 0 if confirmed is None else int(confirmed["confirmation_count"])
                ),
                "base_increment": (
                    float(soft["base_increment"])
                    if soft is not None
                    else None if confirmed is None else float(confirmed["base_increment"])
                ),
                "soft_start_ratio": None if soft is None else float(soft["soft_start_ratio"]),
            }
        )
    return sorted(states, key=_edge_key)


def _rank(items: Sequence[Mapping[str, Any]], node_id: str, limit: int) -> int:
    return next((int(item["rank"]) for item in items if item["node_id"] == node_id), limit + 1)


def _metrics(engine: NeuronGraphRAG, case: Mapping[str, Any], now: float) -> dict[str, Any]:
    limit = int(case["retrieval_limit"])
    channels = engine.search_channels(str(case["query"]), limit=limit, now=now)
    relation = [
        {
            "node_id": hit.node.node_id,
            "rank": hit.rank,
            "score": hit.channel_score,
        }
        for hit in channels.relation.hits
    ]
    target_id = str(case["used_node_id"])
    target_rank = _rank(relation, target_id, limit)
    target_score = next((float(item["score"]) for item in relation if item["node_id"] == target_id), 0.0)
    competitor = max(
        (float(item["score"]) for item in relation if item["node_id"] != target_id),
        default=0.0,
    )
    direct = engine.search(str(case["direct_query"]), limit=limit, now=now + 0.01)
    direct_items = [
        {"node_id": hit.node.node_id, "rank": rank}
        for rank, hit in enumerate(direct.hits, 1)
    ]
    reverse = engine.search_channels(str(case["reverse_query"]), limit=limit, now=now + 0.02)
    reverse_items = [
        {"node_id": hit.node.node_id, "rank": hit.rank}
        for hit in reverse.relation.hits
    ]
    return {
        "target_rank": target_rank,
        "target_mrr": 1.0 / target_rank,
        "target_hit_at_k": target_rank <= int(case["hit_k"]),
        "score_margin": target_score - competitor,
        "top_k": [item["node_id"] for item in relation[: int(case["hit_k"])]],
        "direct_rank": _rank(direct_items, str(case["direct_node_id"]), limit),
        "reverse_rank": _rank(reverse_items, str(case["reverse_node_id"]), limit),
    }


def _feedback_semantics(feedback: FeedbackReceipt | None) -> dict[str, Any] | None:
    if feedback is None:
        return None
    return {
        "channel": feedback.channel,
        "reinforced": [asdict(item) for item in feedback.reinforced_edges],
        "normalized": [asdict(item) for item in feedback.normalized_sibling_edges],
        "evidence": [asdict(item) for item in feedback.evidence],
    }


def _source_use(
    engine: NeuronGraphRAG,
    arm_id: str,
    trace_id: str,
    node_id: str,
    key: str,
    now: float,
) -> dict[str, Any]:
    events = tuple(SourceUseEvent(node_id, stage) for stage in ("selected", "validated", "used"))
    if arm_id == "control":
        payload = json.dumps(
            {"trace_id": trace_id, "events": [asdict(event) for event in events]},
            sort_keys=True,
            separators=(",", ":"),
        )
        arguments = {
            "idempotency_key": key,
            "payload_json": payload,
            "receipt_id": f"control-{key}",
            "trace_id": trace_id,
            "created_at": now,
            "events": tuple((event.node_id, event.stage) for event in events),
            "apply_feedback": None,
            "confirmation_candidate": False,
        }
        receipt = engine.store.record_source_use(**arguments)
        replay = engine.store.record_source_use(**arguments)
        return {
            "stages": [event["stage"] for event in receipt["events"]],
            "changed": [bool(event["changed"]) for event in receipt["events"]],
            "newly_used_node_ids": receipt["newly_used_node_ids"],
            "feedback": None,
            "idempotency_replay_equal": replay == receipt,
        }
    ledger = FeedbackLedger(engine)
    receipt = ledger.record_source_use(trace_id, events, idempotency_key=key, now=now)
    replay = ledger.record_source_use(trace_id, events, idempotency_key=key, now=now)
    return {
        "stages": [item.stage for item in receipt.events],
        "changed": [item.changed for item in receipt.events],
        "newly_used_node_ids": list(receipt.newly_used_node_ids),
        "feedback": _feedback_semantics(receipt.feedback),
        "idempotency_replay_equal": replay == receipt,
    }


def _outcome(
    engine: NeuronGraphRAG,
    arm_id: str,
    trace_id: str,
    node_id: str,
    outcome: str,
    key: str,
    now: float,
) -> dict[str, Any]:
    summary = "fixed registered outcome"
    if arm_id == "control":
        payload = json.dumps(
            {
                "trace_id": trace_id,
                "node_ids": [node_id],
                "outcome": outcome,
                "summary": summary,
                "external_ref": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        arguments = {
            "idempotency_key": key,
            "payload_json": payload,
            "outcome_id": f"control-{key}",
            "trace_id": trace_id,
            "node_ids": (node_id,),
            "outcome": outcome,
            "summary": summary,
            "external_ref": None,
            "recorded_at": now,
        }
        receipt = engine.store.record_outcome(**arguments)
        replay = engine.store.record_outcome(**arguments)
        return {
            "outcome": outcome,
            "reinforcement_applied": False,
            "confirmations": [],
            "credited_paths": [],
            "normalized": [],
            "idempotency_replay_equal": replay == receipt,
        }
    ledger = FeedbackLedger(engine)
    receipt = ledger.record_outcome(
        trace_id,
        [node_id],
        outcome,
        summary,
        idempotency_key=key,
        now=now,
    )
    replay = ledger.record_outcome(
        trace_id,
        [node_id],
        outcome,
        summary,
        idempotency_key=key,
        now=now,
    )
    return {
        "outcome": receipt.outcome,
        "reinforcement_applied": receipt.reinforcement_applied,
        "confirmations": [asdict(item) for item in receipt.confirmations],
        "credited_paths": [asdict(item) for item in receipt.credited_paths],
        "normalized": [asdict(item) for item in receipt.normalized_sibling_edges],
        "idempotency_replay_equal": replay == receipt,
    }


def _checkpoint(
    engine: NeuronGraphRAG,
    case: Mapping[str, Any],
    name: str,
    now: float,
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = _metrics(engine, case, now)
    edges = _edge_state(engine)
    if baseline is None:
        changed_edges: list[str] = []
        entry: list[str] = []
        exit_items: list[str] = []
        non_target_churn = 0
    else:
        previous = {_edge_key(item): item for item in baseline["edges"]}
        changed_edges = [
            _edge_key(item)
            for item in edges
            if any(
                item[field] != previous[_edge_key(item)][field]
                for field in ("weight", "reinforced_count", "evidence_count", "confirmation_count")
            )
        ]
        current_top = set(metrics["top_k"])
        baseline_top = set(baseline["metrics"]["top_k"])
        entry = sorted(current_top - baseline_top)
        exit_items = sorted(baseline_top - current_top)
        excluded = {str(case["used_node_id"]), str(case["source_node_id"])}
        non_target_churn = len((current_top ^ baseline_top) - excluded)
    return {
        "name": name,
        "metrics": metrics,
        "edges": edges,
        "changed_edges": changed_edges,
        "top_k_entry": entry,
        "top_k_exit": exit_items,
        "non_target_churn": non_target_churn,
    }


def _select_trace(engine: NeuronGraphRAG, case: Mapping[str, Any], now: float) -> str:
    channels = engine.search_channels(
        str(case["query"]), limit=int(case["retrieval_limit"]), now=now
    )
    trace = channels.relation if case["search_surface"] == "relation" else channels.lexical
    node_id = str(case["used_node_id"])
    if node_id not in {hit.node.node_id for hit in trace.hits}:
        raise RuntimeError("registered used node is absent from its search trace")
    if case["search_surface"] == "relation" and case["case_role"] in {"confirmed", "corrected"}:
        paths = engine.store.retrieval_paths(trace.trace_id, node_id)
        expected = tuple(
            str(case["credited_edge"][field]) for field in EDGE_FIELDS
        )
        if not any(
            any(
                (step["source_id"], step["target_id"], step["edge_type"]) == expected
                for step in path["steps"]
            )
            for path in paths
        ):
            raise RuntimeError("registered credited edge is absent from relation provenance")
    return trace.trace_id


def _run_case(
    engine: NeuronGraphRAG,
    arm_id: str,
    case: Mapping[str, Any],
    case_index: int,
) -> dict[str, Any]:
    clock = 100_000.0 + case_index * 10_000.0
    baseline = _checkpoint(engine, case, "baseline", clock, None)
    checkpoints = [baseline]
    receipts: list[dict[str, Any]] = []
    for iteration in range(1, 4):
        trace_id = _select_trace(engine, case, clock + iteration * 100.0)
        use = _source_use(
            engine,
            arm_id,
            trace_id,
            str(case["used_node_id"]),
            f"{arm_id}-{case['case_id']}-use-{iteration}",
            clock + iteration * 100.0 + 0.1,
        )
        checkpoints.append(
            _checkpoint(
                engine,
                case,
                f"used_{iteration}",
                clock + iteration * 100.0 + 10.0,
                baseline,
            )
        )
        outcome = _outcome(
            engine,
            arm_id,
            trace_id,
            str(case["used_node_id"]),
            str(case["outcome"]),
            f"{arm_id}-{case['case_id']}-outcome-{iteration}",
            clock + iteration * 100.0 + 20.0,
        )
        receipts.append({"iteration": iteration, "source_use": use, "outcome": outcome})
        checkpoints.append(
            _checkpoint(
                engine,
                case,
                f"outcome_{iteration}",
                clock + iteration * 100.0 + 30.0,
                baseline,
            )
        )
    return {
        "case_id": case["case_id"],
        "case_role": case["case_role"],
        "checkpoints": checkpoints,
        "receipts": receipts,
    }


def _run_arm(
    snapshot: Path,
    arm: Mapping[str, Any],
    base_engine_config: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    with TemporaryDirectory() as directory:
        clone = Path(directory) / "arm.sqlite"
        shutil.copyfile(snapshot, clone)
        with NeuronGraphRAG(clone, config=_config(arm, base_engine_config)) as engine:
            initial = _edge_state(engine)
            observed_cases = [
                _run_case(engine, str(arm["arm_id"]), case, index)
                for index, case in enumerate(cases)
            ]
            final = _edge_state(engine)
        return {
            "arm_id": arm["arm_id"],
            "policy": arm["policy"],
            "effective_config_provenance": effective_config_provenance(
                _config(arm, base_engine_config)
            ),
            "initial_edges": initial,
            "cases": observed_cases,
            "final_edges": final,
        }


def _checkpoint_by_name(case: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in case["checkpoints"] if item["name"] == name)


def _case_by_role(arm: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return next(item for item in arm["cases"] if item["case_role"] == role)


def _edge_at(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return next(item for item in checkpoint["edges"] if _edge_key(item) == key)


def _delta(case: Mapping[str, Any], checkpoint: str, key: str) -> float:
    baseline = _edge_at(_checkpoint_by_name(case, "baseline"), key)
    current = _edge_at(_checkpoint_by_name(case, checkpoint), key)
    return float(current["weight"]) - float(baseline["weight"])


def evaluate_gates(
    preflight: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    control = arms["control"]
    used = arms["used_q3_s1"]
    confirmed = arms["confirmed_r05_s1"]
    soft = arms["soft_start_r025_r05_s1"]
    registered = {str(case["case_id"]): case for case in cases}
    all_receipts = [
        receipt
        for arm in arms.values()
        for case in arm["cases"]
        for receipt in case["receipts"]
    ]
    protocol_integrity = bool(preflight["passed"]) and all(
        bool(arm["fresh_clone_replay"]) for arm in arms.values()
    ) and all(
        receipt["source_use"]["idempotency_replay_equal"]
        and receipt["outcome"]["idempotency_replay_equal"]
        for receipt in all_receipts
    )

    confirmed_case = _case_by_role(soft, "confirmed")
    confirmed_key = _edge_key(registered[str(confirmed_case["case_id"])]["credited_edge"])
    soft_receipts = confirmed_case["receipts"]
    provisional = _delta(confirmed_case, "used_1", confirmed_key)
    first_total = _delta(confirmed_case, "outcome_1", confirmed_key)
    later = [
        receipt["outcome"]["confirmations"][0]
        for receipt in soft_receipts[1:]
        if receipt["outcome"]["confirmations"]
    ]
    confirmed_reference = _delta(
        _case_by_role(confirmed, "confirmed"), "outcome_1", confirmed_key
    )
    soft_schedule = (
        provisional > 0.0
        and first_total <= confirmed_reference + 1e-12
        and abs(first_total - confirmed_reference) <= 1e-12
        and [item["multiplier"] for item in later] == [0.5, 0.25]
        and later[1]["actual_delta"] <= later[0]["actual_delta"]
    )

    used_case = _case_by_role(used, "confirmed")
    confirmed_only_case = _case_by_role(confirmed, "confirmed")
    policy_boundaries = (
        _delta(used_case, "used_1", confirmed_key) == 0.0
        and _delta(used_case, "used_2", confirmed_key) == 0.0
        and _delta(used_case, "used_3", confirmed_key) > 0.0
        and _delta(confirmed_only_case, "used_1", confirmed_key) == 0.0
        and _delta(confirmed_only_case, "outcome_1", confirmed_key) > 0.0
        and all(initial == final for initial, final in zip(control["initial_edges"], control["final_edges"]))
    )

    soft_first = _checkpoint_by_name(confirmed_case, "used_1")["metrics"]["target_mrr"]
    used_first = _checkpoint_by_name(used_case, "used_1")["metrics"]["target_mrr"]
    confirmed_first = _checkpoint_by_name(confirmed_only_case, "used_1")["metrics"]["target_mrr"]
    soft_final = _checkpoint_by_name(confirmed_case, "outcome_3")["metrics"]["target_mrr"]
    used_final = _checkpoint_by_name(used_case, "outcome_3")["metrics"]["target_mrr"]
    latency_quality = (
        provisional > 0.0
        and _delta(used_case, "used_1", confirmed_key) == 0.0
        and _delta(confirmed_only_case, "used_1", confirmed_key) == 0.0
        and soft_first >= min(used_first, confirmed_first)
        and soft_final >= used_final
    )

    corrected = _case_by_role(soft, "corrected")
    corrected_config = registered[str(corrected["case_id"])]
    corrected_key = _edge_key(corrected_config["credited_edge"])
    corrected_used = _edge_at(_checkpoint_by_name(corrected, "used_1"), corrected_key)
    corrected_base = float(corrected_used["base_increment"] or 0.0)
    corrected_cost = _delta(corrected, "outcome_3", corrected_key)
    negative_cost = (
        corrected_cost > 0.0
        and corrected_cost <= corrected_base * float(thresholds["negative_provisional_ratio_max"]) + 1e-12
        and _delta(corrected, "used_1", corrected_key) == corrected_cost
        and not any(
            receipt["outcome"]["reinforcement_applied"] for receipt in corrected["receipts"]
        )
    )

    control_roles = {"lexical", "zero_hop"}
    control_safety = True
    for arm in arms.values():
        for case in arm["cases"]:
            baseline = _checkpoint_by_name(case, "baseline")
            final = _checkpoint_by_name(case, "outcome_3")
            if case["case_role"] in control_roles and final["changed_edges"]:
                control_safety = False
            if (
                final["metrics"]["direct_rank"] > baseline["metrics"]["direct_rank"]
                or final["metrics"]["reverse_rank"] > baseline["metrics"]["reverse_rank"]
            ):
                control_safety = False

    locality = True
    for arm_id, arm in arms.items():
        initial = {_edge_key(item): item for item in arm["initial_edges"]}
        final = {_edge_key(item): item for item in arm["final_edges"]}
        changed = {
            key
            for key in initial
            if any(
                initial[key][field] != final[key][field]
                for field in ("weight", "reinforced_count", "evidence_count", "confirmation_count")
            )
        }
        allowed = set() if arm_id == "control" else {
            edge
            for case in cases
            if case["case_role"] in {"confirmed", "corrected"}
            for edge in case["allowed_mutation_edges"]
        }
        if not changed <= allowed:
            locality = False
    locality = locality and bool(preflight["snapshot_unchanged"])
    return {
        "protocol-integrity": protocol_integrity,
        "soft-start-schedule": soft_schedule,
        "policy-boundaries": policy_boundaries,
        "learning-latency-and-quality": latency_quality,
        "negative-provisional-bound": negative_cost,
        "control-rank-and-mutation-safety": control_safety,
        "mutation-locality-and-source-isolation": locality,
    }


def _preflight(
    snapshot: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    output = ROOT / str(manifest["outputs"][stage])
    development = ROOT / str(manifest["outputs"]["development"])
    holdout = ROOT / str(manifest["outputs"]["holdout"])
    fixture = artifacts["fixture"]
    cases = fixture["stages"][stage]
    snapshot_before = _sha256(snapshot)
    with sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        schema = _schema_identity(connection)
        row_counts = {
            name: int(connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
            for name in manifest["snapshot"]["row_counts"]
        }
        node_ids = {
            str(row[0]) for row in connection.execute("SELECT node_id FROM nodes")
        }
        edge_keys = {
            "|".join(str(item) for item in row)
            for row in connection.execute(
                "SELECT source_id, target_id, edge_type FROM edges"
            )
        }
    registered_nodes = {
        str(case[field])
        for case in cases
        for field in ("used_node_id", "source_node_id", "direct_node_id", "reverse_node_id")
    }
    registered_edges = {
        _edge_key(case["credited_edge"])
        for case in cases
        if case["case_role"] in {"confirmed", "corrected"}
    }
    trace_eligibility = True
    try:
        with TemporaryDirectory() as directory:
            probe = Path(directory) / "probe.sqlite"
            shutil.copyfile(snapshot, probe)
            probe_arm = next(
                arm
                for arm in artifacts["schedule"]["arms"]
                if arm["arm_id"] == "soft_start_r025_r05_s1"
            )
            with NeuronGraphRAG(
                probe,
                config=_config(
                    probe_arm, artifacts["schedule"]["base_engine_config"]
                ),
            ) as engine:
                for index, case in enumerate(cases):
                    _select_trace(engine, case, 80_000.0 + index)
    except Exception:  # noqa: BLE001 - the public report records only pass/fail
        trace_eligibility = False
    checks = {
        "snapshot_hash": snapshot_before == manifest["snapshot"]["snapshot_sha256"],
        "snapshot_size": snapshot.stat().st_size == manifest["snapshot"]["snapshot_size"],
        "snapshot_schema": schema["schema_sha256"]
        == manifest["snapshot"]["schema_sha256"]
        and schema["table_names"] == manifest["snapshot"]["table_names"],
        "snapshot_row_counts": row_counts == manifest["snapshot"]["row_counts"],
        "registered_nodes": registered_nodes <= node_ids,
        "registered_edges": registered_edges <= edge_keys,
        "trace_eligibility": trace_eligibility,
        "arm_order": [arm["arm_id"] for arm in artifacts["schedule"]["arms"]] == list(POLICIES),
        "case_identity": len({case["case_id"] for case in cases}) == len(cases),
        "event_order": artifacts["schedule"]["event_order"]
        == ["used_1", "outcome_1", "used_2", "outcome_2", "used_3", "outcome_3"],
        "registered_output_absent": not output.exists(),
        "result_free_stage_order": stage != "development" or not holdout.exists(),
        "conditional_holdout": stage == "development"
        or (
            development.exists()
            and _verified_development_passed(development, manifest, artifacts)
        ),
        "privacy": True,
        "placeholder_round_trip": artifacts["audit"]["placeholder_round_trip_passed"] is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "snapshot_sha256_before": snapshot_before,
        "snapshot_sha256_after": _sha256(snapshot),
        "snapshot_unchanged": _sha256(snapshot) == snapshot_before,
    }


def _gate_ids(gate: Mapping[str, Any]) -> list[str]:
    values = [str(item["gate_id"]) for item in gate["gates"]]
    if not values or len(values) != len(set(values)):
        raise ValueError("gate IDs must be non-empty and unique")
    return values


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_encoded(payload))


def _verified_development_passed(
    path: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> bool:
    payload = read_json(path)
    verify_observed_payload(payload, manifest, artifacts)
    return payload["stage"] == "development" and payload["all_hard_gates_pass"] is True


def _interpretation(decision: str) -> dict[str, str]:
    summaries = {
        "支持": "固定したlocal snapshotとhard gateの範囲でsoft-start cutover候補を支持する。",
        "不支持": "固定したlocal snapshotとhard gateの範囲でsoft-startの比較優位は成立しなかった。",
        "判定不能": "固定protocolを完走できず、soft-start cutover可否は判定不能である。",
    }
    return {
        "decision": decision,
        "summary": summaries[decision],
        "scope": "local snapshot限定。external corpus、production quality、library defaultへ一般化せず、source databaseとlive configを変更しない。",
    }


def verify_observed_payload(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    assert_public_payload(payload)
    if list(payload) != artifacts["result_schema"]["top_level_fields"]:
        raise ValueError("observed top-level field order differs from frozen schema")
    if payload["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("observed protocol identity differs")
    if payload["stage"] not in STAGES:
        raise ValueError("observed stage differs from registration")
    if payload["snapshot"] != manifest["snapshot"]:
        raise ValueError("observed snapshot identity differs from registration")
    if payload["protocol_hashes"] != manifest["artifact_sha256"]:
        raise ValueError("observed protocol hashes differ from registration")
    gate_ids = _gate_ids(artifacts["gate"])
    if [item["gate_id"] for item in payload["gates"]] != gate_ids:
        raise ValueError("observed gate order differs")
    if payload["status"] == "completed":
        expected_order = [
            arm["arm_id"] for arm in artifacts["schedule"]["arms"]
        ]
        if [arm["arm_id"] for arm in payload["arms"]] != expected_order:
            raise ValueError("observed arm order differs from registration")
        semantic_fields = (
            "arm_id",
            "policy",
            "effective_config_provenance",
            "initial_edges",
            "cases",
            "final_edges",
        )
        for arm in payload["arms"]:
            semantic = {field: arm[field] for field in semantic_fields}
            if arm["semantic_sha256"] != _semantic_hash(semantic):
                raise ValueError("observed arm semantic hash differs")
        arms = {arm["arm_id"]: arm for arm in payload["arms"]}
        computed = evaluate_gates(
            payload["preflight"],
            artifacts["fixture"]["stages"][payload["stage"]],
            arms,
            artifacts["gate"]["thresholds"],
        )
        if {item["gate_id"]: item["passed"] for item in payload["gates"]} != computed:
            raise ValueError("gate results are not recomputable")
        if payload["failure_code"] is not None:
            raise ValueError("completed result cannot carry a failure code")
        expected = "支持" if all(computed.values()) else "不支持"
    elif payload["status"] == "indeterminate":
        if payload["arms"] or any(item["passed"] for item in payload["gates"]):
            raise ValueError("indeterminate result must not claim observations")
        if payload["failure_code"] not in artifacts["result_schema"]["failure_codes"]:
            raise ValueError("unknown failure code")
        expected = "判定不能"
    else:
        raise ValueError("unknown result status")
    if payload["all_hard_gates_pass"] is not all(item["passed"] for item in payload["gates"]):
        raise ValueError("all_hard_gates_pass is not recomputable")
    if payload["interpretation_ja"] != _interpretation(expected):
        raise ValueError("interpretation differs from frozen decision mapping")


def run_registered_stage(stage: str, snapshot: Path) -> Path:
    if stage not in STAGES:
        raise ValueError("stage must be development or holdout")
    manifest, artifacts = _load_protocol()
    output = ROOT / str(manifest["outputs"][stage])
    if output.exists():
        raise FileExistsError(f"registered output already exists: {stage}")
    preflight: dict[str, Any] = {
        "checks": {},
        "passed": False,
        "snapshot_sha256_before": None,
        "snapshot_sha256_after": None,
        "snapshot_unchanged": False,
    }
    arms: list[dict[str, Any]] = []
    failure_code: str | None = None
    try:
        preflight = _preflight(snapshot, manifest, artifacts, stage)
        if not preflight["passed"]:
            raise RuntimeError("preflight")
        cases = artifacts["fixture"]["stages"][stage]
        for arm in artifacts["schedule"]["arms"]:
            primary = _run_arm(
                snapshot, arm, artifacts["schedule"]["base_engine_config"], cases
            )
            replay = _run_arm(
                snapshot, arm, artifacts["schedule"]["base_engine_config"], cases
            )
            primary["semantic_sha256"] = _semantic_hash(primary)
            primary["fresh_clone_replay"] = replay == {
                key: primary[key]
                for key in ("arm_id", "policy", "effective_config_provenance", "initial_edges", "cases", "final_edges")
            }
            arms.append(primary)
        preflight["snapshot_sha256_after"] = _sha256(snapshot)
        preflight["snapshot_unchanged"] = (
            preflight["snapshot_sha256_after"] == preflight["snapshot_sha256_before"]
        )
        computed = evaluate_gates(
            preflight,
            cases,
            {arm["arm_id"]: arm for arm in arms},
            artifacts["gate"]["thresholds"],
        )
        status = "completed"
        decision = "支持" if all(computed.values()) else "不支持"
    except Exception as error:  # noqa: BLE001 - a one-time protocol failure is evidence
        failure_code = "preflight-failed" if str(error) == "preflight" else "execution-failed"
        arms = []
        computed = {gate_id: False for gate_id in _gate_ids(artifacts["gate"])}
        status = "indeterminate"
        decision = "判定不能"
    gate_ids = _gate_ids(artifacts["gate"])
    payload = {
        "protocol_id": manifest["protocol_id"],
        "stage": stage,
        "status": status,
        "failure_code": failure_code,
        "snapshot": manifest["snapshot"],
        "protocol_hashes": manifest["artifact_sha256"],
        "preflight": preflight,
        "arms": arms,
        "gates": [{"gate_id": gate_id, "passed": computed[gate_id]} for gate_id in gate_ids],
        "all_hard_gates_pass": all(computed.values()),
        "interpretation_ja": _interpretation(decision),
    }
    verify_observed_payload(payload, manifest, artifacts)
    write_json_exclusive(output, payload)
    verify_observed_payload(read_json(output), manifest, artifacts)
    return output


def verify_registered_result(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError("stage must be development or holdout")
    manifest, artifacts = _load_protocol()
    verify_observed_payload(read_json(ROOT / str(manifest["outputs"][stage])), manifest, artifacts)


def preflight_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest, artifacts = _load_current_protocol()
    report = {
        stage: _preflight(snapshot, manifest, artifacts, stage)
        for stage in STAGES
        if stage == "development"
    }
    assert_public_payload(report)
    return report

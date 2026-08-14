from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_feedback import EngineConfig, NeuronGraphRAG
from .feedback import FeedbackLedger
from .models import SourceUseEvent


PROTOCOL_ID = "real-task-feedback-shadow-v1"
SCHEMA_VERSION = 1
ARM_IDS = ("used_q3_s1", "confirmed_r05_s1")
SOURCE_USE_STAGES = ("selected", "validated", "used")
POSITIVE_EVIDENCE_KINDS = {
    "test_passed",
    "citation_verified",
    "review_accepted",
}
NEGATIVE_EVIDENCE_KINDS = {"rollback_or_correction"}
OUTCOMES = {"pending", "confirmed", "corrected", "rolled_back"}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_canonical_json(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    raw = artifact.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON artifact: {artifact}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError(f"non-canonical JSON artifact: {artifact}")
    return value


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(payload)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_packet(packet: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_id",
        "packet_id",
        "slot",
        "supersedes_packet_id",
        "task",
        "database_snapshot",
        "retrieval",
        "source_use",
        "outcome",
        "efficiency",
        "captured_at",
    }
    _exact_keys(packet, required, "packet")
    if packet["schema_version"] != SCHEMA_VERSION or packet["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unknown packet protocol")
    _nonempty_string(packet["packet_id"], "packet_id")
    if isinstance(packet["slot"], bool) or not isinstance(packet["slot"], int) or packet["slot"] < 1:
        raise ValueError("slot must be a positive integer")
    if packet["supersedes_packet_id"] is not None:
        _nonempty_string(packet["supersedes_packet_id"], "supersedes_packet_id")

    task = _mapping(packet["task"], "task")
    _exact_keys(
        task,
        {"task_url", "repository", "base_commit", "close_condition", "eligible_at"},
        "task",
    )
    for field in task:
        _nonempty_string(task[field], f"task.{field}")

    snapshot = _mapping(packet["database_snapshot"], "database_snapshot")
    _exact_keys(snapshot, {"sha256"}, "database_snapshot")
    _sha256(snapshot["sha256"], "database_snapshot.sha256")

    retrieval = _mapping(packet["retrieval"], "retrieval")
    _exact_keys(
        retrieval,
        {"query", "limit", "candidates", "used_node_id", "credited_path"},
        "retrieval",
    )
    _nonempty_string(retrieval["query"], "retrieval.query")
    if isinstance(retrieval["limit"], bool) or not isinstance(retrieval["limit"], int) or retrieval["limit"] < 1:
        raise ValueError("retrieval.limit must be a positive integer")
    candidates = _sequence(retrieval["candidates"], "retrieval.candidates")
    if not candidates or len(candidates) > retrieval["limit"]:
        raise ValueError("retrieval.candidates must be non-empty and fit limit")
    candidate_ids: list[str] = []
    for index, raw_candidate in enumerate(candidates):
        candidate = _mapping(raw_candidate, f"candidate[{index}]")
        _exact_keys(candidate, {"node_id", "source_url", "content_sha256"}, "candidate")
        _nonempty_string(candidate["node_id"], "candidate.node_id")
        _nonempty_string(candidate["source_url"], "candidate.source_url")
        _sha256(candidate["content_sha256"], "candidate.content_sha256")
        candidate_ids.append(candidate["node_id"])
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate node IDs must be unique")
    _nonempty_string(retrieval["used_node_id"], "retrieval.used_node_id")
    if retrieval["used_node_id"] not in candidate_ids:
        raise ValueError("used node must be one of the captured candidates")
    _validate_path(retrieval["credited_path"], retrieval["used_node_id"])

    source_use = _sequence(packet["source_use"], "source_use")
    if len(source_use) != len(SOURCE_USE_STAGES):
        raise ValueError("source_use must contain exactly three stages")
    for index, raw_event in enumerate(source_use):
        event = _mapping(raw_event, f"source_use[{index}]")
        _exact_keys(event, {"stage", "node_id"}, "source_use event")
        if event["stage"] != SOURCE_USE_STAGES[index]:
            raise ValueError("source_use stages must be selected, validated, used")
        if event["node_id"] != retrieval["used_node_id"]:
            raise ValueError("source_use node must equal retrieval.used_node_id")

    outcome = _mapping(packet["outcome"], "outcome")
    _exact_keys(outcome, {"status", "summary", "external_ref", "evidence"}, "outcome")
    if outcome["status"] not in OUTCOMES:
        raise ValueError("unknown outcome status")
    _nonempty_string(outcome["summary"], "outcome.summary")
    if outcome["external_ref"] is not None:
        _nonempty_string(outcome["external_ref"], "outcome.external_ref")
    evidence = _sequence(outcome["evidence"], "outcome.evidence")
    kinds = {_validate_evidence(item, retrieval["used_node_id"]) for item in evidence}
    if outcome["status"] == "confirmed" and not kinds & POSITIVE_EVIDENCE_KINDS:
        raise ValueError("confirmed outcome requires positive objective evidence")
    if outcome["status"] in {"corrected", "rolled_back"} and not kinds & NEGATIVE_EVIDENCE_KINDS:
        raise ValueError("corrected or rolled_back outcome requires negative evidence")
    if outcome["status"] == "pending" and evidence:
        raise ValueError("pending outcome must not contain observed evidence")

    _validate_efficiency(packet["efficiency"])
    _nonempty_string(packet["captured_at"], "captured_at")


def capture_packet(packet: Mapping[str, Any], registry_dir: str | Path) -> Path:
    validate_packet(packet)
    registry = Path(registry_dir)
    with _registry_lock(registry):
        packets = _read_registry(registry)
        packet_id = str(packet["packet_id"])
        if any(item["packet_id"] == packet_id for item in packets):
            raise FileExistsError(f"packet ID already registered: {packet_id}")

        supersedes = packet["supersedes_packet_id"]
        if supersedes is None:
            expected_slot = max((int(item["slot"]) for item in packets), default=0) + 1
            if packet["slot"] != expected_slot:
                raise ValueError(f"new packet must use sequential slot {expected_slot}")
            if any(item["slot"] == packet["slot"] for item in packets):
                raise ValueError(f"slot is already registered: {packet['slot']}")
        else:
            by_id = {str(item["packet_id"]): item for item in packets}
            prior_packet = by_id.get(str(supersedes))
            if prior_packet is None:
                raise ValueError("superseded packet is not registered")
            if any(item["supersedes_packet_id"] == supersedes for item in packets):
                raise ValueError("superseded packet already has a successor")
            if packet["slot"] != prior_packet["slot"]:
                raise ValueError("superseding packet must retain its slot")
            for field in ("task", "database_snapshot", "retrieval", "source_use"):
                if packet[field] != prior_packet[field]:
                    raise ValueError(f"superseding packet changed immutable field: {field}")

        output = registry / f"{int(packet['slot']):04d}-{packet_id}.json"
        write_json_exclusive(output, packet)
        return output


def load_effective_registry(registry_dir: str | Path) -> list[dict[str, Any]]:
    registry = Path(registry_dir)
    with _registry_lock(registry):
        packets = _read_registry(registry)
    if not packets:
        raise ValueError("packet registry is empty")
    by_id = {str(packet["packet_id"]): packet for packet in packets}
    if len(by_id) != len(packets):
        raise ValueError("registry packet IDs must be unique")
    roots: dict[int, dict[str, Any]] = {}
    successor: dict[str, dict[str, Any]] = {}
    for packet in packets:
        supersedes = packet["supersedes_packet_id"]
        if supersedes is None:
            slot = int(packet["slot"])
            if slot in roots:
                raise ValueError(f"registry contains duplicate root slot: {slot}")
            roots[slot] = packet
        else:
            if supersedes not in by_id:
                raise ValueError("registry correction references an absent packet")
            if supersedes in successor:
                raise ValueError("registry correction chain branches")
            successor[str(supersedes)] = packet
    if sorted(roots) != list(range(1, len(roots) + 1)):
        raise ValueError("registry root slots must be sequential from one")
    effective = []
    visited: set[str] = set()
    for slot in sorted(roots):
        current = roots[slot]
        while str(current["packet_id"]) in successor:
            packet_id = str(current["packet_id"])
            if packet_id in visited:
                raise ValueError("registry correction chain contains a cycle")
            visited.add(packet_id)
            current = successor[packet_id]
            if current["slot"] != slot:
                raise ValueError("registry correction changed slot")
        effective.append(current)
    reachable = visited | {str(packet["packet_id"]) for packet in effective}
    if reachable != set(by_id):
        raise ValueError("registry contains an unreachable correction")
    return effective


def _read_registry(registry: Path) -> list[dict[str, Any]]:
    packets = [read_canonical_json(path) for path in sorted(registry.glob("*.json"))]
    for item in packets:
        validate_packet(item)
    return packets


@contextmanager
def _registry_lock(registry: Path) -> Any:
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / ".registry.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError("packet registry is locked by another writer") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink()


def verify_packet_against_snapshot(packet: Mapping[str, Any], snapshot_path: str | Path) -> None:
    validate_packet(packet)
    snapshot = Path(snapshot_path)
    live_sidecars = [
        path
        for path in (Path(str(snapshot) + "-wal"), Path(str(snapshot) + "-journal"))
        if path.exists() and path.stat().st_size
    ]
    if live_sidecars:
        raise ValueError("database snapshot has an uncheckpointed live sidecar")
    if sha256_file(snapshot) != packet["database_snapshot"]["sha256"]:
        raise ValueError("database snapshot hash mismatch")
    with tempfile.TemporaryDirectory() as directory:
        verification_clone = Path(directory) / "verify.db"
        shutil.copyfile(snapshot, verification_clone)
        with NeuronGraphRAG(verification_clone, config=_arm_config("confirmed_r05_s1")) as engine:
            nodes = {node.node_id: node for node in engine.store.list_nodes()}
            for candidate in packet["retrieval"]["candidates"]:
                node = nodes.get(candidate["node_id"])
                if node is None:
                    raise ValueError(f"captured node is absent: {candidate['node_id']}")
                if node.metadata.get("source_url") != candidate["source_url"]:
                    raise ValueError(f"source identity mismatch: {candidate['node_id']}")
                if sha256_text(node.text) != candidate["content_sha256"]:
                    raise ValueError(f"content hash mismatch: {candidate['node_id']}")
            trace = engine.search_channels(
                str(packet["retrieval"]["query"]),
                limit=int(packet["retrieval"]["limit"]),
                now=1_000.0,
            ).relation
            actual_ids = [hit.node.node_id for hit in trace.hits]
            expected_ids = [item["node_id"] for item in packet["retrieval"]["candidates"]]
            if actual_ids != expected_ids:
                raise ValueError("captured candidates do not match frozen snapshot retrieval")
            hit = next(item for item in trace.hits if item.node.node_id == packet["retrieval"]["used_node_id"])
            if _selected_path(hit) != packet["retrieval"]["credited_path"]:
                raise ValueError("credited path mismatch")


def replay_packet(packet: Mapping[str, Any], snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets([packet], snapshot_path)


def replay_registry(registry_dir: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets(load_effective_registry(registry_dir), snapshot_path)


def replay_packets(
    packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path
) -> dict[str, Any]:
    ordered_packets = list(packets)
    _validate_batch_inputs(ordered_packets, snapshot_path)
    source_hash_before = sha256_file(snapshot_path)
    arms = {}
    for arm_id in ARM_IDS:
        first = _run_arm_batch(ordered_packets, snapshot_path, arm_id)
        second = _run_arm_batch(ordered_packets, snapshot_path, arm_id)
        if first != second:
            raise RuntimeError(f"non-deterministic replay: {arm_id}")
        arms[arm_id] = first
    if sha256_file(snapshot_path) != source_hash_before:
        raise RuntimeError("source snapshot changed during replay")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "packet_ids": [packet["packet_id"] for packet in ordered_packets],
        "slots": [packet["slot"] for packet in ordered_packets],
        "snapshot_sha256": source_hash_before,
        "arms": arms,
        "comparison": _comparison(arms),
        "efficiency": [
            {"packet_id": packet["packet_id"], **packet["efficiency"]}
            for packet in ordered_packets
        ],
        "replay": {
            "fresh_clone_per_arm": True,
            "cumulative_slot_order": True,
            "repeated_semantic_replay": 2,
            "deterministic": True,
            "source_snapshot_unchanged": True,
        },
    }
    validate_result(result)
    return result


def verify_result_against_packets(
    result: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    snapshot_path: str | Path,
) -> None:
    validate_result(result)
    expected = replay_packets(packets, snapshot_path)
    if result != expected:
        raise ValueError("stored result does not match exact semantic replay")


def verify_result_against_registry(
    result: Mapping[str, Any], registry_dir: str | Path, snapshot_path: str | Path
) -> None:
    verify_result_against_packets(
        result, load_effective_registry(registry_dir), snapshot_path
    )


def _validate_batch_inputs(
    packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path
) -> None:
    if not packets:
        raise ValueError("replay requires at least one packet")
    for packet in packets:
        verify_packet_against_snapshot(packet, snapshot_path)
    slots = [int(packet["slot"]) for packet in packets]
    if slots != list(range(1, len(packets) + 1)):
        raise ValueError("batch packets must be in sequential slot order")
    packet_ids = [str(packet["packet_id"]) for packet in packets]
    if len(set(packet_ids)) != len(packet_ids):
        raise ValueError("batch packet IDs must be unique")
    snapshot_hashes = {
        str(packet["database_snapshot"]["sha256"]) for packet in packets
    }
    if len(snapshot_hashes) != 1 or snapshot_hashes != {sha256_file(snapshot_path)}:
        raise ValueError("batch packets must share the exact replay snapshot")


def validate_result(result: Mapping[str, Any]) -> None:
    _exact_keys(
        result,
        {"schema_version", "protocol_id", "packet_ids", "slots", "snapshot_sha256", "arms", "comparison", "efficiency", "replay"},
        "result",
    )
    if result["schema_version"] != SCHEMA_VERSION or result["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unknown result protocol")
    packet_ids = _sequence(result["packet_ids"], "result.packet_ids")
    slots = _sequence(result["slots"], "result.slots")
    if not packet_ids or any(not isinstance(item, str) or not item for item in packet_ids):
        raise ValueError("result.packet_ids must be non-empty strings")
    if len(set(packet_ids)) != len(packet_ids):
        raise ValueError("result.packet_ids must be unique")
    if slots != list(range(1, len(packet_ids) + 1)):
        raise ValueError("result.slots must be sequential from one")
    _sha256(result["snapshot_sha256"], "result.snapshot_sha256")
    arms = _mapping(result["arms"], "arms")
    if set(arms) != set(ARM_IDS):
        raise ValueError("result must contain exactly the two frozen arms")
    for arm_id in ARM_IDS:
        _validate_arm_result(arms[arm_id], arm_id)
        arm_packets = arms[arm_id]["packets"]
        if [packet["packet_id"] for packet in arm_packets] != packet_ids:
            raise ValueError(f"arm packet ID order mismatch: {arm_id}")
        if [packet["slot"] for packet in arm_packets] != slots:
            raise ValueError(f"arm slot order mismatch: {arm_id}")
    comparison = _mapping(result["comparison"], "comparison")
    if comparison != _comparison(arms):
        raise ValueError("result comparison is not recomputable from arm metrics")
    efficiency = _sequence(result["efficiency"], "result.efficiency")
    if len(efficiency) != len(packet_ids):
        raise ValueError("result efficiency must have one row per packet")
    for packet_id, raw_row in zip(packet_ids, efficiency, strict=True):
        row = _mapping(raw_row, "result efficiency row")
        _exact_keys(
            row,
            {"packet_id", "tool_calls", "research_count", "elapsed_seconds", "token_count"},
            "result efficiency row",
        )
        if row["packet_id"] != packet_id:
            raise ValueError("result efficiency packet order mismatch")
        _validate_efficiency({key: value for key, value in row.items() if key != "packet_id"})
    replay = _mapping(result["replay"], "replay")
    if replay != {
        "fresh_clone_per_arm": True,
        "cumulative_slot_order": True,
        "repeated_semantic_replay": 2,
        "deterministic": True,
        "source_snapshot_unchanged": True,
    }:
        raise ValueError("result replay proof is incomplete")


def _validate_arm_result(value: Any, arm_id: str) -> None:
    arm = _mapping(value, f"arm.{arm_id}")
    _exact_keys(
        arm,
        {
            "arm_id",
            "policy",
            "config",
            "packets",
            "final_edge_state",
        },
        f"arm.{arm_id}",
    )
    if arm["arm_id"] != arm_id or arm["policy"] != ("used" if arm_id == "used_q3_s1" else "confirmed"):
        raise ValueError(f"arm identity mismatch: {arm_id}")
    expected_config = _arm_result_config(arm_id)
    if arm["config"] != expected_config:
        raise ValueError(f"arm config mismatch: {arm_id}")
    packets = _sequence(arm["packets"], f"arm.{arm_id}.packets")
    if not packets:
        raise ValueError(f"arm packet replay is empty: {arm_id}")
    for packet in packets:
        _validate_packet_replay(packet, arm_id)
    final_edge_state = _validate_edge_state(
        arm["final_edge_state"], f"arm.{arm_id}.final_edge_state"
    )
    if final_edge_state != packets[-1]["edge_state_after"]:
        raise ValueError(f"arm final edge state mismatch: {arm_id}")


def _validate_packet_replay(value: Any, arm_id: str) -> None:
    packet = _mapping(value, f"arm.{arm_id}.packet")
    _exact_keys(
        packet,
        {
            "packet_id",
            "slot",
            "before",
            "after",
            "rank_delta",
            "score_delta",
            "edge_state_before",
            "edge_state_after",
            "edge_delta",
            "non_target_churn",
            "source_use",
            "outcome",
            "idempotency_replay",
        },
        f"arm.{arm_id}.packet",
    )
    _nonempty_string(packet["packet_id"], "packet replay ID")
    if isinstance(packet["slot"], bool) or not isinstance(packet["slot"], int) or packet["slot"] < 1:
        raise ValueError("packet replay slot must be positive")
    before = _validate_ranking(packet["before"], f"arm.{arm_id}.before")
    after = _validate_ranking(packet["after"], f"arm.{arm_id}.after")
    if packet["rank_delta"] != after["rank"] - before["rank"]:
        raise ValueError(f"rank delta mismatch: {arm_id}")
    if not _same_number(packet["score_delta"], after["score"] - before["score"]):
        raise ValueError(f"score delta mismatch: {arm_id}")
    before_edges = _validate_edge_state(packet["edge_state_before"], "edge_state_before")
    after_edges = _validate_edge_state(packet["edge_state_after"], "edge_state_after")
    if packet["edge_delta"] != _edge_delta(before_edges, after_edges):
        raise ValueError(f"edge delta mismatch: {arm_id}")
    churn = _mapping(packet["non_target_churn"], "non_target_churn")
    _exact_keys(churn, {"before", "after", "changed"}, "non_target_churn")
    if not isinstance(churn["before"], list) or not isinstance(churn["after"], list):
        raise ValueError("non-target churn orders must be arrays")
    if churn["changed"] is not (churn["before"] != churn["after"]):
        raise ValueError("non-target churn flag mismatch")
    source_use = _mapping(packet["source_use"], "source_use result")
    _exact_keys(source_use, {"events", "newly_used_node_ids", "feedback"}, "source_use result")
    if packet["outcome"] is not None:
        outcome = _mapping(packet["outcome"], "outcome result")
        _exact_keys(
            outcome,
            {"outcome", "reinforcement_applied", "confirmations", "credited_paths", "normalized"},
            "outcome result",
        )
    if packet["idempotency_replay"] is not True:
        raise ValueError(f"idempotency replay is not proven: {arm_id}")


def _validate_ranking(value: Any, name: str) -> Mapping[str, Any]:
    ranking = _mapping(value, name)
    _exact_keys(ranking, {"rank", "score"}, name)
    if isinstance(ranking["rank"], bool) or not isinstance(ranking["rank"], int) or ranking["rank"] < 1:
        raise ValueError(f"{name}.rank must be a positive integer")
    if isinstance(ranking["score"], bool) or not isinstance(ranking["score"], (int, float)) or not math.isfinite(ranking["score"]):
        raise ValueError(f"{name}.score must be finite")
    return ranking


def _validate_edge_state(value: Any, name: str) -> list[Mapping[str, Any]]:
    rows = _sequence(value, name)
    validated = []
    identities = []
    for row in rows:
        edge = _mapping(row, name)
        _exact_keys(
            edge,
            {"source_id", "target_id", "edge_type", "weight", "reinforced_count", "evidence_count", "confirmation_count"},
            name,
        )
        identity = tuple(edge[field] for field in ("source_id", "target_id", "edge_type"))
        identities.append(identity)
        for field in ("weight", "reinforced_count", "evidence_count", "confirmation_count"):
            value_field = edge[field]
            if isinstance(value_field, bool) or not isinstance(value_field, (int, float)) or value_field < 0:
                raise ValueError(f"{name}.{field} must be non-negative")
        validated.append(edge)
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError(f"{name} edge identities must be ordered and unique")
    return validated


def _validate_efficiency(value: Any) -> None:
    efficiency = _mapping(value, "efficiency")
    _exact_keys(
        efficiency,
        {"tool_calls", "research_count", "elapsed_seconds", "token_count"},
        "efficiency",
    )
    for field in ("tool_calls", "research_count", "token_count"):
        count = efficiency[field]
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError(f"efficiency.{field} must be null or a non-negative integer")
    elapsed = efficiency["elapsed_seconds"]
    if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("efficiency.elapsed_seconds must be null or non-negative finite")


def _same_number(left: Any, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isfinite(left)
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    )


def build_placeholder_packet(fixture: Mapping[str, Any], snapshot_path: str | Path) -> dict[str, Any]:
    if fixture.get("placeholder_only") is not True:
        raise ValueError("placeholder fixture must declare placeholder_only")
    for node in fixture["documents"]:
        if not str(node["node_id"]).startswith("placeholder-") or "example.invalid" not in str(node["source_url"]):
            raise ValueError("fixture contains a non-placeholder source identity")
    with NeuronGraphRAG(snapshot_path, config=_arm_config("confirmed_r05_s1")) as engine:
        trace = engine.search_channels(
            fixture["packet_seed"]["query"],
            limit=fixture["packet_seed"]["limit"],
            now=1_000.0,
        ).relation
        used = fixture["packet_seed"]["used_node_id"]
        hit = next(item for item in trace.hits if item.node.node_id == used)
        candidates = [
            {
                "node_id": item.node.node_id,
                "source_url": item.node.metadata["source_url"],
                "content_sha256": sha256_text(item.node.text),
            }
            for item in trace.hits
        ]
    seed = fixture["packet_seed"]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "packet_id": seed["packet_id"],
        "slot": 1,
        "supersedes_packet_id": None,
        "task": seed["task"],
        "database_snapshot": {"sha256": sha256_file(snapshot_path)},
        "retrieval": {
            "query": seed["query"],
            "limit": seed["limit"],
            "candidates": candidates,
            "used_node_id": used,
            "credited_path": _selected_path(hit),
        },
        "source_use": [{"stage": stage, "node_id": used} for stage in SOURCE_USE_STAGES],
        "outcome": seed["outcome"],
        "efficiency": seed["efficiency"],
        "captured_at": seed["captured_at"],
    }
    validate_packet(packet)
    return packet


def create_placeholder_snapshot(fixture: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    with NeuronGraphRAG(path, config=_arm_config("confirmed_r05_s1")) as engine:
        for node in fixture["documents"]:
            engine.add_document(
                node["node_id"],
                node["text"],
                metadata={"source_url": node["source_url"]},
            )
        for edge in fixture["edges"]:
            engine.add_edge(
                edge["source_id"], edge["target_id"], edge["edge_type"], weight=edge["weight"]
            )


def probe_placeholder(fixture_path: str | Path) -> dict[str, Any]:
    fixture = read_canonical_json(fixture_path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snapshot = root / "placeholder.db"
        create_placeholder_snapshot(fixture, snapshot)
        packet = build_placeholder_packet(fixture, snapshot)
        registered = capture_packet(packet, root / "registry")
        captured = read_canonical_json(registered)
        result = replay_packet(captured, snapshot)
        result_path = root / "placeholder.result.json"
        write_json_exclusive(result_path, result)
        verify_result_against_packets(
            read_canonical_json(result_path), [captured], snapshot
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "placeholder_only": True,
            "packet_round_trip": True,
            "replay_round_trip": True,
            "exclusive_writer_verified": True,
        }


def _run_arm_batch(
    packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path, arm_id: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory) / "shadow.db"
        shutil.copyfile(snapshot_path, clone)
        with NeuronGraphRAG(clone, config=_arm_config(arm_id)) as engine:
            return {
                "arm_id": arm_id,
                "policy": "used" if arm_id == "used_q3_s1" else "confirmed",
                "config": _arm_result_config(arm_id),
                "packets": [
                    _run_packet_on_engine(engine, packet, arm_id, index)
                    for index, packet in enumerate(packets, start=1)
                ],
                "final_edge_state": _edge_state(engine),
            }


def _run_packet_on_engine(
    engine: NeuronGraphRAG,
    packet: Mapping[str, Any],
    arm_id: str,
    index: int,
) -> dict[str, Any]:
    clock = 1_000.0 + index * 10.0
    before_edges = _edge_state(engine)
    trace = engine.search_channels(
        packet["retrieval"]["query"],
        limit=packet["retrieval"]["limit"],
        now=clock,
    ).relation
    used_node_id = packet["retrieval"]["used_node_id"]
    if used_node_id not in {hit.node.node_id for hit in trace.hits}:
        raise ValueError(
            f"used node is absent from cumulative replay trace at slot {packet['slot']}"
        )
    ledger = FeedbackLedger(engine)
    events = tuple(SourceUseEvent(used_node_id, stage) for stage in SOURCE_USE_STAGES)
    key = f"shadow:{packet['packet_id']}:{arm_id}:use"
    receipt = ledger.record_source_use(
        trace.trace_id, events, idempotency_key=key, now=clock + 1.0
    )
    repeated = ledger.record_source_use(
        trace.trace_id, events, idempotency_key=key, now=clock + 9.0
    )
    if _source_use_semantics(receipt) != _source_use_semantics(repeated):
        raise RuntimeError("source-use idempotency replay mismatch")

    outcome_semantics = None
    status = packet["outcome"]["status"]
    if status != "pending":
        outcome_key = f"shadow:{packet['packet_id']}:{arm_id}:outcome"
        outcome = ledger.record_outcome(
            trace.trace_id,
            [used_node_id],
            status,
            packet["outcome"]["summary"],
            idempotency_key=outcome_key,
            external_ref=packet["outcome"]["external_ref"],
            now=clock + 2.0,
        )
        repeated_outcome = ledger.record_outcome(
            trace.trace_id,
            [used_node_id],
            status,
            packet["outcome"]["summary"],
            idempotency_key=outcome_key,
            external_ref=packet["outcome"]["external_ref"],
            now=clock + 9.0,
        )
        outcome_semantics = _outcome_semantics(outcome)
        if outcome_semantics != _outcome_semantics(repeated_outcome):
            raise RuntimeError("outcome idempotency replay mismatch")
    after_edges = _edge_state(engine)
    post = engine.search_channels(
        packet["retrieval"]["query"],
        limit=packet["retrieval"]["limit"],
        now=clock + 3.0,
    ).relation
    before_ranking = _ranking(trace, used_node_id)
    after_ranking = _ranking(post, used_node_id)
    return {
        "packet_id": packet["packet_id"],
        "slot": packet["slot"],
        "before": before_ranking,
        "after": after_ranking,
        "rank_delta": after_ranking["rank"] - before_ranking["rank"],
        "score_delta": after_ranking["score"] - before_ranking["score"],
        "edge_state_before": before_edges,
        "edge_state_after": after_edges,
        "edge_delta": _edge_delta(before_edges, after_edges),
        "non_target_churn": _non_target_churn(trace, post, used_node_id),
        "source_use": _source_use_semantics(receipt),
        "outcome": outcome_semantics,
        "idempotency_replay": True,
    }


def _arm_result_config(arm_id: str) -> dict[str, Any]:
    return {
        "relation_feedback_evidence_quorum": 3 if arm_id == "used_q3_s1" else 1,
        "confirmed_outcome_reinforcement": arm_id == "confirmed_r05_s1",
        "confirmation_decay_ratio": 0.5 if arm_id == "confirmed_r05_s1" else None,
        "sibling_feedback_normalization": 1.0,
    }


def _arm_config(arm_id: str) -> EngineConfig:
    common = {
        "sparse_weight": 1.0,
        "dense_weight": 0.0,
        "seed_count": 1,
        "max_hops": 2,
        "hop_decay": 0.7,
        "feedback_learning_rate": 0.2,
        "maximum_edge_weight": 3.0,
        "sibling_feedback_normalization": 1.0,
    }
    if arm_id == "used_q3_s1":
        return EngineConfig(**common, relation_feedback_evidence_quorum=3)
    if arm_id == "confirmed_r05_s1":
        return EngineConfig(
            **common,
            relation_feedback_evidence_quorum=1,
            confirmed_outcome_reinforcement=True,
            confirmation_decay_ratio=0.5,
        )
    raise ValueError(f"unknown arm: {arm_id}")


def _selected_path(hit: Any) -> dict[str, Any]:
    paths = [path for path in hit.paths if path.steps]
    if not paths:
        raise ValueError("used relation node has no credited path")
    selected = max(paths, key=lambda path: (path.contribution, path.seed_id))
    return {
        "node_id": hit.node.node_id,
        "seed_id": selected.seed_id,
        "steps": [
            {
                "source_id": step.source_id,
                "target_id": step.target_id,
                "edge_type": step.edge_type,
            }
            for step in selected.steps
        ],
    }


def _validate_path(value: Any, used_node_id: str) -> None:
    path = _mapping(value, "credited_path")
    _exact_keys(path, {"node_id", "seed_id", "steps"}, "credited_path")
    if path["node_id"] != used_node_id:
        raise ValueError("credited path node must equal used node")
    _nonempty_string(path["seed_id"], "credited_path.seed_id")
    steps = _sequence(path["steps"], "credited_path.steps")
    if not steps:
        raise ValueError("credited path must contain an edge")
    for raw_step in steps:
        step = _mapping(raw_step, "credited path step")
        _exact_keys(step, {"source_id", "target_id", "edge_type"}, "credited path step")
        for field in step:
            _nonempty_string(step[field], f"credited_path.{field}")
    if steps[-1]["target_id"] != used_node_id:
        raise ValueError("credited path must terminate at used node")


def _validate_evidence(value: Any, used_node_id: str) -> str:
    evidence = _mapping(value, "evidence")
    required = {"kind", "node_id", "external_ref", "target_commit", "details"}
    _exact_keys(evidence, required, "evidence")
    kind = evidence["kind"]
    if kind not in POSITIVE_EVIDENCE_KINDS | NEGATIVE_EVIDENCE_KINDS:
        raise ValueError("unsupported evidence kind")
    if evidence["node_id"] != used_node_id:
        raise ValueError("evidence node must equal used node")
    _nonempty_string(evidence["external_ref"], "evidence.external_ref")
    _nonempty_string(evidence["target_commit"], "evidence.target_commit")
    details = _mapping(evidence["details"], "evidence.details")
    if kind == "test_passed":
        if details.get("exit_code") != 0 or not details.get("command"):
            raise ValueError("test_passed requires a command and exit_code 0")
    elif kind == "citation_verified":
        if details.get("literal_match") is not True or not details.get("claim"):
            raise ValueError("citation_verified requires a claim and literal_match true")
    elif kind == "review_accepted":
        if details.get("accepted") is not True:
            raise ValueError("review_accepted requires accepted true")
    elif kind == "rollback_or_correction":
        if not details.get("reason"):
            raise ValueError("rollback_or_correction requires a reason")
    return str(kind)


def _edge_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    rows = []
    for edge in engine.store.list_edges():
        confirmation = engine.store.connection.execute(
            "SELECT confirmation_count FROM confirmed_edge_state WHERE source_id=? AND target_id=? AND edge_type=?",
            (edge.source_id, edge.target_id, edge.edge_type),
        ).fetchone()
        rows.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "reinforced_count": edge.reinforced_count,
                "evidence_count": engine.store.feedback_evidence_count(edge.source_id, edge.target_id, edge.edge_type),
                "confirmation_count": 0 if confirmation is None else int(confirmation["confirmation_count"]),
            }
        )
    return rows


def _edge_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for left, right in zip(before, after, strict=True):
        if tuple(left[field] for field in ("source_id", "target_id", "edge_type")) != tuple(right[field] for field in ("source_id", "target_id", "edge_type")):
            raise RuntimeError("edge identity changed")
        result.append(
            {
                "source_id": left["source_id"],
                "target_id": left["target_id"],
                "edge_type": left["edge_type"],
                "weight_delta": right["weight"] - left["weight"],
                "reinforced_count_delta": right["reinforced_count"] - left["reinforced_count"],
                "evidence_count_delta": right["evidence_count"] - left["evidence_count"],
                "confirmation_count_delta": right["confirmation_count"] - left["confirmation_count"],
            }
        )
    return result


def _ranking(trace: Any, node_id: str) -> dict[str, Any]:
    for hit in trace.hits:
        if hit.node.node_id == node_id:
            return {"rank": hit.rank, "score": hit.graph_activation}
    return {"rank": len(trace.hits) + 1, "score": 0.0}


def _non_target_churn(before: Any, after: Any, target: str) -> dict[str, Any]:
    before_ids = [hit.node.node_id for hit in before.hits if hit.node.node_id != target]
    after_ids = [hit.node.node_id for hit in after.hits if hit.node.node_id != target]
    return {"before": before_ids, "after": after_ids, "changed": before_ids != after_ids}


def _source_use_semantics(receipt: Any) -> dict[str, Any]:
    return {
        "events": [asdict(event) for event in receipt.events],
        "newly_used_node_ids": list(receipt.newly_used_node_ids),
        "feedback": None if receipt.feedback is None else {
            "reinforced": [asdict(item) for item in receipt.feedback.reinforced_edges],
            "normalized": [asdict(item) for item in receipt.feedback.normalized_sibling_edges],
            "evidence": [asdict(item) for item in receipt.feedback.evidence],
        },
    }


def _outcome_semantics(receipt: Any) -> dict[str, Any]:
    return {
        "outcome": receipt.outcome,
        "reinforcement_applied": receipt.reinforcement_applied,
        "confirmations": [asdict(item) for item in receipt.confirmations],
        "credited_paths": [asdict(item) for item in receipt.credited_paths],
        "normalized": [asdict(item) for item in receipt.normalized_sibling_edges],
    }


def _comparison(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    used = arms["used_q3_s1"]
    confirmed = arms["confirmed_r05_s1"]
    used_packets = used["packets"]
    confirmed_packets = confirmed["packets"]
    return {
        "rank_delta_confirmed_minus_used": [
            {
                "packet_id": left["packet_id"],
                "slot": left["slot"],
                "value": right["rank_delta"] - left["rank_delta"],
            }
            for left, right in zip(used_packets, confirmed_packets, strict=True)
        ],
        "score_delta_confirmed_minus_used": [
            {
                "packet_id": left["packet_id"],
                "slot": left["slot"],
                "value": right["score_delta"] - left["score_delta"],
            }
            for left, right in zip(used_packets, confirmed_packets, strict=True)
        ],
        "final_changed_edge_count": {
            arm_id: sum(
                not math.isclose(
                    after["weight"] - before["weight"], 0.0, abs_tol=1e-12
                )
                for before, after in zip(
                    arm["packets"][0]["edge_state_before"],
                    arm["final_edge_state"],
                    strict=True,
                )
            )
            for arm_id, arm in arms.items()
        },
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{name} fields do not match the frozen schema")


def _nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a sha256 digest") from error

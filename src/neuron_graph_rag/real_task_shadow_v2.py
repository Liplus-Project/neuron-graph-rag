from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import real_task_shadow as v1
from .config_provenance import (
    FEEDBACK_CONFIG_FIELDS as FEEDBACK_FIELDS,
    SEARCH_SURFACES,
    config_fingerprint as _fingerprint,
    effective_config,
    effective_config_provenance,
    effective_search_surface,
    search_with_surface,
)
from .evidence_feedback import EngineConfig, NeuronGraphRAG
from .feedback import FeedbackLedger
from .models import SourceUseEvent


PROTOCOL_ID = "real-task-feedback-shadow-v2"
SCHEMA_VERSION = 2
ARM_IDS = v1.ARM_IDS
SOURCE_USE_STAGES = v1.SOURCE_USE_STAGES
ARM_OVERRIDES = {
    "used_q3_s1": {
        "relation_feedback_evidence_quorum": 3,
        "confirmed_outcome_reinforcement": False,
        "confirmation_decay_ratio": None,
        "sibling_feedback_normalization": 1.0,
    },
    "confirmed_r05_s1": {
        "relation_feedback_evidence_quorum": 1,
        "confirmed_outcome_reinforcement": True,
        "confirmation_decay_ratio": 0.5,
        "sibling_feedback_normalization": 1.0,
    },
}
def _capture_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query": packet["retrieval"]["query"],
        "limit": packet["retrieval"]["limit"],
        "candidates": packet["retrieval"]["candidates"],
        "credited_path": packet["retrieval"]["credited_path"],
        "searched_at": packet["capture"]["searched_at"],
        "search_surface": packet["capture"]["search_surface"],
        "effective_config": packet["capture"]["effective_config"],
        "retrieval_config_fingerprint": packet["capture"]["retrieval_config_fingerprint"],
        "feedback_config_fingerprint": packet["capture"]["feedback_config_fingerprint"],
        "full_config_fingerprint": packet["capture"]["full_config_fingerprint"],
    }


def bind_capture_fingerprint(packet: Mapping[str, Any]) -> str:
    return _fingerprint(_capture_payload(packet))


def _as_v1_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    converted = {key: value for key, value in packet.items() if key != "capture"}
    converted["schema_version"] = v1.SCHEMA_VERSION
    converted["protocol_id"] = v1.PROTOCOL_ID
    return converted


def validate_packet(packet: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "protocol_id", "packet_id", "slot",
        "supersedes_packet_id", "task", "database_snapshot", "retrieval",
        "source_use", "outcome", "efficiency", "captured_at", "capture",
    }
    v1._exact_keys(packet, required, "packet")
    if packet["schema_version"] != SCHEMA_VERSION or packet["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unknown packet protocol")
    v1.validate_packet(_as_v1_packet(packet))
    capture = v1._mapping(packet["capture"], "capture")
    v1._exact_keys(
        capture,
        {
            "searched_at", "search_surface", "effective_config",
            "retrieval_config_fingerprint",
            "feedback_config_fingerprint", "full_config_fingerprint",
            "capture_fingerprint",
        },
        "capture",
    )
    searched_at = capture["searched_at"]
    if isinstance(searched_at, bool) or not isinstance(searched_at, (int, float)) or not math.isfinite(searched_at) or searched_at <= 0:
        raise ValueError("capture.searched_at must be a positive finite Unix timestamp")
    if capture["search_surface"] not in SEARCH_SURFACES:
        raise ValueError("capture.search_surface is unknown")
    config = v1._mapping(capture["effective_config"], "capture.effective_config")
    v1._exact_keys(config, {"retrieval", "feedback"}, "capture.effective_config")
    retrieval = v1._mapping(config["retrieval"], "capture.effective_config.retrieval")
    feedback = v1._mapping(config["feedback"], "capture.effective_config.feedback")
    engine_config = EngineConfig(**{**retrieval, **feedback})
    expected = effective_config(engine_config)
    if config != expected:
        raise ValueError("capture effective config is not canonical or complete")
    if capture["search_surface"] != effective_search_surface(engine_config):
        raise ValueError("capture search surface does not match effective config")
    for key, value in (
        ("retrieval_config_fingerprint", retrieval),
        ("feedback_config_fingerprint", feedback),
        ("full_config_fingerprint", config),
    ):
        v1._sha256(capture[key], f"capture.{key}")
        if capture[key] != _fingerprint(value):
            raise ValueError(f"capture {key} mismatch")
    v1._sha256(capture["capture_fingerprint"], "capture.capture_fingerprint")
    if capture["capture_fingerprint"] != bind_capture_fingerprint(packet):
        raise ValueError("capture fingerprint mismatch")


def _read_registry(registry: Path) -> list[dict[str, Any]]:
    packets = [v1.read_canonical_json(path) for path in sorted(registry.glob("*.json"))]
    for packet in packets:
        validate_packet(packet)
    return packets


def capture_packet(packet: Mapping[str, Any], registry_dir: str | Path) -> Path:
    validate_packet(packet)
    registry = Path(registry_dir)
    with v1._registry_lock(registry):
        packets = _read_registry(registry)
        packet_id = str(packet["packet_id"])
        if any(item["packet_id"] == packet_id for item in packets):
            raise FileExistsError(f"packet ID already registered: {packet_id}")
        supersedes = packet["supersedes_packet_id"]
        if supersedes is None:
            expected_slot = max((int(item["slot"]) for item in packets), default=0) + 1
            if packet["slot"] != expected_slot:
                raise ValueError(f"new packet must use sequential slot {expected_slot}")
        else:
            prior = next((item for item in packets if item["packet_id"] == supersedes), None)
            if prior is None:
                raise ValueError("superseded packet is not registered")
            if any(item["supersedes_packet_id"] == supersedes for item in packets):
                raise ValueError("superseded packet already has a successor")
            if packet["slot"] != prior["slot"]:
                raise ValueError("superseding packet must retain its slot")
            for field in ("task", "database_snapshot", "retrieval", "source_use", "capture"):
                if packet[field] != prior[field]:
                    raise ValueError(f"superseding packet changed immutable field: {field}")
        output = registry / f"{int(packet['slot']):04d}-{packet_id}.json"
        v1.write_json_exclusive(output, packet)
        return output


def load_effective_registry(registry_dir: str | Path) -> list[dict[str, Any]]:
    registry = Path(registry_dir)
    with v1._registry_lock(registry):
        packets = _read_registry(registry)
    if not packets:
        raise ValueError("packet registry is empty")
    by_id = {str(packet["packet_id"]): packet for packet in packets}
    if len(by_id) != len(packets):
        raise ValueError("registry packet IDs must be unique")
    roots: dict[int, dict[str, Any]] = {}
    successors: dict[str, dict[str, Any]] = {}
    for packet in packets:
        prior = packet["supersedes_packet_id"]
        if prior is None:
            slot = int(packet["slot"])
            if slot in roots:
                raise ValueError("registry contains duplicate root slot")
            roots[slot] = packet
        else:
            if prior not in by_id or prior in successors:
                raise ValueError("invalid registry correction chain")
            successors[str(prior)] = packet
    if sorted(roots) != list(range(1, len(roots) + 1)):
        raise ValueError("registry root slots must be sequential from one")
    result = []
    reachable: set[str] = set()
    for slot in sorted(roots):
        current = roots[slot]
        while str(current["packet_id"]) in successors:
            reachable.add(str(current["packet_id"]))
            current = successors[str(current["packet_id"])]
            if current["slot"] != slot:
                raise ValueError("registry correction changed slot")
        reachable.add(str(current["packet_id"]))
        result.append(current)
    if reachable != set(by_id):
        raise ValueError("registry contains an unreachable correction")
    return result


def _engine_config(packet: Mapping[str, Any], arm_id: str) -> EngineConfig:
    if arm_id not in ARM_OVERRIDES:
        raise ValueError(f"unknown arm: {arm_id}")
    captured = packet["capture"]["effective_config"]
    values = {**captured["retrieval"], **captured["feedback"]}
    values.update(ARM_OVERRIDES[arm_id])
    return EngineConfig(**values)


def _capture_engine_config(packet: Mapping[str, Any]) -> EngineConfig:
    captured = packet["capture"]["effective_config"]
    return EngineConfig(**{**captured["retrieval"], **captured["feedback"]})


def _arm_provenance(packet: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    provenance = effective_config_provenance(_engine_config(packet, arm_id))
    provenance["search_surface"] = packet["capture"]["search_surface"]
    provenance["overridden_feedback_fields"] = sorted(ARM_OVERRIDES[arm_id])
    return provenance


def verify_packet_against_snapshot(packet: Mapping[str, Any], snapshot_path: str | Path) -> None:
    validate_packet(packet)
    snapshot = Path(snapshot_path)
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(snapshot) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("database snapshot has an uncheckpointed live sidecar")
    if v1.sha256_file(snapshot) != packet["database_snapshot"]["sha256"]:
        raise ValueError("database snapshot hash mismatch")
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory) / "verify.db"
        shutil.copyfile(snapshot, clone)
        with NeuronGraphRAG(clone, config=_capture_engine_config(packet)) as engine:
            nodes = {node.node_id: node for node in engine.store.list_nodes()}
            for candidate in packet["retrieval"]["candidates"]:
                node = nodes.get(candidate["node_id"])
                if node is None or node.metadata.get("source_url") != candidate["source_url"]:
                    raise ValueError("captured source identity mismatch")
                if v1.sha256_text(node.text) != candidate["content_sha256"]:
                    raise ValueError("captured source content hash mismatch")
            trace = search_with_surface(
                engine,
                str(packet["retrieval"]["query"]),
                limit=int(packet["retrieval"]["limit"]),
                search_surface=str(packet["capture"]["search_surface"]),
                now=float(packet["capture"]["searched_at"]),
            )
            expected_ids = [item["node_id"] for item in packet["retrieval"]["candidates"]]
            if [hit.node.node_id for hit in trace.hits] != expected_ids:
                raise ValueError("captured candidates do not match exact runtime replay")
            hit = next(item for item in trace.hits if item.node.node_id == packet["retrieval"]["used_node_id"])
            if v1._selected_path(hit) != packet["retrieval"]["credited_path"]:
                raise ValueError("credited path mismatch")


def _validate_batch(packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path) -> None:
    if not packets:
        raise ValueError("replay requires at least one packet")
    for packet in packets:
        verify_packet_against_snapshot(packet, snapshot_path)
    if [packet["slot"] for packet in packets] != list(range(1, len(packets) + 1)):
        raise ValueError("batch packets must be in sequential slot order")
    if len({packet["packet_id"] for packet in packets}) != len(packets):
        raise ValueError("batch packet IDs must be unique")
    if len({packet["database_snapshot"]["sha256"] for packet in packets}) != 1:
        raise ValueError("batch packets must share the exact replay snapshot")
    capture_configs = [packet["capture"]["effective_config"] for packet in packets]
    if any(config != capture_configs[0] for config in capture_configs[1:]):
        raise ValueError("batch packets must share one effective capture config")
    search_surfaces = {
        packet["capture"]["search_surface"]
        for packet in packets
    }
    if len(search_surfaces) != 1:
        raise ValueError("batch packets must share one capture search surface")
    times = [float(packet["capture"]["searched_at"]) for packet in packets]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("batch capture timestamps must increase in slot order")


def _surface_ranking(trace: Any, node_id: str) -> dict[str, Any]:
    for position, hit in enumerate(trace.hits, start=1):
        if hit.node.node_id == node_id:
            score = (
                hit.channel_score
                if hasattr(hit, "channel_score")
                else hit.final_score
            )
            return {"rank": getattr(hit, "rank", position), "score": score}
    return {"rank": len(trace.hits) + 1, "score": 0.0}


def _run_packet(engine: NeuronGraphRAG, packet: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    clock = float(packet["capture"]["searched_at"])
    before_edges = v1._edge_state(engine)
    trace = search_with_surface(
        engine,
        packet["retrieval"]["query"],
        limit=packet["retrieval"]["limit"],
        search_surface=packet["capture"]["search_surface"],
        now=clock,
    )
    expected_ids = [item["node_id"] for item in packet["retrieval"]["candidates"]]
    if [hit.node.node_id for hit in trace.hits] != expected_ids:
        raise ValueError("cumulative replay candidates diverged from capture")
    used = packet["retrieval"]["used_node_id"]
    hit = next(item for item in trace.hits if item.node.node_id == used)
    if v1._selected_path(hit) != packet["retrieval"]["credited_path"]:
        raise ValueError("cumulative replay credited path diverged from capture")
    ledger = FeedbackLedger(engine)
    events = tuple(SourceUseEvent(used, stage) for stage in SOURCE_USE_STAGES)
    key = f"shadow-v2:{packet['packet_id']}:{arm_id}:use"
    receipt = ledger.record_source_use(trace.trace_id, events, idempotency_key=key, now=clock + 0.1)
    repeat = ledger.record_source_use(trace.trace_id, events, idempotency_key=key, now=clock + 0.9)
    if v1._source_use_semantics(receipt) != v1._source_use_semantics(repeat):
        raise RuntimeError("source-use idempotency replay mismatch")
    outcome_value = None
    status = packet["outcome"]["status"]
    if status != "pending":
        outcome_key = f"shadow-v2:{packet['packet_id']}:{arm_id}:outcome"
        outcome = ledger.record_outcome(
            trace.trace_id, [used], status, packet["outcome"]["summary"],
            idempotency_key=outcome_key, external_ref=packet["outcome"]["external_ref"], now=clock + 0.2,
        )
        repeat_outcome = ledger.record_outcome(
            trace.trace_id, [used], status, packet["outcome"]["summary"],
            idempotency_key=outcome_key, external_ref=packet["outcome"]["external_ref"], now=clock + 0.9,
        )
        outcome_value = v1._outcome_semantics(outcome)
        if outcome_value != v1._outcome_semantics(repeat_outcome):
            raise RuntimeError("outcome idempotency replay mismatch")
    after_edges = v1._edge_state(engine)
    post = search_with_surface(
        engine,
        packet["retrieval"]["query"],
        limit=packet["retrieval"]["limit"],
        search_surface=packet["capture"]["search_surface"],
        now=clock + 0.3,
    )
    before_rank = _surface_ranking(trace, used)
    after_rank = _surface_ranking(post, used)
    return {
        "packet_id": packet["packet_id"], "slot": packet["slot"],
        "before": before_rank, "after": after_rank,
        "rank_delta": after_rank["rank"] - before_rank["rank"],
        "score_delta": after_rank["score"] - before_rank["score"],
        "edge_state_before": before_edges, "edge_state_after": after_edges,
        "edge_delta": v1._edge_delta(before_edges, after_edges),
        "non_target_churn": v1._non_target_churn(trace, post, used),
        "source_use": v1._source_use_semantics(receipt), "outcome": outcome_value,
        "idempotency_replay": True,
    }


def _run_arm(packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path, arm_id: str) -> dict[str, Any]:
    provenance = _arm_provenance(packets[0], arm_id)
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory) / "shadow.db"
        shutil.copyfile(snapshot_path, clone)
        with NeuronGraphRAG(clone, config=_engine_config(packets[0], arm_id)) as engine:
            return {
                "arm_id": arm_id,
                "policy": "used" if arm_id == "used_q3_s1" else "confirmed",
                **provenance,
                "packets": [_run_packet(engine, packet, arm_id) for packet in packets],
                "final_edge_state": v1._edge_state(engine),
            }


def validate_result(result: Mapping[str, Any]) -> None:
    v1._exact_keys(
        result,
        {
            "schema_version", "protocol_id", "packet_ids", "slots",
            "snapshot_sha256", "capture_config", "capture_search_surface",
            "arms", "comparison", "efficiency", "replay",
        },
        "result",
    )
    if result["schema_version"] != SCHEMA_VERSION or result["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unknown result protocol")
    arms = v1._mapping(result["arms"], "arms")
    v1._exact_keys(arms, set(ARM_IDS), "arms")
    retrievals = []
    for arm_id in ARM_IDS:
        arm = v1._mapping(arms[arm_id], f"arms.{arm_id}")
        if arm.get("arm_id") != arm_id or arm.get("overridden_feedback_fields") != sorted(ARM_OVERRIDES[arm_id]):
            raise ValueError("arm policy provenance mismatch")
        if arm.get("search_surface") != result["capture_search_surface"]:
            raise ValueError("arm search surface diverged from capture")
        provenance = {
            "effective_config": arm.get("effective_config"),
            "retrieval_config_fingerprint": arm.get("retrieval_config_fingerprint"),
            "feedback_config_fingerprint": arm.get("feedback_config_fingerprint"),
            "full_config_fingerprint": arm.get("full_config_fingerprint"),
        }
        expected = effective_config_provenance(EngineConfig(**{**provenance["effective_config"]["retrieval"], **provenance["effective_config"]["feedback"]}))
        if provenance != expected:
            raise ValueError("arm effective config provenance mismatch")
        retrievals.append(provenance["effective_config"]["retrieval"])
    if retrievals[0] != retrievals[1] or retrievals[0] != result["capture_config"]["retrieval"]:
        raise ValueError("shadow arms do not share capture retrieval config")


def replay_packets(packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path) -> dict[str, Any]:
    ordered = list(packets)
    _validate_batch(ordered, snapshot_path)
    source_hash = v1.sha256_file(snapshot_path)
    arms = {}
    for arm_id in ARM_IDS:
        first = _run_arm(ordered, snapshot_path, arm_id)
        if first != _run_arm(ordered, snapshot_path, arm_id):
            raise RuntimeError(f"non-deterministic replay: {arm_id}")
        arms[arm_id] = first
    if v1.sha256_file(snapshot_path) != source_hash:
        raise RuntimeError("source snapshot changed during replay")
    result = {
        "schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID,
        "packet_ids": [packet["packet_id"] for packet in ordered],
        "slots": [packet["slot"] for packet in ordered], "snapshot_sha256": source_hash,
        "capture_config": ordered[0]["capture"]["effective_config"],
        "capture_search_surface": ordered[0]["capture"]["search_surface"],
        "arms": arms, "comparison": v1._comparison(arms),
        "efficiency": [{"packet_id": packet["packet_id"], **packet["efficiency"]} for packet in ordered],
        "replay": {
            "fresh_clone_per_arm": True, "cumulative_slot_order": True,
            "capture_time_reused": True, "capture_retrieval_config_reused": True,
            "repeated_semantic_replay": 2, "deterministic": True,
            "source_snapshot_unchanged": True,
        },
    }
    validate_result(result)
    return result


def replay_packet(packet: Mapping[str, Any], snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets([packet], snapshot_path)


def replay_registry(registry_dir: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets(load_effective_registry(registry_dir), snapshot_path)


def verify_result_against_packets(result: Mapping[str, Any], packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path) -> None:
    validate_result(result)
    if result != replay_packets(packets, snapshot_path):
        raise ValueError("stored result does not match exact semantic replay")


def verify_result_against_registry(result: Mapping[str, Any], registry_dir: str | Path, snapshot_path: str | Path) -> None:
    verify_result_against_packets(result, load_effective_registry(registry_dir), snapshot_path)


def build_packet_from_search(fixture: Mapping[str, Any], search: Mapping[str, Any], snapshot_path: str | Path) -> dict[str, Any]:
    if fixture.get("placeholder_only") is not True:
        raise ValueError("placeholder fixture must declare placeholder_only")
    if any(not str(node["node_id"]).startswith("placeholder-") or "example.invalid" not in str(node["source_url"]) for node in fixture["documents"]):
        raise ValueError("fixture contains a non-placeholder source identity")
    seed = fixture["packet_seed"]
    hits = search["hits"]
    used = seed["used_node_id"]
    hit = next(item for item in hits if item["node_id"] == used)
    paths = [path for path in hit["paths"] if path["steps"]]
    selected = max(paths, key=lambda path: (path["contribution"], path["seed_id"]))
    provenance = search["effective_config_provenance"]
    packet = {
        "schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID,
        "packet_id": seed["packet_id"], "slot": seed.get("slot", 1),
        "supersedes_packet_id": None, "task": seed["task"],
        "database_snapshot": {"sha256": v1.sha256_file(snapshot_path)},
        "retrieval": {
            "query": search["query"], "limit": seed["limit"],
            "candidates": [
                {"node_id": item["node_id"], "source_url": item["metadata"]["source_url"], "content_sha256": v1.sha256_text(item["text"])}
                for item in hits
            ],
            "used_node_id": used,
            "credited_path": {
                "node_id": used, "seed_id": selected["seed_id"],
                "steps": [{key: step[key] for key in ("source_id", "target_id", "edge_type")} for step in selected["steps"]],
            },
        },
        "source_use": [{"stage": stage, "node_id": used} for stage in SOURCE_USE_STAGES],
        "outcome": seed["outcome"], "efficiency": seed["efficiency"],
        "captured_at": seed["captured_at"],
        "capture": {
            "searched_at": search["created_at"],
            **provenance,
            "capture_fingerprint": "sha256:" + "0" * 64,
        },
    }
    packet["capture"]["capture_fingerprint"] = bind_capture_fingerprint(packet)
    validate_packet(packet)
    return packet


def create_placeholder_snapshot(fixture: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    with NeuronGraphRAG(path) as engine:
        for node in fixture["documents"]:
            engine.add_document(node["node_id"], node["text"], metadata={"source_url": node["source_url"]})
        for edge in fixture["edges"]:
            engine.add_edge(edge["source_id"], edge["target_id"], edge["edge_type"], weight=edge["weight"])


def probe_placeholder(fixture_path: str | Path) -> dict[str, Any]:
    from neuron_graph_rag_mcp.server import CONTRACT_VERSION, FeedbackMCPAdapter

    fixture = v1.read_canonical_json(fixture_path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snapshot = root / "placeholder.db"
        create_placeholder_snapshot(fixture, snapshot)
        adapter = FeedbackMCPAdapter(
            snapshot,
            config=EngineConfig(
                relation_feedback_evidence_quorum=3,
                sibling_feedback_normalization=1.0,
            ),
        )
        try:
            search = adapter._search(
                {
                    "contract_version": CONTRACT_VERSION,
                    "query": fixture["packet_seed"]["query"],
                    "limit": fixture["packet_seed"]["limit"],
                }
            )
        finally:
            adapter.close()
        packet = build_packet_from_search(fixture, search, snapshot)
        registered = capture_packet(packet, root / "registry")
        captured = v1.read_canonical_json(registered)
        result = replay_packet(captured, snapshot)
        output = root / "placeholder.result.json"
        v1.write_json_exclusive(output, result)
        verify_result_against_packets(v1.read_canonical_json(output), [captured], snapshot)
        return {
            "protocol_id": PROTOCOL_ID,
            "placeholder_only": True,
            "mcp_search_capture_round_trip": True,
            "capture_time_reused": True,
            "effective_config_provenance_verified": True,
            "replay_round_trip": True,
            "exclusive_writer_verified": True,
        }

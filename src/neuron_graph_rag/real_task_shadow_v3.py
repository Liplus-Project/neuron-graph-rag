from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import real_task_shadow as v1
from . import real_task_shadow_v2 as v2
from .evidence_feedback import EngineConfig


PROTOCOL_ID = "real-task-feedback-shadow-v3"
SCHEMA_VERSION = 3
ARM_IDS = v2.ARM_IDS
SOURCE_USE_STAGES = v2.SOURCE_USE_STAGES
PROTOCOL_ARTIFACTS = {
    "lifecycle_fixture": "tests/fixtures/real_task_shadow_v3.placeholder.json",
    "packet_schema": "tests/fixtures/real_task_shadow_v3.packet-schema.json",
    "result_schema": "tests/fixtures/real_task_shadow_v3.result-schema.json",
}
CAPTURE_CONTRACT = {
    "config_partition": "retrieval_and_feedback",
    "mcp_provenance": "effective_runtime_config",
    "search_time": "captured_trace_created_at",
    "snapshot_verification": "explicit_local_snapshot_only",
}
LIFECYCLE_CONTRACT = {
    "final_aggregate": "exclusive_create_once_after_packets",
    "manifest_mutation_after_registration": "forbidden",
    "packet_registry": "append_only_exclusive_canonical",
    "repository_integrity": "hash_registry_result_binding_without_snapshot",
    "stages": ["empty", "packets", "final_aggregate"],
}
REPLAY_CONTRACT = {
    "arm_clone_scope": "one_fresh_clone_per_arm_per_ordered_batch",
    "capture_retrieval_config": "shared_across_arms",
    "capture_time": "reused_per_packet",
    "result_verification": "exact_semantic_replay_from_packets_and_snapshot",
    "slot_order": "sequential_from_one",
}


def _as_v2_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(packet)
    converted["schema_version"] = v2.SCHEMA_VERSION
    converted["protocol_id"] = v2.PROTOCOL_ID
    return converted


def _as_v3_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(packet)
    converted["schema_version"] = SCHEMA_VERSION
    converted["protocol_id"] = PROTOCOL_ID
    return converted


def _as_v2_result(result: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(result)
    converted["schema_version"] = v2.SCHEMA_VERSION
    converted["protocol_id"] = v2.PROTOCOL_ID
    return converted


def _as_v3_result(result: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(result)
    converted["schema_version"] = SCHEMA_VERSION
    converted["protocol_id"] = PROTOCOL_ID
    return converted


def bind_capture_fingerprint(packet: Mapping[str, Any]) -> str:
    return v2.bind_capture_fingerprint(packet)


def validate_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != SCHEMA_VERSION or packet.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unknown packet protocol")
    v2.validate_packet(_as_v2_packet(packet))


def _read_registry(
    registry: Path, *, allow_active_lock: bool = False
) -> list[dict[str, Any]]:
    if not registry.exists():
        return []
    if not registry.is_dir():
        raise ValueError("packet registry must be a directory")
    entries = [
        path
        for path in registry.iterdir()
        if not (allow_active_lock and path.name == ".registry.lock" and path.is_file())
    ]
    if any(not path.is_file() or path.suffix != ".json" for path in entries):
        raise ValueError("packet registry contains a non-packet entry")
    packets = [v1.read_canonical_json(path) for path in sorted(entries)]
    for packet in packets:
        validate_packet(packet)
    return packets


def capture_packet(packet: Mapping[str, Any], registry_dir: str | Path) -> Path:
    validate_packet(packet)
    registry = Path(registry_dir)
    with v1._registry_lock(registry):
        packets = _read_registry(registry, allow_active_lock=True)
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


def _effective_registry(
    packets: Sequence[Mapping[str, Any]], *, require_nonempty: bool
) -> list[dict[str, Any]]:
    if not packets:
        if require_nonempty:
            raise ValueError("packet registry is empty")
        return []
    by_id = {str(packet["packet_id"]): dict(packet) for packet in packets}
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
            roots[slot] = dict(packet)
        else:
            if str(prior) not in by_id or str(prior) in successors:
                raise ValueError("invalid registry correction chain")
            successors[str(prior)] = dict(packet)
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


def load_effective_registry(registry_dir: str | Path) -> list[dict[str, Any]]:
    registry = Path(registry_dir)
    with v1._registry_lock(registry):
        packets = _read_registry(registry, allow_active_lock=True)
    return _effective_registry(packets, require_nonempty=True)


def verify_packet_against_snapshot(packet: Mapping[str, Any], snapshot_path: str | Path) -> None:
    validate_packet(packet)
    v2.verify_packet_against_snapshot(_as_v2_packet(packet), snapshot_path)


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != SCHEMA_VERSION or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unknown result protocol")
    v2.validate_result(_as_v2_result(result))
    packet_ids = v1._sequence(result["packet_ids"], "result.packet_ids")
    slots = v1._sequence(result["slots"], "result.slots")
    if not packet_ids or any(not isinstance(item, str) or not item for item in packet_ids):
        raise ValueError("result.packet_ids must be non-empty strings")
    if len(set(packet_ids)) != len(packet_ids):
        raise ValueError("result.packet_ids must be unique")
    if slots != list(range(1, len(packet_ids) + 1)):
        raise ValueError("result.slots must be sequential from one")
    v1._sha256(result["snapshot_sha256"], "result.snapshot_sha256")
    arms = v1._mapping(result["arms"], "result.arms")
    for arm_id in ARM_IDS:
        arm = v1._mapping(arms[arm_id], f"arm.{arm_id}")
        v1._exact_keys(
            arm,
            {
                "arm_id",
                "effective_config",
                "feedback_config_fingerprint",
                "final_edge_state",
                "full_config_fingerprint",
                "overridden_feedback_fields",
                "packets",
                "policy",
                "retrieval_config_fingerprint",
                "search_surface",
            },
            f"arm.{arm_id}",
        )
        arm_packets = v1._sequence(arm["packets"], f"arm.{arm_id}.packets")
        if [packet["packet_id"] for packet in arm_packets] != packet_ids:
            raise ValueError(f"arm packet ID order mismatch: {arm_id}")
        if [packet["slot"] for packet in arm_packets] != slots:
            raise ValueError(f"arm slot order mismatch: {arm_id}")
        for packet in arm_packets:
            v1._validate_packet_replay(packet, arm_id)
        final_edges = v1._validate_edge_state(
            arm["final_edge_state"], f"arm.{arm_id}.final_edge_state"
        )
        if final_edges != arm_packets[-1]["edge_state_after"]:
            raise ValueError(f"arm final edge state mismatch: {arm_id}")
    comparison = v1._mapping(result["comparison"], "comparison")
    if comparison != v1._comparison(arms):
        raise ValueError("result comparison is not recomputable from arm metrics")
    efficiency = v1._sequence(result["efficiency"], "result.efficiency")
    if len(efficiency) != len(packet_ids):
        raise ValueError("result efficiency must have one row per packet")
    for packet_id, raw_row in zip(packet_ids, efficiency, strict=True):
        row = v1._mapping(raw_row, "result efficiency row")
        v1._exact_keys(
            row,
            {"elapsed_seconds", "packet_id", "research_count", "token_count", "tool_calls"},
            "result efficiency row",
        )
        if row["packet_id"] != packet_id:
            raise ValueError("result efficiency packet order mismatch")
        v1._validate_efficiency(
            {key: value for key, value in row.items() if key != "packet_id"}
        )
    if result["replay"] != {
        "fresh_clone_per_arm": True,
        "cumulative_slot_order": True,
        "capture_time_reused": True,
        "capture_retrieval_config_reused": True,
        "repeated_semantic_replay": 2,
        "deterministic": True,
        "source_snapshot_unchanged": True,
    }:
        raise ValueError("result replay proof is incomplete")


def replay_packets(
    packets: Sequence[Mapping[str, Any]], snapshot_path: str | Path
) -> dict[str, Any]:
    ordered = list(packets)
    for packet in ordered:
        validate_packet(packet)
    result = _as_v3_result(
        v2.replay_packets([_as_v2_packet(packet) for packet in ordered], snapshot_path)
    )
    validate_result(result)
    return result


def replay_packet(packet: Mapping[str, Any], snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets([packet], snapshot_path)


def replay_registry(registry_dir: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    return replay_packets(load_effective_registry(registry_dir), snapshot_path)


def verify_result_against_packets(
    result: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    snapshot_path: str | Path,
) -> None:
    validate_result(result)
    if result != replay_packets(packets, snapshot_path):
        raise ValueError("stored result does not match exact semantic replay")


def verify_result_against_registry(
    result: Mapping[str, Any], registry_dir: str | Path, snapshot_path: str | Path
) -> None:
    verify_result_against_packets(
        result, load_effective_registry(registry_dir), snapshot_path
    )


def build_packet_from_search(
    fixture: Mapping[str, Any], search: Mapping[str, Any], snapshot_path: str | Path
) -> dict[str, Any]:
    packet = _as_v3_packet(v2.build_packet_from_search(fixture, search, snapshot_path))
    validate_packet(packet)
    return packet


def create_placeholder_snapshot(fixture: Mapping[str, Any], output: str | Path) -> None:
    v2.create_placeholder_snapshot(fixture, output)


def write_final_aggregate(result: Mapping[str, Any], output: str | Path) -> None:
    validate_result(result)
    v1.write_json_exclusive(output, result)


def _relative_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} must be a non-empty relative path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{name} must stay below its root")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{name} escapes its root")
    return resolved


def _verify_hashes(root: Path, values: Any, name: str) -> None:
    hashes = v1._mapping(values, name)
    if not hashes:
        raise ValueError(f"{name} must not be empty")
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{name}.{relative} must be a sha256 hex digest")
        try:
            int(expected, 16)
        except ValueError as error:
            raise ValueError(
                f"{name}.{relative} must be a sha256 hex digest"
            ) from error
        path = _relative_path(root, relative, f"{name}.{relative}")
        if not path.is_file():
            raise ValueError(f"frozen artifact is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")


def audit_repository_lifecycle(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    registered_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = v1.read_canonical_json(manifest_path)
    v1._exact_keys(
        manifest,
        {
            "arms",
            "artifact_sha256",
            "capture_contract",
            "freeze_observation_status",
            "legacy_artifact_sha256",
            "lifecycle_contract",
            "protocol_artifacts",
            "protocol_id",
            "registered_outputs",
            "replay_contract",
            "result_free",
            "schema_version",
            "supersedes_protocol_id",
        },
        "manifest",
    )
    if manifest["protocol_id"] != PROTOCOL_ID or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unknown lifecycle manifest protocol")
    if manifest["supersedes_protocol_id"] != v2.PROTOCOL_ID:
        raise ValueError("lifecycle manifest has an unknown predecessor")
    if manifest["arms"] != list(ARM_IDS):
        raise ValueError("lifecycle manifest changed the frozen arms")
    if manifest["result_free"] is not True or manifest["freeze_observation_status"] != "not_started_at_protocol_commit":
        raise ValueError("lifecycle manifest is not a result-free freeze")
    for field, expected in (
        ("protocol_artifacts", PROTOCOL_ARTIFACTS),
        ("capture_contract", CAPTURE_CONTRACT),
        ("lifecycle_contract", LIFECYCLE_CONTRACT),
        ("replay_contract", REPLAY_CONTRACT),
    ):
        if manifest[field] != expected:
            raise ValueError(f"lifecycle manifest changed the frozen {field}")
    root = Path(repository_root).resolve()
    output_root = root if registered_root is None else Path(registered_root).resolve()
    _verify_hashes(root, manifest["artifact_sha256"], "artifact_sha256")
    _verify_hashes(root, manifest["legacy_artifact_sha256"], "legacy_artifact_sha256")
    outputs = v1._mapping(manifest["registered_outputs"], "registered_outputs")
    v1._exact_keys(outputs, {"final_aggregate", "packet_registry"}, "registered_outputs")
    registry = _relative_path(output_root, outputs["packet_registry"], "packet_registry")
    aggregate = _relative_path(output_root, outputs["final_aggregate"], "final_aggregate")
    packets = _read_registry(registry)
    effective = _effective_registry(packets, require_nonempty=False)
    if aggregate.parent.exists():
        allowed = {aggregate.resolve()} if aggregate.exists() else set()
        entries = {path.resolve() for path in aggregate.parent.iterdir()}
        if entries != allowed:
            raise ValueError("observed output contains more than the one-time final aggregate")
    has_aggregate = aggregate.exists()
    if has_aggregate:
        if not effective:
            raise ValueError("final aggregate exists without effective packets")
        result = v1.read_canonical_json(aggregate)
        validate_result(result)
        if result["packet_ids"] != [packet["packet_id"] for packet in effective]:
            raise ValueError("final aggregate packet IDs do not bind the effective registry")
        if result["slots"] != [packet["slot"] for packet in effective]:
            raise ValueError("final aggregate slots do not bind the effective registry")
        snapshot_hashes = {packet["database_snapshot"]["sha256"] for packet in effective}
        if snapshot_hashes != {result["snapshot_sha256"]}:
            raise ValueError("final aggregate snapshot does not bind the effective registry")
        if result["capture_config"] != effective[0]["capture"]["effective_config"]:
            raise ValueError("final aggregate config does not bind the effective registry")
        if result["capture_search_surface"] != effective[0]["capture"]["search_surface"]:
            raise ValueError("final aggregate surface does not bind the effective registry")
    stage = "final_aggregate" if has_aggregate else "packets" if packets else "empty"
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "packet_file_count": len(packets),
        "effective_packet_count": len(effective),
        "final_aggregate": has_aggregate,
    }


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
        write_final_aggregate(result, output)
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

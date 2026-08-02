from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .d1_fixture import load_fixture
from .engine import EngineConfig, NeuronGraphRAG


BLIND_SELECTION_SCHEMA_VERSION = 1
CHANNELS = {"lexical", "relation"}
SELECTIONS = CHANNELS | {"abstain"}
FORBIDDEN_PACKET_KEYS = {
    "acceptable_rank",
    "cohort",
    "expected_node_id",
    "expected_path",
    "expected_relation_empty",
    "gate",
    "gold",
    "intended_channel",
    "prior_result",
}
RESPONSE_KEYS = {
    "case_id",
    "selected_channel",
    "trace_id",
    "node_id",
    "rationale",
}


def read_blind_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != BLIND_SELECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported blind selection schema version")
    if manifest.get("experiment_id") != "d1-liplus-blind-channel-selection-v2":
        raise ValueError("Unexpected blind selection experiment ID")
    if len(manifest.get("gate", [])) != 12:
        raise ValueError("Blind selection manifest must freeze twelve gates")
    _audit_v1_hashes(manifest_path, manifest)
    prompt_path = _repo_root(manifest_path) / manifest["judge_prompt"]["path"]
    if not _matches_frozen_text_bytes(
        prompt_path, str(manifest["judge_prompt"]["sha256"])
    ):
        raise ValueError("Frozen judge prompt hash mismatch")
    _validate_prompt(prompt_path.read_text(encoding="utf-8"))
    return manifest


def audit_result_free_freeze(path: str | Path) -> None:
    manifest_path = Path(path)
    manifest = read_blind_manifest(manifest_path)
    root = _repo_root(manifest_path)
    existing = [
        item for item in manifest.get("frozen_absent_paths", []) if (root / item).exists()
    ]
    if existing:
        raise ValueError(f"Result-free freeze contains observed artifacts: {existing}")


def generate_blind_packet(
    manifest_path: str | Path,
    split_name: str,
    *,
    development_result_path: str | Path | None = None,
) -> dict[str, Any]:
    if split_name not in {"development", "holdout"}:
        raise ValueError("split_name must be development or holdout")
    manifest_path = Path(manifest_path)
    manifest = read_blind_manifest(manifest_path)
    if split_name == "holdout":
        if development_result_path is None:
            raise ValueError("Holdout packet requires a development result")
        development = _read_json(Path(development_result_path))
        if development.get("stage") != "development" or not development.get(
            "gate_passed"
        ):
            raise ValueError("Stop rule forbids generating the holdout packet")
        if development.get("manifest_sha256") != _byte_sha256(manifest_path):
            raise ValueError("Development result does not match frozen manifest")
    elif development_result_path is not None:
        raise ValueError("development_result_path is only valid for holdout")

    root = _repo_root(manifest_path)
    v1_manifest = _read_json(root / manifest["v1_manifest"])
    split = v1_manifest[split_name]
    fixture_path = (root / manifest["v1_manifest"]).parent / split["fixture"]
    gold_path = (root / manifest["v1_manifest"]).parent / split["gold"]
    gold = _read_json(gold_path)
    config = EngineConfig(
        **{
            key: value
            for key, value in v1_manifest["shared_config"].items()
            if key != "limit"
        }
    )
    limit = int(v1_manifest["shared_config"]["limit"])
    cases = [
        _make_packet_case(
            fixture_path,
            case_id=f"case-{index:04d}",
            query=str(case["query"]),
            config=config,
            limit=limit,
        )
        for index, case in enumerate(gold["cases"], start=1)
    ]
    packet = {
        "schema_version": BLIND_SELECTION_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": split_name,
        "protocol_manifest_sha256": _byte_sha256(manifest_path),
        "judge_prompt_sha256": manifest["judge_prompt"]["sha256"],
        "lane_semantics": {
            "lexical": "BM25-only ordering from direct query-to-document term matching.",
            "relation": "Edge-only graph activation ordering after query-based seed selection.",
            "rank_scope": "Ranks are meaningful only within their own lane.",
        },
        "response_schema": {
            "case_id": "opaque case ID",
            "selected_channel": "lexical | relation | abstain",
            "trace_id": "selected lane trace ID or null",
            "node_id": "selected hit node ID or null",
            "rationale": "short packet-based reason",
        },
        "cases": cases,
    }
    validate_packet(packet)
    return packet


def validate_packet(packet: dict[str, Any]) -> None:
    if packet.get("schema_version") != BLIND_SELECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported packet schema version")
    cases = packet.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("Blind packet must contain exactly four cases")
    _reject_forbidden_packet_keys(packet)
    case_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if re.fullmatch(r"case-\d{4}", case_id) is None or case_id in case_ids:
            raise ValueError("Packet case IDs must be unique and opaque")
        case_ids.add(case_id)
        lanes = case.get("lanes")
        if not isinstance(lanes, dict) or set(lanes) != CHANNELS:
            raise ValueError(f"Packet case has invalid lanes: {case_id}")
        trace_ids = {str(lanes[channel].get("trace_id", "")) for channel in CHANNELS}
        if "" in trace_ids or len(trace_ids) != 2:
            raise ValueError(f"Packet traces must be independent: {case_id}")
        if not isinstance(case.get("edge_weight_audit", {}).get("unchanged"), bool):
            raise ValueError(f"Packet case lacks edge audit: {case_id}")
        for channel in CHANNELS:
            hits = lanes[channel].get("hits")
            if not isinstance(hits, list):
                raise ValueError(f"Packet lane hits must be an array: {case_id}")
            for expected_rank, hit in enumerate(hits, start=1):
                if hit.get("rank") != expected_rank or not hit.get("node_id"):
                    raise ValueError(f"Packet lane ranks must be contiguous: {case_id}")
                if channel == "lexical" and "paths" in hit:
                    raise ValueError("Lexical packet hits must not carry graph paths")
                if channel == "relation":
                    paths = hit.get("paths")
                    if not isinstance(paths, list) or not paths:
                        raise ValueError("Relation packet hits require edge paths")
                    for path in paths:
                        if not path.get("raw_steps") or not path.get("projected_steps"):
                            raise ValueError("Relation paths must exclude zero-hop paths")
                        if project_path(path["raw_steps"]) != path["projected_steps"]:
                            raise ValueError("Projected relation path differs from raw path")


def project_path(raw_steps: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "source_id": str(step["source_id"]),
            "target_id": str(step["target_id"]),
            "edge_type": str(step["edge_type"]),
        }
        for step in raw_steps
    ]


def capture_judge_response(
    packet_path: str | Path,
    raw_response: str,
    *,
    judge_id: str,
    model: str,
    agent_type: str,
    executed_at: str | None = None,
) -> dict[str, Any]:
    packet_path = Path(packet_path)
    packet = _read_json(packet_path)
    validate_packet(packet)
    parsed = parse_judge_response(raw_response)
    validate_judge_responses(packet, parsed)
    return {
        "schema_version": BLIND_SELECTION_SCHEMA_VERSION,
        "packet_sha256": _byte_sha256(packet_path),
        "judge_id": _required_text(judge_id, "judge_id"),
        "model": _required_text(model, "model"),
        "agent_type": _required_text(agent_type, "agent_type"),
        "executed_at": _required_text(
            executed_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "executed_at",
        ),
        "raw_response": raw_response,
        "parsed_responses": parsed,
    }


def parse_judge_response(raw_response: str) -> list[dict[str, Any]]:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip("\r\n")
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"responses"}:
        raise ValueError("Judge response must be the frozen JSON object")
    responses = payload["responses"]
    if not isinstance(responses, list):
        raise ValueError("Judge response must contain a responses array")
    return responses


def validate_judge_responses(
    packet: dict[str, Any], responses: list[dict[str, Any]]
) -> None:
    validate_packet(packet)
    cases = {str(case["case_id"]): case for case in packet["cases"]}
    if len(responses) != len(cases):
        raise ValueError("Judge must answer every packet case exactly once")
    seen: set[str] = set()
    for response in responses:
        if not isinstance(response, dict) or set(response) != RESPONSE_KEYS:
            raise ValueError("Judge response fields differ from frozen schema")
        case_id = str(response["case_id"])
        if case_id not in cases or case_id in seen:
            raise ValueError("Judge response has an unknown or duplicate case ID")
        seen.add(case_id)
        selected = str(response["selected_channel"])
        if selected not in SELECTIONS:
            raise ValueError(f"Unknown selected channel: {selected}")
        if not str(response["rationale"]).strip():
            raise ValueError("Judge rationale must not be empty")
        if selected == "abstain":
            if response["trace_id"] is not None or response["node_id"] is not None:
                raise ValueError("Abstention requires null trace_id and node_id")
            continue
        lane = cases[case_id]["lanes"][selected]
        if response["trace_id"] != lane["trace_id"]:
            raise ValueError("Selected trace does not belong to selected channel")
        members = {hit["node_id"] for hit in lane["hits"]}
        if response["node_id"] not in members:
            raise ValueError("Selected node does not belong to selected trace")


def aggregate_blind_results(
    manifest_path: str | Path,
    packet_path: str | Path,
    response_paths: Iterable[str | Path],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    packet_path = Path(packet_path)
    manifest = read_blind_manifest(manifest_path)
    packet = _read_json(packet_path)
    validate_packet(packet)
    if packet.get("experiment_id") != manifest["experiment_id"]:
        raise ValueError("Packet experiment ID does not match frozen manifest")
    if packet.get("protocol_manifest_sha256") != _byte_sha256(manifest_path):
        raise ValueError("Packet does not match frozen blind manifest")
    if packet.get("judge_prompt_sha256") != manifest["judge_prompt"]["sha256"]:
        raise ValueError("Packet judge prompt does not match frozen manifest")
    response_path_list = [Path(path) for path in response_paths]
    artifacts = [_read_json(path) for path in response_path_list]
    if len(artifacts) != 3:
        raise ValueError("Exactly three fresh judge artifacts are required")
    packet_hash = _byte_sha256(packet_path)
    judge_ids: set[str] = set()
    for artifact in artifacts:
        if artifact.get("schema_version") != BLIND_SELECTION_SCHEMA_VERSION:
            raise ValueError("Unsupported judge artifact schema version")
        if artifact.get("packet_sha256") != packet_hash:
            raise ValueError("Judge artifact does not match packet")
        judge_id = str(artifact.get("judge_id", ""))
        if not judge_id or judge_id in judge_ids:
            raise ValueError("Judge IDs must be non-empty and unique")
        judge_ids.add(judge_id)
        for metadata_name in ("model", "agent_type", "executed_at"):
            _required_text(str(artifact.get(metadata_name, "")), metadata_name)
        parsed = artifact.get("parsed_responses", [])
        if parse_judge_response(str(artifact.get("raw_response", ""))) != parsed:
            raise ValueError("Judge raw response and parsed response differ")
        validate_judge_responses(packet, parsed)

    root = _repo_root(manifest_path)
    v1_manifest_path = root / manifest["v1_manifest"]
    v1_manifest = _read_json(v1_manifest_path)
    split_name = str(packet["stage"])
    gold_path = v1_manifest_path.parent / v1_manifest[split_name]["gold"]
    gold_cases = _read_json(gold_path)["cases"]
    packet_cases = packet["cases"]
    if len(gold_cases) != len(packet_cases):
        raise ValueError("Packet and evaluation cases differ")
    response_maps = [
        {item["case_id"]: item for item in artifact["parsed_responses"]}
        for artifact in artifacts
    ]
    evaluated_cases = [
        _evaluate_majority(packet_case, gold_case, response_maps)
        for packet_case, gold_case in zip(packet_cases, gold_cases, strict=True)
    ]
    judge_correct_counts = [
        sum(
            response_maps[index][case["case_id"]]["node_id"]
            == gold["expected_node_id"]
            for case, gold in zip(packet_cases, gold_cases, strict=True)
        )
        for index in range(3)
    ]
    relation_cases = [case for case in evaluated_cases if case["cohort"] == "relation"]
    control_cases = [
        case
        for case in evaluated_cases
        if case["cohort"] in {"direct_lookup", "directional_negative"}
    ]
    packet_clean = _packet_has_no_score_fields(packet)
    gate = {
        "packet_and_prompt_exclude_answer_fields": packet_clean,
        "judge_responses_are_valid_and_trace_bound": True,
        "majority_exists_for_every_case": all(case["has_majority"] for case in evaluated_cases),
        "majority_selects_every_expected_node": all(case["node_correct"] for case in evaluated_cases),
        "lexical_controls_select_lexical": all(case["channel_correct"] for case in control_cases),
        "relation_case_selects_relation": all(case["channel_correct"] for case in relation_cases),
        "relation_path_matches_after_projection": all(case["path_matched"] for case in relation_cases),
        "each_judge_selects_three_expected_nodes": all(count >= 3 for count in judge_correct_counts),
        "no_cross_lane_score_comparison": packet_clean,
        "search_does_not_change_edge_weights": all(case["edge_weight_unchanged"] for case in evaluated_cases),
        "v1_artifacts_match_frozen_bytes": True,
        "observed_artifact_overwrite_is_refused": bool(manifest["stop_rule"]["refuse_overwrite"]),
    }
    majority_correct = sum(case["node_correct"] for case in evaluated_cases)
    channel_correct = sum(case["channel_correct"] for case in evaluated_cases)
    majority_abstentions = sum(case["selected_channel"] == "abstain" for case in evaluated_cases)
    all_votes = [
        (response[case["case_id"]]["selected_channel"], response[case["case_id"]]["node_id"])
        for case in packet_cases
        for response in response_maps
    ]
    agreement_count = sum(
        len(
            {
                (
                    response[case["case_id"]]["selected_channel"],
                    response[case["case_id"]]["node_id"],
                )
                for response in response_maps
            }
        )
        == 1
        for case in packet_cases
    )
    union_oracle = sum(
        any(
            hit["node_id"] == gold["expected_node_id"]
            for lane in case["lanes"].values()
            for hit in lane["hits"]
        )
        for case, gold in zip(packet_cases, gold_cases, strict=True)
    )
    result = {
        "schema_version": BLIND_SELECTION_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": split_name,
        "manifest_sha256": _byte_sha256(manifest_path),
        "packet_sha256": packet_hash,
        "judge_artifacts": [
            {
                "judge_id": artifact["judge_id"],
                "model": artifact["model"],
                "agent_type": artifact["agent_type"],
                "executed_at": artifact["executed_at"],
                "artifact_sha256": _byte_sha256(path),
            }
            for path, artifact in zip(response_path_list, artifacts, strict=True)
        ],
        "cases": evaluated_cases,
        "metrics": {
            "judge_expected_node_accuracy": [count / len(packet_cases) for count in judge_correct_counts],
            "majority_expected_node_accuracy": majority_correct / len(packet_cases),
            "majority_channel_accuracy": channel_correct / len(packet_cases),
            "majority_abstention_rate": majority_abstentions / len(packet_cases),
            "unanimous_agreement_rate": agreement_count / len(packet_cases),
            "selected_node_mrr": sum(
                float(case["selected_reciprocal_rank"])
                for case in evaluated_cases
            )
            / len(evaluated_cases),
            "union_oracle_coverage": union_oracle / len(packet_cases),
            "union_oracle_gap": (union_oracle - majority_correct) / len(packet_cases),
            "total_judge_votes": len(all_votes),
        },
        "gate": gate,
        "gate_passed": all(gate.values()),
    }
    if split_name == "development":
        result["holdout_status"] = (
            "eligible_for_single_packet" if result["gate_passed"] else "not_opened_no_candidate"
        )
    else:
        result["claim_scope"] = (
            "limited_support_on_frozen_holdout"
            if result["gate_passed"]
            else "no_support_on_frozen_holdout"
        )
    return result


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _make_packet_case(
    fixture_path: Path,
    *,
    case_id: str,
    query: str,
    config: EngineConfig,
    limit: int,
) -> dict[str, Any]:
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before = _edge_snapshot(engine)
        channels = engine.search_channels(query, limit=limit, now=1_000.0)
        after = _edge_snapshot(engine)
    return {
        "case_id": case_id,
        "query": query,
        "lanes": {
            "lexical": {
                "trace_id": channels.lexical.trace_id,
                "hits": [_packet_hit(hit, include_paths=False) for hit in channels.lexical.hits],
            },
            "relation": {
                "trace_id": channels.relation.trace_id,
                "hits": [_packet_hit(hit, include_paths=True) for hit in channels.relation.hits],
            },
        },
        "agreement_node_ids": list(channels.agreement_node_ids),
        "edge_weight_audit": {
            "before_sha256": _json_sha256(before),
            "after_sha256": _json_sha256(after),
            "unchanged": before == after,
        },
    }


def _packet_hit(hit: Any, *, include_paths: bool) -> dict[str, Any]:
    metadata = dict(hit.node.metadata)
    source_metadata = {
        key: metadata[key]
        for key in ("repo", "type", "doc_path", "source_url", "state", "updated_at")
        if key in metadata
    }
    value: dict[str, Any] = {
        "rank": int(hit.rank),
        "node_id": hit.node.node_id,
        "title": str(metadata.get("doc_path") or hit.node.text.splitlines()[0]),
        "content": hit.node.text,
        "source_metadata": source_metadata,
    }
    if include_paths:
        value["paths"] = [
            {
                "raw_steps": [dict(step) for step in path["steps"]],
                "projected_steps": project_path(path["steps"]),
            }
            for path in hit.explain()["paths"]
        ]
    return value


def _evaluate_majority(
    packet_case: dict[str, Any],
    gold_case: dict[str, Any],
    response_maps: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    case_id = packet_case["case_id"]
    votes = [response[case_id] for response in response_maps]
    counts = Counter((vote["selected_channel"], vote["node_id"]) for vote in votes)
    pair, count = counts.most_common(1)[0]
    has_majority = count >= 2 and pair[0] != "abstain"
    selected_channel = str(pair[0]) if has_majority else "abstain"
    selected_node = pair[1] if has_majority else None
    selected_trace = (
        packet_case["lanes"][selected_channel]["trace_id"] if has_majority else None
    )
    selected_hit = next(
        (
            hit
            for hit in packet_case["lanes"].get(selected_channel, {}).get("hits", [])
            if hit["node_id"] == selected_node
        ),
        None,
    )
    selected_rank = int(selected_hit["rank"]) if selected_hit else None
    expected_node = str(gold_case["expected_node_id"])
    intended = str(gold_case["intended_channel"])
    expected_path = gold_case.get("expected_path", [])
    observed_projected = [
        path["projected_steps"]
        for path in (selected_hit or {}).get("paths", [])
    ]
    path_matched = (
        bool(expected_path)
        and any(path == expected_path for path in observed_projected)
        and all(path for path in observed_projected)
    ) if intended == "relation" else True
    return {
        "case_id": case_id,
        "cohort": gold_case["cohort"],
        "vote_counts": [
            {"selected_channel": channel, "node_id": node_id, "count": votes_count}
            for (channel, node_id), votes_count in sorted(counts.items(), key=str)
        ],
        "has_majority": has_majority,
        "selected_channel": selected_channel,
        "selected_trace_id": selected_trace,
        "selected_node_id": selected_node,
        "selected_rank": selected_rank,
        "selected_reciprocal_rank": (
            1.0 / selected_rank if selected_rank is not None else 0.0
        ),
        "node_correct": selected_node == expected_node,
        "channel_correct": selected_channel == intended,
        "raw_selected_paths": (selected_hit or {}).get("paths", []),
        "projected_selected_paths": observed_projected,
        "path_matched": path_matched,
        "edge_weight_unchanged": packet_case["edge_weight_audit"]["unchanged"],
    }


def _audit_v1_hashes(manifest_path: Path, manifest: dict[str, Any]) -> None:
    root = _repo_root(manifest_path)
    entries = manifest.get("frozen_v1_bytes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Blind manifest must freeze v1 byte hashes")
    for entry in entries:
        if not _matches_frozen_text_bytes(
            root / entry["path"], str(entry["sha256"])
        ):
            raise ValueError(f"Frozen v1 byte hash mismatch: {entry['path']}")


def _matches_frozen_text_bytes(path: Path, expected_sha256: str) -> bool:
    raw = path.read_bytes()
    if _bytes_sha256(raw) == expected_sha256:
        return True
    if b"\r" not in raw:
        alternate = raw.replace(b"\n", b"\r\n")
    elif b"\r" not in raw.replace(b"\r\n", b"") and b"\n" not in raw.replace(
        b"\r\n", b""
    ):
        alternate = raw.replace(b"\r\n", b"\n")
    else:
        return False
    return alternate != raw and _bytes_sha256(alternate) == expected_sha256


def _validate_prompt(prompt: str) -> None:
    lowered = prompt.casefold()
    forbidden_literals = (
        "acceptable_rank",
        "cohort",
        "expected_path",
        "intended_channel",
        "expected_node_id",
        "prior_result",
        "channels-dev-",
        "channels-holdout-",
    )
    if any(literal in lowered for literal in forbidden_literals):
        raise ValueError("Judge prompt contains a forbidden answer field")
    if "laneをまたいで数値比較しない" not in prompt:
        raise ValueError("Judge prompt must forbid cross-lane numeric comparison")


def _reject_forbidden_packet_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in FORBIDDEN_PACKET_KEYS or "score" in lowered:
                raise ValueError(f"Packet contains forbidden field: {key}")
            _reject_forbidden_packet_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_packet_keys(child)


def _packet_has_no_score_fields(packet: dict[str, Any]) -> bool:
    try:
        _reject_forbidden_packet_keys(packet)
    except ValueError:
        return False
    return True


def _edge_snapshot(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    return [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "reinforced_count": edge.reinforced_count,
        }
        for edge in engine.store.list_edges()
    ]


def _repo_root(manifest_path: Path) -> Path:
    return manifest_path.resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _byte_sha256(path: Path) -> str:
    return _bytes_sha256(path.read_bytes())


def _bytes_sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _required_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text

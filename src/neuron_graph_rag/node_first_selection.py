from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .blind_selection import (
    CHANNELS,
    RESPONSE_KEYS,
    SELECTIONS,
    _make_packet_case,
    _matches_frozen_text_bytes,
)
from .engine import EngineConfig


NODE_FIRST_SCHEMA_VERSION = 1
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


def read_node_first_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != NODE_FIRST_SCHEMA_VERSION:
        raise ValueError("Unsupported node-first manifest schema version")
    if manifest.get("experiment_id") != "d1-liplus-node-first-blind-selection-v3":
        raise ValueError("Unexpected node-first experiment ID")
    if len(manifest.get("gate", [])) != 12:
        raise ValueError("Node-first manifest must freeze twelve gates")
    if manifest.get("case_count") != 4 or manifest.get("judges_per_case") != 3:
        raise ValueError("Node-first protocol requires four cases and three judges")
    _audit_prior_artifacts(manifest_path, manifest)
    prompt = _repo_root(manifest_path) / manifest["judge_prompt"]["path"]
    if _canonical_checkout_sha256(prompt) != manifest["judge_prompt"][
        "canonical_sha256"
    ]:
        raise ValueError("Frozen node-first judge prompt hash mismatch")
    _validate_prompt(prompt.read_text(encoding="utf-8"))
    return manifest


def audit_node_first_result_freeze(path: str | Path) -> None:
    manifest_path = Path(path)
    manifest = read_node_first_manifest(manifest_path)
    root = _repo_root(manifest_path)
    existing = [
        str(path.relative_to(root))
        for path in _observed_paths(root, manifest)
        if path.exists()
    ]
    if existing:
        raise ValueError(f"Result-free freeze contains observed artifacts: {existing}")


def generate_node_first_stage(
    manifest_path: str | Path,
    split_name: str,
    *,
    development_result_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    if split_name not in {"development", "holdout"}:
        raise ValueError("split_name must be development or holdout")
    manifest_path = Path(manifest_path)
    manifest = read_node_first_manifest(manifest_path)
    manifest_hash = _canonical_checkout_sha256(manifest_path)
    if split_name == "holdout":
        if development_result_path is None:
            raise ValueError("Holdout stage requires a development result")
        development = _read_json(Path(development_result_path))
        if development.get("stage") != "development" or not development.get(
            "gate_passed"
        ):
            raise ValueError("Stop rule forbids generating the holdout stage")
        if development.get("manifest_canonical_sha256") != manifest_hash:
            raise ValueError("Development result does not match node-first manifest")
    elif development_result_path is not None:
        raise ValueError("development_result_path is only valid for holdout")

    root = _repo_root(manifest_path)
    v1_manifest_path = root / manifest["v1_manifest"]
    v1_manifest = _read_json(v1_manifest_path)
    split = v1_manifest[split_name]
    fixture_path = v1_manifest_path.parent / split["fixture"]
    gold = _read_json(v1_manifest_path.parent / split["gold"])
    config = EngineConfig(
        **{
            key: value
            for key, value in v1_manifest["shared_config"].items()
            if key != "limit"
        }
    )
    limit = int(v1_manifest["shared_config"]["limit"])
    overrides = manifest["query_overrides"][split_name]
    cases = []
    for index, source_case in enumerate(gold["cases"], start=1):
        case_id = f"case-{index:04d}"
        query = str(overrides.get(case_id, source_case["query"]))
        cases.append(
            _make_packet_case(
                fixture_path,
                case_id=case_id,
                query=query,
                config=config,
                limit=limit,
            )
        )

    common = {
        "schema_version": NODE_FIRST_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": split_name,
        "protocol_manifest_canonical_sha256": manifest_hash,
        "judge_prompt_canonical_sha256": manifest["judge_prompt"][
            "canonical_sha256"
        ],
        "lane_semantics": {
            "lexical": "BM25-only ordering from direct query-to-document term matching.",
            "relation": "Edge-only graph activation ordering after query-based seed selection.",
            "rank_scope": "Ranks are meaningful only within their own lane.",
        },
        "response_schema": {
            "case_id": "the single opaque case ID",
            "selected_channel": "lexical | relation | abstain",
            "trace_id": "selected lane trace ID or null",
            "node_id": "selected hit node ID or null",
            "rationale": "short packet-based reason",
        },
    }
    root_paths = manifest["artifact_paths"][split_name]
    case_artifacts: list[tuple[Path, dict[str, Any]]] = []
    stage_refs = []
    for case in cases:
        case_packet = {**common, "case": case}
        validate_case_packet(case_packet)
        relative_path = root_paths["case_packet_template"].format(
            case_id=case["case_id"]
        )
        case_path = root / relative_path
        case_hash = _bytes_sha256(_json_document_bytes(case_packet))
        case_artifacts.append((case_path, case_packet))
        stage_refs.append(
            {
                "case_id": case["case_id"],
                "path": relative_path,
                "sha256": case_hash,
            }
        )
    stage_packet = {
        **common,
        "cases": cases,
        "case_packet_artifacts": stage_refs,
        "invocation_contract": {
            "cases_per_invocation": 1,
            "fresh_judges_per_case": int(manifest["judges_per_case"]),
            "context_reuse": "forbidden",
        },
    }
    validate_stage_packet(stage_packet)
    return stage_packet, case_artifacts


def write_node_first_stage_exclusive(
    manifest_path: str | Path,
    split_name: str,
    stage_packet: dict[str, Any],
    case_artifacts: list[tuple[Path, dict[str, Any]]],
) -> None:
    manifest_path = Path(manifest_path)
    manifest = read_node_first_manifest(manifest_path)
    root = _repo_root(manifest_path)
    stage_path = root / manifest["artifact_paths"][split_name]["stage_packet"]
    targets = [stage_path, *(path for path, _ in case_artifacts)]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite stage artifacts: {existing}")
    for path, payload in case_artifacts:
        write_json_exclusive(path, payload)
    write_json_exclusive(stage_path, stage_packet)


def validate_stage_packet(packet: dict[str, Any]) -> None:
    cases = packet.get("cases")
    refs = packet.get("case_packet_artifacts")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("Node-first stage packet must contain four cases")
    if not isinstance(refs, list) or len(refs) != 4:
        raise ValueError("Node-first stage packet must reference four case packets")
    _validate_common_packet(packet)
    case_ids = []
    for case in cases:
        _validate_case(case)
        case_ids.append(case["case_id"])
    if len(set(case_ids)) != 4:
        raise ValueError("Stage packet case IDs must be unique")
    if [ref.get("case_id") for ref in refs] != case_ids:
        raise ValueError("Stage packet references differ from case order")
    invocation = packet.get("invocation_contract", {})
    if invocation.get("cases_per_invocation") != 1:
        raise ValueError("Node-first invocation must contain exactly one case")


def validate_case_packet(packet: dict[str, Any]) -> None:
    if "cases" in packet or "case_packet_artifacts" in packet:
        raise ValueError("Case packet must not contain another case")
    _validate_common_packet(packet)
    case = packet.get("case")
    if not isinstance(case, dict):
        raise ValueError("Case packet must contain one case object")
    _validate_case(case)


def parse_single_response(raw_response: str) -> dict[str, Any]:
    payload = json.loads(raw_response.strip())
    if not isinstance(payload, dict) or set(payload) != RESPONSE_KEYS:
        raise ValueError("Judge response must be the frozen single JSON object")
    return payload


def validate_single_response(
    case_packet: dict[str, Any], response: dict[str, Any]
) -> None:
    validate_case_packet(case_packet)
    if not isinstance(response, dict) or set(response) != RESPONSE_KEYS:
        raise ValueError("Judge response fields differ from frozen schema")
    case = case_packet["case"]
    if response["case_id"] != case["case_id"]:
        raise ValueError("Judge response must answer the packet's only case")
    selected = str(response["selected_channel"])
    if selected not in SELECTIONS:
        raise ValueError(f"Unknown selected channel: {selected}")
    if not str(response["rationale"]).strip():
        raise ValueError("Judge rationale must not be empty")
    if selected == "abstain":
        if response["trace_id"] is not None or response["node_id"] is not None:
            raise ValueError("Abstention requires null trace_id and node_id")
        return
    lane = case["lanes"][selected]
    if response["trace_id"] != lane["trace_id"]:
        raise ValueError("Selected trace does not belong to selected channel")
    if response["node_id"] not in {hit["node_id"] for hit in lane["hits"]}:
        raise ValueError("Selected node does not belong to selected trace")


def capture_single_response(
    case_packet_path: str | Path,
    raw_response: str,
    *,
    judge_id: str,
    model: str,
    agent_type: str,
    executed_at: str | None = None,
) -> dict[str, Any]:
    case_packet_path = Path(case_packet_path)
    case_packet = _read_json(case_packet_path)
    response = parse_single_response(raw_response)
    validate_single_response(case_packet, response)
    return {
        "schema_version": NODE_FIRST_SCHEMA_VERSION,
        "experiment_id": case_packet["experiment_id"],
        "stage": case_packet["stage"],
        "case_id": case_packet["case"]["case_id"],
        "case_packet_sha256": _byte_sha256(case_packet_path),
        "judge_id": _required_text(judge_id, "judge_id"),
        "model": _required_text(model, "model"),
        "agent_type": _required_text(agent_type, "agent_type"),
        "executed_at": _required_text(
            executed_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "executed_at",
        ),
        "raw_response": raw_response,
        "parsed_response": response,
    }


def preflight_node_first_capture(
    manifest_path: str | Path,
    stage_packet_path: str | Path,
    case_packet_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Verify the frozen stage and every referenced case packet before capture."""
    manifest_path = Path(manifest_path)
    stage_packet_path = Path(stage_packet_path)
    _, _, _, refs = _preflight_node_first_stage(manifest_path, stage_packet_path)
    if case_packet_path is None:
        return None

    root = _repo_root(manifest_path)
    requested_path = Path(case_packet_path).resolve()
    for case_id, ref in refs.items():
        if (root / str(ref["path"])).resolve() == requested_path:
            return _read_json(requested_path)
    raise ValueError("Case packet is not referenced by the frozen stage packet")


def aggregate_node_first_results(
    manifest_path: str | Path,
    stage_packet_path: str | Path,
    response_paths: Iterable[str | Path],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    stage_packet_path = Path(stage_packet_path)
    manifest, stage_packet, cases, refs = _preflight_node_first_stage(
        manifest_path, stage_packet_path
    )
    split_name = str(stage_packet["stage"])
    manifest_hash = _canonical_checkout_sha256(manifest_path)
    response_path_list = [Path(path) for path in response_paths]
    expected_count = int(manifest["case_count"]) * int(manifest["judges_per_case"])
    if len(response_path_list) != expected_count:
        raise ValueError("Node-first aggregation requires exactly twelve responses")
    artifacts = [_read_json(path) for path in response_path_list]
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    judge_ids: set[str] = set()
    for path, artifact in zip(response_path_list, artifacts, strict=True):
        if artifact.get("schema_version") != NODE_FIRST_SCHEMA_VERSION:
            raise ValueError("Unsupported response artifact schema version")
        if artifact.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError("Response artifact experiment ID differs from manifest")
        if artifact.get("stage") != split_name:
            raise ValueError("Response artifact stage differs from stage packet")
        case_id = str(artifact.get("case_id", ""))
        if case_id not in cases:
            raise ValueError("Response artifact has an unknown case ID")
        if artifact.get("case_packet_sha256") != refs[case_id]["sha256"]:
            raise ValueError("Response artifact differs from frozen case packet")
        judge_id = _required_text(str(artifact.get("judge_id", "")), "judge_id")
        if judge_id in judge_ids:
            raise ValueError("Every invocation must use a fresh judge ID")
        judge_ids.add(judge_id)
        for field in ("model", "agent_type", "executed_at"):
            _required_text(str(artifact.get(field, "")), field)
        parsed = artifact.get("parsed_response")
        if parse_single_response(str(artifact.get("raw_response", ""))) != parsed:
            raise ValueError("Judge raw response and parsed response differ")
        case_packet = _case_packet_from_stage(stage_packet, cases[case_id])
        validate_single_response(case_packet, parsed)
        grouped[case_id].append((path, artifact))
    if any(
        len(grouped[case_id]) != int(manifest["judges_per_case"])
        for case_id in cases
    ):
        raise ValueError("Each case must have exactly three fresh responses")

    root = _repo_root(manifest_path)
    v1_manifest_path = root / manifest["v1_manifest"]
    v1_manifest = _read_json(v1_manifest_path)
    gold_cases = _read_json(
        v1_manifest_path.parent / v1_manifest[split_name]["gold"]
    )["cases"]
    if len(gold_cases) != len(cases):
        raise ValueError("Stage and gold case counts differ")
    gold_by_id = {
        f"case-{index:04d}": case
        for index, case in enumerate(gold_cases, start=1)
    }
    relation_gold = gold_by_id[str(manifest["relation_case_id"])]
    frozen_relation_path = relation_gold.get("expected_path", [])
    evaluated = [
        _evaluate_case(
            cases[case_id],
            gold_by_id[case_id],
            grouped[case_id],
            frozen_relation_path,
        )
        for case_id in sorted(cases)
    ]
    all_votes = [vote for case in evaluated for vote in case["responses"]]
    relation_votes = [vote for vote in all_votes if vote["selected_channel"] == "relation"]
    correct_votes = [vote for vote in all_votes if vote["node_correct"]]
    correct_relation_votes = [
        vote
        for vote in correct_votes
        if vote["selected_channel"] == "relation"
    ]
    union_oracle = sum(
        any(
            hit["node_id"] == gold_by_id[case_id]["expected_node_id"]
            for lane in cases[case_id]["lanes"].values()
            for hit in lane["hits"]
        )
        for case_id in cases
    )
    gate = {
        "packet_and_prompt_exclude_answer_fields": True,
        "all_responses_valid_and_trace_bound": True,
        "node_majority_exists_for_every_case": all(case["has_node_majority"] for case in evaluated),
        "majority_selects_every_expected_node": all(case["node_correct"] for case in evaluated),
        "relation_task_selects_edge_target": next(
            case for case in evaluated if case["case_id"] == manifest["relation_case_id"]
        )["node_correct"],
        "selected_relation_paths_match": all(vote["path_matched"] for vote in relation_votes),
        "one_case_per_invocation": all(vote["single_case_valid"] for vote in all_votes),
        "abstain_is_never_majority": all(case["majority_node_id"] is not None for case in evaluated),
        "search_and_scoring_do_not_change_edges": all(case["edge_weight_unchanged"] for case in evaluated),
        "node_correctness_is_independent_of_channel": _has_no_gold_channel_gate(manifest),
        "v1_v2_artifacts_match_frozen_bytes": True,
        "observed_artifact_overwrite_is_refused": bool(manifest["stop_rule"]["refuse_overwrite"]),
    }
    majority_correct = sum(case["node_correct"] for case in evaluated)
    correct_channel_split = Counter(
        vote["selected_channel"] for vote in correct_votes
    )
    result = {
        "schema_version": NODE_FIRST_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": split_name,
        "manifest_canonical_sha256": manifest_hash,
        "stage_packet_sha256": _byte_sha256(stage_packet_path),
        "response_artifacts": [
            {
                "case_id": artifact["case_id"],
                "judge_id": artifact["judge_id"],
                "model": artifact["model"],
                "agent_type": artifact["agent_type"],
                "executed_at": artifact["executed_at"],
                "artifact_sha256": _byte_sha256(path),
            }
            for path, artifact in zip(response_path_list, artifacts, strict=True)
        ],
        "cases": evaluated,
        "metrics": {
            "judge_node_accuracy": [
                {"judge_id": vote["judge_id"], "correct": vote["node_correct"]}
                for vote in all_votes
            ],
            "majority_node_accuracy": majority_correct / len(evaluated),
            "node_majority_agreement": sum(case["unanimous_node"] for case in evaluated) / len(evaluated),
            "abstain_rate": sum(vote["selected_channel"] == "abstain" for vote in all_votes) / len(all_votes),
            "selected_node_mrr": sum(vote["reciprocal_rank"] for vote in all_votes) / len(all_votes),
            "lexical_trace_usage": sum(vote["selected_channel"] == "lexical" for vote in all_votes) / len(all_votes),
            "relation_trace_usage": len(relation_votes) / len(all_votes),
            "correct_node_channel_split": dict(sorted(correct_channel_split.items())),
            "path_backed_correct_node_rate": (
                sum(vote["path_matched"] for vote in correct_relation_votes)
                / len(correct_relation_votes)
                if correct_relation_votes
                else 0.0
            ),
            "union_oracle_coverage": union_oracle / len(evaluated),
            "union_oracle_gap": (union_oracle - majority_correct) / len(evaluated),
        },
        "gate": gate,
        "gate_passed": all(gate.values()),
    }
    if split_name == "development":
        result["holdout_status"] = (
            "eligible_for_single_stage"
            if result["gate_passed"]
            else "not_opened_no_candidate"
        )
    else:
        result["claim_scope"] = (
            "blind_node_first_selection_supported_on_frozen_minimal_holdout"
            if result["gate_passed"]
            else "no_support_on_frozen_minimal_holdout"
        )
    return result


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_json_document_bytes(payload))


def _evaluate_case(
    case: dict[str, Any],
    gold: dict[str, Any],
    artifact_pairs: list[tuple[Path, dict[str, Any]]],
    frozen_relation_path: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_node = str(gold["expected_node_id"])
    votes = []
    for _, artifact in artifact_pairs:
        response = artifact["parsed_response"]
        selected_channel = str(response["selected_channel"])
        selected_node = response["node_id"]
        selected_hit = None
        if selected_channel in CHANNELS:
            selected_hit = next(
                hit
                for hit in case["lanes"][selected_channel]["hits"]
                if hit["node_id"] == selected_node
            )
        projected_paths = [
            path["projected_steps"]
            for path in (selected_hit or {}).get("paths", [])
        ]
        path_matched = (
            selected_channel != "relation"
            or (
                bool(frozen_relation_path)
                and any(path == frozen_relation_path for path in projected_paths)
                and all(path for path in projected_paths)
            )
        )
        rank = int(selected_hit["rank"]) if selected_hit else None
        votes.append(
            {
                "judge_id": artifact["judge_id"],
                "selected_channel": selected_channel,
                "selected_trace_id": response["trace_id"],
                "selected_node_id": selected_node,
                "selected_rank": rank,
                "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
                "node_correct": selected_node == expected_node,
                "path_matched": path_matched,
                "projected_selected_paths": projected_paths,
                "single_case_valid": response["case_id"] == case["case_id"],
                "feedback_provenance": {
                    "channel": selected_channel,
                    "would_reinforce_edges": selected_channel == "relation",
                    "credited_path_available": bool(projected_paths),
                },
            }
        )
    node_counts = Counter(vote["selected_node_id"] for vote in votes)
    majority_node, majority_count = node_counts.most_common(1)[0]
    has_majority = majority_count >= 2 and majority_node is not None
    majority_node = majority_node if has_majority else None
    return {
        "case_id": case["case_id"],
        "cohort": gold["cohort"],
        "expected_node_id": expected_node,
        "node_vote_counts": [
            {"node_id": node_id, "count": count}
            for node_id, count in sorted(node_counts.items(), key=str)
        ],
        "channel_vote_counts": dict(
            sorted(Counter(vote["selected_channel"] for vote in votes).items())
        ),
        "has_node_majority": has_majority,
        "majority_node_id": majority_node,
        "node_correct": majority_node == expected_node,
        "unanimous_node": len(node_counts) == 1,
        "edge_weight_unchanged": bool(case["edge_weight_audit"]["unchanged"]),
        "responses": votes,
    }


def _case_packet_from_stage(
    stage_packet: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: stage_packet[key]
        for key in (
            "schema_version",
            "experiment_id",
            "stage",
            "protocol_manifest_canonical_sha256",
            "judge_prompt_canonical_sha256",
            "lane_semantics",
            "response_schema",
        )
    } | {"case": case}


def _preflight_node_first_stage(
    manifest_path: Path, stage_packet_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    manifest = read_node_first_manifest(manifest_path)
    stage_packet = _read_json(stage_packet_path)
    validate_stage_packet(stage_packet)
    manifest_hash = _canonical_checkout_sha256(manifest_path)
    if stage_packet.get("protocol_manifest_canonical_sha256") != manifest_hash:
        raise ValueError("Stage packet does not match node-first manifest")
    if stage_packet.get("experiment_id") != manifest["experiment_id"]:
        raise ValueError("Stage packet experiment ID differs from manifest")
    if stage_packet.get("judge_prompt_canonical_sha256") != manifest[
        "judge_prompt"
    ]["canonical_sha256"]:
        raise ValueError("Stage packet judge prompt differs from manifest")
    if stage_packet.get("stage") not in {"development", "holdout"}:
        raise ValueError("Unknown stage packet split")

    refs = {
        str(ref["case_id"]): ref for ref in stage_packet["case_packet_artifacts"]
    }
    cases = {str(case["case_id"]): case for case in stage_packet["cases"]}
    root = _repo_root(manifest_path)
    for case_id, case in cases.items():
        ref = refs[case_id]
        expected_packet = _case_packet_from_stage(stage_packet, case)
        expected_hash = _bytes_sha256(_json_document_bytes(expected_packet))
        if ref.get("sha256") != expected_hash:
            raise ValueError("Stage case packet reference hash is invalid")
        packet_path = root / str(ref["path"])
        if not packet_path.is_file():
            raise ValueError(f"Frozen case packet is missing: {ref['path']}")
        if _byte_sha256(packet_path) != ref["sha256"]:
            raise ValueError(
                f"Frozen case packet differs from stage reference: {ref['path']}"
            )
        validate_case_packet(_read_json(packet_path))
    return manifest, stage_packet, cases, refs


def _validate_common_packet(packet: dict[str, Any]) -> None:
    if packet.get("schema_version") != NODE_FIRST_SCHEMA_VERSION:
        raise ValueError("Unsupported node-first packet schema version")
    if packet.get("experiment_id") != "d1-liplus-node-first-blind-selection-v3":
        raise ValueError("Unexpected node-first packet experiment ID")
    _reject_forbidden_packet_keys(packet)


def _validate_case(case: dict[str, Any]) -> None:
    case_id = str(case.get("case_id", ""))
    if re.fullmatch(r"case-\d{4}", case_id) is None:
        raise ValueError("Node-first case ID must be opaque")
    lanes = case.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != CHANNELS:
        raise ValueError("Node-first case must contain both independent lanes")
    trace_ids = {str(lanes[channel].get("trace_id", "")) for channel in CHANNELS}
    if "" in trace_ids or len(trace_ids) != 2:
        raise ValueError("Node-first case traces must be independent")
    if not isinstance(case.get("edge_weight_audit", {}).get("unchanged"), bool):
        raise ValueError("Node-first case lacks edge audit")
    for channel in CHANNELS:
        hits = lanes[channel].get("hits")
        if not isinstance(hits, list):
            raise ValueError("Node-first lane hits must be an array")
        for rank, hit in enumerate(hits, start=1):
            if hit.get("rank") != rank or not hit.get("node_id"):
                raise ValueError("Node-first lane ranks must be contiguous")
            if channel == "lexical" and "paths" in hit:
                raise ValueError("Lexical hit must not carry graph paths")
            if channel == "relation":
                paths = hit.get("paths")
                if not isinstance(paths, list) or not paths:
                    raise ValueError("Relation hit must carry non-zero-hop paths")
                for path in paths:
                    raw = path.get("raw_steps")
                    projected = path.get("projected_steps")
                    if not raw or not projected or _project_path(raw) != projected:
                        raise ValueError("Relation path projection is invalid")


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


def _validate_prompt(prompt: str) -> None:
    lowered = prompt.casefold()
    forbidden = (
        "acceptable_rank",
        "cohort",
        "expected_node_id",
        "expected_path",
        "intended_channel",
        "prior_result",
        "channels-dev-",
        "channels-holdout-",
    )
    if any(value in lowered for value in forbidden):
        raise ValueError("Node-first prompt contains a forbidden answer field")
    if "packetには一つのcaseしかありません" not in prompt:
        raise ValueError("Node-first prompt must freeze single-case cardinality")
    if "laneをまたいで数値比較しない" not in prompt:
        raise ValueError("Node-first prompt must forbid cross-lane comparison")


def _audit_prior_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> None:
    root = _repo_root(manifest_path)
    entries = manifest.get("frozen_prior_artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Node-first manifest must freeze prior artifacts")
    versions = {entry.get("version") for entry in entries}
    if versions != {"v1", "v2"}:
        raise ValueError("Node-first manifest must freeze v1 and v2 artifacts")
    for entry in entries:
        if not _matches_frozen_text_bytes(
            root / entry["path"], str(entry["sha256"])
        ):
            raise ValueError(f"Frozen prior artifact mismatch: {entry['path']}")


def _observed_paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    paths = []
    for split_name in ("development", "holdout"):
        spec = manifest["artifact_paths"][split_name]
        paths.append(root / spec["stage_packet"])
        paths.append(root / spec["result"])
        for case_number in range(1, int(manifest["case_count"]) + 1):
            case_id = f"case-{case_number:04d}"
            paths.append(
                root / spec["case_packet_template"].format(case_id=case_id)
            )
            for judge_number in range(1, int(manifest["judges_per_case"]) + 1):
                paths.append(
                    root
                    / spec["response_template"].format(
                        case_id=case_id, judge_number=judge_number
                    )
                )
    return paths


def _has_no_gold_channel_gate(manifest: dict[str, Any]) -> bool:
    forbidden = {
        "lexical_controls_select_lexical",
        "relation_case_selects_relation",
        "majority_channel_matches_gold",
    }
    return not forbidden & set(manifest["gate"])


def _project_path(raw_steps: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "source_id": str(step["source_id"]),
            "target_id": str(step["target_id"]),
            "edge_type": str(step["edge_type"]),
        }
        for step in raw_steps
    ]


def _canonical_checkout_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if b"\r" not in raw:
        canonical = raw
    else:
        remaining = raw.replace(b"\r\n", b"")
        if b"\r" in remaining or b"\n" in remaining:
            raise ValueError(f"Mixed or bare-CR newlines are not canonical: {path}")
        canonical = raw.replace(b"\r\n", b"\n")
    return _bytes_sha256(canonical)


def _json_document_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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


def _required_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text

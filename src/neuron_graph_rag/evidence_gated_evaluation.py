from __future__ import annotations

import hashlib
import json
import os
import subprocess
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .evidence_feedback import EngineConfig, NeuronGraphRAG

PROTOCOL = "evidence-gated-feedback-v1"
FIXTURE_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.fixture.json")
GOLD_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.gold.json")
SCHEDULE_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.schedule.json")
GATE_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.gate.json")
AUDIT_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.audit.json")
MANIFEST_PATH = Path("tests/fixtures/evidence_gated_feedback_v1.manifest.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _flatten_strings(child)]
    return []


def identity_projection(
    value: Any,
    *,
    allowlist: set[str],
    forbidden_fragments: tuple[str, ...],
) -> dict[str, set[str]]:
    projected = {
        "case_ids": set(),
        "cluster_ids": set(),
        "node_ids": set(),
        "document_paths": set(),
        "source_urls": set(),
        "queries": set(),
        "credited_edges": set(),
    }

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        if {"source_id", "target_id", "edge_type"} <= set(item):
            projected["credited_edges"].add(
                "|".join(
                    (
                        str(item["source_id"]),
                        str(item["target_id"]),
                        str(item["edge_type"]),
                    )
                )
            )
        for key, child in item.items():
            lowered = key.casefold()
            if any(fragment in lowered for fragment in forbidden_fragments):
                continue
            if key in allowlist:
                values = _flatten_strings(child)
                if key == "case_id":
                    projected["case_ids"].update(values)
                elif key == "cluster_id":
                    projected["cluster_ids"].update(values)
                elif key in {"node_id", "node_ids", "source_id", "target_id"}:
                    projected["node_ids"].update(values)
                elif key in {"document_path", "document_paths", "path", "paths"}:
                    projected["document_paths"].update(map(_canonical_path, values))
                elif key in {"source_url", "source_urls"}:
                    projected["source_urls"].update(values)
                elif key in {"query", "queries"}:
                    projected["queries"].update(
                        " ".join(text.casefold().split()) for text in values
                    )
            visit(child)

    visit(value)
    return projected


def _merge_projection(
    target: dict[str, set[str]], source: dict[str, set[str]]
) -> None:
    for category, values in source.items():
        target[category].update(values)


def validate_identity_audit(
    repo_root: Path, fixture: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    allowlist = {str(key) for key in audit["identity_field_allowlist"]}
    forbidden = tuple(str(value).casefold() for value in audit["forbidden_field_fragments"])
    split_projection = {
        stage: identity_projection(
            split,
            allowlist=allowlist,
            forbidden_fragments=forbidden,
        )
        for stage, split in fixture["splits"].items()
    }
    cross_split_overlap = {
        category: sorted(
            split_projection["development"][category]
            & split_projection["holdout"][category]
        )
        for category in split_projection["development"]
        if split_projection["development"][category]
        & split_projection["holdout"][category]
    }
    if cross_split_overlap != audit["expected_cross_split_overlap"]:
        raise ValueError(f"Unexpected cross-split identity overlap: {cross_split_overlap}")

    prior = {category: set() for category in split_projection["development"]}
    prior_counts: dict[str, dict[str, int]] = {}
    for relative in audit["prior_identity_sources"]:
        path = repo_root / str(relative)
        projection = identity_projection(
            load_json(path),
            allowlist=allowlist,
            forbidden_fragments=forbidden,
        )
        _merge_projection(prior, projection)
        prior_counts[str(relative)] = {
            category: len(values)
            for category, values in projection.items()
            if values
        }
    current = {category: set() for category in prior}
    for projection in split_projection.values():
        _merge_projection(current, projection)
    prior_overlap = {
        category: sorted(current[category] & prior[category])
        for category in current
        if current[category] & prior[category]
    }
    if prior_overlap != audit["expected_prior_overlap"]:
        raise ValueError(f"Unexpected prior identity overlap: {prior_overlap}")
    return {
        "cross_split_overlap": cross_split_overlap,
        "prior_overlap": prior_overlap,
        "split_identity_counts": {
            stage: {
                category: len(values)
                for category, values in projection.items()
                if values
            }
            for stage, projection in split_projection.items()
        },
        "prior_identity_counts": prior_counts,
    }


def _validate_fixture(repo_root: Path, fixture: dict[str, Any]) -> None:
    if set(fixture["splits"]) != {"development", "holdout"}:
        raise ValueError("Fixture must contain development and holdout")
    for stage, split in fixture["splits"].items():
        cases = split.get("cases")
        if not isinstance(cases, list) or {case["role"] for case in cases} != {
            "headroom",
            "ceiling",
        }:
            raise ValueError(f"Unexpected case roles in {stage}")
        for case in cases:
            corpus_text = (repo_root / case["document_path"]).read_text(
                encoding="utf-8"
            )
            node_ids = {node["node_id"] for node in case["nodes"]}
            if len(node_ids) != len(case["nodes"]):
                raise ValueError(f"Duplicate node identity in {case['case_id']}")
            if not {
                case["seed_node_id"],
                case["target_node_id"],
                case["sibling_node_id"],
                case["direct_node_id"],
                case["directional_seed_node_id"],
                case["directional_target_node_id"],
            } <= node_ids:
                raise ValueError(f"Case node role is missing in {case['case_id']}")
            if any(node["text"] not in corpus_text for node in case["nodes"]):
                raise ValueError(f"Corpus text mismatch in {case['case_id']}")
            roles = {edge["role"] for edge in case["edges"]}
            if roles != {"credited", "sibling", "directional_negative"}:
                raise ValueError(f"Unexpected edge roles in {case['case_id']}")


def _validate_schedule(schedule: dict[str, Any], gold: dict[str, Any]) -> None:
    if schedule["checkpoints"] != [0, 1, 2, 3, 4, 10]:
        raise ValueError("Unexpected feedback checkpoints")
    if schedule["checkpoints"] != gold["checkpoints"]:
        raise ValueError("Schedule and gold checkpoints differ")
    variants = {
        item["name"]: (
            item["relation_feedback_evidence_quorum"],
            item["sibling_feedback_normalization"],
        )
        for item in schedule["variants"]
    }
    if variants != {
        "current": (1, 0.0),
        "evidence-only": (3, 0.0),
        "local-only": (1, 1.0),
        "combined": (3, 1.0),
    }:
        raise ValueError("Unexpected variant matrix")
    if schedule["determinism_replays"] != 2:
        raise ValueError("Determinism replay count must be two")


def validate_protocol(repo_root: Path) -> dict[str, Any]:
    manifest = load_json(repo_root / MANIFEST_PATH)
    fixture = load_json(repo_root / FIXTURE_PATH)
    gold = load_json(repo_root / GOLD_PATH)
    schedule = load_json(repo_root / SCHEDULE_PATH)
    gate = load_json(repo_root / GATE_PATH)
    audit = load_json(repo_root / AUDIT_PATH)
    for value in (manifest, fixture, gold, schedule, gate, audit):
        if value.get("protocol") != PROTOCOL:
            raise ValueError("Unexpected protocol identity")

    hashes: dict[str, str] = {}
    for relative, expected in sorted(manifest["registered_artifacts"].items()):
        path = repo_root / relative
        raw = path.read_bytes()
        raw.decode("utf-8", errors="strict")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Registered artifact hash mismatch: {relative}")
        hashes[relative] = actual

    source = manifest["evaluated_source"]
    source_path = repo_root / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise ValueError("Evaluated candidate source hash mismatch")
    _git(repo_root, "cat-file", "-e", f"{source['commit']}^{{commit}}")
    _git(repo_root, "merge-base", "--is-ancestor", source["commit"], "HEAD")

    _validate_fixture(repo_root, fixture)
    _validate_schedule(schedule, gold)
    if gate["hard_gates"] != manifest["hard_gates"]:
        raise ValueError("Manifest and gate hard-gate order differ")
    audit_result = validate_identity_audit(repo_root, fixture, audit)
    return {
        "manifest": manifest,
        "fixture": fixture,
        "gold": gold,
        "schedule": schedule,
        "gate": gate,
        "audit": audit_result,
        "artifact_hashes": hashes,
    }


def _edge_by_role(case: dict[str, Any], role: str) -> dict[str, Any]:
    return next(edge for edge in case["edges"] if edge["role"] == role)


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (edge["source_id"], edge["target_id"], edge["edge_type"])


def _rank(hits: Any, node_id: str) -> int | None:
    return next(
        (rank for rank, hit in enumerate(hits, start=1) if hit.node.node_id == node_id),
        None,
    )


def _make_engine(
    case: dict[str, Any], variant: dict[str, Any], schedule: dict[str, Any]
) -> NeuronGraphRAG:
    config = EngineConfig(
        **schedule["engine_config"],
        relation_feedback_evidence_quorum=variant[
            "relation_feedback_evidence_quorum"
        ],
        sibling_feedback_normalization=variant["sibling_feedback_normalization"],
    )
    engine = NeuronGraphRAG(config=config)
    for node in case["nodes"]:
        engine.add_document(
            node["node_id"],
            node["text"],
            metadata={
                "document_path": case["document_path"],
                "source_url": node["source_url"],
            },
        )
    for edge in case["edges"]:
        engine.add_edge(
            edge["source_id"],
            edge["target_id"],
            edge["edge_type"],
            weight=float(edge["weight"]),
        )
    return engine


def _churn(baseline: list[str], current: list[str]) -> int:
    return sum(
        left != right
        for left, right in zip_longest(baseline, current, fillvalue=None)
    )


def _run_checkpoint(
    case: dict[str, Any],
    variant: dict[str, Any],
    checkpoint: int,
    schedule: dict[str, Any],
) -> dict[str, Any]:
    credited = _edge_by_role(case, "credited")
    sibling = _edge_by_role(case, "sibling")
    directional = _edge_by_role(case, "directional_negative")
    with _make_engine(case, variant, schedule) as engine:
        initial = {
            role: engine.store.edge(*_edge_key(edge))
            for role, edge in (
                ("credited", credited),
                ("sibling", sibling),
                ("directional_negative", directional),
            )
        }
        events: list[dict[str, Any]] = []
        for index in range(checkpoint):
            timestamp = schedule["base_timestamp"] + index * schedule["feedback_interval"]
            trace = engine.search_channels(
                case["queries"]["relation"],
                limit=schedule["limit"],
                now=timestamp,
            ).relation
            target_hit = next(
                hit for hit in trace.hits if hit.node.node_id == case["target_node_id"]
            )
            selected_path = max(
                target_hit.paths,
                key=lambda path: (path.contribution, path.seed_id),
            )
            projected = [
                (step.source_id, step.target_id, step.edge_type)
                for step in selected_path.steps
            ]
            receipt = engine.record_success(
                trace.trace_id,
                [case["target_node_id"]],
                now=timestamp + 1.0,
            )
            evidence = next(
                (
                    item
                    for item in receipt.evidence
                    if (item.source_id, item.target_id, item.edge_type)
                    == _edge_key(credited)
                ),
                None,
            )
            reinforced_delta = sum(
                item.new_weight - item.old_weight for item in receipt.reinforced_edges
            )
            sibling_delta = sum(
                item.old_weight - item.new_weight
                for item in receipt.normalized_sibling_edges
            )
            events.append(
                {
                    "event": index + 1,
                    "trace_channel": receipt.channel,
                    "trace_credit_matches": projected == [list(_edge_key(credited))]
                    or projected == [_edge_key(credited)],
                    "evidence_count": evidence.count if evidence else 0,
                    "quorum": evidence.quorum if evidence else 0,
                    "activated": evidence.activated if evidence else False,
                    "reinforced_delta": reinforced_delta,
                    "sibling_delta": sibling_delta,
                }
            )

        relation = engine.search_channels(
            case["queries"]["relation"],
            limit=schedule["limit"],
            now=schedule["evaluation_timestamp"],
        )
        direct = engine.search(
            case["queries"]["direct"],
            limit=schedule["limit"],
            now=schedule["evaluation_timestamp"] + 1.0,
        )
        negative = engine.search_channels(
            case["queries"]["directional_negative"],
            limit=schedule["limit"],
            now=schedule["evaluation_timestamp"] + 2.0,
        ).relation
        order = [hit.node.node_id for hit in relation.relation.hits]
        scores = {
            hit.node.node_id: hit.channel_score for hit in relation.relation.hits
        }
        recomputed_order = [
            node_id
            for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        final_edges = {
            role: engine.store.edge(*_edge_key(edge))
            for role, edge in (
                ("credited", credited),
                ("sibling", sibling),
                ("directional_negative", directional),
            )
        }
        changed_roles = sorted(
            role
            for role in final_edges
            if (
                final_edges[role].weight != initial[role].weight
                or final_edges[role].reinforced_count
                != initial[role].reinforced_count
            )
        )
        return {
            "feedback_count": checkpoint,
            "target_rank": _rank(relation.relation.hits, case["target_node_id"]),
            "relation_order": order,
            "relation_scores": scores,
            "score_margin": (
                scores[case["target_node_id"]]
                - scores.get(case["sibling_node_id"], 0.0)
            ),
            "controls": {
                "direct_rank": _rank(direct.hits, case["direct_node_id"]),
                "lexical_rank": _rank(
                    relation.lexical.hits, case["seed_node_id"]
                ),
                "directional_negative_rank": _rank(
                    negative.hits, case["directional_target_node_id"]
                ),
            },
            "edges": {
                role: {
                    "weight": edge.weight,
                    "reinforced_count": edge.reinforced_count,
                }
                for role, edge in final_edges.items()
            },
            "evidence_count": engine.store.feedback_evidence_count(*_edge_key(credited)),
            "feedback_rows": engine.store.count_feedback(),
            "events": events,
            "changed_edge_roles": changed_roles,
            "ordering_recomputed": recomputed_order == order,
            "rank_recomputed": (
                _rank(relation.relation.hits, case["target_node_id"])
                == order.index(case["target_node_id"]) + 1
            ),
        }


def _atomic_rollback_audit(stage: str) -> dict[str, bool]:
    prefix = f"q76-{stage}-atomic"
    config = EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=2,
        relation_feedback_evidence_quorum=1,
        sibling_feedback_normalization=1.0,
    )
    with NeuronGraphRAG(config=config) as engine:
        source, middle, target = (
            f"{prefix}-source",
            f"{prefix}-middle",
            f"{prefix}-target",
        )
        engine.add_document(source, f"{prefix}query")
        engine.add_document(middle, "atomic middle")
        engine.add_document(target, "atomic target")
        engine.add_edge(source, middle, "supports", weight=0.6)
        engine.add_edge(middle, target, "supports", weight=0.6)
        trace = engine.search_channels(f"{prefix}query", limit=5, now=100.0).relation
        before = engine.store.edge(source, middle, "supports")
        engine.store.connection.execute(
            "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
            (middle, target, "supports"),
        )
        engine.store.connection.commit()
        try:
            engine.record_success(trace.trace_id, [target], now=101.0)
        except KeyError:
            pass
        credited_rollback = (
            engine.store.edge(source, middle, "supports") == before
            and engine.store.count_feedback_evidence() == 0
            and engine.store.count_feedback() == 0
        )

    with NeuronGraphRAG(config=config) as engine:
        source, target = f"{prefix}-sibling-source", f"{prefix}-sibling-target"
        engine.add_document(source, f"{prefix}siblingquery")
        engine.add_document(target, "atomic sibling target")
        engine.add_edge(source, target, "supports", weight=0.6)
        trace = engine.search_channels(
            f"{prefix}siblingquery", limit=5, now=200.0
        ).relation
        before = engine.store.edge(source, target, "supports")
        try:
            engine.store.apply_evidence_gated_success_feedback(
                f"{prefix}-feedback",
                trace.trace_id,
                201.0,
                [target],
                [(source, target, "supports", 0.1, 2.0)],
                [(source, ((source, f"{prefix}-missing", "supports"),), 1.0)],
                evidence_quorum=1,
            )
        except KeyError:
            pass
        sibling_rollback = (
            engine.store.edge(source, target, "supports") == before
            and engine.store.count_feedback_evidence() == 0
            and engine.store.count_feedback() == 0
        )
    return {
        "credited_failure_rollback": credited_rollback,
        "sibling_failure_rollback": sibling_rollback,
    }


def _variant(case: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return case["variants"][name]


def _checkpoint(case: dict[str, Any], name: str, count: int) -> dict[str, Any]:
    return next(
        item for item in _variant(case, name) if item["feedback_count"] == count
    )


def _evaluate_gates(
    cases: list[dict[str, Any]], atomicity: dict[str, bool], gold: dict[str, Any]
) -> dict[str, bool]:
    headroom = next(case for case in cases if case["role"] == "headroom")
    ceiling = next(case for case in cases if case["role"] == "ceiling")
    all_points = [point for case in cases for points in case["variants"].values() for point in points]
    trace_credit = all(
        event["trace_channel"] == "relation" and event["trace_credit_matches"]
        for point in all_points
        for event in point["events"]
    )
    combined_pre = all(
        (
            _checkpoint(case, "combined", count)["evidence_count"] == count
            and _checkpoint(case, "combined", count)["edges"]
            == _checkpoint(case, "combined", 0)["edges"]
            and _checkpoint(case, "combined", count)["target_rank"]
            == _checkpoint(case, "combined", 0)["target_rank"]
            and _checkpoint(case, "combined", count)["controls"]
            == _checkpoint(case, "combined", 0)["controls"]
        )
        for case in cases
        for count in (1, 2)
    )
    delayed_flip = (
        _checkpoint(headroom, "current", 1)["target_rank"] == 1
        and _checkpoint(headroom, "local-only", 1)["target_rank"] == 1
        and _checkpoint(headroom, "combined", 0)["target_rank"] == 2
        and _checkpoint(headroom, "combined", 1)["target_rank"] == 2
        and _checkpoint(headroom, "combined", 2)["target_rank"] == 2
        and _checkpoint(headroom, "combined", 3)["target_rank"] == 1
    )
    current_mrr = sum(
        1.0 / _checkpoint(case, "current", 10)["target_rank"] for case in cases
    ) / len(cases)
    combined_mrr = sum(
        1.0 / _checkpoint(case, "combined", 10)["target_rank"] for case in cases
    ) / len(cases)
    final_non_regression = (
        combined_mrr >= current_mrr
        and all(
            _checkpoint(ceiling, name, count)["target_rank"] == 1
            for name in ceiling["variants"]
            for count in gold["checkpoints"]
        )
    )
    combined_churn = sum(
        _checkpoint(headroom, "combined", count)["top_k_churn"]
        for count in (1, 2)
    )
    current_churn = sum(
        _checkpoint(headroom, "current", count)["top_k_churn"]
        for count in (1, 2)
    )
    controls_stable = all(
        point["controls"] == points[0]["controls"]
        for case in cases
        for points in case["variants"].values()
        for point in points
    )
    pre_quorum_churn = combined_churn < current_churn and controls_stable

    evidence_cp3 = _checkpoint(headroom, "evidence-only", 3)
    combined_cp3 = _checkpoint(headroom, "combined", 3)
    local_cp1 = _checkpoint(headroom, "local-only", 1)
    ablation = (
        [event["activated"] for event in evidence_cp3["events"]]
        == [False, False, True]
        and [event["activated"] for event in combined_cp3["events"]]
        == [False, False, True]
        and _checkpoint(headroom, "local-only", 1)["events"][0]["activated"]
        and combined_cp3["events"][-1]["reinforced_delta"]
        == evidence_cp3["events"][-1]["reinforced_delta"]
        and combined_cp3["events"][-1]["sibling_delta"]
        == combined_cp3["events"][-1]["reinforced_delta"]
        and local_cp1["events"][0]["sibling_delta"]
        == local_cp1["events"][0]["reinforced_delta"]
    )
    mutation_scope = True
    for case in cases:
        for name, points in case["variants"].items():
            for point in points:
                activated = any(event["activated"] for event in point["events"])
                expected = []
                if activated:
                    expected.append("credited")
                    if name in {"local-only", "combined"}:
                        expected.append("sibling")
                mutation_scope &= point["changed_edge_roles"] == sorted(expected)
                mutation_scope &= point["edges"]["directional_negative"] == points[0]["edges"]["directional_negative"]
    deterministic = all(
        point["deterministic_replay_match"]
        and point["ordering_recomputed"]
        and point["rank_recomputed"]
        and point["top_k_churn_recomputed"]
        for point in all_points
    )
    return {
        "protocol_integrity": True,
        "trace_credit": trace_credit,
        "combined_pre_quorum_stability": combined_pre,
        "delayed_headroom_flip": delayed_flip,
        "final_relation_non_regression": final_non_regression,
        "pre_quorum_churn_reduction": pre_quorum_churn,
        "ablation_consistency": ablation,
        "mutation_scope": bool(mutation_scope),
        "atomic_rollback": all(atomicity.values()),
        "deterministic_replay": deterministic,
    }


def evaluate_stage(repo_root: Path, stage: str) -> dict[str, Any]:
    validated = validate_protocol(repo_root)
    if stage not in {"development", "holdout"}:
        raise ValueError("stage must be development or holdout")
    if stage == "holdout":
        development_path = repo_root / validated["manifest"]["exclusive_outputs"][
            "development"
        ]
        if not development_path.is_file() or not load_json(development_path).get(
            "all_pass"
        ):
            raise ValueError("Holdout requires a passing development output")

    cases: list[dict[str, Any]] = []
    schedule = validated["schedule"]
    for case in validated["fixture"]["splits"][stage]["cases"]:
        case_output = {
            "case_id": case["case_id"],
            "cluster_id": case["cluster_id"],
            "role": case["role"],
            "variants": {},
        }
        for variant in schedule["variants"]:
            points: list[dict[str, Any]] = []
            for checkpoint in schedule["checkpoints"]:
                replays = [
                    _run_checkpoint(case, variant, checkpoint, schedule)
                    for _ in range(schedule["determinism_replays"])
                ]
                point = replays[0]
                point["deterministic_replay_match"] = replays[0] == replays[1]
                points.append(point)
            baseline_order = points[0]["relation_order"]
            for point in points:
                point["top_k_churn"] = _churn(
                    baseline_order, point["relation_order"]
                )
                point["top_k_churn_recomputed"] = point["top_k_churn"] == _churn(
                    baseline_order, point["relation_order"]
                )
            case_output["variants"][variant["name"]] = points
        cases.append(case_output)

    atomicity = _atomic_rollback_audit(stage)
    gates = _evaluate_gates(cases, atomicity, validated["gold"])
    expected_gate_order = validated["gate"]["hard_gates"]
    if list(gates) != expected_gate_order:
        raise ValueError("Evaluator gate order differs from the registered gate")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": stage,
        "run_count": 1,
        "evaluated_source": validated["manifest"]["evaluated_source"],
        "registered_artifact_hashes": validated["artifact_hashes"],
        "identity_audit": validated["audit"],
        "cases": cases,
        "atomicity": atomicity,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": validated["manifest"]["claim_boundary"],
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def run_and_write_stage(repo_root: Path, stage: str) -> dict[str, Any]:
    validated = validate_protocol(repo_root)
    relative = validated["manifest"]["exclusive_outputs"][stage]
    output_path = repo_root / relative
    if output_path.exists():
        raise FileExistsError(f"Observed output already exists: {relative}")
    result = evaluate_stage(repo_root, stage)
    write_json_exclusive(output_path, result)
    return result


def validate_observed_outputs(repo_root: Path) -> dict[str, str]:
    validated = validate_protocol(repo_root)
    outputs = validated["manifest"]["exclusive_outputs"]
    states: dict[str, str] = {}
    development: dict[str, Any] | None = None
    for stage in ("development", "holdout"):
        path = repo_root / outputs[stage]
        if not path.exists():
            states[stage] = "absent"
            continue
        value = load_json(path)
        if (
            value.get("protocol") != PROTOCOL
            or value.get("stage") != stage
            or value.get("run_count") != 1
        ):
            raise ValueError(f"Invalid observed output identity: {stage}")
        if list(value.get("gates", {})) != validated["gate"]["hard_gates"]:
            raise ValueError(f"Observed gate order differs: {stage}")
        states[stage] = "pass" if value.get("all_pass") else "fail"
        if stage == "development":
            development = value
    if states.get("holdout") != "absent" and (
        development is None or not development.get("all_pass")
    ):
        raise ValueError("Holdout exists without passing development")
    return states

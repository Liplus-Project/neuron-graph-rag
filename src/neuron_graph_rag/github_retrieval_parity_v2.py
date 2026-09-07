"""Result-free retrieval parity v2 protocol and append-only lifecycle audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .engine import EngineConfig, NeuronGraphRAG
from .exclusion_intent import (
    RetrievalIntent,
    apply_exclusion_intent,
    decompose_exclusion_intent,
)
from .github_source import GitHubDocument, GitHubSnapshot, index_github_snapshot
from .models import DocumentNode, SearchHit

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STEM = "github_retrieval_parity_v2"
PROTOCOL_ID = "github-rag-vs-ngr-retrieval-parity-v2"
STAGES = ("development", "holdout")
COHORTS = (
    "direct_lexical",
    "semantic_paraphrase",
    "relation_linked",
    "negative_control",
    "over_exclusion_control",
)
GATE_IDS = (
    "protocol-integrity",
    "source-provenance-integrity",
    "deterministic-replay",
    "direct-safety",
    "negative-safety",
    "over-exclusion-safety",
    "cohort-non-regression",
    "expected-source-completeness",
    "relation-path-completeness-non-regression",
    "source-path-explanation-integrity",
)
CLAIM_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "capture_sha256",
    "one_time_claim",
)
GATE_FIELDS = ("gate_id", "hard", "passed", "details")
CAPTURE_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "captured_at",
    "service",
    "tool",
    "cases",
)
CAPTURE_CASE_FIELDS = ("case_id", "request", "raw_search", "raw_stored_content")
RESULT_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "status",
    "failure_code",
    "capture_sha256",
    "protocol_hashes",
    "source",
    "request_contract",
    "cases",
    "cohorts",
    "deterministic_replay",
    "resources",
    "gates",
    "all_hard_gates_pass",
    "raw_github_rag_mcp_capture",
    "interpretation_ja",
)
RUNTIME_PATHS = (
    "src/neuron_graph_rag/dynamics.py",
    "src/neuron_graph_rag/engine.py",
    "src/neuron_graph_rag/exclusion_intent.py",
    "src/neuron_graph_rag/github_source.py",
    "src/neuron_graph_rag/judgments.py",
    "src/neuron_graph_rag/models.py",
    "src/neuron_graph_rag/ontology.py",
    "src/neuron_graph_rag/precision_control.py",
    "src/neuron_graph_rag/retrieval.py",
    "src/neuron_graph_rag/storage.py",
)

MANIFEST_PATH = FIXTURES / f"{STEM}.manifest.json"
CORPUS_PATH = FIXTURES / f"{STEM}.corpus.json"
QUERIES_PATH = FIXTURES / f"{STEM}.queries.json"
GOLD_PATH = FIXTURES / f"{STEM}.gold.json"
GATE_PATH = FIXTURES / f"{STEM}.gate.json"
CAPTURE_SCHEMA_PATH = FIXTURES / f"{STEM}.capture-schema.json"
CLAIM_SCHEMA_PATH = FIXTURES / f"{STEM}.claim-schema.json"
RESULT_SCHEMA_PATH = FIXTURES / f"{STEM}.result-schema.json"
AUDIT_PATH = FIXTURES / f"{STEM}.result-free-audit.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_protocol(
    root: Path = ROOT, *, require_result_free: bool = True
) -> dict[str, Any]:
    relatives = {
        "manifest": MANIFEST_PATH.relative_to(ROOT),
        "corpus": CORPUS_PATH.relative_to(ROOT),
        "queries": QUERIES_PATH.relative_to(ROOT),
        "gold": GOLD_PATH.relative_to(ROOT),
        "gate": GATE_PATH.relative_to(ROOT),
        "capture_schema": CAPTURE_SCHEMA_PATH.relative_to(ROOT),
        "claim_schema": CLAIM_SCHEMA_PATH.relative_to(ROOT),
        "result_schema": RESULT_SCHEMA_PATH.relative_to(ROOT),
        "audit": AUDIT_PATH.relative_to(ROOT),
    }
    paths = {name: root / relative for name, relative in relatives.items()}
    protocol: dict[str, Any] = {
        "root": root,
        "paths": paths,
        "manifest": read_json(paths["manifest"]),
        "corpus": GitHubSnapshot.read(paths["corpus"]),
        "queries": read_json(paths["queries"]),
        "gold": read_json(paths["gold"]),
        "gate": read_json(paths["gate"]),
        "capture_schema": read_json(paths["capture_schema"]),
        "claim_schema": read_json(paths["claim_schema"]),
        "result_schema": read_json(paths["result_schema"]),
        "audit": read_json(paths["audit"]),
    }
    validate_protocol(protocol)
    verify_frozen_artifacts(protocol, require_result_free=require_result_free)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    manifest = _mapping(protocol, "manifest")
    corpus = protocol["corpus"]
    queries = _mapping(protocol, "queries")
    gold = _mapping(protocol, "gold")
    gate = _mapping(protocol, "gate")
    if not isinstance(corpus, GitHubSnapshot):
        raise TypeError("corpus must be a GitHub snapshot")
    for artifact in (
        manifest,
        queries,
        gold,
        gate,
        _mapping(protocol, "capture_schema"),
        _mapping(protocol, "claim_schema"),
        _mapping(protocol, "result_schema"),
        _mapping(protocol, "audit"),
    ):
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("protocol_id mismatch")
    if manifest.get("issue") != 216 or manifest.get("phase") != "result-free-freeze-v2":
        raise ValueError("manifest issue or phase mismatch")

    expected_protocol_artifacts = {
        name: path.relative_to(ROOT).as_posix()
        for name, path in {
            "corpus": CORPUS_PATH,
            "queries": QUERIES_PATH,
            "gold": GOLD_PATH,
            "gate": GATE_PATH,
            "capture_schema": CAPTURE_SCHEMA_PATH,
            "claim_schema": CLAIM_SCHEMA_PATH,
            "result_schema": RESULT_SCHEMA_PATH,
            "audit": AUDIT_PATH,
        }.items()
    }
    if dict(_mapping(manifest, "protocol_artifacts")) != expected_protocol_artifacts:
        raise ValueError("protocol artifact registry mismatch")
    if dict(_mapping(manifest, "lifecycle_contract")) != {
        "registry_policy": "append-only-exclusive-create",
        "stage_order": ["development", "conditional-holdout"],
        "failure_policy": "preserve-and-refuse-retry",
        "holdout_open_condition": "development-all-hard-gates-pass",
        "freeze_identity_scope": "protocol-artifacts-and-production-runtime",
        "observation_identity_scope": "registered-capture-claim-result",
    }:
        raise ValueError("lifecycle contract mismatch")
    capture_schema = _mapping(protocol, "capture_schema")
    claim_schema = _mapping(protocol, "claim_schema")
    result_schema = _mapping(protocol, "result_schema")
    if (
        tuple(capture_schema.get("top_level_fields", ())) != CAPTURE_FIELDS
        or tuple(capture_schema.get("case_fields", ())) != CAPTURE_CASE_FIELDS
        or tuple(claim_schema.get("fields", ())) != CLAIM_FIELDS
        or tuple(result_schema.get("top_level_fields", ())) != RESULT_FIELDS
        or result_schema.get("failure_codes")
        != ["hard-gate-failed", "execution-failed"]
    ):
        raise ValueError("capture, claim, or result schema mismatch")

    source = _mapping(manifest, "source")
    expected_source = {
        "repository": "Liplus-Project/liplus-language",
        "commit": "7a2d1a3a9e6fb821c836333900a9f1d145bd1832",
        "path_glob": "docs/*.md",
        "document_count": 20,
        "access": "public-read-only-github-snapshot-adapter",
        "generated_by": "tools/acquire_github_snapshot.py",
    }
    if {key: source.get(key) for key in expected_source} != expected_source:
        raise ValueError("frozen source contract mismatch")
    paths = source.get("paths")
    if (
        corpus.repository != source["repository"]
        or corpus.commit != source["commit"]
        or not isinstance(paths, list)
        or len(paths) != 20
        or sorted(paths) != [document.path for document in corpus.documents]
        or any(not re.fullmatch(r"docs/[^/]+\.md", path) for path in paths)
    ):
        raise ValueError("corpus does not match the frozen docs surface")
    _validate_snapshot_hashes(corpus)

    documents = {document.path: document for document in corpus.documents}
    identities = {
        _source_identity(corpus.repository, document.path)
        for document in corpus.documents
    }
    relationships = manifest.get("relationships")
    if not isinstance(relationships, list) or len(relationships) != 2:
        raise ValueError("exactly two frozen relation edges are required")
    relation_pairs: set[tuple[str, str]] = set()
    for item in relationships:
        row = _mapping_value(item, "relationship")
        source_id = _required_string(row, "source_id")
        target_id = _required_string(row, "target_id")
        if source_id not in identities or target_id not in identities:
            raise ValueError("relationship identity is outside the corpus")
        if row.get("edge_type") != "links_to":
            raise ValueError("relationship edge type mismatch")
        evidence = _required_string(row, "evidence_token")
        source_path = _path_from_source_id(corpus.repository, source_id)
        if evidence not in documents[source_path].content:
            raise ValueError("relationship evidence token is absent from source")
        relation_pairs.add((source_id, target_id))

    defaults = _mapping(queries, "request_defaults")
    if defaults != {
        "repo": corpus.repository,
        "type": "doc",
        "top_k": 10,
        "fusion": "rrf",
        "rerank": True,
        "graph_expand": True,
        "graph_hops": 2,
    }:
        raise ValueError("request defaults are not the frozen common request")
    if _mapping(queries, "ngr_engine_config") != asdict(EngineConfig()):
        raise ValueError("NGR engine config is not the production default")

    query_stages = _mapping(queries, "stages")
    gold_stages = _mapping(gold, "stages")
    stage_identities: dict[str, set[str]] = {}
    all_case_ids: set[str] = set()
    for stage in STAGES:
        cases = _list_value(query_stages, stage)
        rows = _list_value(gold_stages, stage)
        if len(cases) != len(COHORTS) or len(rows) != len(cases):
            raise ValueError("each stage must contain one case per cohort")
        gold_by_id = {
            _required_string(row, "case_id"): row
            for row in rows
            if isinstance(row, Mapping)
        }
        if len(gold_by_id) != len(rows):
            raise ValueError("gold case ids must be unique")
        stage_sources: set[str] = set()
        cohorts: list[str] = []
        for case in cases:
            if not isinstance(case, Mapping):
                raise TypeError("query cases must be objects")
            if set(case) != {"case_id", "cohort", "query"}:
                raise ValueError("query case fields mismatch")
            case_id = _required_string(case, "case_id")
            cohort = _required_string(case, "cohort")
            _required_string(case, "query")
            if case_id in all_case_ids or case_id not in gold_by_id:
                raise ValueError("query and gold case ids must be unique and aligned")
            all_case_ids.add(case_id)
            cohorts.append(cohort)
            row = gold_by_id[case_id]
            if row.get("cohort") != cohort:
                raise ValueError("query and gold cohort mismatch")
            allowed = {
                "case_id",
                "cohort",
                "expected_source_ids",
                "forbidden_source_ids",
                "protected_safe_source_ids",
            }
            if cohort == "relation_linked":
                allowed.add("relation_seed_source_id")
            if set(row) != allowed:
                raise ValueError("gold case fields mismatch")
            expected = _source_id_list(
                row, "expected_source_ids", identities, required=True
            )
            forbidden = _source_id_list(row, "forbidden_source_ids", identities)
            protected = _source_id_list(row, "protected_safe_source_ids", identities)
            stage_sources.update(expected + forbidden + protected)
            if cohort == "negative_control" and not forbidden:
                raise ValueError("negative control requires a forbidden source")
            if cohort == "over_exclusion_control" and (
                not protected or not set(protected).issubset(expected)
            ):
                raise ValueError(
                    "over-exclusion control requires protected expected sources"
                )
            if cohort == "over_exclusion_control":
                intent = decompose_exclusion_intent(_required_string(case, "query"))
                if not intent.exclusion_clauses:
                    raise ValueError(
                        "over-exclusion control requires an explicit exclusion clause"
                    )
                for source_id in protected:
                    decision = _candidate_exclusion_decision(
                        source_id, intent, corpus, documents
                    )
                    if (
                        decision.get("accepted") is not True
                        or set(decision.get("negated_mentions", ()))
                        != set(intent.exclusion_clauses)
                        or decision.get("matched_exclusions") != []
                    ):
                        raise ValueError(
                            "protected-safe source lacks the frozen candidate-side-negation premise"
                        )
                for source_id in forbidden:
                    decision = _candidate_exclusion_decision(
                        source_id, intent, corpus, documents
                    )
                    if decision.get("accepted") is not False or not decision.get(
                        "matched_exclusions"
                    ):
                        raise ValueError(
                            "unsafe comparison source lacks a non-negated exclusion mention"
                        )
            if cohort == "relation_linked":
                seed = _required_string(row, "relation_seed_source_id")
                stage_sources.add(seed)
                if any((seed, target) not in relation_pairs for target in expected):
                    raise ValueError("relation gold does not match a frozen edge")
        if tuple(cohorts) != COHORTS:
            raise ValueError("cohort order mismatch")
        stage_identities[stage] = stage_sources
    if not stage_identities["development"].isdisjoint(stage_identities["holdout"]):
        raise ValueError("development and holdout source identities must be disjoint")

    predecessor_cases: set[str] = set()
    predecessor_paths: set[str] = set()
    predecessor_hashes: set[str] = set()
    predecessors = manifest.get("predecessors")
    if not isinstance(predecessors, list) or {
        row.get("name") for row in predecessors if isinstance(row, Mapping)
    } != {"retrieval-parity-v1", "source-grounded-relation-v3"}:
        raise ValueError("predecessor registry mismatch")
    root = Path(protocol["root"])
    for predecessor in predecessors:
        if not isinstance(predecessor, Mapping):
            raise TypeError("predecessor registry rows must be objects")
        for relative, expected_hash in _mapping(predecessor, "identity_sha256").items():
            if _sha256(root / str(relative)) != expected_hash:
                raise ValueError(f"predecessor identity drifted: {relative}")
            predecessor_paths.add(str(relative))
            predecessor_hashes.add(str(expected_hash))
        case_ids = predecessor.get("case_ids")
        if not isinstance(case_ids, list) or not all(
            isinstance(item, str) for item in case_ids
        ):
            raise ValueError("predecessor case identity registry mismatch")
        predecessor_cases.update(case_ids)
        for relative, expected_hash in _mapping(
            predecessor, "observation_sha256"
        ).items():
            if _sha256(root / str(relative)) != expected_hash:
                raise ValueError(f"predecessor observation drifted: {relative}")
            predecessor_paths.add(str(relative))
            predecessor_hashes.add(str(expected_hash))
    if all_case_ids & predecessor_cases:
        raise ValueError("v2 query identity overlaps a predecessor")
    runtime_hashes = _mapping(manifest, "runtime_sha256")
    if tuple(runtime_hashes) != RUNTIME_PATHS:
        raise ValueError("production search runtime registry mismatch")
    new_paths = set(_frozen_hashes(manifest)) | _registry_paths(manifest)
    if new_paths & predecessor_paths:
        raise ValueError("v2 artifact registry overlaps a predecessor")
    if set(_frozen_hashes(manifest).values()) & predecessor_hashes:
        raise ValueError("v2 artifact bytes reuse a predecessor artifact")

    gate_ids = tuple(
        _required_string(row, "gate_id")
        for row in _list_value(gate, "gates")
        if isinstance(row, Mapping)
    )
    if gate_ids != GATE_IDS or not all(
        row.get("hard") is True for row in gate["gates"]
    ):
        raise ValueError("hard gate order mismatch")
    expected_audit = {
        "protocol_id": PROTOCOL_ID,
        "phase": "result-free-freeze-v2",
        "development_query_execution_count": 0,
        "holdout_query_execution_count": 0,
        "development_capture_count": 0,
        "development_claim_count": 0,
        "development_result_count": 0,
        "holdout_capture_count": 0,
        "holdout_claim_count": 0,
        "holdout_result_count": 0,
        "predecessor_query_gold_capture_result_reuse_count": 0,
        "shared_database_open_count": 0,
        "feedback_or_outcome_connected": False,
        "performance": "not assessed",
    }
    if dict(_mapping(protocol, "audit")) != expected_audit:
        raise ValueError("result-free audit mismatch")


def verify_frozen_artifacts(
    protocol: Mapping[str, Any], *, require_result_free: bool = True
) -> None:
    root = Path(protocol["root"])
    manifest = _mapping(protocol, "manifest")
    artifacts = _frozen_hashes(manifest)
    if not artifacts:
        raise ValueError("frozen artifact hash registry must not be empty")
    for relative, expected in artifacts.items():
        if _sha256(root / str(relative)) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")
    if require_result_free:
        for relative in _registry_paths(manifest):
            if (root / relative).exists():
                raise ValueError(
                    "registered capture, claim, and result must be absent at freeze"
                )


def register_capture(
    stage: str, source: Path, protocol_commit: str, root: Path = ROOT
) -> Path:
    protocol = load_protocol(root, require_result_free=False)
    verify_protocol_commit(protocol_commit, protocol)
    _validate_stage(stage)
    _assert_stage_can_start(stage, protocol)
    raw = source.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, Mapping):
        raise TypeError("capture must be a JSON object")
    queries, _ = _stage_cases(protocol, stage)
    _validate_capture(value, stage, protocol_commit, protocol, queries)
    target = _stage_path(protocol, stage, "capture")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def verify_protocol_commit(protocol_commit: str, protocol: Mapping[str, Any]) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", protocol_commit) is None:
        raise ValueError("protocol_commit must be a full lowercase commit SHA")
    root = Path(protocol["root"])
    _git_bytes(root, f"{protocol_commit}^{{commit}}")
    branch_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "origin/main"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if branch_check.returncode != 0:
        raise ValueError("protocol_commit must be merged into origin/main")
    manifest_relative = MANIFEST_PATH.relative_to(ROOT).as_posix()
    parent = f"{protocol_commit}^1"
    _git_bytes(root, f"{parent}^{{commit}}")
    if (
        subprocess.run(
            ["git", "cat-file", "-e", f"{parent}:{manifest_relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    ):
        raise ValueError("protocol_commit must first introduce the v2 manifest")
    introductions = subprocess.run(
        [
            "git",
            "log",
            "--first-parent",
            "--diff-filter=A",
            "--format=%H",
            protocol_commit,
            "--",
            manifest_relative,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if introductions.returncode != 0 or introductions.stdout.splitlines() != [
        protocol_commit
    ]:
        raise ValueError("protocol_commit must uniquely introduce the v2 manifest")
    if (
        _git_bytes(root, f"{protocol_commit}:{manifest_relative}")
        != (root / manifest_relative).read_bytes()
    ):
        raise ValueError("running manifest drifted from the frozen merge commit")
    manifest = _mapping(protocol, "manifest")
    for relative, expected in _frozen_hashes(manifest).items():
        committed = hashlib.sha256(
            _git_bytes(root, f"{protocol_commit}:{relative}")
        ).hexdigest()
        if committed != expected or _sha256(root / str(relative)) != expected:
            raise ValueError(f"protocol commit artifact mismatch: {relative}")
    for relative in _registry_paths(manifest):
        if (
            subprocess.run(
                ["git", "cat-file", "-e", f"{protocol_commit}:{relative}"],
                cwd=root,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        ):
            raise ValueError("freeze commit must not contain observation artifacts")


def run_registered_stage(stage: str, protocol_commit: str, root: Path = ROOT) -> Path:
    protocol = load_protocol(root, require_result_free=False)
    verify_protocol_commit(protocol_commit, protocol)
    _validate_stage(stage)
    _assert_stage_execution_can_start(stage, protocol)
    capture_path = _stage_path(protocol, stage, "capture")
    capture_sha256 = _sha256(capture_path)
    claim_path = _stage_path(protocol, stage, "claim")
    write_json_exclusive(
        claim_path,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": protocol_commit,
            "stage": stage,
            "capture_sha256": capture_sha256,
            "one_time_claim": True,
        },
    )
    try:
        payload = _execute_stage(stage, protocol_commit, capture_sha256, protocol)
    except Exception as error:  # noqa: BLE001 - the immutable result records any failure
        payload = _failure_payload(
            stage, protocol_commit, capture_sha256, protocol, error
        )
    result_path = _stage_path(protocol, stage, "result")
    write_json_exclusive(result_path, payload)
    verify_registered_result(stage, root)
    return result_path


def verify_registered_result(stage: str, root: Path = ROOT) -> None:
    protocol = load_protocol(root, require_result_free=False)
    _verify_registered_result(stage, protocol)


def _verify_registered_result(stage: str, protocol: Mapping[str, Any]) -> None:
    _validate_stage(stage)
    claim = read_json(_stage_path(protocol, stage, "claim"))
    if tuple(claim) != CLAIM_FIELDS:
        raise ValueError("stage claim field order mismatch")
    capture_path = _stage_path(protocol, stage, "capture")
    capture_sha256 = _sha256(capture_path)
    if claim != {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": claim.get("protocol_commit"),
        "stage": stage,
        "capture_sha256": capture_sha256,
        "one_time_claim": True,
    }:
        raise ValueError("stage claim identity mismatch")
    protocol_commit = claim.get("protocol_commit")
    if not isinstance(protocol_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", protocol_commit
    ):
        raise ValueError("stage claim protocol_commit must be full lowercase hex")
    result = read_json(_stage_path(protocol, stage, "result"))
    verify_result_payload(result, _mapping(protocol, "result_schema"))
    if (
        result.get("protocol_commit") != protocol_commit
        or result.get("capture_sha256") != capture_sha256
    ):
        raise ValueError("result does not match the stage claim")
    manifest_hashes = _frozen_hashes(_mapping(protocol, "manifest"))
    if result.get("protocol_hashes") != manifest_hashes:
        raise ValueError("result protocol hashes do not match the manifest")
    capture = read_json(capture_path)
    if result.get("raw_github_rag_mcp_capture") != capture:
        raise ValueError("result raw capture does not match the registered capture")
    queries, gold = _stage_cases(protocol, stage)
    github_cases = _validate_capture(capture, stage, protocol_commit, protocol, queries)
    if result.get("failure_code") == "execution-failed":
        if result.get("cases") != [] or any(
            item.get("passed") for item in result["gates"]
        ):
            raise ValueError("execution failure must not carry successful evaluation")
        return
    cases = result.get("cases")
    if not isinstance(cases, list):
        raise TypeError("result cases must be a list")
    result_by_id = {
        str(case.get("case_id")): case for case in cases if isinstance(case, Mapping)
    }
    if len(result_by_id) != len(queries):
        raise ValueError("result case identity registry mismatch")
    recomputed_cases = []
    for query in queries:
        case_id = _required_string(query, "case_id")
        stored = result_by_id.get(case_id)
        if not isinstance(stored, Mapping):
            raise TypeError("result case is missing")
        ngr = _mapping_value(stored.get("ngr"), "result NGR metrics")
        hits = ngr.get("hits")
        if not isinstance(hits, list):
            raise TypeError("result NGR hits must be a list")
        _validate_ngr_hits(hits, protocol)
        recomputed_cases.append(
            _case_metrics(
                query,
                gold[case_id],
                github_cases[case_id],
                hits,
                _top_k(protocol),
            )
        )
    if cases != recomputed_cases:
        raise ValueError("result case metrics do not recompute from raw evidence")
    cohorts = _cohort_metrics(cases)
    if result.get("cohorts") != cohorts:
        raise ValueError("result cohort metrics do not recompute")
    deterministic = (
        _mapping_value(result.get("deterministic_replay"), "deterministic_replay").get(
            "passed"
        )
        is True
    )
    gates = _evaluate_gates(cases, cohorts, deterministic)
    if result.get("gates") != gates:
        raise ValueError("result gates do not recompute")


def verify_result_payload(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    if list(payload) != schema.get("top_level_fields"):
        raise ValueError("result top-level field order mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("stage") not in STAGES:
        raise ValueError("result protocol identity mismatch")
    if not isinstance(payload.get("protocol_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", str(payload["protocol_commit"])
    ):
        raise ValueError("result protocol_commit must be full lowercase hex")
    if not isinstance(payload.get("capture_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", str(payload["capture_sha256"])
    ):
        raise ValueError("result capture_sha256 must be full lowercase hex")
    gates = payload.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATE_IDS):
        raise ValueError("result hard gate order mismatch")
    for index, item in enumerate(gates):
        if not isinstance(item, Mapping) or tuple(item) != GATE_FIELDS:
            raise ValueError("result hard gate shape mismatch")
        if item.get("gate_id") != GATE_IDS[index] or item.get("hard") is not True:
            raise ValueError("result hard gate order mismatch")
        if type(item.get("passed")) is not bool or not isinstance(
            item.get("details"), Mapping
        ):
            raise ValueError("result hard gate values mismatch")
    all_pass = all(item["passed"] for item in gates)
    if payload.get("all_hard_gates_pass") is not all_pass:
        raise ValueError("result all_hard_gates_pass mismatch")
    status = payload.get("status")
    if status == "passed":
        if not all_pass or payload.get("failure_code") is not None:
            raise ValueError("passed result must pass every hard gate")
    elif status == "failed":
        if all_pass or payload.get("failure_code") not in schema.get(
            "failure_codes", []
        ):
            raise ValueError("failed result must fail with a known code")
    else:
        raise ValueError("result status must be passed or failed")


def _execute_stage(
    stage: str,
    protocol_commit: str,
    capture_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    capture = read_json(_stage_path(protocol, stage, "capture"))
    queries, gold = _stage_cases(protocol, stage)
    github_cases = _validate_capture(capture, stage, protocol_commit, protocol, queries)
    first, first_resources = _run_ngr_once(protocol, queries)
    second, second_resources = _run_ngr_once(protocol, queries)
    deterministic = first == second
    top_k = _top_k(protocol)
    cases = [
        _case_metrics(
            case,
            gold[str(case["case_id"])],
            github_cases[str(case["case_id"])],
            first[str(case["case_id"])],
            top_k,
        )
        for case in queries
    ]
    cohorts = _cohort_metrics(cases)
    gates = _evaluate_gates(cases, cohorts, deterministic)
    all_pass = all(item["passed"] for item in gates)
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "status": "passed" if all_pass else "failed",
        "failure_code": None if all_pass else "hard-gate-failed",
        "capture_sha256": capture_sha256,
        "protocol_hashes": _frozen_hashes(_mapping(protocol, "manifest")),
        "source": {
            "repository": corpus.repository,
            "commit": corpus.commit,
            "fingerprint": corpus.fingerprint(),
            "document_count": len(corpus.documents),
        },
        "request_contract": dict(
            _mapping(_mapping(protocol, "queries"), "request_defaults")
        ),
        "cases": cases,
        "cohorts": cohorts,
        "deterministic_replay": {
            "passed": deterministic,
            "normalized_result_sha256": _json_sha256(first),
        },
        "resources": {
            "latency_hard_gate": False,
            "first_replay": first_resources,
            "second_replay": second_resources,
        },
        "gates": gates,
        "all_hard_gates_pass": all_pass,
        "raw_github_rag_mcp_capture": capture,
        "interpretation_ja": "固定された共通document surfaceの比較結果。latencyとresourceは記録のみでhard gateではない。",
    }


def _failure_payload(
    stage: str,
    protocol_commit: str,
    capture_sha256: str,
    protocol: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "status": "failed",
        "failure_code": "execution-failed",
        "capture_sha256": capture_sha256,
        "protocol_hashes": _frozen_hashes(_mapping(protocol, "manifest")),
        "source": {"repository": corpus.repository, "commit": corpus.commit},
        "request_contract": dict(
            _mapping(_mapping(protocol, "queries"), "request_defaults")
        ),
        "cases": [],
        "cohorts": {},
        "deterministic_replay": {"passed": False, "not_evaluated": True},
        "resources": {"latency_hard_gate": False},
        "gates": [
            {
                "gate_id": gate_id,
                "hard": True,
                "passed": False,
                "details": {"not_evaluated": True},
            }
            for gate_id in GATE_IDS
        ],
        "all_hard_gates_pass": False,
        "raw_github_rag_mcp_capture": read_json(
            _stage_path(protocol, stage, "capture")
        ),
        "interpretation_ja": f"一回性stageは{type(error).__name__}で停止した。再実行しない。",
    }


def _run_ngr_once(
    protocol: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    config = EngineConfig(
        **dict(_mapping(_mapping(protocol, "queries"), "ngr_engine_config"))
    )
    started = time.perf_counter()
    with TemporaryDirectory(prefix="ngr-retrieval-parity-v2-") as directory:
        database = Path(directory) / "parity.sqlite"
        with NeuronGraphRAG(database, config=config) as engine:
            index_github_snapshot(engine, corpus)
            for item in _list_value(_mapping(protocol, "manifest"), "relationships"):
                relation = _mapping_value(item, "relationship")
                engine.add_edge(
                    _required_string(relation, "source_id"),
                    _required_string(relation, "target_id"),
                    _required_string(relation, "edge_type"),
                )
            results: dict[str, Any] = {}
            for case in cases:
                trace = engine.search(
                    _required_string(case, "query"), limit=_top_k(protocol), now=0.0
                )
                results[_required_string(case, "case_id")] = [
                    {
                        "rank": rank,
                        "source_id": hit.node.node_id,
                        "repository": hit.node.metadata["repository"],
                        "path": hit.node.metadata["path"],
                        "source_url": hit.node.metadata["source_url"],
                        "commit": hit.node.metadata["commit"],
                        "blob_sha": hit.node.metadata["blob_sha"],
                        "content_sha256": hit.node.metadata["content_sha256"],
                        "explanation": hit.explain(),
                    }
                    for rank, hit in enumerate(trace.hits, start=1)
                ]
        resources = {
            "elapsed_seconds": time.perf_counter() - started,
            "database_bytes": database.stat().st_size,
            "database_scope": "fresh-temporary-deleted-after-replay",
            "feedback_connected": False,
        }
    return results, resources


def _validate_ngr_hits(
    hits: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    if len(hits) > _top_k(protocol):
        raise ValueError("result NGR hit count exceeds frozen top_k")
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    documents = {document.path: document for document in corpus.documents}
    seen: set[str] = set()
    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, Mapping) or hit.get("rank") != rank:
            raise ValueError("result NGR ranks must be consecutive")
        source_id = _required_string(hit, "source_id")
        path = _path_from_source_id(corpus.repository, source_id)
        document = documents.get(path)
        if document is None or source_id in seen:
            raise ValueError("result NGR source identity mismatch")
        seen.add(source_id)
        if {
            "repository": hit.get("repository"),
            "path": hit.get("path"),
            "source_url": hit.get("source_url"),
            "commit": hit.get("commit"),
            "blob_sha": hit.get("blob_sha"),
            "content_sha256": hit.get("content_sha256"),
        } != {
            "repository": corpus.repository,
            "path": path,
            "source_url": document.source_url,
            "commit": corpus.commit,
            "blob_sha": document.blob_sha,
            "content_sha256": document.content_sha256,
        }:
            raise ValueError("result NGR source provenance mismatch")
        explanation = _mapping_value(hit.get("explanation"), "NGR explanation")
        if explanation.get("node_id") != source_id or not isinstance(
            explanation.get("paths"), list
        ):
            raise ValueError("result NGR explanation identity mismatch")


def _validate_capture(
    capture: Mapping[str, Any],
    stage: str,
    protocol_commit: str,
    protocol: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    schema = _mapping(protocol, "capture_schema")
    if list(capture) != schema.get("top_level_fields"):
        raise ValueError("capture top-level field order mismatch")
    if (
        capture.get("protocol_id") != PROTOCOL_ID
        or capture.get("stage") != stage
        or capture.get("protocol_commit") != protocol_commit
    ):
        raise ValueError("capture protocol identity mismatch")
    if capture.get("service") != "github-rag-mcp" or capture.get("tool") != "search":
        raise ValueError("capture must preserve github-rag-mcp search output")
    _required_string(capture, "captured_at")
    rows = capture.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("capture must contain every frozen case exactly once")
    expected_cases = {_required_string(case, "case_id"): case for case in cases}
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    documents = {document.path: document for document in corpus.documents}
    defaults = dict(_mapping(_mapping(protocol, "queries"), "request_defaults"))
    captured: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or list(row) != schema.get("case_fields"):
            raise ValueError("capture case field order mismatch")
        case_id = _required_string(row, "case_id")
        if case_id not in expected_cases or case_id in captured:
            raise ValueError("capture case identity mismatch or duplicate")
        if row.get("request") != {
            "query": expected_cases[case_id]["query"],
            **defaults,
        }:
            raise ValueError("capture request differs from the frozen common request")
        raw_search = _mapping_value(row.get("raw_search"), "raw_search")
        if (
            raw_search.get("mode") != "search"
            or raw_search.get("filters_unmatched") != []
        ):
            raise ValueError("github-rag-mcp search did not use a matched filter")
        results = raw_search.get("results")
        if not isinstance(results, list) or len(results) > _top_k(protocol):
            raise ValueError("raw search result count exceeds frozen top_k")
        vector_paths: dict[str, str] = {}
        keyword: list[dict[str, Any]] = []
        for rank, item in enumerate(results, start=1):
            if (
                not isinstance(item, Mapping)
                or item.get("repo") != corpus.repository
                or item.get("type") != "doc"
            ):
                raise ValueError("raw search result left the frozen document surface")
            path = _required_string(item, "doc_path")
            vector_id = _required_string(item, "vector_id")
            if path not in documents or vector_id in vector_paths:
                raise ValueError("raw search result identity mismatch")
            vector_paths[vector_id] = path
            keyword.append(
                {
                    "rank": rank,
                    "source_id": _source_identity(corpus.repository, path),
                    "raw": item,
                }
            )
        fetch = _mapping_value(row.get("raw_stored_content"), "raw_stored_content")
        if (
            fetch.get("mode") != "fetch"
            or fetch.get("content_source") != "index"
            or fetch.get("content_max_chars") != 8000
            or fetch.get("not_found") != []
        ):
            raise ValueError("stored-content capture provenance mismatch")
        fetched = fetch.get("results")
        if not isinstance(fetched, list):
            raise TypeError("stored-content results must be a list")
        fetched_by_id = {
            _required_string(item, "vector_id"): item
            for item in fetched
            if isinstance(item, Mapping)
        }
        if len(fetched_by_id) != len(fetched) or set(fetched_by_id) != set(
            vector_paths
        ):
            raise ValueError("stored-content capture must cover every keyword result")
        for vector_id, path in vector_paths.items():
            item = fetched_by_id[vector_id]
            expected_content = _indexed_content(path, documents[path].content)
            if (
                item.get("repo") != corpus.repository
                or item.get("type") != "doc"
                or item.get("doc_path") != path
                or item.get("content") != expected_content
            ):
                raise ValueError("stored-content source identity mismatch")
            if item.get("content_chars") != _js_length(expected_content):
                raise ValueError("stored-content character count mismatch")
            if item.get("content_truncated") is not (
                _js_length(path + "\n\n" + documents[path].content) >= 8000
            ):
                raise ValueError("stored-content truncation provenance mismatch")
        graph: list[dict[str, Any]] = []
        graph_results = raw_search.get("graph_results", [])
        if not isinstance(graph_results, list) or len(graph_results) > _top_k(protocol):
            raise ValueError("graph results exceed frozen top_k")
        for rank, item in enumerate(graph_results, start=1):
            if (
                not isinstance(item, Mapping)
                or item.get("repo") != corpus.repository
                or item.get("type") != "doc"
                or item.get("doc_path") not in documents
            ):
                raise ValueError("graph result left the frozen document surface")
            graph.append(
                {
                    "rank": rank,
                    "source_id": _source_identity(
                        corpus.repository, str(item["doc_path"])
                    ),
                    "raw": item,
                }
            )
        captured[case_id] = {
            "keyword": keyword,
            "graph": graph,
            "raw_search": raw_search,
            "raw_stored_content": fetch,
        }
    if set(captured) != set(expected_cases):
        raise ValueError("capture does not cover the frozen cases")
    return captured


def _case_metrics(
    case: Mapping[str, Any],
    gold: Mapping[str, Any],
    github: Mapping[str, Any],
    ngr_hits: Sequence[Mapping[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    cohort = _required_string(case, "cohort")
    expected = list(gold["expected_source_ids"])
    forbidden = list(gold["forbidden_source_ids"])
    protected = list(gold["protected_safe_source_ids"])
    github_rank = lambda source_id: _github_rank(github, source_id, cohort)
    ngr_rank = lambda source_id: _ngr_rank(ngr_hits, source_id, cohort)
    github_relation = (
        _relation_completeness_github(github, expected)
        if cohort == "relation_linked"
        else 1.0
    )
    ngr_relation = (
        _relation_completeness_ngr(ngr_hits, expected)
        if cohort == "relation_linked"
        else 1.0
    )
    return {
        "case_id": _required_string(case, "case_id"),
        "cohort": cohort,
        "query": _required_string(case, "query"),
        "expected_source_ids": expected,
        "forbidden_source_ids": forbidden,
        "protected_safe_source_ids": protected,
        "github_rag_mcp": _metrics(
            expected, forbidden, protected, github_rank, top_k, github_relation
        ),
        "ngr": {
            **_metrics(expected, forbidden, protected, ngr_rank, top_k, ngr_relation),
            "hits": list(ngr_hits),
        },
        "raw_github_rag_mcp": {
            "search": github["raw_search"],
            "stored_content": github["raw_stored_content"],
        },
    }


def _metrics(
    expected: Sequence[str],
    forbidden: Sequence[str],
    protected: Sequence[str],
    rank_for: Any,
    top_k: int,
    relation_path_completeness: float,
) -> dict[str, Any]:
    ranks = [rank_for(source_id) for source_id in expected]
    observed = [rank for rank in ranks if isinstance(rank, int) and rank <= top_k]
    first_rank = min(observed) if observed else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in observed)
    ideal = sum(
        1.0 / math.log2(index + 1) for index in range(1, min(len(expected), top_k) + 1)
    )
    return {
        "expected_ranks": ranks,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg_at_k": 0.0 if ideal == 0.0 else dcg / ideal,
        "recall_at_k": len(observed) / len(expected),
        "forbidden_hit": any(
            rank_for(source_id) is not None for source_id in forbidden
        ),
        "protected_safe_retained": all(
            rank_for(source_id) is not None for source_id in protected
        ),
        "relation_path_completeness": relation_path_completeness,
    }


def _cohort_metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["cohort"])].append(case)
    if set(grouped) != set(COHORTS):
        raise ValueError("result cohort registry mismatch")
    metrics = ("mrr", "ndcg_at_k", "recall_at_k", "relation_path_completeness")
    return {
        cohort: {
            retriever: {
                metric: sum(float(case[retriever][metric]) for case in grouped[cohort])
                / len(grouped[cohort])
                for metric in metrics
            }
            for retriever in ("github_rag_mcp", "ngr")
        }
        for cohort in COHORTS
    }


def _evaluate_gates(
    cases: Sequence[Mapping[str, Any]], cohorts: Mapping[str, Any], deterministic: bool
) -> list[dict[str, Any]]:
    by_cohort = {str(case["cohort"]): case for case in cases}
    compared = ("mrr", "ndcg_at_k", "recall_at_k")

    def non_regression(case: Mapping[str, Any]) -> bool:
        return all(
            float(case["ngr"][metric]) >= float(case["github_rag_mcp"][metric])
            for metric in compared
        )

    direct = by_cohort["direct_lexical"]
    negative = by_cohort["negative_control"]
    over = by_cohort["over_exclusion_control"]
    cohort_non_regression = all(
        float(cohorts[cohort]["ngr"][metric])
        >= float(cohorts[cohort]["github_rag_mcp"][metric])
        for cohort in COHORTS
        for metric in compared
    )
    complete = all(
        float(case[retriever]["recall_at_k"]) == 1.0
        for case in cases
        for retriever in ("github_rag_mcp", "ngr")
    )
    relation = by_cohort["relation_linked"]
    relation_non_regression = (
        float(relation["ngr"]["relation_path_completeness"])
        >= float(relation["github_rag_mcp"]["relation_path_completeness"])
        and float(relation["ngr"]["relation_path_completeness"]) == 1.0
    )
    explanations = all(
        all(
            hit.get("repository")
            and hit.get("path")
            and hit.get("commit")
            and hit.get("blob_sha")
            and hit.get("content_sha256")
            and isinstance(hit.get("explanation"), Mapping)
            for hit in case["ngr"]["hits"]
        )
        for case in cases
    )
    verdicts = {
        "protocol-integrity": True,
        "source-provenance-integrity": True,
        "deterministic-replay": deterministic,
        "direct-safety": non_regression(direct) and direct["ngr"]["recall_at_k"] == 1.0,
        "negative-safety": non_regression(negative)
        and not negative["github_rag_mcp"]["forbidden_hit"]
        and not negative["ngr"]["forbidden_hit"],
        "over-exclusion-safety": non_regression(over)
        and over["github_rag_mcp"]["protected_safe_retained"]
        and over["ngr"]["protected_safe_retained"]
        and not over["github_rag_mcp"]["forbidden_hit"]
        and not over["ngr"]["forbidden_hit"],
        "cohort-non-regression": cohort_non_regression,
        "expected-source-completeness": complete,
        "relation-path-completeness-non-regression": relation_non_regression,
        "source-path-explanation-integrity": explanations,
    }
    return [
        {"gate_id": gate_id, "hard": True, "passed": verdicts[gate_id], "details": {}}
        for gate_id in GATE_IDS
    ]


def audit_repository_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    protocol = load_protocol(root, require_result_free=False)
    development = _stage_registry(protocol, "development")
    holdout = _stage_registry(protocol, "holdout")
    if development["result"] is None and any(holdout.values()):
        raise ValueError("holdout opened before development completed")
    eligible = bool(
        development["result"]
        and development["result"].get("all_hard_gates_pass") is True
    )
    if development["result"] is not None and not eligible and any(holdout.values()):
        raise ValueError("holdout opened after development closed")
    if development["result"] is None:
        phase = (
            "development-claimed"
            if development["claim"]
            else "development-captured"
            if development["capture"]
            else "result-free"
        )
    elif not eligible:
        phase = "development-closed"
    elif holdout["result"] is not None:
        phase = "holdout-completed"
    elif holdout["claim"]:
        phase = "holdout-claimed"
    elif holdout["capture"]:
        phase = "holdout-captured"
    else:
        phase = "holdout-eligible"
    return {
        "status": "observation-registry-valid",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "development": {
            "capture": development["capture"],
            "claim": development["claim"],
            "result": development["result"] is not None,
            "all_hard_gates_pass": eligible,
        },
        "holdout": {
            "eligible": eligible,
            "capture": holdout["capture"],
            "claim": holdout["claim"],
            "result": holdout["result"] is not None,
        },
    }


def audit_result_free(root: Path = ROOT) -> dict[str, Any]:
    state = audit_repository_lifecycle(root)
    if state["phase"] != "result-free":
        raise ValueError("repository observation registry is not result-free")
    protocol = load_protocol(root)
    corpus = protocol["corpus"]
    assert isinstance(corpus, GitHubSnapshot)
    return {
        "status": "result-free-protocol-valid",
        "protocol_id": PROTOCOL_ID,
        "source_document_count": len(corpus.documents),
        "development_case_count": len(
            _mapping(_mapping(protocol, "queries"), "stages")["development"]
        ),
        "holdout_case_count": len(
            _mapping(_mapping(protocol, "queries"), "stages")["holdout"]
        ),
        "registered_observation_count": 0,
        "performance": "not assessed",
    }


def _stage_registry(protocol: Mapping[str, Any], stage: str) -> dict[str, Any]:
    capture_path = _stage_path(protocol, stage, "capture")
    claim_path = _stage_path(protocol, stage, "claim")
    result_path = _stage_path(protocol, stage, "result")
    capture = capture_path.exists()
    claim = claim_path.exists()
    result = read_json(result_path) if result_path.exists() else None
    if claim and not capture:
        raise ValueError("stage claim exists without capture")
    if result is not None and not claim:
        raise ValueError("stage result exists without claim")
    if result is not None:
        _verify_registered_result(stage, protocol)
    return {"capture": capture, "claim": claim, "result": result}


def protocol_file_inventory(root: Path = ROOT) -> tuple[str, ...]:
    manifest = read_json(root / MANIFEST_PATH.relative_to(ROOT))
    paths = {
        MANIFEST_PATH.relative_to(ROOT).as_posix(),
        *map(str, _frozen_hashes(manifest)),
    }
    for predecessor in manifest.get("predecessors", []):
        paths.update(map(str, _mapping(predecessor, "identity_sha256")))
        paths.update(map(str, _mapping(predecessor, "observation_sha256")))
    return tuple(sorted(paths))


def _assert_stage_can_start(stage: str, protocol: Mapping[str, Any]) -> None:
    if any(
        _stage_path(protocol, stage, kind).exists()
        for kind in ("capture", "claim", "result")
    ):
        raise FileExistsError(f"registered {stage} stage already exists")
    if stage == "holdout":
        development = _stage_path(protocol, "development", "result")
        if not development.exists():
            raise RuntimeError("holdout is closed until development result exists")
        _verify_registered_result("development", protocol)
        if read_json(development).get("all_hard_gates_pass") is not True:
            raise RuntimeError(
                "holdout is closed because development hard gates did not pass"
            )


def _assert_stage_execution_can_start(stage: str, protocol: Mapping[str, Any]) -> None:
    if not _stage_path(protocol, stage, "capture").exists():
        raise FileNotFoundError("registered capture is required before stage execution")
    if (
        _stage_path(protocol, stage, "claim").exists()
        or _stage_path(protocol, stage, "result").exists()
    ):
        raise FileExistsError(f"registered {stage} stage is already claimed")
    if stage == "holdout":
        development = _stage_path(protocol, "development", "result")
        _verify_registered_result("development", protocol)
        if read_json(development).get("all_hard_gates_pass") is not True:
            raise RuntimeError(
                "holdout is closed because development hard gates did not pass"
            )


def _stage_cases(
    protocol: Mapping[str, Any], stage: str
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    queries = [
        row
        for row in _list_value(_mapping(_mapping(protocol, "queries"), "stages"), stage)
        if isinstance(row, Mapping)
    ]
    gold = {
        _required_string(row, "case_id"): row
        for row in _list_value(_mapping(_mapping(protocol, "gold"), "stages"), stage)
        if isinstance(row, Mapping)
    }
    return queries, gold


def _github_rank(results: Mapping[str, Any], source_id: str, cohort: str) -> int | None:
    ranks = [
        int(row["rank"]) for row in results["keyword"] if row["source_id"] == source_id
    ]
    if cohort == "relation_linked":
        ranks.extend(
            int(row["rank"])
            for row in results["graph"]
            if row["source_id"] == source_id
        )
    return min(ranks) if ranks else None


def _ngr_rank(
    hits: Sequence[Mapping[str, Any]], source_id: str, cohort: str
) -> int | None:
    ranks: list[int] = []
    for hit in hits:
        if hit.get("source_id") != source_id:
            continue
        ranks.append(int(hit["rank"]))
        if cohort == "relation_linked" and isinstance(hit.get("explanation"), Mapping):
            explanation = hit["explanation"]
            graph_rank = (
                explanation.get("ranks", {}).get("graph")
                if isinstance(explanation.get("ranks"), Mapping)
                else None
            )
            if isinstance(graph_rank, int):
                ranks.append(graph_rank)
    return min(ranks) if ranks else None


def _relation_completeness_github(
    results: Mapping[str, Any], expected: Sequence[str]
) -> float:
    graph_ids = {row["source_id"] for row in results["graph"]}
    return sum(source_id in graph_ids for source_id in expected) / len(expected)


def _relation_completeness_ngr(
    hits: Sequence[Mapping[str, Any]], expected: Sequence[str]
) -> float:
    completed: set[str] = set()
    for hit in hits:
        source_id = hit.get("source_id")
        explanation = hit.get("explanation")
        if source_id in expected and isinstance(explanation, Mapping):
            paths = explanation.get("paths")
            if isinstance(paths, list) and any(
                isinstance(path, Mapping) and bool(path.get("steps")) for path in paths
            ):
                completed.add(str(source_id))
    return len(completed) / len(expected)


def _registry_paths(manifest: Mapping[str, Any]) -> set[str]:
    return {
        _required_string(_mapping_value(stage_outputs, "stage outputs"), kind)
        for stage_outputs in _mapping(manifest, "outputs").values()
        for kind in ("capture", "claim", "result")
    }


def _frozen_hashes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = dict(_mapping(manifest, "artifact_sha256"))
    runtime = dict(_mapping(manifest, "runtime_sha256"))
    overlap = set(artifacts) & set(runtime)
    if overlap:
        raise ValueError(f"artifact and runtime hash registries overlap: {overlap}")
    return {**artifacts, **runtime}


def _candidate_exclusion_decision(
    source_id: str,
    intent: RetrievalIntent,
    corpus: GitHubSnapshot,
    documents: Mapping[str, GitHubDocument],
) -> Mapping[str, Any]:
    path = _path_from_source_id(corpus.repository, source_id)
    document = documents[path]
    hit = SearchHit(
        node=DocumentNode(
            source_id,
            _indexed_content(path, document.content),
            {
                "repository": corpus.repository,
                "path": path,
                "commit": corpus.commit,
            },
        ),
        sparse_score=0.0,
        dense_score=0.0,
        entry_score=0.0,
        graph_activation=0.0,
        final_score=0.0,
    )
    annotated, _ = apply_exclusion_intent((hit,), intent)
    decision = annotated[0].exclusion_intent
    if not isinstance(decision, Mapping):
        raise TypeError("candidate exclusion decision is absent")
    return decision


def _stage_path(protocol: Mapping[str, Any], stage: str, kind: str) -> Path:
    manifest = _mapping(protocol, "manifest")
    outputs = _mapping_value(
        _mapping(manifest, "outputs").get(stage), f"outputs.{stage}"
    )
    return Path(protocol["root"]) / _required_string(outputs, kind)


def _validate_snapshot_hashes(snapshot: GitHubSnapshot) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", snapshot.commit) is None:
        raise ValueError("snapshot commit must be full lowercase SHA-1")
    for document in snapshot.documents:
        if (
            re.fullmatch(r"[0-9a-f]{40}", document.blob_sha) is None
            or re.fullmatch(r"[0-9a-f]{64}", document.content_sha256) is None
        ):
            raise ValueError("snapshot document hashes must be full lowercase hex")


def _source_id_list(
    container: Mapping[str, Any],
    key: str,
    identities: set[str],
    *,
    required: bool = False,
) -> list[str]:
    value = container.get(key)
    if (
        not isinstance(value, list)
        or (required and not value)
        or any(not isinstance(item, str) or item not in identities for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{key} identity registry mismatch")
    return list(value)


def _path_from_source_id(repository: str, source_id: str) -> str:
    prefix = f"github:{repository}:"
    if not source_id.startswith(prefix):
        raise ValueError("source identity repository mismatch")
    return source_id[len(prefix) :]


def _source_identity(repository: str, path: str) -> str:
    return f"github:{repository}:{path}"


def _indexed_content(path: str, content: str) -> str:
    value = path + "\n\n" + content
    units = value.encode("utf-16-le")
    return units[: 8000 * 2].decode("utf-16-le", errors="ignore")


def _js_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _top_k(protocol: Mapping[str, Any]) -> int:
    value = _mapping(_mapping(protocol, "queries"), "request_defaults").get("top_k")
    if not isinstance(value, int) or value <= 5:
        raise ValueError(
            "top_k must be a frozen integer greater than the old Hit@5 ceiling"
        )
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_bytes(root: Path, object_name: str) -> bytes:
    result = subprocess.run(
        ["git", "show", object_name], cwd=root, check=False, capture_output=True
    )
    if result.returncode != 0:
        raise ValueError(f"git object is unavailable: {object_name}")
    return result.stdout


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")


def _mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping_value(container.get(key), key)


def _mapping_value(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _list_value(container: Mapping[str, Any], key: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _required_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value

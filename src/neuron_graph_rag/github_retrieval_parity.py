"""Result-free GitHub RAG versus NGR document retrieval parity protocol."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .engine import EngineConfig, NeuronGraphRAG
from .github_source import GitHubSnapshot, changed_paths, index_github_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STEM = "github_retrieval_parity_v1"
PROTOCOL_ID = "github-rag-vs-ngr-retrieval-parity-v1"
STAGES = ("development", "holdout")
COHORTS = (
    "direct_lexical",
    "semantic_paraphrase",
    "relation_linked",
    "negative_control",
)
GATE_IDS = (
    "protocol-integrity",
    "source-provenance-integrity",
    "deterministic-replay",
    "update-following",
    "direct-case-non-regression",
    "negative-control-non-regression",
    "cohort-mrr-non-regression",
    "cohort-hit-at-k-non-regression",
    "expected-source-top-k-completeness",
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

MANIFEST_PATH = FIXTURES / f"{STEM}.manifest.json"
CURRENT_CORPUS_PATH = FIXTURES / f"{STEM}.corpus.json"
PREVIOUS_CORPUS_PATH = FIXTURES / f"{STEM}.previous-corpus.json"
QUERIES_PATH = FIXTURES / f"{STEM}.queries.json"
GOLD_PATH = FIXTURES / f"{STEM}.gold.json"
GATE_PATH = FIXTURES / f"{STEM}.gate.json"
CAPTURE_SCHEMA_PATH = FIXTURES / f"{STEM}.capture-schema.json"
RESULT_SCHEMA_PATH = FIXTURES / f"{STEM}.result-schema.json"
AUDIT_PATH = FIXTURES / f"{STEM}.result-free-audit.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def register_capture(stage: str, source: Path, protocol_commit: str) -> Path:
    protocol = load_protocol()
    verify_protocol_commit(protocol_commit, protocol)
    _validate_stage(stage)
    _assert_stage_can_start(stage, protocol)
    raw = source.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, Mapping):
        raise ValueError("capture must be a JSON object")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("stage") != stage:
        raise ValueError("capture protocol_id and stage must match registration")
    if value.get("protocol_commit") != protocol_commit:
        raise ValueError("capture protocol_commit must match the frozen merge commit")
    target = _stage_path(protocol["manifest"], stage, "capture")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def load_protocol(root: Path = ROOT) -> dict[str, Any]:
    paths = {
        "manifest": root / MANIFEST_PATH.relative_to(ROOT),
        "current": root / CURRENT_CORPUS_PATH.relative_to(ROOT),
        "previous": root / PREVIOUS_CORPUS_PATH.relative_to(ROOT),
        "queries": root / QUERIES_PATH.relative_to(ROOT),
        "gold": root / GOLD_PATH.relative_to(ROOT),
        "gate": root / GATE_PATH.relative_to(ROOT),
        "capture_schema": root / CAPTURE_SCHEMA_PATH.relative_to(ROOT),
        "result_schema": root / RESULT_SCHEMA_PATH.relative_to(ROOT),
        "audit": root / AUDIT_PATH.relative_to(ROOT),
    }
    protocol: dict[str, Any] = {
        "root": root,
        "paths": paths,
        "manifest": read_json(paths["manifest"]),
        "current": GitHubSnapshot.read(paths["current"]),
        "previous": GitHubSnapshot.read(paths["previous"]),
        "queries": read_json(paths["queries"]),
        "gold": read_json(paths["gold"]),
        "gate": read_json(paths["gate"]),
        "capture_schema": read_json(paths["capture_schema"]),
        "result_schema": read_json(paths["result_schema"]),
        "audit": read_json(paths["audit"]),
    }
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    manifest = _mapping(protocol, "manifest")
    queries = _mapping(protocol, "queries")
    gold = _mapping(protocol, "gold")
    gate = _mapping(protocol, "gate")
    current = protocol["current"]
    previous = protocol["previous"]
    if not isinstance(current, GitHubSnapshot) or not isinstance(
        previous, GitHubSnapshot
    ):
        raise ValueError("current and previous corpora must be GitHub snapshots")
    for artifact in (
        manifest,
        queries,
        gold,
        gate,
        _mapping(protocol, "capture_schema"),
        _mapping(protocol, "result_schema"),
        _mapping(protocol, "audit"),
    ):
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("protocol_id mismatch")
    source = _mapping(manifest, "source")
    if current.repository != source.get("repository") or current.commit != source.get(
        "commit"
    ):
        raise ValueError("current corpus does not match frozen source")
    if previous.repository != current.repository or previous.commit != source.get(
        "previous_commit"
    ):
        raise ValueError("previous corpus does not match frozen update source")
    current_paths = [document.path for document in current.documents]
    previous_paths = [document.path for document in previous.documents]
    if current_paths != source.get("paths") or previous_paths != current_paths:
        raise ValueError("corpus path list is not the frozen complete surface")
    expected_changed = tuple(source.get("changed_paths", ()))
    if changed_paths(previous, current) != expected_changed:
        raise ValueError("frozen update changed_paths mismatch")
    _validate_snapshot_hashes(current)
    _validate_snapshot_hashes(previous)
    identities = {_source_identity(current.repository, path) for path in current_paths}
    relationships = manifest.get("relationships")
    if not isinstance(relationships, list) or not relationships:
        raise ValueError("frozen relationships must not be empty")
    relationship_tuples: set[tuple[str, str, str]] = set()
    for item in relationships:
        relation = _mapping_value(item, "relationship")
        source_id = _required_string(relation, "source_id")
        target_id = _required_string(relation, "target_id")
        relation_type = _required_string(relation, "edge_type")
        if source_id not in identities or target_id not in identities:
            raise ValueError("relationship identity is outside the corpus")
        relationship_tuples.add((source_id, target_id, relation_type))
    defaults = _mapping(queries, "request_defaults")
    if defaults != {
        "repo": current.repository,
        "type": "doc",
        "top_k": 5,
        "fusion": "rrf",
        "rerank": True,
        "graph_expand": True,
        "graph_hops": 2,
    }:
        raise ValueError("request defaults are not the frozen common request")
    query_stages = _mapping(queries, "stages")
    gold_stages = _mapping(gold, "stages")
    split_identities: dict[str, set[str]] = {}
    all_case_ids: set[str] = set()
    for stage in STAGES:
        cases = _list_value(query_stages, stage)
        rows = _list_value(gold_stages, stage)
        if len(cases) != len(COHORTS) or len(rows) != len(cases):
            raise ValueError("each stage must contain one case per cohort")
        by_id = {
            _required_string(row, "case_id"): row
            for row in rows
            if isinstance(row, Mapping)
        }
        if len(by_id) != len(rows):
            raise ValueError("gold case ids must be unique")
        cohorts: list[str] = []
        stage_identities: set[str] = set()
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("query cases must be objects")
            case_id = _required_string(case, "case_id")
            cohort = _required_string(case, "cohort")
            _required_string(case, "query")
            if case_id in all_case_ids or case_id not in by_id:
                raise ValueError("query and gold case ids must be unique and aligned")
            all_case_ids.add(case_id)
            cohorts.append(cohort)
            row = by_id[case_id]
            if row.get("cohort") != cohort:
                raise ValueError("query and gold cohort mismatch")
            expected = _required_string(row, "expected_source_id")
            if expected not in identities:
                raise ValueError("expected source identity is outside the corpus")
            stage_identities.add(expected)
            forbidden = row.get("forbidden_source_ids", [])
            if not isinstance(forbidden, list) or any(
                item not in identities for item in forbidden
            ):
                raise ValueError("forbidden source identity is outside the corpus")
            stage_identities.update(forbidden)
            relation_seed = row.get("relation_seed_source_id")
            if cohort == "relation_linked":
                if not isinstance(relation_seed, str):
                    raise ValueError("relation case requires a seed identity")
                stage_identities.add(relation_seed)
                if not any(
                    edge[0] == relation_seed and edge[1] == expected
                    for edge in relationship_tuples
                ):
                    raise ValueError("relation gold must match a frozen edge")
        if tuple(cohorts) != COHORTS:
            raise ValueError("cohort order mismatch")
        split_identities[stage] = stage_identities
    if not split_identities["development"].isdisjoint(split_identities["holdout"]):
        raise ValueError("development and holdout gold identities must be disjoint")
    gates = _list_value(gate, "gates")
    gate_ids = tuple(
        _required_string(item, "gate_id") for item in gates if isinstance(item, Mapping)
    )
    if gate_ids != GATE_IDS or not all(item.get("hard") is True for item in gates):
        raise ValueError("hard gate order mismatch")
    predecessor = _mapping(manifest, "predecessor")
    old_case_ids = predecessor.get("case_ids")
    if not isinstance(old_case_ids, list) or set(old_case_ids) & all_case_ids:
        raise ValueError("predecessor and successor case identities must be disjoint")
    artifacts = _mapping(manifest, "artifact_sha256")
    if set(artifacts) & set(_mapping(predecessor, "artifact_sha256")):
        raise ValueError("predecessor artifacts must not be reused")


def verify_frozen_artifacts(protocol: Mapping[str, Any]) -> None:
    root = protocol["root"]
    if not isinstance(root, Path):
        raise ValueError("protocol root must be a Path")
    manifest = _mapping(protocol, "manifest")
    if not _mapping(manifest, "artifact_sha256"):
        raise ValueError("frozen artifact hash registry must not be empty")
    for relative, expected in _mapping(manifest, "artifact_sha256").items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("artifact hashes must map paths to hex digests")
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")
    for stage in STAGES:
        for kind in ("capture", "claim", "result"):
            if _stage_path(manifest, stage, kind).exists():
                raise ValueError(f"registered {stage} {kind} must be absent at freeze")


def verify_protocol_commit(protocol_commit: str, protocol: Mapping[str, Any]) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", protocol_commit) is None:
        raise ValueError("protocol_commit must be a full lowercase commit SHA")
    root = protocol["root"]
    manifest = _mapping(protocol, "manifest")
    _git_bytes(root, f"{protocol_commit}^{{commit}}")
    branch_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "origin/main"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if branch_check.returncode != 0:
        raise ValueError("protocol_commit must be merged into origin/main")
    manifest_relative = MANIFEST_PATH.relative_to(ROOT).as_posix()
    first_parent = f"{protocol_commit}^1"
    try:
        _git_bytes(root, f"{first_parent}^{{commit}}")
    except ValueError as error:
        raise ValueError("protocol_commit must have a valid first parent") from error
    parent_manifest = subprocess.run(
        ["git", "cat-file", "-e", f"{first_parent}:{manifest_relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if parent_manifest.returncode == 0:
        raise ValueError(
            "protocol_commit must be the commit that first introduces the manifest"
        )
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if introductions.returncode != 0 or introductions.stdout.splitlines() != [
        protocol_commit
    ]:
        raise ValueError(
            "protocol_commit must be the manifest's unique first-parent introduction"
        )
    if (
        _git_bytes(root, f"{protocol_commit}:{manifest_relative}")
        != (root / manifest_relative).read_bytes()
    ):
        raise ValueError("running manifest drifted from the frozen merge commit")
    for relative, expected in _mapping(manifest, "artifact_sha256").items():
        actual = hashlib.sha256(
            _git_bytes(root, f"{protocol_commit}:{relative}")
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"protocol commit artifact mismatch: {relative}")
        current = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if current != expected:
            raise ValueError(f"running artifact drifted from freeze: {relative}")
    for stage in STAGES:
        for kind in ("capture", "claim", "result"):
            relative = _required_string(
                _mapping_value(_mapping(manifest, "outputs")[stage], stage), kind
            )
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{protocol_commit}:{relative}"],
                cwd=root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if exists.returncode == 0:
                raise ValueError(
                    "frozen merge commit must not contain observation artifacts"
                )


def prove_writer_verifier_round_trip(path: Path) -> None:
    case_metrics = [
        {
            "case_id": "placeholder-negative",
            "cohort": "negative_control",
            "github_rag_mcp": {
                "rank": 1,
                "reciprocal_rank": 1.0,
                "hit_at_k": 1,
                "forbidden_hit": False,
            },
            "ngr": {
                "rank": 1,
                "reciprocal_rank": 1.0,
                "hit_at_k": 1,
                "forbidden_hit": False,
            },
        }
    ]
    gates = [
        {"gate_id": gate_id, "hard": True, "passed": True, "details": {}}
        for gate_id in GATE_IDS
    ]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": "0" * 40,
        "stage": "development",
        "status": "passed",
        "failure_code": None,
        "capture_sha256": "1" * 64,
        "protocol_hashes": {},
        "source": {"repository": "placeholder/repository", "commit": "2" * 40},
        "cases": case_metrics,
        "cohorts": {},
        "update_following": {"passed": True},
        "deterministic_replay": {"passed": True},
        "resources": {"latency_hard_gate": False},
        "gates": gates,
        "all_hard_gates_pass": True,
        "raw_github_rag_mcp_capture": {"placeholder": True},
        "interpretation_ja": "登録外 placeholder の writer/verifier round-trip。",
    }
    write_json_exclusive(path, payload)
    verify_result_payload(read_json(path), read_json(RESULT_SCHEMA_PATH))
    path.unlink()


def run_registered_stage(stage: str, protocol_commit: str) -> Path:
    protocol = load_protocol()
    verify_protocol_commit(protocol_commit, protocol)
    _validate_stage(stage)
    _assert_stage_can_start(stage, protocol)
    manifest = _mapping(protocol, "manifest")
    capture_path = _stage_path(manifest, stage, "capture")
    claim_path = _stage_path(manifest, stage, "claim")
    result_path = _stage_path(manifest, stage, "result")
    if not capture_path.exists():
        raise FileNotFoundError("registered capture is required before stage execution")
    capture_sha256 = hashlib.sha256(capture_path.read_bytes()).hexdigest()
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
    except Exception as error:
        payload = _failure_payload(
            stage, protocol_commit, capture_sha256, protocol, error
        )
    write_json_exclusive(result_path, payload)
    _verify_registered_result(stage, protocol)
    return result_path


def verify_registered_result(stage: str) -> None:
    protocol = load_protocol()
    _verify_registered_result(stage, protocol)


def _verify_registered_result(stage: str, protocol: Mapping[str, Any]) -> None:
    _validate_stage(stage)
    manifest = _mapping(protocol, "manifest")
    claim = read_json(_stage_path(manifest, stage, "claim"))
    if list(claim) != list(CLAIM_FIELDS):
        raise ValueError("stage claim field order mismatch")
    if claim.get("protocol_id") != PROTOCOL_ID or claim.get("stage") != stage:
        raise ValueError("stage claim protocol identity mismatch")
    if claim.get("one_time_claim") is not True:
        raise ValueError("stage claim must be marked one_time_claim")
    protocol_commit = claim.get("protocol_commit")
    if not isinstance(protocol_commit, str):
        raise ValueError("stage claim protocol_commit must be a string")
    verify_protocol_commit(protocol_commit, protocol)
    capture_sha256 = claim.get("capture_sha256")
    if (
        not isinstance(capture_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", capture_sha256) is None
    ):
        raise ValueError("stage claim capture_sha256 must be full lowercase hex")
    result = read_json(_stage_path(manifest, stage, "result"))
    capture_path = _stage_path(manifest, stage, "capture")
    capture = read_json(capture_path)
    if hashlib.sha256(capture_path.read_bytes()).hexdigest() != capture_sha256:
        raise ValueError("registered capture changed after the stage claim")
    verify_result_payload(result, _mapping(protocol, "result_schema"))
    if result.get("stage") != stage:
        raise ValueError("result stage does not match the stage claim")
    if result.get("protocol_commit") != protocol_commit:
        raise ValueError("result protocol_commit does not match the stage claim")
    if result.get("capture_sha256") != capture_sha256:
        raise ValueError("result does not match the stage claim")
    if result.get("protocol_hashes") != dict(_mapping(manifest, "artifact_sha256")):
        raise ValueError("result protocol hashes do not match the frozen manifest")
    if result.get("raw_github_rag_mcp_capture") != capture:
        raise ValueError("result raw capture does not match the registered capture")


def verify_result_payload(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    fields = schema.get("top_level_fields")
    if not isinstance(fields, list) or list(payload) != fields:
        raise ValueError("result top-level field order mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("stage") not in STAGES:
        raise ValueError("result protocol identity mismatch")
    protocol_commit = payload.get("protocol_commit")
    if (
        not isinstance(protocol_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", protocol_commit) is None
    ):
        raise ValueError("result protocol_commit must be full lowercase hex")
    capture_sha256 = payload.get("capture_sha256")
    if (
        not isinstance(capture_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", capture_sha256) is None
    ):
        raise ValueError("result capture_sha256 must be full lowercase hex")
    gates = payload.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATE_IDS):
        raise ValueError("result hard gate order mismatch")
    for index, item in enumerate(gates):
        if not isinstance(item, Mapping) or list(item) != list(GATE_FIELDS):
            raise ValueError("result hard gate shape mismatch")
        if item.get("gate_id") != GATE_IDS[index]:
            raise ValueError("result hard gate order mismatch")
        if item.get("hard") is not True:
            raise ValueError("result gate must be hard")
        if type(item.get("passed")) is not bool:
            raise ValueError("result gate passed must be boolean")
        if not isinstance(item.get("details"), Mapping):
            raise ValueError("result gate details must be an object")
    all_pass = all(item["passed"] for item in gates)
    if payload.get("all_hard_gates_pass") is not all_pass:
        raise ValueError("result all_hard_gates_pass mismatch")
    status = payload.get("status")
    if status not in ("passed", "failed"):
        raise ValueError("result status must be passed or failed")
    if status == "passed":
        if not all_pass or payload.get("failure_code") is not None:
            raise ValueError("passed result must pass every hard gate")
    elif all_pass or payload.get("failure_code") not in schema.get("failure_codes", []):
        raise ValueError("failed result must fail a hard gate with a known code")


def _execute_stage(
    stage: str,
    protocol_commit: str,
    capture_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    capture = read_json(_stage_path(_mapping(protocol, "manifest"), stage, "capture"))
    queries, gold = _stage_cases(protocol, stage)
    github_cases = _validate_capture(capture, stage, protocol_commit, protocol, queries)
    first, first_resources = _run_ngr_once(protocol, queries)
    second, second_resources = _run_ngr_once(protocol, queries)
    deterministic = first == second
    update = _verify_update_following(protocol)
    case_metrics = [
        _case_metrics(
            case,
            gold[case["case_id"]],
            github_cases[case["case_id"]],
            first[case["case_id"]],
            _top_k(protocol),
        )
        for case in queries
    ]
    cohorts = _cohort_metrics(case_metrics)
    gates = _evaluate_gates(case_metrics, cohorts, deterministic, update)
    all_pass = all(item["passed"] for item in gates)
    current = protocol["current"]
    assert isinstance(current, GitHubSnapshot)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "status": "passed" if all_pass else "failed",
        "failure_code": None if all_pass else "hard-gate-failed",
        "capture_sha256": capture_sha256,
        "protocol_hashes": dict(
            _mapping(_mapping(protocol, "manifest"), "artifact_sha256")
        ),
        "source": {
            "repository": current.repository,
            "commit": current.commit,
            "fingerprint": current.fingerprint(),
            "document_count": len(current.documents),
        },
        "cases": case_metrics,
        "cohorts": cohorts,
        "update_following": update,
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
        "interpretation_ja": (
            "固定された共通 document surface 上の比較結果。"
            "latency と resource は記録のみで hard gate ではない。"
        ),
    }


def _failure_payload(
    stage: str,
    protocol_commit: str,
    capture_sha256: str,
    protocol: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    current = protocol["current"]
    assert isinstance(current, GitHubSnapshot)
    gates = [
        {
            "gate_id": gate_id,
            "hard": True,
            "passed": False,
            "details": {"not_evaluated": True},
        }
        for gate_id in GATE_IDS
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "stage": stage,
        "status": "failed",
        "failure_code": "execution-failed",
        "capture_sha256": capture_sha256,
        "protocol_hashes": dict(
            _mapping(_mapping(protocol, "manifest"), "artifact_sha256")
        ),
        "source": {"repository": current.repository, "commit": current.commit},
        "cases": [],
        "cohorts": {},
        "update_following": {"passed": False, "not_evaluated": True},
        "deterministic_replay": {"passed": False, "not_evaluated": True},
        "resources": {"latency_hard_gate": False},
        "gates": gates,
        "all_hard_gates_pass": False,
        "raw_github_rag_mcp_capture": read_json(
            _stage_path(_mapping(protocol, "manifest"), stage, "capture")
        ),
        "interpretation_ja": (
            f"一回性 stage は {type(error).__name__} で停止した。再実行しない。"
        ),
    }


def _run_ngr_once(
    protocol: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = protocol["current"]
    assert isinstance(current, GitHubSnapshot)
    config = EngineConfig(
        **dict(_mapping(_mapping(protocol, "queries"), "ngr_engine_config"))
    )
    started = time.perf_counter()
    with TemporaryDirectory() as directory:
        database = Path(directory) / "parity.sqlite"
        with NeuronGraphRAG(database, config=config) as engine:
            index_github_snapshot(engine, current)
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
                    _required_string(case, "query"),
                    limit=_top_k(protocol),
                    now=0.0,
                )
                results[_required_string(case, "case_id")] = [
                    {
                        "rank": rank,
                        "source_id": hit.node.node_id,
                        "repository": hit.node.metadata["repository"],
                        "path": hit.node.metadata["path"],
                        "source_url": hit.node.metadata["source_url"],
                        "commit": hit.node.metadata["commit"],
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


def _verify_update_following(protocol: Mapping[str, Any]) -> dict[str, Any]:
    previous = protocol["previous"]
    current = protocol["current"]
    assert isinstance(previous, GitHubSnapshot) and isinstance(current, GitHubSnapshot)
    with NeuronGraphRAG() as engine:
        index_github_snapshot(engine, previous)
        index_github_snapshot(engine, current)
        nodes = {node.node_id: node for node in engine.store.list_nodes()}
    followed = all(
        nodes[current.document_id(document)].metadata["blob_sha"] == document.blob_sha
        and nodes[current.document_id(document)].metadata["content_sha256"]
        == document.content_sha256
        for document in current.documents
    )
    return {
        "passed": followed,
        "before_commit": previous.commit,
        "after_commit": current.commit,
        "changed_paths": list(changed_paths(previous, current)),
    }


def _validate_capture(
    capture: Mapping[str, Any],
    stage: str,
    protocol_commit: str,
    protocol: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    capture_schema = _mapping(protocol, "capture_schema")
    if list(capture) != capture_schema.get("top_level_fields"):
        raise ValueError("capture top-level field order mismatch")
    if capture.get("protocol_id") != PROTOCOL_ID or capture.get("stage") != stage:
        raise ValueError("capture protocol identity mismatch")
    if capture.get("protocol_commit") != protocol_commit:
        raise ValueError("capture is not bound to the frozen merge commit")
    if capture.get("service") != "github-rag-mcp" or capture.get("tool") != "search":
        raise ValueError("capture must preserve github-rag-mcp search output")
    _required_string(capture, "captured_at")
    rows = capture.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("capture must contain every frozen case exactly once")
    expected_cases = {_required_string(case, "case_id"): case for case in cases}
    captured: dict[str, dict[str, Any]] = {}
    current = protocol["current"]
    assert isinstance(current, GitHubSnapshot)
    documents = {document.path: document for document in current.documents}
    defaults = dict(_mapping(_mapping(protocol, "queries"), "request_defaults"))
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("capture case must be an object")
        if list(row) != capture_schema.get("case_fields"):
            raise ValueError("capture case field order mismatch")
        case_id = _required_string(row, "case_id")
        if case_id not in expected_cases or case_id in captured:
            raise ValueError("capture case identity mismatch or duplicate")
        expected_request = {"query": expected_cases[case_id]["query"], **defaults}
        if row.get("request") != expected_request:
            raise ValueError("capture request differs from the frozen common request")
        raw_search = _mapping_value(row.get("raw_search"), "raw_search")
        if (
            raw_search.get("mode") != "search"
            or raw_search.get("filters_unmatched") != []
        ):
            raise ValueError("github-rag-mcp search did not use a valid matched filter")
        search_results = raw_search.get("results")
        if not isinstance(search_results, list):
            raise ValueError("raw search results must be a list")
        if len(search_results) > _top_k(protocol):
            raise ValueError("raw search result count exceeds frozen top_k")
        vector_paths: dict[str, str] = {}
        keyword: list[dict[str, Any]] = []
        for rank, item in enumerate(search_results, start=1):
            if not isinstance(item, Mapping):
                raise ValueError("raw search result must be an object")
            if item.get("repo") != current.repository or item.get("type") != "doc":
                raise ValueError("raw search result left the frozen document surface")
            path = _required_string(item, "doc_path")
            if path not in documents:
                raise ValueError("raw search result path is outside the frozen corpus")
            vector_id = _required_string(item, "vector_id")
            if vector_id in vector_paths:
                raise ValueError("raw search result vector_id must be unique")
            vector_paths[vector_id] = path
            keyword.append(
                {
                    "rank": rank,
                    "source_id": _source_identity(current.repository, path),
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
            raise ValueError("stored-content results must be a list")
        fetched_by_id = {
            _required_string(item, "vector_id"): item
            for item in fetched
            if isinstance(item, Mapping)
        }
        if len(fetched_by_id) != len(fetched) or set(fetched_by_id) != set(
            vector_paths
        ):
            raise ValueError("stored-content capture must cover every search result")
        for vector_id, path in vector_paths.items():
            item = fetched_by_id[vector_id]
            if (
                item.get("repo") != current.repository
                or item.get("type") != "doc"
                or item.get("doc_path") != path
            ):
                raise ValueError("stored-content source identity mismatch")
            expected_content = _indexed_content(path, documents[path].content)
            if item.get("content") != expected_content:
                raise ValueError(
                    "github-rag-mcp indexed content differs from the frozen corpus"
                )
            if item.get("content_chars") != _js_length(expected_content):
                raise ValueError("stored-content character count mismatch")
            if item.get("content_truncated") is not (
                _js_length(path + "\n\n" + documents[path].content) >= 8000
            ):
                raise ValueError("stored-content truncation provenance mismatch")
        graph: list[dict[str, Any]] = []
        graph_results = raw_search.get("graph_results", [])
        if not isinstance(graph_results, list):
            raise ValueError("graph_results must be a list when present")
        for rank, item in enumerate(graph_results, start=1):
            if not isinstance(item, Mapping):
                raise ValueError("graph result must be an object")
            path = item.get("doc_path")
            if (
                item.get("repo") != current.repository
                or item.get("type") != "doc"
                or path not in documents
            ):
                raise ValueError("graph result left the frozen document surface")
            graph.append(
                {
                    "rank": rank,
                    "source_id": _source_identity(current.repository, str(path)),
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
    expected = _required_string(gold, "expected_source_id")
    forbidden = list(gold.get("forbidden_source_ids", []))
    github_rank = _github_rank(github, expected, cohort)
    ngr_rank = _ngr_rank(ngr_hits, expected, cohort)
    github_forbidden = any(
        _github_rank(github, item, "negative_control") is not None for item in forbidden
    )
    ngr_forbidden = any(
        _ngr_rank(ngr_hits, item, "negative_control") is not None for item in forbidden
    )
    relation_path = False
    for hit in ngr_hits:
        if hit.get("source_id") != expected:
            continue
        explanation = hit.get("explanation")
        if isinstance(explanation, Mapping):
            paths = explanation.get("paths", [])
            relation_path = any(
                isinstance(path, Mapping) and bool(path.get("steps")) for path in paths
            )
    return {
        "case_id": _required_string(case, "case_id"),
        "cohort": cohort,
        "query": _required_string(case, "query"),
        "expected_source_id": expected,
        "forbidden_source_ids": forbidden,
        "github_rag_mcp": _metric(github_rank, top_k, github_forbidden),
        "ngr": {
            **_metric(ngr_rank, top_k, ngr_forbidden),
            "hits": list(ngr_hits),
            "expected_relation_path": relation_path,
        },
        "raw_github_rag_mcp": {
            "search": github["raw_search"],
            "stored_content": github["raw_stored_content"],
        },
    }


def _cohort_metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["cohort"])].append(case)
    return {
        cohort: {
            retriever: {
                "mrr": sum(
                    float(case[retriever]["reciprocal_rank"])
                    for case in grouped[cohort]
                )
                / len(grouped[cohort]),
                "hit_at_k": sum(
                    int(case[retriever]["hit_at_k"]) for case in grouped[cohort]
                )
                / len(grouped[cohort]),
            }
            for retriever in ("github_rag_mcp", "ngr")
        }
        for cohort in COHORTS
    }


def _evaluate_gates(
    cases: Sequence[Mapping[str, Any]],
    cohorts: Mapping[str, Any],
    deterministic: bool,
    update: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_cohort = {str(case["cohort"]): case for case in cases}
    direct = by_cohort["direct_lexical"]
    negative = by_cohort["negative_control"]
    cohort_mrr = all(
        cohorts[name]["ngr"]["mrr"] >= cohorts[name]["github_rag_mcp"]["mrr"]
        for name in COHORTS
    )
    cohort_hit = all(
        cohorts[name]["ngr"]["hit_at_k"] >= cohorts[name]["github_rag_mcp"]["hit_at_k"]
        for name in COHORTS
    )
    complete = all(
        case[retriever]["hit_at_k"] == 1
        for case in cases
        for retriever in ("github_rag_mcp", "ngr")
    )
    explanations = (
        all(
            all(
                hit.get("repository")
                and hit.get("path")
                and isinstance(hit.get("explanation"), Mapping)
                for hit in case["ngr"]["hits"]
            )
            for case in cases
        )
        and by_cohort["relation_linked"]["ngr"]["expected_relation_path"]
    )
    verdicts = {
        "protocol-integrity": True,
        "source-provenance-integrity": True,
        "deterministic-replay": deterministic,
        "update-following": update.get("passed") is True,
        "direct-case-non-regression": direct["ngr"]["reciprocal_rank"]
        >= direct["github_rag_mcp"]["reciprocal_rank"]
        and direct["ngr"]["hit_at_k"] == 1,
        "negative-control-non-regression": negative["ngr"]["reciprocal_rank"]
        >= negative["github_rag_mcp"]["reciprocal_rank"]
        and not negative["ngr"]["forbidden_hit"]
        and not negative["github_rag_mcp"]["forbidden_hit"],
        "cohort-mrr-non-regression": cohort_mrr,
        "cohort-hit-at-k-non-regression": cohort_hit,
        "expected-source-top-k-completeness": complete,
        "source-path-explanation-integrity": explanations,
    }
    return [
        {"gate_id": gate_id, "hard": True, "passed": verdicts[gate_id], "details": {}}
        for gate_id in GATE_IDS
    ]


def _metric(rank: int | None, top_k: int, forbidden_hit: bool) -> dict[str, Any]:
    return {
        "rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "hit_at_k": int(rank is not None and rank <= top_k),
        "forbidden_hit": forbidden_hit,
    }


def _github_rank(results: Mapping[str, Any], source_id: str, cohort: str) -> int | None:
    ranks = [
        item["rank"] for item in results["keyword"] if item["source_id"] == source_id
    ]
    if cohort == "relation_linked":
        ranks.extend(
            item["rank"] for item in results["graph"] if item["source_id"] == source_id
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
        if cohort == "relation_linked":
            explanation = hit.get("explanation")
            if isinstance(explanation, Mapping):
                graph_rank = (
                    explanation.get("ranks", {}).get("graph")
                    if isinstance(explanation.get("ranks"), Mapping)
                    else None
                )
                if isinstance(graph_rank, int):
                    ranks.append(graph_rank)
    return min(ranks) if ranks else None


def _stage_cases(
    protocol: Mapping[str, Any], stage: str
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    query_rows = _list_value(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    gold_rows = _list_value(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    cases = [row for row in query_rows if isinstance(row, Mapping)]
    gold = {
        _required_string(row, "case_id"): row
        for row in gold_rows
        if isinstance(row, Mapping)
    }
    return cases, gold


def _assert_stage_can_start(stage: str, protocol: Mapping[str, Any]) -> None:
    manifest = _mapping(protocol, "manifest")
    for kind in ("claim", "result"):
        if _stage_path(manifest, stage, kind).exists():
            raise FileExistsError(f"registered {stage} {kind} already exists")
    if stage == "holdout":
        development = _stage_path(manifest, "development", "result")
        if not development.exists():
            raise RuntimeError("holdout is closed until development result exists")
        _verify_registered_result("development", protocol)
        result = read_json(development)
        if result.get("all_hard_gates_pass") is not True:
            raise RuntimeError(
                "holdout is closed because development hard gates did not pass"
            )


def _stage_path(manifest: Mapping[str, Any], stage: str, kind: str) -> Path:
    outputs = _mapping(manifest, "outputs")
    stage_outputs = _mapping_value(outputs.get(stage), f"outputs.{stage}")
    return ROOT / _required_string(stage_outputs, kind)


def _validate_snapshot_hashes(snapshot: GitHubSnapshot) -> None:
    for document in snapshot.documents:
        if re.fullmatch(r"[0-9a-f]{40}", document.blob_sha) is None:
            raise ValueError("snapshot blob SHA must be a full lowercase SHA-1")
        if re.fullmatch(r"[0-9a-f]{64}", document.content_sha256) is None:
            raise ValueError("snapshot content SHA-256 must be full lowercase hex")


def _indexed_content(path: str, content: str) -> str:
    value = path + "\n\n" + content
    if _js_length(value) <= 8000:
        return value
    return value.encode("utf-16-le")[:16000].decode("utf-16-le", errors="ignore")


def _js_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _source_identity(repository: str, path: str) -> str:
    return f"github:{repository}:{path}"


def _top_k(protocol: Mapping[str, Any]) -> int:
    value = _mapping(_mapping(protocol, "queries"), "request_defaults").get("top_k")
    if not isinstance(value, int):
        raise ValueError("top_k must be an integer")
    return value


def _json_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, object_name: str) -> bytes:
    result = subprocess.run(
        ["git", "show", object_name],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"git object is unavailable: {object_name}")
    return result.stdout


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of: {', '.join(STAGES)}")


def _mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping_value(container.get(key), key)


def _mapping_value(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list_value(container: Mapping[str, Any], key: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value

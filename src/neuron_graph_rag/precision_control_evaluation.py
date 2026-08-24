from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .precision_control import RULES, PrecisionControl

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STEM = "github_precision_control_v1"
PROTOCOL_ID = "github-ngr-precision-control-v1"
STAGES = ("development", "holdout")
COHORTS = (
    "direct_lexical",
    "semantic_paraphrase",
    "relation_linked",
    "negative_control",
)
GATE_IDS = (
    "protocol-integrity",
    "identity-separation",
    "deterministic-fresh-db",
    "direct-case-non-regression",
    "semantic-case-non-regression",
    "relation-case-non-regression",
    "cohort-mrr-hit-at-k-non-regression",
    "negative-forbidden-strict-improvement",
    "expected-source-top-k-completeness",
    "relation-source-path-provenance",
    "immutable-post-ranking-isolation",
)
ARTIFACT_KINDS = (
    "corpus",
    "queries",
    "gold",
    "candidates",
    "gate",
    "result-schema",
    "result-free-audit",
)
MANIFEST_PATH = FIXTURES / f"{STEM}.manifest.json"
RESULT_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "status",
    "claim_sha256",
    "protocol_hashes",
    "baseline",
    "candidates",
    "selected_candidate_id",
    "gates",
    "all_hard_gates_pass",
)
CLAIM_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "protocol_hashes",
    "one_time_claim",
)
GATE_FIELDS = ("gate_id", "hard", "passed", "details")
BASELINE_FIELDS = ("baseline_id", "cases", "cohorts", "state")
CANDIDATE_FIELDS = (
    "candidate_id",
    "cases",
    "cohorts",
    "explanations",
    "state",
    "all_hard_gates_pass",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_protocol(root: Path = ROOT) -> dict[str, Any]:
    fixture_root = root / "tests" / "fixtures"
    manifest_path = fixture_root / f"{STEM}.manifest.json"
    protocol: dict[str, Any] = {
        "root": root,
        "manifest_path": manifest_path,
        "manifest": read_json(manifest_path),
    }
    for kind in ARTIFACT_KINDS:
        protocol[kind.replace("-", "_")] = read_json(
            fixture_root / f"{STEM}.{kind}.json"
        )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    manifest = _mapping(protocol, "manifest")
    for kind in ARTIFACT_KINDS:
        artifact = _mapping(protocol, kind.replace("-", "_"))
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"protocol_id mismatch in {kind}")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("manifest protocol_id mismatch")
    _verify_hash_registry(root, _mapping(manifest, "artifact_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "v1_immutable_sha256"))
    _validate_corpus(root, protocol)
    _validate_cases(protocol)
    _validate_candidates(protocol)
    gates = _list(_mapping(protocol, "gate"), "gates")
    if tuple(_string(item, "gate_id") for item in gates) != GATE_IDS:
        raise ValueError("hard gate order is not frozen")
    if any(
        set(item) != {"gate_id", "hard"} or item["hard"] is not True for item in gates
    ):
        raise ValueError("all frozen gates must be exact and hard")
    schema = _mapping(protocol, "result_schema")
    if tuple(schema.get("required_fields", ())) != RESULT_FIELDS:
        raise ValueError("result schema fields mismatch")
    if tuple(schema.get("gate_required_fields", ())) != GATE_FIELDS:
        raise ValueError("gate schema fields mismatch")
    if tuple(schema.get("baseline_required_fields", ())) != BASELINE_FIELDS:
        raise ValueError("baseline schema fields mismatch")
    if tuple(schema.get("candidate_required_fields", ())) != CANDIDATE_FIELDS:
        raise ValueError("candidate schema fields mismatch")
    audit = _mapping(protocol, "result_free_audit")
    if audit.get("phase") != "freeze":
        raise ValueError("result-free audit phase mismatch")
    if audit.get("registered_query_execution_count") != 0:
        raise ValueError("registered query execution must remain zero at freeze")
    if audit.get("observed_result_count") != 0:
        raise ValueError("observed result count must remain zero at freeze")
    if any(
        audit.get(key) is not False
        for key in (
            "shared_database_opened",
            "existing_experiment_database_opened",
            "feedback_or_outcome_recorded",
            "github_rag_mcp_called",
            "v1_artifact_reused_or_modified",
        )
    ):
        raise ValueError("result-free safety audit failed")


def _validate_corpus(root: Path, protocol: Mapping[str, Any]) -> None:
    manifest = _mapping(protocol, "manifest")
    corpus = _mapping(protocol, "corpus")
    source = _mapping(manifest, "source")
    if corpus.get("repository") != source.get("repository"):
        raise ValueError("corpus repository mismatch")
    commit = _full_commit(corpus.get("commit"))
    if commit != source.get("commit"):
        raise ValueError("corpus commit mismatch")
    documents = _list(corpus, "documents")
    if len(documents) != 20:
        raise ValueError("the frozen corpus must contain exactly 20 documents")
    paths: set[str] = set()
    for item in documents:
        if set(item) != {"path", "content_sha256"}:
            raise ValueError("corpus document shape mismatch")
        path = _string(item, "path")
        if path in paths or not re.fullmatch(r"(?:README\.md|docs/[\w.-]+\.md)", path):
            raise ValueError("corpus paths must be unique repository Markdown paths")
        paths.add(path)
        raw = _git_bytes(root, commit, path)
        if sha256_bytes(raw) != _string(item, "content_sha256"):
            raise ValueError(f"source content hash mismatch: {path}")
    relationships = _list(corpus, "relationships")
    relation_keys = set()
    for relation in relationships:
        if set(relation) != {"source_path", "target_path", "edge_type"}:
            raise ValueError("relationship shape mismatch")
        key = (
            _string(relation, "source_path"),
            _string(relation, "target_path"),
            _string(relation, "edge_type"),
        )
        if key[0] not in paths or key[1] not in paths or key in relation_keys:
            raise ValueError("relationship is outside the frozen corpus")
        relation_keys.add(key)


def _validate_cases(protocol: Mapping[str, Any]) -> None:
    corpus = _mapping(protocol, "corpus")
    queries = _mapping(protocol, "queries")
    gold = _mapping(protocol, "gold")
    paths = {_string(item, "path") for item in _list(corpus, "documents")}
    relations = {
        (
            _string(item, "source_path"),
            _string(item, "target_path"),
            _string(item, "edge_type"),
        )
        for item in _list(corpus, "relationships")
    }
    query_stages = _mapping(queries, "stages")
    gold_stages = _mapping(gold, "stages")
    query_ids: set[str] = set()
    identity_splits: dict[str, set[str]] = {}
    for stage in STAGES:
        cases = _list(query_stages, stage)
        rows = _list(gold_stages, stage)
        if len(cases) != 8 or len(rows) != 8:
            raise ValueError("each stage requires eight registered cases")
        if tuple(_string(case, "cohort") for case in cases) != tuple(
            cohort for cohort in COHORTS for _ in range(2)
        ):
            raise ValueError("each cohort must have exactly two cases in frozen order")
        by_id = {_string(row, "case_id"): row for row in rows}
        if len(by_id) != 8:
            raise ValueError("gold case ids must be unique")
        identities: set[str] = set()
        for case in cases:
            if set(case) != {"case_id", "cohort", "query"}:
                raise ValueError("query case shape mismatch")
            case_id = _string(case, "case_id")
            _string(case, "query")
            if case_id in query_ids or case_id not in by_id:
                raise ValueError("query identities must be unique and gold-aligned")
            query_ids.add(case_id)
            row = by_id[case_id]
            if row.get("cohort") != case.get("cohort"):
                raise ValueError("query/gold cohort mismatch")
            expected = row.get("expected_paths")
            forbidden = row.get("forbidden_paths")
            if not isinstance(expected, list) or not isinstance(forbidden, list):
                raise ValueError("gold identity lists are required")
            if any(
                not isinstance(path, str) or path not in paths
                for path in expected + forbidden
            ):
                raise ValueError("gold identity is outside the corpus")
            identities.update(expected)
            identities.update(forbidden)
            if case["cohort"] == "negative_control":
                if expected or len(forbidden) != 1:
                    raise ValueError("negative controls require one forbidden identity")
            elif len(expected) != 1 or forbidden:
                raise ValueError("positive cases require one expected identity")
            if case["cohort"] == "relation_linked":
                seed = _string(row, "relation_seed_path")
                edge_type = _string(row, "relation_edge_type")
                identities.add(seed)
                if (seed, expected[0], edge_type) not in relations:
                    raise ValueError("relation gold does not match the frozen edge")
        identity_splits[stage] = identities
    if not identity_splits["development"].isdisjoint(identity_splits["holdout"]):
        raise ValueError("development and holdout gold identities overlap")
    v1_corpus = read_json(
        _path(protocol, "root")
        / "tests/fixtures/github_retrieval_parity_v1.corpus.json"
    )
    v1_repository = _string(v1_corpus, "repository")
    repository = _string(corpus, "repository")
    current_ids = {f"github:{repository}:doc:{path}" for path in paths}
    v1_ids = {
        f"github:{v1_repository}:doc:{_string(item, 'path')}"
        for item in _list(v1_corpus, "documents")
    }
    if not current_ids.isdisjoint(v1_ids):
        raise ValueError("successor source identities overlap parity v1")
    v1_queries = read_json(
        _path(protocol, "root")
        / "tests/fixtures/github_retrieval_parity_v1.queries.json"
    )
    v1_case_ids = {
        _string(case, "case_id")
        for stage in STAGES
        for case in _list(_mapping(v1_queries, "stages"), stage)
    }
    if not query_ids.isdisjoint(v1_case_ids):
        raise ValueError("successor query identities overlap parity v1")


def _validate_candidates(protocol: Mapping[str, Any]) -> None:
    artifact = _mapping(protocol, "candidates")
    candidates = _list(artifact, "candidates")
    if len(candidates) != 5:
        raise ValueError("candidate set cardinality must remain finite and fixed")
    controls = [PrecisionControl.from_mapping(dict(item)) for item in candidates]
    if len({control.candidate_id for control in controls}) != len(controls):
        raise ValueError("candidate ids must be unique")
    enabled = {rule for control in controls for rule in control.rules()}
    if enabled != set(RULES):
        raise ValueError("candidate axes are incomplete")
    selection = artifact.get("selection_rule")
    if (
        not isinstance(selection, str)
        or "first candidate" not in selection
        or "do not open holdout" not in selection
    ):
        raise ValueError("pre-fixed selection and zero-pass stop rules are required")
    derivation = artifact.get("derivation")
    if not isinstance(derivation, str) or "no v1 observed value" not in derivation:
        raise ValueError("result-independent threshold derivation is required")


def verify_protocol_commit(protocol_commit: str, protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    commit = _full_commit(protocol_commit)
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError("protocol commit must be an origin/main ancestor")
    manifest_relative = str(_path(protocol, "manifest_path").relative_to(root)).replace(
        "\\", "/"
    )
    _git_bytes(root, commit, manifest_relative)
    first_parent = _git_output(root, "rev-parse", f"{commit}^1").strip()
    parent_probe = subprocess.run(
        ["git", "cat-file", "-e", f"{first_parent}:{manifest_relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if parent_probe.returncode == 0:
        raise ValueError(
            "protocol manifest must be introduced by the freeze merge commit"
        )
    manifest = json.loads(
        _git_bytes(root, commit, manifest_relative).decode("utf-8", errors="strict")
    )
    if not isinstance(manifest, dict):
        raise ValueError("committed manifest must be an object")
    for path, expected in _mapping(manifest, "artifact_sha256").items():
        if not isinstance(path, str) or not isinstance(expected, str):
            raise ValueError("committed artifact registry shape mismatch")
        if sha256_bytes(_git_bytes(root, commit, path)) != expected:
            raise ValueError(f"committed protocol artifact mismatch: {path}")


def register_stage_claim(stage: str, protocol_commit: str, root: Path = ROOT) -> Path:
    protocol = load_protocol(root)
    verify_protocol_commit(protocol_commit, protocol)
    _assert_stage_can_start(protocol, stage)
    target = _output_path(protocol, stage, "runtime_claim")
    write_json_exclusive(
        target,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": protocol_commit,
            "stage": stage,
            "protocol_hashes": dict(
                _mapping(_mapping(protocol, "manifest"), "artifact_sha256")
            ),
            "one_time_claim": True,
        },
    )
    return target


def write_stage_result(
    stage: str, payload: Mapping[str, Any], root: Path = ROOT
) -> Path:
    protocol = load_protocol(root)
    claim_path = _output_path(protocol, stage, "runtime_claim")
    if not claim_path.exists():
        raise ValueError("registered runtime claim is required")
    verify_result_payload(protocol, stage, payload, claim_path.read_bytes())
    target = _output_path(protocol, stage, "runtime_result")
    write_json_exclusive(target, payload)
    return target


def verify_result_payload(
    protocol: Mapping[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    claim_raw: bytes,
) -> None:
    if set(payload) != set(RESULT_FIELDS):
        raise ValueError("result fields must exactly match the frozen schema")
    claim = json.loads(claim_raw.decode("utf-8", errors="strict"))
    if not isinstance(claim, dict) or set(claim) != set(CLAIM_FIELDS):
        raise ValueError("claim fields must exactly match the frozen schema")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or claim.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("result/claim protocol mismatch")
    if payload.get("stage") != stage or claim.get("stage") != stage:
        raise ValueError("result/claim stage mismatch")
    if payload.get("protocol_commit") != claim.get("protocol_commit"):
        raise ValueError("result/claim commit mismatch")
    if payload.get("claim_sha256") != sha256_bytes(claim_raw):
        raise ValueError("result claim hash mismatch")
    hashes = dict(_mapping(_mapping(protocol, "manifest"), "artifact_sha256"))
    if (
        payload.get("protocol_hashes") != hashes
        or claim.get("protocol_hashes") != hashes
    ):
        raise ValueError("protocol hash registry mismatch")
    if claim.get("one_time_claim") is not True:
        raise ValueError("one-time claim must be true")
    if payload.get("status") not in {"passed", "failed"}:
        raise ValueError("result status must be passed or failed")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict) or set(baseline) != set(BASELINE_FIELDS):
        raise ValueError("baseline shape mismatch")
    if baseline.get("baseline_id") != "current-ngr":
        raise ValueError("baseline identity mismatch")
    if not isinstance(baseline.get("cases"), list):
        raise ValueError("baseline cases must be an array")
    if not isinstance(baseline.get("cohorts"), dict):
        raise ValueError("baseline cohorts must be an object")
    if not isinstance(baseline.get("state"), dict):
        raise ValueError("baseline state must be an object")
    expected_candidates = [
        _string(item, "candidate_id")
        for item in _list(_mapping(protocol, "candidates"), "candidates")
    ]
    rows = payload.get("candidates")
    if not isinstance(rows, list) or any(
        not isinstance(item, dict) or set(item) != set(CANDIDATE_FIELDS)
        for item in rows
    ):
        raise ValueError("result candidate shape mismatch")
    if [_string(item, "candidate_id") for item in rows] != expected_candidates:
        raise ValueError("result candidate order mismatch")
    for item in rows:
        if not isinstance(item.get("cases"), list):
            raise ValueError("candidate cases must be an array")
        if not isinstance(item.get("cohorts"), dict):
            raise ValueError("candidate cohorts must be an object")
        if not isinstance(item.get("explanations"), list) or any(
            not isinstance(value, dict) for value in item["explanations"]
        ):
            raise ValueError("candidate explanations must be objects")
        if not isinstance(item.get("state"), dict):
            raise ValueError("candidate state must be an object")
        if not isinstance(item.get("all_hard_gates_pass"), bool):
            raise ValueError("candidate hard gate summary must be boolean")
    gates = payload.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATE_IDS):
        raise ValueError("result gates length mismatch")
    for gate, gate_id in zip(gates, GATE_IDS, strict=True):
        if not isinstance(gate, dict) or set(gate) != set(GATE_FIELDS):
            raise ValueError("result gate shape mismatch")
        if gate.get("gate_id") != gate_id or gate.get("hard") is not True:
            raise ValueError("result hard gate identity mismatch")
        if not isinstance(gate.get("passed"), bool) or not isinstance(
            gate.get("details"), dict
        ):
            raise ValueError("result hard gate types mismatch")
    all_pass = all(gate["passed"] for gate in gates)
    if payload.get("all_hard_gates_pass") is not all_pass:
        raise ValueError("all_hard_gates_pass mismatch")
    if (payload.get("status") == "passed") is not all_pass:
        raise ValueError("result status and hard gates disagree")
    selected = payload.get("selected_candidate_id")
    first_passing = next(
        (item["candidate_id"] for item in rows if item["all_hard_gates_pass"] is True),
        None,
    )
    if selected != first_passing:
        raise ValueError("selected candidate violates the pre-fixed first-pass rule")
    if all_pass is not (selected is not None):
        raise ValueError("global gates and candidate selection disagree")


def archive_stage(stage: str, root: Path = ROOT) -> Path:
    protocol = load_protocol(root)
    return _archive_stage(protocol, stage)


def _archive_stage(protocol: Mapping[str, Any], stage: str) -> Path:
    root = _path(protocol, "root")
    claim_runtime = _output_path(protocol, stage, "runtime_claim")
    result_runtime = _output_path(protocol, stage, "runtime_result")
    claim_archive = _output_path(protocol, stage, "archive_claim")
    result_archive = _output_path(protocol, stage, "archive_result")
    transport = _output_path(protocol, stage, "transport")
    if claim_archive.exists() or result_archive.exists() or transport.exists():
        raise FileExistsError("archive stage already exists")
    if not claim_runtime.exists() or not result_runtime.exists():
        raise ValueError("complete runtime claim/result pair is required")
    claim_raw = claim_runtime.read_bytes()
    result_raw = result_runtime.read_bytes()
    verify_result_payload(
        protocol,
        stage,
        json.loads(result_raw.decode("utf-8", errors="strict")),
        claim_raw,
    )
    claim_archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(claim_runtime, claim_archive)
    os.replace(result_runtime, result_archive)
    write_json_exclusive(
        transport,
        {
            "protocol_id": PROTOCOL_ID,
            "stage": stage,
            "reason": "phase-boundary runtime-to-main evidence archival",
            "files": [
                {
                    "runtime_path": str(claim_runtime.relative_to(root)).replace(
                        "\\", "/"
                    ),
                    "archive_path": str(claim_archive.relative_to(root)).replace(
                        "\\", "/"
                    ),
                    "sha256": sha256_bytes(claim_raw),
                    "byte_identity": claim_archive.read_bytes() == claim_raw,
                },
                {
                    "runtime_path": str(result_runtime.relative_to(root)).replace(
                        "\\", "/"
                    ),
                    "archive_path": str(result_archive.relative_to(root)).replace(
                        "\\", "/"
                    ),
                    "sha256": sha256_bytes(result_raw),
                    "byte_identity": result_archive.read_bytes() == result_raw,
                },
            ],
        },
    )
    return transport


def verify_phase_state(protocol: Mapping[str, Any]) -> dict[str, str]:
    phases: dict[str, str] = {}
    for stage in STAGES:
        runtime = [
            _output_path(protocol, stage, key)
            for key in ("runtime_claim", "runtime_result")
        ]
        archive = [
            _output_path(protocol, stage, key)
            for key in ("archive_claim", "archive_result", "transport")
        ]
        if any(path.exists() for path in runtime):
            raise ValueError(
                "runtime observation artifacts must be archived before commit"
            )
        if not any(path.exists() for path in archive):
            phases[stage] = "unobserved"
            continue
        if not all(path.exists() for path in archive):
            raise ValueError("archived observation stage is incomplete")
        claim_raw = archive[0].read_bytes()
        result_raw = archive[1].read_bytes()
        verify_result_payload(
            protocol,
            stage,
            json.loads(result_raw.decode("utf-8", errors="strict")),
            claim_raw,
        )
        transport = read_json(archive[2])
        files = _list(transport, "files")
        if len(files) != 2 or any(
            item.get("byte_identity") is not True for item in files
        ):
            raise ValueError("transport byte identity is not verified")
        expected = [sha256_bytes(claim_raw), sha256_bytes(result_raw)]
        if [item.get("sha256") for item in files] != expected:
            raise ValueError("transport hashes do not match archive bytes")
        phases[stage] = "archived"
    if phases["holdout"] == "archived":
        development = read_json(_output_path(protocol, "development", "archive_result"))
        if development.get("all_hard_gates_pass") is not True:
            raise ValueError("holdout archive requires passing development")
    return phases


def _assert_stage_can_start(protocol: Mapping[str, Any], stage: str) -> None:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    for key in (
        "runtime_claim",
        "runtime_result",
        "archive_claim",
        "archive_result",
        "transport",
    ):
        if _output_path(protocol, stage, key).exists():
            raise FileExistsError(f"{stage} has already started")
    if stage == "holdout":
        result_path = _output_path(protocol, "development", "archive_result")
        if not result_path.exists():
            raise ValueError("holdout requires archived development evidence")
        if read_json(result_path).get("all_hard_gates_pass") is not True:
            raise ValueError("holdout is closed after a failed development gate")


def prove_archive_round_trip() -> dict[str, str]:
    protocol = load_protocol()
    with tempfile.TemporaryDirectory() as directory:
        test_root = Path(directory)
        synthetic = dict(protocol)
        synthetic["root"] = test_root
        claim_path = _output_path(synthetic, "development", "runtime_claim")
        hashes = dict(_mapping(_mapping(protocol, "manifest"), "artifact_sha256"))
        claim = {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": "0" * 40,
            "stage": "development",
            "protocol_hashes": hashes,
            "one_time_claim": True,
        }
        write_json_exclusive(claim_path, claim)
        claim_raw = claim_path.read_bytes()
        gates = [
            {"gate_id": gate_id, "hard": True, "passed": True, "details": {}}
            for gate_id in GATE_IDS
        ]
        candidates = []
        for index, item in enumerate(
            _list(_mapping(protocol, "candidates"), "candidates")
        ):
            candidates.append(
                {
                    "candidate_id": item["candidate_id"],
                    "cases": [],
                    "cohorts": {},
                    "explanations": [],
                    "state": {},
                    "all_hard_gates_pass": index == 0,
                }
            )
        result = {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": "0" * 40,
            "stage": "development",
            "status": "passed",
            "claim_sha256": sha256_bytes(claim_raw),
            "protocol_hashes": hashes,
            "baseline": {
                "baseline_id": "current-ngr",
                "cases": [],
                "cohorts": {},
                "state": {},
            },
            "candidates": candidates,
            "selected_candidate_id": candidates[0]["candidate_id"],
            "gates": gates,
            "all_hard_gates_pass": True,
        }
        write_json_exclusive(
            _output_path(synthetic, "development", "runtime_result"), result
        )
        _archive_stage(synthetic, "development")
        phases = verify_phase_state(synthetic)
        if phases["development"] != "archived":
            raise AssertionError("synthetic archive round trip failed")
        return phases


def _verify_hash_registry(root: Path, registry: Mapping[str, Any]) -> None:
    if not registry:
        raise ValueError("hash registry must not be empty")
    for relative, expected in registry.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("hash registry shape mismatch")
        path = root / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"missing committed path: {commit}:{path}")
    return completed.stdout


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout.decode("ascii", errors="strict")


def _full_commit(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit must be lowercase full 40-hex")
    return value


def _output_path(protocol: Mapping[str, Any], stage: str, key: str) -> Path:
    outputs = _mapping(_mapping(protocol, "manifest"), "outputs")
    row = _mapping(outputs, stage)
    relative = _string(row, key)
    return _path(protocol, "root") / relative


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be an object")
    return item


def _list(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, dict) for row in item):
        raise ValueError(f"{key} must be an array of objects")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _path(value: Mapping[str, Any], key: str) -> Path:
    item = value.get(key)
    if not isinstance(item, Path):
        raise ValueError(f"{key} must be a path")
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the frozen precision-control protocol"
    )
    parser.add_argument("command", choices=("audit", "probe"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "registered_query_execution_count": 0,
        "phase": verify_phase_state(protocol),
    }
    if args.command == "probe":
        result["archive_round_trip"] = prove_archive_round_trip()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

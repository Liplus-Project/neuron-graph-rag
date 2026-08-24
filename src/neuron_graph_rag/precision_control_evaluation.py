from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    "gates",
    "all_hard_gates_pass",
)
CASE_FIELDS = ("case_id", "cohort", "ranked_hits", "returned_source_paths")
HIT_FIELDS = (
    "source_path",
    "rank",
    "final_score",
    "entry_score",
    "normalized_graph_score",
    "source_provenance",
    "relation_paths",
)
SOURCE_FIELDS = ("repository", "commit", "path", "content_sha256")
RELATION_PATH_FIELDS = ("seed_path", "target_path", "edge_type", "step_count")
COHORT_FIELDS = ("cohort", "case_ids", "mrr", "hit_at_5")
EXPLANATION_FIELDS = ("case_id", "decisions")
DECISION_FIELDS = (
    "candidate_id",
    "source_path",
    "accepted",
    "applied_rules",
    "thresholds",
    "pre_filter_rank",
    "pre_filter_score",
    "top_score",
    "top_score_ratio",
    "top_score_margin",
    "entry_signal_present",
    "graph_signal_present",
    "rule_results",
    "source_provenance",
)
STATE_FIELDS = (
    "fresh_database_id",
    "replay_database_id",
    "ranking_sha256",
    "replay_ranking_sha256",
    "score_sha256",
    "replay_score_sha256",
    "activation_sha256",
    "replay_activation_sha256",
    "edge_sha256_before",
    "edge_sha256_after",
    "feedback_count_before",
    "feedback_count_after",
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
    exact_schema = {
        "case_required_fields": CASE_FIELDS,
        "hit_required_fields": HIT_FIELDS,
        "source_required_fields": SOURCE_FIELDS,
        "relation_path_required_fields": RELATION_PATH_FIELDS,
        "cohort_required_fields": COHORT_FIELDS,
        "explanation_required_fields": EXPLANATION_FIELDS,
        "decision_required_fields": DECISION_FIELDS,
        "state_required_fields": STATE_FIELDS,
    }
    for key, fields in exact_schema.items():
        if tuple(schema.get(key, ())) != fields:
            raise ValueError(f"{key} mismatch")
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


def evaluate_result_payload(
    protocol: Mapping[str, Any],
    stage: str,
    claim_raw: bytes,
    baseline_raw: Mapping[str, Any],
    candidate_raws: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive every aggregate, gate, status, and selection from raw result rows."""

    if stage not in STAGES:
        raise ValueError("unknown stage")
    claim = json.loads(claim_raw.decode("utf-8", errors="strict"))
    if not isinstance(claim, dict) or set(claim) != set(CLAIM_FIELDS):
        raise ValueError("claim fields must exactly match the frozen schema")
    hashes = dict(_mapping(_mapping(protocol, "manifest"), "artifact_sha256"))
    if claim != {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": claim.get("protocol_commit"),
        "stage": stage,
        "protocol_hashes": hashes,
        "one_time_claim": True,
    }:
        raise ValueError("claim does not match the frozen protocol")
    if not re.fullmatch(r"[0-9a-f]{40}", str(claim["protocol_commit"])):
        raise ValueError("claim protocol commit must be lowercase full 40-hex")

    baseline = _validate_baseline_raw(protocol, stage, baseline_raw)
    baseline["cohorts"] = _compute_cohorts(protocol, stage, baseline["cases"])
    expected_controls = [
        PrecisionControl.from_mapping(dict(item))
        for item in _list(_mapping(protocol, "candidates"), "candidates")
    ]
    if len(candidate_raws) != len(expected_controls):
        raise ValueError("candidate raw result count mismatch")
    candidates: list[dict[str, Any]] = []
    for raw, control in zip(candidate_raws, expected_controls, strict=True):
        candidate = _validate_candidate_raw(protocol, stage, raw, control, baseline)
        candidate["cohorts"] = _compute_cohorts(protocol, stage, candidate["cases"])
        candidate["gates"] = _candidate_gates(protocol, stage, baseline, candidate)
        candidate["all_hard_gates_pass"] = all(
            gate["passed"] for gate in candidate["gates"]
        )
        candidates.append(candidate)

    database_ids = [
        state[key]
        for state in [baseline["state"], *[item["state"] for item in candidates]]
        for key in ("fresh_database_id", "replay_database_id")
    ]
    if len(set(database_ids)) != len(database_ids):
        for candidate in candidates:
            candidate["gates"][2]["passed"] = False
            candidate["all_hard_gates_pass"] = False

    selected = next(
        (item["candidate_id"] for item in candidates if item["all_hard_gates_pass"]),
        None,
    )
    if selected is not None:
        selected_row = next(
            item for item in candidates if item["candidate_id"] == selected
        )
        global_gates = copy_json(selected_row["gates"])
    else:
        global_gates = []
        for index, gate_id in enumerate(GATE_IDS):
            global_gates.append(
                {
                    "gate_id": gate_id,
                    "hard": True,
                    "passed": all(
                        item["gates"][index]["passed"] for item in candidates
                    ),
                    "details": {"aggregation": "all-candidates-when-none-selected"},
                }
            )
    all_pass = selected is not None and all(gate["passed"] for gate in global_gates)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": claim["protocol_commit"],
        "stage": stage,
        "status": "passed" if all_pass else "failed",
        "claim_sha256": sha256_bytes(claim_raw),
        "protocol_hashes": hashes,
        "baseline": baseline,
        "candidates": candidates,
        "selected_candidate_id": selected,
        "gates": global_gates,
        "all_hard_gates_pass": all_pass,
    }


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _validate_baseline_raw(
    protocol: Mapping[str, Any], stage: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    if set(raw) != {"baseline_id", "cases", "state"}:
        raise ValueError("baseline raw fields mismatch")
    if raw.get("baseline_id") != "current-ngr":
        raise ValueError("baseline identity mismatch")
    cases = _validate_case_rows(protocol, stage, raw.get("cases"))
    for case in cases:
        if case["returned_source_paths"] != [
            hit["source_path"] for hit in case["ranked_hits"][:5]
        ]:
            raise ValueError("baseline must preserve the current top-five ranking")
    state = _validate_state(raw.get("state"))
    return {"baseline_id": "current-ngr", "cases": cases, "state": state}


def _validate_candidate_raw(
    protocol: Mapping[str, Any],
    stage: str,
    raw: Mapping[str, Any],
    control: PrecisionControl,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if set(raw) != {"candidate_id", "cases", "explanations", "state"}:
        raise ValueError("candidate raw fields mismatch")
    if raw.get("candidate_id") != control.candidate_id:
        raise ValueError("candidate raw identity/order mismatch")
    cases = _validate_case_rows(protocol, stage, raw.get("cases"))
    if [case["ranked_hits"] for case in cases] != [
        case["ranked_hits"] for case in baseline["cases"]
    ]:
        raise ValueError("candidate changed pre-filter scores, ranks, or provenance")
    explanations = raw.get("explanations")
    if not isinstance(explanations, list) or len(explanations) != len(cases):
        raise ValueError("candidate explanations must cover every case")
    for case, explanation in zip(cases, explanations, strict=True):
        if not isinstance(explanation, dict) or set(explanation) != set(
            EXPLANATION_FIELDS
        ):
            raise ValueError("candidate explanation row shape mismatch")
        if explanation.get("case_id") != case["case_id"]:
            raise ValueError("candidate explanation case mismatch")
        decisions = explanation.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(
            case["ranked_hits"]
        ):
            raise ValueError("candidate decisions must cover every ranked hit")
        expected_decisions = [
            _decision_from_hit(control, hit, case["ranked_hits"][0]["final_score"])
            for hit in case["ranked_hits"]
        ]
        if decisions != expected_decisions:
            raise ValueError("candidate decision is not recomputable from raw hit data")
        accepted = [
            decision["source_path"]
            for decision in expected_decisions
            if decision["accepted"]
        ][:5]
        if case["returned_source_paths"] != accepted:
            raise ValueError("candidate returned paths do not match filter decisions")
    state = _validate_state(raw.get("state"))
    return {
        "candidate_id": control.candidate_id,
        "cases": cases,
        "explanations": copy_json(explanations),
        "state": state,
    }


def _validate_case_rows(
    protocol: Mapping[str, Any], stage: str, value: object
) -> list[dict[str, Any]]:
    queries = _list(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    if not isinstance(value, list) or len(value) != len(queries):
        raise ValueError("case rows must exactly cover the registered stage")
    corpus = _mapping(protocol, "corpus")
    repository = _string(corpus, "repository")
    commit = _string(corpus, "commit")
    documents = {
        _string(item, "path"): _string(item, "content_sha256")
        for item in _list(corpus, "documents")
    }
    normalized: list[dict[str, Any]] = []
    for row, query in zip(value, queries, strict=True):
        if not isinstance(row, dict) or set(row) != set(CASE_FIELDS):
            raise ValueError("case result shape mismatch")
        if (
            row.get("case_id") != query["case_id"]
            or row.get("cohort") != query["cohort"]
        ):
            raise ValueError("case result identity/order mismatch")
        hits = row.get("ranked_hits")
        if not isinstance(hits, list) or len(hits) != len(documents):
            raise ValueError("ranked hits must cover the complete frozen corpus")
        hit_paths: list[str] = []
        prior_score = math.inf
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict) or set(hit) != set(HIT_FIELDS):
                raise ValueError("ranked hit shape mismatch")
            path = _string(hit, "source_path")
            if path not in documents or path in hit_paths or hit.get("rank") != rank:
                raise ValueError("ranked hit identity/rank mismatch")
            hit_paths.append(path)
            scores = [
                _finite_number(hit.get(key), key)
                for key in (
                    "final_score",
                    "entry_score",
                    "normalized_graph_score",
                )
            ]
            if scores[0] > prior_score:
                raise ValueError("ranked hit scores must be non-increasing")
            prior_score = scores[0]
            source = hit.get("source_provenance")
            if not isinstance(source, dict) or set(source) != set(SOURCE_FIELDS):
                raise ValueError("source provenance shape mismatch")
            if source != {
                "repository": repository,
                "commit": commit,
                "path": path,
                "content_sha256": documents[path],
            }:
                raise ValueError("source provenance does not match frozen corpus")
            relation_paths = hit.get("relation_paths")
            if not isinstance(relation_paths, list):
                raise ValueError("relation paths must be an array")
            for relation in relation_paths:
                if not isinstance(relation, dict) or set(relation) != set(
                    RELATION_PATH_FIELDS
                ):
                    raise ValueError("relation path shape mismatch")
                if relation.get("step_count") != 1:
                    raise ValueError("relation provenance must be edge-only")
                if (
                    relation.get("seed_path") not in documents
                    or relation.get("target_path") not in documents
                ):
                    raise ValueError("relation path is outside the corpus")
                _string(relation, "edge_type")
        returned = row.get("returned_source_paths")
        if (
            not isinstance(returned, list)
            or len(returned) > 5
            or len(set(returned)) != len(returned)
            or any(path not in hit_paths for path in returned)
        ):
            raise ValueError("returned source paths are invalid")
        normalized.append(copy_json(row))
    return normalized


def _validate_state(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(STATE_FIELDS):
        raise ValueError("state result shape mismatch")
    for key in STATE_FIELDS[2:10]:
        if not isinstance(value.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[key]
        ):
            raise ValueError(f"state hash mismatch: {key}")
    first_id = value.get("fresh_database_id")
    replay_id = value.get("replay_database_id")
    if (
        not isinstance(first_id, str)
        or not first_id
        or not isinstance(replay_id, str)
        or not replay_id
        or first_id == replay_id
    ):
        raise ValueError("fresh and replay database identities must differ")
    for key in ("feedback_count_before", "feedback_count_after"):
        if not isinstance(value.get(key), int) or isinstance(value[key], bool):
            raise ValueError("feedback counts must be integers")
    return copy_json(value)


def _decision_from_hit(
    control: PrecisionControl, hit: Mapping[str, Any], top_score: float
) -> dict[str, Any]:
    score = float(hit["final_score"])
    ratio = score / top_score if top_score > 0.0 else 0.0
    margin = top_score - score
    entry_signal = float(hit["entry_score"]) > 0.0
    graph_signal = float(hit["normalized_graph_score"]) > 0.0 and bool(
        hit["relation_paths"]
    )
    rule_results: dict[str, bool] = {}
    if control.minimum_final_score is not None:
        rule_results[RULES[0]] = score >= control.minimum_final_score
    if control.minimum_top_score_ratio is not None:
        rule_results[RULES[1]] = ratio >= control.minimum_top_score_ratio
    if control.maximum_top_score_margin is not None:
        rule_results[RULES[2]] = margin <= control.maximum_top_score_margin
    if control.require_entry_graph_signal_agreement:
        rule_results[RULES[3]] = entry_signal and graph_signal
    return {
        "candidate_id": control.candidate_id,
        "source_path": hit["source_path"],
        "accepted": all(rule_results.values()),
        "applied_rules": list(control.rules()),
        "thresholds": control.thresholds(),
        "pre_filter_rank": hit["rank"],
        "pre_filter_score": score,
        "top_score": top_score,
        "top_score_ratio": ratio,
        "top_score_margin": margin,
        "entry_signal_present": entry_signal,
        "graph_signal_present": graph_signal,
        "rule_results": rule_results,
        "source_provenance": copy_json(hit["source_provenance"]),
    }


def _compute_cohorts(
    protocol: Mapping[str, Any], stage: str, cases: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    gold_rows = {
        row["case_id"]: row
        for row in _list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        cohort_cases = [case for case in cases if case["cohort"] == cohort]
        reciprocal_ranks: list[float] = []
        hits: list[float] = []
        for case in cohort_cases:
            returned = case["returned_source_paths"]
            gold = gold_rows[case["case_id"]]
            paths = (
                gold["forbidden_paths"]
                if cohort == "negative_control"
                else gold["expected_paths"]
            )
            ranks = [returned.index(path) + 1 for path in paths if path in returned]
            best = min(ranks) if ranks else None
            reciprocal_ranks.append(0.0 if best is None else 1.0 / best)
            hits.append(0.0 if best is None else 1.0)
        rows.append(
            {
                "cohort": cohort,
                "case_ids": [case["case_id"] for case in cohort_cases],
                "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "hit_at_5": sum(hits) / len(hits),
            }
        )
    return rows


def _candidate_gates(
    protocol: Mapping[str, Any],
    stage: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    candidate_cases = {case["case_id"]: case for case in candidate["cases"]}
    gold_rows = {
        row["case_id"]: row
        for row in _list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }

    def expected_rank(case: Mapping[str, Any]) -> int | None:
        expected = gold_rows[case["case_id"]]["expected_paths"]
        if not expected or expected[0] not in case["returned_source_paths"]:
            return None
        return case["returned_source_paths"].index(expected[0]) + 1

    def non_regression(cohort: str) -> bool:
        for case_id, before in baseline_cases.items():
            if before["cohort"] != cohort:
                continue
            before_rank = expected_rank(before)
            after_rank = expected_rank(candidate_cases[case_id])
            if before_rank is None or after_rank is None or after_rank > before_rank:
                return False
        return True

    baseline_cohorts = {row["cohort"]: row for row in baseline["cohorts"]}
    candidate_cohorts = {row["cohort"]: row for row in candidate["cohorts"]}
    cohort_non_regression = all(
        candidate_cohorts[cohort][metric] >= baseline_cohorts[cohort][metric]
        for cohort in COHORTS[:3]
        for metric in ("mrr", "hit_at_5")
    )
    negative_improvement = True
    for case_id, before in baseline_cases.items():
        if before["cohort"] != "negative_control":
            continue
        forbidden = gold_rows[case_id]["forbidden_paths"]
        before_ranks = [
            before["returned_source_paths"].index(path) + 1
            for path in forbidden
            if path in before["returned_source_paths"]
        ]
        after = candidate_cases[case_id]
        after_ranks = [
            after["returned_source_paths"].index(path) + 1
            for path in forbidden
            if path in after["returned_source_paths"]
        ]
        if not before_ranks or len(after_ranks) >= len(before_ranks):
            negative_improvement = False
            break
        if after_ranks and min(after_ranks) <= min(before_ranks):
            negative_improvement = False
            break
    completeness = all(
        all(
            path in candidate_cases[case_id]["returned_source_paths"]
            for path in row["expected_paths"]
        )
        and all(
            path in baseline_cases[case_id]["returned_source_paths"]
            for path in row["expected_paths"]
        )
        for case_id, row in gold_rows.items()
        if row["cohort"] != "negative_control"
    )
    relation_provenance = True
    for case_id, gold in gold_rows.items():
        if gold["cohort"] != "relation_linked":
            continue
        expected = gold["expected_paths"][0]
        required = {
            "seed_path": gold["relation_seed_path"],
            "target_path": expected,
            "edge_type": gold["relation_edge_type"],
            "step_count": 1,
        }
        for cases in (baseline_cases, candidate_cases):
            case = cases[case_id]
            hit = next(
                item for item in case["ranked_hits"] if item["source_path"] == expected
            )
            if (
                expected not in case["returned_source_paths"]
                or required not in hit["relation_paths"]
            ):
                relation_provenance = False
    states = [baseline["state"], candidate["state"]]
    deterministic = all(
        state["ranking_sha256"] == state["replay_ranking_sha256"]
        and state["score_sha256"] == state["replay_score_sha256"]
        and state["activation_sha256"] == state["replay_activation_sha256"]
        for state in states
    )
    before_state = baseline["state"]
    after_state = candidate["state"]
    immutable = (
        after_state["score_sha256"] == before_state["score_sha256"]
        and after_state["activation_sha256"] == before_state["activation_sha256"]
        and after_state["edge_sha256_before"] == before_state["edge_sha256_before"]
        and all(
            state["edge_sha256_before"] == state["edge_sha256_after"]
            and state["feedback_count_before"] == 0
            and state["feedback_count_after"] == 0
            for state in states
        )
    )
    passes = (
        True,
        True,
        deterministic,
        non_regression("direct_lexical"),
        non_regression("semantic_paraphrase"),
        non_regression("relation_linked"),
        cohort_non_regression,
        negative_improvement,
        completeness,
        relation_provenance,
        immutable,
    )
    return [
        {
            "gate_id": gate_id,
            "hard": True,
            "passed": passed,
            "details": {"evaluator": "raw-result-v1"},
        }
        for gate_id, passed in zip(GATE_IDS, passes, strict=True)
    ]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def verify_result_payload(
    protocol: Mapping[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    claim_raw: bytes,
) -> None:
    if set(payload) != set(RESULT_FIELDS):
        raise ValueError("result fields must exactly match the frozen schema")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict) or set(baseline) != set(BASELINE_FIELDS):
        raise ValueError("baseline shape mismatch")
    _validate_cohort_rows(baseline.get("cohorts"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or any(
        not isinstance(item, dict) or set(item) != set(CANDIDATE_FIELDS)
        for item in rows
    ):
        raise ValueError("result candidate shape mismatch")
    for item in rows:
        _validate_cohort_rows(item.get("cohorts"))
        _validate_gate_rows(item.get("gates"))
        if not isinstance(item.get("all_hard_gates_pass"), bool):
            raise ValueError("candidate hard gate summary must be boolean")
    _validate_gate_rows(payload.get("gates"))
    if not isinstance(payload.get("all_hard_gates_pass"), bool):
        raise ValueError("global hard gate summary must be boolean")
    if payload.get("status") not in {"passed", "failed"}:
        raise ValueError("result status mismatch")
    if payload.get("selected_candidate_id") is not None and not isinstance(
        payload["selected_candidate_id"], str
    ):
        raise ValueError("selected candidate type mismatch")
    expected = evaluate_result_payload(
        protocol,
        stage,
        claim_raw,
        {
            "baseline_id": baseline["baseline_id"],
            "cases": baseline["cases"],
            "state": baseline["state"],
        },
        [
            {
                "candidate_id": item["candidate_id"],
                "cases": item["cases"],
                "explanations": item["explanations"],
                "state": item["state"],
            }
            for item in rows
        ],
    )
    if dict(payload) != expected:
        raise ValueError("result does not match evaluator recomputation")


def _validate_cohort_rows(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(COHORTS):
        raise ValueError("cohort rows must cover every frozen cohort")
    for cohort, row in zip(COHORTS, value, strict=True):
        if not isinstance(row, dict) or set(row) != set(COHORT_FIELDS):
            raise ValueError("cohort row shape mismatch")
        if row.get("cohort") != cohort:
            raise ValueError("cohort row order mismatch")
        case_ids = row.get("case_ids")
        if (
            not isinstance(case_ids, list)
            or len(case_ids) != 2
            or any(not isinstance(case_id, str) for case_id in case_ids)
        ):
            raise ValueError("cohort case ids must contain exactly two strings")
        _finite_number(row.get("mrr"), "mrr")
        _finite_number(row.get("hit_at_5"), "hit_at_5")


def _validate_gate_rows(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(GATE_IDS):
        raise ValueError("gate rows must cover every frozen hard gate")
    for gate_id, row in zip(GATE_IDS, value, strict=True):
        if not isinstance(row, dict) or set(row) != set(GATE_FIELDS):
            raise ValueError("gate row shape mismatch")
        if row.get("gate_id") != gate_id or row.get("hard") is not True:
            raise ValueError("gate identity/hard flag mismatch")
        if not isinstance(row.get("passed"), bool):
            raise ValueError("gate passed must be boolean")
        if not isinstance(row.get("details"), dict):
            raise ValueError("gate details must be an object")


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


def build_synthetic_evaluated_result(
    protocol: Mapping[str, Any], stage: str, claim_raw: bytes
) -> dict[str, Any]:
    """Build non-empty result-shaped mechanics data without executing a query."""

    corpus = _mapping(protocol, "corpus")
    repository = _string(corpus, "repository")
    commit = _string(corpus, "commit")
    documents = _list(corpus, "documents")
    content_hashes = {
        _string(item, "path"): _string(item, "content_sha256") for item in documents
    }
    all_paths = list(content_hashes)
    queries = _list(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    gold = {
        row["case_id"]: row
        for row in _list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }
    scores = [0.9, 0.8, 0.7, 0.6, 0.1] + [0.09 - (index * 0.004) for index in range(15)]
    baseline_cases: list[dict[str, Any]] = []
    for query in queries:
        row = gold[query["case_id"]]
        if query["cohort"] == "negative_control":
            forbidden = row["forbidden_paths"][0]
            leading = [path for path in all_paths if path != forbidden][:4]
            ordered_paths = (
                leading
                + [forbidden]
                + [
                    path
                    for path in all_paths
                    if path not in leading and path != forbidden
                ]
            )
        else:
            expected = row["expected_paths"][0]
            ordered_paths = [expected] + [
                path for path in all_paths if path != expected
            ]
        ranked_hits = []
        for rank, (path, score) in enumerate(
            zip(ordered_paths, scores, strict=True), start=1
        ):
            relation_paths: list[dict[str, Any]] = []
            if (
                query["cohort"] == "relation_linked"
                and path == row["expected_paths"][0]
            ):
                relation_paths.append(
                    {
                        "seed_path": row["relation_seed_path"],
                        "target_path": path,
                        "edge_type": row["relation_edge_type"],
                        "step_count": 1,
                    }
                )
            ranked_hits.append(
                {
                    "source_path": path,
                    "rank": rank,
                    "final_score": score,
                    "entry_score": 0.5,
                    "normalized_graph_score": 0.4 if relation_paths else 0.0,
                    "source_provenance": {
                        "repository": repository,
                        "commit": commit,
                        "path": path,
                        "content_sha256": content_hashes[path],
                    },
                    "relation_paths": relation_paths,
                }
            )
        baseline_cases.append(
            {
                "case_id": query["case_id"],
                "cohort": query["cohort"],
                "ranked_hits": ranked_hits,
                "returned_source_paths": ordered_paths[:5],
            }
        )
    common_score = sha256_bytes(b"synthetic-score")
    common_activation = sha256_bytes(b"synthetic-activation")
    common_edge = sha256_bytes(b"synthetic-edge")

    def state(arm: str) -> dict[str, Any]:
        ranking = sha256_bytes(f"synthetic-ranking:{arm}".encode("ascii"))
        return {
            "fresh_database_id": f"synthetic:{stage}:{arm}:primary",
            "replay_database_id": f"synthetic:{stage}:{arm}:replay",
            "ranking_sha256": ranking,
            "replay_ranking_sha256": ranking,
            "score_sha256": common_score,
            "replay_score_sha256": common_score,
            "activation_sha256": common_activation,
            "replay_activation_sha256": common_activation,
            "edge_sha256_before": common_edge,
            "edge_sha256_after": common_edge,
            "feedback_count_before": 0,
            "feedback_count_after": 0,
        }

    candidate_raws = []
    for item in _list(_mapping(protocol, "candidates"), "candidates"):
        control = PrecisionControl.from_mapping(dict(item))
        cases = copy_json(baseline_cases)
        explanations = []
        for case in cases:
            top_score = case["ranked_hits"][0]["final_score"]
            decisions = [
                _decision_from_hit(control, hit, top_score)
                for hit in case["ranked_hits"]
            ]
            case["returned_source_paths"] = [
                decision["source_path"]
                for decision in decisions
                if decision["accepted"]
            ][:5]
            explanations.append({"case_id": case["case_id"], "decisions": decisions})
        candidate_raws.append(
            {
                "candidate_id": control.candidate_id,
                "cases": cases,
                "explanations": explanations,
                "state": state(control.candidate_id),
            }
        )
    return evaluate_result_payload(
        protocol,
        stage,
        claim_raw,
        {
            "baseline_id": "current-ngr",
            "cases": baseline_cases,
            "state": state("current-ngr"),
        },
        candidate_raws,
    )


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
        result = build_synthetic_evaluated_result(protocol, "development", claim_raw)
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

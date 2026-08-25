from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STEM = "github_cross_encoder_precision_v2"
PROTOCOL_ID = "github-ngr-cross-encoder-precision-v2"
STAGES = ("development", "holdout")
COHORTS = (
    "direct_lexical",
    "semantic_paraphrase",
    "relation_linked",
    "negative_control",
)
CANDIDATE_IDS = (
    "bge-base-rrf-threshold",
    "bge-base-ce-threshold",
    "bge-v2-m3-rrf-threshold",
    "bge-v2-m3-ce-threshold",
)
GATE_IDS = (
    "protocol-source-model-lock-integrity",
    "identity-separation",
    "cpu-offline-deterministic-fresh-process",
    "positive-case-rank-non-regression",
    "positive-cohort-mrr-hit-at-5-non-regression",
    "negative-non-worsening-and-aggregate-strict-improvement",
    "positive-expected-source-top-5-completeness",
    "relation-source-edge-only-provenance",
    "cross-encoder-fusion-threshold-rank-recomputation",
    "ngr-prefilter-state-immutability",
    "default-surface-immutability",
)
ARTIFACT_KINDS = (
    "corpus",
    "queries",
    "gold",
    "models",
    "candidates",
    "gate",
    "result-schema",
    "result-free-audit",
    "requirements.lock",
)
CLAIM_FIELDS = {
    "protocol_id",
    "protocol_commit",
    "stage",
    "protocol_hashes",
    "one_time_claim",
}
RESULT_FIELDS = {
    "protocol_id",
    "protocol_commit",
    "stage",
    "status",
    "claim_sha256",
    "protocol_hashes",
    "baseline",
    "models",
    "candidates",
    "selected_candidate_id",
    "gates",
    "all_hard_gates_pass",
}
ERROR_FIELDS = {"protocol_id", "stage", "claim_sha256", "error"}
TRANSPORT_FIELDS = {
    "protocol_id",
    "stage",
    "reason",
    "stage_execution_count",
    "files",
}
TRANSPORT_FILE_FIELDS = {
    "runtime_path",
    "archive_path",
    "sha256",
    "byte_identity",
}
STATE_FIELDS = {
    "fresh_database_id",
    "replay_database_id",
    "ranking_sha256",
    "replay_ranking_sha256",
    "activation_sha256",
    "replay_activation_sha256",
    "edge_sha256_before",
    "edge_sha256_after",
    "feedback_count_before",
    "feedback_count_after",
    "sqlite_sha256_before",
    "sqlite_sha256_after",
    "cpu_only",
    "offline",
    "fresh_process",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
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


def project_passages(
    text: str, window: int = 480, overlap: int = 80
) -> list[dict[str, Any]]:
    if window != 480 or overlap != 80 or overlap >= window:
        raise ValueError("only the frozen 480/80 code-point projection is allowed")
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(text):
        chunk = text[start : start + window]
        if chunk:
            rows.append(
                {
                    "chunk_index": index,
                    "start_codepoint": start,
                    "end_codepoint": start + len(chunk),
                    "text": chunk,
                }
            )
        if start + window >= len(text):
            break
        start += window - overlap
        index += 1
    return rows


def sigmoid(raw_logit: float) -> float:
    if not math.isfinite(raw_logit):
        raise ValueError("cross-encoder logit must be finite")
    if raw_logit >= 0:
        z = math.exp(-raw_logit)
        return 1.0 / (1.0 + z)
    z = math.exp(raw_logit)
    return z / (1.0 + z)


def load_protocol(root: Path = ROOT) -> dict[str, Any]:
    fixture_root = root / "tests" / "fixtures"
    protocol: dict[str, Any] = {
        "root": root,
        "source_root": root,
        "manifest_path": fixture_root / f"{STEM}.manifest.json",
        "manifest": read_json(fixture_root / f"{STEM}.manifest.json"),
    }
    for kind in ARTIFACT_KINDS:
        key = kind.replace(".", "_").replace("-", "_")
        suffix = "" if kind == "requirements.lock" else ".json"
        path = fixture_root / f"{STEM}.{kind}{suffix}"
        protocol[key] = (
            path.read_text(encoding="utf-8", errors="strict")
            if kind == "requirements.lock"
            else read_json(path)
        )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    manifest = _mapping(protocol, "manifest")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("manifest protocol_id mismatch")
    _verify_hash_registry(root, _mapping(manifest, "artifact_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "predecessor_immutable_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "v1_immutable_sha256"))
    for kind in ARTIFACT_KINDS[:-1]:
        artifact = _mapping(protocol, kind.replace("-", "_").replace(".", "_"))
        if artifact.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"protocol_id mismatch in {kind}")
    corpus = _mapping(protocol, "corpus")
    source = _mapping(manifest, "source")
    if corpus.get("repository") != source.get("repository") or corpus.get(
        "commit"
    ) != source.get("commit"):
        raise ValueError("source registry mismatch")
    documents = _object_list(corpus, "documents")
    if len(documents) != 24:
        raise ValueError("frozen corpus must contain exactly 24 documents")
    paths: set[str] = set()
    for row in documents:
        if set(row) != {"path", "content_sha256"}:
            raise ValueError("corpus document shape mismatch")
        path = _string(row, "path")
        if path in paths or not path.endswith(".md"):
            raise ValueError("corpus paths must be unique Markdown files")
        paths.add(path)
        if sha256_bytes(_git_bytes(root, _string(corpus, "commit"), path)) != _string(
            row, "content_sha256"
        ):
            raise ValueError(f"source content hash mismatch: {path}")
    relationships: set[tuple[str, str, str]] = set()
    for relation in _object_list(corpus, "relationships"):
        if set(relation) != {"source_path", "target_path", "edge_type"}:
            raise ValueError("frozen relationship shape mismatch")
        key = (
            _string(relation, "source_path"),
            _string(relation, "target_path"),
            _string(relation, "edge_type"),
        )
        if (
            key in relationships
            or key[0] not in paths
            or key[1] not in paths
            or key[0] == key[1]
            or key[2] != "informs"
        ):
            raise ValueError("frozen relationship must be a unique in-corpus edge")
        relationships.add(key)
    _validate_cases(protocol, paths, relationships)
    models = _object_list(_mapping(protocol, "models"), "models")
    if [row.get("model_id") for row in models] != [
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3",
    ]:
        raise ValueError("model order mismatch")
    expected_models = (
        (
            "BAAI/bge-reranker-base",
            "2cfc18c9415c912f9d8155881c133215df768a70",
            "mit",
            1_112_206_140,
        ),
        (
            "BAAI/bge-reranker-v2-m3",
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "apache-2.0",
            2_271_071_852,
        ),
    )
    for row, (model_id, revision, license_id, weight_size) in zip(
        models, expected_models, strict=True
    ):
        if set(row) != {
            "model_id",
            "revision",
            "license",
            "repository_api",
            "model_safetensors_size",
            "required_files",
        }:
            raise ValueError("model registry shape mismatch")
        if (
            row.get("model_id") != model_id
            or row.get("revision") != revision
            or row.get("license") != license_id
            or row.get("model_safetensors_size") != weight_size
            or row.get("repository_api")
            != f"https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true"
        ):
            raise ValueError("model repository metadata mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("revision"))):
            raise ValueError("model revision must be immutable")
        files = _object_list(row, "required_files")
        if not files or any(
            set(item) != {"path", "size", "git_blob_id", "lfs_sha256"} for item in files
        ):
            raise ValueError("model file registry shape mismatch")
        if any(
            not isinstance(item["size"], int) or item["size"] <= 0 for item in files
        ):
            raise ValueError("model file sizes must be positive")
    candidates = _object_list(_mapping(protocol, "candidates"), "candidates")
    if tuple(row.get("candidate_id") for row in candidates) != CANDIDATE_IDS:
        raise ValueError("candidate order mismatch")
    schema = _mapping(protocol, "result_schema")
    for key, fields in (
        ("claim_exact_fields", CLAIM_FIELDS),
        ("result_exact_fields", RESULT_FIELDS),
        ("error_exact_fields", ERROR_FIELDS),
        ("transport_exact_fields", TRANSPORT_FIELDS),
        ("transport_file_exact_fields", TRANSPORT_FILE_FIELDS),
    ):
        value = schema.get(key)
        if (
            not isinstance(value, list)
            or len(value) != len(fields)
            or set(value) != fields
        ):
            raise ValueError(f"{key} does not exactly freeze lifecycle fields")
    passage = _mapping(schema, "passage")
    execution = _mapping(schema, "execution")
    if passage != {
        "window_codepoints": 480,
        "overlap_codepoints": 80,
        "input": ["query", "chunk_text"],
        "padding": True,
        "truncation": True,
        "max_length": 512,
        "document_score": "max raw logit; earliest chunk wins ties",
    }:
        raise ValueError("passage projection contract mismatch")
    if execution != {
        "device": "cpu",
        "dtype": "float32",
        "eval": True,
        "no_grad": True,
        "batch_size": 8,
    }:
        raise ValueError("execution contract mismatch")
    gates = _object_list(_mapping(protocol, "gate"), "gates")
    if tuple(row.get("gate_id") for row in gates) != GATE_IDS or any(
        set(row) != {"gate_id", "hard"} or row["hard"] is not True for row in gates
    ):
        raise ValueError("hard gate contract mismatch")
    audit = _mapping(protocol, "result_free_audit")
    for key in (
        "freeze_registered_query_execution_count",
        "freeze_model_inference_count",
        "freeze_observed_result_count",
    ):
        if audit.get(key) != 0:
            raise ValueError(f"{key} must remain zero at freeze")
    if audit.get("count_scope") != (
        "v2 result-free freeze only; historical v1 observation excluded"
    ):
        raise ValueError("result-free count scope mismatch")
    for key in (
        "shared_database_opened",
        "existing_experiment_database_opened",
        "github_rag_mcp_called",
        "feedback_or_outcome_recorded",
        "model_weights_downloaded",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"{key} must remain false at freeze")


def _validate_cases(
    protocol: Mapping[str, Any],
    paths: set[str],
    relationships: set[tuple[str, str, str]],
) -> None:
    queries = _mapping(_mapping(protocol, "queries"), "stages")
    gold = _mapping(_mapping(protocol, "gold"), "stages")
    all_case_ids: set[str] = set()
    all_queries: set[str] = set()
    identities: dict[str, set[str]] = {}
    for stage in STAGES:
        cases = _object_list(queries, stage)
        rows = _object_list(gold, stage)
        if len(cases) != 8 or len(rows) != 8:
            raise ValueError("each stage requires exactly eight cases")
        if tuple(row.get("cohort") for row in cases) != tuple(
            c for c in COHORTS for _ in range(2)
        ):
            raise ValueError("cohort order mismatch")
        by_id = {_string(row, "case_id"): row for row in rows}
        stage_ids: set[str] = set()
        for case in cases:
            if set(case) != {"case_id", "cohort", "language", "query"}:
                raise ValueError("query case shape mismatch")
            case_id = _string(case, "case_id")
            query_text = _string(case, "query")
            if (
                case_id in all_case_ids
                or case_id not in by_id
                or query_text in all_queries
            ):
                raise ValueError("query identity mismatch")
            all_case_ids.add(case_id)
            all_queries.add(query_text)
            row = by_id[case_id]
            if row.get("cohort") != case.get("cohort"):
                raise ValueError("query/gold cohort mismatch")
            expected = row.get("expected_paths")
            forbidden = row.get("forbidden_paths")
            if (
                not isinstance(expected, list)
                or not isinstance(forbidden, list)
                or any(path not in paths for path in expected + forbidden)
            ):
                raise ValueError("gold identity outside corpus")
            if case["cohort"] == "negative_control":
                if expected or len(forbidden) != 1:
                    raise ValueError("negative case shape mismatch")
            elif len(expected) != 1 or forbidden:
                raise ValueError("positive case shape mismatch")
            if case["cohort"] == "relation_linked":
                seed = _string(row, "relation_seed_path")
                edge_type = _string(row, "relation_edge_type")
                if (
                    seed not in paths
                    or edge_type != "informs"
                    or (seed, expected[0], edge_type) not in relationships
                ):
                    raise ValueError("relation seed mismatch")
                stage_ids.add(seed)
            stage_ids.update(expected + forbidden)
        identities[stage] = stage_ids
    if not identities["development"].isdisjoint(identities["holdout"]):
        raise ValueError("development and holdout identities overlap")
    for predecessor in ("github_precision_control_v1", "github_retrieval_parity_v1"):
        old_queries = read_json(
            _path(protocol, "root") / f"tests/fixtures/{predecessor}.queries.json"
        )
        old_ids = {
            _string(row, "case_id")
            for stage in STAGES
            for row in _object_list(_mapping(old_queries, "stages"), stage)
        }
        if all_case_ids & old_ids:
            raise ValueError("query identity overlaps predecessor")
        old_corpus = read_json(
            _path(protocol, "root") / f"tests/fixtures/{predecessor}.corpus.json"
        )
        old_paths = {
            _string(row, "path") for row in _object_list(old_corpus, "documents")
        }
        if paths & old_paths:
            raise ValueError("source identity overlaps predecessor")


def verify_protocol_commit(protocol_commit: str, protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    commit = _full_commit(protocol_commit)
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    ):
        raise ValueError("protocol commit must be an origin/main ancestor")
    relative = str(_path(protocol, "manifest_path").relative_to(root)).replace(
        "\\", "/"
    )
    _git_bytes(root, commit, relative)
    parent = _git_output(root, "rev-parse", f"{commit}^1").strip()
    if (
        subprocess.run(
            ["git", "cat-file", "-e", f"{parent}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    ):
        raise ValueError(
            "protocol manifest must be introduced by the freeze merge commit"
        )
    committed = json.loads(
        _git_bytes(root, commit, relative).decode("utf-8", errors="strict")
    )
    for path, expected in _mapping(committed, "artifact_sha256").items():
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


def evaluate_result_payload(
    protocol: Mapping[str, Any],
    stage: str,
    claim_raw: bytes,
    baseline_raw: Mapping[str, Any],
    model_raws: list[Mapping[str, Any]],
) -> dict[str, Any]:
    claim = _validate_claim(protocol, stage, claim_raw)
    baseline = _validate_baseline(protocol, stage, baseline_raw)
    models = _validate_models(protocol, stage, model_raws, baseline)
    state_rows = [baseline["state"], *[model["state"] for model in models]]
    candidates = [
        _derive_candidate(protocol, stage, baseline, models, candidate_id, state_rows)
        for candidate_id in CANDIDATE_IDS
    ]
    selected = next(
        (row["candidate_id"] for row in candidates if row["all_hard_gates_pass"]), None
    )
    global_gates = copy.deepcopy(
        next(
            (row["gates"] for row in candidates if row["candidate_id"] == selected),
            candidates[0]["gates"],
        )
    )
    all_pass = selected is not None
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": claim["protocol_commit"],
        "stage": stage,
        "status": "passed" if all_pass else "failed",
        "claim_sha256": sha256_bytes(claim_raw),
        "protocol_hashes": claim["protocol_hashes"],
        "baseline": baseline,
        "models": models,
        "candidates": candidates,
        "selected_candidate_id": selected,
        "gates": global_gates,
        "all_hard_gates_pass": all_pass,
    }


def verify_result_payload(
    protocol: Mapping[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    claim_raw: bytes,
) -> None:
    if set(payload) != RESULT_FIELDS:
        raise ValueError("result fields must exactly match the frozen schema")
    if payload.get("status") not in ("passed", "failed") or not isinstance(
        payload.get("all_hard_gates_pass"), bool
    ):
        raise ValueError("result status shape mismatch")
    _validate_gate_rows(_object_list(payload, "gates"))
    candidates = _object_list(payload, "candidates")
    if tuple(row.get("candidate_id") for row in candidates) != CANDIDATE_IDS:
        raise ValueError("result candidate order mismatch")
    for candidate in candidates:
        if not isinstance(candidate.get("all_hard_gates_pass"), bool):
            raise TypeError("candidate pass summary must be boolean")
        _validate_gate_rows(_object_list(candidate, "gates"))
    recomputed = evaluate_result_payload(
        protocol,
        stage,
        claim_raw,
        _raw_baseline(_mapping(payload, "baseline")),
        [_raw_model(row) for row in _object_list(payload, "models")],
    )
    if payload != recomputed:
        raise ValueError("result is not the unique evaluation of its raw rows")


def _validate_gate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(GATE_IDS):
        raise ValueError("gate cardinality mismatch")
    for gate_id, row in zip(GATE_IDS, rows, strict=True):
        if set(row) != {"gate_id", "hard", "passed", "details"}:
            raise ValueError("gate shape mismatch")
        if row.get("gate_id") != gate_id or row.get("hard") is not True:
            raise ValueError("gate identity mismatch")
        if not isinstance(row.get("passed"), bool) or not isinstance(
            row.get("details"), dict
        ):
            raise TypeError("gate types mismatch")


def write_stage_result(
    stage: str, payload: Mapping[str, Any], root: Path = ROOT
) -> Path:
    protocol = load_protocol(root)
    claim_path = _output_path(protocol, stage, "runtime_claim")
    if any(
        _output_path(protocol, stage, key).exists()
        for key in ("runtime_error", "archive_error", "archive_result", "transport")
    ):
        raise FileExistsError("result is exclusive with existing error evidence")
    if not claim_path.is_file():
        raise ValueError("registered runtime claim is required")
    claim = claim_path.read_bytes()
    verify_result_payload(protocol, stage, payload, claim)
    target = _output_path(protocol, stage, "runtime_result")
    write_json_exclusive(target, payload)
    return target


def write_stage_error(stage: str, message: str, root: Path = ROOT) -> Path:
    protocol = load_protocol(root)
    claim = _output_path(protocol, stage, "runtime_claim")
    if any(
        _output_path(protocol, stage, key).exists()
        for key in ("runtime_result", "archive_result", "archive_error", "transport")
    ):
        raise FileExistsError("error is exclusive with existing result evidence")
    if not claim.is_file() or not isinstance(message, str) or not message:
        raise ValueError("claim and non-empty error message are required")
    claim_raw = claim.read_bytes()
    _validate_claim(protocol, stage, claim_raw)
    target = _output_path(protocol, stage, "runtime_error")
    error_payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "claim_sha256": sha256_bytes(claim_raw),
        "error": message,
    }
    _validate_error_payload(stage, error_payload, claim_raw)
    write_json_exclusive(target, error_payload)
    return target


def _validate_claim(
    protocol: Mapping[str, Any], stage: str, raw: bytes
) -> dict[str, Any]:
    claim = json.loads(raw.decode("utf-8", errors="strict"))
    hashes = dict(_mapping(_mapping(protocol, "manifest"), "artifact_sha256"))
    if (
        not isinstance(claim, dict)
        or set(claim) != CLAIM_FIELDS
        or claim
        != {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": claim.get("protocol_commit"),
            "stage": stage,
            "protocol_hashes": hashes,
            "one_time_claim": True,
        }
        or claim.get("one_time_claim") is not True
    ):
        raise ValueError("claim does not exactly bind the frozen protocol")
    _full_commit(claim["protocol_commit"])
    return claim


def _validate_error_payload(
    stage: str, payload: Mapping[str, Any], claim_raw: bytes
) -> None:
    if (
        set(payload) != ERROR_FIELDS
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("stage") != stage
        or payload.get("claim_sha256") != sha256_bytes(claim_raw)
        or not isinstance(payload.get("error"), str)
        or not payload["error"]
    ):
        raise ValueError("error evidence does not exactly bind the stage claim")


def _validate_baseline(
    protocol: Mapping[str, Any], stage: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        set(raw) != {"baseline_id", "cases", "state"}
        or raw.get("baseline_id") != "current-ngr"
    ):
        raise ValueError("baseline raw shape mismatch")
    cases = _validate_case_rows(
        protocol,
        stage,
        _object_list(raw, "cases"),
        expected_hits=24,
        require_logits=False,
    )
    state = _validate_state(_mapping(raw, "state"))
    return {
        "baseline_id": "current-ngr",
        "cases": cases,
        "cohorts": _cohorts(protocol, stage, cases),
        "state": state,
    }


def _validate_models(
    protocol: Mapping[str, Any],
    stage: str,
    raws: list[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = _object_list(_mapping(protocol, "models"), "models")
    if len(raws) != 2:
        raise ValueError("exactly two model runs required")
    rows = []
    for raw, spec in zip(raws, frozen, strict=True):
        if (
            set(raw) != {"model_id", "revision", "cases", "state", "metrics"}
            or raw.get("model_id") != spec.get("model_id")
            or raw.get("revision") != spec.get("revision")
        ):
            raise ValueError("model raw identity mismatch")
        cases = _validate_case_rows(
            protocol,
            stage,
            _object_list(raw, "cases"),
            expected_hits=20,
            require_logits=True,
        )
        base_by_id = {row["case_id"]: row for row in _object_list(baseline, "cases")}
        for case in cases:
            expected_hits = base_by_id[case["case_id"]]["ranked_hits"][:20]
            if [
                (hit["source_path"], hit["rank"], hit["ngr_score"])
                for hit in case["ranked_hits"]
            ] != [
                (hit["source_path"], hit["rank"], hit["ngr_score"])
                for hit in expected_hits
            ]:
                raise ValueError("cross-encoder input must be the frozen NGR top-20")
        metrics = _mapping(raw, "metrics")
        expected_metric_fields = {
            "latency_ms",
            "peak_rss_bytes",
            "cache_bytes",
            "pair_count",
            "window_count",
        }
        if set(metrics) != expected_metric_fields:
            raise ValueError("model metrics shape mismatch")
        latency = metrics.get("latency_ms")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise TypeError("latency metric must be finite and non-negative")
        for key in ("peak_rss_bytes", "cache_bytes", "pair_count", "window_count"):
            value = metrics.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"{key} must be a non-negative integer")
        raw_chunk_count = sum(
            len(hit["chunks"]) for case in cases for hit in case["ranked_hits"]
        )
        if (
            metrics["pair_count"] != raw_chunk_count
            or metrics["window_count"] != raw_chunk_count
        ):
            raise ValueError("pair/window metrics must match raw chunk cardinality")
        rows.append(
            {
                "model_id": raw["model_id"],
                "revision": raw["revision"],
                "cases": cases,
                "state": _validate_state(_mapping(raw, "state")),
                "metrics": dict(metrics),
            }
        )
    return rows


def _validate_case_rows(
    protocol: Mapping[str, Any],
    stage: str,
    rows: list[dict[str, Any]],
    expected_hits: int,
    require_logits: bool,
) -> list[dict[str, Any]]:
    frozen = _object_list(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    if [row.get("case_id") for row in rows] != [row["case_id"] for row in frozen]:
        raise ValueError("case order mismatch")
    corpus_hashes = {
        _string(row, "path"): _string(row, "content_sha256")
        for row in _object_list(_mapping(protocol, "corpus"), "documents")
    }
    corpus_paths = set(corpus_hashes)
    frozen_relations = {
        (
            _string(relation, "source_path"),
            _string(relation, "target_path"),
            _string(relation, "edge_type"),
        )
        for relation in _object_list(_mapping(protocol, "corpus"), "relationships")
    }
    result = copy.deepcopy(rows)
    for row, query in zip(result, frozen, strict=True):
        required = {"case_id", "cohort", "ranked_hits"}
        if set(row) != required or row.get("cohort") != query.get("cohort"):
            raise ValueError("case shape mismatch")
        hits = _object_list(row, "ranked_hits")
        if len(hits) != expected_hits:
            raise ValueError("ranked hit cardinality mismatch")
        seen = set()
        previous_order_key: tuple[float, str] | None = None
        for rank, hit in enumerate(hits, 1):
            fields = {
                "source_path",
                "rank",
                "ngr_score",
                "source_sha256",
                "relation_paths",
            }
            if require_logits:
                fields |= {"chunks", "raw_logit", "winning_chunk_index"}
            if (
                set(hit) != fields
                or not isinstance(hit.get("rank"), int)
                or isinstance(hit.get("rank"), bool)
                or hit.get("rank") != rank
            ):
                raise ValueError("ranked hit shape mismatch")
            path = _string(hit, "source_path")
            if path in seen or hit.get("source_sha256") != corpus_hashes.get(path):
                raise ValueError("ranked source identity mismatch")
            seen.add(path)
            if not isinstance(hit.get("ngr_score"), (int, float)) or isinstance(
                hit.get("ngr_score"), bool
            ):
                raise TypeError("NGR score must be numeric")
            ngr_score = float(hit["ngr_score"])
            if not math.isfinite(ngr_score):
                raise ValueError("NGR score must be finite")
            order_key = (-ngr_score, path)
            if previous_order_key is not None and order_key < previous_order_key:
                raise ValueError("NGR hits must be score-desc/source-identity-asc")
            previous_order_key = order_key
            relations = _object_list(hit, "relation_paths")
            relation_keys: set[tuple[str, str, str]] = set()
            for relation in relations:
                if set(relation) != {
                    "seed_path",
                    "target_path",
                    "edge_type",
                    "step_count",
                }:
                    raise ValueError("relation path shape mismatch")
                seed = _string(relation, "seed_path")
                target = _string(relation, "target_path")
                edge_type = _string(relation, "edge_type")
                relation_key = (seed, target, edge_type)
                if (
                    seed not in corpus_paths
                    or target != path
                    or target not in corpus_paths
                    or not isinstance(relation.get("step_count"), int)
                    or isinstance(relation.get("step_count"), bool)
                    or relation.get("step_count") != 1
                    or relation_key not in frozen_relations
                    or relation_key in relation_keys
                ):
                    raise ValueError("relation path must be one frozen in-corpus edge")
                relation_keys.add(relation_key)
            if require_logits:
                chunks = _object_list(hit, "chunks")
                if not chunks or any(
                    set(chunk)
                    != {
                        "chunk_index",
                        "start_codepoint",
                        "end_codepoint",
                        "text_sha256",
                        "raw_logit",
                    }
                    for chunk in chunks
                ):
                    raise ValueError("chunk score shape mismatch")
                expected_chunks = _source_passages(protocol, path)
                if len(chunks) != len(expected_chunks):
                    raise ValueError("chunk cardinality does not match frozen source")
                logits = []
                for index, (chunk, expected_chunk) in enumerate(
                    zip(chunks, expected_chunks, strict=True)
                ):
                    if (
                        not isinstance(chunk.get("chunk_index"), int)
                        or isinstance(chunk.get("chunk_index"), bool)
                        or chunk.get("chunk_index") != index
                        or not isinstance(chunk.get("start_codepoint"), int)
                        or isinstance(chunk.get("start_codepoint"), bool)
                        or chunk.get("start_codepoint")
                        != expected_chunk["start_codepoint"]
                        or not isinstance(chunk.get("end_codepoint"), int)
                        or isinstance(chunk.get("end_codepoint"), bool)
                        or chunk.get("end_codepoint") != expected_chunk["end_codepoint"]
                        or chunk.get("text_sha256") != expected_chunk["text_sha256"]
                        or not isinstance(chunk.get("raw_logit"), (int, float))
                        or isinstance(chunk.get("raw_logit"), bool)
                        or not math.isfinite(float(chunk["raw_logit"]))
                    ):
                        raise ValueError("chunk score mismatch")
                    logits.append(float(chunk["raw_logit"]))
                winner = min(
                    (
                        index
                        for index, value in enumerate(logits)
                        if value == max(logits)
                    ),
                    default=-1,
                )
                if (
                    not isinstance(hit.get("winning_chunk_index"), int)
                    or isinstance(hit.get("winning_chunk_index"), bool)
                    or hit.get("winning_chunk_index") != winner
                    or not isinstance(hit.get("raw_logit"), (int, float))
                    or isinstance(hit.get("raw_logit"), bool)
                    or not math.isfinite(float(hit["raw_logit"]))
                    or float(hit.get("raw_logit")) != logits[winner]
                ):
                    raise ValueError("document max-chunk score mismatch")
    return result


def _validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if set(state) != STATE_FIELDS:
        raise ValueError("state shape mismatch")
    if any(
        state.get(key) is not True for key in ("cpu_only", "offline", "fresh_process")
    ):
        raise ValueError("CPU/offline/fresh-process state required")
    for key in (
        "ranking_sha256",
        "replay_ranking_sha256",
        "activation_sha256",
        "replay_activation_sha256",
        "edge_sha256_before",
        "edge_sha256_after",
        "sqlite_sha256_before",
        "sqlite_sha256_after",
    ):
        if not isinstance(state.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", state[key]
        ):
            raise TypeError(f"{key} must be a lowercase SHA-256")
    database_ids = (state.get("fresh_database_id"), state.get("replay_database_id"))
    if any(not isinstance(value, str) or not value for value in database_ids):
        raise TypeError("database ids must be non-empty strings")
    if database_ids[0] == database_ids[1]:
        raise ValueError("fresh and replay database identities must differ")
    for key in ("feedback_count_before", "feedback_count_after"):
        value = state.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TypeError(f"{key} must be a non-negative integer")
    return dict(state)


def _source_passages(
    protocol: Mapping[str, Any], source_path: str
) -> tuple[dict[str, Any], ...]:
    corpus = _mapping(protocol, "corpus")
    root = _path(protocol, "source_root")
    return _frozen_source_passages(str(root), _string(corpus, "commit"), source_path)


@cache
def _frozen_source_passages(
    root_value: str, commit: str, source_path: str
) -> tuple[dict[str, Any], ...]:
    raw = _git_bytes(Path(root_value), commit, source_path)
    text = raw.decode("utf-8", errors="strict")
    return tuple(
        {
            "chunk_index": chunk["chunk_index"],
            "start_codepoint": chunk["start_codepoint"],
            "end_codepoint": chunk["end_codepoint"],
            "text_sha256": sha256_bytes(chunk["text"].encode("utf-8")),
        }
        for chunk in project_passages(text)
    )


def _derive_candidate(
    protocol: Mapping[str, Any],
    stage: str,
    baseline: Mapping[str, Any],
    models: list[dict[str, Any]],
    candidate_id: str,
    state_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model = models[0] if candidate_id.startswith("bge-base") else models[1]
    fusion = "rrf" if "-rrf-" in candidate_id else "ce"
    cases = []
    for model_case in _object_list(model, "cases"):
        ce_hits = [
            hit for hit in model_case["ranked_hits"] if float(hit["raw_logit"]) >= 0.0
        ]
        ce_sorted = sorted(
            ce_hits, key=lambda hit: (-float(hit["raw_logit"]), hit["source_path"])
        )
        ce_rank = {
            hit["source_path"]: rank
            for rank, hit in enumerate(
                sorted(
                    model_case["ranked_hits"],
                    key=lambda hit: (-float(hit["raw_logit"]), hit["source_path"]),
                ),
                1,
            )
        }
        if fusion == "rrf":
            ranked = sorted(
                ce_hits,
                key=lambda hit: (
                    -(1 / (60 + hit["rank"]) + 1 / (60 + ce_rank[hit["source_path"]])),
                    hit["source_path"],
                ),
            )
        else:
            ranked = ce_sorted
        returned = [hit["source_path"] for hit in ranked[:5]]
        cases.append(
            {
                "case_id": model_case["case_id"],
                "cohort": model_case["cohort"],
                "returned_source_paths": returned,
                "scores": [
                    {
                        "source_path": hit["source_path"],
                        "raw_logit": hit["raw_logit"],
                        "sigmoid": sigmoid(float(hit["raw_logit"])),
                        "ce_rank": ce_rank[hit["source_path"]],
                        "ngr_rank": hit["rank"],
                        "final_score": (
                            1 / (60 + hit["rank"])
                            + 1 / (60 + ce_rank[hit["source_path"]])
                        )
                        if fusion == "rrf"
                        else hit["raw_logit"],
                    }
                    for hit in ranked
                ],
            }
        )
    cohorts = _cohorts(protocol, stage, cases)
    gates = _candidate_gates(protocol, stage, baseline, cases, cohorts, state_rows)
    return {
        "candidate_id": candidate_id,
        "cases": cases,
        "cohorts": cohorts,
        "gates": gates,
        "all_hard_gates_pass": all(row["passed"] for row in gates),
    }


def _candidate_gates(
    protocol: Mapping[str, Any],
    stage: str,
    baseline: Mapping[str, Any],
    cases: list[dict[str, Any]],
    cohorts: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold = {
        _string(row, "case_id"): row
        for row in _object_list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }
    base_cases = {row["case_id"]: row for row in _object_list(baseline, "cases")}
    candidate_cases = {row["case_id"]: row for row in cases}
    positive_case_ok = True
    complete = True
    relation_ok = True
    base_forbidden = candidate_forbidden = 0
    negative_nonworse = True
    for case_id, row in gold.items():
        base_paths = [
            hit["source_path"] for hit in base_cases[case_id]["ranked_hits"][:5]
        ]
        candidate_paths = candidate_cases[case_id]["returned_source_paths"]
        if row["expected_paths"]:
            target = row["expected_paths"][0]
            base_rank = base_paths.index(target) + 1 if target in base_paths else None
            candidate_rank = (
                candidate_paths.index(target) + 1 if target in candidate_paths else None
            )
            if base_rank is not None and (
                candidate_rank is None or candidate_rank > base_rank
            ):
                positive_case_ok = False
            if candidate_rank is None:
                complete = False
            if row["cohort"] == "relation_linked":
                hit = next(
                    (
                        hit
                        for hit in base_cases[case_id]["ranked_hits"]
                        if hit["source_path"] == target
                    ),
                    None,
                )
                relation_ok &= bool(
                    hit
                    and any(
                        path
                        == {
                            "seed_path": row["relation_seed_path"],
                            "target_path": target,
                            "edge_type": row["relation_edge_type"],
                            "step_count": 1,
                        }
                        for path in hit["relation_paths"]
                    )
                )
        else:
            forbidden = row["forbidden_paths"][0]
            base_present = forbidden in base_paths
            candidate_present = forbidden in candidate_paths
            base_forbidden += int(base_present)
            candidate_forbidden += int(candidate_present)
            negative_nonworse &= not (candidate_present and not base_present)
    base_cohorts = {
        row["cohort"]: row
        for row in _cohorts(protocol, stage, _object_list(baseline, "cases"))
    }
    cohort_ok = all(
        row["mrr"] >= base_cohorts[row["cohort"]]["mrr"]
        and row["hit_at_5"] >= base_cohorts[row["cohort"]]["hit_at_5"]
        for row in cohorts
        if row["cohort"] != "negative_control"
    )
    database_ids = [
        state[key]
        for state in state_rows
        for key in ("fresh_database_id", "replay_database_id")
    ]
    deterministic = all(
        state["ranking_sha256"] == state["replay_ranking_sha256"]
        and state["activation_sha256"] == state["replay_activation_sha256"]
        for state in state_rows
    ) and len(set(database_ids)) == len(database_ids)
    immutable = all(
        state["edge_sha256_before"] == state["edge_sha256_after"]
        and state["feedback_count_before"] == state["feedback_count_after"] == 0
        and state["sqlite_sha256_before"] == state["sqlite_sha256_after"]
        for state in state_rows
    )
    passed = (
        True,
        True,
        deterministic,
        positive_case_ok,
        cohort_ok,
        negative_nonworse and candidate_forbidden < base_forbidden,
        complete,
        relation_ok,
        True,
        immutable,
        True,
    )
    return [
        {
            "gate_id": gate_id,
            "hard": True,
            "passed": bool(value),
            "details": {"recomputed": True},
        }
        for gate_id, value in zip(GATE_IDS, passed, strict=True)
    ]


def _cohorts(
    protocol: Mapping[str, Any], stage: str, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    gold = {
        _string(row, "case_id"): row
        for row in _object_list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }
    rows = []
    for cohort in COHORTS:
        selected = [row for row in cases if row["cohort"] == cohort]
        reciprocal = []
        hits = []
        for case in selected:
            target = gold[case["case_id"]]["expected_paths"]
            paths = (
                case["returned_source_paths"]
                if "returned_source_paths" in case
                else [hit["source_path"] for hit in case["ranked_hits"][:5]]
            )
            rank = paths.index(target[0]) + 1 if target and target[0] in paths else None
            reciprocal.append(0.0 if rank is None else 1.0 / rank)
            hits.append(1.0 if rank is not None else 0.0)
        rows.append(
            {
                "cohort": cohort,
                "case_ids": [row["case_id"] for row in selected],
                "mrr": sum(reciprocal) / len(selected),
                "hit_at_5": sum(hits) / len(selected),
            }
        )
    return rows


def _raw_baseline(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_id": value.get("baseline_id"),
        "cases": copy.deepcopy(value.get("cases")),
        "state": copy.deepcopy(value.get("state")),
    }


def _raw_model(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value.get(key))
        for key in ("model_id", "revision", "cases", "state", "metrics")
    }


def archive_stage(stage: str, root: Path = ROOT) -> Path:
    return _archive_stage(load_protocol(root), stage)


def _archive_stage(protocol: Mapping[str, Any], stage: str) -> Path:
    claim_runtime = _output_path(protocol, stage, "runtime_claim")
    result_runtime = _output_path(protocol, stage, "runtime_result")
    error_runtime = _output_path(protocol, stage, "runtime_error")
    claim_archive = _output_path(protocol, stage, "archive_claim")
    result_archive = _output_path(protocol, stage, "archive_result")
    error_archive = _output_path(protocol, stage, "archive_error")
    transport = _output_path(protocol, stage, "transport")
    if any(
        path.exists()
        for path in (claim_archive, result_archive, error_archive, transport)
    ):
        raise FileExistsError("archive stage already exists")
    if not claim_runtime.is_file() or result_runtime.exists() == error_runtime.exists():
        raise ValueError("claim and exactly one result/error are required")
    claim_raw = claim_runtime.read_bytes()
    _validate_claim(protocol, stage, claim_raw)
    payload_runtime = result_runtime if result_runtime.exists() else error_runtime
    payload_archive = result_archive if result_runtime.exists() else error_archive
    payload_raw = payload_runtime.read_bytes()
    if result_runtime.exists():
        verify_result_payload(
            protocol,
            stage,
            json.loads(payload_raw.decode("utf-8", errors="strict")),
            claim_raw,
        )
    else:
        _validate_error_payload(
            stage,
            json.loads(payload_raw.decode("utf-8", errors="strict")),
            claim_raw,
        )
    payload_archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(claim_runtime, claim_archive)
    os.replace(payload_runtime, payload_archive)
    files = []
    for runtime, archive, raw in (
        (claim_runtime, claim_archive, claim_raw),
        (payload_runtime, payload_archive, payload_raw),
    ):
        files.append(
            {
                "runtime_path": _relative(protocol, runtime),
                "archive_path": _relative(protocol, archive),
                "sha256": sha256_bytes(raw),
                "byte_identity": archive.read_bytes() == raw,
            }
        )
    transport_payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "reason": "phase-boundary byte-preserving archival",
        "stage_execution_count": 1,
        "files": files,
    }
    _validate_transport_payload(
        protocol, stage, transport_payload, claim_raw, payload_archive, payload_raw
    )
    write_json_exclusive(transport, transport_payload)
    return transport


def _validate_transport_payload(
    protocol: Mapping[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    claim_raw: bytes,
    evidence_path: Path,
    evidence_raw: bytes,
) -> None:
    if (
        set(payload) != TRANSPORT_FIELDS
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("stage") != stage
        or payload.get("reason") != "phase-boundary byte-preserving archival"
        or payload.get("stage_execution_count") != 1
        or isinstance(payload.get("stage_execution_count"), bool)
    ):
        raise ValueError("transport manifest shape mismatch")
    files = _object_list(payload, "files")
    if (
        len(files) != 2
        or any(set(row) != TRANSPORT_FILE_FIELDS for row in files)
        or any(row.get("byte_identity") is not True for row in files)
    ):
        raise ValueError("transport file registry shape mismatch")
    claim_archive = _output_path(protocol, stage, "archive_claim")
    claim_runtime = _output_path(protocol, stage, "runtime_claim")
    evidence_kind = (
        "result" if evidence_path.name.endswith("observed.json") else "error"
    )
    evidence_runtime = _output_path(protocol, stage, f"runtime_{evidence_kind}")
    expected = (
        {
            "runtime_path": _relative(protocol, claim_runtime),
            "archive_path": _relative(protocol, claim_archive),
            "sha256": sha256_bytes(claim_raw),
            "byte_identity": True,
        },
        {
            "runtime_path": _relative(protocol, evidence_runtime),
            "archive_path": _relative(protocol, evidence_path),
            "sha256": sha256_bytes(evidence_raw),
            "byte_identity": True,
        },
    )
    if tuple(files) != expected:
        raise ValueError("transport manifest does not exactly bind archived bytes")
    if (
        claim_archive.read_bytes() != claim_raw
        or evidence_path.read_bytes() != evidence_raw
    ):
        raise ValueError("transport byte identity check failed")


def verify_phase_state(protocol: Mapping[str, Any]) -> dict[str, str]:
    states = {}
    for stage in STAGES:
        runtime = [
            _output_path(protocol, stage, key)
            for key in ("runtime_claim", "runtime_result", "runtime_error")
        ]
        if any(path.exists() for path in runtime):
            raise ValueError("runtime artifacts must be archived before commit")
        claim = _output_path(protocol, stage, "archive_claim")
        result = _output_path(protocol, stage, "archive_result")
        error = _output_path(protocol, stage, "archive_error")
        transport = _output_path(protocol, stage, "transport")
        if not any(path.exists() for path in (claim, result, error, transport)):
            states[stage] = "unobserved"
            continue
        if (
            not claim.is_file()
            or not transport.is_file()
            or result.exists() == error.exists()
        ):
            raise ValueError("archived stage is incomplete")
        claim_raw = claim.read_bytes()
        _validate_claim(protocol, stage, claim_raw)
        payload = result if result.exists() else error
        if result.exists():
            value = read_json(result)
            verify_result_payload(protocol, stage, value, claim_raw)
            states[stage] = (
                "archived-passed" if value["all_hard_gates_pass"] else "archived-failed"
            )
        else:
            value = read_json(error)
            _validate_error_payload(stage, value, claim_raw)
            states[stage] = "archived-error"
        transport_value = read_json(transport)
        _validate_transport_payload(
            protocol,
            stage,
            transport_value,
            claim_raw,
            payload,
            payload.read_bytes(),
        )
    if states["holdout"] != "unobserved" and states["development"] != "archived-passed":
        raise ValueError("holdout requires passing development")
    return states


def _assert_stage_can_start(protocol: Mapping[str, Any], stage: str) -> None:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    if any(
        _output_path(protocol, stage, key).exists()
        for key in (
            "runtime_claim",
            "runtime_result",
            "runtime_error",
            "archive_claim",
            "archive_result",
            "archive_error",
            "transport",
        )
    ):
        raise FileExistsError(f"{stage} has already started")
    if stage == "holdout":
        phases = verify_phase_state(protocol)
        if phases != {
            "development": "archived-passed",
            "holdout": "unobserved",
        }:
            raise ValueError("holdout is closed unless development passed")


def build_synthetic_evaluated_result(
    protocol: Mapping[str, Any], stage: str, claim_raw: bytes
) -> dict[str, Any]:
    documents = _object_list(_mapping(protocol, "corpus"), "documents")
    paths = [row["path"] for row in documents]
    hashes = {row["path"]: row["content_sha256"] for row in documents}
    queries = _object_list(_mapping(_mapping(protocol, "queries"), "stages"), stage)
    gold = {
        row["case_id"]: row
        for row in _object_list(_mapping(_mapping(protocol, "gold"), "stages"), stage)
    }
    baseline_cases = []
    for query in queries:
        row = gold[query["case_id"]]
        target = (row["expected_paths"] or row["forbidden_paths"])[0]
        if query["cohort"] == "negative_control":
            ordered = (
                [path for path in paths if path != target][:4]
                + [target]
                + [path for path in paths if path != target][4:]
            )
        else:
            ordered = [target] + [path for path in paths if path != target]
        hits = []
        for rank, path in enumerate(ordered, 1):
            relation = []
            if query["cohort"] == "relation_linked" and path == target:
                relation = [
                    {
                        "seed_path": row["relation_seed_path"],
                        "target_path": target,
                        "edge_type": "informs",
                        "step_count": 1,
                    }
                ]
            hits.append(
                {
                    "source_path": path,
                    "rank": rank,
                    "ngr_score": 1.0 / rank,
                    "source_sha256": hashes[path],
                    "relation_paths": relation,
                }
            )
        baseline_cases.append(
            {
                "case_id": query["case_id"],
                "cohort": query["cohort"],
                "ranked_hits": hits,
            }
        )
    common = sha256_bytes(b"synthetic-state")

    def state(name: str) -> dict[str, Any]:
        ranking = sha256_bytes(name.encode("utf-8"))
        return {
            "fresh_database_id": f"{stage}:{name}:fresh",
            "replay_database_id": f"{stage}:{name}:replay",
            "ranking_sha256": ranking,
            "replay_ranking_sha256": ranking,
            "activation_sha256": common,
            "replay_activation_sha256": common,
            "edge_sha256_before": common,
            "edge_sha256_after": common,
            "feedback_count_before": 0,
            "feedback_count_after": 0,
            "sqlite_sha256_before": common,
            "sqlite_sha256_after": common,
            "cpu_only": True,
            "offline": True,
            "fresh_process": True,
        }

    baseline = {
        "baseline_id": "current-ngr",
        "cases": baseline_cases,
        "state": state("baseline"),
    }
    model_raws = []
    for model_index, spec in enumerate(
        _object_list(_mapping(protocol, "models"), "models")
    ):
        model_cases = []
        raw_chunk_count = 0
        for case in baseline_cases:
            target = (
                gold[case["case_id"]]["expected_paths"]
                or gold[case["case_id"]]["forbidden_paths"]
            )[0]
            ranked_hits = []
            for hit in case["ranked_hits"][:20]:
                positive = (
                    case["cohort"] != "negative_control"
                    and hit["source_path"] == target
                )
                forbidden = (
                    case["cohort"] == "negative_control"
                    and hit["source_path"] == target
                )
                logit = (
                    4.0
                    if positive
                    else (-4.0 if forbidden else 1.0 - hit["rank"] / 25.0)
                )
                source_chunks = _source_passages(protocol, hit["source_path"])
                raw_chunk_count += len(source_chunks)
                ranked_hits.append(
                    {
                        **copy.deepcopy(hit),
                        "chunks": [
                            {
                                **chunk,
                                "raw_logit": logit,
                            }
                            for chunk in source_chunks
                        ],
                        "raw_logit": logit,
                        "winning_chunk_index": 0,
                    }
                )
            model_cases.append(
                {
                    "case_id": case["case_id"],
                    "cohort": case["cohort"],
                    "ranked_hits": ranked_hits,
                }
            )
        model_raws.append(
            {
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "cases": model_cases,
                "state": state(f"model-{model_index}"),
                "metrics": {
                    "latency_ms": 0.0,
                    "peak_rss_bytes": 0,
                    "cache_bytes": 0,
                    "pair_count": raw_chunk_count,
                    "window_count": raw_chunk_count,
                },
            }
        )
    return evaluate_result_payload(protocol, stage, claim_raw, baseline, model_raws)


def prove_archive_round_trip() -> dict[str, dict[str, str]]:
    source = load_protocol()
    results: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as directory:
        unobserved = dict(source)
        unobserved["root"] = Path(directory)
        results["unobserved"] = verify_phase_state(unobserved)
    for scenario in (
        "development-passed",
        "holdout-passed",
        "development-failed",
        "development-error",
    ):
        with tempfile.TemporaryDirectory() as directory:
            protocol = dict(source)
            protocol["root"] = Path(directory)
            hashes = dict(_mapping(_mapping(source, "manifest"), "artifact_sha256"))

            def claim(
                stage: str,
                bound_protocol: Mapping[str, Any] = protocol,
                bound_hashes: dict[str, Any] = hashes,
            ) -> bytes:
                value = {
                    "protocol_id": PROTOCOL_ID,
                    "protocol_commit": "0" * 40,
                    "stage": stage,
                    "protocol_hashes": bound_hashes,
                    "one_time_claim": True,
                }
                path = _output_path(bound_protocol, stage, "runtime_claim")
                write_json_exclusive(path, value)
                return path.read_bytes()

            raw = claim("development")
            if scenario == "development-error":
                write_json_exclusive(
                    _output_path(protocol, "development", "runtime_error"),
                    {
                        "protocol_id": PROTOCOL_ID,
                        "stage": "development",
                        "claim_sha256": sha256_bytes(raw),
                        "error": "synthetic",
                    },
                )
            else:
                result = build_synthetic_evaluated_result(source, "development", raw)
                if scenario == "development-failed":
                    for model in result["models"]:
                        model["state"]["replay_ranking_sha256"] = "f" * 64
                    result = evaluate_result_payload(
                        source,
                        "development",
                        raw,
                        _raw_baseline(result["baseline"]),
                        [_raw_model(row) for row in result["models"]],
                    )
                write_json_exclusive(
                    _output_path(protocol, "development", "runtime_result"), result
                )
            _archive_stage(protocol, "development")
            if scenario == "holdout-passed":
                holdout_raw = claim("holdout")
                write_json_exclusive(
                    _output_path(protocol, "holdout", "runtime_result"),
                    build_synthetic_evaluated_result(source, "holdout", holdout_raw),
                )
                _archive_stage(protocol, "holdout")
            results[scenario] = verify_phase_state(protocol)
    return results


def _verify_hash_registry(root: Path, registry: Mapping[str, Any]) -> None:
    if not registry:
        raise ValueError("hash registry cannot be empty")
    for relative, expected in registry.items():
        if not isinstance(relative, str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected)
        ):
            raise ValueError("hash registry shape mismatch")
        path = root / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")


def _output_path(protocol: Mapping[str, Any], stage: str, key: str) -> Path:
    return _path(protocol, "root") / _string(
        _mapping(_mapping(_mapping(protocol, "manifest"), "outputs"), stage), key
    )


def _relative(protocol: Mapping[str, Any], path: Path) -> str:
    return str(path.relative_to(_path(protocol, "root"))).replace("\\", "/")


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"missing committed path: {commit}:{path}")
    return completed.stdout


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout.decode("ascii", errors="strict")


def _full_commit(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit must be lowercase full 40-hex")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    row = value.get(key)
    if not isinstance(row, dict):
        raise TypeError(f"{key} must be an object")
    return row


def _object_list(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} must be an array of objects")
    return rows


def _string(value: Mapping[str, Any], key: str) -> str:
    row = value.get(key)
    if not isinstance(row, str) or not row:
        raise ValueError(f"{key} must be a non-empty string")
    return row


def _path(value: Mapping[str, Any], key: str) -> Path:
    row = value.get(key)
    if not isinstance(row, Path):
        raise TypeError(f"{key} must be a path")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the result-free cross-encoder precision protocol"
    )
    parser.add_argument("command", choices=("audit", "probe"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    output: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "freeze_registered_query_execution_count": 0,
        "freeze_model_inference_count": 0,
        "freeze_observed_result_count": 0,
        "historical_v1_observation_included": False,
        "phase": verify_phase_state(protocol),
    }
    if args.command == "probe":
        output["archive_round_trip"] = prove_archive_round_trip()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

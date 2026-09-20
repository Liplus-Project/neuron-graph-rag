from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import full_corpus_rerank_oracle as v1

PROTOCOL_ID = "github-retrieval-parity-v5-full-corpus-rerank-oracle-v2"
ROOT = Path(__file__).resolve().parents[2]
MODULE = "neuron_graph_rag.full_corpus_rerank_oracle_v2"
MANIFEST = Path("tests/fixtures/full_corpus_rerank_oracle_v2.manifest.json")
SCHEMA = Path("tests/fixtures/full_corpus_rerank_oracle_v2.schema.json")
QUERY = Path("tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
PACKAGE_INIT = Path("tests/fixtures/full_corpus_rerank_oracle_v2.package_init.py")
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")
CLAIM = Path("tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json")
RESULT = Path("tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json")
ERROR = Path("tests/evidence/full_corpus_rerank_oracle_v2/development.error.json")
ATTESTATION = Path("preflight.attestation.json")
CASE_ID = "v5path-dev-semantic-axis-separation"
EXPECTED_QUERY = "異なる判断軸の矛盾を優先順位で潰さず境界へ戻して解く原則"
EXPECTED_SOURCE_ID = (
    "github:Liplus-Project/neuron-graph-rag:"
    "corpora/github-retrieval-parity-v4/rules/model/axis-separation.md"
)
RERANK_CUTOFF = 20
BATCH_SIZE = 8
CPU_THREADS = 4
WINDOW_CODEPOINTS = 480
OVERLAP_CODEPOINTS = 80
MODEL_KINDS = ("base", "v2-m3")
MODEL_DISTRIBUTIONS = v1.MODEL_DISTRIBUTIONS

WORKER_REGISTERED_FILES = {
    Path("src/neuron_graph_rag/__init__.py"),
    Path("src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py"),
    Path("src/neuron_graph_rag/full_corpus_rerank_oracle.py"),
    Path("src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py"),
    MANIFEST,
    QUERY,
    SCHEMA,
    CORPUS,
    MODEL_REGISTRY,
}
FINALIZER_REGISTERED_FILES = WORKER_REGISTERED_FILES | {GOLD}
FORBIDDEN_REGISTERED_PATHS = {
    Path("tests/fixtures/github_retrieval_parity_v5.queries.json"),
    Path("tests/fixtures/github_retrieval_parity_v5.gold.json"),
    Path("tests/evidence/full_corpus_rerank_oracle_v1"),
    Path("tests/fixtures/github_retrieval_parity_v5.holdout.json"),
    Path("ngr.sqlite3"),
}

canonical_json_bytes = v1.canonical_json_bytes
sha256_bytes = v1.sha256_bytes
sha256_file = v1.sha256_file
read_json = v1.read_json
write_json_exclusive = v1.write_json_exclusive
_exact_keys = v1._exact_keys
_network_is_disabled = v1._network_is_disabled
_verify_model_files = v1._verify_model_files
_load_model = v1._load_model
_score_documents = v1._score_documents
_dependency_versions = v1._dependency_versions
_validate_document_rows = v1._validate_document_rows
classify_ranks = v1.classify_ranks
project_passages = v1.project_passages


def _query(root: Path = ROOT) -> str:
    payload = read_json(root / QUERY)
    _exact_keys(
        payload, {"case_id", "protocol_id", "query", "schema_version"}, "query bundle"
    )
    if (
        payload["schema_version"] != 1
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["case_id"] != CASE_ID
        or payload["query"] != EXPECTED_QUERY
    ):
        raise ValueError("development-only query bundle identity mismatch")
    return str(payload["query"])


def _gold(root: Path = ROOT) -> str:
    payload = read_json(root / GOLD)
    _exact_keys(
        payload,
        {"case_id", "expected_source_id", "protocol_id", "schema_version"},
        "gold bundle",
    )
    if (
        payload["schema_version"] != 1
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["case_id"] != CASE_ID
        or payload["expected_source_id"] != EXPECTED_SOURCE_ID
    ):
        raise ValueError("development-only gold bundle identity mismatch")
    return str(payload["expected_source_id"])


def _manifest(root: Path = ROOT, *, verify_files: bool = True) -> dict[str, Any]:
    manifest = read_json(root / MANIFEST)
    _exact_keys(
        manifest,
        {
            "aggregation",
            "artifact_sha256",
            "case_id",
            "classification_policy",
            "copy_map",
            "corpus_document_count",
            "forbidden_registered_paths",
            "models",
            "projection",
            "protocol_id",
            "purpose",
            "rerank_cutoff",
            "runtime",
            "schema_version",
            "source_files",
            "worker_registered_files",
        },
        "manifest",
    )
    expected_runtime = {
        "batch_size": BATCH_SIZE,
        "cpu_threads": CPU_THREADS,
        "device": "cpu",
        "fresh_runtime": True,
        "network": "disabled",
        "platform": "linux-x86_64",
        "worker_gold_mount": "absent",
    }
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["case_id"] != CASE_ID
        or manifest["corpus_document_count"] != 93
        or manifest["rerank_cutoff"] != RERANK_CUTOFF
        or manifest["aggregation"] != "maximum-raw-logit-per-document"
        or manifest["projection"]
        != {
            "coverage": "start-to-end",
            "overlap": OVERLAP_CODEPOINTS,
            "tokenizer_max_length": 512,
            "unit": "unicode-codepoint",
            "window": WINDOW_CODEPOINTS,
        }
        or manifest["runtime"] != expected_runtime
    ):
        raise ValueError("manifest identity or execution contract mismatch")
    policy = {
        "all_models_at_or_above_cutoff": "candidate_generation_bottleneck",
        "all_models_below_cutoff": "semantic_discrimination_bottleneck",
        "mixed": "undetermined",
    }
    if manifest["classification_policy"] != policy:
        raise ValueError("classification policy mismatch")
    expected_sources = {
        "corpus": CORPUS.as_posix(),
        "gold": GOLD.as_posix(),
        "model_registry": MODEL_REGISTRY.as_posix(),
        "package_init": PACKAGE_INIT.as_posix(),
        "query": QUERY.as_posix(),
        "schema": SCHEMA.as_posix(),
        "v1_helpers": "src/neuron_graph_rag/full_corpus_rerank_oracle.py",
        "v2_runner": "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py",
        "projection": "src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py",
    }
    if manifest["source_files"] != expected_sources:
        raise ValueError("manifest source file contract mismatch")
    expected_copy_map = {
        **{path.as_posix(): path.as_posix() for path in WORKER_REGISTERED_FILES},
        GOLD.as_posix(): GOLD.as_posix(),
    }
    expected_copy_map["src/neuron_graph_rag/__init__.py"] = PACKAGE_INIT.as_posix()
    if manifest["copy_map"] != expected_copy_map:
        raise ValueError("manifest copy map mismatch")
    if manifest["worker_registered_files"] != sorted(
        path.as_posix() for path in WORKER_REGISTERED_FILES
    ):
        raise ValueError("worker registered allowlist mismatch")
    if manifest["forbidden_registered_paths"] != sorted(
        path.as_posix() for path in FORBIDDEN_REGISTERED_PATHS
    ):
        raise ValueError("forbidden registered path inventory mismatch")
    artifacts = manifest["artifact_sha256"]
    expected_artifacts = set(expected_sources.values())
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("manifest artifact hash registry mismatch")
    for relative, expected in artifacts.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid registered SHA-256: {relative}")
        if verify_files and sha256_file(root / relative) != expected:
            raise ValueError(f"registered artifact changed: {relative}")
    models = manifest["models"]
    if not isinstance(models, list) or [row.get("kind") for row in models] != list(
        MODEL_KINDS
    ):
        raise ValueError("manifest model order mismatch")
    return manifest


def _documents(root: Path = ROOT) -> list[dict[str, Any]]:
    return v1._documents(root)


def _model_specs(
    root: Path = ROOT, *, registered: bool = False
) -> list[dict[str, Any]]:
    manifest = _manifest(root, verify_files=not registered)
    registry = read_json(root / MODEL_REGISTRY).get("models")
    if not isinstance(registry, list):
        raise TypeError("model registry must contain a model list")
    results = []
    for expected in manifest["models"]:
        selected = [
            row
            for row in registry
            if isinstance(row, dict)
            and row.get("model_id") == expected.get("model_id")
            and row.get("revision") == expected.get("revision")
        ]
        if len(selected) != 1:
            raise ValueError(f"pinned model is missing: {expected.get('kind')}")
        results.append({**selected[0], "kind": expected["kind"]})
    return results


def _registered_regular_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _assert_forbidden_absent(root: Path) -> None:
    for relative in FORBIDDEN_REGISTERED_PATHS:
        if (root / relative).exists():
            raise ValueError(f"forbidden registered input exists: {relative.as_posix()}")
    if any(root.rglob("*holdout*")):
        raise ValueError("holdout-named input exists in registered environment")


def _validate_registered_tree(
    root: Path, *, stage: str, terminal: str | None = None
) -> dict[str, str]:
    manifest = _manifest(root, verify_files=False)
    expected = set(WORKER_REGISTERED_FILES)
    if stage == "finalizer":
        expected.add(GOLD)
    elif stage not in {"preflight", "claim", "worker"}:
        raise ValueError(f"unknown registered stage: {stage}")
    if stage in {"worker", "finalizer"}:
        expected.add(CLAIM)
    if terminal == "result":
        expected.add(RESULT)
    elif terminal == "error":
        expected.add(ERROR)
    elif terminal is not None:
        raise ValueError(f"unknown terminal state: {terminal}")
    actual = _registered_regular_files(root)
    if actual != expected:
        raise ValueError(
            "registered file allowlist mismatch: "
            f"missing={sorted(path.as_posix() for path in expected - actual)!r} "
            f"extra={sorted(path.as_posix() for path in actual - expected)!r}"
        )
    _assert_forbidden_absent(root)
    hashes = {path.as_posix(): sha256_file(root / path) for path in expected}
    copy_map = manifest["copy_map"]
    artifacts = manifest["artifact_sha256"]
    for destination in WORKER_REGISTERED_FILES:
        if destination == MANIFEST:
            continue
        source = str(copy_map[destination.as_posix()])
        if hashes[destination.as_posix()] != artifacts[source]:
            raise ValueError(f"registered file hash mismatch: {destination.as_posix()}")
    if stage == "finalizer" and hashes[GOLD.as_posix()] != artifacts[GOLD.as_posix()]:
        raise ValueError("finalizer gold hash mismatch")
    return hashes


def _registered_source_module_sha256(root: Path, source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    completed = subprocess.run(
        ["git", "show", f"{source_commit}:src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout)


def preflight(
    root: Path, cache: Path, runtime_root: Path, source_commit: str
) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("registered oracle preflight requires Linux x86_64")
    if not _network_is_disabled():
        raise RuntimeError("registered oracle preflight requires disabled outbound network")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    registered_hashes = _validate_registered_tree(root, stage="preflight")
    query_payload = read_json(root / QUERY)
    if "expected_source_id" in query_payload or "stages" in query_payload:
        raise ValueError("query bundle contains gold or stage data")
    query = _query(root)
    documents = _documents(root)
    model_rows = []
    for spec in _model_specs(root, registered=True):
        hashes = _verify_model_files(spec, cache)
        runtime = _load_model(spec, cache)
        tokenizer, model, torch = runtime
        encoded = tokenizer(
            ["synthetic query"],
            ["synthetic passage"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            value = float(model(**encoded).logits.reshape(-1)[0])
        if not math.isfinite(value):
            raise ValueError("synthetic model probe emitted a non-finite logit")
        del runtime, tokenizer, model
        gc.collect()
        model_rows.append(
            {
                "kind": spec["kind"],
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "model_files_sha256": hashes,
            }
        )
    if runtime_root.exists():
        raise FileExistsError("registered runtime root must be fresh before preflight")
    runtime_root.mkdir(parents=True)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "status": "preflight_valid",
        "registered_query_execution_count": 0,
        "synthetic_forward_inference_count": len(model_rows),
        "corpus_document_count": len(documents),
        "query_sha256": sha256_bytes(query.encode("utf-8")),
        "network": "disabled-by-container",
        "holdout_bearing_input_file_count": 0,
        "shared_database_open_count": 0,
        "gold_present": False,
        "registered_source_sha256": registered_hashes,
        "models": model_rows,
    }
    write_json_exclusive(runtime_root / ATTESTATION, payload)
    return payload


def claim(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    registered_hashes = _validate_registered_tree(root, stage="claim")
    attestation = read_json(runtime_root / ATTESTATION)
    if (
        attestation.get("status") != "preflight_valid"
        or attestation.get("source_commit") != source_commit
        or attestation.get("registered_query_execution_count") != 0
        or attestation.get("holdout_bearing_input_file_count") != 0
        or attestation.get("gold_present") is not False
        or attestation.get("registered_source_sha256") != registered_hashes
    ):
        raise ValueError("preflight attestation is not bound to this registered source")
    if any((root / path).exists() for path in (CLAIM, RESULT, ERROR)):
        raise FileExistsError("registered oracle evidence is append-only and already exists")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "source_commit": source_commit,
        "manifest_sha256": sha256_file(root / MANIFEST),
        "preflight_attestation_sha256": sha256_file(runtime_root / ATTESTATION),
        "retry_count": 0,
        "registered_query_count": 1,
        "model_worker_count": 2,
        "holdout_bearing_input_file_count": 0,
        "worker_gold_present": False,
    }
    write_json_exclusive(root / CLAIM, payload)
    return payload


def worker(root: Path, cache: Path, runtime_root: Path, kind: str) -> dict[str, Any]:
    _validate_registered_tree(root, stage="worker")
    if (root / GOLD).exists():
        raise ValueError("worker environment must not contain the gold bundle")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("registered oracle worker requires Linux x86_64")
    if not _network_is_disabled():
        raise RuntimeError("registered oracle worker requires disabled outbound network")
    specs = {row["kind"]: row for row in _model_specs(root, registered=True)}
    if kind not in specs:
        raise ValueError(f"unknown model kind: {kind}")
    output = runtime_root / f"{kind}.json"
    if output.exists():
        raise FileExistsError(f"worker output already exists: {kind}")
    query = _query(root)
    documents = _documents(root)
    spec = specs[kind]
    model_hashes = _verify_model_files(spec, cache)
    started = time.perf_counter()
    runtime = _load_model(spec, cache)
    try:
        rows, pair_count = _score_documents(query, documents, runtime)
    finally:
        del runtime
        gc.collect()
    elapsed = time.perf_counter() - started
    import resource

    payload = {
        "kind": kind,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "model_files_sha256": model_hashes,
        "dependencies": _dependency_versions(),
        "metrics": {
            "runtime_seconds": elapsed,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "pair_count": pair_count,
            "document_count": len(rows),
            "chunk_count": sum(row["chunk_count"] for row in rows),
        },
        "ranking_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "documents": rows,
    }
    write_json_exclusive(output, payload)
    return {"kind": kind, "status": "worker_complete", "ranking_sha256": payload["ranking_sha256"]}


def _expected_registered_source_hashes(
    root: Path, *, registered: bool
) -> dict[str, str]:
    manifest = _manifest(root, verify_files=not registered)
    copy_map = manifest["copy_map"]
    hashes = {}
    for destination in FINALIZER_REGISTERED_FILES:
        source = destination if registered else Path(str(copy_map[destination.as_posix()]))
        hashes[destination.as_posix()] = sha256_file(root / source)
    return hashes


def _result_payload(
    root: Path,
    source_commit: str,
    worker_payloads: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    expected = _gold(root)
    models = []
    ranks = []
    for worker_payload in worker_payloads:
        rows = worker_payload.get("documents")
        if not isinstance(rows, list):
            raise TypeError("worker documents must be a list")
        matches = [row for row in rows if row.get("source_id") == expected]
        if len(matches) != 1:
            raise ValueError("gold source is missing or duplicated in model ranking")
        rank = int(matches[0]["rank"])
        ranks.append(rank)
        models.append(
            {
                **worker_payload,
                "expected_source_rank": rank,
                "expected_source_within_cutoff": rank <= RERANK_CUTOFF,
            }
        )
    manifest = _manifest(root, verify_files=False)
    input_paths = [MANIFEST, CORPUS, QUERY, GOLD, MODEL_REGISTRY, SCHEMA]
    registered_hashes = _expected_registered_source_hashes(root, registered=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "purpose": manifest["purpose"],
        "source_commit": source_commit,
        "case_id": CASE_ID,
        "query": _query(root),
        "expected_source_id": expected,
        "corpus_document_count": 93,
        "rerank_cutoff": RERANK_CUTOFF,
        "classification_policy": manifest["classification_policy"],
        "classification": classify_ranks(ranks),
        "environment": {
            "os": "linux",
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu_threads": CPU_THREADS,
            "network": "disabled",
            "fresh_runtime": True,
            "holdout_bearing_input_file_count": 0,
            "shared_database_open_count": 0,
            "runtime_seconds": time.perf_counter() - started,
        },
        "inputs_sha256": {
            path.as_posix(): sha256_file(root / path) for path in input_paths
        },
        "registered_source_sha256": registered_hashes,
        "source_module_sha256": sha256_file(
            root / "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py"
        ),
        "models": models,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_result(
    payload: Mapping[str, Any], root: Path = ROOT, *, registered: bool = False
) -> None:
    expected_fields = {
        "case_id", "classification", "classification_policy", "corpus_document_count",
        "environment", "expected_source_id", "inputs_sha256", "models", "payload_sha256",
        "protocol_id", "purpose", "query", "registered_source_sha256", "rerank_cutoff",
        "schema_version", "source_commit", "source_module_sha256",
    }
    _exact_keys(payload, expected_fields, "result")
    manifest = _manifest(root, verify_files=not registered)
    if (
        payload["schema_version"] != 1
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["purpose"] != manifest["purpose"]
        or payload["case_id"] != CASE_ID
        or payload["query"] != EXPECTED_QUERY
        or payload["expected_source_id"] != EXPECTED_SOURCE_ID
        or payload["corpus_document_count"] != 93
        or payload["rerank_cutoff"] != RERANK_CUTOFF
        or payload["classification_policy"] != manifest["classification_policy"]
        or not re.fullmatch(r"[0-9a-f]{40}", str(payload["source_commit"]))
    ):
        raise ValueError("result identity mismatch")
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise TypeError("result environment must be an object")
    _exact_keys(
        environment,
        {"architecture", "cpu_threads", "fresh_runtime", "holdout_bearing_input_file_count", "network", "os", "python", "runtime_seconds", "shared_database_open_count"},
        "result environment",
    )
    if (
        environment["os"] != "linux"
        or environment["architecture"] != "x86_64"
        or environment["cpu_threads"] != CPU_THREADS
        or environment["network"] != "disabled"
        or environment["fresh_runtime"] is not True
        or environment["holdout_bearing_input_file_count"] != 0
        or environment["shared_database_open_count"] != 0
        or not isinstance(environment["runtime_seconds"], (int, float))
        or environment["runtime_seconds"] <= 0
    ):
        raise ValueError("result environment contract mismatch")
    input_paths = (MANIFEST, CORPUS, QUERY, GOLD, MODEL_REGISTRY, SCHEMA)
    expected_inputs = {path.as_posix(): sha256_file(root / path) for path in input_paths}
    if payload["inputs_sha256"] != expected_inputs:
        raise ValueError("result input hash binding mismatch")
    expected_module_hash = (
        sha256_file(root / "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py")
        if registered
        else _registered_source_module_sha256(root, str(payload["source_commit"]))
    )
    if payload["source_module_sha256"] != expected_module_hash:
        raise ValueError("result runner hash binding mismatch")
    if payload["registered_source_sha256"] != _expected_registered_source_hashes(
        root, registered=registered
    ):
        raise ValueError("result registered source hash binding mismatch")
    models = payload["models"]
    if not isinstance(models, list) or [row.get("kind") for row in models] != list(MODEL_KINDS):
        raise ValueError("result model order mismatch")
    specs = {
        row["kind"]: row for row in _model_specs(root, registered=registered)
    }
    ranks = []
    model_fields = {"dependencies", "documents", "expected_source_rank", "expected_source_within_cutoff", "kind", "metrics", "model_files_sha256", "model_id", "ranking_sha256", "revision"}
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("result model must be an object")
        _exact_keys(model, model_fields, "result model")
        spec = specs[model["kind"]]
        if model["model_id"] != spec["model_id"] or model["revision"] != spec["revision"]:
            raise ValueError("result model identity mismatch")
        required = spec.get("required_files")
        if not isinstance(required, list):
            raise TypeError("pinned model required_files must be a list")
        expected_model_files = {str(row["path"]): row for row in required}
        if set(model["model_files_sha256"]) != set(expected_model_files):
            raise ValueError("result model file hash set mismatch")
        for relative, observed in model["model_files_sha256"].items():
            expected_lfs = expected_model_files[relative].get("lfs_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", str(observed)) or (
                expected_lfs is not None and observed != expected_lfs
            ):
                raise ValueError("result pinned model file SHA-256 mismatch")
        if not isinstance(model["dependencies"], dict) or set(model["dependencies"]) != set(MODEL_DISTRIBUTIONS):
            raise ValueError("result dependency version set mismatch")
        metrics = model["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("result model metrics must be an object")
        _exact_keys(metrics, {"chunk_count", "document_count", "pair_count", "peak_rss_bytes", "runtime_seconds"}, "result model metrics")
        if metrics["document_count"] != 93 or metrics["pair_count"] != metrics["chunk_count"] or metrics["peak_rss_bytes"] <= 0 or metrics["runtime_seconds"] <= 0:
            raise ValueError("result model metrics mismatch")
        _validate_document_rows(model["documents"], str(model["kind"]))
        if model["ranking_sha256"] != sha256_bytes(canonical_json_bytes(model["documents"])):
            raise ValueError("result ranking hash mismatch")
        matches = [row for row in model["documents"] if row["source_id"] == EXPECTED_SOURCE_ID]
        if len(matches) != 1 or model["expected_source_rank"] != matches[0]["rank"]:
            raise ValueError("result gold rank does not match raw document rows")
        rank = int(model["expected_source_rank"])
        ranks.append(rank)
        if model["expected_source_within_cutoff"] is not (rank <= RERANK_CUTOFF):
            raise ValueError("result cutoff membership mismatch")
    if payload["classification"] != classify_ranks(ranks):
        raise ValueError("result bottleneck classification mismatch")
    without_hash = dict(payload)
    claimed_hash = without_hash.pop("payload_sha256")
    if claimed_hash != sha256_bytes(canonical_json_bytes(without_hash)):
        raise ValueError("result payload hash mismatch")


def finalize(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _validate_registered_tree(root, stage="finalizer")
    claim_payload = read_json(root / CLAIM)
    if claim_payload.get("source_commit") != source_commit or claim_payload.get("retry_count") != 0:
        raise ValueError("claim is not bound to this finalization")
    worker_payloads = [read_json(runtime_root / f"{kind}.json") for kind in MODEL_KINDS]
    started = time.perf_counter()
    payload = _result_payload(root, source_commit, worker_payloads, started)
    validate_result(payload, root, registered=True)
    write_json_exclusive(root / RESULT, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "observed_valid",
        "classification": payload["classification"],
        "ranks": {row["kind"]: row["expected_source_rank"] for row in payload["models"]},
        "payload_sha256": payload["payload_sha256"],
    }


def record_error(root: Path, source_commit: str, message: str) -> dict[str, Any]:
    if (root / RESULT).exists():
        raise FileExistsError("cannot append error after result")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "source_commit": source_commit,
        "retry_count": 0,
        "error": message,
    }
    write_json_exclusive(root / ERROR, payload)
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    manifest = _manifest(root)
    _query(root)
    _gold(root)
    documents = _documents(root)
    evidence = {
        "claim": (root / CLAIM).exists(),
        "result": (root / RESULT).exists(),
        "error": (root / ERROR).exists(),
    }
    if evidence["result"] and evidence["error"]:
        raise ValueError("result and error evidence cannot coexist")
    if (evidence["result"] or evidence["error"]) and not evidence["claim"]:
        raise ValueError("terminal evidence requires the append-only claim")
    result = None
    if evidence["claim"]:
        claim_payload = read_json(root / CLAIM)
        if (
            claim_payload.get("protocol_id") != PROTOCOL_ID
            or claim_payload.get("case_id") != CASE_ID
            or claim_payload.get("manifest_sha256") != sha256_file(root / MANIFEST)
            or claim_payload.get("retry_count") != 0
            or claim_payload.get("registered_query_count") != 1
            or claim_payload.get("model_worker_count") != 2
            or claim_payload.get("holdout_bearing_input_file_count") != 0
            or claim_payload.get("worker_gold_present") is not False
        ):
            raise ValueError("claim identity mismatch")
    if evidence["result"]:
        result = read_json(root / RESULT)
        validate_result(result, root)
    if evidence["error"]:
        failure = read_json(root / ERROR)
        if failure.get("protocol_id") != PROTOCOL_ID or failure.get("retry_count") != 0:
            raise ValueError("error evidence identity mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "observed_valid" if result is not None else ("failed" if evidence["error"] else "result_free"),
        "corpus_document_count": len(documents),
        "case_id": CASE_ID,
        "holdout_bearing_input_file_count": 0,
        "worker_gold_present": False,
        "github_rag_request_count": 0,
        "shared_database_open_count": 0,
        "classification": result["classification"] if result is not None else None,
        "ranks": ({row["kind"]: row["expected_source_rank"] for row in result["models"]} if result is not None else None),
        "forbidden_registered_paths": manifest["forbidden_registered_paths"],
        "evidence": evidence,
    }


def probe(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    _query(root)
    _gold(root)
    text = "x" * 1200
    chunks = project_passages(text)
    if chunks[0]["start_codepoint"] != 0 or chunks[-1]["end_codepoint"] != len(text):
        raise ValueError("synthetic projection coverage failed")
    if classify_ranks([5, 20]) != "candidate_generation_bottleneck" or classify_ranks([21, 93]) != "semantic_discrimination_bottleneck" or classify_ranks([20, 21]) != "undetermined":
        raise ValueError("classification probe failed")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "synthetic_probe_valid",
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "holdout_bearing_input_file_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "probe"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=ROOT)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, required=True)
    preflight_parser.add_argument("--cache", type=Path, required=True)
    preflight_parser.add_argument("--runtime-root", type=Path, required=True)
    preflight_parser.add_argument("--source-commit", required=True)
    claim_parser = subparsers.add_parser("claim")
    claim_parser.add_argument("--root", type=Path, required=True)
    claim_parser.add_argument("--runtime-root", type=Path, required=True)
    claim_parser.add_argument("--source-commit", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    worker_parser.add_argument("--cache", type=Path, required=True)
    worker_parser.add_argument("--runtime-root", type=Path, required=True)
    worker_parser.add_argument("--kind", choices=MODEL_KINDS, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--root", type=Path, required=True)
    finalize_parser.add_argument("--runtime-root", type=Path, required=True)
    finalize_parser.add_argument("--source-commit", required=True)
    error_parser = subparsers.add_parser("record-error")
    error_parser.add_argument("--root", type=Path, required=True)
    error_parser.add_argument("--source-commit", required=True)
    error_parser.add_argument("--message", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "audit":
        result = audit(arguments.root)
    elif arguments.command == "probe":
        result = probe(arguments.root)
    elif arguments.command == "preflight":
        result = preflight(arguments.root, arguments.cache, arguments.runtime_root, arguments.source_commit)
    elif arguments.command == "claim":
        result = claim(arguments.root, arguments.runtime_root, arguments.source_commit)
    elif arguments.command == "worker":
        result = worker(arguments.root, arguments.cache, arguments.runtime_root, arguments.kind)
    elif arguments.command == "finalize":
        result = finalize(arguments.root, arguments.runtime_root, arguments.source_commit)
    else:
        result = record_error(arguments.root, arguments.source_commit, arguments.message)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

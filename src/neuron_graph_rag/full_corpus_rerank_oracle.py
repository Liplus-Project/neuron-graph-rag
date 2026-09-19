from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cross_encoder_precision_v2_evaluation import project_passages

PROTOCOL_ID = "github-retrieval-parity-v5-full-corpus-rerank-oracle-v1"
ROOT = Path(__file__).resolve().parents[2]
MODULE = "neuron_graph_rag.full_corpus_rerank_oracle"
MANIFEST = Path("tests/fixtures/full_corpus_rerank_oracle_v1.manifest.json")
SCHEMA = Path("tests/fixtures/full_corpus_rerank_oracle_v1.schema.json")
RESULT = Path("tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json")
CLAIM = Path("tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json")
ERROR = Path("tests/evidence/full_corpus_rerank_oracle_v1/development.error.json")
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
QUERIES = Path("tests/fixtures/github_retrieval_parity_v5.queries.json")
GOLD = Path("tests/fixtures/github_retrieval_parity_v5.gold.json")
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")
CASE_ID = "v5path-dev-semantic-axis-separation"
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
MODEL_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "psutil",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields mismatch: expected={sorted(expected)!r} "
            f"actual={sorted(value)!r}"
        )


def _source_id(repository: str, path: str) -> str:
    return f"github:{repository}:{path}"


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    manifest = read_json(root / MANIFEST)
    _exact_keys(
        manifest,
        {
            "aggregation",
            "artifact_sha256",
            "case_id",
            "classification_policy",
            "corpus_document_count",
            "models",
            "projection",
            "protocol_id",
            "purpose",
            "rerank_cutoff",
            "runtime",
            "schema_version",
            "source_files",
        },
        "manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["case_id"] != CASE_ID
        or manifest["corpus_document_count"] != 93
        or manifest["rerank_cutoff"] != RERANK_CUTOFF
        or manifest["aggregation"] != "maximum-raw-logit-per-document"
        or manifest["projection"]
        != {
            "unit": "unicode-codepoint",
            "window": WINDOW_CODEPOINTS,
            "overlap": OVERLAP_CODEPOINTS,
            "coverage": "start-to-end",
            "tokenizer_max_length": 512,
        }
        or manifest["runtime"]
        != {
            "batch_size": BATCH_SIZE,
            "cpu_threads": CPU_THREADS,
            "device": "cpu",
            "network": "disabled",
            "platform": "linux-x86_64",
            "fresh_runtime": True,
        }
    ):
        raise ValueError("manifest identity or execution contract mismatch")
    expected_policy = {
        "all_models_at_or_above_cutoff": "candidate_generation_bottleneck",
        "all_models_below_cutoff": "semantic_discrimination_bottleneck",
        "mixed": "undetermined",
    }
    if manifest["classification_policy"] != expected_policy:
        raise ValueError("classification policy mismatch")
    source_files = manifest["source_files"]
    if source_files != {
        "corpus": CORPUS.as_posix(),
        "gold": GOLD.as_posix(),
        "model_registry": MODEL_REGISTRY.as_posix(),
        "queries": QUERIES.as_posix(),
        "schema": SCHEMA.as_posix(),
    }:
        raise ValueError("manifest source file contract mismatch")
    artifact_sha256 = manifest["artifact_sha256"]
    if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != set(
        source_files.values()
    ):
        raise ValueError("manifest artifact hash registry mismatch")
    for relative, expected in artifact_sha256.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid registered SHA-256: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"registered artifact changed: {relative}")
    models = manifest["models"]
    if not isinstance(models, list) or [row.get("kind") for row in models] != list(
        MODEL_KINDS
    ):
        raise ValueError("manifest model order mismatch")
    return manifest


def _target_query(root: Path = ROOT) -> str:
    queries = read_json(root / QUERIES)
    rows = queries.get("stages", {}).get("development", [])
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    if len(matches) != 1 or set(matches[0]) != {"case_id", "cohort", "query"}:
        raise ValueError("target development query is missing or malformed")
    if matches[0]["cohort"] != "semantic_paraphrase":
        raise ValueError("target query cohort mismatch")
    return str(matches[0]["query"])


def _expected_source(root: Path = ROOT) -> str:
    gold = read_json(root / GOLD)
    rows = gold.get("stages", {}).get("development", [])
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    if len(matches) != 1:
        raise ValueError("target development gold is missing or duplicated")
    expected = matches[0].get("expected_source_ids")
    if expected != [EXPECTED_SOURCE_ID]:
        raise ValueError("target development gold identity mismatch")
    return EXPECTED_SOURCE_ID


def _documents(root: Path = ROOT) -> list[dict[str, Any]]:
    corpus = read_json(root / CORPUS)
    repository = corpus.get("repository")
    rows = corpus.get("documents")
    if repository != "Liplus-Project/neuron-graph-rag" or not isinstance(rows, list):
        raise ValueError("v5 source corpus identity mismatch")
    if len(rows) != 93:
        raise ValueError("v5 source corpus must contain exactly 93 documents")
    documents: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("corpus document must be an object")
        path = str(row.get("path"))
        content = str(row.get("content"))
        content_sha256 = str(row.get("content_sha256"))
        if sha256_bytes(content.encode("utf-8")) != content_sha256:
            raise ValueError(f"corpus document content hash mismatch: {path}")
        source_id = _source_id(str(repository), path)
        if source_id in source_ids:
            raise ValueError(f"duplicate source ID: {source_id}")
        source_ids.add(source_id)
        documents.append(
            {
                "source_id": source_id,
                "path": path,
                "content": content,
                "content_sha256": content_sha256,
            }
        )
    if EXPECTED_SOURCE_ID not in source_ids:
        raise ValueError("gold document is absent from exact corpus")
    return documents


def _model_specs(root: Path = ROOT) -> list[dict[str, Any]]:
    manifest = _manifest(root)
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


def _snapshot_path(cache: Path, model_id: str, revision: str) -> Path:
    path = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"pinned model snapshot is unavailable: {path}")
    return path


def _verify_model_files(spec: Mapping[str, Any], cache: Path) -> dict[str, str]:
    snapshot = _snapshot_path(cache, str(spec["model_id"]), str(spec["revision"]))
    hashes: dict[str, str] = {}
    required = spec.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError("pinned model required_files are missing")
    for row in required:
        if not isinstance(row, dict):
            raise TypeError("model required file row must be an object")
        relative = str(row.get("path"))
        path = snapshot / relative
        if not path.is_file() or path.stat().st_size != row.get("size"):
            raise ValueError(f"pinned model file size mismatch: {relative}")
        observed = sha256_file(path)
        lfs_sha256 = row.get("lfs_sha256")
        if lfs_sha256 is not None and observed != lfs_sha256:
            raise ValueError(f"pinned model file hash mismatch: {relative}")
        hashes[relative] = observed
    return hashes


def _load_model(spec: Mapping[str, Any], cache: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(CPU_THREADS)
    snapshot = _snapshot_path(cache, str(spec["model_id"]), str(spec["revision"]))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("model is not in pinned CPU eval mode")
    return tokenizer, model, torch


def _score_documents(
    query: str,
    documents: Sequence[Mapping[str, Any]],
    runtime: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], int]:
    tokenizer, model, torch = runtime
    scored: list[dict[str, Any]] = []
    pair_count = 0
    for document in documents:
        text = str(document["content"])
        chunks = project_passages(
            text, window=WINDOW_CODEPOINTS, overlap=OVERLAP_CODEPOINTS
        )
        if not chunks or chunks[0]["start_codepoint"] != 0:
            raise ValueError("passage projection omitted the document start")
        if chunks[-1]["end_codepoint"] != len(text):
            raise ValueError("passage projection omitted the document end")
        scores: list[float] = []
        for offset in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[offset : offset + BATCH_SIZE]
            encoded = tokenizer(
                [query] * len(batch),
                [str(chunk["text"]) for chunk in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to("cpu") for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).to(dtype=torch.float32)
            values = [float(value) for value in logits.tolist()]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("cross-encoder emitted a non-finite logit")
            scores.extend(values)
        pair_count += len(chunks)
        best = max(scores)
        winner = min(index for index, value in enumerate(scores) if value == best)
        chunk = chunks[winner]
        scored.append(
            {
                "source_id": document["source_id"],
                "path": document["path"],
                "character_count": len(text),
                "chunk_count": len(chunks),
                "best_chunk_score": best,
                "winning_chunk_index": winner,
                "winning_chunk_start_codepoint": chunk["start_codepoint"],
                "winning_chunk_end_codepoint": chunk["end_codepoint"],
                "winning_chunk_sha256": sha256_bytes(
                    str(chunk["text"]).encode("utf-8")
                ),
            }
        )
    scored.sort(key=lambda row: (-row["best_chunk_score"], row["source_id"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return scored, pair_count


def _dependency_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in MODEL_DISTRIBUTIONS}


def _network_is_disabled() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            return False
    except OSError:
        return True


def _worker_payload(root: Path, cache: Path, kind: str) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("registered oracle worker requires Linux x86_64")
    if not _network_is_disabled():
        raise RuntimeError("registered oracle worker requires disabled outbound network")
    specs = {row["kind"]: row for row in _model_specs(root)}
    if kind not in specs:
        raise ValueError(f"unknown model kind: {kind}")
    query = _target_query(root)
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

    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "kind": kind,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "model_files_sha256": model_hashes,
        "dependencies": _dependency_versions(),
        "metrics": {
            "runtime_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes,
            "pair_count": pair_count,
            "document_count": len(rows),
            "chunk_count": sum(row["chunk_count"] for row in rows),
        },
        "ranking_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "documents": rows,
    }


def classify_ranks(ranks: Sequence[int], cutoff: int = RERANK_CUTOFF) -> str:
    if not ranks or cutoff < 1 or any(rank < 1 for rank in ranks):
        raise ValueError("ranks and cutoff must be positive")
    within = [rank <= cutoff for rank in ranks]
    if all(within):
        return "candidate_generation_bottleneck"
    if not any(within):
        return "semantic_discrimination_bottleneck"
    return "undetermined"


def _result_payload(
    root: Path,
    source_commit: str,
    worker_payloads: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    expected = _expected_source(root)
    models = []
    ranks = []
    for worker in worker_payloads:
        rows = worker.get("documents")
        if not isinstance(rows, list):
            raise TypeError("worker documents must be a list")
        matches = [row for row in rows if row.get("source_id") == expected]
        if len(matches) != 1:
            raise ValueError("gold source is missing or duplicated in model ranking")
        rank = int(matches[0]["rank"])
        ranks.append(rank)
        models.append(
            {
                **worker,
                "expected_source_rank": rank,
                "expected_source_within_cutoff": rank <= RERANK_CUTOFF,
            }
        )
    manifest = _manifest(root)
    input_paths = [MANIFEST, CORPUS, QUERIES, GOLD, MODEL_REGISTRY, SCHEMA]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "purpose": manifest["purpose"],
        "source_commit": source_commit,
        "case_id": CASE_ID,
        "query": _target_query(root),
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
            "shared_database_open_count": 0,
            "runtime_seconds": time.perf_counter() - started,
        },
        "inputs_sha256": {
            path.as_posix(): sha256_file(root / path) for path in input_paths
        },
        "source_module_sha256": "",
        "models": models,
    }
    module_path = root / "src/neuron_graph_rag/full_corpus_rerank_oracle.py"
    payload["source_module_sha256"] = sha256_file(module_path)
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _validate_document_rows(rows: object, name: str) -> None:
    if not isinstance(rows, list) or len(rows) != 93:
        raise ValueError(f"{name} must contain 93 document rows")
    expected_fields = {
        "best_chunk_score",
        "character_count",
        "chunk_count",
        "path",
        "rank",
        "source_id",
        "winning_chunk_end_codepoint",
        "winning_chunk_index",
        "winning_chunk_sha256",
        "winning_chunk_start_codepoint",
    }
    ranks = set()
    source_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{name} document row must be an object")
        _exact_keys(row, expected_fields, f"{name} document row")
        if not isinstance(row["rank"], int) or not 1 <= row["rank"] <= 93:
            raise ValueError(f"{name} document rank is invalid")
        if not isinstance(row["best_chunk_score"], (int, float)) or not math.isfinite(
            row["best_chunk_score"]
        ):
            raise ValueError(f"{name} document score is invalid")
        if (
            not isinstance(row["character_count"], int)
            or row["character_count"] < 1
            or not isinstance(row["chunk_count"], int)
            or row["chunk_count"] < 1
            or not isinstance(row["winning_chunk_index"], int)
            or not 0 <= row["winning_chunk_index"] < row["chunk_count"]
            or not isinstance(row["winning_chunk_start_codepoint"], int)
            or not isinstance(row["winning_chunk_end_codepoint"], int)
            or not 0 <= row["winning_chunk_start_codepoint"]
            < row["winning_chunk_end_codepoint"]
            <= row["character_count"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(row["winning_chunk_sha256"]))
        ):
            raise ValueError(f"{name} document projection metadata is invalid")
        ranks.add(row["rank"])
        source_ids.add(row["source_id"])
    if ranks != set(range(1, 94)) or len(source_ids) != 93:
        raise ValueError(f"{name} ranking is incomplete or duplicated")
    expected_order = sorted(rows, key=lambda row: (-row["best_chunk_score"], row["source_id"]))
    if [row["source_id"] for row in rows] != [row["source_id"] for row in expected_order]:
        raise ValueError(f"{name} ranking order does not match raw logits")


def validate_result(payload: Mapping[str, Any], root: Path = ROOT) -> None:
    _exact_keys(
        payload,
        {
            "case_id",
            "classification",
            "classification_policy",
            "corpus_document_count",
            "environment",
            "expected_source_id",
            "inputs_sha256",
            "models",
            "payload_sha256",
            "protocol_id",
            "purpose",
            "query",
            "rerank_cutoff",
            "schema_version",
            "source_commit",
            "source_module_sha256",
        },
        "result",
    )
    manifest = _manifest(root)
    if (
        payload["schema_version"] != 1
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["purpose"] != manifest["purpose"]
        or payload["case_id"] != CASE_ID
        or payload["query"] != _target_query(root)
        or payload["expected_source_id"] != _expected_source(root)
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
        {
            "architecture",
            "cpu_threads",
            "fresh_runtime",
            "network",
            "os",
            "python",
            "runtime_seconds",
            "shared_database_open_count",
        },
        "result environment",
    )
    if (
        environment["os"] != "linux"
        or environment["architecture"] != "x86_64"
        or environment["cpu_threads"] != CPU_THREADS
        or environment["network"] != "disabled"
        or environment["fresh_runtime"] is not True
        or environment["shared_database_open_count"] != 0
        or not isinstance(environment["runtime_seconds"], (int, float))
        or environment["runtime_seconds"] <= 0
    ):
        raise ValueError("result environment contract mismatch")
    expected_inputs = {
        path.as_posix(): sha256_file(root / path)
        for path in (MANIFEST, CORPUS, QUERIES, GOLD, MODEL_REGISTRY, SCHEMA)
    }
    if payload["inputs_sha256"] != expected_inputs:
        raise ValueError("result input hash binding mismatch")
    module_path = root / "src/neuron_graph_rag/full_corpus_rerank_oracle.py"
    if payload["source_module_sha256"] != sha256_file(module_path):
        raise ValueError("result runner hash binding mismatch")
    models = payload["models"]
    if not isinstance(models, list) or [row.get("kind") for row in models] != list(
        MODEL_KINDS
    ):
        raise ValueError("result model order mismatch")
    specs = {row["kind"]: row for row in _model_specs(root)}
    ranks = []
    model_fields = {
        "dependencies",
        "documents",
        "expected_source_rank",
        "expected_source_within_cutoff",
        "kind",
        "metrics",
        "model_files_sha256",
        "model_id",
        "ranking_sha256",
        "revision",
    }
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("result model must be an object")
        _exact_keys(model, model_fields, "result model")
        spec = specs[model["kind"]]
        if model["model_id"] != spec["model_id"] or model["revision"] != spec["revision"]:
            raise ValueError("result model identity mismatch")
        required_files = spec.get("required_files")
        if not isinstance(required_files, list):
            raise TypeError("pinned model required_files must be a list")
        expected_model_files = {str(row["path"]): row for row in required_files}
        if set(model["model_files_sha256"]) != set(expected_model_files):
            raise ValueError("result model file hash set mismatch")
        for relative, observed in model["model_files_sha256"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", str(observed)):
                raise ValueError("result model file SHA-256 is malformed")
            expected_lfs = expected_model_files[relative].get("lfs_sha256")
            if expected_lfs is not None and observed != expected_lfs:
                raise ValueError("result pinned model file SHA-256 mismatch")
        dependencies = model["dependencies"]
        if not isinstance(dependencies, dict) or set(dependencies) != set(
            MODEL_DISTRIBUTIONS
        ):
            raise ValueError("result dependency version set mismatch")
        metrics = model["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("result model metrics must be an object")
        _exact_keys(
            metrics,
            {
                "chunk_count",
                "document_count",
                "pair_count",
                "peak_rss_bytes",
                "runtime_seconds",
            },
            "result model metrics",
        )
        if (
            metrics["document_count"] != 93
            or metrics["pair_count"] != metrics["chunk_count"]
            or not isinstance(metrics["peak_rss_bytes"], int)
            or metrics["peak_rss_bytes"] <= 0
            or not isinstance(metrics["runtime_seconds"], (int, float))
            or metrics["runtime_seconds"] <= 0
        ):
            raise ValueError("result model metrics mismatch")
        _validate_document_rows(model["documents"], str(model["kind"]))
        if model["ranking_sha256"] != sha256_bytes(
            canonical_json_bytes(model["documents"])
        ):
            raise ValueError("result ranking hash mismatch")
        matches = [
            row
            for row in model["documents"]
            if row["source_id"] == EXPECTED_SOURCE_ID
        ]
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


def preflight(root: Path, cache: Path) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("registered oracle preflight requires Linux x86_64")
    if not _network_is_disabled():
        raise RuntimeError("registered oracle preflight requires disabled outbound network")
    _manifest(root)
    _target_query(root)
    _expected_source(root)
    documents = _documents(root)
    model_rows = []
    for spec in _model_specs(root):
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
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "preflight_valid",
        "registered_query_execution_count": 0,
        "synthetic_forward_inference_count": len(model_rows),
        "corpus_document_count": len(documents),
        "network": "disabled-by-container",
        "shared_database_open_count": 0,
        "models": model_rows,
    }


def run_once(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _manifest(root)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if not _network_is_disabled():
        raise RuntimeError("registered oracle run requires disabled outbound network")
    if runtime_root.exists():
        raise FileExistsError("registered oracle runtime root must be fresh")
    runtime_root.mkdir(parents=True)
    claim_path = root / CLAIM
    result_path = root / RESULT
    error_path = root / ERROR
    if any(path.exists() for path in (claim_path, result_path, error_path)):
        raise FileExistsError("registered oracle evidence is append-only and already exists")
    claim = {
        "protocol_id": PROTOCOL_ID,
        "case_id": CASE_ID,
        "source_commit": source_commit,
        "manifest_sha256": sha256_file(root / MANIFEST),
        "retry_count": 0,
        "registered_query_count": 1,
        "model_worker_count": 2,
    }
    write_json_exclusive(claim_path, claim)
    started = time.perf_counter()
    worker_payloads = []
    try:
        for kind in MODEL_KINDS:
            worker_output = runtime_root / f"{kind}.json"
            command = [
                sys.executable,
                "-m",
                MODULE,
                "worker",
                "--root",
                str(root),
                "--cache",
                str(cache),
                "--kind",
                kind,
                "--output",
                str(worker_output),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "NO_PROXY": "*",
                    "OMP_NUM_THREADS": str(CPU_THREADS),
                    "MKL_NUM_THREADS": str(CPU_THREADS),
                }
            )
            subprocess.run(command, cwd=root, env=environment, check=True)
            worker_payloads.append(read_json(worker_output))
        payload = _result_payload(root, source_commit, worker_payloads, started)
        validate_result(payload, root)
        write_json_exclusive(result_path, payload)
        return payload
    except BaseException as error:
        failure = {
            "protocol_id": PROTOCOL_ID,
            "case_id": CASE_ID,
            "source_commit": source_commit,
            "retry_count": 0,
            "error": f"{type(error).__name__}: {error}",
        }
        write_json_exclusive(error_path, failure)
        raise


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    _target_query(root)
    _expected_source(root)
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
    claim = None
    if evidence["claim"]:
        claim = read_json(root / CLAIM)
        _exact_keys(
            claim,
            {
                "case_id",
                "manifest_sha256",
                "model_worker_count",
                "protocol_id",
                "registered_query_count",
                "retry_count",
                "source_commit",
            },
            "claim",
        )
        if (
            claim["protocol_id"] != PROTOCOL_ID
            or claim["case_id"] != CASE_ID
            or claim["manifest_sha256"] != sha256_file(root / MANIFEST)
            or claim["retry_count"] != 0
            or claim["registered_query_count"] != 1
            or claim["model_worker_count"] != 2
            or not re.fullmatch(r"[0-9a-f]{40}", str(claim["source_commit"]))
        ):
            raise ValueError("claim identity mismatch")
    if evidence["result"]:
        result = read_json(root / RESULT)
        validate_result(result, root)
        if claim is None or result["source_commit"] != claim["source_commit"]:
            raise ValueError("result source commit is not bound to the claim")
    if evidence["error"]:
        failure = read_json(root / ERROR)
        _exact_keys(
            failure,
            {"case_id", "error", "protocol_id", "retry_count", "source_commit"},
            "error evidence",
        )
        if (
            failure["protocol_id"] != PROTOCOL_ID
            or failure["case_id"] != CASE_ID
            or failure["retry_count"] != 0
            or claim is None
            or failure["source_commit"] != claim["source_commit"]
        ):
            raise ValueError("error evidence identity mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "observed_valid" if evidence["result"] else "result_free_valid",
        "corpus_document_count": len(documents),
        "case_id": CASE_ID,
        "holdout_read_count": 0,
        "github_rag_request_count": 0,
        "shared_database_open_count": 0,
        "evidence": evidence,
    }


def probe(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    text = "x" * 1200
    chunks = project_passages(text)
    if chunks[0]["start_codepoint"] != 0 or chunks[-1]["end_codepoint"] != len(text):
        raise ValueError("synthetic projection coverage failed")
    if classify_ranks([5, 20]) != "candidate_generation_bottleneck":
        raise ValueError("candidate-generation classification probe failed")
    if classify_ranks([21, 93]) != "semantic_discrimination_bottleneck":
        raise ValueError("semantic-discrimination classification probe failed")
    if classify_ranks([20, 21]) != "undetermined":
        raise ValueError("mixed classification probe failed")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "synthetic_probe_valid",
        "synthetic_chunk_count": len(chunks),
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "probe"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=ROOT)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, default=ROOT)
    preflight_parser.add_argument("--cache", type=Path, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    worker_parser.add_argument("--cache", type=Path, required=True)
    worker_parser.add_argument("--kind", choices=MODEL_KINDS, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, required=True)
    run_parser.add_argument("--runtime-root", type=Path, required=True)
    run_parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "audit":
        result = audit(arguments.root)
    elif arguments.command == "probe":
        result = probe(arguments.root)
    elif arguments.command == "preflight":
        result = preflight(arguments.root, arguments.cache)
    elif arguments.command == "worker":
        result = _worker_payload(arguments.root, arguments.cache, arguments.kind)
        write_json_exclusive(arguments.output, result)
    else:
        result = run_once(
            arguments.root,
            arguments.cache,
            arguments.runtime_root,
            arguments.source_commit,
        )
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Holdout-absent CPU two-stage retrieval experiment for the v5 hard query."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import e5_structural_centroid_ablation as e5
from . import full_corpus_rerank_oracle as v1
from . import structural_representation_length_bias_ablation as ce
from . import structural_representation_length_bias_diagnostic as structural

PROTOCOL_ID = "github-retrieval-parity-v5-practical-two-stage-retrieval-v1"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/practical_two_stage_retrieval_v1.manifest.json")
SCHEMA = Path("tests/fixtures/practical_two_stage_retrieval_v1.schema.json")
MODELS = Path("tests/fixtures/practical_two_stage_retrieval_v1.models.json")
QUERY = Path("tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
PARITY = Path("tests/evidence/practical_two_stage_retrieval_v1/result_free_minilm_parity.json")
CLAIM = Path("tests/evidence/practical_two_stage_retrieval_v1/development.claim.json")
RESULT = Path("tests/evidence/practical_two_stage_retrieval_v1/development.observed.json")
ERROR = Path("tests/evidence/practical_two_stage_retrieval_v1/development.error.json")
ATTESTATION = Path("preflight.attestation.json")
MODEL_KINDS = ("minilm", "v2-m3")
CANDIDATE_K = 50
CUTOFF = 20
PRACTICAL_SECONDS = 600.0
PARITY_TOLERANCE = 1e-6

canonical_json_bytes = v1.canonical_json_bytes
sha256_bytes = v1.sha256_bytes
sha256_file = v1.sha256_file
read_json = v1.read_json
write_json_exclusive = v1.write_json_exclusive


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "frozen_pre_registered_execution":
        raise ValueError("frozen manifest identity mismatch")
    if value["stage_1"]["candidate_k"] != CANDIDATE_K or value["ranking"]["cutoff"] != CUTOFF:
        raise ValueError("ranking contract mismatch")
    if value["runtime"]["practical_target_seconds_per_pipeline"] != PRACTICAL_SECONDS:
        raise ValueError("runtime target mismatch")
    if value["minilm_parity_gate"]["max_absolute_difference_tolerance"] != PARITY_TOLERANCE:
        raise ValueError("parity tolerance mismatch")
    for relative, digest in value.get("artifact_sha256", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256_file(root / relative) != digest:
            raise ValueError(f"frozen artifact changed: {relative}")
    return value


def _models(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    rows = read_json(root / MODELS).get("models")
    if not isinstance(rows, list):
        raise ValueError("model registry missing")
    result = {str(row["kind"]): row for row in rows}
    if tuple(result) != ("e5", "minilm", "v2-m3"):
        raise ValueError("model order mismatch")
    return result


def _verify_model(spec: Mapping[str, Any], cache: Path) -> dict[str, str]:
    observed = v1._verify_model_files(spec, cache)
    expected = {str(row["path"]): str(row["sha256"]) for row in spec["required_files"]}
    if observed != expected:
        raise ValueError(f"pinned model hash mismatch: {spec['kind']}")
    return observed


def _registered_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _validate_tree(root: Path, stage: str) -> dict[str, str]:
    manifest = _manifest(root)
    expected = set(manifest["worker_registered_files"])
    if stage in {"worker", "finalizer"}:
        expected.add(CLAIM.as_posix())
    if stage == "finalizer":
        expected.add(GOLD.as_posix())
    actual = _registered_files(root)
    if actual != expected:
        raise ValueError(
            f"registered allowlist mismatch: missing={sorted(expected-actual)!r} extra={sorted(actual-expected)!r}"
        )
    for forbidden in manifest["forbidden_registered_paths"]:
        if (root / forbidden).exists():
            raise ValueError(f"forbidden registered path exists: {forbidden}")
    return {relative: sha256_file(root / relative) for relative in sorted(expected)}


def _query(root: Path) -> str:
    value = read_json(root / QUERY)
    if set(value) != {"schema_version", "protocol_id", "case_id", "query"}:
        raise ValueError("query bundle fields mismatch")
    return str(value["query"])


def _gold(root: Path) -> str:
    value = read_json(root / GOLD)
    if set(value) != {"schema_version", "protocol_id", "case_id", "expected_source_id"}:
        raise ValueError("gold bundle fields mismatch")
    return str(value["expected_source_id"])


def parity(root: Path, cache: Path, output: Path) -> dict[str, Any]:
    """Compare direct MiniLM loading with the Transformers pipeline on synthetic pairs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    spec = _models(root)["minilm"]
    model_hashes = _verify_model(spec, cache)
    snapshot = v1._snapshot_path(cache, spec["model_id"], spec["revision"])
    queries = ["synthetic retrieval question", "日本語の合成検索質問"]
    passages = ["a synthetic passage for loader parity", "ローダー比較だけに使う合成文書"]
    direct_tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    direct_model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32
    ).to("cpu").eval()
    direct_inputs = direct_tokenizer(
        queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.inference_mode():
        direct_logits = direct_model(**direct_inputs).logits.reshape(-1).to(torch.float32).tolist()
    reference_tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    reference_model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32
    ).to("cpu").eval()
    reference_inputs = reference_tokenizer(
        queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    classifier = pipeline(
        "text-classification", model=reference_model, tokenizer=reference_tokenizer,
        device=-1, function_to_apply="none",
    )
    pipeline_rows = classifier(
        [{"text": query, "text_pair": passage} for query, passage in zip(queries, passages, strict=True)],
        batch_size=len(queries), truncation=True, max_length=512,
    )
    reference_logits = [float(row["score"]) for row in pipeline_rows]
    token_fields_equal = all(
        direct_inputs[name].tolist() == reference_inputs[name].tolist()
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in direct_inputs or name in reference_inputs
    )
    differences = [abs(left - right) for left, right in zip(direct_logits, reference_logits, strict=True)]
    passed = token_fields_equal and max(differences) <= PARITY_TOLERANCE
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "result_free_parity_valid" if passed else "result_free_parity_failed",
        "registered_query_execution_count": 0,
        "synthetic_pairs": [{"query": q, "passage": p} for q, p in zip(queries, passages, strict=True)],
        "loader_boundary": "direct AutoTokenizer/AutoModelForSequenceClassification versus Transformers text-classification pipeline loaded independently from the same offline snapshot",
        "logit_contract": "single raw sequence-classification logit; no sigmoid or softmax",
        "tokenization": {"truncation": True, "max_length": 512, "padding": True},
        "token_fields_equal": token_fields_equal,
        "direct_logits": direct_logits,
        "pipeline_logits": reference_logits,
        "absolute_differences": differences,
        "max_absolute_difference_tolerance": PARITY_TOLERANCE,
        "model_files_sha256": model_hashes,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(output, payload)
    if not passed:
        raise ValueError("MiniLM loader parity failed")
    return payload


def _require_offline_linux() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not v1._network_is_disabled():
        raise RuntimeError("registered execution requires offline Linux x86_64")


def preflight_e5(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _require_offline_linux()
    hashes = _validate_tree(root, "preflight")
    if (root / GOLD).exists() or runtime_root.exists():
        raise ValueError("preflight requires absent gold and fresh runtime")
    runtime_root.mkdir(parents=True)
    spec = _models(root)["e5"]
    model_hashes = _verify_model(spec, cache)
    backend = e5.DirectE5(root, cache)
    rows = backend.embed(["query: synthetic query", "passage: synthetic passage"])
    if len(rows) != 2 or any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("E5 synthetic forward failed")
    payload = {
        "protocol_id": PROTOCOL_ID, "component": "e5", "status": "preflight_component_valid",
        "source_commit": source_commit, "registered_query_execution_count": 0,
        "gold_present": False, "registered_source_sha256": hashes,
        "model_files_sha256": model_hashes,
    }
    write_json_exclusive(runtime_root / "preflight.e5.json", payload)
    return payload


def preflight_ce(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _require_offline_linux()
    hashes = _validate_tree(root, "preflight")
    if (root / GOLD).exists() or not runtime_root.is_dir():
        raise ValueError("preflight requires absent gold and initialized runtime")
    parity_value = read_json(root / PARITY)
    if parity_value.get("status") != "result_free_parity_valid" or sha256_file(root / PARITY) != _manifest(root)["minilm_parity_gate"]["evidence_sha256"]:
        raise ValueError("MiniLM parity evidence mismatch")
    models = []
    for kind in MODEL_KINDS:
        spec = _models(root)[kind]
        model_hashes = _verify_model(spec, cache)
        runtime = v1._load_model(spec, cache)
        tokenizer, model, torch = runtime
        encoded = tokenizer(["synthetic query"], ["synthetic passage"], return_tensors="pt")
        with torch.inference_mode():
            value = float(model(**encoded).logits.reshape(-1)[0])
        if not math.isfinite(value):
            raise ValueError("cross-encoder synthetic forward failed")
        models.append({"kind": kind, "model_files_sha256": model_hashes})
        del runtime, tokenizer, model
        gc.collect()
    payload = {
        "protocol_id": PROTOCOL_ID, "component": "cross-encoders", "status": "preflight_component_valid",
        "source_commit": source_commit, "registered_query_execution_count": 0,
        "gold_present": False, "registered_source_sha256": hashes,
        "parity_evidence_sha256": sha256_file(root / PARITY), "models": models,
    }
    write_json_exclusive(runtime_root / "preflight.ce.json", payload)
    return payload


def preflight_bind(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    hashes = _validate_tree(root, "preflight")
    components = [read_json(runtime_root / name) for name in ("preflight.e5.json", "preflight.ce.json")]
    if any(row.get("source_commit") != source_commit or row.get("registered_source_sha256") != hashes for row in components):
        raise ValueError("component preflight binding mismatch")
    payload = {
        "protocol_id": PROTOCOL_ID, "status": "preflight_valid", "source_commit": source_commit,
        "registered_query_execution_count": 0, "gold_present": False,
        "holdout_bearing_input_file_count": 0, "shared_database_open_count": 0,
        "corpus_document_count": 93, "registered_source_sha256": hashes,
        "component_sha256": {
            "e5": sha256_file(runtime_root / "preflight.e5.json"),
            "cross_encoders": sha256_file(runtime_root / "preflight.ce.json"),
        },
    }
    write_json_exclusive(runtime_root / ATTESTATION, payload)
    return payload


def claim(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    hashes = _validate_tree(root, "claim")
    preflight = read_json(runtime_root / ATTESTATION)
    if preflight.get("source_commit") != source_commit or preflight.get("registered_source_sha256") != hashes:
        raise ValueError("preflight binding mismatch")
    if any((root / path).exists() for path in (CLAIM, RESULT, ERROR)):
        raise FileExistsError("append-only evidence exists")
    payload = {
        "protocol_id": PROTOCOL_ID, "source_commit": source_commit,
        "manifest_sha256": sha256_file(root / MANIFEST),
        "preflight_attestation_sha256": sha256_file(runtime_root / ATTESTATION),
        "registered_pipeline_count": 1, "retry_count": 0, "candidate_k": CANDIDATE_K,
        "stage_2_worker_count": 2, "gold_present_in_workers": False,
    }
    write_json_exclusive(root / CLAIM, payload)
    return payload


def stage1(root: Path, cache: Path, runtime_root: Path) -> dict[str, Any]:
    _validate_tree(root, "worker")
    if (root / GOLD).exists():
        raise ValueError("worker gold must be absent")
    query = _query(root)
    documents = v1._documents(root)
    model_hashes = _verify_model(_models(root)["e5"], cache)
    started = time.perf_counter()
    backend = e5.DirectE5(root, cache)
    query_vector = backend.embed(["query: " + query])[0]
    scored, truncation = e5._score_representation(documents, backend, query_vector, "structural")
    ranking = []
    for document in scored:
        ranking.append({
            "source_id": document["source_id"], "path": document["path"],
            "chunk_count": document["chunk_count"],
            "document_score": e5._cosine(query_vector, document["centroid_embedding"]),
        })
    ranking.sort(key=lambda row: (-row["document_score"], row["source_id"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    import resource
    payload = {
        "stage": 1, "kind": "e5-structural-centroid", "model_id": _models(root)["e5"]["model_id"],
        "revision": _models(root)["e5"]["revision"], "candidate_k": CANDIDATE_K,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "model_files_sha256": model_hashes,
        "query_embedding_sha256": sha256_bytes(canonical_json_bytes(query_vector)),
        "truncation": truncation, "documents": scored, "ranking": ranking,
        "ranking_sha256": sha256_bytes(canonical_json_bytes(ranking)),
        "candidate_source_ids": [row["source_id"] for row in ranking[:CANDIDATE_K]],
    }
    write_json_exclusive(runtime_root / "stage1.json", payload)
    return {"status": "stage1_complete", "candidate_count": CANDIDATE_K}


def stage2(root: Path, cache: Path, runtime_root: Path, kind: str) -> dict[str, Any]:
    _validate_tree(root, "worker")
    if (root / GOLD).exists():
        raise ValueError("worker gold must be absent")
    stage1_value = read_json(runtime_root / "stage1.json")
    candidate_ids = stage1_value.get("candidate_source_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != CANDIDATE_K or len(set(candidate_ids)) != CANDIDATE_K:
        raise ValueError("stage 1 candidate packet is incomplete")
    all_documents = {row["source_id"]: row for row in v1._documents(root)}
    if any(source_id not in all_documents for source_id in candidate_ids):
        raise ValueError("candidate source is absent from corpus")
    documents = [all_documents[source_id] for source_id in candidate_ids]
    query = _query(root)
    spec = _models(root)[kind]
    model_hashes = _verify_model(spec, cache)
    started = time.perf_counter()
    runtime = v1._load_model(spec, cache)
    try:
        scored, truncation = ce._score_representation(query, documents, runtime, "structural")
    finally:
        del runtime
        gc.collect()
    ranking = []
    for document in scored:
        logits = [float(chunk["raw_logit"]) for chunk in document["chunks"]]
        ranking.append({
            "source_id": document["source_id"], "path": document["path"],
            "chunk_count": document["chunk_count"], "document_score": ce._nlme(logits),
        })
    ranking.sort(key=lambda row: (-row["document_score"], row["source_id"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    import resource
    payload = {
        "stage": 2, "kind": kind, "model_id": spec["model_id"], "revision": spec["revision"],
        "candidate_k": CANDIDATE_K, "candidate_source_ids_sha256": sha256_bytes(canonical_json_bytes(candidate_ids)),
        "stage1_packet_sha256": sha256_file(runtime_root / "stage1.json"),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "model_files_sha256": model_hashes, "truncation": truncation,
        "documents": scored, "ranking": ranking,
        "ranking_sha256": sha256_bytes(canonical_json_bytes(ranking)),
    }
    write_json_exclusive(runtime_root / f"stage2.{kind}.json", payload)
    return {"status": "stage2_complete", "kind": kind, "candidate_count": len(ranking)}


def finalize(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _validate_tree(root, "finalizer")
    expected = _gold(root)
    stage1_value = read_json(runtime_root / "stage1.json")
    candidates = stage1_value["candidate_source_ids"]
    if expected not in candidates:
        raise ValueError("expected source missing from stage 1 candidates")
    stage1_rank = next(row["rank"] for row in stage1_value["ranking"] if row["source_id"] == expected)
    models = []
    for kind in MODEL_KINDS:
        worker = read_json(runtime_root / f"stage2.{kind}.json")
        if worker.get("stage1_packet_sha256") != sha256_file(runtime_root / "stage1.json"):
            raise ValueError("stage 2 packet binding mismatch")
        match = next(row for row in worker["ranking"] if row["source_id"] == expected)
        worker["expected_source_rank"] = match["rank"]
        worker["expected_source_within_cutoff"] = match["rank"] <= CUTOFF
        worker["top_distractor"] = next(row for row in worker["ranking"] if row["source_id"] != expected)
        worker["pipeline_runtime_seconds"] = stage1_value["runtime_seconds"] + worker["runtime_seconds"]
        worker["practical_target_met"] = worker["pipeline_runtime_seconds"] <= PRACTICAL_SECONDS
        models.append(worker)
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "source_commit": source_commit,
        "expected_source_id": expected, "corpus_document_count": 93, "candidate_k": CANDIDATE_K,
        "stage1_expected_source_rank": stage1_rank, "stage1_candidate_inclusion": True,
        "cutoff": CUTOFF, "primary_success": all(row["expected_source_within_cutoff"] for row in models),
        "practical_success": all(row["practical_target_met"] for row in models),
        "practical_target_seconds_per_pipeline": PRACTICAL_SECONDS,
        "runtime_measurement_boundary": _manifest(root)["runtime"]["measurement_boundary"],
        "environment": {"os": "linux", "architecture": platform.machine(), "network": "disabled", "fresh_runtime": True, "holdout_bearing_input_file_count": 0, "shared_database_open_count": 0},
        "inputs_sha256": {path.as_posix(): sha256_file(root / path) for path in (MANIFEST, SCHEMA, MODELS, QUERY, GOLD, CORPUS, PARITY)},
        "stage1": stage1_value, "stage2_models": models,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / RESULT, payload)
    return {
        "status": "observed_valid", "stage1_rank": stage1_rank,
        "ranks": {row["kind"]: row["expected_source_rank"] for row in models},
        "pipeline_runtime_seconds": {row["kind"]: row["pipeline_runtime_seconds"] for row in models},
        "primary_success": payload["primary_success"], "practical_success": payload["practical_success"],
    }


def record_error(root: Path, source_commit: str, message: str) -> dict[str, Any]:
    if not (root / CLAIM).exists() or (root / RESULT).exists():
        raise FileExistsError("error evidence requires claim and no result")
    payload = {"protocol_id": PROTOCOL_ID, "source_commit": source_commit, "retry_count": 0, "error": message}
    write_json_exclusive(root / ERROR, payload)
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    evidence = {"claim": (root / CLAIM).exists(), "result": (root / RESULT).exists(), "error": (root / ERROR).exists()}
    if evidence["result"] and evidence["error"]:
        raise ValueError("result and error coexist")
    result = read_json(root / RESULT) if evidence["result"] else None
    if result and result.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in result.items() if k != "payload_sha256"})):
        raise ValueError("result payload hash mismatch")
    return {
        "protocol_id": PROTOCOL_ID, "status": "observed_valid" if result else "result_free_frozen",
        "registered_pipeline_count": 1 if result else 0, "holdout_bearing_input_file_count": 0,
        "evidence": evidence, "primary_success": result.get("primary_success") if result else None,
        "practical_success": result.get("practical_success") if result else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "probe"):
        sub.add_parser(name).add_argument("--root", type=Path, default=ROOT)
    p = sub.add_parser("parity"); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    for name in ("preflight-e5", "preflight-ce"):
        p = sub.add_parser(name); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("preflight-bind"); p.add_argument("--root", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("claim"); p.add_argument("--root", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("stage1"); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True)
    p = sub.add_parser("stage2"); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--kind", choices=MODEL_KINDS, required=True)
    p = sub.add_parser("finalize"); p.add_argument("--root", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("record-error"); p.add_argument("--root", type=Path, required=True); p.add_argument("--source-commit", required=True); p.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    if args.command in {"audit", "probe"}: result = audit(args.root)
    elif args.command == "parity": result = parity(args.root, args.cache, args.output)
    elif args.command == "preflight-e5": result = preflight_e5(args.root, args.cache, args.runtime_root, args.source_commit)
    elif args.command == "preflight-ce": result = preflight_ce(args.root, args.cache, args.runtime_root, args.source_commit)
    elif args.command == "preflight-bind": result = preflight_bind(args.root, args.runtime_root, args.source_commit)
    elif args.command == "claim": result = claim(args.root, args.runtime_root, args.source_commit)
    elif args.command == "stage1": result = stage1(args.root, args.cache, args.runtime_root)
    elif args.command == "stage2": result = stage2(args.root, args.cache, args.runtime_root, args.kind)
    elif args.command == "finalize": result = finalize(args.root, args.runtime_root, args.source_commit)
    else: result = record_error(args.root, args.source_commit, args.message)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

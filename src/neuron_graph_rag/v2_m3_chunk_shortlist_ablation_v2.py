"""Holdout-absent CPU v2-m3 chunk-shortlist ablation for the v5 hard query."""
from __future__ import annotations

import argparse
import gc
import json
import math
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

PROTOCOL_ID = "github-retrieval-parity-v5-v2-m3-chunk-shortlist-ablation-v2"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/v2_m3_chunk_shortlist_ablation_v2.manifest.json")
SCHEMA = Path("tests/fixtures/v2_m3_chunk_shortlist_ablation_v2.schema.json")
MODELS = Path("tests/fixtures/v2_m3_chunk_shortlist_ablation_v2.models.json")
QUERY = Path("tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
EVIDENCE = Path("tests/evidence/v2_m3_chunk_shortlist_ablation_v2")
PREFLIGHT = EVIDENCE / "development.preflight.json"
CLAIM = EVIDENCE / "development.claim.json"
STAGE1_EVIDENCE = EVIDENCE / "development.stage1.worker.json"
STAGE2_EVIDENCE = EVIDENCE / "development.stage2.worker.json"
RESULT = EVIDENCE / "development.observed.json"
ERROR = EVIDENCE / "development.error.json"
ATTESTATION = Path("preflight.attestation.json")
STAGE1_PACKET = Path("stage1.worker.json")
STAGE2_PACKET = Path("stage2.worker.json")
CANDIDATE_K = 50
SHORTLIST_ARMS = (2, 4, 8)
MAX_SHORTLIST = 8
CUTOFF = 20
PRACTICAL_SECONDS = 600.0
BATCH_SIZE = 8

canonical_json_bytes = v1.canonical_json_bytes
sha256_bytes = v1.sha256_bytes
sha256_file = v1.sha256_file
read_json = v1.read_json
write_json_exclusive = v1.write_json_exclusive


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "frozen_pre_registered_execution":
        raise ValueError("frozen manifest identity mismatch")
    if value["stage_1"]["candidate_k"] != CANDIDATE_K:
        raise ValueError("candidate contract mismatch")
    if tuple(value["shortlist"]["arms"]) != SHORTLIST_ARMS or value["shortlist"]["max_chunks"] != MAX_SHORTLIST:
        raise ValueError("shortlist contract mismatch")
    if value["ranking"]["cutoff"] != CUTOFF or value["runtime"]["practical_target_seconds"] != PRACTICAL_SECONDS:
        raise ValueError("success contract mismatch")
    for relative, digest in value.get("artifact_sha256", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256_file(root / relative) != digest:
            raise ValueError(f"frozen artifact changed: {relative}")
    return value


def _models(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    rows = read_json(root / MODELS).get("models")
    if not isinstance(rows, list):
        raise ValueError("model registry missing")
    result = {str(row["kind"]): row for row in rows}
    if tuple(result) != ("e5", "v2-m3"):
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
    forbidden_paths = [
        *manifest["forbidden_registered_paths"],
        *manifest["stage_forbidden_registered_paths"][stage],
    ]
    for forbidden in forbidden_paths:
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
        "protocol_id": PROTOCOL_ID,
        "component": "e5",
        "status": "preflight_component_valid",
        "source_commit": source_commit,
        "registered_query_execution_count": 0,
        "gold_present": False,
        "registered_source_sha256": hashes,
        "model_files_sha256": model_hashes,
    }
    write_json_exclusive(runtime_root / "preflight.e5.json", payload)
    return payload


def preflight_v2_m3(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _require_offline_linux()
    hashes = _validate_tree(root, "preflight")
    if (root / GOLD).exists() or not runtime_root.is_dir():
        raise ValueError("preflight requires absent gold and initialized runtime")
    spec = _models(root)["v2-m3"]
    model_hashes = _verify_model(spec, cache)
    runtime = v1._load_model(spec, cache)
    tokenizer, model, torch = runtime
    encoded = tokenizer(["synthetic query"], ["synthetic passage"], return_tensors="pt")
    with torch.inference_mode():
        value = float(model(**encoded).logits.reshape(-1)[0])
    if not math.isfinite(value):
        raise ValueError("v2-m3 synthetic forward failed")
    del runtime, tokenizer, model
    gc.collect()
    payload = {
        "protocol_id": PROTOCOL_ID,
        "component": "v2-m3",
        "status": "preflight_component_valid",
        "source_commit": source_commit,
        "registered_query_execution_count": 0,
        "gold_present": False,
        "registered_source_sha256": hashes,
        "model_files_sha256": model_hashes,
    }
    write_json_exclusive(runtime_root / "preflight.v2-m3.json", payload)
    return payload


def preflight_bind(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    hashes = _validate_tree(root, "preflight")
    components = [read_json(runtime_root / name) for name in ("preflight.e5.json", "preflight.v2-m3.json")]
    if any(row.get("source_commit") != source_commit or row.get("registered_source_sha256") != hashes for row in components):
        raise ValueError("component preflight binding mismatch")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "preflight_valid",
        "source_commit": source_commit,
        "registered_query_execution_count": 0,
        "gold_present": False,
        "holdout_bearing_input_file_count": 0,
        "mixed_v5_query_or_gold_file_count": 0,
        "shared_database_open_count": 0,
        "corpus_document_count": 93,
        "registered_source_sha256": hashes,
        "component_sha256": {
            "e5": sha256_file(runtime_root / "preflight.e5.json"),
            "v2-m3": sha256_file(runtime_root / "preflight.v2-m3.json"),
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
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "manifest_sha256": sha256_file(root / MANIFEST),
        "preflight_attestation_sha256": sha256_file(runtime_root / ATTESTATION),
        "registered_pipeline_count": 1,
        "retry_count": 0,
        "candidate_k": CANDIDATE_K,
        "shortlist_arms": list(SHORTLIST_ARMS),
        "stage_2_model_forward_pass_count": 1,
        "gold_present_in_workers": False,
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
            "source_id": document["source_id"],
            "path": document["path"],
            "chunk_count": document["chunk_count"],
            "document_score": e5._cosine(query_vector, document["centroid_embedding"]),
        })
    ranking.sort(key=lambda row: (-row["document_score"], row["source_id"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    import resource
    payload = {
        "stage": 1,
        "kind": "e5-structural-centroid",
        "model_id": _models(root)["e5"]["model_id"],
        "revision": _models(root)["e5"]["revision"],
        "candidate_k": CANDIDATE_K,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "model_files_sha256": model_hashes,
        "query_embedding_sha256": sha256_bytes(canonical_json_bytes(query_vector)),
        "truncation": truncation,
        "documents": scored,
        "ranking": ranking,
        "ranking_sha256": sha256_bytes(canonical_json_bytes(ranking)),
        "candidate_source_ids": [row["source_id"] for row in ranking[:CANDIDATE_K]],
    }
    write_json_exclusive(runtime_root / STAGE1_PACKET, payload)
    return {"status": "stage1_complete", "candidate_count": CANDIDATE_K}


def _shortlist_arms(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    arms = []
    for size in SHORTLIST_ARMS:
        ranking = []
        for document in documents:
            selected = list(document["chunks"])[:size]
            logits = [float(chunk["raw_logit"]) for chunk in selected]
            ranking.append({
                "source_id": document["source_id"],
                "path": document["path"],
                "available_chunk_count": document["all_chunk_count"],
                "selected_chunk_count": len(selected),
                "document_score": ce._nlme(logits),
            })
        ranking.sort(key=lambda row: (-row["document_score"], row["source_id"]))
        for rank, row in enumerate(ranking, 1):
            row["rank"] = rank
        arms.append({
            "m": size,
            "aggregation": "normalized_log_mean_exp",
            "temperature": 1.0,
            "ranking": ranking,
            "ranking_sha256": sha256_bytes(canonical_json_bytes(ranking)),
        })
    return arms


def _ordered_shortlist(chunks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    indices = [int(row["chunk_index"]) for row in chunks]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate stage 1 chunk index")
    return sorted(chunks, key=lambda row: (-float(row["query_cosine"]), int(row["chunk_index"])))


def stage2(root: Path, cache: Path, runtime_root: Path) -> dict[str, Any]:
    _validate_tree(root, "worker")
    if (root / GOLD).exists():
        raise ValueError("worker gold must be absent")
    stage1_value = read_json(runtime_root / STAGE1_PACKET)
    candidate_ids = stage1_value.get("candidate_source_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != CANDIDATE_K or len(set(candidate_ids)) != CANDIDATE_K:
        raise ValueError("stage 1 candidate packet is incomplete")
    stage1_documents = {row["source_id"]: row for row in stage1_value["documents"]}
    corpus_documents = {row["source_id"]: row for row in v1._documents(root)}
    if any(source_id not in stage1_documents or source_id not in corpus_documents for source_id in candidate_ids):
        raise ValueError("candidate source is absent from stage 1 or corpus")
    query = _query(root)
    spec = _models(root)["v2-m3"]
    model_hashes = _verify_model(spec, cache)
    started = time.perf_counter()
    runtime = v1._load_model(spec, cache)
    tokenizer, model, torch = runtime
    output_documents: list[dict[str, Any]] = []
    passages: list[str] = []
    metadata: list[dict[str, Any]] = []
    try:
        for source_id in candidate_ids:
            corpus = corpus_documents[source_id]
            stage1_document = stage1_documents[source_id]
            body = str(corpus["content"])
            body_chunks = v1.project_passages(body, window=480, overlap=80)
            stage1_chunks = list(stage1_document["chunks"])
            if len(body_chunks) != stage1_document["chunk_count"] or len(body_chunks) != len(stage1_chunks):
                raise ValueError("stage 1 chunk count drift")
            for body_chunk, stage1_chunk in zip(body_chunks, stage1_chunks, strict=True):
                if (
                    int(stage1_chunk["chunk_index"]) != int(body_chunk["chunk_index"])
                    or stage1_chunk["body_chunk_sha256"] != sha256_bytes(str(body_chunk["text"]).encode("utf-8"))
                ):
                    raise ValueError("stage 1 chunk identity drift")
            ordered = _ordered_shortlist(stage1_chunks)
            selected = ordered[:MAX_SHORTLIST]
            document_chunks = []
            for shortlist_rank, stage1_chunk in enumerate(selected, 1):
                chunk_index = int(stage1_chunk["chunk_index"])
                body_chunk = body_chunks[chunk_index]
                prefix = structural.structural_prefix(str(corpus["path"]), body, int(body_chunk["start_codepoint"]))
                passage = prefix + str(body_chunk["text"])
                pair_length = len(tokenizer.encode(query, passage, add_special_tokens=True, truncation=False))
                row = {
                    "shortlist_rank": shortlist_rank,
                    "chunk_index": chunk_index,
                    "e5_query_cosine": float(stage1_chunk["query_cosine"]),
                    "start_codepoint": body_chunk["start_codepoint"],
                    "end_codepoint": body_chunk["end_codepoint"],
                    "body_chunk_sha256": stage1_chunk["body_chunk_sha256"],
                    "prefix_codepoint_length": len(prefix),
                    "pair_token_length_before_truncation": pair_length,
                    "pair_exceeds_512_before_truncation": pair_length > 512,
                }
                passages.append(passage)
                metadata.append(row)
                document_chunks.append(row)
            output_documents.append({
                "source_id": source_id,
                "path": corpus["path"],
                "all_chunk_count": len(body_chunks),
                "selected_chunk_count": len(selected),
                "chunks": document_chunks,
            })
        if len(passages) > CANDIDATE_K * MAX_SHORTLIST:
            raise ValueError("shortlist pair budget exceeded")
        scores: list[float] = []
        for offset in range(0, len(passages), BATCH_SIZE):
            batch = passages[offset:offset + BATCH_SIZE]
            encoded = tokenizer(
                [query] * len(batch), batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).to(dtype=torch.float32).tolist()
            scores.extend(float(value) for value in logits)
        if len(scores) != len(metadata) or any(not math.isfinite(value) for value in scores):
            raise ValueError("v2-m3 shortlist logits are incomplete")
        for row, score in zip(metadata, scores, strict=True):
            row["raw_logit"] = score
        arms = _shortlist_arms(output_documents)
    finally:
        del runtime, tokenizer, model
        gc.collect()
    import resource
    payload = {
        "stage": 2,
        "kind": "v2-m3-chunk-shortlist",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "candidate_k": CANDIDATE_K,
        "shortlist_arms": list(SHORTLIST_ARMS),
        "max_shortlist": MAX_SHORTLIST,
        "candidate_source_ids_sha256": sha256_bytes(canonical_json_bytes(candidate_ids)),
        "stage1_packet_sha256": sha256_file(runtime_root / STAGE1_PACKET),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "model_files_sha256": model_hashes,
        "selected_pair_count": len(passages),
        "forward_batch_count": math.ceil(len(passages) / BATCH_SIZE),
        "batch_size": BATCH_SIZE,
        "documents": output_documents,
        "arms": arms,
    }
    write_json_exclusive(runtime_root / STAGE2_PACKET, payload)
    return {"status": "stage2_complete", "candidate_count": CANDIDATE_K, "selected_pair_count": len(passages)}


def finalize(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _validate_tree(root, "finalizer")
    expected = _gold(root)
    stage1_value = read_json(runtime_root / STAGE1_PACKET)
    stage2_value = read_json(runtime_root / STAGE2_PACKET)
    candidates = stage1_value["candidate_source_ids"]
    if expected not in candidates:
        raise ValueError("expected source missing from stage 1 candidates")
    if stage2_value.get("stage1_packet_sha256") != sha256_file(runtime_root / STAGE1_PACKET):
        raise ValueError("stage 2 packet binding mismatch")
    if tuple(stage2_value.get("shortlist_arms", ())) != SHORTLIST_ARMS:
        raise ValueError("stage 2 arm contract mismatch")
    if stage2_value.get("selected_pair_count", 0) > CANDIDATE_K * MAX_SHORTLIST:
        raise ValueError("stage 2 pair budget exceeded")
    stage1_rank = next(row["rank"] for row in stage1_value["ranking"] if row["source_id"] == expected)
    observed_arms = []
    for arm in stage2_value["arms"]:
        match = next(row for row in arm["ranking"] if row["source_id"] == expected)
        observed_arms.append({
            "m": arm["m"],
            "expected_source_rank": match["rank"],
            "expected_source_within_cutoff": match["rank"] <= CUTOFF,
            "ranking_sha256": arm["ranking_sha256"],
            "top_distractor": next(row for row in arm["ranking"] if row["source_id"] != expected),
        })
    pipeline_runtime = float(stage1_value["runtime_seconds"]) + float(stage2_value["runtime_seconds"])
    quality_success = any(row["expected_source_within_cutoff"] for row in observed_arms)
    practical_success = pipeline_runtime <= PRACTICAL_SECONDS
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "expected_source_id": expected,
        "corpus_document_count": 93,
        "candidate_k": CANDIDATE_K,
        "stage1_expected_source_rank": stage1_rank,
        "stage1_candidate_inclusion": True,
        "shortlist_arms": list(SHORTLIST_ARMS),
        "cutoff": CUTOFF,
        "quality_primary_success": quality_success,
        "practical_success": practical_success,
        "combined_success": quality_success and practical_success,
        "practical_target_seconds": PRACTICAL_SECONDS,
        "stage1_runtime_seconds": stage1_value["runtime_seconds"],
        "stage2_runtime_seconds": stage2_value["runtime_seconds"],
        "pipeline_runtime_seconds": pipeline_runtime,
        "stage1_peak_rss_bytes": stage1_value["peak_rss_bytes"],
        "stage2_peak_rss_bytes": stage2_value["peak_rss_bytes"],
        "selected_pair_count": stage2_value["selected_pair_count"],
        "forward_batch_count": stage2_value["forward_batch_count"],
        "arms": observed_arms,
        "full_chunk_baseline": {"source": "#242", "expected_source_rank": 15, "pipeline_runtime_seconds": 717.949333017008},
        "runtime_measurement_boundary": _manifest(root)["runtime"]["measurement_boundary"],
        "environment": {
            "os": "linux",
            "architecture": platform.machine(),
            "network": "disabled",
            "fresh_runtime": True,
            "holdout_bearing_input_file_count": 0,
            "mixed_v5_query_or_gold_file_count": 0,
            "shared_database_open_count": 0,
        },
        "inputs_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in (MANIFEST, SCHEMA, MODELS, QUERY, GOLD, CORPUS)
        },
        "worker_packets_sha256": {
            STAGE1_EVIDENCE.as_posix(): sha256_file(runtime_root / STAGE1_PACKET),
            STAGE2_EVIDENCE.as_posix(): sha256_file(runtime_root / STAGE2_PACKET),
        },
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / RESULT, payload)
    return {
        "status": "observed_valid",
        "stage1_rank": stage1_rank,
        "ranks": {str(row["m"]): row["expected_source_rank"] for row in observed_arms},
        "pipeline_runtime_seconds": pipeline_runtime,
        "quality_primary_success": quality_success,
        "practical_success": practical_success,
        "combined_success": payload["combined_success"],
    }


def record_error(root: Path, source_commit: str, message: str) -> dict[str, Any]:
    if not (root / CLAIM).exists() or (root / RESULT).exists():
        raise FileExistsError("error evidence requires claim and no result")
    payload = {"protocol_id": PROTOCOL_ID, "source_commit": source_commit, "retry_count": 0, "error": message}
    write_json_exclusive(root / ERROR, payload)
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    evidence = {
        "preflight": (root / PREFLIGHT).exists(),
        "claim": (root / CLAIM).exists(),
        "stage1": (root / STAGE1_EVIDENCE).exists(),
        "stage2": (root / STAGE2_EVIDENCE).exists(),
        "result": (root / RESULT).exists(),
        "error": (root / ERROR).exists(),
    }
    if evidence["result"] and evidence["error"]:
        raise ValueError("result and error coexist")
    result = read_json(root / RESULT) if evidence["result"] else None
    if result:
        if result.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in result.items() if k != "payload_sha256"})):
            raise ValueError("result payload hash mismatch")
        for relative, digest in result["worker_packets_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError(f"worker packet hash mismatch: {relative}")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "observed_valid" if result else "result_free_frozen",
        "registered_pipeline_count": 1 if result else 0,
        "holdout_bearing_input_file_count": 0,
        "evidence": evidence,
        "quality_primary_success": result.get("quality_primary_success") if result else None,
        "practical_success": result.get("practical_success") if result else None,
        "combined_success": result.get("combined_success") if result else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "probe"):
        sub.add_parser(name).add_argument("--root", type=Path, default=ROOT)
    for name in ("preflight-e5", "preflight-v2-m3"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--runtime-root", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
    for name in ("preflight-bind", "claim", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--runtime-root", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
    for name in ("stage1", "stage2"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--runtime-root", type=Path, required=True)
    command = sub.add_parser("record-error")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    command.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    if args.command in {"audit", "probe"}:
        result = audit(args.root)
    elif args.command == "preflight-e5":
        result = preflight_e5(args.root, args.cache, args.runtime_root, args.source_commit)
    elif args.command == "preflight-v2-m3":
        result = preflight_v2_m3(args.root, args.cache, args.runtime_root, args.source_commit)
    elif args.command == "preflight-bind":
        result = preflight_bind(args.root, args.runtime_root, args.source_commit)
    elif args.command == "claim":
        result = claim(args.root, args.runtime_root, args.source_commit)
    elif args.command == "stage1":
        result = stage1(args.root, args.cache, args.runtime_root)
    elif args.command == "stage2":
        result = stage2(args.root, args.cache, args.runtime_root)
    elif args.command == "finalize":
        result = finalize(args.root, args.runtime_root, args.source_commit)
    else:
        result = record_error(args.root, args.source_commit, args.message)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

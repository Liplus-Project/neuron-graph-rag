"""Finalizer-only recovery for the completed #242 gold-blind worker packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_ID = "github-retrieval-parity-v5-practical-two-stage-finalizer-recovery-v2"
SOURCE_PROTOCOL_ID = "github-retrieval-parity-v5-practical-two-stage-retrieval-v1"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/practical_two_stage_retrieval_recovery_v2.manifest.json")
SCHEMA = Path("tests/fixtures/practical_two_stage_retrieval_recovery_v2.schema.json")
SOURCE_MANIFEST = Path("tests/fixtures/practical_two_stage_retrieval_v1.manifest.json")
SOURCE_SCHEMA = Path("tests/fixtures/practical_two_stage_retrieval_v1.schema.json")
SOURCE_MODELS = Path("tests/fixtures/practical_two_stage_retrieval_v1.models.json")
SOURCE_RUNNER = Path("src/neuron_graph_rag/practical_two_stage_retrieval.py")
SOURCE_PARITY = Path("tests/evidence/practical_two_stage_retrieval_v1/result_free_minilm_parity.json")
SOURCE_EVIDENCE = Path("tests/evidence/practical_two_stage_retrieval_v1")
PREFLIGHT = SOURCE_EVIDENCE / "development.preflight.json"
SOURCE_CLAIM = SOURCE_EVIDENCE / "development.claim.json"
STAGE1 = SOURCE_EVIDENCE / "development.stage1.worker.json"
MINILM = SOURCE_EVIDENCE / "development.stage2.minilm.worker.json"
V2_M3 = SOURCE_EVIDENCE / "development.stage2.v2-m3.worker.json"
SOURCE_ERROR = SOURCE_EVIDENCE / "development.error.json"
RECOVERY_V1_EVIDENCE = Path("tests/evidence/practical_two_stage_retrieval_finalizer_recovery_v1")
RECOVERY_V1_CLAIM = RECOVERY_V1_EVIDENCE / "development.claim.json"
RECOVERY_V1_RESULT = RECOVERY_V1_EVIDENCE / "development.recovered.json"
RECOVERY_V1_ERROR = RECOVERY_V1_EVIDENCE / "development.error.json"
RECOVERY_EVIDENCE = Path("tests/evidence/practical_two_stage_retrieval_finalizer_recovery_v2")
CLAIM = RECOVERY_EVIDENCE / "development.claim.json"
RESULT = RECOVERY_EVIDENCE / "development.recovered.json"
ERROR = RECOVERY_EVIDENCE / "development.error.json"
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
MODEL_KINDS = ("minilm", "v2-m3")
DOCUMENT_COUNT = 93
CANDIDATE_K = 50
CUTOFF = 20
PRACTICAL_SECONDS = 600.0


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "frozen_before_gold_mount":
        raise ValueError("recovery manifest identity mismatch")
    if value.get("registered_query_execution_count") != 0 or value.get("model_forward_inference_count") != 0:
        raise ValueError("recovery inference count mismatch")
    for relative, digest in value.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen recovery artifact changed: {relative}")
    return value


def _registered_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}


def _validate_tree(root: Path, stage: str) -> dict[str, str]:
    manifest = _manifest(root)
    expected = set(manifest["claim_registered_files"])
    if stage == "finalizer":
        expected.update((CLAIM.as_posix(), GOLD.as_posix()))
    actual = _registered_files(root)
    if actual != expected:
        raise ValueError(f"recovery allowlist mismatch: missing={sorted(expected-actual)!r} extra={sorted(actual-expected)!r}")
    for forbidden in manifest["forbidden_registered_paths"]:
        if (root / forbidden).exists():
            raise ValueError(f"forbidden recovery input exists: {forbidden}")
    return {relative: sha256_file(root / relative) for relative in sorted(expected)}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite numeric value: {label}")
    return float(value)


def _verify_source_evidence(root: Path) -> dict[str, Any]:
    manifest = _manifest(root)
    expected = manifest["source_evidence_sha256"]
    actual = {relative: sha256_file(root / relative) for relative in expected}
    if actual != expected:
        raise ValueError("immutable source evidence hash mismatch")
    preflight = read_json(root / PREFLIGHT)
    source_claim = read_json(root / SOURCE_CLAIM)
    source_error = read_json(root / SOURCE_ERROR)
    source_manifest = read_json(root / SOURCE_MANIFEST)
    if preflight.get("protocol_id") != SOURCE_PROTOCOL_ID or preflight.get("status") != "preflight_valid":
        raise ValueError("source preflight mismatch")
    if preflight.get("registered_query_execution_count") != 0 or preflight.get("gold_present") is not False:
        raise ValueError("source preflight boundary mismatch")
    if preflight.get("holdout_bearing_input_file_count") != 0 or preflight.get("shared_database_open_count") != 0:
        raise ValueError("source preflight isolation mismatch")
    if source_claim.get("protocol_id") != SOURCE_PROTOCOL_ID or source_claim.get("registered_pipeline_count") != 1:
        raise ValueError("source claim mismatch")
    if source_claim.get("retry_count") != 0 or source_claim.get("gold_present_in_workers") is not False:
        raise ValueError("source claim boundary mismatch")
    if source_error.get("protocol_id") != SOURCE_PROTOCOL_ID or source_error.get("retry_count") != 0:
        raise ValueError("source error mismatch")
    source_commit = manifest["source_commit"]
    if {preflight.get("source_commit"), source_claim.get("source_commit"), source_error.get("source_commit")} != {source_commit}:
        raise ValueError("source commit mismatch")
    if source_claim.get("preflight_attestation_sha256") != expected[PREFLIGHT.as_posix()]:
        raise ValueError("source claim/preflight binding mismatch")
    if source_claim.get("manifest_sha256") != sha256_file(root / SOURCE_MANIFEST):
        raise ValueError("source claim/manifest binding mismatch")
    error_text = str(source_error.get("error"))
    if "command failed" not in error_text or " finalize " not in error_text:
        raise ValueError("source failure cause mismatch")
    if (root / SOURCE_EVIDENCE / "development.observed.json").exists():
        raise ValueError("failed source protocol must not have result evidence")
    recovery_v1_claim = read_json(root / RECOVERY_V1_CLAIM)
    recovery_v1_error = read_json(root / RECOVERY_V1_ERROR)
    if recovery_v1_claim.get("protocol_id") != "github-retrieval-parity-v5-practical-two-stage-finalizer-recovery-v1":
        raise ValueError("recovery-v1 claim identity mismatch")
    if recovery_v1_claim.get("status") != "worker_packets_complete_gold_absent":
        raise ValueError("recovery-v1 claim status mismatch")
    if recovery_v1_claim.get("registered_query_execution_count") != 0 or recovery_v1_claim.get("model_forward_inference_count") != 0:
        raise ValueError("recovery-v1 claim inference boundary mismatch")
    if recovery_v1_error.get("protocol_id") != recovery_v1_claim["protocol_id"] or recovery_v1_error.get("retry_count") != 0:
        raise ValueError("recovery-v1 error mismatch")
    if (root / RECOVERY_V1_RESULT).exists():
        raise ValueError("failed recovery-v1 must not have result evidence")
    source_artifacts = source_manifest["artifact_sha256"]
    for relative in (SOURCE_RUNNER, SOURCE_SCHEMA, SOURCE_MODELS, SOURCE_PARITY):
        if source_artifacts.get(relative.as_posix()) != sha256_file(root / relative):
            raise ValueError(f"source artifact binding mismatch: {relative}")
    if preflight["registered_source_sha256"].get(SOURCE_RUNNER.as_posix()) != sha256_file(root / SOURCE_RUNNER):
        raise ValueError("preflight/source runner binding mismatch")
    return {"preflight": preflight, "claim": source_claim, "error": source_error, "manifest": source_manifest}


def _sorted_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = [dict(row) for row in rows]
    for row in expected:
        row.pop("rank", None)
    expected.sort(key=lambda row: (-float(row["document_score"]), str(row["source_id"])))
    for rank, row in enumerate(expected, 1):
        row["rank"] = rank
    return expected


def _walk_for_gold(value: Any) -> None:
    forbidden = {"expected_source_id", "expected_source_rank", "expected_source_within_cutoff", "top_distractor", "gold"}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("gold-derived field exists in worker packet")
        for child in value.values():
            _walk_for_gold(child)
    elif isinstance(value, list):
        for child in value:
            _walk_for_gold(child)


def _verify_stage1(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = read_json(root / STAGE1)
    required = {"stage", "kind", "model_id", "revision", "candidate_k", "runtime_seconds", "peak_rss_bytes", "model_files_sha256", "query_embedding_sha256", "truncation", "documents", "ranking", "ranking_sha256", "candidate_source_ids"}
    if set(packet) != required or packet["stage"] != 1 or packet["kind"] != "e5-structural-centroid" or packet["candidate_k"] != CANDIDATE_K:
        raise ValueError("stage 1 packet fields mismatch")
    _walk_for_gold(packet)
    if _finite(packet["runtime_seconds"], "stage1 runtime") <= 0 or not isinstance(packet["peak_rss_bytes"], int) or packet["peak_rss_bytes"] <= 0:
        raise ValueError("stage 1 runtime diagnostics mismatch")
    documents = packet["documents"]
    ranking = packet["ranking"]
    if not isinstance(documents, list) or len(documents) != DOCUMENT_COUNT or not isinstance(ranking, list) or len(ranking) != DOCUMENT_COUNT:
        raise ValueError("stage 1 document count mismatch")
    document_ids = [row.get("source_id") for row in documents]
    ranking_ids = [row.get("source_id") for row in ranking]
    if len(set(document_ids)) != DOCUMENT_COUNT or set(document_ids) != set(ranking_ids):
        raise ValueError("stage 1 source set mismatch")
    for document in documents:
        chunks = document.get("chunks")
        vector = document.get("centroid_embedding")
        if not isinstance(chunks, list) or not chunks or document.get("chunk_count") != len(chunks):
            raise ValueError("stage 1 chunk completeness mismatch")
        if not isinstance(vector, list) or not vector or not math.isclose(math.sqrt(sum(_finite(x, "centroid") ** 2 for x in vector)), 1.0, abs_tol=2e-6):
            raise ValueError("stage 1 centroid mismatch")
        for index, chunk in enumerate(chunks):
            if chunk.get("chunk_index") != index or len(str(chunk.get("body_chunk_sha256"))) != 64:
                raise ValueError("stage 1 chunk identity mismatch")
            _finite(chunk.get("query_cosine"), "stage1 query cosine")
    if ranking != _sorted_ranking(ranking) or packet["ranking_sha256"] != sha256_bytes(canonical_json_bytes(ranking)):
        raise ValueError("stage 1 ranking mismatch")
    candidates = packet["candidate_source_ids"]
    if candidates != ranking_ids[:CANDIDATE_K] or len(set(candidates)) != CANDIDATE_K:
        raise ValueError("stage 1 candidate selection mismatch")
    model_spec = read_json(root / SOURCE_MODELS)["models"][0]
    if packet["model_id"] != model_spec["model_id"] or packet["revision"] != model_spec["revision"]:
        raise ValueError("stage 1 model identity mismatch")
    expected_hashes = {row["path"]: row["sha256"] for row in model_spec["required_files"]}
    if packet["model_files_sha256"] != expected_hashes:
        raise ValueError("stage 1 model hashes mismatch")
    return packet, {"runtime_seconds": packet["runtime_seconds"], "peak_rss_bytes": packet["peak_rss_bytes"], "candidate_source_ids_sha256": sha256_bytes(canonical_json_bytes(candidates)), "ranking_sha256": packet["ranking_sha256"]}


def _nlme(scores: Sequence[float]) -> float:
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(value - maximum) for value in scores) / len(scores))


def _verify_stage2(root: Path, path: Path, kind: str, stage1: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = read_json(root / path)
    required = {"stage", "kind", "model_id", "revision", "candidate_k", "candidate_source_ids_sha256", "stage1_packet_sha256", "runtime_seconds", "peak_rss_bytes", "model_files_sha256", "truncation", "documents", "ranking", "ranking_sha256"}
    if set(packet) != required or packet["stage"] != 2 or packet["kind"] != kind or packet["candidate_k"] != CANDIDATE_K:
        raise ValueError(f"stage 2 packet fields mismatch: {kind}")
    _walk_for_gold(packet)
    candidates = stage1["candidate_source_ids"]
    if packet["candidate_source_ids_sha256"] != sha256_bytes(canonical_json_bytes(candidates)) or packet["stage1_packet_sha256"] != sha256_file(root / STAGE1):
        raise ValueError(f"stage 2 candidate binding mismatch: {kind}")
    if _finite(packet["runtime_seconds"], f"{kind} runtime") <= 0 or not isinstance(packet["peak_rss_bytes"], int) or packet["peak_rss_bytes"] <= 0:
        raise ValueError(f"stage 2 runtime diagnostics mismatch: {kind}")
    documents = packet["documents"]
    ranking = packet["ranking"]
    if not isinstance(documents, list) or len(documents) != CANDIDATE_K or not isinstance(ranking, list) or len(ranking) != CANDIDATE_K:
        raise ValueError(f"stage 2 candidate count mismatch: {kind}")
    if {row.get("source_id") for row in documents} != set(candidates) or {row.get("source_id") for row in ranking} != set(candidates):
        raise ValueError(f"stage 2 source set mismatch: {kind}")
    by_id = {}
    for document in documents:
        chunks = document.get("chunks")
        if not isinstance(chunks, list) or not chunks or document.get("chunk_count") != len(chunks):
            raise ValueError(f"stage 2 chunk completeness mismatch: {kind}")
        for index, chunk in enumerate(chunks):
            if chunk.get("chunk_index") != index or len(str(chunk.get("body_chunk_sha256"))) != 64:
                raise ValueError(f"stage 2 chunk identity mismatch: {kind}")
            _finite(chunk.get("raw_logit"), f"{kind} raw logit")
        by_id[document["source_id"]] = document
    recomputed = []
    for source_id in candidates:
        document = by_id[source_id]
        recomputed.append({"source_id": source_id, "path": document["path"], "chunk_count": document["chunk_count"], "document_score": _nlme([float(chunk["raw_logit"]) for chunk in document["chunks"]])})
    recomputed = _sorted_ranking(recomputed)
    for stored, calculated in zip(ranking, recomputed, strict=True):
        for field in ("source_id", "path", "chunk_count", "rank"):
            if stored.get(field) != calculated.get(field):
                raise ValueError(f"stage 2 NLME ranking order mismatch: {kind}")
        if not math.isclose(float(stored["document_score"]), float(calculated["document_score"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"stage 2 NLME score mismatch: {kind}")
    if packet["ranking_sha256"] != sha256_bytes(canonical_json_bytes(ranking)):
        raise ValueError(f"stage 2 NLME ranking mismatch: {kind}")
    model_spec = next(row for row in read_json(root / SOURCE_MODELS)["models"] if row["kind"] == kind)
    if packet["model_id"] != model_spec["model_id"] or packet["revision"] != model_spec["revision"]:
        raise ValueError(f"stage 2 model identity mismatch: {kind}")
    expected_hashes = {row["path"]: row["sha256"] for row in model_spec["required_files"]}
    if packet["model_files_sha256"] != expected_hashes:
        raise ValueError(f"stage 2 model hashes mismatch: {kind}")
    return packet, {"runtime_seconds": packet["runtime_seconds"], "peak_rss_bytes": packet["peak_rss_bytes"], "ranking_sha256": packet["ranking_sha256"], "document_count": len(ranking)}


def _verify_packets(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    stage1, stage1_diagnostics = _verify_stage1(root)
    workers = []
    diagnostics = {"stage1": stage1_diagnostics, "stage2": {}}
    for kind, path in zip(MODEL_KINDS, (MINILM, V2_M3), strict=True):
        worker, values = _verify_stage2(root, path, kind, stage1)
        workers.append(worker)
        diagnostics["stage2"][kind] = values
    return stage1, workers, diagnostics


def claim(root: Path = ROOT) -> dict[str, Any]:
    registered = _validate_tree(root, "claim")
    if any((root / path).exists() for path in (CLAIM, RESULT, ERROR, GOLD)):
        raise FileExistsError("recovery claim requires absent gold and no recovery evidence")
    source = _verify_source_evidence(root)
    stage1, workers, diagnostics = _verify_packets(root)
    manifest = _manifest(root)
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "worker_packets_complete_gold_absent",
        "source_protocol_id": SOURCE_PROTOCOL_ID, "source_protocol_status": "failed_finalizer_gold_stream_path",
        "source_commit": manifest["source_commit"], "registered_query_execution_count": 0,
        "model_forward_inference_count": 0, "retry_count": 0, "gold_present": False,
        "holdout_bearing_input_file_count": 0, "registered_source_sha256": registered,
        "source_evidence_sha256": manifest["source_evidence_sha256"], "worker_packet_diagnostics": diagnostics,
        "source_preflight_component_sha256": source["preflight"]["component_sha256"],
        "model_files_sha256": {"e5": stage1["model_files_sha256"], **{row["kind"]: row["model_files_sha256"] for row in workers}},
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / CLAIM, payload)
    return {"status": payload["status"], "payload_sha256": payload["payload_sha256"]}


def _gold(root: Path) -> str:
    value = read_json(root / GOLD)
    if set(value) != {"schema_version", "protocol_id", "case_id", "expected_source_id"}:
        raise ValueError("development-only gold fields mismatch")
    return str(value["expected_source_id"])


def finalize(root: Path = ROOT) -> dict[str, Any]:
    registered = _validate_tree(root, "finalizer")
    if (root / RESULT).exists() or (root / ERROR).exists():
        raise FileExistsError("recovery result or error already exists")
    manifest = _manifest(root)
    if sha256_file(root / GOLD) != manifest["development_gold_sha256"]:
        raise ValueError("development-only gold hash mismatch")
    claim_value = read_json(root / CLAIM)
    if claim_value.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in claim_value.items() if k != "payload_sha256"})):
        raise ValueError("recovery claim payload hash mismatch")
    if claim_value.get("status") != "worker_packets_complete_gold_absent":
        raise ValueError("recovery claim did not prove packet completeness")
    _verify_source_evidence(root)
    stage1, workers, diagnostics = _verify_packets(root)
    if claim_value.get("worker_packet_diagnostics") != diagnostics:
        raise ValueError("recovery claim diagnostics drift")
    expected = _gold(root)
    if expected not in stage1["candidate_source_ids"]:
        raise ValueError("expected source missing from stage 1 candidates")
    stage1_rank = next(row["rank"] for row in stage1["ranking"] if row["source_id"] == expected)
    observed = []
    for worker in workers:
        match = next(row for row in worker["ranking"] if row["source_id"] == expected)
        pipeline_runtime = float(stage1["runtime_seconds"]) + float(worker["runtime_seconds"])
        observed.append({
            "kind": worker["kind"], "expected_source_rank": match["rank"],
            "expected_source_within_cutoff": match["rank"] <= CUTOFF,
            "stage2_runtime_seconds": worker["runtime_seconds"], "pipeline_runtime_seconds": pipeline_runtime,
            "practical_target_met": pipeline_runtime <= PRACTICAL_SECONDS,
            "peak_rss_bytes": worker["peak_rss_bytes"], "ranking_sha256": worker["ranking_sha256"],
            "top_distractor": next(row for row in worker["ranking"] if row["source_id"] != expected),
        })
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "recovered_from_complete_gold_blind_worker_packets",
        "source_protocol_id": SOURCE_PROTOCOL_ID, "source_protocol_status": "failed_finalizer_gold_stream_path",
        "derivation": "completed_gold_blind_stage1_and_stage2_worker_packets", "source_commit": manifest["source_commit"],
        "registered_query_execution_count": 0, "model_forward_inference_count": 0, "retry_count": 0,
        "expected_source_id": expected, "corpus_document_count": DOCUMENT_COUNT, "candidate_k": CANDIDATE_K,
        "stage1_expected_source_rank": stage1_rank, "stage1_candidate_inclusion": True, "cutoff": CUTOFF,
        "primary_success": all(row["expected_source_within_cutoff"] for row in observed),
        "practical_success": all(row["practical_target_met"] for row in observed),
        "practical_target_seconds_per_pipeline": PRACTICAL_SECONDS,
        "runtime_measurement_boundary": read_json(root / SOURCE_MANIFEST)["runtime"]["measurement_boundary"],
        "stage1_runtime_seconds": stage1["runtime_seconds"], "stage1_peak_rss_bytes": stage1["peak_rss_bytes"],
        "stage2_models": observed, "inputs_sha256": registered,
        "source_evidence_sha256": manifest["source_evidence_sha256"], "recovery_claim_sha256": sha256_file(root / CLAIM),
        "environment": {"network": "not_used_by_stdlib_finalizer", "holdout_bearing_input_file_count": 0, "mixed_v5_query_or_gold_file_count": 0, "shared_database_open_count": 0},
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / RESULT, payload)
    return {"status": payload["status"], "stage1_rank": stage1_rank, "ranks": {row["kind"]: row["expected_source_rank"] for row in observed}, "pipeline_runtime_seconds": {row["kind"]: row["pipeline_runtime_seconds"] for row in observed}, "primary_success": payload["primary_success"], "practical_success": payload["practical_success"]}


def record_error(root: Path, message: str) -> dict[str, Any]:
    if not (root / CLAIM).exists() or (root / RESULT).exists():
        raise FileExistsError("recovery error requires claim and no result")
    payload = {"protocol_id": PROTOCOL_ID, "registered_query_execution_count": 0, "model_forward_inference_count": 0, "retry_count": 0, "error": message}
    write_json_exclusive(root / ERROR, payload)
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    _verify_source_evidence(root)
    evidence = {"claim": (root / CLAIM).exists(), "result": (root / RESULT).exists(), "error": (root / ERROR).exists()}
    if evidence["result"] and evidence["error"]:
        raise ValueError("recovery result and error coexist")
    result = read_json(root / RESULT) if evidence["result"] else None
    if result and result.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in result.items() if k != "payload_sha256"})):
        raise ValueError("recovered result payload hash mismatch")
    return {"protocol_id": PROTOCOL_ID, "status": result.get("status") if result else ("packet_complete" if evidence["claim"] else "recovery_frozen"), "source_protocol_status": "failed_finalizer_gold_stream_path", "registered_query_execution_count": 0, "model_forward_inference_count": 0, "holdout_bearing_input_file_count": 0, "evidence": evidence, "primary_success": result.get("primary_success") if result else None, "practical_success": result.get("practical_success") if result else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("claim", "finalize", "audit"):
        sub.add_parser(name).add_argument("--root", type=Path, default=ROOT)
    p = sub.add_parser("record-error"); p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    if args.command == "claim": result = claim(args.root)
    elif args.command == "finalize": result = finalize(args.root)
    elif args.command == "audit": result = audit(args.root)
    else: result = record_error(args.root, args.message)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

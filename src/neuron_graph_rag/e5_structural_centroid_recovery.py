"""Finalizer-only recovery for the completed #240 gold-blind worker packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_ID = "github-retrieval-parity-v5-e5-structural-centroid-finalizer-recovery-v1"
SOURCE_PROTOCOL_ID = "github-retrieval-parity-v5-e5-structural-centroid-ablation-v1"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/e5_structural_centroid_recovery_v1.manifest.json")
SCHEMA = Path("tests/fixtures/e5_structural_centroid_recovery_v1.schema.json")
SOURCE_MANIFEST = Path(
    "tests/fixtures/e5_structural_centroid_ablation_v1.manifest.json"
)
SOURCE_SCHEMA = Path("tests/fixtures/e5_structural_centroid_ablation_v1.schema.json")
MODEL_REGISTRY = Path("tests/fixtures/e5_structural_centroid_ablation_v1.model.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
SOURCE_EVIDENCE = Path("tests/evidence/e5_structural_centroid_ablation_v1")
PREFLIGHT = SOURCE_EVIDENCE / "development.preflight.json"
SOURCE_CLAIM = SOURCE_EVIDENCE / "development.claim.json"
WORKER = SOURCE_EVIDENCE / "development.worker.json"
SOURCE_ERROR = SOURCE_EVIDENCE / "development.error.json"
RECOVERY_EVIDENCE = Path("tests/evidence/e5_structural_centroid_finalizer_recovery_v1")
CLAIM = RECOVERY_EVIDENCE / "development.claim.json"
RESULT = RECOVERY_EVIDENCE / "development.recovered.json"
ARMS = ("body_max", "structural_max", "body_centroid", "structural_centroid")
REPRESENTATIONS = ("body", "structural")
DIMENSIONS = 384
DOCUMENT_COUNT = 93
PASSAGE_COUNT = 2065


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "frozen_before_gold_mount"
    ):
        raise ValueError("recovery manifest identity mismatch")
    if (
        value.get("registered_query_execution_count") != 0
        or value.get("model_forward_inference_count") != 0
    ):
        raise ValueError("recovery execution count mismatch")
    for relative, digest in value.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen recovery artifact changed: {relative}")
    return value


def _registered_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _validate_tree(root: Path, stage: str) -> dict[str, str]:
    manifest = _manifest(root)
    expected = set(manifest["claim_registered_files"])
    if stage == "finalizer":
        expected.update((CLAIM.as_posix(), GOLD.as_posix()))
    actual = _registered_files(root)
    if actual != expected:
        raise ValueError(
            f"recovery allowlist mismatch: missing={sorted(expected - actual)!r} extra={sorted(actual - expected)!r}"
        )
    for forbidden in manifest["forbidden_registered_paths"]:
        if (root / forbidden).exists():
            raise ValueError(f"forbidden recovery input exists: {forbidden}")
    return {relative: sha256_file(root / relative) for relative in sorted(expected)}


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"non-finite numeric value: {label}")
    return float(value)


def _unit_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != DIMENSIONS:
        raise ValueError(f"embedding dimension mismatch: {label}")
    vector = [_finite_number(item, label) for item in value]
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(f"embedding norm mismatch: {label}")
    return vector


def _rankdata(values: Sequence[float | int]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end + 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def _pearson(left: Sequence[float | int], right: Sequence[float | int]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal length at least two")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    if denominator == 0:
        raise ValueError("correlation input variance must be non-zero")
    return numerator / denominator


def _spearman(rows: Sequence[Mapping[str, Any]]) -> float:
    return _pearson(
        _rankdata([int(row["chunk_count"]) for row in rows]),
        _rankdata([float(row["document_score"]) for row in rows]),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _verify_source_evidence(root: Path) -> dict[str, Any]:
    manifest = _manifest(root)
    expected = manifest["source_evidence_sha256"]
    actual = {relative: sha256_file(root / relative) for relative in expected}
    if actual != expected:
        raise ValueError("immutable v1 evidence hash mismatch")
    preflight = read_json(root / PREFLIGHT)
    source_claim = read_json(root / SOURCE_CLAIM)
    source_error = read_json(root / SOURCE_ERROR)
    if (
        preflight.get("protocol_id") != SOURCE_PROTOCOL_ID
        or preflight.get("status") != "preflight_valid"
    ):
        raise ValueError("v1 preflight mismatch")
    if (
        preflight.get("registered_query_execution_count") != 0
        or preflight.get("gold_present") is not False
    ):
        raise ValueError("v1 preflight boundary mismatch")
    if (
        preflight.get("holdout_bearing_input_file_count") != 0
        or preflight.get("corpus_document_count") != DOCUMENT_COUNT
    ):
        raise ValueError("v1 preflight input mismatch")
    if (
        source_claim.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_claim.get("registered_query_count") != 1
    ):
        raise ValueError("v1 claim mismatch")
    if (
        source_claim.get("retry_count") != 0
        or source_claim.get("gold_present_in_worker") is not False
    ):
        raise ValueError("v1 claim boundary mismatch")
    if (
        source_error.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_error.get("retry_count") != 0
    ):
        raise ValueError("v1 error mismatch")
    source_manifest = read_json(root / SOURCE_MANIFEST)
    error_text = source_error.get("error", "")
    if GOLD.as_posix() not in source_manifest.get("forbidden_registered_paths", []):
        raise ValueError("v1 manifest does not expose the finalizer contradiction")
    if " finalize " not in error_text or GOLD.name not in error_text:
        raise ValueError("v1 failure cause mismatch")
    source_commit = manifest["source_commit"]
    if {
        preflight.get("source_commit"),
        source_claim.get("source_commit"),
        source_error.get("source_commit"),
    } != {source_commit}:
        raise ValueError("v1 source commit mismatch")
    if (
        source_claim.get("preflight_attestation_sha256")
        != expected[PREFLIGHT.as_posix()]
    ):
        raise ValueError("v1 claim/preflight binding mismatch")
    return {"preflight": preflight, "claim": source_claim, "error": source_error}


def _verify_worker(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _manifest(root)
    worker = read_json(root / WORKER)
    expected_keys = {
        "model_id",
        "revision",
        "model_files_sha256",
        "dependencies",
        "runtime_seconds",
        "peak_rss_bytes",
        "query_embedding",
        "query_embedding_sha256",
        "representations",
        "arms",
        "body_at_or_below_512_but_structural_above_512_count",
        "structural_increases_truncation",
    }
    if set(worker) != expected_keys:
        raise ValueError("worker top-level fields mismatch")
    model = read_json(root / MODEL_REGISTRY)
    if (
        worker["model_id"] != model["model_id"]
        or worker["revision"] != model["revision"]
    ):
        raise ValueError("worker model identity mismatch")
    if worker["model_files_sha256"] != model["files_sha256"]:
        raise ValueError("worker model file hashes mismatch")
    if worker["dependencies"] != manifest["dependencies"]:
        raise ValueError("worker dependency versions mismatch")
    if _finite_number(worker["runtime_seconds"], "runtime_seconds") <= 0:
        raise ValueError("worker runtime must be positive")
    if not isinstance(worker["peak_rss_bytes"], int) or worker["peak_rss_bytes"] <= 0:
        raise ValueError("worker peak RSS mismatch")
    query = _unit_vector(worker["query_embedding"], "query_embedding")
    if (
        sha256_bytes(canonical_json_bytes(worker["query_embedding"]))
        != worker["query_embedding_sha256"]
    ):
        raise ValueError("query embedding hash mismatch")

    representations = worker["representations"]
    if not isinstance(representations, list) or [
        row.get("representation") for row in representations
    ] != list(REPRESENTATIONS):
        raise ValueError("worker representation order mismatch")
    source_ids: set[str] | None = None
    rep_documents: dict[str, dict[str, dict[str, Any]]] = {}
    truncation_exceeded: dict[str, int] = {}
    total_passages: dict[str, int] = {}
    for representation in representations:
        name = representation["representation"]
        if set(representation) != {
            "representation",
            "runtime_seconds",
            "truncation",
            "documents",
        }:
            raise ValueError(f"representation fields mismatch: {name}")
        if (
            _finite_number(representation["runtime_seconds"], f"{name}.runtime_seconds")
            <= 0
        ):
            raise ValueError("representation runtime must be positive")
        documents = representation["documents"]
        if not isinstance(documents, list) or len(documents) != DOCUMENT_COUNT:
            raise ValueError(f"document count mismatch: {name}")
        current_ids = [row.get("source_id") for row in documents]
        if (
            any(not isinstance(item, str) or not item for item in current_ids)
            or len(set(current_ids)) != DOCUMENT_COUNT
        ):
            raise ValueError(f"source IDs mismatch: {name}")
        if source_ids is None:
            source_ids = set(current_ids)
        elif set(current_ids) != source_ids:
            raise ValueError("representation source set mismatch")
        by_id: dict[str, dict[str, Any]] = {}
        counted = 0
        exceeded = 0
        token_lengths: list[int] = []
        for document in documents:
            if set(document) != {
                "source_id",
                "path",
                "character_count",
                "chunk_count",
                "centroid_embedding",
                "chunks",
            }:
                raise ValueError(f"document fields mismatch: {name}")
            if (
                not isinstance(document["path"], str)
                or not isinstance(document["character_count"], int)
                or document["character_count"] < 0
            ):
                raise ValueError(f"document metadata mismatch: {name}")
            chunks = document["chunks"]
            if (
                not isinstance(chunks, list)
                or not chunks
                or document["chunk_count"] != len(chunks)
            ):
                raise ValueError(f"chunk count mismatch: {name}")
            _unit_vector(document["centroid_embedding"], f"{name}.centroid")
            for index, chunk in enumerate(chunks):
                if set(chunk) != {
                    "chunk_index",
                    "start_codepoint",
                    "end_codepoint",
                    "body_chunk_sha256",
                    "body_chunk_codepoint_length",
                    "prefix_codepoint_length",
                    "passage_token_length_before_truncation",
                    "passage_exceeds_512_before_truncation",
                    "query_cosine",
                }:
                    raise ValueError(f"chunk fields mismatch: {name}")
                if chunk["chunk_index"] != index or not (
                    0
                    <= chunk["start_codepoint"]
                    < chunk["end_codepoint"]
                    <= document["character_count"]
                ):
                    raise ValueError(f"chunk offsets mismatch: {name}")
                if (
                    chunk["body_chunk_codepoint_length"]
                    != chunk["end_codepoint"] - chunk["start_codepoint"]
                ):
                    raise ValueError(f"chunk length mismatch: {name}")
                if (
                    not isinstance(chunk["body_chunk_sha256"], str)
                    or len(chunk["body_chunk_sha256"]) != 64
                ):
                    raise ValueError(f"chunk hash mismatch: {name}")
                if (
                    not isinstance(chunk["prefix_codepoint_length"], int)
                    or chunk["prefix_codepoint_length"] < 0
                ):
                    raise ValueError(f"prefix length mismatch: {name}")
                token_length = chunk["passage_token_length_before_truncation"]
                if not isinstance(token_length, int) or token_length <= 0:
                    raise ValueError(f"token length mismatch: {name}")
                if chunk["passage_exceeds_512_before_truncation"] is not (
                    token_length > 512
                ):
                    raise ValueError(f"truncation flag mismatch: {name}")
                _finite_number(chunk["query_cosine"], f"{name}.query_cosine")
                token_lengths.append(token_length)
                exceeded += int(token_length > 512)
            counted += len(chunks)
            by_id[document["source_id"]] = document
        if counted != PASSAGE_COUNT:
            raise ValueError(f"passage count mismatch: {name}")
        summary = representation["truncation"]
        expected_summary = {
            "passage_count": counted,
            "passages_exceeding_512_before_truncation": exceeded,
            "passage_token_length": {
                "min": min(token_lengths),
                "max": max(token_lengths),
                "mean": sum(token_lengths) / len(token_lengths),
            },
        }
        if summary != expected_summary:
            raise ValueError(f"truncation summary mismatch: {name}")
        rep_documents[name] = by_id
        truncation_exceeded[name] = exceeded
        total_passages[name] = counted

    increased = 0
    for source_id in sorted(source_ids or ()):
        body = rep_documents["body"][source_id]
        structural = rep_documents["structural"][source_id]
        if (body["path"], body["character_count"], body["chunk_count"]) != (
            structural["path"],
            structural["character_count"],
            structural["chunk_count"],
        ):
            raise ValueError("body/structural document identity mismatch")
        for left, right in zip(body["chunks"], structural["chunks"], strict=True):
            fields = (
                "chunk_index",
                "start_codepoint",
                "end_codepoint",
                "body_chunk_sha256",
                "body_chunk_codepoint_length",
            )
            if tuple(left[field] for field in fields) != tuple(
                right[field] for field in fields
            ):
                raise ValueError("body/structural chunk identity mismatch")
            if left["prefix_codepoint_length"] != 0:
                raise ValueError("body prefix must be empty")
            increased += int(
                not left["passage_exceeds_512_before_truncation"]
                and right["passage_exceeds_512_before_truncation"]
            )
    if worker["body_at_or_below_512_but_structural_above_512_count"] != increased:
        raise ValueError("structural truncation increase count mismatch")
    if worker["structural_increases_truncation"] is not (increased > 0):
        raise ValueError("structural truncation increase flag mismatch")

    arms = worker["arms"]
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != list(ARMS):
        raise ValueError("worker arm order mismatch")
    verified_arms: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_id = arm["arm_id"]
        representation, aggregation = arm_id.rsplit("_", 1)
        if set(arm) != {
            "arm_id",
            "representation",
            "aggregation",
            "documents",
            "spearman_chunk_count_vs_document_score",
            "ranking_sha256",
        }:
            raise ValueError(f"arm fields mismatch: {arm_id}")
        if arm["representation"] != representation or arm["aggregation"] != aggregation:
            raise ValueError(f"arm identity mismatch: {arm_id}")
        documents = arm["documents"]
        if not isinstance(documents, list) or len(documents) != DOCUMENT_COUNT:
            raise ValueError(f"arm document count mismatch: {arm_id}")
        expected_rows = []
        for source_id in sorted(source_ids or ()):
            document = rep_documents[representation][source_id]
            if aggregation == "max":
                score = max(
                    float(chunk["query_cosine"]) for chunk in document["chunks"]
                )
            else:
                score = _dot(
                    query, [float(item) for item in document["centroid_embedding"]]
                )
            expected_rows.append(
                {
                    "source_id": source_id,
                    "path": document["path"],
                    "chunk_count": document["chunk_count"],
                    "document_score": score,
                }
            )
        expected_rows.sort(key=lambda row: (-row["document_score"], row["source_id"]))
        for rank, row in enumerate(expected_rows, 1):
            row["rank"] = rank
        if documents != expected_rows:
            raise ValueError(f"arm ranking does not match frozen formula: {arm_id}")
        if arm["ranking_sha256"] != sha256_bytes(canonical_json_bytes(documents)):
            raise ValueError(f"ranking hash mismatch: {arm_id}")
        if arm["spearman_chunk_count_vs_document_score"] != _spearman(documents):
            raise ValueError(f"Spearman mismatch: {arm_id}")
        verified_arms[arm_id] = arm
    forbidden_keys = {
        "expected_source_id",
        "expected_source_rank",
        "expected_source_within_cutoff",
        "top_distractor",
        "gold",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden_keys.intersection(value):
                raise ValueError("gold-derived field exists in worker packet")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(worker)
    diagnostics = {
        "document_count": DOCUMENT_COUNT,
        "passage_count": total_passages,
        "passages_exceeding_512_before_truncation": truncation_exceeded,
        "body_at_or_below_512_but_structural_above_512_count": increased,
        "structural_increases_truncation": increased > 0,
        "ranking_sha256": {
            arm_id: verified_arms[arm_id]["ranking_sha256"] for arm_id in ARMS
        },
        "spearman_chunk_count_vs_document_score": {
            arm_id: verified_arms[arm_id]["spearman_chunk_count_vs_document_score"]
            for arm_id in ARMS
        },
    }
    return worker, diagnostics


def claim(root: Path = ROOT) -> dict[str, Any]:
    registered = _validate_tree(root, "claim")
    if (root / CLAIM).exists() or (root / RESULT).exists() or (root / GOLD).exists():
        raise FileExistsError(
            "recovery claim requires no prior recovery evidence and no gold"
        )
    source = _verify_source_evidence(root)
    _, diagnostics = _verify_worker(root)
    manifest = _manifest(root)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "worker_packet_complete_gold_absent",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_protocol_status": "failed_finalizer_allowlist_contradiction",
        "source_commit": manifest["source_commit"],
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "gold_present": False,
        "holdout_bearing_input_file_count": 0,
        "registered_source_sha256": registered,
        "source_evidence_sha256": manifest["source_evidence_sha256"],
        "worker_packet_sha256": sha256_file(root / WORKER),
        "model_files_sha256": source["preflight"]["model_files_sha256"],
        "packet_diagnostics": diagnostics,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / CLAIM, payload)
    return {
        "status": payload["status"],
        "worker_packet_sha256": payload["worker_packet_sha256"],
        "payload_sha256": payload["payload_sha256"],
    }


def _gold(root: Path) -> str:
    value = read_json(root / GOLD)
    if set(value) != {"schema_version", "protocol_id", "case_id", "expected_source_id"}:
        raise ValueError("development-only gold fields mismatch")
    return str(value["expected_source_id"])


def finalize(root: Path = ROOT) -> dict[str, Any]:
    registered = _validate_tree(root, "finalizer")
    if (root / RESULT).exists():
        raise FileExistsError("recovered result already exists")
    manifest = _manifest(root)
    if sha256_file(root / GOLD) != manifest["development_gold_sha256"]:
        raise ValueError("development-only gold hash mismatch")
    claim_value = read_json(root / CLAIM)
    claim_without_hash = {
        key: value for key, value in claim_value.items() if key != "payload_sha256"
    }
    if claim_value.get("payload_sha256") != sha256_bytes(
        canonical_json_bytes(claim_without_hash)
    ):
        raise ValueError("recovery claim payload hash mismatch")
    if claim_value.get("status") != "worker_packet_complete_gold_absent":
        raise ValueError("recovery claim did not prove packet completeness")
    if claim_value.get("worker_packet_sha256") != sha256_file(root / WORKER):
        raise ValueError("recovery claim/worker binding mismatch")
    _verify_source_evidence(root)
    worker, diagnostics = _verify_worker(root)
    if claim_value.get("packet_diagnostics") != diagnostics:
        raise ValueError("recovery claim diagnostics drift")
    expected = _gold(root)
    by_arm: dict[str, dict[str, Any]] = {}
    observed_arms = []
    for arm in worker["arms"]:
        match = next(
            (row for row in arm["documents"] if row["source_id"] == expected), None
        )
        if match is None:
            raise ValueError("expected source absent from complete worker packet")
        top_distractor = next(
            row for row in arm["documents"] if row["source_id"] != expected
        )
        observed = {
            "arm_id": arm["arm_id"],
            "representation": arm["representation"],
            "aggregation": arm["aggregation"],
            "expected_source_rank": match["rank"],
            "expected_source_within_cutoff": match["rank"] <= 20,
            "top_distractor": top_distractor,
            "spearman_chunk_count_vs_document_score": arm[
                "spearman_chunk_count_vs_document_score"
            ],
            "ranking_sha256": arm["ranking_sha256"],
        }
        observed_arms.append(observed)
        by_arm[arm["arm_id"]] = observed
    primary = [arm for arm in ARMS[1:] if by_arm[arm]["expected_source_rank"] <= 20]
    directional = [
        arm
        for arm in ARMS[1:]
        if by_arm[arm]["expected_source_rank"]
        < by_arm["body_max"]["expected_source_rank"]
    ]
    attenuation = {
        representation: abs(
            by_arm[f"{representation}_centroid"][
                "spearman_chunk_count_vs_document_score"
            ]
        )
        < abs(by_arm[f"{representation}_max"]["spearman_chunk_count_vs_document_score"])
        for representation in REPRESENTATIONS
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "recovered_from_complete_gold_blind_worker_packet",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_protocol_status": "failed_finalizer_allowlist_contradiction",
        "derivation": "completed_gold_blind_worker_packet",
        "source_commit": manifest["source_commit"],
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "retry_count": 0,
        "expected_source_id": expected,
        "corpus_document_count": DOCUMENT_COUNT,
        "cutoff": 20,
        "primary_success": bool(primary),
        "primary_success_arms": primary,
        "directional_evidence_arms": directional,
        "length_bias_attenuation": attenuation,
        "arms": observed_arms,
        "truncation": {
            "passage_count": diagnostics["passage_count"],
            "passages_exceeding_512_before_truncation": diagnostics[
                "passages_exceeding_512_before_truncation"
            ],
            "body_at_or_below_512_but_structural_above_512_count": diagnostics[
                "body_at_or_below_512_but_structural_above_512_count"
            ],
            "structural_increases_truncation": diagnostics[
                "structural_increases_truncation"
            ],
        },
        "runtime_seconds": worker["runtime_seconds"],
        "peak_rss_bytes": worker["peak_rss_bytes"],
        "environment": {
            "network": "not_used_by_stdlib_finalizer",
            "fresh_registered_tree": True,
            "holdout_bearing_input_file_count": 0,
            "mixed_v5_query_or_gold_file_count": 0,
            "shared_database_open_count": 0,
        },
        "inputs_sha256": registered,
        "source_evidence_sha256": manifest["source_evidence_sha256"],
        "worker_packet_sha256": sha256_file(root / WORKER),
        "recovery_claim_sha256": sha256_file(root / CLAIM),
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / RESULT, payload)
    return {
        "status": payload["status"],
        "ranks": {row["arm_id"]: row["expected_source_rank"] for row in observed_arms},
        "primary_success": payload["primary_success"],
        "primary_success_arms": primary,
        "directional_evidence_arms": directional,
        "length_bias_attenuation": attenuation,
        "payload_sha256": payload["payload_sha256"],
    }


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    source = _verify_source_evidence(root)
    source_result = root / SOURCE_EVIDENCE / "development.observed.json"
    if source_result.exists():
        raise ValueError("failed v1 protocol must not have result evidence")
    evidence = {"claim": (root / CLAIM).exists(), "result": (root / RESULT).exists()}
    result = read_json(root / RESULT) if evidence["result"] else None
    if result:
        without_hash = {
            key: value for key, value in result.items() if key != "payload_sha256"
        }
        if result.get("payload_sha256") != sha256_bytes(
            canonical_json_bytes(without_hash)
        ):
            raise ValueError("recovered result payload hash mismatch")
        if (
            result.get("registered_query_execution_count") != 0
            or result.get("model_forward_inference_count") != 0
        ):
            raise ValueError("recovery performed registered inference")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": result.get("status")
        if result
        else ("packet_complete" if evidence["claim"] else "recovery_frozen"),
        "source_protocol_status": "failed_finalizer_allowlist_contradiction",
        "source_error": source["error"]["error"],
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "holdout_bearing_input_file_count": 0,
        "evidence": evidence,
        "primary_success": result.get("primary_success") if result else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("claim", "finalize", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "claim":
        result = claim(args.root)
    elif args.command == "finalize":
        result = finalize(args.root)
    else:
        result = audit(args.root)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

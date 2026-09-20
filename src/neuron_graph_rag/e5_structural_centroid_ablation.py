"""Holdout-absent multilingual-e5-small structural/centroid ablation."""
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

from . import full_corpus_rerank_oracle as v1
from . import structural_representation_length_bias_diagnostic as structural

PROTOCOL_ID = "github-retrieval-parity-v5-e5-structural-centroid-ablation-v1"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/e5_structural_centroid_ablation_v1.manifest.json")
SCHEMA = Path("tests/fixtures/e5_structural_centroid_ablation_v1.schema.json")
MODEL_REGISTRY = Path("tests/fixtures/e5_structural_centroid_ablation_v1.model.json")
QUERY = Path("tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
GOLD = Path("tests/fixtures/full_corpus_rerank_oracle_v2.gold.json")
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
PARITY = Path("tests/evidence/e5_structural_centroid_ablation_v1/result_free_parity.json")
CLAIM = Path("tests/evidence/e5_structural_centroid_ablation_v1/development.claim.json")
RESULT = Path("tests/evidence/e5_structural_centroid_ablation_v1/development.observed.json")
ERROR = Path("tests/evidence/e5_structural_centroid_ablation_v1/development.error.json")
ATTESTATION = Path("preflight.attestation.json")
ARMS = ("body_max", "structural_max", "body_centroid", "structural_centroid")
REPRESENTATIONS = ("body", "structural")
MAX_LENGTH = 512
PARITY_MAX_ABS_TOLERANCE = 1e-6
PARITY_MIN_COSINE = 0.999999

canonical_json_bytes = v1.canonical_json_bytes
sha256_bytes = v1.sha256_bytes
sha256_file = v1.sha256_file
read_json = v1.read_json
write_json_exclusive = v1.write_json_exclusive


def _manifest(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "frozen_pre_registered_execution":
        raise ValueError("frozen manifest identity mismatch")
    if [row.get("arm_id") for row in value.get("arms", [])] != list(ARMS):
        raise ValueError("frozen arm order mismatch")
    if value["ranking"]["cutoff"] != 20:
        raise ValueError("cutoff mismatch")
    if value["parity_gate"]["max_absolute_difference_tolerance"] != PARITY_MAX_ABS_TOLERANCE:
        raise ValueError("parity tolerance mismatch")
    if value["parity_gate"]["minimum_cosine_similarity"] != PARITY_MIN_COSINE:
        raise ValueError("parity cosine mismatch")
    for relative, digest in value.get("artifact_sha256", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256_file(root / relative) != digest:
            raise ValueError(f"frozen artifact changed: {relative}")
    return value


def _model(root: Path = ROOT) -> dict[str, Any]:
    value = read_json(root / MODEL_REGISTRY)
    if value.get("model_id") != "intfloat/multilingual-e5-small":
        raise ValueError("model identity mismatch")
    if value.get("revision") != "614241f622f53c4eeff9890bdc4f31cfecc418b3":
        raise ValueError("model revision mismatch")
    return value


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


def _registered_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _validate_tree(root: Path, stage: str) -> dict[str, str]:
    manifest = _manifest(root)
    expected = set(manifest["worker_registered_files"])
    if stage in {"worker", "finalizer"}:
        expected.add(CLAIM.as_posix())
    if stage == "finalizer":
        expected.add(GOLD.as_posix())
    actual = _registered_files(root)
    if actual != expected:
        raise ValueError(f"registered allowlist mismatch: missing={sorted(expected-actual)!r} extra={sorted(actual-expected)!r}")
    for forbidden in manifest["forbidden_registered_paths"]:
        if (root / forbidden).exists():
            raise ValueError(f"forbidden registered path exists: {forbidden}")
    return {relative: sha256_file(root / relative) for relative in sorted(expected)}


def _verify_model_files(root: Path, cache: Path) -> dict[str, str]:
    spec = _model(root)
    result = {}
    for relative, expected in spec["files_sha256"].items():
        path = cache / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pinned model file mismatch: {relative}")
        result[relative] = expected
    return result


class DirectE5:
    def __init__(self, root: Path, cache: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        spec = _model(root)
        snapshot = cache / spec["snapshot_root"]
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(snapshot / "onnx/model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=self._options(ort),
        )

    @staticmethod
    def _options(ort: Any) -> Any:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        return options

    def token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        np = self.np
        result: list[list[float]] = []
        for offset in range(0, len(texts), 8):
            encodings = self.tokenizer.encode_batch(list(texts[offset : offset + 8]))
            rows = []
            for encoding in encodings:
                ids = encoding.ids[:MAX_LENGTH]
                mask = encoding.attention_mask[:MAX_LENGTH]
                types = encoding.type_ids[:MAX_LENGTH]
                rows.append((ids, mask, types))
            width = max(len(row[0]) for row in rows)
            pad = self.tokenizer.token_to_id("<pad>")
            arrays = {}
            for name, index, fill in (("input_ids", 0, pad), ("attention_mask", 1, 0), ("token_type_ids", 2, 0)):
                arrays[name] = np.asarray([row[index] + [fill] * (width - len(row[index])) for row in rows], dtype=np.int64)
            hidden = self.session.run(None, arrays)[0]
            mask = arrays["attention_mask"][:, :, None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if bool((norms == 0).any()):
                raise ValueError("zero-norm embedding")
            result.extend((pooled / norms).astype(np.float32).tolist())
        return result


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def parity(root: Path, cache: Path, output: Path) -> dict[str, Any]:
    """Result-free comparison against #230 FastEmbed on synthetic text only."""
    from .semantic_retrieval import MultilingualE5

    _verify_model_files(root, cache)
    raw = ["synthetic multilingual parity", "A compact synthetic passage.", "日本語を含む合成文書です。"]
    texts = ["query: " + raw[0], "passage: " + raw[1], "passage: " + raw[2]]
    direct = DirectE5(root, cache).embed(texts)
    fastembed = MultilingualE5(cache)
    reference = [list(fastembed.query(raw[0]))] + [list(row) for row in fastembed.passages(raw[1:])]
    comparisons = []
    for label, left, right in zip(("query", "passage-en", "passage-ja"), direct, reference, strict=True):
        maximum = max(abs(a - b) for a, b in zip(left, right, strict=True))
        cosine = _cosine(left, right)
        comparisons.append({"label": label, "max_absolute_difference": maximum, "cosine_similarity": cosine})
    passed = all(row["max_absolute_difference"] <= PARITY_MAX_ABS_TOLERANCE and row["cosine_similarity"] >= PARITY_MIN_COSINE for row in comparisons)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "result_free_parity_valid" if passed else "result_free_parity_failed",
        "registered_query_execution_count": 0,
        "texts": texts,
        "prefixes": {"query": "query: ", "passage": "passage: "},
        "pooling": "attention-mask mean pooling then L2 normalization",
        "max_length": MAX_LENGTH,
        "max_absolute_difference_tolerance": PARITY_MAX_ABS_TOLERANCE,
        "minimum_cosine_similarity": PARITY_MIN_COSINE,
        "comparisons": comparisons,
        "model_files_sha256": _verify_model_files(root, cache),
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    if not passed:
        raise ValueError("direct ONNX implementation did not match FastEmbed")
    write_json_exclusive(output, payload)
    return payload


def _rankdata(values: Sequence[float | int]) -> list[float]:
    return structural._rankdata(values)


def _spearman(rows: Sequence[Mapping[str, Any]]) -> float:
    return structural._pearson(
        _rankdata([int(row["chunk_count"]) for row in rows]),
        _rankdata([float(row["document_score"]) for row in rows]),
    )


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    values = [sum(column) / len(vectors) for column in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("zero-norm document centroid")
    return [value / norm for value in values]


def _score_representation(
    documents: Sequence[Mapping[str, Any]], backend: DirectE5, query_vector: Sequence[float], representation: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    token_lengths: list[int] = []
    exceeded = 0
    for document in documents:
        body = str(document["content"])
        chunks = v1.project_passages(body, window=480, overlap=80)
        texts = []
        rows = []
        for index, chunk in enumerate(chunks):
            prefix = "" if representation == "body" else structural.structural_prefix(str(document["path"]), body, int(chunk["start_codepoint"]))
            passage = "passage: " + prefix + str(chunk["text"])
            token_length = backend.token_length(passage)
            token_lengths.append(token_length)
            exceeded += int(token_length > MAX_LENGTH)
            texts.append(passage)
            rows.append({
                "chunk_index": index,
                "start_codepoint": chunk["start_codepoint"],
                "end_codepoint": chunk["end_codepoint"],
                "body_chunk_sha256": sha256_bytes(str(chunk["text"]).encode("utf-8")),
                "body_chunk_codepoint_length": len(str(chunk["text"])),
                "prefix_codepoint_length": len(prefix),
                "passage_token_length_before_truncation": token_length,
                "passage_exceeds_512_before_truncation": token_length > MAX_LENGTH,
            })
        vectors = backend.embed(texts)
        for row, vector in zip(rows, vectors, strict=True):
            row["query_cosine"] = _cosine(query_vector, vector)
        output.append({
            "source_id": document["source_id"],
            "path": document["path"],
            "character_count": len(body),
            "chunk_count": len(chunks),
            "centroid_embedding": _centroid(vectors),
            "chunks": rows,
        })
    return output, {
        "passage_count": len(token_lengths),
        "passages_exceeding_512_before_truncation": exceeded,
        "passage_token_length": {"min": min(token_lengths), "max": max(token_lengths), "mean": sum(token_lengths) / len(token_lengths)},
    }


def _arms(representations: Sequence[Mapping[str, Any]], query_vector: Sequence[float]) -> list[dict[str, Any]]:
    result = []
    by_id = {row["representation"]: row for row in representations}
    for arm_id in ARMS:
        representation, aggregation = arm_id.rsplit("_", 1)
        rows = []
        for document in by_id[representation]["documents"]:
            score = max(float(chunk["query_cosine"]) for chunk in document["chunks"])
            if aggregation == "centroid":
                score = _cosine(query_vector, document["centroid_embedding"])
            rows.append({"source_id": document["source_id"], "path": document["path"], "chunk_count": document["chunk_count"], "document_score": score})
        rows.sort(key=lambda row: (-row["document_score"], row["source_id"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        result.append({"arm_id": arm_id, "representation": representation, "aggregation": aggregation, "documents": rows, "spearman_chunk_count_vs_document_score": _spearman(rows), "ranking_sha256": sha256_bytes(canonical_json_bytes(rows))})
    return result


def preflight(root: Path, cache: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not v1._network_is_disabled():
        raise RuntimeError("preflight requires offline Linux x86_64")
    hashes = _validate_tree(root, "preflight")
    if (root / GOLD).exists():
        raise ValueError("gold must be absent from preflight")
    parity_evidence = read_json(root / PARITY)
    if parity_evidence.get("status") != "result_free_parity_valid" or sha256_file(root / PARITY) != _manifest(root)["parity_gate"]["evidence_sha256"]:
        raise ValueError("result-free parity gate mismatch")
    documents = v1._documents(root)
    for document in documents:
        for chunk in v1.project_passages(document["content"], window=480, overlap=80):
            structural.structural_prefix(document["path"], document["content"], chunk["start_codepoint"])
    model_hashes = _verify_model_files(root, cache)
    backend = DirectE5(root, cache)
    synthetic = backend.embed(["query: synthetic query", "passage: synthetic passage"])
    if len(synthetic) != 2 or any(not math.isfinite(x) for row in synthetic for x in row):
        raise ValueError("synthetic forward failed")
    if runtime_root.exists():
        raise FileExistsError("runtime root must be fresh")
    runtime_root.mkdir(parents=True)
    payload = {"protocol_id": PROTOCOL_ID, "status": "preflight_valid", "source_commit": source_commit, "registered_query_execution_count": 0, "gold_present": False, "holdout_bearing_input_file_count": 0, "corpus_document_count": len(documents), "registered_source_sha256": hashes, "model_files_sha256": model_hashes, "parity_evidence_sha256": sha256_file(root / PARITY)}
    write_json_exclusive(runtime_root / ATTESTATION, payload)
    return payload


def claim(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    hashes = _validate_tree(root, "claim")
    preflight_value = read_json(runtime_root / ATTESTATION)
    if preflight_value.get("source_commit") != source_commit or preflight_value.get("registered_source_sha256") != hashes:
        raise ValueError("preflight binding mismatch")
    if any((root / path).exists() for path in (CLAIM, RESULT, ERROR)):
        raise FileExistsError("append-only evidence exists")
    payload = {"protocol_id": PROTOCOL_ID, "source_commit": source_commit, "manifest_sha256": sha256_file(root / MANIFEST), "preflight_attestation_sha256": sha256_file(runtime_root / ATTESTATION), "registered_query_count": 1, "retry_count": 0, "worker_count": 1, "gold_present_in_worker": False}
    write_json_exclusive(root / CLAIM, payload)
    return payload


def worker(root: Path, cache: Path, runtime_root: Path) -> dict[str, Any]:
    _validate_tree(root, "worker")
    if (root / GOLD).exists():
        raise ValueError("worker gold must be absent")
    query = _query(root)
    documents = v1._documents(root)
    started = time.perf_counter()
    backend = DirectE5(root, cache)
    query_vector = backend.embed(["query: " + query])[0]
    representations = []
    for representation in REPRESENTATIONS:
        rep_started = time.perf_counter()
        rows, truncation = _score_representation(documents, backend, query_vector, representation)
        representations.append({"representation": representation, "runtime_seconds": time.perf_counter() - rep_started, "truncation": truncation, "documents": rows})
    body = {row["source_id"]: row for row in representations[0]["documents"]}
    metadata = {row["source_id"]: row for row in representations[1]["documents"]}
    increased = 0
    for source_id in body:
        for left, right in zip(body[source_id]["chunks"], metadata[source_id]["chunks"], strict=True):
            identity = ("start_codepoint", "end_codepoint", "body_chunk_sha256")
            if tuple(left[k] for k in identity) != tuple(right[k] for k in identity):
                raise ValueError("body chunk identity drift")
            increased += int(not left["passage_exceeds_512_before_truncation"] and right["passage_exceeds_512_before_truncation"])
    import importlib.metadata
    import resource

    payload = {
        "model_id": _model(root)["model_id"],
        "revision": _model(root)["revision"],
        "model_files_sha256": _verify_model_files(root, cache),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime", "tokenizers")},
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "query_embedding": query_vector,
        "query_embedding_sha256": sha256_bytes(canonical_json_bytes(query_vector)),
        "representations": representations,
        "arms": _arms(representations, query_vector),
        "body_at_or_below_512_but_structural_above_512_count": increased,
        "structural_increases_truncation": increased > 0,
    }
    write_json_exclusive(runtime_root / "worker.json", payload)
    del backend
    gc.collect()
    return {"status": "worker_complete"}


def finalize(root: Path, runtime_root: Path, source_commit: str) -> dict[str, Any]:
    _validate_tree(root, "finalizer")
    expected = _gold(root)
    worker_value = read_json(runtime_root / "worker.json")
    by_arm = {}
    for arm in worker_value["arms"]:
        match = next(row for row in arm["documents"] if row["source_id"] == expected)
        arm["expected_source_rank"] = match["rank"]
        arm["expected_source_within_cutoff"] = match["rank"] <= 20
        arm["top_distractor"] = next(row for row in arm["documents"] if row["source_id"] != expected)
        by_arm[arm["arm_id"]] = arm
    primary = [arm for arm in ARMS[1:] if by_arm[arm]["expected_source_rank"] <= 20]
    directional = [arm for arm in ARMS[1:] if by_arm[arm]["expected_source_rank"] < by_arm["body_max"]["expected_source_rank"]]
    attenuation = {rep: abs(by_arm[f"{rep}_centroid"]["spearman_chunk_count_vs_document_score"]) < abs(by_arm[f"{rep}_max"]["spearman_chunk_count_vs_document_score"]) for rep in REPRESENTATIONS}
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "source_commit": source_commit, "expected_source_id": expected, "corpus_document_count": 93, "cutoff": 20, "primary_success": bool(primary), "primary_success_arms": primary, "directional_evidence_arms": directional, "length_bias_attenuation": attenuation, "environment": {"os": "linux", "architecture": platform.machine(), "network": "disabled", "fresh_runtime": True, "holdout_bearing_input_file_count": 0, "shared_database_open_count": 0}, "inputs_sha256": {path.as_posix(): sha256_file(root / path) for path in (MANIFEST, SCHEMA, MODEL_REGISTRY, QUERY, GOLD, CORPUS, PARITY)}, "worker": worker_value}
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    write_json_exclusive(root / RESULT, payload)
    return {"status": "observed_valid", "primary_success": payload["primary_success"], "primary_success_arms": primary, "directional_evidence_arms": directional, "length_bias_attenuation": attenuation, "ranks": {arm: by_arm[arm]["expected_source_rank"] for arm in ARMS}, "payload_sha256": payload["payload_sha256"]}


def record_error(root: Path, source_commit: str, message: str) -> dict[str, Any]:
    if not (root / CLAIM).exists() or (root / RESULT).exists():
        raise FileExistsError("error evidence requires claim and no result")
    payload = {"protocol_id": PROTOCOL_ID, "source_commit": source_commit, "retry_count": 0, "error": message}
    write_json_exclusive(root / ERROR, payload)
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    _manifest(root)
    parity_value = read_json(root / PARITY)
    if parity_value.get("status") != "result_free_parity_valid" or parity_value.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in parity_value.items() if k != "payload_sha256"})):
        raise ValueError("parity evidence invalid")
    evidence = {"claim": (root / CLAIM).exists(), "result": (root / RESULT).exists(), "error": (root / ERROR).exists()}
    if evidence["result"] and evidence["error"]:
        raise ValueError("result and error coexist")
    result = read_json(root / RESULT) if evidence["result"] else None
    if result and result.get("payload_sha256") != sha256_bytes(canonical_json_bytes({k: v for k, v in result.items() if k != "payload_sha256"})):
        raise ValueError("result payload hash mismatch")
    return {"protocol_id": PROTOCOL_ID, "status": "observed_valid" if result else "result_free_frozen", "registered_query_execution_count": 1 if result else 0, "holdout_bearing_input_file_count": 0, "parity_status": parity_value["status"], "evidence": evidence, "primary_success": result.get("primary_success") if result else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "probe"):
        sub.add_parser(name).add_argument("--root", type=Path, default=ROOT)
    p = sub.add_parser("parity"); p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, default=PARITY)
    p = sub.add_parser("preflight"); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("claim"); p.add_argument("--root", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("worker"); p.add_argument("--root", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True)
    p = sub.add_parser("finalize"); p.add_argument("--root", type=Path, required=True); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("record-error"); p.add_argument("--root", type=Path, required=True); p.add_argument("--source-commit", required=True); p.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    if args.command in {"audit", "probe"}: result = audit(args.root)
    elif args.command == "parity": result = parity(args.root, args.cache, args.output)
    elif args.command == "preflight": result = preflight(args.root, args.cache, args.runtime_root, args.source_commit)
    elif args.command == "claim": result = claim(args.root, args.runtime_root, args.source_commit)
    elif args.command == "worker": result = worker(args.root, args.cache, args.runtime_root)
    elif args.command == "finalize": result = finalize(args.root, args.runtime_root, args.source_commit)
    else: result = record_error(args.root, args.source_commit, args.message)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import full_corpus_rerank_oracle_v2 as v2

PROTOCOL_ID = "github-retrieval-parity-v5-structural-length-bias-ablation-v1"
ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path("tests/fixtures/github_retrieval_parity_v4.corpus.json")
V2_RESULT = Path(
    "tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json"
)
MANIFEST = Path(
    "tests/fixtures/structural_representation_length_bias_ablation_v1.manifest.json"
)
DIAGNOSTIC = Path(
    "tests/evidence/structural_representation_length_bias_ablation_v1/"
    "result_free_diagnostic.json"
)
SNAPSHOT_PREFIX = "corpora/github-retrieval-parity-v4/"
ATX_HEADING = re.compile(r"^(#{1,6}) (.*)$")
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
MISSING = "[missing]"

canonical_json_bytes = v2.canonical_json_bytes
sha256_bytes = v2.sha256_bytes
sha256_file = v2.sha256_file
read_json = v2.read_json


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
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    if denominator == 0:
        raise ValueError("correlation input variance must be non-zero")
    return numerator / denominator


def _correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    chunk_counts = [int(row["chunk_count"]) for row in rows]
    scores = [float(row["best_chunk_score"]) for row in rows]
    ranks = [int(row["rank"]) for row in rows]
    return {
        "pearson_chunk_count_vs_best_score": _pearson(chunk_counts, scores),
        "spearman_chunk_count_vs_best_score": _pearson(
            _rankdata(chunk_counts), _rankdata(scores)
        ),
        "spearman_chunk_count_vs_rank": _pearson(
            _rankdata(chunk_counts), _rankdata(ranks)
        ),
    }


def _atx_headings(text: str) -> list[dict[str, Any]]:
    headings = []
    offset = 0
    fence_marker: str | None = None
    fence_length = 0
    for line_with_ending in text.splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_marker and len(marker) >= fence_length:
                fence_marker = None
                fence_length = 0
        elif fence_marker is None:
            heading = ATX_HEADING.match(line)
            if heading:
                headings.append(
                    {
                        "level": len(heading.group(1)),
                        "text": heading.group(2),
                        "start_codepoint": offset,
                    }
                )
        offset += len(line_with_ending)
    return headings


def _normalize_metadata(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized if normalized else MISSING


def structural_fields(path: str, content: str, chunk_start: int) -> dict[str, Any]:
    posix_path = path.replace("\\", "/")
    if not posix_path.startswith(SNAPSHOT_PREFIX):
        raise ValueError(
            f"corpus path is outside the frozen snapshot prefix: {posix_path}"
        )
    headings = _atx_headings(content)
    title = next(
        (
            _normalize_metadata(str(row["text"]))
            for row in headings
            if row["level"] == 1
        ),
        MISSING,
    )
    stack: list[dict[str, Any]] = []
    for heading in headings:
        if int(heading["start_codepoint"]) >= chunk_start:
            break
        level = int(heading["level"])
        stack = [row for row in stack if int(row["level"]) < level]
        stack.append(heading)
    return {
        "repository_path": _normalize_metadata(
            posix_path.removeprefix(SNAPSHOT_PREFIX)
        ),
        "filename": _normalize_metadata(PurePosixPath(posix_path).name),
        "document_title": title,
        "heading_chain": (
            " > ".join(_normalize_metadata(str(row["text"])) for row in stack)
            if stack
            else MISSING
        ),
    }


def structural_prefix(path: str, content: str, chunk_start: int) -> str:
    fields = structural_fields(path, content, chunk_start)
    return (
        f"path: {fields['repository_path']}\n"
        f"filename: {fields['filename']}\n"
        f"title: {fields['document_title']}\n"
        f"headings: {fields['heading_chain']}\n\n"
    )


def _winning_chunk(
    document: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    text = str(document["content"])
    start = int(row["winning_chunk_start_codepoint"])
    end = int(row["winning_chunk_end_codepoint"])
    chunk = text[start:end]
    if sha256_bytes(chunk.encode("utf-8")) != row["winning_chunk_sha256"]:
        raise ValueError("v2 winning chunk does not match the exact corpus")
    return {
        "rank": row["rank"],
        "source_id": row["source_id"],
        "path": row["path"],
        "character_count": row["character_count"],
        "chunk_count": row["chunk_count"],
        "best_chunk_score": row["best_chunk_score"],
        "winning_chunk_index": row["winning_chunk_index"],
        "winning_chunk_start_codepoint": start,
        "winning_chunk_end_codepoint": end,
        "winning_chunk_sha256": row["winning_chunk_sha256"],
        "winning_chunk_text": chunk,
        "structural_fields": structural_fields(str(row["path"]), text, start),
    }


def diagnose(root: Path = ROOT) -> dict[str, Any]:
    v2_audit = v2.audit(root)
    if v2_audit["status"] != "observed_valid":
        raise ValueError("result-free diagnosis requires the valid v2 observation")
    result = read_json(root / V2_RESULT)
    corpus = read_json(root / CORPUS)
    repository = corpus.get("repository")
    corpus_rows = corpus.get("documents")
    if repository != "Liplus-Project/neuron-graph-rag" or not isinstance(
        corpus_rows, list
    ):
        raise ValueError("exact corpus identity mismatch")
    documents = {
        f"github:{repository}:{row['path']}": row
        for row in corpus_rows
        if isinstance(row, dict)
    }
    if len(documents) != 93:
        raise ValueError("exact corpus must contain 93 unique documents")
    expected_source = str(result["expected_source_id"])
    models = []
    for model in result["models"]:
        rows = model["documents"]
        gold_rows = [row for row in rows if row["source_id"] == expected_source]
        distractors = [row for row in rows if row["source_id"] != expected_source]
        if len(gold_rows) != 1 or not distractors:
            raise ValueError("v2 ranking lacks gold or distractor evidence")
        gold = gold_rows[0]
        top_distractor = min(distractors, key=lambda row: int(row["rank"]))
        models.append(
            {
                "kind": model["kind"],
                "model_id": model["model_id"],
                "revision": model["revision"],
                "correlations": _correlations(rows),
                "gold": _winning_chunk(documents[gold["source_id"]], gold),
                "top_distractor": _winning_chunk(
                    documents[top_distractor["source_id"]], top_distractor
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "result_free_diagnostic",
        "purpose": (
            "有効なv2 evidenceとexact corpusだけからchunk-count相関とwinning chunkの"
            "構造情報を再構成し、登録query再実行前のablation設計判断に使う。"
        ),
        "registered_query_execution_count": 0,
        "github_rag_request_count": 0,
        "shared_database_open_count": 0,
        "corpus_document_count": len(documents),
        "expected_source_id": expected_source,
        "inputs_sha256": {
            CORPUS.as_posix(): sha256_file(root / CORPUS),
            V2_RESULT.as_posix(): sha256_file(root / V2_RESULT),
        },
        "models": models,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def audit(root: Path = ROOT) -> dict[str, Any]:
    expected = diagnose(root)
    observed = read_json(root / DIAGNOSTIC)
    if observed != expected:
        raise ValueError("result-free diagnostic drifted from v2 evidence or exact corpus")
    manifest = read_json(root / MANIFEST)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "frozen_pre_registered_execution"
        or manifest.get("registered_query_execution_count") != 0
        or manifest.get("diagnostic_sha256") != sha256_file(root / DIAGNOSTIC)
    ):
        raise ValueError("proposed manifest identity or diagnostic binding mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "result_free_manifest_frozen",
        "registered_query_execution_count": 0,
        "corpus_document_count": observed["corpus_document_count"],
        "diagnostic_payload_sha256": observed["payload_sha256"],
        "proposed_arm_count": len(manifest["arms"]),
        "parent_judgment_required": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("diagnose", "freeze", "audit"))
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    if arguments.action == "audit":
        result = audit(arguments.root)
    else:
        result = diagnose(arguments.root)
        if arguments.action == "freeze":
            write_json_exclusive(arguments.root / DIAGNOSTIC, result)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

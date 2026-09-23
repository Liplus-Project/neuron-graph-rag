from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from neuron_graph_rag.cpu_shortlist_retrieval import (
    CpuShortlistRetriever,
    LocalPinnedE5,
    LocalPinnedV2M3,
)
from neuron_graph_rag.models import DocumentNode

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v1.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _nodes(corpus: dict) -> list[DocumentNode]:
    prefix = str(corpus["path_prefix"])
    repository = str(corpus["repository"])
    return [
        DocumentNode(
            f"github:{repository}:{row['path']}",
            str(row["content"]),
            {"path": str(row["path"]).removeprefix(prefix)},
        )
        for row in corpus["documents"]
    ]


def run(args: argparse.Namespace) -> dict:
    manifest = _read(MANIFEST)
    corpus_path = ROOT / manifest["corpus"]["path"]
    if _sha256(corpus_path) != manifest["corpus"]["sha256"]:
        raise ValueError("fixed benchmark corpus changed")
    corpus = _read(corpus_path)
    nodes = _nodes(corpus)
    if len(nodes) != manifest["corpus"]["document_count"]:
        raise ValueError("fixed benchmark document count changed")
    output = Path(args.output)
    cache = Path(args.cache)
    if output.exists() or cache.exists():
        raise FileExistsError("benchmark requires absent output and a cold cache path")
    retriever = CpuShortlistRetriever(
        cache,
        LocalPinnedE5(args.e5_snapshot, threads=4),
        LocalPinnedV2M3(args.reranker_snapshot, threads=4),
    )
    cold = retriever.update_cache(nodes)
    warm = retriever.update_cache(nodes)
    changed = list(nodes)
    changed[0] = DocumentNode(nodes[0].node_id, nodes[0].text + "\nindex update probe\n", nodes[0].metadata)
    update = retriever.update_cache(changed)
    restore = retriever.update_cache(nodes)
    cases = []
    runtime_gate = True
    quality_gate = True
    for case in manifest["queries"]:
        trace = retriever.search(case["query"], nodes, limit=len(nodes), timeout_seconds=60.0)
        ranks = {hit.node.node_id: rank for rank, hit in enumerate(trace.hits, 1)}
        expected_rank = ranks.get(case["expected_source_id"])
        case_quality = expected_rank is not None and expected_rank <= manifest["quality_gate"]["maximum_expected_rank"]
        case_runtime = trace.diagnostics["total_seconds"] <= manifest["runtime"]["warm_query_target_seconds"]
        quality_gate = quality_gate and case_quality
        runtime_gate = runtime_gate and case_runtime
        cases.append({
            "case_id": case["case_id"],
            "query": case["query"],
            "expected_source_id": case["expected_source_id"],
            "expected_rank": expected_rank,
            "top_source_ids": [hit.node.node_id for hit in trace.hits[:5]],
            "quality_gate_passed": case_quality,
            "runtime_gate_passed": case_runtime,
            "diagnostics": trace.diagnostics,
        })
    source_commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "status": "observed",
        "source_commit": source_commit,
        "manifest_sha256": _sha256(MANIFEST),
        "corpus_sha256": _sha256(corpus_path),
        "environment": {"cpu_threads": 4, "platform": os.name},
        "cache": {
            "cold": cold.__dict__ if hasattr(cold, "__dict__") else {name: getattr(cold, name) for name in cold.__slots__},
            "warm": warm.__dict__ if hasattr(warm, "__dict__") else {name: getattr(warm, name) for name in warm.__slots__},
            "one_document_update": update.__dict__ if hasattr(update, "__dict__") else {name: getattr(update, name) for name in update.__slots__},
            "restore": restore.__dict__ if hasattr(restore, "__dict__") else {name: getattr(restore, name) for name in restore.__slots__},
        },
        "cases": cases,
        "gates": {
            "warm_query_within_60_seconds": runtime_gate,
            "source_grounded_expected_rank_at_most_5": quality_gate,
            "user_guide_may_be_published": runtime_gate and quality_gate,
        },
    }
    _write_exclusive(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e5-snapshot", required=True)
    parser.add_argument("--reranker-snapshot", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["gates"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

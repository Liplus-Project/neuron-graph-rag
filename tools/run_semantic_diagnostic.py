"""CPU-only exploratory diagnostic; reads development evidence, never holdout queries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_graph_rag.engine import NeuronGraphRAG
from neuron_graph_rag.github_source import GitHubSnapshot, index_github_snapshot
from neuron_graph_rag.models import DocumentNode
from neuron_graph_rag.semantic_retrieval import (
    MultilingualE5, SemanticRetriever, attach_semantic_retriever,
)


def timed(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def ranking(scores):
    return sorted(scores, key=lambda key: (-scores[key], key))


def metrics(ids, case):
    expected = case["expected_source_ids"]
    ranks = [ids.index(key) + 1 if key in ids else None for key in expected]
    return {
        "expected_ranks": ranks,
        "recall_at_10": sum(r is not None and r <= 10 for r in ranks) / len(expected),
        "forbidden_top10": [x for x in case.get("forbidden_source_ids", []) if x in ids[:10]],
        "top10": ids[:10],
    }


def probes(retriever):
    docs = [
        DocumentNode("refund", "Customers can get their money back within thirty days of purchase."),
        DocumentNode("delivery", "Your parcel usually arrives three working days after dispatch."),
        DocumentNode("password", "To regain account access, request a password reset link by email."),
        DocumentNode("backup", "Backups are encrypted and retained for ninety days."),
    ]
    queries = [
        ("ja_en", "買った品物の代金を返してもらえる期限は？", "refund"),
        ("es_en", "¿Cuánto tarda en llegar mi paquete?", "delivery"),
        ("en_paraphrase", "I forgot my login secret. How can I sign in again?", "password"),
    ]
    results = []
    for name, query, expected in queries:
        scores, elapsed = timed(lambda: retriever.score(query, docs))
        ids = ranking(scores)
        results.append(dict(name=name, query=query, expected=expected,
                            rank=ids.index(expected) + 1, seconds=elapsed, scores=scores))
    long_doc = DocumentNode("long", "Office supplies are inventoried weekly. " * 250 +
                            "Emergency submarines use violet beacons to guide rescuers.")
    scores, elapsed = timed(lambda: retriever.score(
        "潜水艦が救助隊を導くために使う信号は？", [*docs, long_doc]))
    results.append(dict(name="long_tail", rank=ranking(scores).index("long") + 1,
                        chunks=len(retriever.backend.chunks(long_doc.text)), seconds=elapsed, scores=scores))
    query = "How long are encrypted backups retained?"
    before = retriever.score(query, docs)
    misses = retriever.cache_misses
    changed = [*docs[:-1], DocumentNode("backup", "Fresh bread is baked every morning.")]
    after = retriever.score(query, changed)
    results.append(dict(name="update_same_id", before=before["backup"], after=after["backup"],
                        new_cache_misses=retriever.cache_misses - misses))
    negative_docs = [DocumentNode("allowed", "The system supports encrypted offline backups."),
                     DocumentNode("forbidden", "The system supports unencrypted cloud backups.")]
    query = "Find encrypted offline backups, not unencrypted cloud backups"
    raw = retriever.score(query, negative_docs)
    with NeuronGraphRAG() as engine:
        attach_semantic_retriever(engine, retriever)
        for doc in negative_docs:
            engine.add_document(doc.node_id, doc.text)
        hits = engine.search(query, limit=2).hits
        results.append(dict(name="negation", query=query, dense_scores=raw,
                            dense_ranking=ranking(raw), hybrid_ranking=[h.node.node_id for h in hits]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; existing observations are never overwritten")
    backend = MultilingualE5(args.model_cache)
    _, load_seconds = timed(lambda: backend.query("load measurement"))
    retriever = SemanticRetriever(backend)
    # Development-only file already contains queries and expected/forbidden IDs.
    # Do not load the mixed development/holdout fixture or import its protocol runner.
    source = ROOT / "tests/evidence/github_retrieval_parity_v5/development.observed.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    snapshot = GitHubSnapshot.read(ROOT / "tests/fixtures/github_retrieval_parity_v4.corpus.json")
    results = []
    with tempfile.TemporaryDirectory(prefix="ngr-semantic230-") as directory:
        with NeuronGraphRAG(Path(directory) / "isolated.db") as engine:
            index_github_snapshot(engine, snapshot)
            nodes = engine.store.list_nodes()
            _, indexing = timed(lambda: retriever.score("index warmup", nodes))
            for case in previous["cases"]:
                query = case["query"]
                baseline, base_seconds = timed(lambda: engine.search(query, limit=10))
                dense, dense_seconds = timed(lambda: retriever.score(query, nodes))
                original = engine.dense_retriever
                attach_semantic_retriever(engine, retriever)
                hybrid, hybrid_seconds = timed(lambda: engine.search(query, limit=10))
                attach_semantic_retriever(engine, original)
                results.append(dict(
                    case_id=case["case_id"], cohort=case["cohort"], query=query,
                    baseline=metrics([h.node.node_id for h in baseline.hits], case),
                    semantic=metrics(ranking(dense), case),
                    semantic_hybrid=metrics([h.node.node_id for h in hybrid.hits], case),
                    seconds=dict(baseline=base_seconds, semantic=dense_seconds, hybrid=hybrid_seconds),
                ))
                print(case["case_id"], {k:results[-1][k]["expected_ranks"] for k in
                      ("baseline", "semantic", "semantic_hybrid")}, flush=True)
    probe_results = probes(retriever)
    files = {}
    for file in sorted(args.model_cache.rglob("*")):
        if file.is_file() and file.suffix in (".onnx", ".json"):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(file.relative_to(args.model_cache))] = digest.hexdigest()
    result = dict(
        purpose="known v5 development exploratory diagnostic; not unseen/confirmatory evaluation",
        model=backend.model_name, model_files_sha256=files,
        providers=backend._model.model.model.get_providers(), threads=backend.threads,
        versions={p:importlib.metadata.version(p) for p in ("fastembed", "onnxruntime", "tokenizers", "numpy")},
        python=platform.python_version(), platform=platform.platform(),
        corpus_documents=len(snapshot.documents), development_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seconds=dict(model_load_and_first_query=load_seconds, initial_index_and_query=indexing),
        cases=results, independent_probes=probe_results,
        cache=dict(hits=retriever.cache_hits, misses=retriever.cache_misses),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

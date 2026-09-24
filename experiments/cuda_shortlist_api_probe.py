"""Validate the public CUDA engine API against the fixed v3 corpus on a real GPU.

All output, cache and models must be supplied outside the repository. This is a
new observation, not a modification of the frozen v3 or exploratory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from neuron_graph_rag.cpu_shortlist_retrieval import LocalPinnedE5
from neuron_graph_rag.cuda_shortlist_retrieval import (
    CudaShortlistRetriever, LocalPinnedCudaV2M3, attach_cuda_shortlist_retriever,
)
from neuron_graph_rag.engine import NeuronGraphRAG

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v3.json"
FROZEN = ROOT / "tests/evidence/cpu_shortlist_benchmark_v3/observed.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "e5-snapshot", "reranker-snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    corpus_path = ROOT / manifest["corpus"]["path"]
    expected_files = {
        "e5_onnx_model": (args.e5_snapshot / "onnx/model.onnx", manifest["models"]["e5"]["onnx_model_sha256"]),
        "e5_tokenizer": (args.e5_snapshot / "tokenizer.json", manifest["models"]["e5"]["tokenizer_sha256"]),
        "v2_m3_weights": (args.reranker_snapshot / "model.safetensors", manifest["models"]["v2_m3"]["weights_sha256"]),
        "v2_m3_tokenizer": (args.reranker_snapshot / "tokenizer.json", manifest["models"]["v2_m3"]["tokenizer_sha256"]),
    }
    if args.e5_snapshot.name != manifest["models"]["e5"]["revision"] or args.reranker_snapshot.name != manifest["models"]["v2_m3"]["revision"]:
        raise ValueError("snapshot revision path mismatch")
    hashes = {name: sha256(path) for name, (path, _) in expected_files.items()}
    if any(hashes[name] != expected for name, (_, expected) in expected_files.items()):
        raise ValueError("model hash mismatch")
    if sha256(corpus_path) != manifest["corpus"]["sha256"]:
        raise ValueError("corpus hash mismatch")
    if not args.cache.is_file():
        raise FileNotFoundError("copy the frozen v3 E5 cache outside the repository first")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    prefix, repository = corpus["path_prefix"], corpus["repository"]
    retriever = CudaShortlistRetriever(
        args.cache, LocalPinnedE5(args.e5_snapshot, threads=4),
        LocalPinnedCudaV2M3(args.reranker_snapshot),
    )
    cases = []
    with NeuronGraphRAG(":memory:") as engine:
        attach_cuda_shortlist_retriever(engine, retriever)
        for row in corpus["documents"]:
            engine.add_document(
                f"github:{repository}:{row['path']}", row["content"],
                metadata={"path": row["path"].removeprefix(prefix)},
            )
        receipt = engine.update_cuda_shortlist_cache()
        expected_fingerprint = frozen["cache"]["cold"]["cache_fingerprint"]
        if receipt.cache_fingerprint != expected_fingerprint or receipt.unchanged != 93:
            raise ValueError("cache fingerprint or corpus identity mismatch")
        # Warm the lazy model before the three measured calls.
        engine.search_cuda_shortlist(manifest["queries"][0]["query"], limit=50, timeout_seconds=120)
        for fixed, previous in zip(manifest["queries"], frozen["cases"], strict=True):
            started = time.perf_counter()
            trace = engine.search_cuda_shortlist(fixed["query"], limit=50, timeout_seconds=120)
            wall_seconds = time.perf_counter() - started
            source_order = [hit.node.node_id for hit in trace.hits]
            rank = source_order.index(fixed["expected_source_id"]) + 1
            if trace.diagnostics["forward_pairs"] != previous["diagnostics"]["forward_pairs"]:
                raise ValueError("pair count differs from frozen v3")
            cases.append({
                "case_id": fixed["case_id"], "wall_seconds": wall_seconds,
                "expected_rank": rank, "pair_count": trace.diagnostics["forward_pairs"],
                "source_order": source_order, "diagnostics": trace.diagnostics,
            })
    retriever.close()
    allocated_after_close = int(torch.cuda.memory_allocated(0))
    result = {
        "protocol": "cuda-shortlist-public-api-v1", "status": "observed",
        "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "model_sha256": hashes, "corpus_sha256": sha256(corpus_path),
        "manifest_sha256": sha256(MANIFEST), "cache_fingerprint": receipt.cache_fingerprint,
        "gpu_name": torch.cuda.get_device_name(0), "cases": cases,
        "cuda_allocated_after_close_bytes": allocated_after_close,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"cases": [(case["case_id"], case["wall_seconds"], case["expected_rank"], case["pair_count"]) for case in cases], "gpu": result["gpu_name"]}))


if __name__ == "__main__":
    main()

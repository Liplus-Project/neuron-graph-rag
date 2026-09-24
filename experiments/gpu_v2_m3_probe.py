"""Exploratory, same-host CPU/CUDA reranker probe for the frozen v3 shortlist."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

from neuron_graph_rag.cpu_shortlist_retrieval import (
    CpuShortlistRetriever, LocalPinnedE5, _peak_rss_bytes,
)
from neuron_graph_rag.models import DocumentNode

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v3.json"
OBSERVED = ROOT / "tests/evidence/cpu_shortlist_benchmark_v3/observed.json"
BATCH_SIZE = 8
MAX_LENGTH = 512


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(args: argparse.Namespace, manifest: dict, observed: dict) -> dict:
    if manifest["protocol_id"] != "cpu-shortlist-retrieval-benchmark-v3":
        raise ValueError("unexpected fixture")
    corpus = ROOT / manifest["corpus"]["path"]
    if sha256(corpus) != manifest["corpus"]["sha256"]:
        raise ValueError("corpus hash mismatch")
    expected = {
        "e5_onnx_model": (args.e5_snapshot / "onnx/model.onnx", manifest["models"]["e5"]["onnx_model_sha256"]),
        "e5_tokenizer": (args.e5_snapshot / "tokenizer.json", manifest["models"]["e5"]["tokenizer_sha256"]),
        "v2_m3_weights": (args.reranker_snapshot / "model.safetensors", manifest["models"]["v2_m3"]["weights_sha256"]),
        "v2_m3_tokenizer": (args.reranker_snapshot / "tokenizer.json", manifest["models"]["v2_m3"]["tokenizer_sha256"]),
    }
    if args.e5_snapshot.name != manifest["models"]["e5"]["revision"] or args.reranker_snapshot.name != manifest["models"]["v2_m3"]["revision"]:
        raise ValueError("snapshot revision path mismatch")
    actual = {name: sha256(path) for name, (path, _) in expected.items()}
    if any(actual[name] != value for name, (_, value) in expected.items()):
        raise ValueError("model hash mismatch")
    if actual != observed["environment"]["model_sha256"]:
        raise ValueError("model differs from frozen v3 observation")
    if sha256(MANIFEST) != observed["manifest_sha256"]:
        raise ValueError("fixture differs from frozen v3 observation")
    return {"corpus_sha256": sha256(corpus), "manifest_sha256": sha256(MANIFEST), "model_sha256": actual}


class PairCapture:
    model_id = "BAAI/bge-reranker-v2-m3"
    revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [0.0] * len(pairs)


def make_cases(args: argparse.Namespace, manifest: dict, observed: dict) -> list[dict]:
    corpus = read_json(ROOT / manifest["corpus"]["path"])
    prefix, repository = corpus["path_prefix"], corpus["repository"]
    nodes = [DocumentNode(f"github:{repository}:{row['path']}", row["content"],
                          {"path": row["path"].removeprefix(prefix)}) for row in corpus["documents"]]
    if len(nodes) != manifest["corpus"]["document_count"]:
        raise ValueError("document count mismatch")
    capture = PairCapture()
    retriever = CpuShortlistRetriever(args.cache, LocalPinnedE5(args.e5_snapshot, threads=4), capture)
    if not args.cache.exists():
        receipt = retriever.update_cache(nodes)
        if receipt.cache_fingerprint != observed["cache"]["cold"]["cache_fingerprint"]:
            raise ValueError("new cache fingerprint mismatch")
    cases = []
    for fixed, previous in zip(manifest["queries"], observed["cases"], strict=True):
        capture.pairs.clear()
        trace = retriever.search(fixed["query"], nodes, limit=len(nodes), timeout_seconds=120)
        if trace.diagnostics["cache_fingerprint"] != previous["diagnostics"]["cache_fingerprint"]:
            raise ValueError("copied cache fingerprint mismatch")
        if len(capture.pairs) != previous["diagnostics"]["forward_pairs"]:
            raise ValueError("pair count differs from frozen v3")
        # The original search has stable candidate and chunk ordering. Its fake
        # logits leave stage-1 order intact, so replay each source's selected chunks.
        # Recover exact pair ownership through the cache's selected passage rows.
        # Candidate order is stage-1 order, distinct from fake rerank ordering.
        ordered = sorted(trace.hits, key=lambda hit: hit.stage1_rank)
        owners = [hit.node.node_id for hit in ordered for _ in hit.selected_chunks]
        if len(owners) != len(capture.pairs):
            raise ValueError("pair ownership mismatch")
        cases.append({"case_id": fixed["case_id"], "query": fixed["query"],
                      "expected_source_id": fixed["expected_source_id"],
                      "pairs": list(capture.pairs), "owners": owners,
                      "pair_count": len(owners)})
    return cases


def rank(owners: list[str], logits: list[float]) -> list[str]:
    grouped: dict[str, list[float]] = {}
    for owner, value in zip(owners, logits, strict=True):
        grouped.setdefault(owner, []).append(value)
    scores = {}
    for owner, values in grouped.items():
        maximum = max(values)
        scores[owner] = maximum + math.log(sum(math.exp(value - maximum) for value in values) / len(values))
    return sorted(scores, key=lambda owner: (-scores[owner], owner))


def score(model, tokenizer, torch, pairs: list[tuple[str, str]], device: str) -> list[float]:
    result = []
    with torch.inference_mode():
        for offset in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[offset:offset + BATCH_SIZE]
            encoded = tokenizer([a for a, _ in batch], [b for _, b in batch],
                                padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt")
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            result.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    if len(result) != len(pairs) or any(not math.isfinite(x) for x in result):
        raise ValueError("invalid reranker logits")
    return result


def run_device(device: str, cases: list[dict], snapshot: Path) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to torch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True).float().to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    cold_started = time.perf_counter()
    score(model, tokenizer, torch, cases[0]["pairs"], device)
    cold_seconds = time.perf_counter() - cold_started
    measured = []
    for case in cases:
        started = time.perf_counter()
        logits = score(model, tokenizer, torch, case["pairs"], device)
        seconds = time.perf_counter() - started
        ranked = rank(case["owners"], logits)
        measured.append({"case_id": case["case_id"], "seconds": seconds,
                         "pair_count": len(logits), "source_order": ranked,
                         "expected_rank": ranked.index(case["expected_source_id"]) + 1,
                         "logits": logits})
    memory = {"process_peak_rss_bytes": _peak_rss_bytes()}
    if device == "cuda":
        memory.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "device_name": torch.cuda.get_device_name(0),
                       "device_capability": torch.cuda.get_device_capability(0)})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"load_seconds": load_seconds, "cold_case_seconds": cold_seconds,
            "warm_cases": measured, "memory": memory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "e5-snapshot", "reranker-snapshot", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"protocol": "exploratory-v2-m3-cpu-cuda-v1", "status": "running",
              "settings": {"batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
                           "dtype": "float32", "cpu_threads": 4}}
    stage = "verify_inputs"
    failure = None
    try:
        import torch
        manifest, observed = read_json(MANIFEST), read_json(OBSERVED)
        result["identity"] = verify_inputs(args, manifest, observed)
        stage = "make_cases"
        cache_was_present = args.cache.exists()
        cases = make_cases(args, manifest, observed)
        result["identity"]["cache_copy_sha256" if cache_was_present else "new_cache_sha256"] = sha256(args.cache)
        result["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "tokenizers", "huggingface_hub", "onnxruntime", "numpy")},
            "torch_cuda": torch.version.cuda,
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
            ).stdout.strip(),
        }
        stage = "cpu"
        result["cpu"] = run_device("cpu", cases, args.reranker_snapshot)
        stage = "cuda"
        result["gpu"] = run_device("cuda", cases, args.reranker_snapshot)
        comparisons = []
        for left, right in zip(result["cpu"]["warm_cases"], result["gpu"]["warm_cases"], strict=True):
            comparisons.append({"case_id": left["case_id"], "speedup": left["seconds"] / right["seconds"],
                                "source_order_equal": left["source_order"] == right["source_order"],
                                "top5_equal": left["source_order"][:5] == right["source_order"][:5],
                                "max_abs_logit_delta": max(abs(a - b) for a, b in zip(left["logits"], right["logits"], strict=True))})
        result["comparisons"] = comparisons
        result["status"] = "observed"
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"stage": stage, "type": type(exc).__name__,
                             "message": str(exc), "traceback": traceback.format_exc()}
        failure = exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if failure is not None:
        raise RuntimeError(f"probe failed at {stage}; details saved to {args.output}") from failure
    print(json.dumps({"speedups": [row["speedup"] for row in result["comparisons"]],
                      "peak_vram_bytes": result["gpu"]["memory"]["peak_allocated_bytes"]}))


if __name__ == "__main__":
    main()

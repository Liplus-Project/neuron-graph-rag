"""Explicit CUDA v2-m3 shortlist retrieval; optional libraries load on first search."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .cpu_shortlist_retrieval import (
    MODEL_BATCH_SIZE,
    RERANKER_MODEL_ID,
    RERANKER_REVISION,
    CpuShortlistRetriever,
    CpuShortlistTrace,
    SearchCancelled,
    SearchTimeout,
)


class CudaUnavailableError(RuntimeError):
    """CUDA cannot be used by the installed PyTorch runtime."""


class LocalPinnedCudaV2M3:
    """Lazy, reusable FP32 CUDA reranker for a caller-supplied local snapshot."""

    model_id = RERANKER_MODEL_ID
    revision = RERANKER_REVISION

    def __init__(self, snapshot: str | Path, *, device: int = 0):
        if device < 0:
            raise ValueError("device must be non-negative")
        self.snapshot = Path(snapshot)
        self.device = device
        self._runtime: tuple[Any, Any, Any] | None = None

    def _torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install CUDA-enabled torch for CUDA shortlist retrieval") from exc
        if not torch.cuda.is_available() or self.device >= torch.cuda.device_count():
            raise CudaUnavailableError(f"CUDA device {self.device} is unavailable to PyTorch")
        torch.cuda.init()
        return torch

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is None:
            torch = self._torch()
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install transformers for CUDA shortlist retrieval") from exc
            required = ("config.json", "model.safetensors", "tokenizer.json")
            missing = [name for name in required if not (self.snapshot / name).is_file()]
            if missing:
                raise FileNotFoundError(f"pinned v2-m3 snapshot is incomplete: {', '.join(missing)}")
            tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.snapshot, local_files_only=True,
            ).float().to(f"cuda:{self.device}").eval()
            self._runtime = tokenizer, model, torch
        return self._runtime

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        tokenizer, model, torch = self._load()
        result: list[float] = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), MODEL_BATCH_SIZE):
                batch = pairs[offset:offset + MODEL_BATCH_SIZE]
                encoded = tokenizer(
                    [query for query, _ in batch], [passage for _, passage in batch],
                    padding=True, truncation=True, max_length=512, return_tensors="pt",
                )
                encoded = {name: tensor.to(f"cuda:{self.device}") for name, tensor in encoded.items()}
                result.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
        torch.cuda.synchronize(self.device)
        if any(not math.isfinite(value) for value in result):
            raise ValueError("CUDA reranker returned a non-finite logit")
        return result

    def close(self) -> None:
        """Drop this backend's model and release reusable PyTorch CUDA blocks."""
        if self._runtime is not None:
            torch = self._runtime[2]
            self._runtime = None
            torch.cuda.empty_cache()


class CudaShortlistRetriever(CpuShortlistRetriever):
    """The existing E5 shortlist with an explicitly selected CUDA reranker."""

    def __init__(self, cache_path: str | Path, e5_backend: Any, reranker: LocalPinnedCudaV2M3):
        if not isinstance(reranker, LocalPinnedCudaV2M3):
            raise TypeError("CUDA shortlist requires LocalPinnedCudaV2M3")
        super().__init__(cache_path, e5_backend, reranker)

    def search(self, query: str, nodes: Any, **kwargs: Any) -> CpuShortlistTrace:
        started = time.monotonic()
        timeout_seconds = kwargs.get("timeout_seconds", 60.0)
        cancelled = kwargs.get("cancelled")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cancelled is not None and cancelled():
            raise SearchCancelled("CUDA shortlist operation was cancelled")
        torch = self.reranker._torch()
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise SearchTimeout("CUDA shortlist operation exceeded its explicit timeout")
        # The first explicit search loads the model; its allocation is included.
        torch.cuda.reset_peak_memory_stats(self.reranker.device)
        kwargs["timeout_seconds"] = remaining
        trace = super().search(query, nodes, **kwargs)
        torch.cuda.synchronize(self.reranker.device)
        trace.diagnostics.update({
            "cuda_device": torch.cuda.get_device_name(self.reranker.device),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.reranker.device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.reranker.device)),
        })
        return trace

    def close(self) -> None:
        self.reranker.close()


def attach_cuda_shortlist_retriever(engine: Any, retriever: CudaShortlistRetriever) -> Any:
    """Add GPU-named methods only to the supplied engine instance."""
    if not isinstance(retriever, CudaShortlistRetriever):
        raise TypeError("retriever must be CudaShortlistRetriever")

    def update_cuda_shortlist_cache(**kwargs: Any) -> Any:
        return retriever.update_cache(engine.store.list_nodes(), **kwargs)

    def search_cuda_shortlist(query: str, **kwargs: Any) -> CpuShortlistTrace:
        return retriever.search(query, engine.store.list_nodes(), **kwargs)

    engine._cuda_shortlist_retriever = retriever
    engine.update_cuda_shortlist_cache = update_cuda_shortlist_cache
    engine.search_cuda_shortlist = search_cuda_shortlist
    return engine

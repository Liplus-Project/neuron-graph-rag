"""Opt-in CPU E5 retrieval. Importing this module never imports model libraries."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from .models import DocumentNode
from .retrieval import DenseRetriever


class MultilingualE5:
    """Lazy FastEmbed backend with explicit E5 prefixes and checked input lengths."""

    model_name = "intfloat/multilingual-e5-small"

    def __init__(self, cache_dir: str | Path, *, threads: int = 4):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.cache_dir = str(cache_dir)
        self.threads = threads
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                from fastembed.common.model_description import ModelSource, PoolingType
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise ImportError("Install fastembed>=0.8,<0.9 for semantic retrieval") from exc
            if not any(m["model"] == self.model_name for m in TextEmbedding.list_supported_models()):
                TextEmbedding.add_custom_model(
                    model=self.model_name, pooling=PoolingType.MEAN,
                    normalization=True, sources=ModelSource(hf=self.model_name),
                    dim=384, model_file="onnx/model.onnx",
                )
            model = TextEmbedding(
                model_name=self.model_name, cache_dir=self.cache_dir,
                threads=self.threads, providers=["CPUExecutionProvider"], cuda=False,
            )
            tokenizer = Tokenizer.from_str(model.model.tokenizer.to_str())
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
            self._model = model
        return self._model

    def _fits(self, text: str) -> bool:
        self._load()
        return len(self._tokenizer.encode(text).ids) <= 512

    def chunks(self, text: str) -> list[str]:
        # Overlapping character windows preserve all text, including the tail.
        # Token checks subdivide unusually token-dense multilingual windows.
        def split(part):
            if self._fits("passage: " + part):
                return [part]
            if len(part) < 2:
                raise ValueError("Cannot fit passage within model token limit")
            middle = len(part) // 2
            return split(part[:middle]) + split(part[middle:])

        parts = []
        for start in range(0, len(text), 850):
            parts.extend(split(text[start:start + 1000]))
            if start + 1000 >= len(text):
                break
        return parts or [""]

    def query(self, text: str):
        model = self._load()
        prefixed = "query: " + text
        if not self._fits(prefixed):
            raise ValueError("Query exceeds E5's 512-token limit; shorten it explicitly")
        return tuple(float(v) for v in next(iter(model.embed([prefixed]))))

    def passages(self, texts: Sequence[str]):
        model = self._load()
        prefixed = ["passage: " + text for text in texts]
        if not all(self._fits(text) for text in prefixed):
            raise ValueError("Passage exceeds E5's 512-token limit")
        return [tuple(float(v) for v in vector) for vector in model.embed(prefixed, batch_size=8)]


class SemanticRetriever:
    """Max chunk cosine with bounded, content-addressed in-memory document cache.

    Use one instance per backend configuration; cache is never shared across models.
    Like the SQLite engine, this instance is intended for serial use.
    """

    def __init__(self, backend: MultilingualE5, *, cache_size: int = 256):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.backend = backend
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        self._cache.clear()

    def score(self, query: str, nodes: Sequence[DocumentNode]) -> dict[str, float]:
        if not nodes or not query.strip():
            return {node.node_id: 0.0 for node in nodes}
        query_vector = self.backend.query(query)
        scores = {}
        for node in nodes:
            key = hashlib.sha256(node.text.encode("utf-8")).digest()
            if key in self._cache:
                self.cache_hits += 1
                vectors = self._cache.pop(key)
            else:
                self.cache_misses += 1
                chunks = self.backend.chunks(node.text)
                vectors = self.backend.passages(chunks)
                if len(vectors) != len(chunks):
                    raise ValueError("Backend returned an inconsistent passage count")
            self._cache[key] = vectors
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            scores[node.node_id] = max(
                (max(0.0, DenseRetriever._cosine(query_vector, v)) for v in vectors),
                default=0.0,
            )
        return scores


def attach_semantic_retriever(engine, retriever: SemanticRetriever):
    """Explicitly replace dense retrieval for both documents and judgment channels.

    Call before searching; existing sparse/graph weights and exclusion rules remain.
    No standard engine code or defaults are modified.
    """
    engine.dense_retriever = retriever
    engine.judgments.dense_retriever = retriever
    return engine

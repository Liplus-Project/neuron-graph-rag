"""Opt-in CPU two-stage retrieval with a persistent E5 document cache.

The module itself imports only the Python standard library. Model libraries are
loaded by the concrete backends on their first explicit operation.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .models import DocumentNode

E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CANDIDATE_K = 50
CHUNKS_PER_DOCUMENT = 2
NLME_TAU = 1.0
WINDOW_CODEPOINTS = 480
OVERLAP_CODEPOINTS = 80
MODEL_BATCH_SIZE = 8
CACHE_SCHEMA_VERSION = "1"
STRUCTURAL_CONTRACT = "path-title-atx-v1:codepoint-480-overlap-80"

_ATX_HEADING = re.compile(r"^(#{1,6}) (.*)$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_MISSING = "[missing]"


class CpuShortlistError(RuntimeError):
    """Base error for the explicit shortlist surface."""


class CacheIdentityError(CpuShortlistError):
    """The cache schema, model, or structural identity does not match."""


class CacheStaleError(CpuShortlistError):
    """The cache does not exactly represent the supplied corpus."""


class SearchCancelled(CpuShortlistError):
    """The caller cancelled an explicit cache or query operation."""


class SearchTimeout(CpuShortlistError):
    """The explicit timeout elapsed; no alternate ranking was returned."""


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    stage: str
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class CacheUpdateReceipt:
    added: int
    updated: int
    removed: int
    unchanged: int
    chunk_count: int
    e5_seconds: float
    total_seconds: float
    peak_rss_bytes: int
    cache_fingerprint: str


@dataclass(frozen=True, slots=True)
class CpuShortlistChunk:
    index: int
    start_codepoint: int
    end_codepoint: int
    e5_cosine: float


@dataclass(frozen=True, slots=True)
class CpuShortlistHit:
    node: DocumentNode
    score: float
    stage1_rank: int
    stage1_score: float
    selected_chunks: tuple[CpuShortlistChunk, ...]


@dataclass(frozen=True, slots=True)
class CpuShortlistTrace:
    query: str
    hits: tuple[CpuShortlistHit, ...]
    diagnostics: dict[str, Any]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _path(node: DocumentNode) -> str:
    raw = node.metadata.get("path", node.node_id)
    value = str(raw).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value or node.node_id


def _content_identity(node: DocumentNode) -> str:
    return hashlib.sha256(
        _canonical_json({"node_id": node.node_id, "path": _path(node), "text": node.text})
    ).hexdigest()


def _headings(text: str) -> list[tuple[int, str, int]]:
    result: list[tuple[int, str, int]] = []
    offset = 0
    fence_marker: str | None = None
    fence_length = 0
    for line_with_ending in text.splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker, fence_length = marker[0], len(marker)
            elif marker[0] == fence_marker and len(marker) >= fence_length:
                fence_marker, fence_length = None, 0
        elif fence_marker is None:
            heading = _ATX_HEADING.match(line)
            if heading:
                result.append((len(heading.group(1)), heading.group(2), offset))
        offset += len(line_with_ending)
    return result


def _normalized_field(value: str) -> str:
    value = " ".join(value.split())
    return value or _MISSING


def structural_prefix(path: str, content: str, chunk_start: int) -> str:
    """Return the #244 structural prefix without its frozen-snapshot path guard."""
    path = path.replace("\\", "/")
    headings = _headings(content)
    title = next(
        (_normalized_field(text) for level, text, _ in headings if level == 1),
        _MISSING,
    )
    stack: list[tuple[int, str, int]] = []
    for heading in headings:
        level, _, start = heading
        if start >= chunk_start:
            break
        stack = [row for row in stack if row[0] < level]
        stack.append(heading)
    chain = (
        " > ".join(_normalized_field(row[1]) for row in stack)
        if stack else _MISSING
    )
    return (
        f"path: {_normalized_field(path)}\n"
        f"filename: {_normalized_field(PurePosixPath(path).name)}\n"
        f"title: {title}\n"
        f"headings: {chain}\n\n"
    )


def structural_chunks(node: DocumentNode) -> list[tuple[int, int, int, str]]:
    """Project exact 480/80 code-point windows and attach structural prefixes."""
    result = []
    step = WINDOW_CODEPOINTS - OVERLAP_CODEPOINTS
    start = 0
    index = 0
    while True:
        end = min(len(node.text), start + WINDOW_CODEPOINTS)
        body = node.text[start:end]
        result.append((index, start, end, structural_prefix(_path(node), node.text, start) + body))
        if end >= len(node.text):
            break
        index += 1
        start += step
    return result


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("embedding must have non-zero norm")
    return tuple(value / norm for value in values)


def _centroid(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("document must have at least one chunk embedding")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding dimensions are inconsistent")
    return _normalize([sum(vector[i] for vector in vectors) / len(vectors) for i in range(dimension)])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions are inconsistent")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _pack(vector: Sequence[float]) -> bytes:
    values = tuple(float(value) for value in vector)
    return struct.pack(f"<{len(values)}f", *values)


def _unpack(blob: bytes, dimension: int) -> tuple[float, ...]:
    if dimension < 1 or len(blob) != dimension * 4:
        raise CacheStaleError("cached embedding shape is invalid")
    return tuple(struct.unpack(f"<{dimension}f", blob))


def _peak_rss_bytes() -> int:
    if os.name != "nt":
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if os.uname().sysname == "Darwin" else value * 1024

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


class LocalPinnedE5:
    """Lazy ONNX E5 backend bound to one caller-supplied local snapshot."""

    model_id = E5_MODEL_ID
    revision = E5_REVISION

    def __init__(self, snapshot: str | Path, *, threads: int = 4):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.snapshot = Path(snapshot)
        self.threads = threads
        self._runtime: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is None:
            try:
                import numpy as np
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise ImportError("Install numpy, onnxruntime, and tokenizers for CPU shortlist retrieval") from exc
            tokenizer_path = self.snapshot / "tokenizer.json"
            model_path = self.snapshot / "onnx" / "model.onnx"
            if not tokenizer_path.is_file() or not model_path.is_file():
                raise FileNotFoundError("pinned E5 snapshot is incomplete")
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
            self._runtime = np, tokenizer, session
        return self._runtime

    def _embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        np, tokenizer, session = self._load()
        result = []
        for offset in range(0, len(texts), MODEL_BATCH_SIZE):
            encodings = tokenizer.encode_batch(list(texts[offset:offset + MODEL_BATCH_SIZE]))
            rows = []
            for encoding in encodings:
                rows.append((encoding.ids[:512], encoding.attention_mask[:512], encoding.type_ids[:512]))
            width = max(len(row[0]) for row in rows)
            pad = tokenizer.token_to_id("<pad>")
            arrays = {
                name: np.asarray([row[index] + [fill] * (width - len(row[index])) for row in rows], dtype=np.int64)
                for name, index, fill in (("input_ids", 0, pad), ("attention_mask", 1, 0), ("token_type_ids", 2, 0))
            }
            hidden = session.run(None, arrays)[0]
            mask = arrays["attention_mask"][:, :, None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            result.extend(_normalize(row) for row in pooled.tolist())
        return result

    def embed_query(self, query: str) -> tuple[float, ...]:
        return self._embed(["query: " + query])[0]

    def embed_passages(self, passages: Sequence[str]) -> list[tuple[float, ...]]:
        return self._embed(["passage: " + passage for passage in passages])


class LocalPinnedV2M3:
    """Lazy local-only v2-m3 backend bound to the pinned revision snapshot."""

    model_id = RERANKER_MODEL_ID
    revision = RERANKER_REVISION

    def __init__(self, snapshot: str | Path, *, threads: int = 4):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.snapshot = Path(snapshot)
        self.threads = threads
        self._runtime: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install torch and transformers for CPU shortlist reranking") from exc
            if not (self.snapshot / "config.json").is_file():
                raise FileNotFoundError("pinned v2-m3 snapshot is incomplete")
            torch.set_num_threads(self.threads)
            torch.set_num_interop_threads(1)
            tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(self.snapshot, local_files_only=True)
            model.eval()
            self._runtime = tokenizer, model, torch
        return self._runtime

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        tokenizer, model, torch = self._load()
        if not pairs:
            return []
        encoded = tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits.reshape(-1).tolist()
        result = [float(value) for value in logits]
        if any(not math.isfinite(value) for value in result):
            raise ValueError("reranker returned a non-finite logit")
        return result


def download_cpu_shortlist_models(cache_dir: str | Path) -> dict[str, Path]:
    """Explicitly download both pinned snapshots; this is never called implicitly."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to download shortlist models") from exc
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "e5": Path(snapshot_download(E5_MODEL_ID, revision=E5_REVISION, cache_dir=root / "e5")),
        "v2-m3": Path(snapshot_download(RERANKER_MODEL_ID, revision=RERANKER_REVISION, cache_dir=root / "v2-m3")),
    }


class CpuShortlistRetriever:
    """Persistent E5 shortlist plus v2-m3 NLME reranking."""

    def __init__(self, cache_path: str | Path, e5_backend: Any, reranker: Any):
        self.cache_path = Path(cache_path)
        self.e5 = e5_backend
        self.reranker = reranker
        if (getattr(e5_backend, "model_id", None), getattr(e5_backend, "revision", None)) != (E5_MODEL_ID, E5_REVISION):
            raise ValueError("E5 backend identity is not the pinned revision")
        if (getattr(reranker, "model_id", None), getattr(reranker, "revision", None)) != (RERANKER_MODEL_ID, RERANKER_REVISION):
            raise ValueError("reranker identity is not the pinned revision")

    @staticmethod
    def _expected_metadata() -> dict[str, str]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "e5_model_id": E5_MODEL_ID,
            "e5_revision": E5_REVISION,
            "structural_contract": STRUCTURAL_CONTRACT,
        }

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(
              node_id TEXT PRIMARY KEY, path TEXT NOT NULL, content_sha256 TEXT NOT NULL,
              embedding BLOB NOT NULL, dimension INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks(
              node_id TEXT NOT NULL REFERENCES documents(node_id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL, start_codepoint INTEGER NOT NULL,
              end_codepoint INTEGER NOT NULL, passage TEXT NOT NULL,
              embedding BLOB NOT NULL, dimension INTEGER NOT NULL,
              PRIMARY KEY(node_id, chunk_index)
            );
            """
        )
        return connection

    def _ensure_identity(self, connection: sqlite3.Connection, mismatch: str) -> None:
        if mismatch not in {"error", "rebuild"}:
            raise ValueError("mismatch must be 'error' or 'rebuild'")
        expected = self._expected_metadata()
        actual = dict(connection.execute("SELECT key, value FROM metadata"))
        if not actual:
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())
        elif actual != expected:
            if mismatch == "error":
                raise CacheIdentityError(f"cache identity mismatch: expected={expected!r} actual={actual!r}")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM metadata")
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())

    @staticmethod
    def _check(deadline: float | None, cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise SearchCancelled("CPU shortlist operation was cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise SearchTimeout("CPU shortlist operation exceeded its explicit timeout")

    @staticmethod
    def _emit(progress: Callable[[ProgressUpdate], None] | None, stage: str, completed: int, total: int) -> None:
        if progress is not None:
            progress(ProgressUpdate(stage, completed, total))

    def _fingerprint(self, connection: sqlite3.Connection) -> str:
        rows = list(connection.execute("SELECT node_id, content_sha256 FROM documents ORDER BY node_id"))
        return hashlib.sha256(_canonical_json({"metadata": self._expected_metadata(), "documents": rows})).hexdigest()

    @staticmethod
    def _unique_nodes(nodes: Sequence[DocumentNode]) -> list[DocumentNode]:
        result = sorted(nodes, key=lambda node: node.node_id)
        if len({node.node_id for node in result}) != len(result):
            raise ValueError("node IDs must be unique")
        return result

    def update_cache(
        self, nodes: Sequence[DocumentNode], *, mismatch: str = "error",
        timeout_seconds: float | None = None,
        progress: Callable[[ProgressUpdate], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> CacheUpdateReceipt:
        started = time.monotonic()
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = None if timeout_seconds is None else started + timeout_seconds
        nodes = self._unique_nodes(nodes)
        with closing(self._connect()) as connection, connection:
            self._ensure_identity(connection, mismatch)
            existing = dict(connection.execute("SELECT node_id, content_sha256 FROM documents"))
            desired = {node.node_id: _content_identity(node) for node in nodes}
            removed_ids = sorted(set(existing) - set(desired))
            changed = [node for node in nodes if existing.get(node.node_id) != desired[node.node_id]]
            added = sum(node.node_id not in existing for node in changed)
            updated = len(changed) - added
            computed: list[tuple[DocumentNode, str, tuple[float, ...], list[tuple[int, int, int, str, tuple[float, ...]]]]] = []
            e5_started = time.monotonic()
            self._emit(progress, "cache", 0, len(changed))
            for number, node in enumerate(changed, 1):
                self._check(deadline, cancelled)
                chunks = structural_chunks(node)
                vectors: list[tuple[float, ...]] = []
                for offset in range(0, len(chunks), MODEL_BATCH_SIZE):
                    self._check(deadline, cancelled)
                    batch = chunks[offset:offset + MODEL_BATCH_SIZE]
                    produced = self.e5.embed_passages([row[3] for row in batch])
                    if len(produced) != len(batch):
                        raise ValueError("E5 backend returned an inconsistent passage count")
                    vectors.extend(_normalize(vector) for vector in produced)
                rows = [(*chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]
                computed.append((node, desired[node.node_id], _centroid(vectors), rows))
                self._emit(progress, "cache", number, len(changed))
            e5_seconds = time.monotonic() - e5_started
            self._check(deadline, cancelled)
            connection.executemany("DELETE FROM documents WHERE node_id = ?", ((node_id,) for node_id in removed_ids))
            for node, digest, centroid, chunks in computed:
                connection.execute("DELETE FROM documents WHERE node_id = ?", (node.node_id,))
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (node.node_id, _path(node), digest, _pack(centroid), len(centroid)),
                )
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ((node.node_id, index, start, end, passage, _pack(vector), len(vector)) for index, start, end, passage, vector in chunks),
                )
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            fingerprint = self._fingerprint(connection)
        return CacheUpdateReceipt(
            added, updated, len(removed_ids), len(nodes) - len(changed), chunk_count,
            e5_seconds, time.monotonic() - started, _peak_rss_bytes(), fingerprint,
        )

    def _validate_current_corpus(self, connection: sqlite3.Connection, nodes: Sequence[DocumentNode]) -> None:
        actual = dict(connection.execute("SELECT node_id, content_sha256 FROM documents"))
        expected = {node.node_id: _content_identity(node) for node in nodes}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(node_id for node_id in set(actual) & set(expected) if actual[node_id] != expected[node_id])
            raise CacheStaleError(f"cache is stale: missing={missing!r} extra={extra!r} changed={changed!r}")

    def search(
        self, query: str, nodes: Sequence[DocumentNode], *, limit: int = 5,
        timeout_seconds: float = 60.0,
        progress: Callable[[ProgressUpdate], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> CpuShortlistTrace:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        nodes = self._unique_nodes(nodes)
        if not nodes:
            raise ValueError("Cannot search an empty corpus")
        started = time.monotonic()
        deadline = started + timeout_seconds
        by_id = {node.node_id: node for node in nodes}
        with closing(self._connect()) as connection, connection:
            self._ensure_identity(connection, "error")
            self._validate_current_corpus(connection, nodes)
            self._check(deadline, cancelled)
            query_started = time.monotonic()
            query_vector = _normalize(self.e5.embed_query(query))
            query_seconds = time.monotonic() - query_started
            self._check(deadline, cancelled)
            stage1_started = time.monotonic()
            document_rows = list(connection.execute("SELECT node_id, embedding, dimension FROM documents ORDER BY node_id"))
            stage1 = []
            self._emit(progress, "e5-shortlist", 0, len(document_rows))
            for number, (node_id, blob, dimension) in enumerate(document_rows, 1):
                self._check(deadline, cancelled)
                stage1.append((str(node_id), _cosine(query_vector, _unpack(blob, int(dimension)))))
                self._emit(progress, "e5-shortlist", number, len(document_rows))
            stage1.sort(key=lambda row: (-row[1], row[0]))
            candidates = stage1[:CANDIDATE_K]
            stage1_ranks = {node_id: rank for rank, (node_id, _) in enumerate(stage1, 1)}
            stage1_seconds = time.monotonic() - stage1_started
            selected: dict[str, list[tuple[int, int, int, str, float]]] = {}
            pairs: list[tuple[str, str]] = []
            pair_nodes: list[str] = []
            for node_id, _ in candidates:
                rows = []
                for index, start, end, passage, blob, dimension in connection.execute(
                    "SELECT chunk_index, start_codepoint, end_codepoint, passage, embedding, dimension FROM chunks WHERE node_id = ? ORDER BY chunk_index",
                    (node_id,),
                ):
                    rows.append((int(index), int(start), int(end), str(passage), _cosine(query_vector, _unpack(blob, int(dimension)))))
                rows.sort(key=lambda row: (-row[4], row[0]))
                selected[node_id] = rows[:CHUNKS_PER_DOCUMENT]
                for row in selected[node_id]:
                    pairs.append((query, row[3]))
                    pair_nodes.append(node_id)
            rerank_started = time.monotonic()
            logits: list[float] = []
            self._emit(progress, "v2-m3", 0, len(pairs))
            for offset in range(0, len(pairs), MODEL_BATCH_SIZE):
                self._check(deadline, cancelled)
                batch = pairs[offset:offset + MODEL_BATCH_SIZE]
                produced = [float(value) for value in self.reranker.score_pairs(batch)]
                if len(produced) != len(batch) or any(not math.isfinite(value) for value in produced):
                    raise ValueError("reranker returned inconsistent or non-finite logits")
                logits.extend(produced)
                self._emit(progress, "v2-m3", len(logits), len(pairs))
            reranker_seconds = time.monotonic() - rerank_started
            self._check(deadline, cancelled)
            per_document: dict[str, list[float]] = {node_id: [] for node_id, _ in candidates}
            for node_id, logit in zip(pair_nodes, logits, strict=True):
                per_document[node_id].append(logit)
            ranked = []
            for node_id, stage1_score in candidates:
                scores = per_document[node_id]
                maximum = max(scores)
                score = maximum + math.log(sum(math.exp(value - maximum) for value in scores) / len(scores))
                ranked.append((node_id, score, stage1_score))
            ranked.sort(key=lambda row: (-row[1], row[0]))
            hits = tuple(
                CpuShortlistHit(
                    by_id[node_id], score, stage1_ranks[node_id], stage1_score,
                    tuple(CpuShortlistChunk(index, start, end, cosine) for index, start, end, _, cosine in selected[node_id]),
                )
                for node_id, score, stage1_score in ranked[:limit]
            )
            fingerprint = self._fingerprint(connection)
        total_seconds = time.monotonic() - started
        return CpuShortlistTrace(query, hits, {
            "candidate_k": CANDIDATE_K,
            "chunks_per_document": CHUNKS_PER_DOCUMENT,
            "nlme_tau": NLME_TAU,
            "forward_pairs": len(pairs),
            "e5_query_seconds": query_seconds,
            "e5_stage1_seconds": stage1_seconds,
            "v2_m3_seconds": reranker_seconds,
            "total_seconds": total_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "e5_model": {"id": E5_MODEL_ID, "revision": E5_REVISION},
            "reranker_model": {"id": RERANKER_MODEL_ID, "revision": RERANKER_REVISION},
            "cache_fingerprint": fingerprint,
        })


def attach_cpu_shortlist_retriever(engine: Any, retriever: CpuShortlistRetriever) -> Any:
    """Attach only the standalone opt-in surface; default retrieval is untouched."""
    engine._cpu_shortlist_retriever = retriever
    return engine

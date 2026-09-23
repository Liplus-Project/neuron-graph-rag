from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from neuron_graph_rag.cpu_shortlist_retrieval import (
    CANDIDATE_K,
    CHUNKS_PER_DOCUMENT,
    E5_MODEL_ID,
    E5_REVISION,
    RERANKER_MODEL_ID,
    RERANKER_REVISION,
    CacheIdentityError,
    CacheStaleError,
    CpuShortlistRetriever,
    SearchCancelled,
    attach_cpu_shortlist_retriever,
    structural_chunks,
    _peak_rss_bytes,
)
from neuron_graph_rag.engine import NeuronGraphRAG
from neuron_graph_rag.models import DocumentNode
from neuron_graph_rag.retrieval import DenseRetriever


class FakeE5:
    model_id = E5_MODEL_ID
    revision = E5_REVISION

    def __init__(self):
        self.passages = []

    @staticmethod
    def _vector(text):
        if "target" in text or "対象" in text:
            return (1.0, 0.0)
        if "near" in text:
            return (0.8, 0.6)
        return (0.0, 1.0)

    def embed_query(self, query):
        return (1.0, 0.0)

    def embed_passages(self, passages):
        self.passages.extend(passages)
        return [self._vector(passage) for passage in passages]


class FakeReranker:
    model_id = RERANKER_MODEL_ID
    revision = RERANKER_REVISION

    def __init__(self):
        self.pairs = []

    def score_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [3.0 if "winner" in passage else 1.0 for _, passage in pairs]


class CpuShortlistTests(unittest.TestCase):
    def test_peak_rss_measurement_is_positive(self):
        self.assertGreater(_peak_rss_bytes(), 0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name) / "shortlist.db"
        self.e5 = FakeE5()
        self.reranker = FakeReranker()
        self.retriever = CpuShortlistRetriever(self.cache, self.e5, self.reranker)

    def tearDown(self):
        self.temporary.cleanup()

    def test_import_and_backend_construction_are_lazy(self):
        subprocess.run(
            [sys.executable, "-c", "from neuron_graph_rag.cpu_shortlist_retrieval import LocalPinnedE5, LocalPinnedV2M3; import sys; a=LocalPinnedE5('missing'); b=LocalPinnedV2M3('missing'); assert a._runtime is None and b._runtime is None; assert 'onnxruntime' not in sys.modules and 'transformers' not in sys.modules"],
            check=True,
        )

    def test_structural_windows_cover_tail_and_track_headings(self):
        text = "# Title\n## Alpha\n" + "x" * 600
        chunks = structural_chunks(DocumentNode("n", text, {"path": "docs/a.md"}))
        self.assertEqual([(row[1], row[2]) for row in chunks], [(0, 480), (400, len(text))])
        self.assertIn("path: docs/a.md", chunks[0][3])
        self.assertIn("title: Title", chunks[0][3])
        self.assertIn("headings: Title > Alpha", chunks[1][3])
        self.assertTrue(chunks[-1][3].endswith(text[400:]))

    def test_incremental_update_only_embeds_changed_documents(self):
        nodes = [DocumentNode("a", "target"), DocumentNode("b", "noise")]
        first = self.retriever.update_cache(nodes)
        self.assertEqual((first.added, first.updated, first.removed, first.unchanged), (2, 0, 0, 0))
        seen = len(self.e5.passages)
        second = self.retriever.update_cache(nodes)
        self.assertEqual((second.added, second.updated, second.removed, second.unchanged), (0, 0, 0, 2))
        self.assertEqual(len(self.e5.passages), seen)
        third = self.retriever.update_cache([DocumentNode("a", "target changed"), DocumentNode("c", "noise")])
        self.assertEqual((third.added, third.updated, third.removed, third.unchanged), (1, 1, 1, 0))

    def test_identity_mismatch_errors_or_explicitly_rebuilds(self):
        self.retriever.update_cache([DocumentNode("a", "target")])
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE metadata SET value='wrong' WHERE key='e5_revision'")
        with self.assertRaises(CacheIdentityError):
            self.retriever.update_cache([DocumentNode("a", "target")])
        receipt = self.retriever.update_cache([DocumentNode("a", "target")], mismatch="rebuild")
        self.assertEqual(receipt.added, 1)

    def test_stale_cache_fails_without_ranking_fallback(self):
        self.retriever.update_cache([DocumentNode("a", "target")])
        with self.assertRaises(CacheStaleError):
            self.retriever.search("q", [DocumentNode("a", "changed")])

    def test_fixed_shortlist_nlme_tie_break_and_pair_bound(self):
        nodes = [DocumentNode(f"{index:02d}", "target winner" if index == 3 else "target near") for index in range(55)]
        self.retriever.update_cache(nodes)
        trace = self.retriever.search("q", nodes, limit=55)
        self.assertEqual(trace.hits[0].node.node_id, "03")
        self.assertEqual([hit.node.node_id for hit in trace.hits[1:4]], ["00", "01", "02"])
        self.assertEqual(len(trace.hits), CANDIDATE_K)
        self.assertLessEqual(trace.diagnostics["forward_pairs"], CANDIDATE_K * CHUNKS_PER_DOCUMENT)
        self.assertEqual(trace.diagnostics["nlme_tau"], 1.0)

    def test_cancel_and_progress_are_explicit(self):
        events = []
        with self.assertRaises(SearchCancelled):
            self.retriever.update_cache(
                [DocumentNode("a", "target")],
                cancelled=lambda: True,
                progress=events.append,
            )
        self.assertEqual(events[0].stage, "cache")

    def test_engine_opt_in_does_not_replace_default_dense_retrieval(self):
        with NeuronGraphRAG() as engine:
            default_dense = engine.dense_retriever
            self.assertFalse(hasattr(engine, "search_cpu_shortlist"))
            attach_cpu_shortlist_retriever(engine, self.retriever)
            engine.add_document("a", "target winner")
            engine.add_document("b", "noise")
            engine.update_cpu_shortlist_cache()
            trace = engine.search_cpu_shortlist("q")
            self.assertEqual(trace.hits[0].node.node_id, "a")
            self.assertIs(engine.dense_retriever, default_dense)
            self.assertIsInstance(engine.dense_retriever, DenseRetriever)
            with NeuronGraphRAG() as untouched:
                self.assertFalse(hasattr(untouched, "search_cpu_shortlist"))

    def test_source_grounded_fixture_is_independent_from_v5(self):
        fixture_path = Path(__file__).parent / "fixtures" / "cpu_shortlist_source_grounded.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        encoded = fixture_path.read_bytes()
        self.assertEqual(encoded.decode("utf-8").encode("utf-8"), encoded)
        self.assertNotIn("holdout", encoded.decode("utf-8").lower())
        self.assertNotIn("github_retrieval_parity_v5", encoded.decode("utf-8"))
        ids = {row["node_id"] for row in fixture["documents"]}
        self.assertTrue(all(row["expected_source_id"] in ids for row in fixture["queries"]))


if __name__ == "__main__":
    unittest.main()

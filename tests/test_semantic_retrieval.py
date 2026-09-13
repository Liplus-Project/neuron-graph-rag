from __future__ import annotations

import subprocess
import sys
import unittest

from neuron_graph_rag.engine import NeuronGraphRAG
from neuron_graph_rag.models import DocumentNode
from neuron_graph_rag.retrieval import DenseRetriever
from neuron_graph_rag.semantic_retrieval import (
    MultilingualE5, SemanticRetriever, attach_semantic_retriever,
)


class Backend:
    def __init__(self):
        self.seen = []

    def query(self, text):
        return (1., 0.)

    def chunks(self, text):
        return text.split("|")

    def passages(self, texts):
        self.seen.extend(texts)
        return [(1., 0.) if "target" in text else (0., 1.) for text in texts]


class SemanticTests(unittest.TestCase):
    def test_lazy_import_and_construction(self):
        subprocess.run([sys.executable, "-c", "from neuron_graph_rag.semantic_retrieval import MultilingualE5; "
                        "import sys; m=MultilingualE5('must-not-be-created'); "
                        "assert 'fastembed' not in sys.modules; assert m._model is None"], check=True)

    def test_content_cache_update_eviction_and_tail(self):
        backend = Backend()
        retrieval = SemanticRetriever(backend, cache_size=2)
        self.assertEqual(retrieval.score("q", [DocumentNode("same", "noise|target")])["same"], 1.)
        retrieval.score("q", [DocumentNode("new_id", "noise|target")])
        self.assertEqual(len(backend.seen), 2)
        self.assertEqual(retrieval.score("q", [DocumentNode("same", "noise")])["same"], 0.)
        retrieval.score("q", [DocumentNode("third", "other")])
        self.assertEqual(len(retrieval._cache), 2)
        retrieval.score("q", [DocumentNode("same", "noise|target")])
        self.assertEqual(retrieval.cache_misses, 4)
        retrieval.clear_cache()
        self.assertFalse(retrieval._cache)

    def test_attach_both_channels_and_default_unchanged(self):
        with NeuronGraphRAG() as engine:
            self.assertIsInstance(engine.dense_retriever, DenseRetriever)
            config = engine.config
            retriever = SemanticRetriever(Backend())
            attach_semantic_retriever(engine, retriever)
            self.assertIs(engine.dense_retriever, retriever)
            self.assertIs(engine.judgments.dense_retriever, retriever)
            self.assertIs(engine.config, config)
            engine.add_document("a", "noise")
            engine.add_document("b", "target")
            self.assertEqual(engine.search("q").hits[0].node.node_id, "b")

    def test_empty_does_not_load(self):
        model = MultilingualE5("unused")
        retriever = SemanticRetriever(model)
        self.assertEqual(retriever.score("q", []), {})
        self.assertEqual(retriever.score(" ", [DocumentNode("a", "t")]), {"a": 0.})
        self.assertIsNone(model._model)

    def test_checked_chunks_cover_all_text(self):
        model = MultilingualE5("unused")
        model._fits = lambda text: len(text) <= 80
        text = "日本語 and punctuation! " * 100
        chunks = model.chunks(text)
        self.assertTrue(all(len("passage: " + part) <= 80 for part in chunks))
        self.assertTrue(text.endswith(chunks[-1]))
        # Each original window is preserved by recursive splitting.
        self.assertIn(text[:1000], "".join(chunks))

    def test_prefixes_and_long_query_rejection(self):
        class Model:
            def embed(self, texts, **kwargs):
                self.seen = texts
                return iter([(1., 0.) for _ in texts])
        model = MultilingualE5("unused")
        stub = Model()
        model._load = lambda: stub
        model._fits = lambda text: len(text) < 30
        model.query("hello")
        self.assertEqual(stub.seen, ["query: hello"])
        model.passages(["world"])
        self.assertEqual(stub.seen, ["passage: world"])
        with self.assertRaises(ValueError):
            model.query("x" * 30)
        with self.assertRaises(ValueError):
            model.passages(["x" * 30])


if __name__ == "__main__":
    unittest.main()

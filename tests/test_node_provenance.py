from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.models import DocumentNode
from neuron_graph_rag.storage import SQLiteStore


def provenance(writer="Luna"):
    return {"writer": writer, "source_refs": ["https://github.com/Liplus-Project/neuron-graph-rag/issues/311"]}


class NodeProvenanceTest(unittest.TestCase):
    def test_persistence_replacement_and_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "nodes.sqlite"
            with NeuronGraphRAG(database) as rag:
                original = provenance()
                rag.add_document("a", "first", metadata={"node_provenance": original, "old": 1}, now=10)
                original["writer"] = "mutated"
                rag.store.connection.execute("UPDATE nodes SET last_used_at = 20 WHERE node_id = 'a'")
                rag.store.connection.commit()
                node = rag.add_document("a", "second", metadata={"new": 2}, now=30)
                self.assertEqual(node.metadata, {"new": 2, "node_provenance": provenance()})
                self.assertEqual(node, rag.store.get_node("a"))
                self.assertEqual(rag.store.node_usage("a"), {"created_at": 10, "last_used_at": 20})
                replaced = rag.add_document("a", "third", metadata={"node_provenance": provenance("Lay")})
                self.assertEqual(replaced.metadata, {"node_provenance": provenance("Lay")})
                legacy = rag.add_document("legacy", "old", metadata={"provenance": {"free": "judgment"}})
                self.assertNotIn("node_provenance", legacy.metadata)
                self.assertEqual(legacy.metadata["provenance"], {"free": "judgment"})
            with NeuronGraphRAG(database) as reopened:
                self.assertEqual(reopened.store.get_node("a"), replaced)

    def test_invalid_provenance_has_no_write(self):
        bad = [None, {}, [], "writer", {"writer": "", "source_refs": ["https://github.com/o/r/pull/1"]},
               {"writer": "  ", "source_refs": []}, {"writer": 1, "source_refs": []},
               {"writer": "a", "source_refs": "https://github.com/o/r/issues/1"},
               {"writer": "a", "source_refs": []},
               {"writer": "a", "source_refs": [None]},
               {**provenance(), "extra": True}]
        for url in ["http://github.com/o/r/issues/1", "https://github.com.evil/o/r/issues/1",
                    "https://github.com/o/r/commit/1", "https://github.com/o/r/pull/0",
                    "https://github.com/o/r/issues/1?q=x", "https://github.com/o/r/pull/1#x"]:
            bad.append({"writer": "a", "source_refs": [url]})
        with NeuronGraphRAG() as rag:
            original = rag.add_document("a", "original", metadata={"node_provenance": provenance()})
            usage = rag.store.node_usage("a")
            for value in bad:
                with self.subTest(value=value):
                    for node_id in ["a", "new"]:
                        with self.assertRaises(ValueError):
                            rag.add_document(node_id, "invalid", metadata={"node_provenance": value})
                    self.assertEqual(rag.store.get_node("a"), original)
                    self.assertEqual(rag.store.node_usage("a"), usage)
                    self.assertEqual(len(rag.store.list_nodes()), 1)
            with self.assertRaises(ValueError):
                rag.store.upsert_node(DocumentNode("a", "invalid", {"node_provenance": None}))
            valid = {"writer": "自己申告", "source_refs": ["https://github.com/o/r/pull/12/", "https://github.com/o/r/issues/9"]}
            self.assertEqual(rag.add_document("a", "valid", metadata={"node_provenance": valid}).metadata["node_provenance"], valid)

    def test_omitted_key_waits_for_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "shared.sqlite"
            with SQLiteStore(database) as store:
                store.upsert_node(DocumentNode("a", "old", {"node_provenance": provenance()}))
                ready = threading.Event()
                started = threading.Event()
                proceed = threading.Event()
                results, errors = [], []

                def reingest():
                    try:
                        with SQLiteStore(database) as other:
                            ready.set()
                            if not proceed.wait(5):
                                raise TimeoutError("writer did not proceed")
                            other.connection.set_trace_callback(lambda sql: started.set())
                            results.append(other.upsert_node(DocumentNode("a", "reingested", {"new": True})))
                    except Exception as error:
                        errors.append(error)

                worker = threading.Thread(target=reingest)
                worker.start()
                self.assertTrue(ready.wait(5))
                try:
                    with store.transaction() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        proceed.set()
                        self.assertTrue(started.wait(5))
                        store.upsert_node(DocumentNode("a", "concurrent", {"node_provenance": provenance("Lay")}))
                finally:
                    worker.join(10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(results[0], store.get_node("a"))
                self.assertEqual(results[0].metadata, {"new": True, "node_provenance": provenance("Lay")})

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import EngineConfig, NeuronGraphRAG


class EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NeuronGraphRAG(
            config=EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                entry_weight=0.4,
                graph_weight=0.6,
                seed_count=1,
                max_hops=2,
                hop_decay=0.8,
                activation_half_life_seconds=100.0,
            )
        )
        self.engine.add_document("source", "alpha source fact", confidence=0.73)
        self.engine.add_document("used", "distant used fact", confidence=0.82)
        self.engine.add_document("unused", "distant unused fact", confidence=0.91)
        self.engine.add_edge(
            "source", "used", "supports", weight=0.50, factuality=0.80
        )
        self.engine.add_edge(
            "source", "unused", "mentions", weight=0.40, factuality=0.70
        )

    def tearDown(self) -> None:
        self.engine.close()

    def test_search_explains_entry_and_graph_path(self) -> None:
        trace = self.engine.search("alpha", limit=3, now=1_000.0)
        used = next(hit for hit in trace.hits if hit.node.node_id == "used")

        self.assertEqual(used.sparse_score, 0.0)
        self.assertGreater(used.graph_activation, 0.0)
        self.assertEqual(used.paths[0].seed_id, "source")
        self.assertEqual(used.paths[0].steps[0].edge_type, "supports")
        self.assertIn("scores", used.explain())

    def test_retrieval_does_not_change_edge_weight(self) -> None:
        before = self.engine.store.edge("source", "used", "supports")
        self.engine.search("alpha", limit=3, now=1_000.0)
        after = self.engine.store.edge("source", "used", "supports")

        self.assertEqual(before.weight, after.weight)
        self.assertEqual(after.reinforced_count, 0)
        self.assertEqual(self.engine.store.count_feedback(), 0)

    def test_success_reinforces_only_the_used_path(self) -> None:
        trace = self.engine.search("alpha", limit=3, now=1_000.0)
        before_used = self.engine.store.edge("source", "used", "supports").weight
        before_unused = self.engine.store.edge(
            "source", "unused", "mentions"
        ).weight

        receipt = self.engine.record_success(
            trace.trace_id, ["used"], now=1_001.0
        )
        after_used = self.engine.store.edge("source", "used", "supports").weight
        after_unused = self.engine.store.edge(
            "source", "unused", "mentions"
        ).weight
        repeated = self.engine.search("alpha", limit=3, now=1_002.0)
        first_used = next(hit for hit in trace.hits if hit.node.node_id == "used")
        repeated_used = next(
            hit for hit in repeated.hits if hit.node.node_id == "used"
        )

        self.assertGreater(after_used, before_used)
        self.assertEqual(after_unused, before_unused)
        self.assertEqual(len(receipt.reinforced_edges), 1)
        self.assertGreater(
            repeated_used.graph_activation, first_used.graph_activation
        )
        self.assertEqual(self.engine.store.count_retrievals(), 2)
        self.assertEqual(self.engine.store.count_feedback(), 1)

    def test_activation_decays_without_changing_knowledge_axes(self) -> None:
        self.engine.search("alpha", limit=3, now=1_000.0)
        initial_activation = self.engine.activation("used", now=1_000.0)
        decayed_activation = self.engine.activation("used", now=1_100.0)
        node = self.engine.store.get_node("used")
        edge = self.engine.store.edge("source", "used", "supports")

        self.assertAlmostEqual(decayed_activation, initial_activation / 2.0)
        self.assertEqual(node.confidence, 0.82)
        self.assertEqual(edge.factuality, 0.80)
        self.assertEqual(edge.weight, 0.50)

    def test_success_must_reference_a_retrieved_node(self) -> None:
        trace = self.engine.search("alpha", limit=1, now=1_000.0)
        with self.assertRaises(ValueError):
            self.engine.record_success(trace.trace_id, ["used"], now=1_001.0)

    def test_successful_seed_does_not_reinforce_an_uncredited_edge(self) -> None:
        trace = self.engine.search("alpha", limit=3, now=1_000.0)
        before = self.engine.store.edge("source", "used", "supports").weight

        receipt = self.engine.record_success(
            trace.trace_id, ["source"], now=1_001.0
        )

        self.assertEqual(receipt.reinforced_edges, ())
        self.assertEqual(
            self.engine.store.edge("source", "used", "supports").weight,
            before,
        )
        self.assertEqual(self.engine.store.count_feedback(), 1)

    def test_feedback_and_reinforcement_roll_back_together(self) -> None:
        trace = self.engine.search("alpha", limit=3, now=1_000.0)
        self.engine.store.connection.execute(
            """
            DELETE FROM edges
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            ("source", "used", "supports"),
        )
        self.engine.store.connection.commit()

        with self.assertRaises(KeyError):
            self.engine.record_success(trace.trace_id, ["used"], now=1_001.0)

        self.assertEqual(self.engine.store.count_feedback(), 0)


class PersistenceTest(unittest.TestCase):
    def test_nodes_edges_and_history_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "rag.db"
            with NeuronGraphRAG(database) as engine:
                engine.add_document("a", "alpha")
                engine.add_document("b", "beta")
                engine.add_edge("a", "b", "supports", weight=0.4)
                trace = engine.search("alpha", limit=2, now=1_000.0)
                engine.record_success(trace.trace_id, ["b"], now=1_001.0)

            with NeuronGraphRAG(database) as reopened:
                self.assertEqual(len(reopened.store.list_nodes()), 2)
                self.assertGreater(
                    reopened.store.edge("a", "b", "supports").weight, 0.4
                )
                self.assertEqual(reopened.store.count_retrievals(), 1)
                self.assertEqual(reopened.store.count_feedback(), 1)


if __name__ == "__main__":
    unittest.main()

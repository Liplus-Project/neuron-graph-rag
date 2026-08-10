from __future__ import annotations

import unittest

from neuron_graph_rag import EngineConfig, NeuronGraphRAG


def _config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=1,
        sibling_feedback_normalization=1.0,
    )


def _default_config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=1,
    )


def _populate(engine: NeuronGraphRAG) -> None:
    engine.add_document("source", "alpha lexical source")
    engine.add_document("target", "distant relation target")
    engine.add_document("other", "unrelated sibling")
    engine.add_edge("source", "target", "mentions", weight=0.5)
    engine.add_edge("source", "other", "mentions", weight=0.25)


class SiblingRelationFeedbackTest(unittest.TestCase):
    def test_default_relation_feedback_keeps_uncredited_siblings_unchanged(self) -> None:
        with NeuronGraphRAG(config=_default_config()) as engine:
            _populate(engine)
            result = engine.search_channels("alpha", limit=3, now=1_000.0)
            sibling_before = engine.store.edge("source", "other", "mentions")

            receipt = engine.record_success(
                result.relation.trace_id, ["target"], now=1_001.0
            )

            self.assertEqual(engine.config.sibling_feedback_normalization, 0.0)
            self.assertEqual(receipt.normalized_sibling_edges, ())
            self.assertEqual(
                engine.store.edge("source", "other", "mentions"), sibling_before
            )

    def test_relation_feedback_normalizes_only_uncredited_siblings(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            engine.add_document("separate-source", "unrelated source")
            engine.add_document("separate-target", "unrelated target")
            engine.add_edge("separate-source", "separate-target", "mentions", weight=0.75)
            result = engine.search_channels("alpha", limit=3, now=1_000.0)
            lexical_ranks_before = [
                (hit.node.node_id, hit.rank) for hit in result.lexical.hits
            ]
            target_before = engine.store.edge("source", "target", "mentions")
            sibling_before = engine.store.edge("source", "other", "mentions")
            separate_before = engine.store.edge(
                "separate-source", "separate-target", "mentions"
            )

            receipt = engine.record_success(
                result.relation.trace_id, ["target"], now=1_001.0
            )
            repeated = engine.search_channels("alpha", limit=3, now=1_002.0)
            target_after = engine.store.edge("source", "target", "mentions")
            sibling_after = engine.store.edge("source", "other", "mentions")

            self.assertGreater(target_after.weight, target_before.weight)
            self.assertLess(sibling_after.weight, sibling_before.weight)
            self.assertEqual(sibling_after.reinforced_count, 0)
            self.assertEqual(
                engine.store.edge("separate-source", "separate-target", "mentions"),
                separate_before,
            )
            self.assertEqual(len(receipt.reinforced_edges), 1)
            self.assertEqual(len(receipt.normalized_sibling_edges), 1)
            self.assertEqual(receipt.normalized_sibling_edges[0].target_id, "other")
            self.assertEqual(
                [(hit.node.node_id, hit.rank) for hit in repeated.lexical.hits],
                lexical_ranks_before,
            )
            self.assertEqual(repeated.relation.hits[0].node.node_id, "target")

    def test_multiple_credited_edges_are_not_normalized_as_siblings(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            engine.add_document("also-credited", "another relation target")
            engine.add_edge("source", "also-credited", "mentions", weight=0.4)
            result = engine.search_channels("alpha", limit=4, now=1_000.0)
            target_before = engine.store.edge("source", "target", "mentions")
            also_credited_before = engine.store.edge(
                "source", "also-credited", "mentions"
            )
            sibling_before = engine.store.edge("source", "other", "mentions")

            receipt = engine.record_success(
                result.relation.trace_id, ["target", "also-credited"], now=1_001.0
            )

            self.assertGreater(
                engine.store.edge("source", "target", "mentions").weight,
                target_before.weight,
            )
            self.assertGreater(
                engine.store.edge("source", "also-credited", "mentions").weight,
                also_credited_before.weight,
            )
            self.assertLess(
                engine.store.edge("source", "other", "mentions").weight,
                sibling_before.weight,
            )
            self.assertEqual(
                {edge.target_id for edge in receipt.normalized_sibling_edges}, {"other"}
            )

    def test_opt_in_feedback_keeps_direct_source_result_first(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            direct_before = engine.search("alpha", limit=3, now=1_000.0)
            relation = engine.search_channels("alpha", limit=3, now=1_001.0)

            engine.record_success(
                relation.relation.trace_id, ["target"], now=1_002.0
            )
            direct_after = engine.search("alpha", limit=3, now=1_003.0)

            self.assertEqual(direct_before.hits[0].node.node_id, "source")
            self.assertEqual(direct_after.hits[0].node.node_id, "source")
            self.assertEqual(
                direct_after.hits[0].entry_score, direct_before.hits[0].entry_score
            )

    def test_lexical_and_zero_hop_feedback_do_not_normalize_siblings(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            result = engine.search_channels("alpha", limit=3, now=1_000.0)
            sibling_before = engine.store.edge("source", "other", "mentions")

            lexical = engine.record_success(
                result.lexical.trace_id, ["target"], now=1_001.0
            )
            direct = engine.search("alpha", limit=3, now=1_002.0)
            zero_hop = engine.record_success(
                direct.trace_id, ["source"], now=1_003.0
            )

            self.assertEqual(lexical.normalized_sibling_edges, ())
            self.assertEqual(zero_hop.normalized_sibling_edges, ())
            self.assertEqual(
                engine.store.edge("source", "other", "mentions"), sibling_before
            )

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import (
    EngineConfig,
    FeedbackLedger,
    NeuronGraphRAG,
    SourceUseEvent,
)
from neuron_graph_rag.engine import EngineConfig as ModuleEngineConfig


def _config(*, quorum: int = 3, sibling_ratio: float = 0.0) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=2,
        relation_feedback_evidence_quorum=quorum,
        sibling_feedback_normalization=sibling_ratio,
    )


def _populate(engine: NeuronGraphRAG) -> None:
    engine.add_document("source", "alpha lexical source")
    engine.add_document("target", "distant relation target")
    engine.add_document("sibling", "uncredited sibling")
    engine.add_document("other-source", "separate source")
    engine.add_document("other-target", "separate target")
    engine.add_edge("source", "target", "mentions", weight=0.5)
    engine.add_edge("source", "sibling", "mentions", weight=0.25)
    engine.add_edge("other-source", "other-target", "mentions", weight=0.75)


def _relation_success(
    engine: NeuronGraphRAG, event_index: int, node_id: str = "target"
):
    trace = engine.search_channels(
        "alpha", limit=5, now=1_000.0 + event_index
    ).relation
    return engine.record_success(
        trace.trace_id, [node_id], now=2_000.0 + event_index
    )


class EvidenceQuorumFeedbackTest(unittest.TestCase):
    def test_engine_module_exposes_quorum_config(self) -> None:
        self.assertEqual(
            ModuleEngineConfig(relation_feedback_evidence_quorum=2)
            .relation_feedback_evidence_quorum,
            2,
        )

    def test_quorum_three_activates_on_third_and_each_later_trace(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            initial = engine.store.edge("source", "target", "mentions")

            receipts = [_relation_success(engine, index) for index in range(1, 5)]
            states = [receipt.evidence[0] for receipt in receipts]

            self.assertEqual([state.count for state in states], [1, 2, 3, 4])
            self.assertEqual([state.quorum for state in states], [3, 3, 3, 3])
            self.assertEqual(
                [state.activated for state in states], [False, False, True, True]
            )
            self.assertEqual(receipts[0].reinforced_edges, ())
            self.assertEqual(receipts[1].reinforced_edges, ())
            self.assertEqual(len(receipts[2].reinforced_edges), 1)
            self.assertEqual(len(receipts[3].reinforced_edges), 1)
            stored = engine.store.edge("source", "target", "mentions")
            self.assertGreater(stored.weight, initial.weight)
            self.assertEqual(stored.reinforced_count, 2)
            self.assertEqual(engine.store.count_feedback_evidence(), 4)

    def test_same_trace_and_source_use_replays_do_not_duplicate_evidence(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            trace = engine.search("alpha", limit=5, now=1_000.0)
            first = engine.record_success(trace.trace_id, ["target"], now=1_001.0)
            duplicate = engine.record_success(
                trace.trace_id, ["target"], now=1_002.0
            )
            self.assertEqual(first.evidence[0].count, 1)
            self.assertFalse(first.evidence[0].activated)
            self.assertEqual(duplicate.evidence[0].count, 1)
            self.assertFalse(duplicate.evidence[0].activated)
            self.assertEqual(engine.store.count_feedback_evidence(), 1)

        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger = FeedbackLedger(engine)
            trace = engine.search("alpha", limit=5, now=1_000.0)
            events = (
                SourceUseEvent("target", "selected"),
                SourceUseEvent("target", "validated"),
                SourceUseEvent("target", "used"),
            )
            receipt = ledger.record_source_use(
                trace.trace_id, events, idempotency_key="quorum-source-use"
            )
            replay = ledger.record_source_use(
                trace.trace_id, events, idempotency_key="quorum-source-use"
            )
            duplicate_stage = ledger.record_source_use(
                trace.trace_id,
                [SourceUseEvent("target", "used")],
                idempotency_key="quorum-duplicate-stage",
            )
            self.assertEqual(replay, receipt)
            self.assertIsNotNone(receipt.feedback)
            self.assertEqual(receipt.feedback.evidence[0].count, 1)
            self.assertIsNone(duplicate_stage.feedback)
            self.assertEqual(engine.store.count_feedback_evidence(), 1)

    def test_default_quorum_one_preserves_first_update(self) -> None:
        self.assertEqual(EngineConfig().relation_feedback_evidence_quorum, 1)
        with NeuronGraphRAG(config=_config(quorum=1)) as engine:
            _populate(engine)
            before = engine.store.edge("source", "target", "mentions")
            receipt = _relation_success(engine, 1)
            after = engine.store.edge("source", "target", "mentions")

            self.assertEqual(len(receipt.reinforced_edges), 1)
            self.assertEqual(receipt.evidence[0].count, 1)
            self.assertEqual(receipt.evidence[0].quorum, 1)
            self.assertTrue(receipt.evidence[0].activated)
            self.assertGreater(after.weight, before.weight)
            self.assertEqual(after.reinforced_count, 1)

    def test_quorum_requires_a_positive_non_boolean_integer(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EngineConfig(relation_feedback_evidence_quorum=value)

    def test_evidence_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "persistent.sqlite"
            with NeuronGraphRAG(database, config=_config()) as engine:
                _populate(engine)
                first = _relation_success(engine, 1)
                self.assertEqual(first.evidence[0].count, 1)

            with NeuronGraphRAG(database, config=_config()) as reopened:
                second = _relation_success(reopened, 2)
                third = _relation_success(reopened, 3)
                self.assertEqual(second.evidence[0].count, 2)
                self.assertFalse(second.evidence[0].activated)
                self.assertEqual(third.evidence[0].count, 3)
                self.assertTrue(third.evidence[0].activated)
                self.assertEqual(
                    reopened.store.edge(
                        "source", "target", "mentions"
                    ).reinforced_count,
                    1,
                )

    def test_existing_database_migrates_without_rewriting_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.sqlite"
            with NeuronGraphRAG(database) as engine:
                _populate(engine)
                before = engine.store.edge("source", "target", "mentions")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("DROP TABLE relation_feedback_evidence")
                connection.commit()

            with NeuronGraphRAG(database, config=_config()) as migrated:
                self.assertEqual(len(migrated.store.list_nodes()), 5)
                self.assertEqual(
                    migrated.store.edge("source", "target", "mentions"), before
                )
                self.assertEqual(migrated.store.count_feedback_evidence(), 0)
                receipt = _relation_success(migrated, 1)
                self.assertEqual(receipt.evidence[0].count, 1)

    def test_lexical_zero_hop_and_uncredited_edges_add_no_evidence(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            channels = engine.search_channels("alpha", limit=5, now=1_000.0)
            lexical = engine.record_success(
                channels.lexical.trace_id, ["target"], now=1_001.0
            )
            direct = engine.search("alpha", limit=5, now=1_002.0)
            zero_hop = engine.record_success(
                direct.trace_id, ["source"], now=1_003.0
            )
            self.assertEqual(lexical.evidence, ())
            self.assertEqual(zero_hop.evidence, ())
            self.assertEqual(engine.store.count_feedback_evidence(), 0)

            sibling_before = engine.store.edge("source", "sibling", "mentions")
            other_before = engine.store.edge(
                "other-source", "other-target", "mentions"
            )
            relation = _relation_success(engine, 4)
            self.assertEqual(
                [(item.source_id, item.target_id) for item in relation.evidence],
                [("source", "target")],
            )
            self.assertEqual(
                engine.store.edge("source", "sibling", "mentions"), sibling_before
            )
            self.assertEqual(
                engine.store.edge("other-source", "other-target", "mentions"),
                other_before,
            )

    def test_sibling_normalization_waits_for_actual_activation(self) -> None:
        with NeuronGraphRAG(config=_config(sibling_ratio=1.0)) as engine:
            _populate(engine)
            target_before = engine.store.edge("source", "target", "mentions")
            sibling_before = engine.store.edge("source", "sibling", "mentions")
            other_before = engine.store.edge(
                "other-source", "other-target", "mentions"
            )

            first = _relation_success(engine, 1)
            second = _relation_success(engine, 2)
            self.assertEqual(first.normalized_sibling_edges, ())
            self.assertEqual(second.normalized_sibling_edges, ())
            self.assertEqual(
                engine.store.edge("source", "target", "mentions"), target_before
            )
            self.assertEqual(
                engine.store.edge("source", "sibling", "mentions"), sibling_before
            )

            third = _relation_success(engine, 3)
            target_after = engine.store.edge("source", "target", "mentions")
            sibling_after = engine.store.edge("source", "sibling", "mentions")
            self.assertTrue(third.evidence[0].activated)
            self.assertEqual(len(third.normalized_sibling_edges), 1)
            self.assertAlmostEqual(
                target_after.weight - target_before.weight,
                sibling_before.weight - sibling_after.weight,
            )
            self.assertEqual(
                engine.store.edge("other-source", "other-target", "mentions"),
                other_before,
            )

    def test_multi_edge_failure_rolls_back_evidence_weight_and_feedback(self) -> None:
        with NeuronGraphRAG(config=_config(quorum=1)) as engine:
            engine.add_document("a-source", "alpha source")
            engine.add_document("m-middle", "middle relation")
            engine.add_document("z-target", "target relation")
            engine.add_edge("a-source", "m-middle", "supports", weight=0.5)
            engine.add_edge("m-middle", "z-target", "supports", weight=0.5)
            trace = engine.search("alpha", limit=3, now=1_000.0)
            first_before = engine.store.edge("a-source", "m-middle", "supports")
            engine.store.connection.execute(
                """
                DELETE FROM edges
                WHERE source_id = ? AND target_id = ? AND edge_type = ?
                """,
                ("m-middle", "z-target", "supports"),
            )
            engine.store.connection.commit()

            with self.assertRaises(KeyError):
                engine.record_success(trace.trace_id, ["z-target"], now=1_001.0)

            self.assertEqual(
                engine.store.edge("a-source", "m-middle", "supports"), first_before
            )
            self.assertEqual(engine.store.count_feedback_evidence(), 0)
            self.assertEqual(engine.store.count_feedback(), 0)

    def test_sibling_failure_rolls_back_evidence_weight_and_feedback(self) -> None:
        with NeuronGraphRAG(config=_config(quorum=1)) as engine:
            _populate(engine)
            trace = engine.search("alpha", limit=5, now=1_000.0)
            target_before = engine.store.edge("source", "target", "mentions")

            with self.assertRaises(KeyError):
                engine.store.apply_success_feedback(
                    "injected-feedback",
                    trace.trace_id,
                    1_001.0,
                    ["target"],
                    [("source", "target", "mentions", 0.1, 2.0)],
                    [("source", (("source", "missing", "mentions"),), 1.0)],
                    evidence_quorum=1,
                )

            self.assertEqual(
                engine.store.edge("source", "target", "mentions"), target_before
            )
            self.assertEqual(engine.store.count_feedback_evidence(), 0)
            self.assertEqual(engine.store.count_feedback(), 0)


if __name__ == "__main__":
    unittest.main()

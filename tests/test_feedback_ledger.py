from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import (
    EngineConfig,
    FeedbackContractError,
    FeedbackLedger,
    NeuronGraphRAG,
    SourceUseEvent,
)


class FeedbackLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "ngr.sqlite"
        self.engine = NeuronGraphRAG(self.database)
        self.feedback = FeedbackLedger(self.engine)
        self.engine.add_document("decision", "cache invalidation decision")
        self.engine.add_document("implementation", "implementation details")
        self.engine.add_edge("decision", "implementation", "implemented_by", weight=0.7)
        self.trace = self.engine.search("cache invalidation", limit=2)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary.cleanup()

    def test_ordered_source_use_is_atomic_idempotent_and_reinforces_once(self) -> None:
        node_id = "implementation"
        before = self.engine.store.edge("decision", node_id, "implemented_by").weight
        events = (
            SourceUseEvent(node_id, "selected"),
            SourceUseEvent(node_id, "validated"),
            SourceUseEvent(node_id, "used"),
        )
        receipt = self.feedback.record_source_use(
            self.trace.trace_id, events, idempotency_key="answer-1"
        )
        after = self.engine.store.edge("decision", node_id, "implemented_by").weight
        self.assertEqual(receipt.newly_used_node_ids, (node_id,))
        self.assertIsNotNone(receipt.feedback)
        self.assertGreater(after, before)

        replay = self.feedback.record_source_use(
            self.trace.trace_id, events, idempotency_key="answer-1"
        )
        self.assertEqual(replay, receipt)
        self.assertEqual(
            self.engine.store.edge("decision", node_id, "implemented_by").weight,
            after,
        )
        self.assertEqual(self.engine.store.count_feedback(), 1)

    def test_invalid_batch_rolls_back_all_stage_changes(self) -> None:
        events = (
            SourceUseEvent("implementation", "selected"),
            SourceUseEvent("decision", "used"),
        )
        with self.assertRaises(FeedbackContractError) as raised:
            self.feedback.record_source_use(
                self.trace.trace_id, events, idempotency_key="invalid-1"
            )
        self.assertEqual(raised.exception.code, "invalid_stage_transition")
        self.assertEqual(self.engine.store.source_use_stages(self.trace.trace_id), {})
        self.assertEqual(self.engine.store.count_feedback(), 0)

    def test_non_used_stages_and_duplicate_used_do_not_reinforce(self) -> None:
        node_id = "implementation"
        before = self.engine.store.edge("decision", node_id, "implemented_by").weight
        selected = self.feedback.record_source_use(
            self.trace.trace_id,
            [SourceUseEvent(node_id, "selected")],
            idempotency_key="selected-1",
        )
        validated = self.feedback.record_source_use(
            self.trace.trace_id,
            [SourceUseEvent(node_id, "validated")],
            idempotency_key="validated-1",
        )
        self.assertIsNone(selected.feedback)
        self.assertIsNone(validated.feedback)
        self.assertEqual(self.engine.store.count_feedback(), 0)
        self.assertEqual(
            self.engine.store.edge("decision", node_id, "implemented_by").weight,
            before,
        )

        self.feedback.record_source_use(
            self.trace.trace_id,
            [SourceUseEvent(node_id, "used")],
            idempotency_key="used-1",
        )
        after = self.engine.store.edge("decision", node_id, "implemented_by").weight
        duplicate = self.feedback.record_source_use(
            self.trace.trace_id,
            [SourceUseEvent(node_id, "used")],
            idempotency_key="used-duplicate",
        )
        self.assertFalse(duplicate.events[0].changed)
        self.assertIsNone(duplicate.feedback)
        self.assertEqual(self.engine.store.count_feedback(), 1)
        self.assertEqual(
            self.engine.store.edge("decision", node_id, "implemented_by").weight,
            after,
        )

    def test_idempotency_conflict_does_not_change_ledger(self) -> None:
        self.feedback.record_source_use(
            self.trace.trace_id,
            [SourceUseEvent("implementation", "selected")],
            idempotency_key="shared-key",
        )
        with self.assertRaises(FeedbackContractError) as raised:
            self.feedback.record_source_use(
                self.trace.trace_id,
                [SourceUseEvent("decision", "selected")],
                idempotency_key="shared-key",
            )
        self.assertEqual(raised.exception.code, "idempotency_conflict")
        self.assertEqual(
            self.engine.store.source_use_stages(self.trace.trace_id),
            {"implementation": "selected"},
        )

    def test_reinforcement_rolls_back_if_ledger_write_fails(self) -> None:
        before = self.engine.store.edge(
            "decision", "implementation", "implemented_by"
        )
        events = [
            SourceUseEvent("implementation", "selected"),
            SourceUseEvent("implementation", "validated"),
            SourceUseEvent("implementation", "used"),
        ]
        with (
            patch.object(
                self.engine.store,
                "_save_idempotent_result",
                side_effect=RuntimeError("injected ledger failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected ledger failure"),
        ):
            self.feedback.record_source_use(
                self.trace.trace_id, events, idempotency_key="atomic-failure"
            )
        self.assertEqual(
            self.engine.store.edge("decision", "implementation", "implemented_by"),
            before,
        )
        self.assertEqual(self.engine.store.source_use_stages(self.trace.trace_id), {})
        self.assertEqual(self.engine.store.count_feedback(), 0)

    def test_delayed_outcome_requires_used_source_and_never_changes_weight(self) -> None:
        node_id = "implementation"
        with self.assertRaises(FeedbackContractError) as raised:
            self.feedback.record_outcome(
                self.trace.trace_id,
                [node_id],
                "confirmed",
                "verified later",
                idempotency_key="outcome-before-use",
            )
        self.assertEqual(raised.exception.code, "source_not_used")

        self.feedback.record_source_use(
            self.trace.trace_id,
            [
                SourceUseEvent(node_id, "selected"),
                SourceUseEvent(node_id, "validated"),
                SourceUseEvent(node_id, "used"),
            ],
            idempotency_key="use-before-outcome",
        )
        weight = self.engine.store.edge("decision", node_id, "implemented_by").weight
        receipt = self.feedback.record_outcome(
            self.trace.trace_id,
            [node_id],
            "corrected",
            "a detail was corrected",
            idempotency_key="outcome-1",
        )
        replay = self.feedback.record_outcome(
            self.trace.trace_id,
            [node_id],
            "corrected",
            "a detail was corrected",
            idempotency_key="outcome-1",
        )
        self.assertEqual(receipt, replay)
        self.assertFalse(receipt.reinforcement_applied)
        self.assertEqual(self.engine.store.count_outcomes(), 1)
        self.assertEqual(
            self.engine.store.edge("decision", node_id, "implemented_by").weight,
            weight,
        )


def _parity_config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=1,
        sibling_feedback_normalization=1.0,
    )


def _populate_parity_engine(engine: NeuronGraphRAG) -> None:
    engine.add_document("source", "alpha lexical source")
    engine.add_document("target", "distant relation target")
    engine.add_document("sibling", "unrelated sibling")
    engine.add_edge("source", "target", "mentions", weight=0.5)
    engine.add_edge("source", "sibling", "mentions", weight=0.25)


class FeedbackPlanParityTest(unittest.TestCase):
    def test_normal_credited_path_matches_direct_record_success(self) -> None:
        with (
            NeuronGraphRAG(config=_parity_config()) as direct,
            NeuronGraphRAG(config=_parity_config()) as ledger_engine,
        ):
            _populate_parity_engine(direct)
            _populate_parity_engine(ledger_engine)
            direct_trace = direct.search("alpha", limit=3, now=1_000.0)
            ledger_trace = ledger_engine.search("alpha", limit=3, now=1_000.0)

            direct_receipt = direct.record_success(
                direct_trace.trace_id, ["target"], now=1_001.0
            )
            ledger_receipt = FeedbackLedger(ledger_engine).record_source_use(
                ledger_trace.trace_id,
                [
                    SourceUseEvent("target", "selected"),
                    SourceUseEvent("target", "validated"),
                    SourceUseEvent("target", "used"),
                ],
                idempotency_key="normal-parity",
                now=1_001.0,
            )

            self.assertEqual(
                ledger_receipt.feedback.reinforced_edges,
                direct_receipt.reinforced_edges,
            )
            self.assertEqual(ledger_engine.store.list_edges(), direct.store.list_edges())

    def test_relation_sibling_normalization_matches_direct_record_success(self) -> None:
        with (
            NeuronGraphRAG(config=_parity_config()) as direct,
            NeuronGraphRAG(config=_parity_config()) as ledger_engine,
        ):
            _populate_parity_engine(direct)
            _populate_parity_engine(ledger_engine)
            direct_trace = direct.search_channels("alpha", limit=3, now=1_000.0)
            ledger_trace = ledger_engine.search_channels("alpha", limit=3, now=1_000.0)

            direct_receipt = direct.record_success(
                direct_trace.relation.trace_id, ["target"], now=1_001.0
            )
            ledger_receipt = FeedbackLedger(ledger_engine).record_source_use(
                ledger_trace.relation.trace_id,
                [
                    SourceUseEvent("target", "selected"),
                    SourceUseEvent("target", "validated"),
                    SourceUseEvent("target", "used"),
                ],
                idempotency_key="relation-parity",
                now=1_001.0,
            )

            self.assertEqual(
                ledger_receipt.feedback.reinforced_edges,
                direct_receipt.reinforced_edges,
            )
            self.assertEqual(
                ledger_receipt.feedback.normalized_sibling_edges,
                direct_receipt.normalized_sibling_edges,
            )
            self.assertEqual(ledger_engine.store.list_edges(), direct.store.list_edges())


if __name__ == "__main__":
    unittest.main()

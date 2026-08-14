from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import FeedbackLedger, SourceUseEvent
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG


def _config(*, sibling_ratio: float = 0.0) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=2,
        sibling_feedback_normalization=sibling_ratio,
        confirmed_outcome_reinforcement=True,
        confirmation_decay_ratio=0.5,
    )


def _populate(engine: NeuronGraphRAG) -> None:
    engine.add_document("source", "alpha lexical source")
    engine.add_document("target", "distant relation target")
    engine.add_document("sibling", "uncredited sibling")
    engine.add_document("other-source", "isolated origin")
    engine.add_document("other-target", "isolated destination")
    engine.add_edge("source", "target", "supports", weight=0.5)
    engine.add_edge("source", "sibling", "supports", weight=0.4)
    engine.add_edge("other-source", "other-target", "isolated", weight=0.8)


def _used_relation(
    engine: NeuronGraphRAG, event_index: int, *, node_id: str = "target"
) -> tuple[FeedbackLedger, str]:
    ledger = FeedbackLedger(engine)
    trace = engine.search_channels("alpha", limit=5, now=1_000.0 + event_index).relation
    receipt = ledger.record_source_use(
        trace.trace_id,
        [
            SourceUseEvent(node_id, "selected"),
            SourceUseEvent(node_id, "validated"),
            SourceUseEvent(node_id, "used"),
        ],
        idempotency_key=f"use-{event_index}-{node_id}",
        now=2_000.0 + event_index,
    )
    assert receipt.feedback is None
    return ledger, trace.trace_id


class ConfirmedOutcomeFeedbackTest(unittest.TestCase):
    def test_candidate_requires_explicit_decay_and_remains_default_off(self) -> None:
        default = EngineConfig()
        self.assertFalse(default.confirmed_outcome_reinforcement)
        self.assertIsNone(default.confirmation_decay_ratio)
        with self.assertRaises(ValueError):
            EngineConfig(confirmed_outcome_reinforcement=True)
        with self.assertRaises(ValueError):
            EngineConfig(confirmation_decay_ratio=0.5)
        for value in (0.0, 1.0, -0.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EngineConfig(
                    confirmed_outcome_reinforcement=True,
                    confirmation_decay_ratio=value,
                )

    def test_used_is_non_mutating_and_confirmations_diminish_per_edge(self) -> None:
        with NeuronGraphRAG(config=_config(sibling_ratio=1.0)) as engine:
            _populate(engine)
            target_initial = engine.store.edge("source", "target", "supports")
            sibling_initial = engine.store.edge("source", "sibling", "supports")
            deltas = []
            multipliers = []
            for event_index in range(1, 4):
                ledger, trace_id = _used_relation(engine, event_index)
                self.assertEqual(
                    engine.store.edge("source", "target", "supports").weight,
                    target_initial.weight + sum(deltas),
                )
                receipt = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    f"verified result {event_index}",
                    idempotency_key=f"confirmed-{event_index}",
                    now=3_000.0 + event_index,
                )
                self.assertTrue(receipt.reinforcement_applied)
                self.assertEqual(len(receipt.confirmations), 1)
                confirmation = receipt.confirmations[0]
                self.assertEqual(confirmation.confirmation_count, event_index)
                self.assertEqual(
                    [
                        (step.source_id, step.target_id)
                        for step in receipt.credited_paths[0].steps
                    ],
                    [("source", "target")],
                )
                deltas.append(confirmation.actual_delta)
                multipliers.append(confirmation.multiplier)

            self.assertEqual(multipliers, [1.0, 0.5, 0.25])
            self.assertGreater(deltas[0], deltas[1])
            self.assertGreater(deltas[1], deltas[2])
            self.assertAlmostEqual(deltas[1], deltas[0] * 0.5)
            self.assertAlmostEqual(deltas[2], deltas[0] * 0.25)
            target = engine.store.edge("source", "target", "supports")
            sibling = engine.store.edge("source", "sibling", "supports")
            self.assertEqual(target.reinforced_count, 3)
            self.assertAlmostEqual(target.weight - target_initial.weight, sum(deltas))
            self.assertAlmostEqual(sibling_initial.weight - sibling.weight, sum(deltas))
            self.assertEqual(
                engine.store.edge("other-source", "other-target", "isolated").weight,
                0.8,
            )

    def test_idempotency_same_trace_and_non_confirmed_outcomes_do_not_reinforce(
        self,
    ) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, trace_id = _used_relation(engine, 1)
            before = engine.store.edge("source", "target", "supports")
            first = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "verified result",
                idempotency_key="confirmed-first",
            )
            replay = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "verified result",
                idempotency_key="confirmed-first",
            )
            duplicate_trace = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "same trace repeated",
                idempotency_key="confirmed-same-trace",
            )
            corrected = ledger.record_outcome(
                trace_id,
                ["target"],
                "corrected",
                "later correction",
                idempotency_key="corrected",
            )
            self.assertEqual(replay, first)
            self.assertFalse(duplicate_trace.reinforcement_applied)
            self.assertEqual(duplicate_trace.confirmations, ())
            self.assertFalse(corrected.reinforcement_applied)
            self.assertEqual(engine.store.count_confirmations(), 1)
            after = engine.store.edge("source", "target", "supports")
            self.assertAlmostEqual(
                after.weight - before.weight, first.confirmations[0].actual_delta
            )

    def test_lexical_zero_hop_and_uncredited_nodes_do_not_change_edges(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger = FeedbackLedger(engine)
            channels = engine.search_channels("alpha", limit=5, now=1_000.0)
            direct = engine.search("alpha", limit=5, now=1_001.0)
            before = engine.store.list_edges()
            for index, (trace_id, node_id) in enumerate(
                ((channels.lexical.trace_id, "target"), (direct.trace_id, "source")),
                start=1,
            ):
                ledger.record_source_use(
                    trace_id,
                    [
                        SourceUseEvent(node_id, "selected"),
                        SourceUseEvent(node_id, "validated"),
                        SourceUseEvent(node_id, "used"),
                    ],
                    idempotency_key=f"nonrelation-use-{index}",
                )
                receipt = ledger.record_outcome(
                    trace_id,
                    [node_id],
                    "confirmed",
                    "confirmed without credited relation path",
                    idempotency_key=f"nonrelation-confirmed-{index}",
                )
                self.assertFalse(receipt.reinforcement_applied)
                self.assertEqual(receipt.confirmations, ())
            self.assertEqual(engine.store.list_edges(), before)
            self.assertEqual(engine.store.count_confirmations(), 0)

    def test_restart_migration_and_atomic_receipt_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "confirmed.sqlite"
            with NeuronGraphRAG(database, config=_config()) as engine:
                _populate(engine)
                ledger, trace_id = _used_relation(engine, 1)
                first = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "first persisted confirmation",
                    idempotency_key="persisted-first",
                )
                self.assertEqual(first.confirmations[0].confirmation_count, 1)

            with NeuronGraphRAG(database, config=_config()) as reopened:
                ledger, trace_id = _used_relation(reopened, 2)
                before = reopened.store.edge("source", "target", "supports")
                second = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "second persisted confirmation",
                    idempotency_key="persisted-second",
                )
                self.assertEqual(second.confirmations[0].confirmation_count, 2)
                self.assertEqual(second.confirmations[0].multiplier, 0.5)

                ledger, failing_trace = _used_relation(reopened, 3)
                before_failure = reopened.store.edge("source", "target", "supports")
                with (
                    patch.object(
                        reopened.store,
                        "_save_idempotent_result",
                        side_effect=RuntimeError("injected receipt failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "injected receipt failure"),
                ):
                    ledger.record_outcome(
                        failing_trace,
                        ["target"],
                        "confirmed",
                        "must roll back",
                        idempotency_key="atomic-confirmed",
                    )
                self.assertEqual(
                    reopened.store.edge("source", "target", "supports"),
                    before_failure,
                )
                self.assertEqual(reopened.store.count_confirmations(), 2)
                self.assertEqual(reopened.store.count_outcomes(), 2)
                self.assertGreater(before.weight, 0.5)

    def test_existing_database_adds_confirmation_tables_without_rewriting_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.sqlite"
            with NeuronGraphRAG(database) as engine:
                _populate(engine)
                before = engine.store.list_edges()
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("DROP TABLE confirmed_relation_feedback")
                connection.execute("DROP TABLE confirmed_edge_state")
                connection.execute("DROP TABLE confirmed_source_uses")
                connection.commit()

            with NeuronGraphRAG(database, config=_config()) as migrated:
                self.assertEqual(migrated.store.list_edges(), before)
                self.assertEqual(migrated.store.count_confirmations(), 0)
                ledger, trace_id = _used_relation(migrated, 1)
                receipt = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "migration retained the graph",
                    idempotency_key="migration-confirmed",
                )
                self.assertEqual(receipt.confirmations[0].confirmation_count, 1)

    def test_pre_candidate_used_trace_cannot_be_double_reinforced_after_policy_switch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "policy-switch.sqlite"
            with NeuronGraphRAG(database) as engine:
                _populate(engine)
                ledger = FeedbackLedger(engine)
                trace = engine.search_channels("alpha", limit=5, now=1_000.0).relation
                used = ledger.record_source_use(
                    trace.trace_id,
                    [
                        SourceUseEvent("target", "selected"),
                        SourceUseEvent("target", "validated"),
                        SourceUseEvent("target", "used"),
                    ],
                    idempotency_key="legacy-used",
                )
                self.assertIsNotNone(used.feedback)
                after_used = engine.store.edge("source", "target", "supports")

            with NeuronGraphRAG(database, config=_config()) as candidate:
                receipt = FeedbackLedger(candidate).record_outcome(
                    trace.trace_id,
                    ["target"],
                    "confirmed",
                    "old used policy must not be credited again",
                    idempotency_key="candidate-after-legacy-use",
                )
                self.assertFalse(receipt.reinforcement_applied)
                self.assertEqual(receipt.confirmations, ())
                self.assertEqual(
                    candidate.store.edge("source", "target", "supports"),
                    after_used,
                )


if __name__ == "__main__":
    unittest.main()

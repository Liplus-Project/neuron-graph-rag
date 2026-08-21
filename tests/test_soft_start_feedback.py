from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import FeedbackLedger, SourceUseEvent
from neuron_graph_rag.config_provenance import effective_config_provenance
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG


def _config(*, sibling_ratio: float = 0.0, soft_ratio: float = 0.25) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=2,
        feedback_learning_rate=0.2,
        sibling_feedback_normalization=sibling_ratio,
        soft_start_feedback_reinforcement=True,
        soft_start_feedback_ratio=soft_ratio,
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


def _used_relation(engine: NeuronGraphRAG, index: int) -> tuple[FeedbackLedger, str, object]:
    ledger = FeedbackLedger(engine)
    trace = engine.search_channels("alpha", limit=5, now=1_000.0 + index).relation
    receipt = ledger.record_source_use(
        trace.trace_id,
        [
            SourceUseEvent("target", "selected"),
            SourceUseEvent("target", "validated"),
            SourceUseEvent("target", "used"),
        ],
        idempotency_key=f"soft-use-{index}",
        now=2_000.0 + index,
    )
    return ledger, trace.trace_id, receipt


class SoftStartFeedbackTest(unittest.TestCase):
    def test_config_is_default_off_and_rejects_incompatible_values(self) -> None:
        default = EngineConfig()
        self.assertFalse(default.soft_start_feedback_reinforcement)
        self.assertIsNone(default.soft_start_feedback_ratio)
        default_feedback = effective_config_provenance(default)["effective_config"][
            "feedback"
        ]
        self.assertNotIn("soft_start_feedback_reinforcement", default_feedback)
        self.assertNotIn("soft_start_feedback_ratio", default_feedback)
        active_feedback = effective_config_provenance(_config())["effective_config"][
            "feedback"
        ]
        self.assertTrue(active_feedback["soft_start_feedback_reinforcement"])
        self.assertEqual(active_feedback["soft_start_feedback_ratio"], 0.25)
        for ratio in (0.0, 1.0, math.nan, math.inf, True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                EngineConfig(
                    soft_start_feedback_reinforcement=True,
                    soft_start_feedback_ratio=ratio,
                    confirmation_decay_ratio=0.5,
                )
        with self.assertRaises(ValueError):
            EngineConfig(soft_start_feedback_ratio=0.25)
        with self.assertRaises(ValueError):
            EngineConfig(
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
            )
        with self.assertRaises(ValueError):
            EngineConfig(
                confirmed_outcome_reinforcement=True,
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
            )
        with self.assertRaises(ValueError):
            EngineConfig(
                relation_feedback_evidence_quorum=2,
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
            )

    def test_used_provisional_then_confirmed_remainder_and_decay(self) -> None:
        with NeuronGraphRAG(config=_config(sibling_ratio=1.0)) as engine:
            _populate(engine)
            target_initial = engine.store.edge("source", "target", "supports")
            sibling_initial = engine.store.edge("source", "sibling", "supports")
            ledger, trace_id, used = _used_relation(engine, 1)
            self.assertIsNotNone(used.feedback)
            self.assertEqual(len(used.feedback.reinforced_edges), 1)
            provisional_delta = (
                used.feedback.reinforced_edges[0].new_weight
                - used.feedback.reinforced_edges[0].old_weight
            )
            self.assertGreater(provisional_delta, 0.0)
            self.assertEqual(
                engine.store.edge("source", "sibling", "supports"), sibling_initial
            )
            first = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "first confirmation",
                idempotency_key="soft-confirm-1",
            )
            base_increment = provisional_delta / 0.25
            self.assertEqual(first.confirmations[0].confirmation_count, 1)
            self.assertEqual(first.confirmations[0].multiplier, 0.75)
            self.assertAlmostEqual(
                provisional_delta + first.confirmations[0].actual_delta,
                base_increment,
            )
            self.assertAlmostEqual(
                sibling_initial.weight
                - engine.store.edge("source", "sibling", "supports").weight,
                first.confirmations[0].actual_delta,
            )

            multipliers = [first.confirmations[0].multiplier]
            for index in (2, 3):
                ledger, next_trace, next_used = _used_relation(engine, index)
                self.assertIsNotNone(next_used.feedback)
                self.assertEqual(next_used.feedback.reinforced_edges, ())
                confirmed = ledger.record_outcome(
                    next_trace,
                    ["target"],
                    "confirmed",
                    f"confirmation {index}",
                    idempotency_key=f"soft-confirm-{index}",
                )
                multipliers.append(confirmed.confirmations[0].multiplier)
            self.assertEqual(multipliers, [0.75, 0.5, 0.25])
            self.assertEqual(
                engine.store.edge("other-source", "other-target", "isolated").weight,
                0.8,
            )
            self.assertGreater(
                engine.store.edge("source", "target", "supports").weight,
                target_initial.weight,
            )

    def test_duplicates_negative_outcomes_and_nonrelation_paths_are_nonmutating(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, trace_id, used = _used_relation(engine, 1)
            replay = ledger.record_source_use(
                trace_id,
                [
                    SourceUseEvent("target", "selected"),
                    SourceUseEvent("target", "validated"),
                    SourceUseEvent("target", "used"),
                ],
                idempotency_key="soft-use-1",
            )
            self.assertEqual(replay, used)
            before_negative = engine.store.list_edges()
            negative = ledger.record_outcome(
                trace_id,
                ["target"],
                "corrected",
                "not supported",
                idempotency_key="soft-negative",
            )
            self.assertFalse(negative.reinforcement_applied)
            self.assertEqual(engine.store.list_edges(), before_negative)

            direct = engine.search("alpha", limit=5, now=4_000.0)
            lexical_used = ledger.record_source_use(
                direct.trace_id,
                [
                    SourceUseEvent("source", "selected"),
                    SourceUseEvent("source", "validated"),
                    SourceUseEvent("source", "used"),
                ],
                idempotency_key="soft-lexical-use",
            )
            self.assertIsNotNone(lexical_used.feedback)
            self.assertEqual(lexical_used.feedback.reinforced_edges, ())
            lexical_confirmed = ledger.record_outcome(
                direct.trace_id,
                ["source"],
                "confirmed",
                "direct source",
                idempotency_key="soft-lexical-confirm",
            )
            self.assertFalse(lexical_confirmed.reinforcement_applied)
            self.assertEqual(engine.store.list_edges(), before_negative)

    def test_restart_migration_and_atomic_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "soft-start.sqlite"
            with NeuronGraphRAG(database) as legacy:
                _populate(legacy)
                _, legacy_trace_id, _ = _used_relation(legacy, 0)
                before = legacy.store.list_edges()
            with NeuronGraphRAG(database, config=_config()) as engine:
                self.assertEqual(engine.store.list_edges(), before)
                legacy_confirmation = FeedbackLedger(engine).record_outcome(
                    legacy_trace_id,
                    ["target"],
                    "confirmed",
                    "pre-policy use remains audit-only",
                    idempotency_key="soft-pre-policy-confirm",
                )
                self.assertFalse(legacy_confirmation.reinforcement_applied)
                self.assertEqual(engine.store.list_edges(), before)
                with closing(sqlite3.connect(database)) as connection:
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        )
                    }
                self.assertIn("soft_start_edge_state", tables)
                self.assertIn("soft_start_relation_feedback", tables)

                ledger = FeedbackLedger(engine)
                failing_use_trace = engine.search_channels(
                    "alpha", limit=5, now=900.0
                ).relation
                before_use_failure = engine.store.edge(
                    "source", "target", "supports"
                )
                with (
                    patch.object(
                        engine.store,
                        "_save_idempotent_result",
                        side_effect=RuntimeError("injected use receipt failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "injected use receipt failure"),
                ):
                    ledger.record_source_use(
                        failing_use_trace.trace_id,
                        [
                            SourceUseEvent("target", "selected"),
                            SourceUseEvent("target", "validated"),
                            SourceUseEvent("target", "used"),
                        ],
                        idempotency_key="soft-atomic-use",
                    )
                self.assertEqual(
                    engine.store.edge("source", "target", "supports"),
                    before_use_failure,
                )

                ledger, trace_id, _ = _used_relation(engine, 1)
                before_failure = engine.store.edge("source", "target", "supports")
                with (
                    patch.object(
                        engine.store,
                        "_save_idempotent_result",
                        side_effect=RuntimeError("injected receipt failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "injected receipt failure"),
                ):
                    ledger.record_outcome(
                        trace_id,
                        ["target"],
                        "confirmed",
                        "must roll back",
                        idempotency_key="soft-atomic-confirm",
                    )
                self.assertEqual(
                    engine.store.edge("source", "target", "supports"), before_failure
                )
                self.assertEqual(engine.store.count_outcomes(), 1)

    def test_maximum_weight_and_restart_preserve_the_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "soft-start-restart.sqlite"
            config = _config()
            with NeuronGraphRAG(database, config=config) as engine:
                _populate(engine)
                ledger, trace_id, _ = _used_relation(engine, 1)
                first = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "first persisted confirmation",
                    idempotency_key="soft-persisted-first",
                )
                self.assertEqual(first.confirmations[0].confirmation_count, 1)
            with NeuronGraphRAG(database, config=config) as reopened:
                ledger, trace_id, _ = _used_relation(reopened, 2)
                second = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "second persisted confirmation",
                    idempotency_key="soft-persisted-second",
                )
                self.assertEqual(second.confirmations[0].confirmation_count, 2)
                self.assertEqual(second.confirmations[0].multiplier, 0.5)

            capped = EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                seed_count=1,
                max_hops=2,
                feedback_learning_rate=0.2,
                maximum_edge_weight=0.51,
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
            )
            with NeuronGraphRAG(config=capped) as engine:
                _populate(engine)
                ledger, trace_id, used = _used_relation(engine, 1)
                after_used = engine.store.edge("source", "target", "supports").weight
                self.assertLessEqual(after_used, 0.51)
                confirmed = ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "confirmed",
                    "capped confirmation",
                    idempotency_key="soft-capped-confirm",
                )
                self.assertGreaterEqual(confirmed.confirmations[0].new_weight, after_used)
                self.assertLessEqual(confirmed.confirmations[0].new_weight, 0.51)
                self.assertIsNotNone(used.feedback)


if __name__ == "__main__":
    unittest.main()

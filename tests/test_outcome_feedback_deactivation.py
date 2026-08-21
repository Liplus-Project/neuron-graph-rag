from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import FeedbackLedger, SourceUseEvent
from neuron_graph_rag.config_provenance import effective_config_provenance
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG


def _config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=2,
        feedback_learning_rate=0.2,
        sibling_feedback_normalization=1.0,
        soft_start_feedback_reinforcement=True,
        soft_start_feedback_ratio=0.25,
        confirmation_decay_ratio=0.5,
        outcome_driven_feedback_deactivation=True,
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
    engine: NeuronGraphRAG, index: int
) -> tuple[FeedbackLedger, str, object]:
    ledger = FeedbackLedger(engine)
    trace = engine.search_channels("alpha", limit=5, now=1_000.0 + index).relation
    receipt = ledger.record_source_use(
        trace.trace_id,
        [
            SourceUseEvent("target", "selected"),
            SourceUseEvent("target", "validated"),
            SourceUseEvent("target", "used"),
        ],
        idempotency_key=f"deactivation-use-{index}",
        now=2_000.0 + index,
    )
    return ledger, trace.trace_id, receipt


class OutcomeFeedbackDeactivationTest(unittest.TestCase):
    def test_config_is_default_off_and_requires_soft_start(self) -> None:
        default = EngineConfig()
        self.assertFalse(default.outcome_driven_feedback_deactivation)
        feedback = effective_config_provenance(default)["effective_config"]["feedback"]
        self.assertNotIn("outcome_driven_feedback_deactivation", feedback)
        active = effective_config_provenance(_config())["effective_config"]["feedback"]
        self.assertTrue(active["outcome_driven_feedback_deactivation"])
        with self.assertRaises(ValueError):
            EngineConfig(outcome_driven_feedback_deactivation=True)

    def test_correction_exactly_reverses_credited_and_sibling_mutations(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, trace_id, used = _used_relation(engine, 1)
            self.assertIsNotNone(used.feedback)
            confirmed = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "confirmed first",
                idempotency_key="deactivation-confirmed",
                now=3_000.0,
            )
            self.assertTrue(confirmed.reinforcement_applied)
            self.assertGreater(
                engine.store.edge("source", "target", "supports").weight, 0.5
            )
            self.assertLess(
                engine.store.edge("source", "sibling", "supports").weight, 0.4
            )

            corrected = ledger.record_outcome(
                trace_id,
                ["target"],
                "corrected",
                "credited answer was wrong",
                idempotency_key="deactivation-corrected",
                now=4_000.0,
            )
            self.assertTrue(corrected.deactivation_applied)
            self.assertEqual(len(corrected.reversed_contributions), 2)
            self.assertEqual(
                {item.contribution_kind for item in corrected.reversed_contributions},
                {"soft_start_provisional", "soft_start_confirmation"},
            )
            confirmation = next(
                item
                for item in corrected.reversed_contributions
                if item.contribution_kind == "soft_start_confirmation"
            )
            self.assertEqual(
                {mutation.mutation_role for mutation in confirmation.mutations},
                {"credited", "sibling"},
            )
            self.assertAlmostEqual(
                engine.store.edge("source", "target", "supports").weight, 0.5
            )
            self.assertAlmostEqual(
                engine.store.edge("source", "sibling", "supports").weight, 0.4
            )
            self.assertAlmostEqual(
                engine.store.edge("other-source", "other-target", "isolated").weight,
                0.8,
            )

            replay = ledger.record_outcome(
                trace_id,
                ["target"],
                "corrected",
                "credited answer was wrong",
                idempotency_key="deactivation-corrected",
                now=9_000.0,
            )
            self.assertEqual(replay, corrected)
            self.assertAlmostEqual(
                engine.store.edge("source", "target", "supports").weight, 0.5
            )

    def test_rollback_is_trace_scoped_and_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "deactivation.sqlite"
            with NeuronGraphRAG(database, config=_config()) as engine:
                _populate(engine)
                ledger, first_trace, _ = _used_relation(engine, 1)
                _, second_trace, _ = _used_relation(engine, 2)
                ledger.record_outcome(
                    second_trace,
                    ["target"],
                    "confirmed",
                    "second trace remains active",
                    idempotency_key="deactivation-second-confirmed",
                )
                before_rollback = engine.store.edge(
                    "source", "target", "supports"
                ).weight
                first_delta = engine.store.connection.execute(
                    "SELECT credited_delta FROM feedback_contributions "
                    "WHERE trace_id = ? AND active = 1",
                    (first_trace,),
                ).fetchone()[0]
            with NeuronGraphRAG(database, config=_config()) as engine:
                receipt = FeedbackLedger(engine).record_outcome(
                    first_trace,
                    ["target"],
                    "rolled_back",
                    "first use was rolled back",
                    idempotency_key="deactivation-rollback",
                )
                self.assertTrue(receipt.deactivation_applied)
                self.assertEqual(len(receipt.reversed_contributions), 1)
                self.assertAlmostEqual(
                    engine.store.edge("source", "target", "supports").weight,
                    before_rollback - first_delta,
                )
                active_second = engine.store.connection.execute(
                    "SELECT COUNT(*) FROM feedback_contributions "
                    "WHERE trace_id = ? AND active = 1",
                    (second_trace,),
                ).fetchone()[0]
                self.assertEqual(active_second, 1)

    def test_superseded_edge_is_dormant_until_saved_path_is_confirmed(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, trace_id, _ = _used_relation(engine, 1)
            superseded = ledger.record_outcome(
                trace_id,
                ["target"],
                "superseded",
                "relationship became stale",
                idempotency_key="deactivation-superseded",
            )
            self.assertTrue(superseded.deactivation_applied)
            self.assertTrue(engine.store.edge_is_dormant("source", "target", "supports"))
            self.assertNotIn(
                "target",
                {
                    edge.target_id
                    for edge in engine.store.outgoing_edges("source")
                },
            )

            confirmed = ledger.record_outcome(
                trace_id,
                ["target"],
                "confirmed",
                "saved path is current again",
                idempotency_key="deactivation-reactivated",
            )
            self.assertEqual(len(confirmed.reactivated_edges), 1)
            self.assertFalse(engine.store.edge_is_dormant("source", "target", "supports"))
            self.assertIn(
                "target",
                {
                    edge.target_id
                    for edge in engine.store.outgoing_edges("source")
                },
            )

    def test_duplicate_confirmation_does_not_reactivate_and_reversal_has_floor(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, first_trace, _ = _used_relation(engine, 1)
            ledger.record_outcome(
                first_trace,
                ["target"],
                "confirmed",
                "first independent confirmation",
                idempotency_key="deactivation-floor-confirmed",
            )
            _, saved_trace, _ = _used_relation(engine, 2)
            ledger.record_outcome(
                first_trace,
                ["target"],
                "superseded",
                "relationship became stale",
                idempotency_key="deactivation-floor-superseded",
            )
            duplicate_trace = ledger.record_outcome(
                first_trace,
                ["target"],
                "confirmed",
                "duplicate trace is not new evidence",
                idempotency_key="deactivation-floor-duplicate-confirmed",
            )
            self.assertEqual(duplicate_trace.reactivated_edges, ())
            self.assertTrue(engine.store.edge_is_dormant("source", "target", "supports"))
            independent = ledger.record_outcome(
                saved_trace,
                ["target"],
                "confirmed",
                "independent saved path is current",
                idempotency_key="deactivation-floor-independent-confirmed",
            )
            self.assertEqual(len(independent.reactivated_edges), 1)

            engine.store.connection.execute(
                "UPDATE edges SET weight = 0.49 "
                "WHERE source_id = 'source' AND target_id = 'target' "
                "AND edge_type = 'supports'"
            )
            engine.store.connection.commit()
            corrected = ledger.record_outcome(
                first_trace,
                ["target"],
                "corrected",
                "reverse without punitive floor crossing",
                idempotency_key="deactivation-floor-corrected",
            )
            self.assertTrue(corrected.deactivation_applied)
            self.assertGreaterEqual(
                engine.store.edge("source", "target", "supports").weight, 0.5
            )

    def test_unattributed_and_atomic_failures_do_not_partially_mutate(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            ledger, trace_id, _ = _used_relation(engine, 1)
            before = engine.store.list_edges()
            lexical = engine.search("alpha", limit=5, now=5_000.0)
            ledger.record_source_use(
                lexical.trace_id,
                [
                    SourceUseEvent("source", "selected"),
                    SourceUseEvent("source", "validated"),
                    SourceUseEvent("source", "used"),
                ],
                idempotency_key="deactivation-lexical-use",
            )
            unattributed = ledger.record_outcome(
                lexical.trace_id,
                ["source"],
                "corrected",
                "lexical result",
                idempotency_key="deactivation-lexical-correction",
            )
            self.assertFalse(unattributed.deactivation_applied)
            self.assertEqual(engine.store.list_edges(), before)

            with (
                patch.object(
                    engine.store,
                    "_save_idempotent_result",
                    side_effect=RuntimeError("injected deactivation receipt failure"),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "injected deactivation receipt failure"
                ),
            ):
                ledger.record_outcome(
                    trace_id,
                    ["target"],
                    "corrected",
                    "must roll back atomically",
                    idempotency_key="deactivation-atomic",
                )
            self.assertEqual(engine.store.list_edges(), before)
            self.assertEqual(
                engine.store.connection.execute(
                    "SELECT COUNT(*) FROM delayed_outcomes "
                    "WHERE summary = 'must roll back atomically'"
                ).fetchone()[0],
                0,
            )


if __name__ == "__main__":
    unittest.main()

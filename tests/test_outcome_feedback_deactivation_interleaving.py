from __future__ import annotations

import unittest

from neuron_graph_rag import FeedbackLedger, SourceUseEvent
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG


def _use_relation(
    engine: NeuronGraphRAG,
    ledger: FeedbackLedger,
    *,
    node_id: str,
    index: int,
) -> str:
    trace = engine.search_channels("alpha", limit=5, now=1_000.0 + index).relation
    ledger.record_source_use(
        trace.trace_id,
        [
            SourceUseEvent(node_id, "selected"),
            SourceUseEvent(node_id, "validated"),
            SourceUseEvent(node_id, "used"),
        ],
        idempotency_key=f"interleaved-use-{index}",
    )
    return trace.trace_id


class OutcomeFeedbackDeactivationInterleavingTest(unittest.TestCase):
    def test_sibling_then_credited_reversal_is_bounded_and_order_reversible(self) -> None:
        config = EngineConfig(
            sparse_weight=1.0,
            dense_weight=0.0,
            seed_count=1,
            max_hops=2,
            feedback_learning_rate=2.0,
            sibling_feedback_normalization=1.0,
            maximum_edge_weight=2.0,
            soft_start_feedback_reinforcement=True,
            soft_start_feedback_ratio=0.25,
            confirmation_decay_ratio=0.5,
            outcome_driven_feedback_deactivation=True,
        )
        with NeuronGraphRAG(config=config) as engine:
            engine.add_document("source", "alpha lexical source")
            engine.add_document("first", "first relation target")
            engine.add_document("second", "second relation target")
            engine.add_edge("source", "first", "supports", weight=0.5)
            engine.add_edge("source", "second", "supports", weight=1.9)
            ledger = FeedbackLedger(engine)

            first_trace = _use_relation(
                engine, ledger, node_id="first", index=1
            )
            ledger.record_outcome(
                first_trace,
                ["first"],
                "confirmed",
                "first edge normalizes its sibling",
                idempotency_key="interleaved-first-confirmed",
            )
            second_trace = _use_relation(
                engine, ledger, node_id="second", index=2
            )
            ledger.record_outcome(
                second_trace,
                ["second"],
                "confirmed",
                "sibling later becomes credited",
                idempotency_key="interleaved-second-confirmed",
            )

            ledger.record_outcome(
                first_trace,
                ["first"],
                "corrected",
                "reverse the earlier sibling normalization",
                idempotency_key="interleaved-first-corrected",
            )
            second = engine.store.edge("source", "second", "supports")
            self.assertAlmostEqual(second.weight, 2.0)
            self.assertLessEqual(second.weight, config.maximum_edge_weight)

            ledger.record_outcome(
                second_trace,
                ["second"],
                "corrected",
                "reverse the later credited contribution",
                idempotency_key="interleaved-second-corrected",
            )
            self.assertAlmostEqual(
                engine.store.edge("source", "second", "supports").weight, 1.9
            )
            self.assertAlmostEqual(
                engine.store.edge("source", "first", "supports").weight, 0.5
            )


if __name__ == "__main__":
    unittest.main()

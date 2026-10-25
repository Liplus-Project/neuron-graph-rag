from __future__ import annotations

import unittest

from neuron_graph_rag import EngineConfig, NeuronGraphRAG, PrecisionControl
from neuron_graph_rag.models import ActivationPath, DocumentNode, PathStep, SearchHit
from neuron_graph_rag.precision_control import apply_precision_control


class PrecisionControlMechanicsTest(unittest.TestCase):
    def _engine(self, control: PrecisionControl | None = None) -> NeuronGraphRAG:
        engine = NeuronGraphRAG(
            config=EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                entry_weight=0.5,
                graph_weight=0.5,
                seed_count=1,
                max_hops=1,
                precision_control=control,
            )
        )
        engine.add_document("source", "alpha source", metadata={"path": "source.md"})
        engine.add_document(
            "linked", "unrelated linked", metadata={"path": "linked.md"}
        )
        engine.add_document("noise", "unrelated noise", metadata={"path": "noise.md"})
        engine.add_edge("source", "linked", "informs", weight=0.8)
        return engine

    def test_default_search_and_explicit_none_are_exactly_equivalent(self) -> None:
        with self._engine() as default, self._engine(None) as explicit_none:
            left = default.search("alpha", limit=3, now=100.0)
            right = explicit_none.search("alpha", limit=3, now=100.0)
        self.assertEqual(left.hits, right.hits)
        self.assertEqual(left.diagnostics, right.diagnostics)
        self.assertNotIn("precision_control", left.hits[0].explain())
        self.assertNotIn("precision_control", left.diagnostics)

    def test_opt_in_filters_after_ranking_without_mutating_scores_or_state(
        self,
    ) -> None:
        control = PrecisionControl("floor", minimum_final_score=0.2)
        with self._engine() as baseline, self._engine(control) as filtered:
            before_edge = filtered.store.list_edges()
            baseline_trace = baseline.search("alpha", limit=3, now=100.0)
            filtered_trace = filtered.search("alpha", limit=3, now=100.0)
            after_edge = filtered.store.list_edges()
            baseline_activation = {
                node.node_id: baseline.store.activation(node.node_id)
                for node in baseline.store.list_nodes()
            }
            filtered_activation = {
                node.node_id: filtered.store.activation(node.node_id)
                for node in filtered.store.list_nodes()
            }
            feedback_count = filtered.store.count_feedback()
        baseline_scores = {
            hit.node.node_id: hit.final_score for hit in baseline_trace.hits
        }
        for hit in filtered_trace.hits:
            self.assertEqual(hit.final_score, baseline_scores[hit.node.node_id])
        self.assertEqual(before_edge, after_edge)
        self.assertEqual(baseline_activation, filtered_activation)
        self.assertEqual(feedback_count, 0)
        self.assertIn("precision_control", filtered_trace.diagnostics)

    def test_explanation_is_recomputable_and_keeps_source_and_relation_path(
        self,
    ) -> None:
        path = ActivationPath(
            seed_id="seed",
            contribution=0.4,
            steps=(PathStep("seed", "target", "informs", 0.8, 1.0),),
        )
        hit = SearchHit(
            node=DocumentNode("target", "text", {"path": "target.md"}),
            sparse_score=0.2,
            dense_score=0.1,
            entry_score=0.3,
            graph_activation=0.4,
            final_score=0.6,
            paths=(path,),
            normalized_graph_activation=0.5,
        )
        lower = SearchHit(
            node=DocumentNode("lower", "text", {"path": "lower.md"}),
            sparse_score=0.1,
            dense_score=0.0,
            entry_score=0.1,
            graph_activation=0.0,
            final_score=0.2,
        )
        control = PrecisionControl(
            "combined",
            minimum_final_score=0.15,
            minimum_top_score_ratio=0.4,
            maximum_top_score_margin=0.4,
            require_entry_graph_signal_agreement=True,
        )
        annotated, accepted = apply_precision_control((hit, lower), control)
        explanation = annotated[0].explain()
        decision = explanation["precision_control"]
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["pre_filter_rank"], 1)
        self.assertEqual(decision["pre_filter_score"], hit.final_score)
        self.assertEqual(
            decision["top_score_ratio"],
            decision["pre_filter_score"] / decision["top_score"],
        )
        self.assertEqual(
            decision["top_score_margin"],
            decision["top_score"] - decision["pre_filter_score"],
        )
        self.assertEqual(explanation["paths"][0]["steps"][0]["edge_type"], "informs")
        self.assertEqual(decision["source_provenance"]["path"], "target.md")
        self.assertEqual(accepted[0].node.metadata["path"], "target.md")
        self.assertFalse(annotated[1].precision_control["accepted"])

    def test_invalid_or_empty_controls_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            PrecisionControl("empty")
        with self.assertRaises(ValueError):
            PrecisionControl("bad", minimum_final_score=1.1)
        with self.assertRaises(ValueError):
            PrecisionControl.from_mapping({"candidate_id": "bad", "query": "no"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from neuron_graph_rag.engine import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.experiment import _select_anchored_development, read_manifest
from tools.acquire_d1_fixture import assert_connected
from tools.audit_anchored_fixture import build_audit


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_anchored_hybrid_experiment.manifest.json"
DEVELOPMENT = FIXTURES / "d1_liplus_anchored_hybrid_development.json"
DEVELOPMENT_GOLD = FIXTURES / "d1_liplus_anchored_hybrid_development.gold.json"
HOLDOUT = FIXTURES / "d1_liplus_anchored_hybrid_holdout.json"
HOLDOUT_GOLD = FIXTURES / "d1_liplus_anchored_hybrid_holdout.gold.json"
AUDIT = FIXTURES / "d1_liplus_anchored_hybrid.contamination.json"
PRIOR_FIXTURES = [
    FIXTURES / "d1_liplus_benchmark.json",
    FIXTURES / "d1_liplus_dynamics_holdout.json",
    FIXTURES / "d1_liplus_local_competition_development.json",
    FIXTURES / "d1_liplus_local_competition_holdout.json",
]


class AnchoredFixtureTest(unittest.TestCase):
    def test_manifest_fixes_exact_roles_and_disjoint_connected_splits(self) -> None:
        manifest = read_manifest(MANIFEST)
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(len(manifest["variants"]), 6)
        self.assertEqual(manifest["baselines"], ["current", "bm25-only"])
        self.assertEqual(
            [variant["id"] for variant in manifest["variants"]],
            [
                "current",
                "bm25-only",
                "bm25-graph-additive",
                "anchored-local",
                "anchored-local-query",
                "bm25-anchored-local",
            ],
        )
        for path in (DEVELOPMENT, HOLDOUT):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            assert_connected(fixture)
            self.assertEqual(len(fixture["nodes"]), 5)
            self.assertGreaterEqual(len(fixture["edges"]), 4)
        for name in (
            "d1_liplus_anchored_hybrid_development.provenance.json",
            "d1_liplus_anchored_hybrid_holdout.provenance.json",
        ):
            provenance = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
            evidence = provenance["read_only_evidence"]
            self.assertTrue(all(value == 0 for value in evidence["rows_written"]))
            self.assertTrue(all(value == 0 for value in evidence["changes"]))
            self.assertTrue(all(value is False for value in evidence["changed_db"]))

    def test_audit_uses_only_prior_fixture_identifiers(self) -> None:
        audit = build_audit(
            development_fixture=DEVELOPMENT,
            development_gold=DEVELOPMENT_GOLD,
            holdout_fixture=HOLDOUT,
            holdout_gold=HOLDOUT_GOLD,
            prior_fixtures=PRIOR_FIXTURES,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit, json.loads(AUDIT.read_text(encoding="utf-8")))
        self.assertIn("prior gold and result artifacts are not loaded", audit["prior_usage"])


class AnchoredDynamicsTest(unittest.TestCase):
    @staticmethod
    def _trace(strategy: str):  # type: ignore[no-untyped-def]
        with NeuronGraphRAG(
            config=EngineConfig(
                activation_strategy=strategy,
                seed_count=1,
                recurrent_steps=2,
                recurrent_decay=0.5,
            )
        ) as engine:
            engine.add_document("seed", "source alpha anchor")
            engine.add_document("target", "related target evidence")
            engine.add_edge("seed", "target", "mention")
            return engine.search("source alpha", limit=2, now=1_000.0)

    def test_anchor_is_invariant_and_graph_paths_are_edge_only(self) -> None:
        for strategy in (
            "anchored_local_competition",
            "anchored_local_query_competition",
        ):
            trace = self._trace(strategy)
            self.assertTrue(trace.diagnostics["entry_anchor_invariant"])
            self.assertTrue(trace.diagnostics["graph_signal_excludes_zero_hop"])
            for hit in trace.hits:
                explanation = hit.explain()
                scores = explanation["scores"]
                self.assertEqual(
                    scores["entry_anchor_before_competition"],
                    scores["entry_anchor_after_competition"],
                )
                self.assertIn("sparse_raw", scores)
                self.assertIn("dense_raw", scores)
                self.assertIn("graph_activation_normalized", scores)
                self.assertTrue(
                    all(path["kind"] == "graph" for path in explanation["paths"])
                )

    def test_current_trace_marks_zero_hop_paths(self) -> None:
        trace = self._trace("current_positive_additive")
        self.assertFalse(trace.diagnostics["graph_signal_excludes_zero_hop"])
        kinds = {
            path["kind"]
            for hit in trace.hits
            for path in hit.explain()["paths"]
        }
        self.assertIn("entry_zero_hop", kinds)

    def test_bm25_only_skips_dense_and_graph_calls(self) -> None:
        def forbidden_dense(_: str) -> list[float]:
            raise AssertionError("dense encoder must not be called")

        with NeuronGraphRAG(
            config=EngineConfig(
                dense_weight=0.0,
                graph_weight=0.0,
                use_dense_retrieval=False,
                use_graph_propagation=False,
            ),
            dense_encoder=forbidden_dense,
        ) as engine:
            engine.add_document("only", "bm25 literal")
            engine.store.outgoing_edges = lambda _: (_ for _ in ()).throw(
                AssertionError("graph traversal must not be called")
            )
            trace = engine.search("bm25 literal", limit=1, now=1_000.0)
        self.assertFalse(trace.diagnostics["use_dense_retrieval"])
        self.assertFalse(trace.diagnostics["use_graph_propagation"])
        self.assertEqual(trace.diagnostics["stop_reason"], "graph_disabled")
        self.assertEqual(trace.hits[0].dense_raw_score, 0.0)
        self.assertEqual(trace.hits[0].graph_activation, 0.0)


class AnchoredSelectionTest(unittest.TestCase):
    @staticmethod
    def _variant(
        variant_id: str,
        *,
        direct: float,
        relation: float,
        negative: float,
        anchor: bool = True,
        edge_only: bool = True,
        feedback: bool = True,
    ) -> dict[str, object]:
        return {
            "id": variant_id,
            "metrics": {
                "cohorts": {
                    "direct_lookup": {"mean_reciprocal_rank": direct},
                    "relation": {"mean_reciprocal_rank": relation},
                    "negative_control": {"mean_reciprocal_rank": negative},
                }
            },
            "explanations": [{"matched": True}],
            "feedback": {
                "credited_edges": [{}] if feedback else [],
                "uncredited_edge_changes": [],
                "non_target_rank_changes": [],
            },
            "diagnostics": {
                "mean_expansions": 2.0,
                "entry_anchor_invariant": anchor,
                "graph_signal_excludes_zero_hop": edge_only,
            },
            "structural_complexity": 5,
        }

    def test_candidate_requires_relation_gain_controls_and_anchor_contract(self) -> None:
        variants = [
            self._variant("current", direct=1.0, relation=0.5, negative=1.0),
            self._variant("bm25-only", direct=1.0, relation=0.4, negative=1.0),
            self._variant(
                "additive", direct=1.0, relation=0.7, negative=1.0, edge_only=False
            ),
            self._variant("anchored", direct=1.0, relation=0.7, negative=1.0),
        ]
        selection = _select_anchored_development(variants)  # type: ignore[arg-type]
        self.assertEqual(selection["selected_variant_id"], "anchored")
        self.assertFalse(variants[2]["candidate_gate_passed"])
        self.assertTrue(variants[3]["candidate_gate_passed"])


if __name__ == "__main__":
    unittest.main()

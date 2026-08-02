from __future__ import annotations

import json
import unittest
from pathlib import Path

from neuron_graph_rag.engine import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.experiment import (
    _select_fusion_development,
    _trace_formula_recomputable,
    read_manifest,
)
from tools.acquire_d1_fixture import assert_connected
from tools.audit_anchored_fixture import build_audit


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_fusion_calibration_experiment.manifest.json"
DEVELOPMENT = FIXTURES / "d1_liplus_fusion_calibration_development.json"
DEVELOPMENT_GOLD = FIXTURES / "d1_liplus_fusion_calibration_development.gold.json"
HOLDOUT = FIXTURES / "d1_liplus_fusion_calibration_holdout.json"
HOLDOUT_GOLD = FIXTURES / "d1_liplus_fusion_calibration_holdout.gold.json"
AUDIT = FIXTURES / "d1_liplus_fusion_calibration.contamination.json"
DEVELOPMENT_RESULT = (
    FIXTURES / "d1_liplus_fusion_calibration_experiment.development.result.json"
)
HOLDOUT_RESULT = (
    FIXTURES / "d1_liplus_fusion_calibration_experiment.holdout.result.json"
)
PRIOR_FIXTURES = [
    FIXTURES / "d1_liplus_wiki.json",
    FIXTURES / "d1_liplus_benchmark.json",
    FIXTURES / "d1_liplus_dynamics_holdout.json",
    FIXTURES / "d1_liplus_local_competition_development.json",
    FIXTURES / "d1_liplus_local_competition_holdout.json",
    FIXTURES / "d1_liplus_anchored_hybrid_development.json",
    FIXTURES / "d1_liplus_anchored_hybrid_holdout.json",
]


class FusionCalibrationFreezeTest(unittest.TestCase):
    def test_manifest_fixes_six_variants_and_two_connected_splits(self) -> None:
        manifest = read_manifest(MANIFEST)
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["baselines"], ["current"])
        self.assertEqual(
            [variant["id"] for variant in manifest["variants"]],
            [
                "current",
                "anchored-local-unscaled",
                "anchored-linear-conservative",
                "anchored-linear-mass",
                "anchored-rrf-conservative",
                "anchored-rrf-balanced",
            ],
        )
        for path in (DEVELOPMENT, HOLDOUT):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            assert_connected(fixture)
            self.assertEqual(len(fixture["nodes"]), 3)
            self.assertEqual(len(fixture["edges"]), 2)
        self.assertFalse(DEVELOPMENT_RESULT.exists())
        self.assertFalse(HOLDOUT_RESULT.exists())

    def test_contamination_audit_covers_all_seven_prior_fixtures(self) -> None:
        audit = build_audit(
            development_fixture=DEVELOPMENT,
            development_gold=DEVELOPMENT_GOLD,
            holdout_fixture=HOLDOUT,
            holdout_gold=HOLDOUT_GOLD,
            prior_fixtures=PRIOR_FIXTURES,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["inputs"]["prior_fixtures"]), 7)
        self.assertEqual(audit, json.loads(AUDIT.read_text(encoding="utf-8")))

    def test_provenance_records_zero_writes(self) -> None:
        for name in (
            "d1_liplus_fusion_calibration_development.provenance.json",
            "d1_liplus_fusion_calibration_holdout.provenance.json",
        ):
            provenance = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
            evidence = provenance["read_only_evidence"]
            self.assertTrue(all(value == 0 for value in evidence["rows_written"]))
            self.assertTrue(all(value == 0 for value in evidence["changes"]))
            self.assertTrue(all(value is False for value in evidence["changed_db"]))


class FusionFormulaTest(unittest.TestCase):
    def test_graph_normalization_modes_are_exact(self) -> None:
        scores = {"a": 2.0, "b": 1.0, "c": 0.0}
        self.assertEqual(
            NeuronGraphRAG._normalize_graph_activation(scores, "none"),
            scores,
        )
        self.assertEqual(
            NeuronGraphRAG._normalize_graph_activation(scores, "max"),
            {"a": 1.0, "b": 0.5, "c": 0.0},
        )
        mass = NeuronGraphRAG._normalize_graph_activation(scores, "l1_mass")
        self.assertAlmostEqual(mass["a"], 2.0 / 3.0)
        self.assertAlmostEqual(mass["b"], 1.0 / 3.0)
        self.assertEqual(mass["c"], 0.0)

    @staticmethod
    def _rrf_components(entry_weight: float, graph_weight: float):  # type: ignore[no-untyped-def]
        with NeuronGraphRAG(
            config=EngineConfig(
                entry_weight=entry_weight,
                graph_weight=graph_weight,
                final_fusion_strategy="rrf",
                rrf_k=60,
            )
        ) as engine:
            direct = engine._fusion_components(
                entry=1.0,
                normalized_graph=0.0,
                entry_rank=1,
                graph_rank=None,
                positive_graph_count=2,
            )
            related = engine._fusion_components(
                entry=0.5,
                normalized_graph=1.0,
                entry_rank=2,
                graph_rank=1,
                positive_graph_count=2,
            )
        return sum(direct), sum(related), related

    def test_bottom_centered_rrf_brackets_conservative_and_balanced(self) -> None:
        conservative_direct, conservative_related, components = (
            self._rrf_components(0.8, 0.2)
        )
        balanced_direct, balanced_related, _ = self._rrf_components(0.65, 0.35)
        self.assertGreater(conservative_direct, conservative_related)
        self.assertLess(balanced_direct, balanced_related)
        expected_graph = 0.2 * (1.0 / 61.0 - 1.0 / 63.0)
        self.assertAlmostEqual(components[1], expected_graph)

    def test_trace_exposes_recomputable_components_and_positive_graph_ranks(self) -> None:
        config = EngineConfig(
            activation_strategy="anchored_local_competition",
            entry_weight=0.8,
            graph_weight=0.2,
            final_fusion_strategy="rrf",
            graph_normalization="none",
            seed_count=1,
            recurrent_steps=2,
        )
        with NeuronGraphRAG(config=config) as engine:
            engine.add_document("seed", "source alpha anchor")
            engine.add_document("left", "left related")
            engine.add_document("right", "right related")
            engine.add_edge("seed", "left", "mention")
            engine.add_edge("seed", "right", "mention")
            trace = engine.search("source alpha", limit=3, now=1_000.0)
        self.assertTrue(trace.diagnostics["final_order_recomputable"])
        self.assertTrue(_trace_formula_recomputable(trace, config))
        self.assertEqual(trace.diagnostics["positive_graph_node_count"], 2)
        for hit in trace.hits:
            explanation = hit.explain()
            fusion = explanation["fusion"]
            self.assertAlmostEqual(
                fusion["final"],
                fusion["entry_component"] + fusion["graph_component"],
            )
            if hit.node.node_id == "seed":
                self.assertIsNone(explanation["ranks"]["graph"])
                self.assertEqual(fusion["graph_component"], 0.0)


class FusionSelectionTest(unittest.TestCase):
    @staticmethod
    def _variant(
        variant_id: str,
        ranks: dict[str, int],
        *,
        direct: float,
        relation: float,
        negative: float,
    ) -> dict[str, object]:
        cohorts = {
            "d1": "direct_lookup",
            "d2": "direct_lookup",
            "r1": "relation",
            "r2": "relation",
            "n1": "negative_control",
            "n2": "negative_control",
        }
        return {
            "id": variant_id,
            "metrics": {
                "cohorts": {
                    "direct_lookup": {"mean_reciprocal_rank": direct},
                    "relation": {"mean_reciprocal_rank": relation},
                    "negative_control": {"mean_reciprocal_rank": negative},
                }
            },
            "cases": [
                {"id": case_id, "rank": rank, "cohort": cohorts[case_id]}
                for case_id, rank in ranks.items()
            ],
            "explanations": [{"matched": True}],
            "feedback": {
                "credited_edges": [{}],
                "uncredited_edge_changes": [],
                "non_target_rank_changes": [],
            },
            "diagnostics": {
                "mean_expansions": 2.0,
                "entry_anchor_invariant": True,
                "graph_signal_excludes_zero_hop": True,
                "final_order_recomputable": True,
            },
            "structural_complexity": 5,
        }

    def test_aggregate_non_regression_cannot_hide_individual_control_swap(self) -> None:
        current_ranks = {"d1": 1, "d2": 2, "r1": 2, "r2": 2, "n1": 1, "n2": 2}
        current = self._variant(
            "current",
            current_ranks,
            direct=0.75,
            relation=0.5,
            negative=0.75,
        )
        swapped = self._variant(
            "swapped",
            {**current_ranks, "d1": 2, "d2": 1, "r1": 1},
            direct=0.75,
            relation=0.75,
            negative=0.75,
        )
        passing = self._variant(
            "passing",
            {**current_ranks, "r1": 1},
            direct=0.75,
            relation=0.75,
            negative=0.75,
        )
        selection = _select_fusion_development([current, swapped, passing])  # type: ignore[arg-type]
        self.assertFalse(swapped["candidate_gate_passed"])
        self.assertTrue(passing["candidate_gate_passed"])
        self.assertEqual(selection["selected_variant_id"], "passing")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from neuron_graph_rag.d1_fixture import read_fixture
from neuron_graph_rag.engine import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.experiment import _select_development, read_manifest
from tools.acquire_d1_fixture import assert_connected


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_dynamics_experiment.manifest.json"
HOLDOUT = FIXTURES / "d1_liplus_dynamics_holdout.json"
HOLDOUT_GOLD = FIXTURES / "d1_liplus_dynamics_holdout.gold.json"
HOLDOUT_PROVENANCE = FIXTURES / "d1_liplus_dynamics_holdout.provenance.json"
DEVELOPMENT = FIXTURES / "d1_liplus_benchmark.json"


class DynamicsExperimentFreezeTest(unittest.TestCase):
    def test_manifest_freezes_thirteen_variants_and_disjoint_holdout(self) -> None:
        manifest = read_manifest(MANIFEST)
        self.assertEqual(len(manifest["variants"]), 13)
        self.assertLessEqual(len(manifest["variants"]), manifest["maximum_variants"])
        self.assertEqual(
            {variant["family"] for variant in manifest["variants"]},
            {
                "current_positive_additive",
                "finite_activation_budget",
                "lateral_inhibition",
                "query_conditioned_transmission",
                "recurrent_competition",
            },
        )
        development = read_fixture(DEVELOPMENT)
        holdout = read_fixture(HOLDOUT)
        development_paths = {
            node["metadata"]["doc_path"] for node in development["nodes"]
        }
        holdout_paths = {node["metadata"]["doc_path"] for node in holdout["nodes"]}
        self.assertEqual(development_paths & holdout_paths, set())
        self.assertEqual(len(holdout["nodes"]), 9)
        self.assertEqual(len(holdout["edges"]), 11)
        assert_connected(holdout)

    def test_holdout_gold_is_balanced_and_auditable_without_searching_it(self) -> None:
        gold = json.loads(HOLDOUT_GOLD.read_text(encoding="utf-8"))
        counts = {
            cohort: sum(case["cohort"] == cohort for case in gold["cases"])
            for cohort in ("direct_lookup", "relation", "negative_control")
        }
        self.assertEqual(counts, {cohort: 3 for cohort in counts})
        self.assertTrue(
            all(
                case["source_url"].startswith(
                    "https://github.com/Liplus-Project/liplus-language/wiki/"
                )
                for case in gold["cases"]
            )
        )
        node_ids = {node["node_id"] for node in read_fixture(HOLDOUT)["nodes"]}
        self.assertTrue(all(case["expected_node_id"] in node_ids for case in gold["cases"]))
        self.assertTrue(
            all(
                step["source_id"] in node_ids and step["target_id"] in node_ids
                for case in gold["cases"]
                for step in case.get("expected_path", [])
            )
        )

    def test_holdout_provenance_preserves_read_only_contract(self) -> None:
        provenance = json.loads(HOLDOUT_PROVENANCE.read_text(encoding="utf-8"))
        evidence = provenance["read_only_evidence"]
        self.assertTrue(all(value == 0 for value in evidence["rows_written"]))
        self.assertTrue(all(value == 0 for value in evidence["changes"]))
        self.assertFalse(any(evidence["changed_db"]))
        self.assertTrue(provenance["source"]["schema_fingerprint"].startswith("sha256:"))
        self.assertEqual(provenance["known_gaps"], [])


class DynamicsStrategyTest(unittest.TestCase):
    def _trace(self, config: EngineConfig):
        with NeuronGraphRAG(config=config) as engine:
            engine.add_document("seed", "alpha starting evidence")
            engine.add_document("target", "beta target evidence")
            engine.add_document("distractor", "gamma unrelated evidence")
            engine.add_edge("seed", "target", "supports")
            engine.add_edge("seed", "distractor", "mentions")
            return engine.search("alpha beta supports", limit=3, now=1_000.0)

    def test_every_frozen_strategy_is_deterministic_and_observable(self) -> None:
        manifest = read_manifest(MANIFEST)
        shared = {
            key: value
            for key, value in manifest["shared_config"].items()
            if key != "limit"
        }
        for variant in manifest["variants"]:
            values = dict(shared)
            values["activation_strategy"] = variant["family"]
            values.update(variant["parameters"])
            first = self._trace(EngineConfig(**values))
            second = self._trace(EngineConfig(**values))
            self.assertEqual(
                [hit.node.node_id for hit in first.hits],
                [hit.node.node_id for hit in second.hits],
                variant["id"],
            )
            self.assertEqual(first.diagnostics, second.diagnostics, variant["id"])
            self.assertEqual(first.diagnostics["strategy"], variant["family"])
            self.assertLessEqual(
                first.diagnostics["expansions"],
                EngineConfig(**values).max_propagation_expansions,
            )

    def test_current_strategy_retains_positive_additive_activation(self) -> None:
        trace = self._trace(
            EngineConfig(
                seed_count=1,
                max_hops=1,
                hop_decay=0.7,
                entry_weight=0.25,
                graph_weight=0.75,
            )
        )
        seed = next(hit for hit in trace.hits if hit.node.node_id == "seed")
        target = next(hit for hit in trace.hits if hit.node.node_id == "target")
        self.assertGreater(seed.graph_activation, 0.0)
        self.assertGreater(target.graph_activation, 0.0)
        self.assertEqual(trace.diagnostics["strategy"], "current_positive_additive")


class DevelopmentSelectionRuleTest(unittest.TestCase):
    @staticmethod
    def _variant(
        variant_id: str,
        relation: float,
        negative: float,
        direct: float = 1.0,
        expansions: float = 10.0,
        complexity: int = 1,
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
            "diagnostics": {"mean_expansions": expansions},
            "structural_complexity": complexity,
        }

    def test_selection_keeps_failed_and_dominated_variants(self) -> None:
        variants = [
            self._variant("current", 0.4, 0.7, complexity=0),
            self._variant("failed", 0.5, 0.6),
            self._variant("dominated", 0.5, 0.7, expansions=12.0),
            self._variant("winner", 0.6, 0.7, expansions=8.0),
        ]
        selection = _select_development(variants)  # type: ignore[arg-type]
        self.assertEqual(selection["selected_variant_id"], "winner")
        self.assertEqual(variants[1]["pareto_status"], "failed_gate")
        self.assertEqual(variants[2]["pareto_status"], "dominated")
        self.assertEqual(variants[2]["dominated_by"], ["winner"])

    def test_no_candidate_stops_before_holdout(self) -> None:
        variants = [
            self._variant("current", 0.4, 0.7, complexity=0),
            self._variant("tradeoff", 0.6, 0.6),
        ]
        selection = _select_development(variants)  # type: ignore[arg-type]
        self.assertEqual(selection["selected_variant_id"], "current")
        self.assertEqual(selection["reason"], "no_variant_passed_candidate_gate")


if __name__ == "__main__":
    unittest.main()

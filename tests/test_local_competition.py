from __future__ import annotations

import json
import unittest
from pathlib import Path

from neuron_graph_rag.engine import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.experiment import _select_local_development, read_manifest
from tools.acquire_d1_fixture import assert_connected
from tools.audit_local_competition_fixture import build_audit


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_local_competition_experiment.manifest.json"
DEVELOPMENT = FIXTURES / "d1_liplus_local_competition_development.json"
DEVELOPMENT_GOLD = (
    FIXTURES / "d1_liplus_local_competition_development.gold.json"
)
HOLDOUT = FIXTURES / "d1_liplus_local_competition_holdout.json"
HOLDOUT_GOLD = FIXTURES / "d1_liplus_local_competition_holdout.gold.json"
DEVELOPMENT_RESULT = (
    FIXTURES / "d1_liplus_local_competition_experiment.development.result.json"
)
HOLDOUT_RESULT = (
    FIXTURES / "d1_liplus_local_competition_experiment.holdout.result.json"
)


class LocalCompetitionFixtureTest(unittest.TestCase):
    def test_manifest_and_fixtures_are_frozen_and_disjoint(self) -> None:
        manifest = read_manifest(MANIFEST)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["variants"]), 6)
        self.assertEqual(manifest["baselines"], ["current", "recurrent-balanced"])
        for path in (DEVELOPMENT, HOLDOUT):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            assert_connected(fixture)
            self.assertEqual(len(fixture["nodes"]), 9)
            self.assertEqual(len(fixture["edges"]), 11)
        for name in (
            "d1_liplus_local_competition_development.provenance.json",
            "d1_liplus_local_competition_holdout.provenance.json",
        ):
            provenance = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
            evidence = provenance["read_only_evidence"]
            self.assertTrue(all(value == 0 for value in evidence["rows_written"]))
            self.assertTrue(all(value == 0 for value in evidence["changes"]))
            self.assertTrue(all(value is False for value in evidence["changed_db"]))

    def test_contamination_audit_never_loads_prior_gold_or_results(self) -> None:
        audit = build_audit(
            development_fixture=DEVELOPMENT,
            development_gold=DEVELOPMENT_GOLD,
            holdout_fixture=HOLDOUT,
            holdout_gold=HOLDOUT_GOLD,
            prior_development_fixture=FIXTURES / "d1_liplus_benchmark.json",
            prior_holdout_fixture=FIXTURES / "d1_liplus_dynamics_holdout.json",
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["old_holdout_usage"],
            "fixture identifiers only; old holdout gold and result are not loaded",
        )


class LocalCompetitionDynamicsTest(unittest.TestCase):
    @staticmethod
    def _trace(strategy: str):  # type: ignore[no-untyped-def]
        config = EngineConfig(
            activation_strategy=strategy,
            seed_count=1,
            max_hops=2,
            entry_weight=0.25,
            graph_weight=0.75,
            activation_budget=1.0,
            inhibition_ratio=0.1,
            query_transmission_floor=0.4,
            recurrent_steps=2,
            recurrent_decay=0.5,
            max_active_paths_per_node=4,
        )
        with NeuronGraphRAG(config=config) as engine:
            engine.add_document("seed", "source alpha local recurrent")
            engine.add_document("relevant", "alpha target evidence")
            engine.add_document("other", "unrelated sibling")
            engine.add_document("join", "joined path evidence")
            engine.add_edge("seed", "relevant", "mention")
            engine.add_edge("seed", "other", "mention")
            engine.add_edge("relevant", "join", "mention")
            engine.add_edge("other", "join", "mention")
            return engine.search("source alpha", limit=4, now=1_000.0)

    def test_all_local_variants_are_deterministic_and_bounded(self) -> None:
        strategies = (
            "local_neighbor_competition",
            "local_neighbor_query_competition",
            "local_neighbor_path_competition",
            "local_neighbor_query_path_competition",
        )
        for strategy in strategies:
            first = self._trace(strategy)
            second = self._trace(strategy)
            self.assertEqual(
                [hit.node.node_id for hit in first.hits],
                [hit.node.node_id for hit in second.hits],
            )
            self.assertEqual(first.diagnostics, second.diagnostics)
            self.assertEqual(first.diagnostics["strategy"], strategy)
            self.assertGreater(len(first.diagnostics["competition_sets"]), 0)
            self.assertLessEqual(first.diagnostics["expansions"], 10_000)

    def test_query_conditioning_changes_only_local_sibling_allocation(self) -> None:
        plain = self._trace("local_neighbor_competition")
        query = self._trace("local_neighbor_query_competition")
        plain_set = plain.diagnostics["competition_sets"][0]
        query_set = query.diagnostics["competition_sets"][0]
        self.assertEqual(plain_set["source_id"], "seed")
        self.assertEqual(query_set["source_id"], "seed")
        self.assertEqual(plain_set["neighbor_count"], 2)
        self.assertEqual(query_set["neighbor_count"], 2)
        self.assertEqual(plain_set["mean_query_relevance"], 1.0)
        self.assertLess(query_set["mean_query_relevance"], 1.0)

    def test_path_conditioning_retains_multiple_active_paths(self) -> None:
        node = self._trace("local_neighbor_query_competition")
        path = self._trace("local_neighbor_query_path_competition")
        self.assertGreaterEqual(
            path.diagnostics["active_path_count"],
            node.diagnostics["active_path_count"],
        )
        identities = {
            item["path_identity"] for item in path.diagnostics["competition_sets"]
        }
        self.assertGreater(len(identities), 1)


class LocalCompetitionSelectionTest(unittest.TestCase):
    @staticmethod
    def _variant(
        variant_id: str,
        *,
        direct: float,
        relation: float,
        negative: float,
        path: bool = True,
        feedback: bool = True,
        expansions: float = 5.0,
        complexity: int = 5,
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
            "explanations": [{"matched": path}],
            "feedback": {
                "credited_edges": [{}] if feedback else [],
                "uncredited_edge_changes": [],
                "non_target_rank_changes": [],
            },
            "diagnostics": {"mean_expansions": expansions},
            "structural_complexity": complexity,
        }

    def test_candidate_must_beat_both_relation_baselines(self) -> None:
        variants = [
            self._variant(
                "current", direct=1.0, relation=0.4, negative=1.0, complexity=0
            ),
            self._variant(
                "recurrent-balanced",
                direct=0.7,
                relation=0.6,
                negative=0.5,
                complexity=4,
            ),
            self._variant(
                "local-neighbor",
                direct=1.0,
                relation=0.6,
                negative=1.0,
            ),
            self._variant(
                "local-neighbor-query",
                direct=1.0,
                relation=0.7,
                negative=1.0,
                expansions=4.0,
            ),
        ]
        selection = _select_local_development(variants)  # type: ignore[arg-type]
        self.assertEqual(selection["selected_variant_id"], "local-neighbor-query")
        self.assertFalse(variants[2]["candidate_gate_passed"])
        self.assertTrue(variants[3]["candidate_gate_passed"])


class LocalCompetitionResultAuditTest(unittest.TestCase):
    def test_development_records_all_variants_and_stops_before_holdout(self) -> None:
        result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
        manifest = read_manifest(MANIFEST)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["variant_count"], 6)
        self.assertEqual(
            [variant["id"] for variant in result["variants"]],
            [variant["id"] for variant in manifest["variants"]],
        )
        self.assertEqual(result["selection"]["selected_variant_id"], "current")
        self.assertEqual(result["selection"]["eligible_variant_ids"], [])
        self.assertEqual(
            result["selection"]["reason"],
            "no_local_variant_passed_frozen_gate",
        )
        self.assertEqual(result["holdout_status"], "not_opened_no_candidate")
        self.assertFalse(HOLDOUT_RESULT.exists())

    def test_relation_gain_does_not_hide_control_regressions(self) -> None:
        result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
        by_id = {variant["id"]: variant for variant in result["variants"]}
        for variant_id in ("local-neighbor", "local-neighbor-path"):
            gate = by_id[variant_id]["candidate_gate"]
            self.assertTrue(gate["relation_strictly_above_current"])
            self.assertTrue(gate["relation_strictly_above_recurrent_balanced"])
            self.assertFalse(gate["direct_non_regression"])
            self.assertFalse(gate["negative_non_regression"])
        for variant_id in (
            "local-neighbor-query",
            "local-neighbor-query-path",
        ):
            gate = by_id[variant_id]["candidate_gate"]
            self.assertTrue(gate["negative_non_regression"])
            self.assertFalse(gate["direct_non_regression"])
            self.assertFalse(gate["relation_strictly_above_recurrent_balanced"])
        for variant in result["variants"]:
            self.assertTrue(all(item["matched"] for item in variant["explanations"]))
            self.assertEqual(variant["feedback"]["uncredited_edge_changes"], [])
            self.assertEqual(variant["feedback"]["non_target_rank_changes"], [])


if __name__ == "__main__":
    unittest.main()

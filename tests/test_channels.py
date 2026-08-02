from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.channel_experiment import (
    read_channel_manifest,
    run_channel_holdout,
)
from neuron_graph_rag.d1_fixture import read_fixture


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_channels_experiment.manifest.json"
DEVELOPMENT = FIXTURES / "d1_liplus_channels_development.json"
DEVELOPMENT_GOLD = FIXTURES / "d1_liplus_channels_development.gold.json"
DEVELOPMENT_PROVENANCE = (
    FIXTURES / "d1_liplus_channels_development.provenance.json"
)
HOLDOUT = FIXTURES / "d1_liplus_channels_holdout.json"
HOLDOUT_GOLD = FIXTURES / "d1_liplus_channels_holdout.gold.json"
HOLDOUT_PROVENANCE = FIXTURES / "d1_liplus_channels_holdout.provenance.json"
AUDIT = FIXTURES / "d1_liplus_channels.contamination.json"
DEVELOPMENT_RESULT = (
    FIXTURES / "d1_liplus_channels_experiment.development.result.json"
)
HOLDOUT_RESULT = FIXTURES / "d1_liplus_channels_experiment.holdout.result.json"


def _config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        seed_count=1,
        max_hops=1,
        inhibition_ratio=0.1,
    )


def _populate(engine: NeuronGraphRAG) -> None:
    engine.add_document("source", "alpha lexical source")
    engine.add_document("target", "distant relation target")
    engine.add_document("other", "unrelated control")
    engine.add_edge("source", "target", "mentions", weight=0.5)
    engine.add_edge("source", "other", "mentions", weight=0.25)


class SearchChannelsCoreTest(unittest.TestCase):
    def test_channels_are_independent_and_have_no_combined_order(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            result = engine.search_channels("alpha", limit=2, now=1_000.0)

            self.assertNotEqual(result.lexical.trace_id, result.relation.trace_id)
            self.assertEqual(result.lexical.query, result.relation.query)
            self.assertEqual(result.lexical.channel, "lexical")
            self.assertEqual(result.relation.channel, "relation")
            self.assertEqual(result.lexical.hits[0].node.node_id, "source")
            self.assertEqual(result.lexical.hits[0].paths, ())
            self.assertTrue(result.relation.hits)
            self.assertTrue(
                all(path.steps for hit in result.relation.hits for path in hit.paths)
            )
            self.assertFalse(hasattr(result, "hits"))
            self.assertFalse(hasattr(result, "final_score"))
            self.assertEqual(
                engine.store.retrieval_channel(result.lexical.trace_id), "lexical"
            )
            self.assertEqual(
                engine.store.retrieval_channel(result.relation.trace_id), "relation"
            )

    def test_trace_provenance_controls_feedback(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            result = engine.search_channels("alpha", limit=3, now=1_000.0)
            before = engine.store.edge("source", "target", "mentions")

            lexical = engine.record_success(
                result.lexical.trace_id, ["target"], now=1_001.0
            )
            after_lexical = engine.store.edge("source", "target", "mentions")
            relation = engine.record_success(
                result.relation.trace_id, ["target"], now=1_002.0
            )
            after_relation = engine.store.edge("source", "target", "mentions")

            self.assertEqual(lexical.channel, "lexical")
            self.assertEqual(lexical.reinforced_edges, ())
            self.assertEqual(after_lexical.weight, before.weight)
            self.assertEqual(relation.channel, "relation")
            self.assertEqual(len(relation.reinforced_edges), 1)
            self.assertGreater(after_relation.weight, after_lexical.weight)

    def test_cross_channel_node_trace_misuse_is_rejected_atomically(self) -> None:
        with NeuronGraphRAG(config=_config()) as engine:
            _populate(engine)
            result = engine.search_channels("alpha", limit=1, now=1_000.0)
            self.assertEqual(result.lexical.hits[0].node.node_id, "source")
            self.assertEqual(result.relation.hits[0].node.node_id, "target")
            before = engine.store.edge("source", "target", "mentions")

            with self.assertRaises(ValueError):
                engine.record_success(
                    result.lexical.trace_id, ["target"], now=1_001.0
                )

            self.assertEqual(engine.store.count_feedback(), 0)
            self.assertEqual(
                engine.store.edge("source", "target", "mentions"), before
            )

    def test_channel_trace_provenance_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "channels.db"
            with NeuronGraphRAG(database, config=_config()) as engine:
                _populate(engine)
                result = engine.search_channels("alpha", limit=2, now=1_000.0)
                lexical_trace_id = result.lexical.trace_id
                relation_trace_id = result.relation.trace_id
            with NeuronGraphRAG(database, config=_config()) as reopened:
                self.assertEqual(
                    reopened.store.retrieval_channel(lexical_trace_id), "lexical"
                )
                self.assertEqual(
                    reopened.store.retrieval_channel(relation_trace_id), "relation"
                )


class ChannelFreezeContractTest(unittest.TestCase):
    def test_manifest_freezes_two_disjoint_connected_splits(self) -> None:
        manifest = read_channel_manifest(MANIFEST)
        self.assertEqual(manifest["candidate_id"], "dual-lane")
        self.assertEqual(len(manifest["gate"]), 12)
        development = read_fixture(DEVELOPMENT)
        holdout = read_fixture(HOLDOUT)
        self.assertEqual(len(development["nodes"]), 2)
        self.assertEqual(len(development["edges"]), 1)
        self.assertEqual(len(holdout["nodes"]), 2)
        self.assertEqual(len(holdout["edges"]), 1)
        development_paths = {
            node["metadata"]["doc_path"] for node in development["nodes"]
        }
        holdout_paths = {node["metadata"]["doc_path"] for node in holdout["nodes"]}
        self.assertFalse(development_paths & holdout_paths)

    def test_gold_freezes_exact_four_hard_gate_cases(self) -> None:
        for path in (DEVELOPMENT_GOLD, HOLDOUT_GOLD):
            gold = json.loads(path.read_text(encoding="utf-8"))
            cohorts = [case["cohort"] for case in gold["cases"]]
            self.assertEqual(len(cohorts), 4)
            self.assertEqual(cohorts.count("direct_lookup"), 2)
            self.assertEqual(cohorts.count("relation"), 1)
            self.assertEqual(cohorts.count("directional_negative"), 1)
            self.assertTrue(all(case["acceptable_rank"] == 1 for case in gold["cases"]))

    def test_provenance_and_contamination_are_frozen_without_writes(self) -> None:
        for path in (DEVELOPMENT_PROVENANCE, HOLDOUT_PROVENANCE):
            provenance = json.loads(path.read_text(encoding="utf-8"))
            evidence = provenance["read_only_evidence"]
            self.assertEqual(evidence["query_count"], 5)
            self.assertEqual(evidence["rows_written"], [0] * 5)
            self.assertEqual(evidence["changes"], [0] * 5)
            self.assertEqual(evidence["changed_db"], [False] * 5)
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["inputs"]["prior_fixtures"]), 9)
        self.assertIn("prior gold and result artifacts are not loaded", audit["prior_usage"])

    def test_development_result_is_recorded_and_holdout_remains_absent(self) -> None:
        self.assertTrue(DEVELOPMENT_RESULT.exists())
        self.assertFalse(HOLDOUT_RESULT.exists())


class ChannelResultAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))

    def test_development_preserves_negative_result_and_stop_rule(self) -> None:
        self.assertEqual(self.result["stage"], "development")
        self.assertFalse(self.result["gate_passed"])
        self.assertEqual(
            self.result["selection"],
            {
                "reason": "channel_candidate_failed_frozen_gate",
                "selected_candidate_id": "current",
            },
        )
        self.assertEqual(self.result["holdout_status"], "not_opened_no_candidate")
        passed = [name for name, value in self.result["gate"].items() if value]
        failed = [name for name, value in self.result["gate"].items() if not value]
        self.assertEqual(len(passed), 10)
        self.assertEqual(
            failed,
            [
                "lexical_controls_do_not_regress",
                "relation_paths_match_and_exclude_zero_hop",
            ],
        )

    def test_relation_gain_and_feedback_attribution_are_still_explicit(self) -> None:
        metrics = self.result["metrics"]
        self.assertEqual(metrics["relation_bm25_mrr"], 0.5)
        self.assertEqual(metrics["relation_mrr"], 1.0)
        self.assertEqual(metrics["union_coverage"], 1.0)
        self.assertTrue(self.result["feedback"]["lexical_success_isolated"])
        self.assertTrue(self.result["feedback"]["relation_success_isolated"])
        self.assertTrue(self.result["feedback"]["cross_lane_misuse_rejected"])

    def test_stop_rule_rejects_holdout_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "Stop rule forbids opening"):
            run_channel_holdout(MANIFEST, DEVELOPMENT_RESULT)


if __name__ == "__main__":
    unittest.main()

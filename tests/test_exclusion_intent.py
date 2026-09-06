from __future__ import annotations

import json
import unittest
from pathlib import Path

from neuron_graph_rag import (
    EngineConfig,
    NeuronGraphRAG,
    PrecisionControl,
    decompose_exclusion_intent,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "tests/fixtures/exclusion_intent_v1.cases.json").read_text(
        encoding="utf-8"
    )
)
SOURCE_GROUNDED_CORPUS = json.loads(
    (
        ROOT
        / "tests/fixtures/github_source_grounded_relation_v2.corpus.json"
    ).read_text(encoding="utf-8")
)
SOURCE_GROUNDED_QUERIES = json.loads(
    (
        ROOT
        / "tests/fixtures/github_source_grounded_relation_v2.queries.json"
    ).read_text(encoding="utf-8")
)
SOURCE_GROUNDED_GOLD = json.loads(
    (
        ROOT
        / "tests/fixtures/github_source_grounded_relation_v2.gold.json"
    ).read_text(encoding="utf-8")
)


def lexical_config(
    *, precision_control: PrecisionControl | None = None
) -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        entry_weight=1.0,
        graph_weight=0.0,
        use_dense_retrieval=False,
        use_graph_propagation=False,
        precision_control=precision_control,
    )


class ExclusionIntentParsingTest(unittest.TestCase):
    def test_frozen_parser_cases(self) -> None:
        for case in CASES["parser_cases"]:
            with self.subTest(case=case["id"]):
                if "error" in case:
                    with self.assertRaisesRegex(ValueError, case["error"]):
                        decompose_exclusion_intent(case["query"])
                    continue
                intent = decompose_exclusion_intent(case["query"])
                self.assertEqual(intent.positive_query, case["positive_query"])
                self.assertEqual(
                    intent.exclusion_clauses,
                    tuple(case["exclusion_clauses"]),
                )

    def test_empty_marker_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have a clause"):
            decompose_exclusion_intent("deployment without")

    def test_normal_word_boundaries_do_not_match_exclusion_phrase(self) -> None:
        with NeuronGraphRAG(config=lexical_config()) as engine:
            engine.add_document("trust", "editor trust policy")
            trace = engine.search("editor without Rust", limit=1, now=1.0)
        self.assertEqual([hit.node.node_id for hit in trace.hits], ["trust"])


class ExclusionIntentSearchTest(unittest.TestCase):
    def test_frozen_search_cases(self) -> None:
        for case in CASES["search_cases"]:
            with self.subTest(case=case["id"]):
                with NeuronGraphRAG(config=lexical_config()) as engine:
                    for document in case["documents"]:
                        engine.add_document(
                            document["node_id"], document["text"]
                        )
                    trace = engine.search(
                        case["query"], limit=case["limit"], now=1.0
                    )
                self.assertCountEqual(
                    [hit.node.node_id for hit in trace.hits],
                    case["expected_returned_node_ids"],
                )
                self.assertEqual(
                    trace.diagnostics["exclusion_intent"]["excluded_node_ids"],
                    case["expected_excluded_node_ids"],
                )

    def test_exclusion_runs_after_graph_ranking(self) -> None:
        config = EngineConfig(
            sparse_weight=1.0,
            dense_weight=0.0,
            entry_weight=0.4,
            graph_weight=0.6,
            seed_count=1,
            max_hops=1,
            use_dense_retrieval=False,
        )
        with NeuronGraphRAG(config=config) as engine:
            engine.add_document("source", "alpha deployment")
            engine.add_document("docker", "Docker cluster")
            engine.add_document("safe", "alpha release")
            engine.add_edge("source", "docker", "informs", weight=1.0)
            trace = engine.search("alpha without Docker", limit=2, now=1.0)
        self.assertNotIn("docker", [hit.node.node_id for hit in trace.hits])
        self.assertIn(
            "docker", trace.diagnostics["exclusion_intent"]["excluded_node_ids"]
        )

    def test_known_source_grounded_negative_failures_are_excluded(self) -> None:
        queries = {
            row["case_id"]: row["query"]
            for row in SOURCE_GROUNDED_QUERIES["stages"]["development"]
        }
        gold = {
            row["case_id"]: row["forbidden_path"]
            for row in SOURCE_GROUNDED_GOLD["stages"]["development"]
            if row["forbidden_path"] is not None
        }
        case_ids = (
            "v25-dev-negative-boundary-three",
            "v25-dev-negative-boundary-ten",
        )
        self.assertEqual(
            decompose_exclusion_intent(
                queries["v25-dev-negative-boundary-ten"]
            ).positive_query,
            "Locate the first-credit recovery state",
        )
        for case_id in case_ids:
            with self.subTest(case=case_id):
                with NeuronGraphRAG(config=lexical_config()) as engine:
                    for document in SOURCE_GROUNDED_CORPUS["documents"]:
                        engine.add_document(
                            document["path"],
                            document["content"],
                            metadata={"path": document["path"]},
                        )
                    trace = engine.search(queries[case_id], limit=5, now=1.0)
                returned = {hit.node.node_id for hit in trace.hits}
                excluded = set(
                    trace.diagnostics["exclusion_intent"]["excluded_node_ids"]
                )
                self.assertNotIn(gold[case_id], returned)
                self.assertIn(gold[case_id], excluded)

    def test_exclusion_runs_after_precision_control_and_before_limit(self) -> None:
        control = PrecisionControl("accept-all", minimum_final_score=0.0)
        with NeuronGraphRAG(
            config=lexical_config(precision_control=control)
        ) as engine:
            engine.add_document("a-docker", "deployment Docker")
            engine.add_document("b-safe", "deployment Podman")
            engine.add_document("c-safe", "deployment release")
            trace = engine.search("deployment without Docker", limit=2, now=1.0)
        self.assertEqual(
            [hit.node.node_id for hit in trace.hits], ["b-safe", "c-safe"]
        )
        self.assertIn("precision_control", trace.diagnostics)
        self.assertEqual(
            trace.diagnostics["exclusion_intent"]["excluded_node_ids"],
            ["a-docker"],
        )

    def test_excluded_reason_is_in_diagnostics_and_pass_is_in_hit_explain(
        self,
    ) -> None:
        with NeuronGraphRAG(config=lexical_config()) as engine:
            engine.add_document("excluded", "deployment Docker")
            engine.add_document("safe", "deployment Podman")
            trace = engine.search("deployment without Docker", limit=1, now=1.0)
        decisions = trace.diagnostics["exclusion_intent"]["decisions"]
        excluded = next(row for row in decisions if row["node_id"] == "excluded")
        self.assertEqual(excluded["reason"], "excluded_by_query_clause")
        self.assertEqual(
            excluded["matched_exclusions"],
            [{"clause": "Docker", "reason": "direct_phrase_match"}],
        )
        self.assertTrue(trace.hits[0].explain()["exclusion_intent"]["accepted"])

    def test_no_exclusion_keeps_exact_existing_search_contract(self) -> None:
        with NeuronGraphRAG(config=lexical_config()) as left_engine:
            left_engine.add_document("a", "notebook migration")
            left = left_engine.search("notebook migration", limit=1, now=1.0)
        with NeuronGraphRAG(config=lexical_config()) as right_engine:
            right_engine.add_document("a", "notebook migration")
            right = right_engine.search("notebook migration", limit=1, now=1.0)
        self.assertEqual(left.hits, right.hits)
        self.assertEqual(left.diagnostics, right.diagnostics)
        self.assertNotIn("exclusion_intent", left.hits[0].explain())
        self.assertNotIn("exclusion_intent", left.diagnostics)


if __name__ == "__main__":
    unittest.main()

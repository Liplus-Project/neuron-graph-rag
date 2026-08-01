from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from neuron_graph_rag.benchmark import read_gold, run_benchmark
from neuron_graph_rag.d1_fixture import read_fixture
from tools.acquire_d1_fixture import assert_connected


FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "d1_liplus_benchmark.json"
GOLD = FIXTURES / "d1_liplus_benchmark.gold.json"
PROVENANCE = FIXTURES / "d1_liplus_benchmark.provenance.json"
RESULT = FIXTURES / "d1_liplus_benchmark.result.json"


class RealCorpusBenchmarkContractTest(unittest.TestCase):
    def test_gold_is_canonical_and_fixes_twelve_balanced_cases(self) -> None:
        gold = read_gold(GOLD)
        canonical = json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

        self.assertEqual(GOLD.read_text(encoding="utf-8"), canonical)
        self.assertEqual(len(gold["cases"]), 12)
        self.assertEqual(
            Counter(case["cohort"] for case in gold["cases"]),
            {
                "direct_lookup": 4,
                "relation": 4,
                "negative_control": 4,
            },
        )
        relation_hops = Counter(
            len(case["expected_path"])
            for case in gold["cases"]
            if case["cohort"] == "relation"
        )
        self.assertEqual(relation_hops, {1: 2, 2: 2})

    def test_fixture_is_canonical_connected_and_matches_gold_nodes(self) -> None:
        fixture = read_fixture(FIXTURE)
        gold = read_gold(GOLD)
        canonical = (
            json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )

        self.assertEqual(FIXTURE.read_text(encoding="utf-8"), canonical)
        self.assertEqual(len(fixture["nodes"]), 12)
        self.assertEqual(len(fixture["edges"]), 26)
        self.assertEqual(
            {edge["edge_type"] for edge in fixture["edges"]}, {"mention"}
        )
        assert_connected(fixture)
        node_ids = {node["node_id"] for node in fixture["nodes"]}
        self.assertTrue(
            all(case["expected_node_id"] in node_ids for case in gold["cases"])
        )

    def test_provenance_proves_read_only_acquisition(self) -> None:
        report = json.loads(PROVENANCE.read_text(encoding="utf-8"))

        self.assertEqual(report["result"]["nodes_included"], 12)
        self.assertEqual(report["result"]["edges_included"], 26)
        self.assertTrue(
            all(value == 0 for value in report["read_only_evidence"]["rows_written"])
        )
        self.assertTrue(
            all(value == 0 for value in report["read_only_evidence"]["changes"])
        )
        self.assertFalse(any(report["read_only_evidence"]["changed_db"]))
        self.assertTrue(report["source"]["schema_fingerprint"].startswith("sha256:"))

    def test_checked_result_matches_frozen_inputs(self) -> None:
        checked = json.loads(RESULT.read_text(encoding="utf-8"))
        regenerated = run_benchmark(FIXTURE, GOLD)
        canonical = (
            json.dumps(checked, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )

        self.assertEqual(RESULT.read_text(encoding="utf-8"), canonical)
        self.assertEqual(regenerated, checked)
        self.assertEqual(
            {item["id"]: item["status"] for item in checked["hypotheses"]},
            {
                "H1": "supported",
                "H2": "unsupported",
                "H3": "supported",
                "H4": "supported",
            },
        )
        self.assertTrue(all(item["matched"] for item in checked["explanations"]))
        self.assertEqual(checked["feedback"]["uncredited_edge_changes"], [])
        self.assertEqual(checked["feedback"]["non_target_rank_changes"], [])


if __name__ == "__main__":
    unittest.main()

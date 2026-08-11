from __future__ import annotations

import unittest

from neuron_graph_rag.cli import run_demo
from neuron_graph_rag.evaluation import evaluate


class VerticalSliceTest(unittest.TestCase):
    def test_demo_runs_ingest_search_feedback_and_research(self) -> None:
        result = run_demo()

        self.assertEqual(result["retrieval_count"], 2)
        self.assertEqual(result["feedback_count"], 1)
        self.assertGreater(
            result["implemented_by_weight"]["after"],
            result["implemented_by_weight"]["before"],
        )
        self.assertGreater(
            result["after"]["graph_activation"],
            result["before"]["graph_activation"],
        )
        self.assertTrue(result["success_feedback"]["reinforced_edges"])
        self.assertEqual(
            result["success_feedback"]["evidence"][0]["count"], 1
        )
        self.assertEqual(
            result["success_feedback"]["evidence"][0]["quorum"], 1
        )
        self.assertTrue(
            result["success_feedback"]["evidence"][0]["activated"]
        )

    def test_eval_compares_baseline_and_graph_ranking(self) -> None:
        result = evaluate()

        self.assertEqual(result["cases"], 3)
        self.assertGreaterEqual(
            result["graph_rag"]["mean_reciprocal_rank"],
            result["baseline_hybrid"]["mean_reciprocal_rank"],
        )
        self.assertGreater(result["improved_queries"], 0)


if __name__ == "__main__":
    unittest.main()

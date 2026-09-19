from __future__ import annotations

import inspect
import json
import unittest
from itertools import pairwise

from neuron_graph_rag import full_corpus_rerank_oracle as oracle


class FullCorpusRerankOracleTests(unittest.TestCase):
    def test_manifest_binds_exact_v5_inputs_and_two_pinned_models(self) -> None:
        manifest = oracle._manifest(oracle.ROOT)
        self.assertEqual(manifest["corpus_document_count"], 93)
        self.assertEqual(manifest["rerank_cutoff"], 20)
        self.assertEqual(
            [row["model_id"] for row in manifest["models"]],
            ["BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"],
        )
        self.assertEqual(
            manifest["source_files"],
            {
                "corpus": oracle.CORPUS.as_posix(),
                "gold": oracle.GOLD.as_posix(),
                "model_registry": oracle.MODEL_REGISTRY.as_posix(),
                "queries": oracle.QUERIES.as_posix(),
                "schema": oracle.SCHEMA.as_posix(),
            },
        )

    def test_only_registered_development_case_and_gold_are_selected(self) -> None:
        self.assertEqual(
            oracle._target_query(oracle.ROOT),
            "異なる判断軸の矛盾を優先順位で潰さず境界へ戻して解く原則",
        )
        self.assertEqual(oracle._expected_source(oracle.ROOT), oracle.EXPECTED_SOURCE_ID)
        worker_source = inspect.getsource(oracle._worker_payload)
        self.assertNotIn("_expected_source", worker_source)
        self.assertNotIn("GOLD", worker_source)

    def test_full_corpus_projection_covers_every_document_end_to_end(self) -> None:
        documents = oracle._documents(oracle.ROOT)
        self.assertEqual(len(documents), 93)
        for document in documents:
            text = document["content"]
            chunks = oracle.project_passages(text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["start_codepoint"], 0)
            self.assertEqual(chunks[-1]["end_codepoint"], len(text))
            for left, right in pairwise(chunks):
                self.assertEqual(
                    right["start_codepoint"],
                    left["end_codepoint"] - oracle.OVERLAP_CODEPOINTS,
                )

    def test_classification_requires_model_agreement_at_fixed_cutoff(self) -> None:
        self.assertEqual(
            oracle.classify_ranks([1, 20]), "candidate_generation_bottleneck"
        )
        self.assertEqual(
            oracle.classify_ranks([21, 93]), "semantic_discrimination_bottleneck"
        )
        self.assertEqual(oracle.classify_ranks([20, 21]), "undetermined")
        with self.assertRaises(ValueError):
            oracle.classify_ranks([])

    def test_schema_is_exact_and_describes_all_document_evidence(self) -> None:
        schema = json.loads((oracle.ROOT / oracle.SCHEMA).read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["corpus_document_count"]["const"], 93)
        model = schema["$defs"]["model"]
        self.assertFalse(model["additionalProperties"])
        self.assertEqual(model["properties"]["documents"]["minItems"], 93)
        document = schema["$defs"]["document"]
        self.assertIn("best_chunk_score", document["required"])
        self.assertIn("chunk_count", document["required"])
        self.assertIn("winning_chunk_sha256", document["required"])

    def test_synthetic_probe_never_runs_registered_query_or_model(self) -> None:
        result = oracle.probe(oracle.ROOT)
        self.assertEqual(result["status"], "synthetic_probe_valid")
        self.assertEqual(result["registered_query_execution_count"], 0)
        self.assertEqual(result["model_forward_inference_count"], 0)

    def test_audit_accepts_only_result_free_or_single_terminal_evidence(self) -> None:
        result = oracle.audit(oracle.ROOT)
        self.assertIn(result["status"], {"result_free_valid", "observed_valid"})
        self.assertEqual(result["holdout_read_count"], 0)
        self.assertEqual(result["github_rag_request_count"], 0)
        self.assertEqual(result["shared_database_open_count"], 0)


if __name__ == "__main__":
    unittest.main()

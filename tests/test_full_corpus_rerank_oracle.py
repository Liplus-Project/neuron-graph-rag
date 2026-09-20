from __future__ import annotations

import hashlib
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

    def test_v1_loaders_read_mixed_stage_files_and_execution_is_retired(self) -> None:
        queries = json.loads((oracle.ROOT / oracle.QUERIES).read_text(encoding="utf-8"))
        gold = json.loads((oracle.ROOT / oracle.GOLD).read_text(encoding="utf-8"))
        self.assertEqual(
            {stage: len(rows) for stage, rows in queries["stages"].items()},
            {"development": 5, "holdout": 5},
        )
        self.assertEqual(
            {stage: len(rows) for stage, rows in gold["stages"].items()},
            {"development": 5, "holdout": 5},
        )
        self.assertIn("read_json(root / QUERIES)", inspect.getsource(oracle._target_query))
        self.assertIn("read_json(root / GOLD)", inspect.getsource(oracle._expected_source))
        self.assertTrue(oracle.V1_RETIRED)
        with self.assertRaisesRegex(RuntimeError, "holdout-bearing mixed-stage inputs"):
            oracle._reject_retired_v1_execution()
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

    def test_invalidation_schema_and_evidence_remove_valid_classification(self) -> None:
        schema = json.loads(
            (oracle.ROOT / oracle.INVALIDATION_SCHEMA).read_text(encoding="utf-8")
        )
        evidence = json.loads(
            (oracle.ROOT / oracle.INVALIDATION).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["valid_classification"], {"type": "null"})
        self.assertEqual(evidence["status"], "invalidated")
        self.assertIsNone(evidence["valid_classification"])
        self.assertEqual(evidence["holdout_bearing_input_file_count"], 2)
        self.assertEqual(evidence["unique_holdout_record_count"], 10)
        oracle.validate_invalidation(evidence, oracle.ROOT)

    def test_original_claim_and_result_bytes_are_preserved(self) -> None:
        expected = {
            oracle.CLAIM: "217ad55dffcab57095840c1d69bc17af0f291fc262cd45c5500ec87589fbad08",
            oracle.RESULT: "ea4a64fe75b7f415c25fa26375375f07944078f34637838b66e565b6e88c23ca",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((oracle.ROOT / relative).read_bytes()).hexdigest(), digest
            )

    def test_synthetic_probe_never_runs_registered_query_or_model(self) -> None:
        result = oracle.probe(oracle.ROOT)
        self.assertEqual(result["status"], "synthetic_probe_valid")
        self.assertEqual(result["registered_query_execution_count"], 0)
        self.assertEqual(result["model_forward_inference_count"], 0)

    def test_host_wrapper_exposes_only_audit_and_probe_after_invalidation(self) -> None:
        runner = (
            oracle.ROOT / "tools/run_full_corpus_rerank_oracle_wslc.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('[ValidateSet("audit", "probe")]', runner)
        self.assertNotIn('"preflight"', runner)
        self.assertNotIn('"run"', runner)

    def test_audit_exposes_invalidated_observation_without_valid_classification(self) -> None:
        result = oracle.audit(oracle.ROOT)
        self.assertEqual(result["status"], "observed_invalidated")
        self.assertTrue(result["holdout_contract_violation"])
        self.assertEqual(result["holdout_bearing_input_file_count"], 2)
        self.assertEqual(result["unique_holdout_record_count"], 10)
        self.assertIsNone(result["valid_classification"])
        self.assertEqual(
            result["reported_invalidated_classification"],
            "semantic_discrimination_bottleneck",
        )
        self.assertEqual(result["github_rag_request_count"], 0)
        self.assertEqual(result["shared_database_open_count"], 0)


if __name__ == "__main__":
    unittest.main()

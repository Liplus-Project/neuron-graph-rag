from __future__ import annotations

import hashlib
import json
import unittest

from neuron_graph_rag import structural_representation_length_bias_diagnostic as diagnostic
from neuron_graph_rag import structural_representation_length_bias_ablation as runner


class StructuralRepresentationLengthBiasAblationTests(unittest.TestCase):
    def test_result_free_diagnostic_is_reproducible_and_queryless(self) -> None:
        audit = diagnostic.audit(diagnostic.ROOT)
        self.assertEqual(audit["status"], "result_free_manifest_frozen")
        self.assertEqual(audit["registered_query_execution_count"], 0)
        self.assertEqual(audit["corpus_document_count"], 93)
        self.assertEqual(audit["proposed_arm_count"], 4)
        self.assertFalse(audit["parent_judgment_required"])

    def test_v2_evidence_exposes_length_bias_signal_and_winning_chunks(self) -> None:
        payload = diagnostic.read_json(diagnostic.ROOT / diagnostic.DIAGNOSTIC)
        models = {row["kind"]: row for row in payload["models"]}
        self.assertAlmostEqual(
            models["base"]["correlations"]["spearman_chunk_count_vs_best_score"],
            0.4182360021317228,
        )
        self.assertAlmostEqual(
            models["v2-m3"]["correlations"]["spearman_chunk_count_vs_best_score"],
            0.5742361338600934,
        )
        for model in models.values():
            self.assertEqual(model["gold"]["structural_fields"]["document_title"], "Axis Separation")
            self.assertEqual(model["gold"]["structural_fields"]["heading_chain"], "[missing]")
            self.assertTrue(model["gold"]["winning_chunk_text"].startswith("---\n"))

    def test_structural_prefix_literal_handles_fences_whitespace_and_missing(self) -> None:
        content = "#  A\t title  \n\n```\n## ignored\n```\n## Part   one\nbody\n"
        start = content.index("body")
        prefix = diagnostic.structural_prefix(
            "corpora/github-retrieval-parity-v4/docs/example file.md",
            content,
            start,
        )
        self.assertEqual(
            prefix,
            "path: docs/example file.md\n"
            "filename: example file.md\n"
            "title: A title\n"
            "headings: A title > Part one\n\n",
        )
        self.assertEqual(
            diagnostic.structural_prefix(
                "corpora/github-retrieval-parity-v4/docs/no-heading.md", "body", 0
            ),
            "path: docs/no-heading.md\n"
            "filename: no-heading.md\n"
            "title: [missing]\n"
            "headings: [missing]\n\n",
        )

    def test_manifest_freezes_factorial_formula_success_and_truncation_diagnostics(self) -> None:
        manifest = json.loads(
            (diagnostic.ROOT / diagnostic.MANIFEST).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "frozen_pre_registered_execution")
        self.assertEqual(
            [row["arm_id"] for row in manifest["arms"]],
            ["body_max", "structural_max", "body_nlme", "structural_nlme"],
        )
        self.assertEqual(
            manifest["aggregations"]["normalized_log_mean_exp"]["temperature"], 1.0
        )
        self.assertEqual(manifest["ranking"]["cutoff"], 20)
        structural = manifest["representations"]["structural"]
        self.assertIsNone(structural["prefix_cap"])
        self.assertIn("prefix_token_length", json.dumps(manifest["saved_evidence"]))

    def test_runner_is_result_free_and_nlme_is_chunk_count_normalized(self) -> None:
        audit = runner.audit(runner.ROOT)
        self.assertEqual(audit["status"], "result_free_frozen")
        self.assertEqual(audit["registered_query_execution_count"], 0)
        self.assertAlmostEqual(runner._nlme([2.0]), 2.0)
        self.assertAlmostEqual(runner._nlme([2.0, 2.0]), 2.0)
        self.assertLess(runner._nlme([2.0, -10.0]), 2.0)

    def test_wrapper_keeps_gold_out_of_preflight_and_workers(self) -> None:
        wrapper = (
            runner.ROOT
            / "tools/run_structural_representation_length_bias_ablation_v1_wslc.ps1"
        ).read_text(encoding="utf-8")
        pre_worker = wrapper.split('$goldPath =', 1)[0]
        self.assertNotIn('"gold" = @(', pre_worker)
        self.assertIn("full_corpus_rerank_oracle_v2.gold.json:ro", wrapper)
        self.assertIn("github-structural-length-bias-ablation-v1-runtime", wrapper)

    def test_v1_and_v2_owned_files_are_byte_for_byte_unchanged(self) -> None:
        expected = {
            "docs/experiments/full-corpus-rerank-oracle-v1.md": "7a97a97b9ae774694e9b8a21d84b8fe604521911d6c70d3c3138190366d584bf",
            "src/neuron_graph_rag/full_corpus_rerank_oracle.py": "d4e4c1df3568ac8be45fbd0b5001a336d5c8d6a9e108ca3bd028c6fe77746bf8",
            "tests/fixtures/full_corpus_rerank_oracle_v1.manifest.json": "c356cd5400b3e72b576e1f4aa24ec795440a40d72e81facaaad5d6e33fce2ea7",
            "tests/fixtures/full_corpus_rerank_oracle_v1.schema.json": "88f663af76becc1e440f2c50a5472f3b8242e97c56c13dc3cf9f47c0768fd4f6",
            "tests/fixtures/full_corpus_rerank_oracle_v1.invalidation.schema.json": "c973f6983fb4c35a82976ab971cc0f2cc6a01af959ee32cb10a801f37f8b762c",
            "tests/test_full_corpus_rerank_oracle.py": "75dd8e835f4f53301e7412c8e9ac826cfdc588dd6f458ff4a4822290c01139e9",
            "tools/run_full_corpus_rerank_oracle_wslc.ps1": "d9c305e22ed4e9e150492baa2c2e0d47f2fdd13bd9ba5aa6997709ab84ce48b6",
            "tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json": "217ad55dffcab57095840c1d69bc17af0f291fc262cd45c5500ec87589fbad08",
            "tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json": "ea4a64fe75b7f415c25fa26375375f07944078f34637838b66e565b6e88c23ca",
            "tests/evidence/full_corpus_rerank_oracle_v1/development.invalidation.json": "2f8d0c8774f44b2655dcd4fe553f8beedb6122ff4cc82c5fe50a67916bd5138b",
            "docs/experiments/full-corpus-rerank-oracle-v2.md": "ddc23414e4ee589c71095b609bb75b11ffc11c73617ea3eeb3be7bc56091e9c8",
            "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py": "ad64dbbf4c118045c8684edf3e3670bed160727e387c847a8c09cb01aaaedfbc",
            "tests/fixtures/full_corpus_rerank_oracle_v2.gold.json": "689028b2a6f827bc915f6d9151c1b6b95164b4dbd59715bb930d573c33c846cf",
            "tests/fixtures/full_corpus_rerank_oracle_v2.manifest.json": "331f25d5bfa7c0caf0355f3124558ee721a40ab0a9e6c235272800cc0aa4c689",
            "tests/fixtures/full_corpus_rerank_oracle_v2.package_init.py": "5fc4c8900b0993b6b599c8beefb3c2b3f45b18cd14c25af2dd725fa7ba12f79f",
            "tests/fixtures/full_corpus_rerank_oracle_v2.query.json": "c7bc3c49310737d6cad024e14ee1388417b46ee2f38b90f6b1c3793737295f23",
            "tests/fixtures/full_corpus_rerank_oracle_v2.schema.json": "ac46bf2417bde0f78292d6f33722af19b339048639a874084ff01ed2326b33ba",
            "tests/test_full_corpus_rerank_oracle_v2.py": "6634dfa190bf0d01083d440bc2fc2e7069936a6b34c164cd134fe9b988d5a3b1",
            "tools/run_full_corpus_rerank_oracle_v2_wslc.ps1": "042615177ab685ef980302f21c72326523ab838876d93935bd40a376370a7969",
            "tests/evidence/full_corpus_rerank_oracle_v2/development.preflight.json": "405663d2e3388e079a76043d1e8370983982f8bff565801a898d7f340f79e9e1",
            "tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json": "416ff3e48305df1f1619a71c6c665fecbf07b18cc6bce8e72295688033b0ed17",
            "tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json": "52f2289de5a2da675fd4f980c9752bda387101c2601184fbd01fdd59642dccd4",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((diagnostic.ROOT / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )


if __name__ == "__main__":
    unittest.main()

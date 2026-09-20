from __future__ import annotations

import hashlib
import json
import unittest

from neuron_graph_rag import e5_structural_centroid_ablation as runner
from neuron_graph_rag import structural_representation_length_bias_diagnostic as structural


class E5StructuralCentroidAblationTests(unittest.TestCase):
    def test_manifest_freezes_model_factorial_and_success_criteria(self) -> None:
        manifest = runner._manifest(runner.ROOT)
        self.assertEqual(manifest["status"], "frozen_pre_registered_execution")
        self.assertEqual([row["arm_id"] for row in manifest["arms"]], list(runner.ARMS))
        self.assertEqual(manifest["model"]["query_prefix"], "query: ")
        self.assertEqual(manifest["model"]["passage_prefix"], "passage: ")
        self.assertEqual(manifest["ranking"]["cutoff"], 20)
        self.assertEqual(manifest["projection"], {"unit": "unicode-codepoint", "window": 480, "overlap": 80, "coverage": "start-to-end"})

    def test_result_free_fastembed_parity_is_frozen(self) -> None:
        parity = runner.read_json(runner.ROOT / runner.PARITY)
        self.assertEqual(parity["status"], "result_free_parity_valid")
        self.assertEqual(parity["registered_query_execution_count"], 0)
        self.assertEqual(parity["max_absolute_difference_tolerance"], 1e-6)
        self.assertEqual(parity["minimum_cosine_similarity"], 0.999999)
        self.assertTrue(all(row["max_absolute_difference"] <= 1e-6 for row in parity["comparisons"]))
        self.assertTrue(all(row["cosine_similarity"] >= 0.999999 for row in parity["comparisons"]))
        self.assertEqual(hashlib.sha256((runner.ROOT / runner.PARITY).read_bytes()).hexdigest(), "a18e9dbc7eba522155d0927dbcafea0b2011c3691e3764dd95c0e526cb1f442c")

    def test_centroid_normalizes_mean_and_zero_norm_fails_closed(self) -> None:
        value = runner._centroid([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(value[0], 2 ** -0.5)
        self.assertAlmostEqual(value[1], 2 ** -0.5)
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            runner._centroid([[1.0, 0.0], [-1.0, 0.0]])

    def test_structural_literal_is_reused_without_chunk_changes(self) -> None:
        text = "# Title\n\n## Part\nbody"
        prefix = structural.structural_prefix(
            "corpora/github-retrieval-parity-v4/docs/example.md", text, text.index("body")
        )
        self.assertEqual(prefix, "path: docs/example.md\nfilename: example.md\ntitle: Title\nheadings: Title > Part\n\n")

    def test_audit_is_result_free_before_registered_execution(self) -> None:
        audit = runner.audit(runner.ROOT)
        self.assertEqual(audit["status"], "result_free_frozen")
        self.assertEqual(audit["registered_query_execution_count"], 0)
        self.assertEqual(audit["parity_status"], "result_free_parity_valid")
        self.assertEqual(audit["evidence"], {"claim": False, "result": False, "error": False})

    def test_wrapper_keeps_gold_out_of_preflight_and_worker(self) -> None:
        wrapper = (runner.ROOT / "tools/run_e5_structural_centroid_ablation_v1_wslc.ps1").read_text(encoding="utf-8")
        before_finalizer = wrapper.split("$goldPath =", 1)[0]
        self.assertNotIn("full_corpus_rerank_oracle_v2.gold.json:ro", before_finalizer)
        self.assertIn("full_corpus_rerank_oracle_v2.gold.json:ro", wrapper)
        self.assertIn("--network", wrapper)
        self.assertIn('"none"', wrapper)

    def test_prior_frozen_assets_are_unchanged(self) -> None:
        expected = {
            "src/neuron_graph_rag/structural_representation_length_bias_ablation.py": "ec5e1d2c9bd7cf0403cf8a715c53dccc19039a1944a7775b7b9a3f5b3776c1fa",
            "src/neuron_graph_rag/structural_representation_length_bias_diagnostic.py": "b117e0446fcbe76890e6278dbf08bbb12d42cf4c87329d9a69e6a366bc95de86",
            "tests/fixtures/structural_representation_length_bias_ablation_v1.manifest.json": "8ca6479d6d51658e92c530bf56eb0d4a63bb156dcccbef485122df5616af29a4",
            "tests/fixtures/structural_representation_length_bias_ablation_v1.schema.json": "09990dfd3d51de5cc5691d5851164293df441bba66462751d86229dd8c2fee81",
            "tests/evidence/structural_representation_length_bias_ablation_v1/development.preflight.json": "41355b6d2f17a3b046af34d079c137c3c6aca0bce392c315d88346ba4f97f01f",
            "tests/evidence/structural_representation_length_bias_ablation_v1/development.claim.json": "e14743bb9ceb3eda8c8b7911678fb12da4af9fe4652e7a770ff606d59ec8d5ba",
            "tests/evidence/structural_representation_length_bias_ablation_v1/development.observed.json": "da37bf495c6061a0f48119b6d679a7f42daaee6b625e41446ba9d361022c70e8",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest(), digest, relative)


if __name__ == "__main__":
    unittest.main()

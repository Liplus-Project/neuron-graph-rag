from __future__ import annotations

import hashlib
import unittest

from neuron_graph_rag import v2_m3_chunk_shortlist_ablation as runner


class V2M3ChunkShortlistAblationTests(unittest.TestCase):
    def test_manifest_freezes_candidate_shortlist_runtime_and_success(self) -> None:
        manifest = runner._manifest(runner.ROOT)
        self.assertEqual(manifest["status"], "frozen_pre_registered_execution")
        self.assertEqual(manifest["stage_1"]["candidate_k"], 50)
        self.assertEqual(manifest["shortlist"]["arms"], [2, 4, 8])
        self.assertEqual(manifest["shortlist"]["max_chunks"], 8)
        self.assertEqual(manifest["stage_2"]["aggregation"]["temperature"], 1.0)
        self.assertEqual(manifest["ranking"]["cutoff"], 20)
        self.assertEqual(manifest["runtime"]["practical_target_seconds"], 600.0)
        self.assertEqual(manifest["execution"]["registered_pipeline_count"], 1)
        self.assertEqual(manifest["execution"]["retry_count"], 0)

    def test_models_are_exactly_the_pinned_e5_and_v2_m3(self) -> None:
        models = runner._models(runner.ROOT)
        self.assertEqual(tuple(models), ("e5", "v2-m3"))
        self.assertEqual(
            (models["e5"]["model_id"], models["e5"]["revision"]),
            ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        )
        self.assertEqual(
            (models["v2-m3"]["model_id"], models["v2-m3"]["revision"]),
            ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
        )
        self.assertTrue(all(len(row["sha256"]) == 64 for model in models.values() for row in model["required_files"]))

    def test_arm_aggregation_uses_prefix_logits_and_source_id_tie_break(self) -> None:
        documents = [
            {
                "source_id": "b",
                "path": "b.md",
                "all_chunk_count": 3,
                "chunks": [{"raw_logit": 2.0}, {"raw_logit": 0.0}, {"raw_logit": -1.0}],
            },
            {
                "source_id": "a",
                "path": "a.md",
                "all_chunk_count": 1,
                "chunks": [{"raw_logit": 2.0}],
            },
        ]
        arms = runner._shortlist_arms(documents)
        self.assertEqual([row["m"] for row in arms], [2, 4, 8])
        self.assertEqual(arms[0]["ranking"][0]["source_id"], "a")
        self.assertEqual(arms[0]["ranking"][0]["selected_chunk_count"], 1)
        self.assertEqual(arms[1]["ranking"][1]["selected_chunk_count"], 3)
        self.assertEqual(arms[1]["ranking"], arms[2]["ranking"])

    def test_shortlist_orders_by_cosine_then_chunk_index(self) -> None:
        chunks = [
            {"chunk_index": 3, "query_cosine": 0.8},
            {"chunk_index": 1, "query_cosine": 0.8},
            {"chunk_index": 0, "query_cosine": 0.7},
        ]
        self.assertEqual(
            [row["chunk_index"] for row in runner._ordered_shortlist(chunks)],
            [1, 3, 0],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner._ordered_shortlist([chunks[0], chunks[0]])

    def test_wrapper_scores_stage2_before_gold_and_mounts_volume_root(self) -> None:
        wrapper = (runner.ROOT / "tools/run_v2_m3_chunk_shortlist_ablation_v1_wslc.ps1").read_text(encoding="utf-8")
        run_block = wrapper.split("$ceArguments =", 1)[1]
        self.assertLess(run_block.index('"stage2"'), run_block.index("$copyGold ="))
        self.assertLess(run_block.index("$copyGold ="), run_block.index('"finalize"'))
        self.assertIn('--volume ${volume}:${containerRoot}', wrapper)
        self.assertIn("-C ${containerSource}/tests/fixtures", wrapper)
        self.assertIn('"--network", "none"', wrapper)

    def test_audit_is_result_free_before_registered_execution(self) -> None:
        audit = runner.audit(runner.ROOT)
        self.assertEqual(audit["status"], "result_free_frozen")
        self.assertEqual(audit["registered_pipeline_count"], 0)
        self.assertEqual(audit["holdout_bearing_input_file_count"], 0)
        self.assertFalse(any(audit["evidence"].values()))

    def test_242_frozen_protocol_and_result_are_unchanged(self) -> None:
        expected = {
            "src/neuron_graph_rag/practical_two_stage_retrieval.py": "e5635376019e3e64e00d6a61512359d58e43f3e0ed7ef6b57e8f891dce1ce405",
            "src/neuron_graph_rag/practical_two_stage_retrieval_recovery_v2.py": "deaf5bb4c39b5e55ea04875a6b878111329f1770f91cbfe0a888d2d7b90a65ba",
            "tests/fixtures/practical_two_stage_retrieval_v1.manifest.json": "a4bd0b18b2042f843b981e4879239a5191156efc6658a2dff851dfe522a7552f",
            "tests/fixtures/practical_two_stage_retrieval_recovery_v2.manifest.json": "6ec9504cd48bd971c23573ec892679e836fb191cd3446e4ed25e8f6c8472fd6f",
            "tests/evidence/practical_two_stage_retrieval_finalizer_recovery_v2/development.recovered.json": "7166caaac4c1c581140711231de4d373125fad5daeac72b83c199109ea75311e",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest(), digest, relative)


if __name__ == "__main__":
    unittest.main()

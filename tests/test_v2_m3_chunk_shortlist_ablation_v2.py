from __future__ import annotations

import hashlib
import unittest

from neuron_graph_rag import v2_m3_chunk_shortlist_ablation_v2 as runner


class V2M3ChunkShortlistAblationV2Tests(unittest.TestCase):
    def test_v2_changes_only_the_failed_allowlist_boundary(self) -> None:
        manifest = runner._manifest(runner.ROOT)
        self.assertEqual(manifest["status"], "frozen_pre_registered_execution")
        self.assertEqual(manifest["registered_query_execution_count"], 0)
        self.assertEqual(manifest["stage_1"]["candidate_k"], 50)
        self.assertEqual(manifest["shortlist"]["arms"], [2, 4, 8])
        self.assertEqual(manifest["shortlist"]["max_chunks"], 8)
        self.assertEqual(manifest["stage_2"]["aggregation"]["temperature"], 1.0)
        self.assertEqual(manifest["ranking"]["cutoff"], 20)
        self.assertEqual(manifest["runtime"]["practical_target_seconds"], 600.0)
        self.assertEqual(manifest["execution"]["retry_count"], 0)
        self.assertNotIn(runner.EVIDENCE.as_posix(), manifest["forbidden_registered_paths"])
        self.assertIn((runner.EVIDENCE / "development.observed.json").as_posix(), manifest["stage_forbidden_registered_paths"]["worker"])
        self.assertNotIn((runner.EVIDENCE / "development.claim.json").as_posix(), manifest["stage_forbidden_registered_paths"]["worker"])

    def test_predecessor_failure_evidence_is_hash_locked(self) -> None:
        manifest = runner._manifest(runner.ROOT)
        for relative, digest in manifest["predecessor_failure_evidence_sha256"].items():
            self.assertEqual(hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_models_are_exactly_the_pinned_e5_and_v2_m3(self) -> None:
        models = runner._models(runner.ROOT)
        self.assertEqual(tuple(models), ("e5", "v2-m3"))
        self.assertEqual(models["e5"]["revision"], "614241f622f53c4eeff9890bdc4f31cfecc418b3")
        self.assertEqual(models["v2-m3"]["revision"], "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")

    def test_shortlist_and_arm_derivation_are_unchanged(self) -> None:
        chunks = [
            {"chunk_index": 3, "query_cosine": 0.8},
            {"chunk_index": 1, "query_cosine": 0.8},
            {"chunk_index": 0, "query_cosine": 0.7},
        ]
        self.assertEqual([row["chunk_index"] for row in runner._ordered_shortlist(chunks)], [1, 3, 0])
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
        self.assertEqual(arms[1]["ranking"], arms[2]["ranking"])

    def test_wrapper_keeps_gold_after_complete_workers(self) -> None:
        wrapper = (runner.ROOT / "tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1").read_text(encoding="utf-8")
        run_block = wrapper.split("$ceArguments =", 1)[1]
        self.assertLess(run_block.index('"stage1"'), run_block.index('"stage2"'))
        self.assertLess(run_block.index('"stage2"'), run_block.index("$copyGold ="))
        self.assertLess(run_block.index("$copyGold ="), run_block.index('"finalize"'))
        self.assertIn('"--network", "none"', wrapper)

    def test_registered_execution_is_complete_and_auditable(self) -> None:
        audit = runner.audit(runner.ROOT)
        self.assertEqual(audit["status"], "observed_valid")
        self.assertEqual(audit["registered_pipeline_count"], 1)
        self.assertEqual(
            audit["evidence"],
            {"preflight": True, "claim": True, "stage1": True, "stage2": True, "result": True, "error": False},
        )
        self.assertTrue(audit["quality_primary_success"])
        self.assertTrue(audit["practical_success"])
        self.assertTrue(audit["combined_success"])

    def test_observed_result_and_worker_packets_are_hash_locked(self) -> None:
        expected = {
            "development.claim.json": "2c0eb18d36b864b2ab64a7b953d950545b73f426ae0bc06a4fbb3e5f87a75577",
            "development.observed.json": "44fa02d88c790c762c238732732121790606fc50f73acc6eff43abc2f50c3d4c",
            "development.preflight.json": "f1c93c8724f0e27cb9dcdd9150d5bd7bccf1663cf61913ee37a7b163eda8cf84",
            "development.stage1.worker.json": "5cd4779c744f3fc681a90b1b33e8df1f0fd232a617449bab9727455608bc4a82",
            "development.stage2.worker.json": "d75e20c7f242470976e0ad755efa0b9a254a28199553a3dd3123ee85c092286c",
        }
        for name, digest in expected.items():
            evidence = runner.ROOT / runner.EVIDENCE / name
            self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), digest, name)

        observed = runner.read_json(runner.ROOT / runner.RESULT)
        self.assertEqual(observed["source_commit"], "6df8d836a464788ed818d8d408adecdbc3bea7a2")
        self.assertEqual(observed["stage1_expected_source_rank"], 39)
        self.assertEqual([row["expected_source_rank"] for row in observed["arms"]], [19, 22, 21])
        self.assertEqual(observed["selected_pair_count"], 370)
        self.assertEqual(observed["forward_batch_count"], 47)
        self.assertAlmostEqual(observed["pipeline_runtime_seconds"], 223.94103522299997)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import unittest

from neuron_graph_rag import practical_two_stage_retrieval as runner
from neuron_graph_rag import practical_two_stage_retrieval_recovery as recovery


class PracticalTwoStageRetrievalTests(unittest.TestCase):
    def test_manifest_freezes_pipeline_models_runtime_and_success(self) -> None:
        manifest = runner._manifest(runner.ROOT)
        self.assertEqual(manifest["status"], "frozen_pre_registered_execution")
        self.assertEqual(manifest["stage_1"]["candidate_k"], 50)
        self.assertEqual(manifest["stage_1"]["representation"], "structural")
        self.assertEqual(manifest["stage_2"]["aggregation"]["temperature"], 1.0)
        self.assertEqual(
            [row["kind"] for row in manifest["stage_2"]["models"]],
            ["minilm", "v2-m3"],
        )
        self.assertEqual(manifest["ranking"]["cutoff"], 20)
        self.assertEqual(manifest["runtime"]["practical_target_seconds_per_pipeline"], 600.0)
        self.assertEqual(manifest["execution"]["registered_pipeline_count"], 1)
        self.assertEqual(manifest["execution"]["retry_count"], 0)

    def test_minilm_parity_gate_is_result_free_and_frozen(self) -> None:
        parity = runner.read_json(runner.ROOT / runner.PARITY)
        self.assertEqual(parity["status"], "result_free_parity_valid")
        self.assertEqual(parity["registered_query_execution_count"], 0)
        self.assertTrue(parity["token_fields_equal"])
        self.assertEqual(parity["max_absolute_difference_tolerance"], 1e-6)
        self.assertLessEqual(max(parity["absolute_differences"]), 1e-6)
        self.assertEqual(
            hashlib.sha256((runner.ROOT / runner.PARITY).read_bytes()).hexdigest(),
            "69448b2be0a5fb08b621e261d93d1610039663f6d19b1d8eae17eed65eadd3b0",
        )

    def test_model_registry_fixes_canonical_ids_revisions_and_all_file_hashes(self) -> None:
        models = runner._models(runner.ROOT)
        self.assertEqual(
            (models["minilm"]["model_id"], models["minilm"]["revision"]),
            (
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            ),
        )
        self.assertEqual(models["v2-m3"]["revision"], "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
        for model in models.values():
            self.assertTrue(model["required_files"])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in model["required_files"]))

    def test_nlme_and_tie_break_are_inherited(self) -> None:
        self.assertAlmostEqual(runner.ce._nlme([2.0]), 2.0)
        self.assertAlmostEqual(runner.ce._nlme([2.0, 2.0]), 2.0)
        self.assertLess(runner.ce._nlme([2.0, -10.0]), 2.0)
        self.assertEqual(runner._manifest(runner.ROOT)["inherited_contract"]["tie_break"], "source_id ascending")

    def test_wrapper_keeps_gold_out_until_both_gold_blind_workers_finish(self) -> None:
        wrapper = (runner.ROOT / "tools/run_practical_two_stage_retrieval_v1_wslc.ps1").read_text(encoding="utf-8")
        run_block = wrapper.split("$ceArguments =", 1)[1]
        gold_copy = run_block.index("$copyGold =")
        self.assertLess(run_block.index('"--kind", $kind'), gold_copy)
        self.assertLess(gold_copy, run_block.index('"finalize"'))
        self.assertIn('"--network", "none"', wrapper)
        self.assertIn("github-practical-two-stage-retrieval-v1-runtime", wrapper)

    def test_source_protocol_failure_evidence_is_append_only(self) -> None:
        audit = runner.audit(runner.ROOT)
        self.assertEqual(audit["status"], "result_free_frozen")
        self.assertEqual(audit["registered_pipeline_count"], 0)
        self.assertEqual(audit["evidence"], {"claim": True, "result": False, "error": True})
        expected = recovery._manifest(recovery.ROOT)["source_evidence_sha256"]
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_recovery_manifest_freezes_zero_inference_and_gold_absent_claim(self) -> None:
        manifest = recovery._manifest(recovery.ROOT)
        self.assertEqual(manifest["status"], "frozen_before_gold_mount")
        self.assertEqual(manifest["registered_query_execution_count"], 0)
        self.assertEqual(manifest["model_forward_inference_count"], 0)
        self.assertEqual(manifest["retry_count"], 0)
        self.assertNotIn(recovery.GOLD.as_posix(), manifest["claim_registered_files"])
        self.assertEqual(manifest["ranking"]["candidate_k"], 50)
        self.assertEqual(manifest["ranking"]["temperature"], 1.0)
        self.assertEqual(manifest["ranking"]["practical_target_seconds_per_pipeline"], 600.0)

    def test_recovery_verifies_complete_packets_without_gold(self) -> None:
        stage1, workers, diagnostics = recovery._verify_packets(recovery.ROOT)
        self.assertEqual(len(stage1["ranking"]), 93)
        self.assertEqual(len(stage1["candidate_source_ids"]), 50)
        self.assertEqual([row["kind"] for row in workers], ["minilm", "v2-m3"])
        self.assertEqual({row["kind"]: len(row["ranking"]) for row in workers}, {"minilm": 50, "v2-m3": 50})
        self.assertEqual(set(diagnostics["stage2"]), {"minilm", "v2-m3"})

    def test_recovery_audit_is_frozen_before_gold_mount(self) -> None:
        audit = recovery.audit(recovery.ROOT)
        self.assertEqual(audit["status"], "recovery_frozen")
        self.assertEqual(audit["registered_query_execution_count"], 0)
        self.assertEqual(audit["model_forward_inference_count"], 0)
        self.assertEqual(audit["source_protocol_status"], "failed_finalizer_gold_stream_path")

    def test_recovery_wrapper_adds_gold_only_after_successful_claim(self) -> None:
        wrapper = (recovery.ROOT / "tools/run_practical_two_stage_retrieval_recovery_v1.ps1").read_text(encoding="utf-8")
        claim_block = wrapper.split('if ($Phase -eq "claim")', 1)[1].split('if ($Phase -eq "finalize")', 1)[0]
        finalize_block = wrapper.split('if ($Phase -eq "finalize")', 1)[1].split('if ($Phase -eq "audit")', 1)[0]
        self.assertNotIn("Copy-ExclusiveFile $goldRelative", claim_block)
        self.assertIn("Copy-ExclusiveFile $goldRelative", finalize_block)
        self.assertIn("successful recovery claim is required", finalize_block)

    def test_prior_238_and_240_frozen_assets_are_unchanged(self) -> None:
        expected = {
            "src/neuron_graph_rag/structural_representation_length_bias_ablation.py": "ec5e1d2c9bd7cf0403cf8a715c53dccc19039a1944a7775b7b9a3f5b3776c1fa",
            "tests/fixtures/structural_representation_length_bias_ablation_v1.manifest.json": "8ca6479d6d51658e92c530bf56eb0d4a63bb156dcccbef485122df5616af29a4",
            "tests/evidence/structural_representation_length_bias_ablation_v1/development.observed.json": "da37bf495c6061a0f48119b6d679a7f42daaee6b625e41446ba9d361022c70e8",
            "src/neuron_graph_rag/e5_structural_centroid_ablation.py": "0afa4abf198c411fe0f577ad22fcdc2af3960c3ea35453a32b72443480063e7e",
            "src/neuron_graph_rag/e5_structural_centroid_recovery_v2.py": "b59840c3bac6438c5563986c3b08caf078264ec965f33d125da0e219445e2386",
            "tests/fixtures/e5_structural_centroid_ablation_v1.manifest.json": "b2701a1305eb95bc73fafcb0d33e0e9ece0bf6a8c599759493e2581675b03711",
            "tests/fixtures/e5_structural_centroid_recovery_v2.manifest.json": "0f7a53e253bd35e712071e3103dedccb97217643339f34138ec7bc2182bd6188",
            "tests/evidence/e5_structural_centroid_ablation_v1/development.worker.json": "8321809a4ff8fb11e1120d49f9159d1e900862c80e3798467323cb10e7cd6882",
            "tests/evidence/e5_structural_centroid_finalizer_recovery_v2/development.recovered.json": "0afbe83f9e494b036faaa110e9296e48183df075069dda6f400df5995c1c5b24",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )


if __name__ == "__main__":
    unittest.main()

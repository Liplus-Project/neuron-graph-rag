from __future__ import annotations

import hashlib
import json
import unittest

from neuron_graph_rag import e5_structural_centroid_recovery as recovery
from neuron_graph_rag import e5_structural_centroid_recovery_v2 as recovery_v2


class E5StructuralCentroidRecoveryTests(unittest.TestCase):
    def test_manifest_freezes_finalizer_only_boundary(self) -> None:
        manifest = json.loads(
            (recovery.ROOT / recovery.MANIFEST).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "frozen_before_gold_mount")
        self.assertEqual(manifest["registered_query_execution_count"], 0)
        self.assertEqual(manifest["model_forward_inference_count"], 0)
        self.assertEqual(manifest["retry_count"], 0)
        self.assertEqual(
            manifest["packet_completeness"]["document_count_per_representation"], 93
        )
        self.assertEqual(
            manifest["packet_completeness"]["passage_count_per_representation"], 2065
        )
        self.assertEqual(
            manifest["development_gold_sha256"],
            "689028b2a6f827bc9151c1b6b95164b4dbd59715bb930d573c33c846cf",
        )

    def test_failed_v1_evidence_is_hash_locked_without_a_result(self) -> None:
        manifest = json.loads(
            (recovery.ROOT / recovery.MANIFEST).read_text(encoding="utf-8")
        )
        for relative, digest in manifest["source_evidence_sha256"].items():
            self.assertEqual(
                hashlib.sha256((recovery.ROOT / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )
        self.assertFalse(
            (
                recovery.ROOT / recovery.SOURCE_EVIDENCE / "development.observed.json"
            ).exists()
        )

    def test_claim_allowlist_excludes_gold_and_mixed_inputs(self) -> None:
        manifest = json.loads(
            (recovery.ROOT / recovery.MANIFEST).read_text(encoding="utf-8")
        )
        self.assertNotIn(recovery.GOLD.as_posix(), manifest["claim_registered_files"])
        self.assertIn(
            "tests/fixtures/github_retrieval_parity_v5.gold.json",
            manifest["forbidden_registered_paths"],
        )
        self.assertIn(
            "tests/fixtures/github_retrieval_parity_v5.queries.json",
            manifest["forbidden_registered_paths"],
        )
        self.assertIn(
            "tests/fixtures/full_corpus_rerank_oracle_v2.query.json",
            manifest["forbidden_registered_paths"],
        )

    def test_runner_has_no_model_runtime_and_reads_gold_only_in_finalize_path(
        self,
    ) -> None:
        source = (
            recovery.ROOT / "src/neuron_graph_rag/e5_structural_centroid_recovery.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("onnxruntime", source)
        self.assertNotIn("tokenizers", source)
        claim_body = source.split("def claim(", 1)[1].split("def _gold(", 1)[0]
        self.assertNotIn("_gold(", claim_body)
        self.assertNotIn("read_json(root / GOLD)", claim_body)

    def test_wrapper_mounts_gold_only_after_successful_claim(self) -> None:
        wrapper = (
            recovery.ROOT / "tools/run_e5_structural_centroid_recovery_v1.ps1"
        ).read_text(encoding="utf-8")
        claim_section = wrapper.split('if ($Phase -eq "claim")', 1)[1].split(
            'if ($Phase -eq "finalize")', 1
        )[0]
        finalize_section = wrapper.split('if ($Phase -eq "finalize")', 1)[1].split(
            'if ($Phase -eq "audit")', 1
        )[0]
        self.assertIn("gold must be absent before recovery claim", claim_section)
        self.assertNotIn("Copy-ExclusiveFile $goldRelative", claim_section)
        self.assertIn(
            "successful recovery claim is required before gold mount", finalize_section
        )
        self.assertIn("Copy-ExclusiveFile $goldRelative", finalize_section)

    def test_recovery_schema_names_derived_status(self) -> None:
        schema = json.loads(
            (recovery.ROOT / recovery.SCHEMA).read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["properties"]["status"]["const"],
            "recovered_from_complete_gold_blind_worker_packet",
        )
        self.assertEqual(
            schema["properties"]["registered_query_execution_count"]["const"], 0
        )
        self.assertEqual(
            schema["properties"]["model_forward_inference_count"]["const"], 0
        )

    def test_v2_changes_only_the_failed_gold_hash_contract(self) -> None:
        v1 = json.loads((recovery.ROOT / recovery.MANIFEST).read_text(encoding="utf-8"))
        v2 = json.loads(
            (recovery_v2.ROOT / recovery_v2.MANIFEST).read_text(encoding="utf-8")
        )
        self.assertEqual(v1["packet_completeness"], v2["packet_completeness"])
        self.assertEqual(v1["dependencies"], v2["dependencies"])
        self.assertEqual(v1["ranking"], v2["ranking"])
        self.assertEqual(len(v1["development_gold_sha256"]), 58)
        self.assertEqual(len(v2["development_gold_sha256"]), 64)
        self.assertEqual(
            v2["development_gold_sha256"],
            hashlib.sha256(
                (recovery_v2.ROOT / recovery_v2.impl.GOLD).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(v2["registered_query_execution_count"], 0)
        self.assertEqual(v2["model_forward_inference_count"], 0)

    def test_v2_hash_locks_failed_v1_recovery_without_a_result(self) -> None:
        manifest = json.loads(
            (recovery_v2.ROOT / recovery_v2.MANIFEST).read_text(encoding="utf-8")
        )
        for relative, digest in manifest["recovery_v1_evidence_sha256"].items():
            self.assertEqual(
                hashlib.sha256((recovery_v2.ROOT / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )
        self.assertFalse(
            (
                recovery_v2.ROOT
                / recovery_v2.V1_RECOVERY_EVIDENCE
                / "development.recovered.json"
            ).exists()
        )

    def test_v2_claim_allowlist_keeps_gold_absent(self) -> None:
        manifest = json.loads(
            (recovery_v2.ROOT / recovery_v2.MANIFEST).read_text(encoding="utf-8")
        )
        self.assertNotIn(
            recovery_v2.impl.GOLD.as_posix(), manifest["claim_registered_files"]
        )
        self.assertIn(
            "tests/evidence/e5_structural_centroid_finalizer_recovery_v1/development.error.json",
            manifest["claim_registered_files"],
        )


if __name__ == "__main__":
    unittest.main()

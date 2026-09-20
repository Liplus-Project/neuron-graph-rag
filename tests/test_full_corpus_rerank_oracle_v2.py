from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import full_corpus_rerank_oracle_v2 as oracle


class FullCorpusRerankOracleV2Tests(unittest.TestCase):
    def test_development_bundle_is_split_and_contains_no_stage_container(self) -> None:
        query = json.loads((oracle.ROOT / oracle.QUERY).read_text(encoding="utf-8"))
        gold = json.loads((oracle.ROOT / oracle.GOLD).read_text(encoding="utf-8"))
        self.assertEqual(query["query"], oracle.EXPECTED_QUERY)
        self.assertNotIn("expected_source_id", query)
        self.assertNotIn("stages", query)
        self.assertEqual(gold["expected_source_id"], oracle.EXPECTED_SOURCE_ID)
        self.assertNotIn("query", gold)
        self.assertNotIn("stages", gold)

    def test_manifest_freezes_v1_settings_and_worker_allowlist(self) -> None:
        manifest = oracle._manifest(oracle.ROOT)
        self.assertEqual(manifest["corpus_document_count"], 93)
        self.assertEqual(manifest["rerank_cutoff"], 20)
        self.assertEqual(manifest["projection"]["window"], 480)
        self.assertEqual(manifest["projection"]["overlap"], 80)
        self.assertEqual(manifest["runtime"]["cpu_threads"], 4)
        self.assertEqual(manifest["runtime"]["worker_gold_mount"], "absent")
        self.assertNotIn(oracle.GOLD.as_posix(), manifest["worker_registered_files"])
        self.assertIn(
            "tests/fixtures/github_retrieval_parity_v5.queries.json",
            manifest["forbidden_registered_paths"],
        )
        self.assertIn(
            "tests/fixtures/github_retrieval_parity_v5.gold.json",
            manifest["forbidden_registered_paths"],
        )

    def test_registered_tree_rejects_gold_and_any_extra_file_for_worker(self) -> None:
        manifest = oracle._manifest(oracle.ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for destination in oracle.WORKER_REGISTERED_FILES:
                source = Path(manifest["copy_map"][destination.as_posix()])
                target = root / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(oracle.ROOT / source, target)
            hashes = oracle._validate_registered_tree(root, stage="preflight")
            self.assertEqual(set(hashes), {path.as_posix() for path in oracle.WORKER_REGISTERED_FILES})
            gold = root / oracle.GOLD
            gold.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(oracle.ROOT / oracle.GOLD, gold)
            with self.assertRaisesRegex(ValueError, "registered file allowlist mismatch"):
                oracle._validate_registered_tree(root, stage="preflight")

    def test_worker_never_reads_or_receives_gold(self) -> None:
        source = inspect.getsource(oracle.worker)
        self.assertNotIn("_gold(", source)
        self.assertNotIn("read_json(root / GOLD)", source)
        wrapper = (oracle.ROOT / "tools/run_full_corpus_rerank_oracle_v2_wslc.ps1").read_text(
            encoding="utf-8"
        )
        worker_segment = wrapper.split('foreach ($kind in @("base", "v2-m3"))', 1)[1].split(
            "$goldPath", 1
        )[0]
        self.assertNotIn("full_corpus_rerank_oracle_v2.gold.json", worker_segment)
        self.assertIn("full_corpus_rerank_oracle_v2.gold.json:ro", wrapper)

    def test_schema_records_zero_holdout_bearing_inputs(self) -> None:
        schema = json.loads((oracle.ROOT / oracle.SCHEMA).read_text(encoding="utf-8"))
        environment = schema["$defs"]["environment"]
        self.assertEqual(
            environment["properties"]["holdout_bearing_input_file_count"]["const"], 0
        )
        self.assertEqual(schema["properties"]["corpus_document_count"]["const"], 93)
        self.assertEqual(schema["$defs"]["model"]["properties"]["documents"]["minItems"], 93)

    def test_probe_and_result_free_audit_execute_no_registered_query(self) -> None:
        probe = oracle.probe(oracle.ROOT)
        self.assertEqual(probe["registered_query_execution_count"], 0)
        self.assertEqual(probe["model_forward_inference_count"], 0)
        audit = oracle.audit(oracle.ROOT)
        self.assertEqual(audit["status"], "result_free")
        self.assertEqual(audit["holdout_bearing_input_file_count"], 0)
        self.assertFalse(audit["worker_gold_present"])

    def test_v1_owned_files_are_byte_for_byte_unchanged(self) -> None:
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
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((oracle.ROOT / relative).read_bytes()).hexdigest(), digest,
                relative,
            )


if __name__ == "__main__":
    unittest.main()

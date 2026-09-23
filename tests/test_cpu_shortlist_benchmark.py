from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v1.json"
RESULT = ROOT / "tests/evidence/cpu_shortlist_benchmark_v1/observed.json"
V2_MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v2.json"
V2_RESULT = ROOT / "tests/evidence/cpu_shortlist_benchmark_v2/observed.json"
V1_INTERRUPTED = ROOT / "tests/evidence/cpu_shortlist_benchmark_v1/interrupted.json"


class CpuShortlistBenchmarkTests(unittest.TestCase):
    def test_v1_interruption_preserves_unassessed_result(self):
        evidence = json.loads(V1_INTERRUPTED.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "interrupted_without_result")
        self.assertEqual(evidence["quality_and_runtime"], "not_assessed")
        self.assertFalse(RESULT.exists())

    def test_result_free_contract_or_observed_result(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "fixed_before_observation")
        self.assertEqual(manifest["runtime"], {
            "cpu_threads": 4,
            "warm_query_target_seconds": 60.0,
            "candidate_k": 50,
            "chunks_per_document": 2,
            "max_forward_pairs": 100,
        })
        corpus = ROOT / manifest["corpus"]["path"]
        self.assertEqual(hashlib.sha256(corpus.read_bytes()).hexdigest(), manifest["corpus"]["sha256"])
        self.assertEqual(len(manifest["queries"]), 3)
        self.assertEqual(len({row["case_id"] for row in manifest["queries"]}), 3)
        self.assertTrue(all("v5" in path for path in manifest["excluded_inputs"]))
        if not RESULT.exists():
            return
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["protocol_id"], manifest["protocol_id"])
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
        self.assertEqual(result["corpus_sha256"], manifest["corpus"]["sha256"])
        self.assertEqual(len(result["cases"]), 3)
        for case in result["cases"]:
            self.assertLessEqual(case["diagnostics"]["forward_pairs"], 100)
            self.assertEqual(case["diagnostics"]["candidate_k"], 50)
            self.assertEqual(case["diagnostics"]["chunks_per_document"], 2)

    def test_v2_frozen_protocol_and_optional_result(self):
        v1 = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "fixed_before_observation")
        self.assertEqual(manifest["corpus"], v1["corpus"])
        self.assertEqual(manifest["queries"], v1["queries"])
        self.assertEqual(manifest["quality_gate"], v1["quality_gate"])
        for key in ("cpu_threads", "warm_query_target_seconds", "candidate_k", "chunks_per_document", "max_forward_pairs"):
            self.assertEqual(manifest["runtime"][key], v1["runtime"][key])
        if not V2_RESULT.exists():
            return
        result = json.loads(V2_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["protocol_id"], manifest["protocol_id"])
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(V2_MANIFEST.read_bytes()).hexdigest())
        self.assertEqual(result["environment"]["packages"], manifest["runtime"]["packages"])
        self.assertEqual(len(result["cases"]), 3)
        for case in result["cases"]:
            if "diagnostics" in case:
                self.assertLessEqual(case["diagnostics"]["forward_pairs"], 100)
            else:
                self.assertEqual(case["error"], "SearchTimeout")


if __name__ == "__main__":
    unittest.main()

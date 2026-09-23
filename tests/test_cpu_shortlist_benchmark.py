from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/cpu_shortlist_benchmark_v1.json"
RESULT = ROOT / "tests/evidence/cpu_shortlist_benchmark_v1/observed.json"


class CpuShortlistBenchmarkTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from neuron_graph_rag.github_retrieval_parity import (
    GATE_IDS,
    MANIFEST_PATH,
    PROTOCOL_ID,
    load_protocol,
    verify_registered_result,
)

ROOT = Path(__file__).resolve().parents[1]
STEM = "github_retrieval_parity_v1"
CAPTURE_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.development.capture.json"
CLAIM_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.development.claim.json"
RESULT_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.development.observed.json"
AUDIT_PATH = ROOT / "tests" / "fixtures" / f"{STEM}.observation-audit.json"
OBSERVATION_DOC = ROOT / "docs" / "github-retrieval-parity-observation-v1.md"
PROTOCOL_COMMIT = "b3cc03a15b81f0e395ae564387a46fe57d320f31"


class GitHubRetrievalParityObservationTest(unittest.TestCase):
    def test_development_evidence_is_registered_and_immutable(self) -> None:
        verify_registered_result("development")
        protocol = load_protocol()
        manifest = protocol["manifest"]
        capture = self._read_json(CAPTURE_PATH)
        claim = self._read_json(CLAIM_PATH)
        result = self._read_json(RESULT_PATH)

        self.assertEqual(capture["protocol_id"], PROTOCOL_ID)
        self.assertEqual(capture["protocol_commit"], PROTOCOL_COMMIT)
        self.assertEqual(capture["stage"], "development")
        self.assertEqual(len(capture["cases"]), 4)
        self.assertEqual(
            [row["case_id"] for row in capture["cases"]],
            [row["case_id"] for row in protocol["queries"]["stages"]["development"]],
        )
        for row in capture["cases"]:
            keyword = row["raw_search"]["results"]
            stored = row["raw_stored_content"]["results"]
            self.assertEqual(len(keyword), 5)
            self.assertEqual(len(stored), 5)
            self.assertEqual(
                {item["vector_id"] for item in keyword},
                {item["vector_id"] for item in stored},
            )
            self.assertEqual(row["raw_stored_content"]["not_found"], [])

        self.assertEqual(claim["protocol_commit"], PROTOCOL_COMMIT)
        self.assertIs(claim["one_time_claim"], True)
        self.assertEqual(
            claim["capture_sha256"],
            hashlib.sha256(CAPTURE_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(result["protocol_commit"], PROTOCOL_COMMIT)
        self.assertEqual(result["protocol_hashes"], manifest["artifact_sha256"])
        self.assertEqual(result["raw_github_rag_mcp_capture"], capture)

    def test_failed_gates_close_holdout_without_erasing_other_results(self) -> None:
        protocol = load_protocol()
        manifest = protocol["manifest"]
        result = self._read_json(RESULT_PATH)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_code"], "hard-gate-failed")
        self.assertIs(result["all_hard_gates_pass"], False)
        self.assertEqual([gate["gate_id"] for gate in result["gates"]], list(GATE_IDS))
        self.assertEqual(
            {gate["gate_id"] for gate in result["gates"] if not gate["passed"]},
            {
                "negative-control-non-regression",
                "expected-source-top-k-completeness",
            },
        )
        self.assertTrue(result["deterministic_replay"]["passed"])
        self.assertTrue(result["update_following"]["passed"])
        for kind in ("capture", "claim", "result"):
            self.assertFalse((ROOT / manifest["outputs"]["holdout"][kind]).exists())

    def test_observation_audit_records_one_time_calls_and_shared_db_invariance(
        self,
    ) -> None:
        audit = self._read_json(AUDIT_PATH)
        self.assertEqual(audit["protocol_id"], PROTOCOL_ID)
        self.assertEqual(audit["protocol_commit"], PROTOCOL_COMMIT)
        self.assertEqual(audit["outcome"], "unsupported")
        self.assertEqual(audit["development"]["search_calls"], 4)
        self.assertEqual(audit["development"]["stored_content_fetch_calls"], 4)
        self.assertIs(audit["development"]["capture_registered_once"], True)
        self.assertIs(audit["development"]["stage_executed_once"], True)
        self.assertEqual(audit["holdout"]["search_calls"], 0)
        self.assertEqual(audit["holdout"]["stored_content_fetch_calls"], 0)
        self.assertIs(audit["holdout"]["stage_executed"], False)
        database = audit["shared_database"]
        self.assertEqual(database["sha256_before"], database["sha256_after"])
        self.assertEqual(database["bytes_before"], database["bytes_after"])
        self.assertIs(database["unchanged"], True)
        self.assertIs(audit["safety"]["production_mutation"], False)
        self.assertIs(audit["safety"]["feedback_connected"], False)
        for relative, expected in audit["artifacts"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
            )

    def test_observation_artifacts_are_canonical_utf8_and_scope_is_limited(
        self,
    ) -> None:
        for path in (CAPTURE_PATH, CLAIM_PATH, RESULT_PATH, AUDIT_PATH):
            raw = path.read_bytes()
            json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
        for path in (CLAIM_PATH, RESULT_PATH, AUDIT_PATH):
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        text = OBSERVATION_DOC.read_text(encoding="utf-8", errors="strict")
        self.assertIn("`unsupported`", text)
        self.assertIn("holdout は観測していない", text)
        self.assertIn("固定 repository commit の12文書", text)
        self.assertTrue(MANIFEST_PATH.exists())

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
        if not isinstance(value, dict):
            raise AssertionError(f"JSON object required: {path}")
        return value


if __name__ == "__main__":
    unittest.main()

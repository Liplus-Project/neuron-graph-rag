from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import github_retrieval_parity_v5_observation as observation

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests/evidence/github_retrieval_parity_v5"


def _json(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8", errors="strict"))


class GitHubRetrievalParityV5ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = observation.verify_observation(ROOT)
        cls.preflight = _json("development.preflight.json")
        cls.capture = _json("development.capture.json")
        cls.result = _json("development.observed.json")

    def test_evidence_validates_against_untouched_freeze(self) -> None:
        self.assertEqual(self.summary["freeze_commit"], observation.FREEZE_COMMIT)
        self.assertEqual(self.summary["phase"], "development-closed")
        self.assertEqual(self.summary["preflight_document_count"], 93)
        self.assertEqual(self.summary["registered_query_execution_count"], 5)
        self.assertFalse(self.summary["holdout_opened"])

        manifest = _json_fixture("github_retrieval_parity_v5.manifest.json")
        for relative in (
            "src/neuron_graph_rag/github_retrieval_parity_v5.py",
            "tests/test_github_retrieval_parity_v5.py",
        ):
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, manifest["artifact_sha256"][relative])

    def test_queryless_preflight_and_registered_requests_are_exact(self) -> None:
        documents = self.preflight["documents"]
        paths = [row["doc_path"] for row in documents]
        self.assertEqual(len(paths), 93)
        self.assertEqual(len(set(paths)), 93)
        self.assertTrue(
            all(path.startswith("corpora/github-retrieval-parity-v4/") for path in paths)
        )

        defaults = {
            "repo": "Liplus-Project/neuron-graph-rag",
            "type": "doc",
            "path_prefix": "corpora/github-retrieval-parity-v4/",
            "top_k": 10,
            "fusion": "rrf",
            "rerank": True,
            "graph_expand": True,
            "graph_hops": 2,
        }
        self.assertEqual(len(self.capture["cases"]), 5)
        for case in self.capture["cases"]:
            request = dict(case["request"])
            request.pop("query")
            self.assertEqual(request, defaults)
            self.assertEqual(len(case["raw_search"]["results"]), 10)
            self.assertEqual(case["raw_stored_content"]["not_found"], [])

    def test_metrics_and_hard_gate_failure_are_preserved(self) -> None:
        cohorts = self.result["cohorts"]
        expected = {
            "direct_lexical": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            "semantic_paraphrase": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "relation_linked": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            "negative_control": (
                (1.0, 1.0, 1.0),
                (0.5, 0.6309297535714575, 1.0),
            ),
            "over_exclusion_control": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        }
        for cohort, (github_expected, ngr_expected) in expected.items():
            github = cohorts[cohort]["github_rag_mcp"]
            ngr = cohorts[cohort]["ngr"]
            self.assertEqual(
                (github["mrr"], github["ndcg_at_k"], github["recall_at_k"]),
                github_expected,
            )
            self.assertEqual(
                (ngr["mrr"], ngr["ndcg_at_k"], ngr["recall_at_k"]),
                ngr_expected,
            )

        failed = {gate["gate_id"] for gate in self.result["gates"] if not gate["passed"]}
        self.assertEqual(
            failed,
            {
                "negative-safety",
                "over-exclusion-safety",
                "cohort-non-regression",
                "expected-source-completeness",
            },
        )
        self.assertEqual(self.summary["hard_gates_passed"], 6)
        self.assertEqual(self.summary["hard_gates_total"], 10)
        self.assertFalse(self.summary["all_hard_gates_pass"])

    def test_only_fresh_temporary_database_resources_were_recorded(self) -> None:
        resources = self.result["resources"]
        self.assertFalse(resources["latency_hard_gate"])
        for replay in ("first_replay", "second_replay"):
            row = resources[replay]
            self.assertEqual(
                row["database_scope"], "fresh-temporary-deleted-after-replay"
            )
            self.assertFalse(row["feedback_connected"])
            self.assertEqual(row["database_bytes"], 1_531_904)

        holdout = (
            "holdout.preflight.json",
            "holdout.capture.json",
            "holdout.claim.json",
            "holdout.observed.json",
        )
        self.assertTrue(all(not (EVIDENCE / name).exists() for name in holdout))

    def test_observation_cli_is_import_safe(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "tools/verify_github_retrieval_parity_v5_observation.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["phase"], "development-closed")
        self.assertEqual(payload["registered_query_execution_count"], 5)
        self.assertFalse(payload["holdout_opened"])

    def test_registry_rejects_holdout_or_other_unregistered_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            for name in observation.EVIDENCE_SHA256:
                (evidence / name).touch()
            observation._assert_exact_registry(evidence)
            (evidence / "holdout.capture.json").touch()
            with self.assertRaisesRegex(ValueError, "extra=.*holdout.capture"):
                observation._assert_exact_registry(evidence)


def _json_fixture(name: str) -> dict[str, object]:
    return json.loads(
        (ROOT / "tests/fixtures" / name).read_text(
            encoding="utf-8", errors="strict"
        )
    )


if __name__ == "__main__":
    unittest.main()

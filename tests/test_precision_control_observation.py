from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import precision_control_evaluation as evaluation
from neuron_graph_rag import precision_control_observation as observation
from neuron_graph_rag.engine import EngineConfig
from neuron_graph_rag.precision_control import PrecisionControl


class PrecisionControlObservationMechanicsTests(unittest.TestCase):
    def _synthetic_protocol(self, root: Path) -> dict[str, object]:
        repository = "example/synthetic"
        documents = []
        for path, text in (("one.md", "alpha entry"), ("two.md", "linked target")):
            documents.append(
                {
                    "node_id": f"github:{repository}:doc:{path}",
                    "text": text,
                    "metadata": {
                        "repository": repository,
                        "commit": "0" * 40,
                        "path": path,
                        "content_sha256": observation.sha256_bytes(text.encode()),
                    },
                }
            )
        return {
            "root": root,
            "corpus": {
                "repository": repository,
                "relationships": [
                    {
                        "source_path": "one.md",
                        "target_path": "two.md",
                        "edge_type": "informs",
                    }
                ],
            },
            "queries": {
                "request": {"limit": 1, "now": 0.0},
                "stages": {
                    "synthetic": [
                        {
                            "case_id": "synthetic-001",
                            "cohort": "synthetic",
                            "query": "alpha",
                        }
                    ]
                },
            },
            "documents": documents,
        }

    def test_default_arm_captures_complete_prefilter_with_limit_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic_protocol(Path(directory))
            arm = observation._execute_arm(
                protocol,
                "synthetic",
                protocol["documents"],
                EngineConfig(),
            )
        case = arm["cases"][0]
        self.assertEqual(len(case["ranked_hits"]), 2)
        self.assertEqual(len(case["returned_source_paths"]), 1)
        self.assertEqual(arm["explanations"], [])
        self.assertEqual(arm["feedback_count_before"], 0)
        self.assertEqual(arm["feedback_count_after"], 0)
        self.assertEqual(arm["edge_sha256_before"], arm["edge_sha256_after"])

    def test_candidate_decisions_are_from_actual_search_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic_protocol(Path(directory))
            control = PrecisionControl(
                candidate_id="synthetic-floor", minimum_final_score=0.25
            )
            arm = observation._execute_arm(
                protocol,
                "synthetic",
                protocol["documents"],
                EngineConfig(precision_control=control),
            )
        decisions = arm["explanations"][0]["decisions"]
        hits = arm["cases"][0]["ranked_hits"]
        self.assertEqual(len(decisions), 2)
        self.assertEqual(
            [row["source_path"] for row in decisions],
            [row["source_path"] for row in hits],
        )
        self.assertEqual(
            decisions,
            [
                evaluation._decision_from_hit(control, hit, hits[0]["final_score"])
                for hit in hits
            ],
        )
        self.assertTrue(
            all(row["candidate_id"] == control.candidate_id for row in decisions)
        )
        self.assertEqual(
            arm["cases"][0]["returned_source_paths"],
            [row["source_path"] for row in decisions if row["accepted"]][:1],
        )

    def test_primary_and_replay_are_deterministic_and_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic_protocol(Path(directory))
            first = observation._execute_arm(
                protocol,
                "synthetic",
                protocol["documents"],
                EngineConfig(),
            )
            replay = observation._execute_arm(
                protocol,
                "synthetic",
                protocol["documents"],
                EngineConfig(),
            )
        state = observation._combine_state(first, replay)
        self.assertNotEqual(state["fresh_database_id"], state["replay_database_id"])
        self.assertEqual(state["ranking_sha256"], state["replay_ranking_sha256"])
        self.assertEqual(state["score_sha256"], state["replay_score_sha256"])
        self.assertEqual(state["activation_sha256"], state["replay_activation_sha256"])

    def test_failed_claim_is_preserved_and_retries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim_path = (
                root / "tests/runtime/github_precision_control_v1/dev.claim.json"
            )
            claim_path.parent.mkdir(parents=True)
            claim_raw = b'{"one_time_claim":true}\n'
            claim_path.write_bytes(claim_raw)
            observation._archive_error(
                root,
                "development",
                claim_path,
                claim_raw,
                RuntimeError("synthetic failure"),
            )
            failed_claim = observation._failed_claim_path(root, "development")
            error_path = observation._error_path(root, "development")
            self.assertEqual(failed_claim.read_bytes(), claim_raw)
            error = json.loads(error_path.read_text(encoding="utf-8"))
            self.assertEqual(set(error), set(observation.ERROR_FIELDS))
            with self.assertRaises(FileExistsError):
                observation._reject_failed_retry(root, "development")


class PrecisionControlObservationEvidenceTests(unittest.TestCase):
    def test_committed_development_evidence_is_exact_and_fully_verified(self) -> None:
        protocol = evaluation.load_protocol()
        self.assertEqual(
            evaluation.verify_phase_state(protocol),
            {"development": "unobserved", "holdout": "unobserved"},
        )
        self.assertEqual(
            observation.verify_observation_phase(protocol),
            {"development": "archived", "holdout": "unobserved"},
        )
        audit = observation.preflight()
        self.assertEqual(audit["registered_stage_execution_count"], 1)
        self.assertEqual(audit["post_observation_stage_reexecution_count"], 0)
        manifest = protocol["manifest"]
        development_paths = observation._final_archive_paths(
            evaluation.ROOT, "development"
        )
        development_claim = development_paths["claim"]
        development_result = development_paths["result"]
        lifecycle_transport = development_paths["lifecycle_transport"]
        archive_transport = development_paths["archive_transport"]
        expected_hashes = {
            development_claim: (
                "d3528a66849f8a25fcd4e7030bf199e0b3aa74f3a5d72737340f102ce39006af"
            ),
            development_result: (
                "d553830cbc6006170d6b78b5d864a10495b956954e5658075135b0bf93a0e844"
            ),
            lifecycle_transport: (
                "dfeb0ca23164361d7a4d9560141e8b0a1190576097b1878acb2097c22905c5f2"
            ),
            archive_transport: (
                "a591e2fd57d8856a1eaf94f517869ed1bfccd07ef21f0453255eba6ec27acac7"
            ),
        }
        for path, expected in expected_hashes.items():
            self.assertEqual(evaluation.sha256_bytes(path.read_bytes()), expected)

        result = json.loads(development_result.read_text(encoding="utf-8"))
        evaluation.verify_result_payload(
            protocol, "development", result, development_claim.read_bytes()
        )
        self.assertEqual(len(result["gates"]), len(evaluation.GATE_IDS))
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertFalse(result["all_hard_gates_pass"])
        self.assertTrue(
            all(
                not candidate["all_hard_gates_pass"]
                for candidate in result["candidates"]
            )
        )
        transport = json.loads(lifecycle_transport.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["sha256"] for row in transport["files"]],
            [
                expected_hashes[development_claim],
                expected_hashes[development_result],
            ],
        )
        archival = json.loads(archive_transport.read_text(encoding="utf-8"))
        self.assertTrue(archival["runtime_verified"])
        self.assertTrue(all(row["byte_identity"] is True for row in archival["files"]))
        self.assertEqual(
            [row["sha256"] for row in archival["files"]],
            [
                expected_hashes[development_claim],
                expected_hashes[development_result],
                expected_hashes[lifecycle_transport],
            ],
        )

        for relative in manifest["outputs"]["holdout"].values():
            self.assertFalse((evaluation.ROOT / relative).exists())
        self.assertTrue(
            all(
                not path.exists()
                for path in observation._final_archive_paths(
                    evaluation.ROOT, "holdout"
                ).values()
            )
        )


if __name__ == "__main__":
    unittest.main()

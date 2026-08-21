from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.corpus_integrity import verify_manifest_source_hashes
from neuron_graph_rag.outcome_feedback_deactivation_evaluation import (
    ARMS,
    MANIFEST_PATH,
    PROTOCOL_ID,
    STEM,
    acquire_transactional_snapshot,
    assert_public_payload,
    prove_writer_verifier_round_trip,
    read_json,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return read_json(FIXTURES / f"{STEM}.{name}.json")


class OutcomeFeedbackDeactivationFreezeTest(unittest.TestCase):
    def test_result_free_artifacts_are_canonical_private_free_and_complete(self) -> None:
        for name in (
            "fixture",
            "schedule",
            "gate",
            "result-schema",
            "result-free-audit",
        ):
            path = FIXTURES / f"{STEM}.{name}.json"
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            self.assertEqual(payload["protocol_id"], PROTOCOL_ID)
            assert_public_payload(payload)

        schedule = _fixture("schedule")
        self.assertEqual([arm["arm_id"] for arm in schedule["arms"]], list(ARMS))
        gate = _fixture("gate")
        self.assertEqual(len(gate["gates"]), 8)
        self.assertNotEqual(
            [item["gate_id"] for item in gate["gates"]],
            sorted(item["gate_id"] for item in gate["gates"]),
        )
        audit = _fixture("result-free-audit")
        self.assertFalse(audit["development_executed_in_freeze_issue"])
        self.assertFalse(audit["holdout_executed_in_freeze_issue"])

    def test_case_identity_roles_and_paths_are_frozen(self) -> None:
        fixture = _fixture("fixture")
        seen: set[str] = set()
        stage_nodes: dict[str, set[str]] = {}
        for stage in ("development", "holdout"):
            cases = fixture["stages"][stage]
            self.assertEqual(
                {case["case_role"] for case in cases},
                {"corrected", "rolled_back", "superseded", "unattributed"},
            )
            ids = {case["case_id"] for case in cases}
            self.assertTrue(ids.isdisjoint(seen))
            seen.update(ids)
            stage_nodes[stage] = {case["used_node_id"] for case in cases}
            for case in cases:
                if case["search_surface"] == "relation":
                    self.assertEqual(
                        case["credited_edge"]["target_id"], case["used_node_id"]
                    )
                    self.assertEqual(case["registered_initial_state"]["weight"], 0.5)
        self.assertTrue(stage_nodes["development"].isdisjoint(stage_nodes["holdout"]))

    def test_manifest_uses_registration_bytes_and_outputs_were_absent(self) -> None:
        manifest = read_json(MANIFEST_PATH)
        registered = verify_manifest_source_hashes(
            ROOT, MANIFEST_PATH, manifest["artifact_sha256"]
        )
        self.assertEqual(registered.artifact_sha256, manifest["artifact_sha256"])
        for relative in manifest["outputs"].values():
            completed = subprocess.run(
                ["git", "cat-file", "-e", f"{registered.source_commit}:{relative}"],
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_snapshot_acquisition_is_logically_stable_private_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite"
            snapshot = root / "snapshot.sqlite"
            connection = sqlite3.connect(source)
            try:
                connection.executescript(
                    """
                    CREATE TABLE nodes(node_id TEXT PRIMARY KEY, text TEXT);
                    CREATE TABLE edges(source_id TEXT, target_id TEXT, edge_type TEXT);
                    CREATE TABLE retrievals(trace_id TEXT);
                    CREATE TABLE success_feedback(feedback_id TEXT);
                    CREATE TABLE delayed_outcomes(outcome_id TEXT);
                    INSERT INTO nodes VALUES ('public-id', 'private body');
                    """
                )
                connection.commit()
            finally:
                connection.close()
            provenance = acquire_transactional_snapshot(source, snapshot)
            self.assertEqual(
                provenance["source_logical_sha256_before"],
                provenance["source_logical_sha256_after"],
            )
            self.assertEqual(
                provenance["snapshot_sha256"],
                hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            )
            self.assertNotIn("private body", json.dumps(provenance))
            assert_public_payload(provenance)
            with self.assertRaises(FileExistsError):
                acquire_transactional_snapshot(source, snapshot)

    def test_writer_round_trip_privacy_and_exclusive_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prove_writer_verifier_round_trip(root / "placeholder.json")
            self.assertFalse((root / "placeholder.json").exists())
            target = root / "exclusive.json"
            write_json_exclusive(target, {"safe": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"safe": False})
        for payload in (
            {"node_text": "private body"},
            {"value": "C:\\private\\snapshot.sqlite"},
            {"value": "/private/snapshot.sqlite"},
            {"value": "password=hunter2"},
        ):
            with self.assertRaises(ValueError):
                assert_public_payload(payload)


if __name__ == "__main__":
    unittest.main()

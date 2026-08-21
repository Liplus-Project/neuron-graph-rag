from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.corpus_integrity import verify_manifest_source_hashes
from neuron_graph_rag.soft_start_snapshot_evaluation import (
    MANIFEST_PATH,
    POLICIES,
    STEM,
    acquire_transactional_snapshot,
    assert_public_payload,
    prove_writer_verifier_round_trip,
    read_json,
    verify_registered_result,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return read_json(FIXTURES / f"{STEM}.{name}.json")


class SoftStartSnapshotFreezeTest(unittest.TestCase):
    def test_result_free_artifacts_are_canonical_private_free_and_complete(self) -> None:
        names = (
            "fixture",
            "schedule",
            "gate",
            "result-schema",
            "result-free-audit",
        )
        for name in names:
            path = FIXTURES / f"{STEM}.{name}.json"
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            self.assertEqual(payload["protocol_id"], "soft-start-snapshot-v1")
            assert_public_payload(payload)

        schedule = _fixture("schedule")
        self.assertEqual(
            [arm["arm_id"] for arm in schedule["arms"]], list(POLICIES)
        )
        self.assertEqual(schedule["iterations_per_case"], 3)
        self.assertEqual(
            schedule["event_order"],
            ["used_1", "outcome_1", "used_2", "outcome_2", "used_3", "outcome_3"],
        )
        self.assertEqual(
            schedule["arms"][3]["engine_config"],
            {
                "soft_start_feedback_reinforcement": True,
                "soft_start_feedback_ratio": 0.25,
                "confirmation_decay_ratio": 0.5,
            },
        )
        fixture = _fixture("fixture")
        for stage in ("development", "holdout"):
            self.assertEqual(
                [case["case_role"] for case in fixture["stages"][stage]],
                ["confirmed", "corrected", "lexical", "zero_hop"],
            )

        audit = _fixture("result-free-audit")
        self.assertTrue(audit["result_free"])
        self.assertTrue(audit["registered_outputs_absent"])
        self.assertEqual(audit["source_database_write_count"], 0)
        self.assertTrue(audit["placeholder_round_trip_passed"])

    def test_manifest_reads_registered_commit_bytes_and_registered_outputs_were_absent(self) -> None:
        manifest = read_json(MANIFEST_PATH)
        assert_public_payload(manifest)
        registered = verify_manifest_source_hashes(
            ROOT, MANIFEST_PATH, manifest["artifact_sha256"]
        )
        self.assertEqual(registered.artifact_sha256, manifest["artifact_sha256"])
        for relative in manifest["outputs"].values():
            completed = subprocess.run(
                [
                    "git",
                    "cat-file",
                    "-e",
                    f"{registered.source_commit}:{relative}",
                ],
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_snapshot_acquisition_is_transactional_read_only_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite"
            snapshot = root / "snapshot.sqlite"
            connection = sqlite3.connect(source)
            try:
                connection.execute("CREATE TABLE nodes(node_id TEXT PRIMARY KEY, text TEXT)")
                connection.execute(
                    "CREATE TABLE edges(source_id TEXT, target_id TEXT, edge_type TEXT)"
                )
                connection.execute("CREATE TABLE retrievals(trace_id TEXT)")
                connection.execute("CREATE TABLE success_feedback(feedback_id TEXT)")
                connection.execute("CREATE TABLE delayed_outcomes(outcome_id TEXT)")
                connection.execute("INSERT INTO nodes VALUES ('public-id', 'private body')")
                connection.commit()
            finally:
                connection.close()
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            provenance = acquire_transactional_snapshot(source, snapshot)
            after = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(
                provenance["source_container_sha256_before"],
                provenance["source_container_sha256_after"],
            )
            self.assertEqual(provenance["row_counts"]["nodes"], 1)
            self.assertNotIn("private body", json.dumps(provenance))
            assert_public_payload(provenance)
            with self.assertRaises(FileExistsError):
                acquire_transactional_snapshot(source, snapshot)

    def test_writer_round_trip_is_exclusive_and_privacy_scan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prove_writer_verifier_round_trip(root / "probe.json")
            self.assertFalse((root / "probe.json").exists())
            target = root / "exclusive.json"
            write_json_exclusive(target, {"safe": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"safe": False})
        with self.assertRaisesRegex(ValueError, "private field"):
            assert_public_payload({"node_text": "secret"})
        with self.assertRaisesRegex(ValueError, "absolute private path"):
            assert_public_payload({"value": "C:\\private\\snapshot.sqlite"})
        with self.assertRaisesRegex(ValueError, "absolute private path"):
            assert_public_payload({"value": "/private/snapshot.sqlite"})
        with self.assertRaisesRegex(ValueError, "credential-shaped"):
            assert_public_payload({"value": "password=hunter2"})

    def test_registered_observations_verify_when_present(self) -> None:
        manifest = read_json(MANIFEST_PATH)
        for stage, relative in manifest["outputs"].items():
            if (ROOT / relative).exists():
                verify_registered_result(stage)


if __name__ == "__main__":
    unittest.main()

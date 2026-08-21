from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag.baseline_aware_soft_start_snapshot_evaluation import (
    MANIFEST_PATH,
    POLICIES,
    STEM,
    acquire_transactional_snapshot,
    assert_public_payload,
    derive_q3_first_mutation_event,
    prove_writer_verifier_round_trip,
    read_json,
    run_registered_stage,
    verify_registered_result,
    write_json_exclusive,
)
from neuron_graph_rag.corpus_integrity import verify_manifest_source_hashes


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return read_json(FIXTURES / f"{STEM}.{name}.json")


class BaselineAwareSoftStartSnapshotFreezeTest(unittest.TestCase):
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
            self.assertEqual(
                payload["protocol_id"], "baseline-aware-soft-start-snapshot-v2"
            )
            assert_public_payload(payload)

        schedule = _fixture("schedule")
        self.assertEqual(
            [arm["arm_id"] for arm in schedule["arms"]], list(POLICIES)
        )
        self.assertEqual(schedule["event_budget"], 4)
        self.assertEqual(
            schedule["event_order"],
            [
                "used_1",
                "outcome_1",
                "used_2",
                "outcome_2",
                "used_3",
                "outcome_3",
                "used_4",
                "outcome_4",
            ],
        )
        gate = _fixture("gate")
        self.assertEqual(len(gate["gates"]), 8)
        self.assertEqual(gate["gates"][1]["gate_id"], "baseline-aware-q3-boundary")

    def test_initial_state_derivation_capacity_and_v1_isolation_are_frozen(self) -> None:
        fixture = _fixture("fixture")
        manifest = read_json(MANIFEST_PATH)
        excluded = set(manifest["excluded_v1_development_credited_edges"])
        development_edges: set[str] = set()
        all_case_ids: set[str] = set()
        for stage in ("development", "holdout"):
            for case in fixture["stages"][stage]:
                self.assertNotIn(case["case_id"], all_case_ids)
                all_case_ids.add(case["case_id"])
                if case["case_role"] not in {"confirmed", "corrected"}:
                    continue
                initial = case["registered_initial_state"]
                expected = derive_q3_first_mutation_event(
                    3, initial["evidence_count"]
                )
                self.assertEqual(case["expected_q3_first_mutation_event"], expected)
                self.assertLess(expected, 4)
                key = "|".join(
                    case["credited_edge"][field]
                    for field in ("source_id", "target_id", "edge_type")
                )
                if stage == "development":
                    development_edges.add(key)
        self.assertTrue(development_edges.isdisjoint(excluded))
        audit = _fixture("result-free-audit")
        self.assertFalse(audit["v1_private_snapshot_used"])
        self.assertFalse(audit["v1_observed_result_used"])
        self.assertFalse(audit["v1_development_credited_edge_reused"])

    def test_manifest_reads_registration_bytes_and_outputs_were_absent(self) -> None:
        manifest = read_json(MANIFEST_PATH)
        registered = verify_manifest_source_hashes(
            ROOT, MANIFEST_PATH, manifest["artifact_sha256"]
        )
        self.assertEqual(registered.artifact_sha256, manifest["artifact_sha256"])
        self.assertFalse(
            any("soft_start_snapshot_v1" in path for path in manifest["artifact_sha256"])
        )
        for relative in manifest["outputs"].values():
            completed = subprocess.run(
                ["git", "cat-file", "-e", f"{registered.source_commit}:{relative}"],
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_preflight_failure_creates_no_registered_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.sqlite"
            snapshot.write_bytes(b"not-opened")
            output = root / "development.json"
            manifest = {
                "outputs": {
                    "development": str(output),
                    "holdout": str(root / "holdout.json"),
                }
            }
            report = {
                "checks": {"registered_initial_state": False},
                "passed": False,
                "snapshot_sha256_before": None,
                "snapshot_sha256_after": None,
                "snapshot_unchanged": False,
            }
            with patch(
                "neuron_graph_rag.baseline_aware_soft_start_snapshot_evaluation._load_protocol",
                return_value=(manifest, {}),
            ), patch(
                "neuron_graph_rag.baseline_aware_soft_start_snapshot_evaluation._preflight",
                return_value=report,
            ):
                with self.assertRaisesRegex(RuntimeError, "protocol preflight failed"):
                    run_registered_stage("development", snapshot)
            self.assertFalse(output.exists())

    def test_snapshot_acquisition_is_read_only_private_and_exclusive(self) -> None:
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
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotIn("private body", json.dumps(provenance))
            assert_public_payload(provenance)
            with self.assertRaises(FileExistsError):
                acquire_transactional_snapshot(source, snapshot)

    def test_writer_round_trip_privacy_and_registered_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prove_writer_verifier_round_trip(root / "probe.json")
            self.assertFalse((root / "probe.json").exists())
            target = root / "exclusive.json"
            write_json_exclusive(target, {"safe": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"safe": False})
        for payload in (
            {"node_text": "secret"},
            {"value": "C:\\private\\snapshot.sqlite"},
            {"value": "/private/snapshot.sqlite"},
            {"value": "password=hunter2"},
        ):
            with self.assertRaises(ValueError):
                assert_public_payload(payload)
        if MANIFEST_PATH.exists():
            manifest = read_json(MANIFEST_PATH)
            for stage, relative in manifest["outputs"].items():
                if (ROOT / relative).exists():
                    verify_registered_result(stage)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag import github_retrieval_parity as parity
from neuron_graph_rag.github_retrieval_parity import (
    AUDIT_PATH,
    CAPTURE_SCHEMA_PATH,
    COHORTS,
    CURRENT_CORPUS_PATH,
    GATE_IDS,
    GATE_PATH,
    GOLD_PATH,
    MANIFEST_PATH,
    PREVIOUS_CORPUS_PATH,
    QUERIES_PATH,
    RESULT_SCHEMA_PATH,
    STAGES,
    load_protocol,
    prove_writer_verifier_round_trip,
    validate_protocol,
    verify_frozen_artifacts,
    write_json_exclusive,
)
from neuron_graph_rag.github_source import GitHubSnapshot, index_github_snapshot

ROOT = Path(__file__).resolve().parents[1]


class GitHubRetrievalParityFreezeTest(unittest.TestCase):
    def test_protocol_is_complete_identity_disjoint_and_result_free(self) -> None:
        protocol = load_protocol()
        manifest = protocol["manifest"]
        current = protocol["current"]
        previous = protocol["previous"]
        self.assertEqual(len(current.documents), 12)
        self.assertEqual(len(previous.documents), 12)
        self.assertEqual(
            [item["gate_id"] for item in protocol["gate"]["gates"]],
            list(GATE_IDS),
        )
        identities: dict[str, set[str]] = {}
        for stage in STAGES:
            query_rows = protocol["queries"]["stages"][stage]
            gold_rows = protocol["gold"]["stages"][stage]
            self.assertEqual([item["cohort"] for item in query_rows], list(COHORTS))
            values: set[str] = set()
            for row in gold_rows:
                values.add(row["expected_source_id"])
                values.update(row["forbidden_source_ids"])
                if "relation_seed_source_id" in row:
                    values.add(row["relation_seed_source_id"])
            identities[stage] = values
            for kind in ("capture", "claim", "result"):
                self.assertFalse((ROOT / manifest["outputs"][stage][kind]).exists())
        self.assertTrue(identities["development"].isdisjoint(identities["holdout"]))

    def test_public_snapshots_verify_content_hashes_and_provenance(self) -> None:
        for path in (CURRENT_CORPUS_PATH, PREVIOUS_CORPUS_PATH):
            raw = json.loads(path.read_text(encoding="utf-8", errors="strict"))
            snapshot = GitHubSnapshot.from_mapping(raw)
            for source, document in zip(
                raw["documents"], snapshot.documents, strict=True
            ):
                self.assertEqual(
                    source["content_sha256"],
                    hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
                )
                self.assertIn(snapshot.commit, document.source_url)
        with NeuronGraphRAG() as engine:
            snapshot = GitHubSnapshot.read(CURRENT_CORPUS_PATH)
            index_github_snapshot(engine, snapshot)
            node = engine.store.get_node(snapshot.document_id(snapshot.documents[0]))
        self.assertEqual(
            node.metadata["content_sha256"], snapshot.documents[0].content_sha256
        )

    def test_all_freeze_json_is_canonical_utf8_and_observation_free(self) -> None:
        paths = (
            CURRENT_CORPUS_PATH,
            PREVIOUS_CORPUS_PATH,
            QUERIES_PATH,
            GOLD_PATH,
            GATE_PATH,
            CAPTURE_SCHEMA_PATH,
            RESULT_SCHEMA_PATH,
            AUDIT_PATH,
            MANIFEST_PATH,
        )
        for path in paths:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertFalse(audit["live_github_rag_mcp_search_called"])
        self.assertFalse(audit["registered_query_executed_against_ngr"])

    def test_freeze_audit_and_placeholder_round_trip_do_not_search(self) -> None:
        protocol = load_protocol()
        with patch.object(
            NeuronGraphRAG,
            "search",
            side_effect=AssertionError("registered search forbidden"),
        ):
            verify_frozen_artifacts(protocol)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "placeholder.json"
                prove_writer_verifier_round_trip(target)
                self.assertFalse(target.exists())

    def test_tampered_hash_and_split_identity_fail_closed(self) -> None:
        raw = json.loads(CURRENT_CORPUS_PATH.read_text(encoding="utf-8"))
        raw["documents"][0]["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            GitHubSnapshot.from_mapping(raw)

        protocol = load_protocol()
        tampered = copy.deepcopy(protocol)
        tampered["gold"]["stages"]["holdout"][0]["expected_source_id"] = protocol[
            "gold"
        ]["stages"]["development"][0]["expected_source_id"]
        with self.assertRaisesRegex(ValueError, "identities must be disjoint"):
            validate_protocol(tampered)

    def test_duplicate_observation_and_development_rerun_fail_closed(self) -> None:
        protocol = load_protocol()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(parity, "ROOT", Path(directory)),
        ):
            capture = (
                Path(directory)
                / protocol["manifest"]["outputs"]["development"]["capture"]
            )
            write_json_exclusive(capture, {"first": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(capture, {"second": True})

            claim = (
                Path(directory)
                / protocol["manifest"]["outputs"]["development"]["claim"]
            )
            write_json_exclusive(claim, {"claimed": True})
            with self.assertRaisesRegex(FileExistsError, "development claim"):
                parity._assert_stage_can_start("development", protocol)

    def test_failed_development_keeps_holdout_closed(self) -> None:
        protocol = load_protocol()
        gates = [
            {"gate_id": gate_id, "hard": True, "passed": False, "details": {}}
            for gate_id in GATE_IDS
        ]
        failed = {
            "protocol_id": parity.PROTOCOL_ID,
            "protocol_commit": "0" * 40,
            "stage": "development",
            "status": "failed",
            "failure_code": "hard-gate-failed",
            "capture_sha256": "1" * 64,
            "protocol_hashes": {},
            "source": {},
            "cases": [],
            "cohorts": {},
            "update_following": {"passed": False},
            "deterministic_replay": {"passed": False},
            "resources": {"latency_hard_gate": False},
            "gates": gates,
            "all_hard_gates_pass": False,
            "raw_github_rag_mcp_capture": {},
            "interpretation_ja": "development hard gate failure",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(parity, "ROOT", Path(directory)),
        ):
            development = (
                Path(directory)
                / protocol["manifest"]["outputs"]["development"]["result"]
            )
            write_json_exclusive(development, failed)
            with self.assertRaisesRegex(
                RuntimeError, "development hard gates did not pass"
            ):
                parity._assert_stage_can_start("holdout", protocol)


if __name__ == "__main__":
    unittest.main()

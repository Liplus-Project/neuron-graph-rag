from __future__ import annotations

import copy
import hashlib
import json
import subprocess
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
    def _valid_result(
        self, protocol: dict[str, object], capture: dict[str, object]
    ) -> dict[str, object]:
        gates = [
            {"gate_id": gate_id, "hard": True, "passed": True, "details": {}}
            for gate_id in GATE_IDS
        ]
        return {
            "protocol_id": parity.PROTOCOL_ID,
            "protocol_commit": "0" * 40,
            "stage": "development",
            "status": "passed",
            "failure_code": None,
            "capture_sha256": "1" * 64,
            "protocol_hashes": dict(protocol["manifest"]["artifact_sha256"]),
            "source": {},
            "cases": [],
            "cohorts": {},
            "update_following": {"passed": True},
            "deterministic_replay": {"passed": True},
            "resources": {"latency_hard_gate": False},
            "gates": gates,
            "all_hard_gates_pass": True,
            "raw_github_rag_mcp_capture": capture,
            "interpretation_ja": "test result",
        }

    def _valid_capture(
        self, protocol: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        case = protocol["queries"]["stages"]["development"][0]
        current = protocol["current"]
        document = current.documents[0]
        indexed = parity._indexed_content(document.path, document.content)
        search_item = {
            "repo": current.repository,
            "type": "doc",
            "doc_path": document.path,
            "vector_id": "vector-1",
        }
        stored_item = {
            **search_item,
            "content": indexed,
            "content_chars": parity._js_length(indexed),
            "content_truncated": (
                parity._js_length(document.path + "\n\n" + document.content) >= 8000
            ),
        }
        capture = {
            "protocol_id": parity.PROTOCOL_ID,
            "protocol_commit": "0" * 40,
            "stage": "development",
            "captured_at": "2026-08-25T00:00:00Z",
            "service": "github-rag-mcp",
            "tool": "search",
            "cases": [
                {
                    "case_id": case["case_id"],
                    "request": {
                        "query": case["query"],
                        **protocol["queries"]["request_defaults"],
                    },
                    "raw_search": {
                        "mode": "search",
                        "filters_unmatched": [],
                        "results": [search_item],
                        "graph_results": [],
                    },
                    "raw_stored_content": {
                        "mode": "fetch",
                        "content_source": "index",
                        "content_max_chars": 8000,
                        "not_found": [],
                        "results": [stored_item],
                    },
                }
            ],
        }
        return capture, case

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

    def test_only_manifest_introduction_commit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return completed.stdout.strip()

            git("init")
            git("config", "core.autocrlf", "false")
            git("config", "user.email", "parity@example.invalid")
            git("config", "user.name", "Parity Test")
            (root / "parent.txt").write_text("parent\n", encoding="utf-8")
            git("add", "parent.txt")
            git("commit", "-m", "parent")

            artifact = root / "frozen.txt"
            artifact.write_text("frozen\n", encoding="utf-8")
            manifest_path = root / MANIFEST_PATH.relative_to(ROOT)
            manifest_path.parent.mkdir(parents=True)
            manifest = {
                "artifact_sha256": {
                    "frozen.txt": hashlib.sha256(artifact.read_bytes()).hexdigest()
                },
                "outputs": {
                    stage: {
                        kind: f"outputs/{stage}.{kind}.json"
                        for kind in ("capture", "claim", "result")
                    }
                    for stage in STAGES
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            git("add", "frozen.txt", MANIFEST_PATH.relative_to(ROOT).as_posix())
            git("commit", "-m", "freeze")
            freeze_commit = git("rev-parse", "HEAD")

            (root / "later.txt").write_text("later\n", encoding="utf-8")
            git("add", "later.txt")
            git("commit", "-m", "later")
            later_commit = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", later_commit)

            protocol = {"root": root, "manifest": manifest}
            parity.verify_protocol_commit(freeze_commit, protocol)
            with self.assertRaisesRegex(ValueError, "first introduces the manifest"):
                parity.verify_protocol_commit(later_commit, protocol)

            git("rm", MANIFEST_PATH.relative_to(ROOT).as_posix())
            git("commit", "-m", "remove manifest")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            git("add", MANIFEST_PATH.relative_to(ROOT).as_posix())
            git("commit", "-m", "reintroduce manifest")
            reintroduced_commit = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", reintroduced_commit)
            with self.assertRaisesRegex(ValueError, "unique first-parent introduction"):
                parity.verify_protocol_commit(reintroduced_commit, protocol)

    def test_result_gate_shape_and_status_tampering_fail_closed(self) -> None:
        protocol = load_protocol()
        payload = self._valid_result(protocol, {})
        schema = protocol["result_schema"]
        tampered: list[tuple[str, dict[str, object], str]] = []

        non_mapping = copy.deepcopy(payload)
        non_mapping["gates"][0] = "not-an-object"
        tampered.append(("non-mapping", non_mapping, "gate shape"))

        short = copy.deepcopy(payload)
        short["gates"].pop()
        tampered.append(("missing-gate", short, "gate order"))

        soft = copy.deepcopy(payload)
        soft["gates"][0]["hard"] = False
        tampered.append(("soft-gate", soft, "must be hard"))

        non_boolean = copy.deepcopy(payload)
        non_boolean["gates"][0]["passed"] = 1
        tampered.append(("non-boolean-passed", non_boolean, "must be boolean"))

        pending = copy.deepcopy(payload)
        pending["status"] = "pending"
        tampered.append(("unknown-status", pending, "passed or failed"))

        for name, candidate, message in tampered:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                parity.verify_result_payload(candidate, schema)

    def test_registered_result_integrity_tampering_fails_closed(self) -> None:
        protocol = load_protocol()
        capture = {"raw": "registered"}
        result = self._valid_result(protocol, capture)
        mutations = []

        def add_claim_field(claim: dict[str, object], _: dict[str, object]) -> None:
            claim["extra"] = True

        mutations.append(("claim-fields", add_claim_field, "claim field order"))

        def change_claim_identity(
            claim: dict[str, object], _: dict[str, object]
        ) -> None:
            claim["protocol_id"] = "other"

        mutations.append(
            ("claim-identity", change_claim_identity, "claim protocol identity")
        )

        def change_claim_stage(claim: dict[str, object], _: dict[str, object]) -> None:
            claim["stage"] = "holdout"

        mutations.append(("claim-stage", change_claim_stage, "claim protocol identity"))

        def disable_one_time(claim: dict[str, object], _: dict[str, object]) -> None:
            claim["one_time_claim"] = False

        mutations.append(("claim-once", disable_one_time, "one_time_claim"))

        def change_result_commit(
            _: dict[str, object], candidate: dict[str, object]
        ) -> None:
            candidate["protocol_commit"] = "2" * 40

        mutations.append(
            ("result-commit", change_result_commit, "protocol_commit does not match")
        )

        def change_protocol_hashes(
            _: dict[str, object], candidate: dict[str, object]
        ) -> None:
            candidate["protocol_hashes"] = {}

        mutations.append(("protocol-hashes", change_protocol_hashes, "protocol hashes"))

        def change_raw_capture(
            _: dict[str, object], candidate: dict[str, object]
        ) -> None:
            candidate["raw_github_rag_mcp_capture"] = {"raw": "tampered"}

        mutations.append(("raw-capture", change_raw_capture, "raw capture"))

        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with (
                    patch.object(parity, "ROOT", root),
                    patch.object(parity, "verify_protocol_commit"),
                ):
                    manifest = protocol["manifest"]
                    capture_path = root / manifest["outputs"]["development"]["capture"]
                    write_json_exclusive(capture_path, capture)
                    capture_sha256 = hashlib.sha256(
                        capture_path.read_bytes()
                    ).hexdigest()
                    claim = {
                        "protocol_id": parity.PROTOCOL_ID,
                        "protocol_commit": "0" * 40,
                        "stage": "development",
                        "capture_sha256": capture_sha256,
                        "one_time_claim": True,
                    }
                    candidate = copy.deepcopy(result)
                    candidate["capture_sha256"] = capture_sha256
                    mutate(claim, candidate)
                    write_json_exclusive(
                        root / manifest["outputs"]["development"]["claim"], claim
                    )
                    write_json_exclusive(
                        root / manifest["outputs"]["development"]["result"],
                        candidate,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        parity._verify_registered_result("development", protocol)

    def test_capture_surface_and_vector_identity_tampering_fail_closed(self) -> None:
        protocol = load_protocol()
        capture, case = self._valid_capture(protocol)

        duplicate = copy.deepcopy(capture)
        duplicate["cases"][0]["raw_search"]["results"].append(
            copy.deepcopy(duplicate["cases"][0]["raw_search"]["results"][0])
        )
        with self.assertRaisesRegex(ValueError, "vector_id must be unique"):
            parity._validate_capture(
                duplicate, "development", "0" * 40, protocol, [case]
            )

        for field, value in (
            ("repo", "outside/repository"),
            ("type", "issue"),
            ("doc_path", "outside.md"),
        ):
            tampered = copy.deepcopy(capture)
            graph_item = copy.deepcopy(tampered["cases"][0]["raw_search"]["results"][0])
            graph_item[field] = value
            tampered["cases"][0]["raw_search"]["graph_results"] = [graph_item]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "graph result left"),
            ):
                parity._validate_capture(
                    tampered, "development", "0" * 40, protocol, [case]
                )

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
            "protocol_hashes": dict(protocol["manifest"]["artifact_sha256"]),
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
            patch.object(parity, "verify_protocol_commit"),
        ):
            root = Path(directory)
            manifest = protocol["manifest"]
            capture = {"raw": "registered"}
            capture_path = root / manifest["outputs"]["development"]["capture"]
            write_json_exclusive(capture_path, capture)
            capture_sha256 = hashlib.sha256(capture_path.read_bytes()).hexdigest()
            failed["capture_sha256"] = capture_sha256
            failed["raw_github_rag_mcp_capture"] = capture
            write_json_exclusive(
                root / manifest["outputs"]["development"]["claim"],
                {
                    "protocol_id": parity.PROTOCOL_ID,
                    "protocol_commit": "0" * 40,
                    "stage": "development",
                    "capture_sha256": capture_sha256,
                    "one_time_claim": True,
                },
            )
            development = root / manifest["outputs"]["development"]["result"]
            write_json_exclusive(development, failed)
            with self.assertRaisesRegex(
                RuntimeError, "development hard gates did not pass"
            ):
                parity._assert_stage_can_start("holdout", protocol)


if __name__ == "__main__":
    unittest.main()

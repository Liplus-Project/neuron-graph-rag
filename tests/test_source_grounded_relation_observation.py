from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import source_grounded_relation_observation as observation

ROOT = Path(__file__).resolve().parents[1]
TEST_COMMIT = "a" * 40


def _raw_packet_registry() -> dict[str, dict[str, dict[str, str]]]:
    return {
        stage: {
            arm: {
                run: f"evidence/raw/{stage}/{arm}.{run}.json"
                for run in observation.RUNS
            }
            for arm in observation.ARMS
        }
        for stage in observation.STAGES
    }


def _packet(stage: str, arm: str, run: str) -> dict[str, object]:
    return {
        "protocol_id": observation.PROTOCOL_ID,
        "protocol_commit": TEST_COMMIT,
        "stage": stage,
        "arm": arm,
        "run": run,
        "attempt": 1,
        "retry_count": 0,
        "cases": [],
    }


class SourceGroundedRelationObservationTests(unittest.TestCase):
    def test_protocol_is_complete_fresh_and_result_free(self) -> None:
        result = observation.audit_result_free()
        self.assertEqual(result["status"], "result-free-protocol-valid")
        self.assertEqual(result["source_document_count"], 20)
        self.assertEqual(result["development_case_count"], 8)
        self.assertEqual(result["holdout_case_count"], 8)
        self.assertEqual(result["observed_result_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_all_registered_relations_are_rederived_from_source(self) -> None:
        protocol = observation.load_protocol()
        extracted = observation.extract_source_grounded_relations(protocol["corpus"])
        self.assertEqual(list(extracted), protocol["manifest"]["relationships"])
        self.assertGreaterEqual(len(extracted), 4)
        for relation in extracted:
            self.assertEqual(relation["edge_type"], "markdown_link")
            self.assertEqual(
                relation["acquisition_method"], "markdown-relative-link-regex-v1"
            )

    def test_corpus_bytes_and_blob_ids_match_fixed_git_objects(self) -> None:
        protocol = observation.load_protocol()
        commit = protocol["manifest"]["source"]["commit"]
        for document in protocol["corpus"].documents:
            content = subprocess.run(
                ["git", "show", f"{commit}:{document.path}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            blob = subprocess.run(
                ["git", "rev-parse", f"{commit}:{document.path}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            self.assertEqual(content.decode("utf-8", errors="strict"), document.content)
            self.assertEqual(blob, document.blob_sha)
            self.assertEqual(
                hashlib.sha256(content).hexdigest(), document.content_sha256
            )

    def test_worker_is_gold_blind(self) -> None:
        protocol = observation.load_worker_protocol()
        self.assertNotIn("gold", protocol)
        self.assertNotIn("gate", protocol)
        self.assertNotIn("audit", protocol)
        signature = inspect.signature(observation.run_worker)
        self.assertNotIn("gold", signature.parameters)
        source = inspect.getsource(observation.run_worker)
        self.assertNotIn("finalize_stage", source)
        with self.assertRaisesRegex(ValueError, "must not expose gold"):
            observation.run_worker(
                observation.load_protocol(),
                "development",
                "original-full-query-ngr-default",
                Path("unreachable.sqlite"),
                protocol_commit=TEST_COMMIT,
                run="primary",
            )

    def test_predecessor_stage_formats_are_both_read(self) -> None:
        v19 = observation._read_json(
            ROOT / "tests/fixtures/github_cross_encoder_precision_v8.queries.json",
            require_canonical=False,
        )
        v23 = observation._read_json(
            ROOT
            / "tests/fixtures/github_cross_encoder_precision_v23_real.queries.json",
            require_canonical=False,
        )
        self.assertEqual(len(observation._stage_case_rows(v19)), 16)
        self.assertEqual(len(observation._stage_case_rows(v23)), 16)

    def test_candidate_uses_query_source_reference_and_records_path(self) -> None:
        protocol = observation.load_worker_protocol()
        synthetic = copy.deepcopy(protocol)
        synthetic["queries"] = {
            "stages": {
                "development": [
                    {
                        "case_id": "synthetic-source-reference",
                        "query": "Which related design is linked from Evidence-gated local feedback reinforcement?",
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            result = observation.run_worker(
                synthetic,
                "development",
                "source-grounded-relation-seed",
                Path(directory) / "fresh.sqlite",
                protocol_commit=TEST_COMMIT,
                run="primary",
            )
        case = result["cases"][0]
        self.assertEqual(
            case["referenced_seed_paths"],
            ["docs/evidence-gated-local-feedback-reinforcement.md"],
        )
        target = next(
            hit
            for hit in case["hits"]
            if hit["path"] == "docs/sibling-relation-feedback-normalization.md"
        )
        self.assertEqual(
            target["relation_paths"][0]["steps"][0],
            {
                "source_path": "docs/evidence-gated-local-feedback-reinforcement.md",
                "target_path": "docs/sibling-relation-feedback-normalization.md",
                "edge_type": "markdown_link",
            },
        )

    def test_candidate_is_identical_without_relation_seed(self) -> None:
        protocol = observation.load_worker_protocol()
        synthetic = copy.deepcopy(protocol)
        synthetic["queries"] = {
            "stages": {
                "development": [
                    {
                        "case_id": "synthetic-no-relation-seed",
                        "query": "finite activation budget neural dynamics experiment",
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = observation.run_worker(
                synthetic,
                "development",
                "original-full-query-ngr-default",
                root / "baseline.sqlite",
                protocol_commit=TEST_COMMIT,
                run="primary",
            )
            candidate = observation.run_worker(
                synthetic,
                "development",
                "source-grounded-relation-seed",
                root / "candidate.sqlite",
                protocol_commit=TEST_COMMIT,
                run="primary",
            )
        self.assertEqual(baseline["cases"][0]["hits"], candidate["cases"][0]["hits"])
        self.assertEqual(candidate["cases"][0]["referenced_seed_paths"], [])

    def test_gate_and_arm_identity_are_literal(self) -> None:
        protocol = observation.load_protocol()
        self.assertEqual(tuple(protocol["manifest"]["arms"]), observation.ARMS)
        self.assertEqual(
            tuple(row["gate_id"] for row in protocol["gate"]["gates"]),
            observation.GATE_IDS,
        )
        self.assertEqual(
            protocol["manifest"]["source"]["commit"],
            "74a7ae1b4b9dbe822ef719e4a4b7d0a8b5b3066c",
        )
        self.assertEqual(set(protocol["manifest"]["claims"]), set(observation.STAGES))
        self.assertEqual(
            set(protocol["manifest"]["raw_packets"]), set(observation.STAGES)
        )
        self.assertEqual(
            len(observation._raw_packet_paths(protocol)),
            len(observation.STAGES) * len(observation.ARMS) * len(observation.RUNS),
        )
        self.assertEqual(
            set(protocol["manifest"]["artifact_sha256"]),
            {
                "docs/source-grounded-relation-seed-retrieval-experiment.md",
                "src/neuron_graph_rag/source_grounded_relation_observation.py",
                "tests/fixtures/github_source_grounded_relation_v1.corpus.json",
                "tests/fixtures/github_source_grounded_relation_v1.gate.json",
                "tests/fixtures/github_source_grounded_relation_v1.gold.json",
                "tests/fixtures/github_source_grounded_relation_v1.queries.json",
                "tests/fixtures/github_source_grounded_relation_v1.result-free-audit.json",
                "tests/test_source_grounded_relation_observation.py",
                "tools/acquire_source_grounded_relation_corpus.py",
                "tools/run_source_grounded_relation_observation.py",
            },
        )

    def test_raw_packet_is_exclusive_and_survives_finalizer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {
                "root": root,
                "manifest": {"raw_packets": _raw_packet_registry()},
            }
            packet = _packet("development", observation.ARMS[0], "primary")
            path = observation._persist_worker_packet(
                protocol,
                packet,
                stage="development",
                arm=observation.ARMS[0],
                run="primary",
                protocol_commit=TEST_COMMIT,
            )
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                observation._persist_worker_packet(
                    protocol,
                    packet,
                    stage="development",
                    arm=observation.ARMS[0],
                    run="primary",
                    protocol_commit=TEST_COMMIT,
                )
            with self.assertRaises(FileNotFoundError):
                observation.finalize_stage(
                    protocol, "development", TEST_COMMIT, "0" * 64
                )
            self.assertEqual(path.read_bytes(), original)

    def test_raw_packet_identity_mismatch_fails_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {
                "root": root,
                "manifest": {"raw_packets": _raw_packet_registry()},
            }
            packet = _packet("development", observation.ARMS[1], "primary")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                observation._persist_worker_packet(
                    protocol,
                    packet,
                    stage="development",
                    arm=observation.ARMS[0],
                    run="primary",
                    protocol_commit=TEST_COMMIT,
                )
            self.assertFalse(any(root.rglob("*.json")))

    def test_protocol_commit_binds_manifest_introduction_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Protocol Test"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "core.autocrlf", "false"], cwd=root, check=True
            )
            (root / "README.md").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "base"], cwd=root, check=True
            )
            artifact = root / "artifact.txt"
            artifact.write_text("frozen\n", encoding="utf-8")
            manifest_path = root / "tests/fixtures" / observation.MANIFEST_PATH.name
            manifest_path.parent.mkdir(parents=True)
            manifest = {
                "artifact_sha256": {
                    "artifact.txt": hashlib.sha256(artifact.read_bytes()).hexdigest()
                },
                "claims": {
                    stage: f"evidence/{stage}.claim.json"
                    for stage in observation.STAGES
                },
                "outputs": {
                    stage: f"evidence/{stage}.observed.json"
                    for stage in observation.STAGES
                },
                "raw_packets": _raw_packet_registry(),
            }
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            subprocess.run(
                ["git", "add", "artifact.txt", manifest_path.relative_to(root)],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "freeze"], cwd=root, check=True
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-ref", "refs/remotes/origin/main", commit],
                cwd=root,
                check=True,
            )
            protocol = {"root": root, "manifest": manifest}
            observation.verify_protocol_commit(protocol, commit)

            manifest_path.write_bytes(manifest_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "manifest drifted"):
                observation.verify_protocol_commit(protocol, commit)
            manifest_path.write_bytes(manifest_bytes)

            (root / "README.md").write_text("later\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "later"], cwd=root, check=True
            )
            later = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-ref", "refs/remotes/origin/main", later],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "first introduces"):
                observation.verify_protocol_commit(protocol, later)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from neuron_graph_rag import cross_encoder_precision_v2_evaluation as evaluation
from neuron_graph_rag import cross_encoder_precision_v2_observation as observation


class CrossEncoderPrecisionV2ObservationTest(unittest.TestCase):
    def test_observation_identity_and_offline_environment_are_fixed(self) -> None:
        self.assertEqual(
            observation.PROTOCOL_COMMIT,
            "36c17aac3b49587c97d96bac51db668bf834177b",
        )
        self.assertEqual(
            observation.SOURCE_COMMIT,
            "c32b3049fd3daaa2190faf5e3e85955a195ee88c",
        )
        self.assertEqual(observation.BATCH_SIZE, 8)
        with patch.dict(os.environ, {}, clear=True):
            environment = observation._worker_environment(Path("C:/repo"), offline=True)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONUTF8"], "1")

    def test_combine_state_binds_distinct_fresh_and_replay_runs(self) -> None:
        primary = self._state("primary")
        replay = self._state("replay")
        state = observation._combine_state(primary, replay)
        self.assertEqual(state["fresh_database_id"], "primary-db")
        self.assertEqual(state["replay_database_id"], "replay-db")
        self.assertEqual(state["ranking_sha256"], "primary-ranking")
        self.assertEqual(state["replay_ranking_sha256"], "replay-ranking")
        self.assertTrue(state["cpu_only"])
        self.assertTrue(state["offline"])
        self.assertTrue(state["fresh_process"])

    def test_snapshot_verification_checks_git_and_lfs_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            regular = b"config"
            weights = b"weights"
            (snapshot / "config.json").write_bytes(regular)
            (snapshot / "model.safetensors").write_bytes(weights)
            spec = {
                "model_id": "example/model",
                "revision": "a" * 40,
                "license": "mit",
                "required_files": [
                    {
                        "path": "config.json",
                        "size": len(regular),
                        "git_blob_id": hashlib.sha1(
                            f"blob {len(regular)}\0".encode("ascii") + regular,
                            usedforsecurity=False,
                        ).hexdigest(),
                        "lfs_sha256": None,
                    },
                    {
                        "path": "model.safetensors",
                        "size": len(weights),
                        "git_blob_id": "unused",
                        "lfs_sha256": hashlib.sha256(weights).hexdigest(),
                    },
                ],
            }
            report = observation._verify_snapshot(spec, snapshot)
            self.assertEqual(
                [row["hash_kind"] for row in report["files"]],
                ["git_blob_id", "lfs_sha256"],
            )
            (snapshot / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                observation._verify_snapshot(spec, snapshot)

    def test_verify_preflight_rejects_tampered_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            fixture = root / "tests/fixtures"
            evidence.mkdir(parents=True)
            fixture.mkdir(parents=True)
            lock = fixture / "github_cross_encoder_precision_v2.requirements.lock"
            lock.write_bytes(b"locked\n")
            external = root / "external"
            cache = external / "model-cache"
            cache.mkdir(parents=True)
            model_report = {"models": []}
            dependency = {"lock_sha256": observation.sha256_bytes(lock.read_bytes())}
            report = {
                "protocol_id": observation.PROTOCOL_ID,
                "protocol_commit": observation.PROTOCOL_COMMIT,
                "source_commit": observation.SOURCE_COMMIT,
                "external_root": str(external.resolve()),
                "cache_path": str(cache.resolve()),
                "cache_reused_as_verified_bytes_only": True,
                "v1_evidence_semantic_content_read": False,
                "offline": True,
                "trust_remote_code": False,
                "batch_size": observation.BATCH_SIZE,
                "claim_count": 0,
                "registered_query_execution_count": 0,
                "observed_stage_inference_count": 0,
                "phase": {"development": "unobserved", "holdout": "unobserved"},
                "model_report_sha256": observation.canonical_sha256(model_report),
                "dependency_report_sha256": observation.canonical_sha256(dependency),
                "shared_database_sha256_before": "d" * 64,
                "shared_database_sha256_after": "d" * 64,
            }
            for name, value in (
                ("preflight.json", report),
                ("model-verification.json", model_report),
                ("dependency-report.json", dependency),
                ("preflight-commands.json", {"commands": [{"returncode": 0}]}),
            ):
                (evidence / name).write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(observation, "load_protocol", return_value={}),
                patch.object(observation, "verify_protocol_commit"),
                patch.object(observation, "_verify_model_report"),
                patch.object(
                    observation, "shared_database_path", return_value=root / "shared.db"
                ),
                patch.object(observation, "hash_file_shared", return_value="d" * 64),
            ):
                observation.verify_preflight(root, external, cache)
                report["claim_count"] = 1
                (evidence / "preflight.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    observation.verify_preflight(root, external, cache)

    def test_failed_development_never_starts_holdout_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            cache = root / "cache"
            (root / observation.EVIDENCE).mkdir(parents=True)
            error = RuntimeError("one-shot failure")
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={"shared_database_sha256_before": "e" * 64},
                ),
                patch.object(observation, "_run_stage_once", side_effect=error) as run,
                patch.object(
                    observation, "shared_database_path", return_value=root / "shared.db"
                ),
                patch.object(observation, "hash_file_shared", return_value="e" * 64),
                patch.object(observation, "load_protocol", return_value={}),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={"development": "error", "holdout": "unobserved"},
                ),
                self.assertRaisesRegex(RuntimeError, "one-shot failure"),
            ):
                observation.run_conditional(root, external, cache)
            self.assertEqual(
                run.call_args_list,
                [call("development", root, external, cache, [])],
            )
            failure = json.loads(
                (root / observation.EVIDENCE / "execution-error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["phase"]["holdout"], "unobserved")
            self.assertTrue(failure["shared_database_unchanged"])

    def test_holdout_runs_only_after_development_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            cache = root / "cache"
            (root / observation.EVIDENCE).mkdir(parents=True)
            development = {
                "all_hard_gates_pass": True,
                "selected_candidate_id": "base",
            }
            holdout = {
                "all_hard_gates_pass": True,
                "selected_candidate_id": "base",
            }
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={"shared_database_sha256_before": "f" * 64},
                ),
                patch.object(
                    observation,
                    "_run_stage_once",
                    side_effect=[development, holdout],
                ) as run,
                patch.object(
                    observation, "shared_database_path", return_value=root / "shared.db"
                ),
                patch.object(observation, "hash_file_shared", return_value="f" * 64),
                patch.object(observation, "load_protocol", return_value={}),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={"development": "passed", "holdout": "passed"},
                ),
            ):
                result = observation.run_conditional(root, external, cache)
            self.assertEqual(
                [row.args[0] for row in run.call_args_list], ["development", "holdout"]
            )
            self.assertEqual(result["execution"]["claim_count"], 2)

    def test_six_fresh_worker_packets_archive_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "external/development"
            stage_root.mkdir(parents=True)
            names = [
                f"{kind}-{replay}.json"
                for kind in ("baseline", "base", "v2-m3")
                for replay in ("primary", "replay")
            ]
            for index, name in enumerate(names):
                (stage_root / name).write_bytes(f'{{"worker":{index}}}\n'.encode())
            manifest_path = observation._archive_raw_workers(
                "development", stage_root, root
            )
            manifest = observation.read_json(manifest_path)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["archived_worker_packet_count"], 6)
            for row in manifest["files"]:
                archive = root / row["archive_path"]
                self.assertEqual(
                    observation.sha256_bytes(archive.read_bytes()), row["sha256"]
                )
                self.assertTrue(row["byte_identity"])

    @staticmethod
    def _state(prefix: str) -> dict[str, object]:
        return {
            "database_id": f"{prefix}-db",
            "ranking_sha256": f"{prefix}-ranking",
            "activation_sha256": f"{prefix}-activation",
            "edge_sha256_before": "edge",
            "edge_sha256_after": "edge",
            "feedback_count_before": 0,
            "feedback_count_after": 0,
            "sqlite_sha256_before": "sqlite",
            "sqlite_sha256_after": "sqlite",
        }


class CrossEncoderPrecisionV2ObservationEvidenceTest(unittest.TestCase):
    def test_committed_development_evidence_is_exact_and_fully_verified(self) -> None:
        protocol = evaluation.load_protocol()
        self.assertEqual(
            evaluation.verify_phase_state(protocol),
            {"development": "archived-failed", "holdout": "unobserved"},
        )
        evidence = evaluation.ROOT / observation.EVIDENCE
        expected_hashes = {
            "development.claim.json": (
                "437450a4e8fdcc488b4409ac14cff9133c152c8945a11081e268f93ae08efdbc"
            ),
            "development.observed.json": (
                "83e7cbbc7e09db2189edc535372d317ce69810c5601149d6acf5b2e308bae007"
            ),
            "development.transport.json": (
                "7eafcb3a442bc3a5da94a25c0867b4bb283b468fd60dcd11444c2bb60e9d0838"
            ),
            "development.raw-archive.json": (
                "7e9b3bc45fa6a7c65fad0f9c45414cad18fe17dc6559ea998179911be054aca7"
            ),
            "execution.json": (
                "d41ba9a93b8048c0be15bb3fcf7d830a744ade6813f663e727c2a62e226496cd"
            ),
        }
        for name, expected in expected_hashes.items():
            self.assertEqual(
                evaluation.sha256_bytes((evidence / name).read_bytes()), expected
            )

        claim_raw = (evidence / "development.claim.json").read_bytes()
        result = observation.read_json(evidence / "development.observed.json")
        evaluation.verify_result_payload(protocol, "development", result, claim_raw)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertFalse(result["all_hard_gates_pass"])
        self.assertEqual(len(result["candidates"]), 4)
        self.assertTrue(
            all(
                candidate["all_hard_gates_pass"] is False
                for candidate in result["candidates"]
            )
        )
        self.assertEqual(
            {gate["gate_id"] for gate in result["gates"] if not gate["passed"]},
            {
                "positive-case-rank-non-regression",
                "positive-cohort-mrr-hit-at-5-non-regression",
                "positive-expected-source-top-5-completeness",
                "relation-source-edge-only-provenance",
            },
        )

        transport = observation.read_json(evidence / "development.transport.json")
        self.assertEqual(transport["stage_execution_count"], 1)
        self.assertTrue(all(row["byte_identity"] for row in transport["files"]))
        self.assertEqual(
            [row["sha256"] for row in transport["files"]],
            [
                expected_hashes["development.claim.json"],
                expected_hashes["development.observed.json"],
            ],
        )

        raw_manifest = observation.read_json(
            evidence / "development.raw-archive.json"
        )
        expected_raw_hashes = {
            "baseline-primary.json": (
                "b816256f4e174af7722b027e582196af620fb8d7384af292d96e8ae2115e78dc"
            ),
            "baseline-replay.json": (
                "8fb51c7f1f17af0f74d43410992a45d5eda2863f821147ef4c32ae177ee93989"
            ),
            "base-primary.json": (
                "5d2ae8f0372627846c0bc1346d755ce56c15be8935a0cf669e71e92605b9cb45"
            ),
            "base-replay.json": (
                "b5bc0f1ab5a3eacef4d59e4de125a1f41654e2f9ff8c9e1a72b7c19d9b79ebcf"
            ),
            "v2-m3-primary.json": (
                "ce2638c95ee0383aea5f491804f7dd284c55f6f2c10aa92955fe2595db816d73"
            ),
            "v2-m3-replay.json": (
                "ec014f6167520416a947714c1dc3b8f35db4cc60eeba2c314baf290c5b53b005"
            ),
        }
        self.assertEqual(raw_manifest["stage_execution_count"], 1)
        self.assertEqual(raw_manifest["expected_worker_packet_count"], 6)
        self.assertEqual(raw_manifest["archived_worker_packet_count"], 6)
        self.assertTrue(raw_manifest["complete"])
        self.assertTrue(all(row["byte_identity"] for row in raw_manifest["files"]))
        self.assertEqual(
            {
                Path(row["archive_path"]).name: row["sha256"]
                for row in raw_manifest["files"]
            },
            expected_raw_hashes,
        )

        raw_packets = [
            observation.read_json(evidence / "raw/development" / name)
            for name in expected_raw_hashes
        ]
        self.assertEqual(sum(len(packet["cases"]) for packet in raw_packets), 48)
        self.assertEqual(len({packet["database_id"] for packet in raw_packets}), 6)
        for name, expected in expected_raw_hashes.items():
            self.assertEqual(
                evaluation.sha256_bytes(
                    (evidence / "raw/development" / name).read_bytes()
                ),
                expected,
            )

        execution = observation.read_json(evidence / "execution.json")
        self.assertEqual(execution["claim_count"], 1)
        self.assertEqual(execution["model_stage_process_count"], 4)
        self.assertEqual(len(execution["commands"]), 6)
        self.assertTrue(all(row["returncode"] == 0 for row in execution["commands"]))
        self.assertEqual(
            execution["phase"],
            {"development": "archived-failed", "holdout": "unobserved"},
        )
        self.assertIsNone(execution["selected_candidate"]["development"])
        self.assertEqual(
            execution["shared_database_sha256_before"],
            execution["shared_database_sha256_after"],
        )
        self.assertFalse((evidence / "execution-error.json").exists())
        self.assertFalse((evidence / "development.error.json").exists())
        for relative in protocol["manifest"]["outputs"]["holdout"].values():
            self.assertFalse((evaluation.ROOT / relative).exists())


if __name__ == "__main__":
    unittest.main()

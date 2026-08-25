from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

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


if __name__ == "__main__":
    unittest.main()

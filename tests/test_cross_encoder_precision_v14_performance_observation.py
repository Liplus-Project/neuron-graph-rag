from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v14_performance_observation as observation,
)


class CrossEncoderPrecisionV14PerformanceObservationTest(unittest.TestCase):
    def test_identity_workers_and_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "56d32bac8144b96b03a6813d8732600a3491f8c9",
        )
        self.assertEqual(observation.WSLC_VERSION, "2.9.4.0")
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v14-runtime"
        )
        self.assertEqual(
            observation.V11_ROOT_FREEZE_VOLUME,
            "github-cross-encoder-precision-v11-root-freeze",
        )
        self.assertEqual(
            observation.V13_COMMIT_FREEZE_VOLUME,
            "github-cross-encoder-precision-v13-commit-freeze",
        )
        self.assertEqual(len(observation.WORKERS), 6)
        for value in (
            observation.CONTAINER_ROOT,
            observation.CONTAINER_SOURCE,
            observation.CONTAINER_CACHE,
            observation.CONTAINER_PROTOCOL_SOURCE,
            observation.CONTAINER_DATABASES,
            observation.CONTAINER_RUNS,
            observation.CONTAINER_ARCHIVE,
            observation.CONTAINER_TRANSPORT,
        ):
            self.assertIsInstance(value, PurePosixPath)
            self.assertEqual(
                observation.serialize_container_path(value), value.as_posix()
            )

    def test_host_path_cannot_cross_container_path_boundary(self) -> None:
        with self.assertRaises(TypeError):
            observation.serialize_container_path(  # type: ignore[arg-type]
                Path("/opt/ngr-v14/runtime")
            )
        with tempfile.TemporaryDirectory() as directory:
            spec = observation.host_bind_spec(
                Path(directory), "/input/source", mode="ro"
            )
            self.assertTrue(spec.endswith(":/input/source:ro"))
            with self.assertRaises(ValueError):
                observation.host_bind_spec(
                    Path(directory), "C:\\input\\source", mode="ro"
                )

    def test_container_command_is_offline_and_mounts_only_v14_runtime(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v14-development-baseline-primary",
        )
        rendered = "\n".join(command)
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v14-runtime:/opt/ngr-v14/runtime",
        )
        for forbidden in (
            observation.V13_COMMIT_FREEZE_VOLUME,
            observation.V12_RUNTIME_VOLUME,
            observation.V11_ROOT_FREEZE_VOLUME,
            observation.V10_RUNTIME_VOLUME,
            observation.V10_CACHE_FREEZE_VOLUME,
            str(observation.OLD_V8_ROOT),
            str(observation.SHARED_DATABASE),
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn(
            "neuron_graph_rag.cross_encoder_precision_v14_performance_observation",
            command,
        )

    def test_source_initialization_leaves_model_cache_and_old_root_absent(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e '/opt/ngr-v14/runtime/model-cache'", script)
        self.assertIn("test ! -e '/opt/ngr-v8/runtime'", script)
        self.assertIn("mkdir '/opt/ngr-v14/runtime/source'", script)
        self.assertNotIn("mkdir -p /opt/ngr-v14/runtime/model-cache", script)
        for path in ("databases", "runs", "archive", "transport"):
            self.assertIn(f"/opt/ngr-v14/runtime/{path}", script)
        self.assertNotIn("knowledge.db", script)

    def test_model_copy_uses_frozen_exclusive_verifier(self) -> None:
        with patch.object(
            observation.model_freeze,
            "_container_model_copy_verify",
            return_value={
                "target_exclusive_create": True,
                "revision_count": 2,
                "file_count": 12,
                "total_size": 3_427_616_927,
            },
        ) as verifier:
            result = observation._container_model_copy(
                "/input/models",
                "/opt/ngr-v14/runtime/model-cache",
                "/opt/ngr-v14/runtime/model-verification.json",
            )
        self.assertTrue(result["target_exclusive_create"])
        self.assertEqual(result["revision_count"], 2)
        self.assertEqual(result["file_count"], 12)
        self.assertEqual(result["total_size"], 3_427_616_927)
        self.assertEqual(
            verifier.call_args.args[2],
            "/opt/ngr-v14/runtime/source/tests/fixtures/"
            "github_cross_encoder_precision_v8.models.json",
        )

    def test_configure_uses_git_free_binder_for_actual_v14_roots(self) -> None:
        source_identity = {
            "identity_schema": "ngr.git-free-protocol-identity/v1"
        }
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(observation, "read_json", return_value=source_identity),
            patch.object(
                observation.git_free_freeze,
                "bind_git_free_commit_verifier",
                return_value={"git_free_verifier_bound": True},
            ) as binder,
        ):
            observation._configure_container_harness()
        self.assertIs(binder.call_args.args[0], observation.lifecycle.predecessor)
        self.assertEqual(binder.call_args.kwargs["volume"], observation.VOLUME)
        self.assertEqual(
            binder.call_args.kwargs["root"], Path("/opt/ngr-v14/runtime")
        )
        self.assertEqual(
            binder.call_args.kwargs["source"],
            Path("/opt/ngr-v14/runtime/source"),
        )
        self.assertEqual(
            binder.call_args.kwargs["protocol_source"],
            Path("/opt/ngr-v14/runtime/frozen-source"),
        )
        self.assertEqual(
            binder.call_args.kwargs["evidence"],
            Path(
                "/opt/ngr-v14/runtime/source/tests/evidence/"
                "github_cross_encoder_precision_v14_observation"
            ),
        )
        self.assertIs(binder.call_args.kwargs["identity"], source_identity)

    def test_v14_scope_restores_v10_and_nested_lifecycle_defaults(self) -> None:
        original = {
            "protocol": observation.lifecycle.PROTOCOL_ID,
            "root": observation.lifecycle.CONTAINER_ROOT,
            "configure": observation.lifecycle.lifecycle._configure_container_harness,
        }
        with observation._v14_scope():
            self.assertEqual(observation.lifecycle.PROTOCOL_ID, observation.PROTOCOL_ID)
            self.assertEqual(
                observation.lifecycle.CONTAINER_ROOT, observation.CONTAINER_ROOT
            )
            self.assertIs(
                observation.lifecycle.lifecycle._configure_container_harness,
                observation._configure_container_harness,
            )
        self.assertEqual(observation.lifecycle.PROTOCOL_ID, original["protocol"])
        self.assertEqual(observation.lifecycle.CONTAINER_ROOT, original["root"])
        self.assertIs(
            observation.lifecycle.lifecycle._configure_container_harness,
            original["configure"],
        )

    def test_platform_evidence_names_all_legacy_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "platform-report.json"
            observation._write_lifecycle_json_exclusive(
                path,
                {
                    "path_freeze_volume": observation.V13_COMMIT_FREEZE_VOLUME,
                    "path_freeze_volume_mounted": False,
                    "path_freeze_volume_read": False,
                },
            )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("path_freeze_volume", report)
        self.assertEqual(
            report["v13_commit_freeze_volume"],
            observation.V13_COMMIT_FREEZE_VOLUME,
        )
        for key in (
            "v13_commit_freeze_volume_mounted",
            "v13_commit_freeze_volume_read",
            "v13_commit_freeze_volume_reused",
            "v12_runtime_volume_mounted",
            "v12_runtime_volume_read",
            "v12_runtime_volume_reused",
            "v11_root_freeze_volume_mounted",
            "v11_root_freeze_volume_read",
            "v11_root_freeze_volume_reused",
            "v10_runtime_volume_mounted",
            "v10_runtime_volume_read",
            "v10_runtime_volume_reused",
            "v10_cache_freeze_volume_mounted",
            "v10_cache_freeze_volume_read",
            "v10_cache_freeze_volume_reused",
            "old_v8_root_created",
            "old_v8_root_mounted",
            "old_v8_root_read",
        ):
            self.assertFalse(report[key])

        original = {
            "path_freeze_volume": observation.V13_COMMIT_FREEZE_VOLUME,
            "path_freeze_volume_mounted": False,
            "path_freeze_volume_read": False,
        }
        self.assertEqual(
            observation._canonical_lifecycle_value(report),
            observation.canonical_sha256(original),
        )

    def test_manifest_freezes_v13_closure_and_v8_image_contract(self) -> None:
        contract = observation._stored_freeze_contract(observation.ROOT)
        self.assertEqual(contract["v13_artifact_count"], 15)
        self.assertEqual(
            contract["v13_commit_identity_sha256"],
            observation.V13_COMMIT_IDENTITY_SHA256,
        )
        self.assertEqual(contract["accepted_image"]["id"], observation.IMAGE_ID)
        self.assertEqual(contract["expected_distribution_count"], 29)
        for key in (
            "v13_commit_freeze_volume_mounted",
            "v13_commit_freeze_volume_read",
            "v13_commit_freeze_volume_copied",
            "v13_commit_freeze_volume_reused",
            "v12_runtime_volume_mounted",
            "v12_runtime_volume_read",
            "v12_runtime_volume_reused",
            "v11_root_freeze_volume_mounted",
            "v11_root_freeze_volume_read",
            "v11_root_freeze_volume_copied",
            "v11_root_freeze_volume_reused",
            "v10_runtime_volume_mounted",
            "v10_runtime_volume_read",
            "v10_runtime_volume_reused",
            "v10_cache_freeze_volume_mounted",
            "v10_cache_freeze_volume_read",
            "v10_cache_freeze_volume_reused",
            "old_v8_root_created",
            "old_v8_root_mounted",
            "old_v8_root_read",
        ):
            self.assertFalse(contract[key])

    def test_development_error_is_terminal_and_never_opens_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            with (
                patch.object(
                    observation.lifecycle.lifecycle,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64
                    },
                ),
                patch.object(
                    observation.lifecycle.lifecycle,
                    "_git_output",
                    return_value="b" * 40,
                ),
                patch.object(
                    observation.lifecycle.lifecycle,
                    "_remote_ci_green",
                    return_value={"preflight_evidence_commit": "b" * 40},
                ),
                patch.object(observation.lifecycle, "_sync_preflight_evidence"),
                patch.object(
                    observation.lifecycle.lifecycle,
                    "_hash_shared_database",
                    return_value="a" * 64,
                ),
                patch.object(
                    observation,
                    "_run_stage_host",
                    side_effect=RuntimeError("one-shot failure"),
                ) as stage,
                patch.object(observation.lifecycle, "_export_volume_evidence"),
                self.assertRaisesRegex(RuntimeError, "one-shot failure"),
            ):
                observation.run_once(root)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[0], "development")
            failure = json.loads(
                (evidence / "execution-error.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["retry_count"], 0)
            self.assertEqual(failure["development_claim_count"], 0)
            self.assertEqual(failure["holdout_claim_count"], 0)

    def test_terminal_root_mismatch_is_fail_closed_and_result_free(self) -> None:
        with patch.object(
            observation,
            "_hash_shared_database",
            side_effect=AssertionError("terminal audit must not reopen shared DB"),
        ):
            result = observation.audit_evidence(observation.ROOT)
        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["failure_point"], "development-claim-protocol-root"
        )
        for key in (
            "development_claim_count",
            "holdout_claim_count",
            "worker_process_count",
            "observed_result_count",
            "retry_count",
        ):
            self.assertEqual(result[key], 0)
        self.assertTrue(result["shared_database_unchanged"])
        self.assertEqual(result["performance"], "not assessed")

    def test_preflight_error_is_terminal_and_forbids_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            raw = {
                "protocol_id": observation.PROTOCOL_ID,
                "implementation_commit": "e" * 40,
                "error": "RuntimeError: preflight stopped",
                "runtime_volume_create_count": 1,
                "development_claim_count": 0,
                "holdout_claim_count": 0,
                "registered_query_execution_count": 0,
                "preflight_forward_inference_count": 2,
                "observed_stage_inference_count": 0,
                "result_count": 0,
                "retry_count": 0,
                "shared_database_sha256_before_preflight": "f" * 64,
            }
            raw_path = evidence / "preflight.error.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_before = raw_path.read_bytes()
            with patch.object(
                observation, "_hash_shared_database", return_value="f" * 64
            ):
                terminal = observation.finalize_preflight_error(root)
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertEqual(terminal["retry_count"], 0)
            self.assertFalse(terminal["same_protocol_retry_allowed"])
            self.assertTrue(terminal["shared_database_unchanged"])
            self.assertEqual(terminal["performance"], "not assessed")
            for key in (
                "v13_commit_freeze_volume_mounted",
                "v13_commit_freeze_volume_read",
                "v13_commit_freeze_volume_reused",
                "v12_runtime_volume_mounted",
                "v12_runtime_volume_read",
                "v12_runtime_volume_reused",
                "v11_root_freeze_volume_mounted",
                "v11_root_freeze_volume_read",
                "v11_root_freeze_volume_reused",
                "v10_runtime_volume_mounted",
                "v10_runtime_volume_read",
                "v10_runtime_volume_reused",
                "v10_cache_freeze_volume_mounted",
                "v10_cache_freeze_volume_read",
                "v10_cache_freeze_volume_reused",
                "old_v8_root_created",
                "old_v8_root_mounted",
                "old_v8_root_read",
            ):
                self.assertFalse(terminal[key])
            self.assertTrue(
                (evidence / "observation-evidence-manifest.json").is_file()
            )

    def test_holdout_opens_only_after_selected_all_gate_development(self) -> None:
        scenarios = (
            ({"all_hard_gates_pass": False, "selected_candidate_id": None}, 1),
            ({"all_hard_gates_pass": True, "selected_candidate_id": None}, 1),
            ({"all_hard_gates_pass": True, "selected_candidate_id": "base"}, 2),
        )
        for development, expected_calls in scenarios:
            with self.subTest(development=development), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / observation.EVIDENCE).mkdir(parents=True)
                holdout = {
                    "all_hard_gates_pass": True,
                    "selected_candidate_id": "base",
                }

                def run_stage(
                    stage: str,
                    _root: Path,
                    _rows: list[dict[str, object]],
                    claim_counts: dict[str, int],
                    development_result: dict[str, object] = development,
                    holdout_result: dict[str, object] = holdout,
                ) -> dict[str, object]:
                    claim_counts[stage] += 1
                    return (
                        development_result
                        if stage == "development"
                        else holdout_result
                    )

                with (
                    patch.object(
                        observation.lifecycle.lifecycle,
                        "verify_preflight",
                        return_value={
                            "shared_database_sha256_before_preflight": "c" * 64
                        },
                    ),
                    patch.object(
                        observation.lifecycle.lifecycle,
                        "_git_output",
                        return_value="d" * 40,
                    ),
                    patch.object(
                        observation.lifecycle.lifecycle,
                        "_remote_ci_green",
                        return_value={"preflight_evidence_commit": "d" * 40},
                    ),
                    patch.object(observation.lifecycle, "_sync_preflight_evidence"),
                    patch.object(
                        observation.lifecycle.lifecycle,
                        "_hash_shared_database",
                        return_value="c" * 64,
                    ),
                    patch.object(
                        observation, "_run_stage_host", side_effect=run_stage
                    ) as stage,
                ):
                    result = observation.run_once(root)
                self.assertEqual(stage.call_count, expected_calls)
                self.assertEqual(
                    result["execution"]["holdout_claim_count"],
                    int(expected_calls == 2),
                )

    def test_worker_commands_fix_six_fresh_processes_and_databases(self) -> None:
        rows: list[dict[str, object]] = []
        claims = {"development": 0, "holdout": 0}
        commands: list[list[str]] = []

        def logged(
            command: list[str],
            _root: Path,
            _rows: list[dict[str, object]],
            **_kwargs: object,
        ) -> str:
            commands.append(command)
            if "finalize" in command:
                return '{"all_hard_gates_pass": false, "selected_candidate_id": null}'
            return "{}"

        with (
            patch.object(
                observation.lifecycle.lifecycle,
                "_run_logged",
                side_effect=logged,
            ),
            patch.object(observation.lifecycle, "_export_volume_evidence"),
        ):
            observation._run_stage_host(
                "development", observation.ROOT, rows, claims
            )
        worker_commands = [command for command in commands if "worker" in command]
        self.assertEqual(len(worker_commands), 6)
        self.assertEqual(claims["development"], 1)
        names = {
            command[command.index("--name") + 1] for command in worker_commands
        }
        databases = {
            command[command.index("--database") + 1]
            for command in worker_commands
        }
        self.assertEqual(len(names), 6)
        self.assertEqual(len(databases), 6)
        self.assertTrue(all(name.startswith("ngr-v14-") for name in names))
        self.assertTrue(
            all(
                path.startswith("/opt/ngr-v14/runtime/databases/development/")
                for path in databases
            )
        )
        self.assertTrue(all("\\" not in path for path in databases))

    def test_frozen_rank_candidates_and_top5_remain_unchanged(self) -> None:
        fixture = json.loads(
            (
                observation.ROOT
                / "tests/fixtures/github_cross_encoder_precision_v8.candidates.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(fixture["top_k"], 5)
        self.assertEqual(
            [row["candidate_id"] for row in fixture["candidates"]],
            [
                "bge-base-rrf-rank-only",
                "bge-base-ce-rank-only",
                "bge-v2-m3-rrf-rank-only",
                "bge-v2-m3-ce-rank-only",
            ],
        )

    def test_json_contract_is_utf8_without_replacement_character(self) -> None:
        raw = (observation.ROOT / observation.MANIFEST).read_bytes()
        value = json.loads(raw.decode("utf-8"))
        self.assertIsInstance(value, dict)
        self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

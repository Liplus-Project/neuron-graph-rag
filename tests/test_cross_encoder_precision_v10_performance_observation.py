from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v10_performance_observation as observation,
)


class CrossEncoderPrecisionV10PerformanceObservationTest(unittest.TestCase):
    def test_identity_worker_order_and_runtime_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "e75d1e065441b794ce83b68f62d55747741052e5",
        )
        self.assertEqual(observation.WSLC_VERSION, "2.9.4.0")
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v10-runtime"
        )
        self.assertEqual(
            observation.CACHE_FREEZE_VOLUME,
            "github-cross-encoder-precision-v10-cache-freeze",
        )
        self.assertEqual(
            observation.WORKERS,
            (
                ("baseline", "primary"),
                ("baseline", "replay"),
                ("base", "primary"),
                ("base", "replay"),
                ("v2-m3", "primary"),
                ("v2-m3", "replay"),
            ),
        )
        for value in (
            observation.CONTAINER_ROOT,
            observation.CONTAINER_SOURCE,
            observation.CONTAINER_CACHE,
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
            observation.serialize_container_path(Path("/opt/ngr-v10/runtime"))  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            spec = observation.host_bind_spec(
                Path(directory), "/input/source", mode="ro"
            )
            self.assertTrue(spec.endswith(":/input/source:ro"))
            with self.assertRaises(ValueError):
                observation.host_bind_spec(
                    Path(directory), "C:\\input\\source", mode="ro"
                )

    def test_container_command_is_offline_and_never_mounts_cache_freeze(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v10-development-baseline-primary",
        )
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertIn("--rm", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v10-runtime:/opt/ngr-v10/runtime",
        )
        self.assertNotIn(observation.CACHE_FREEZE_VOLUME, "\n".join(command))
        self.assertNotIn(str(observation.SHARED_DATABASE), command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn(
            "neuron_graph_rag.cross_encoder_precision_v10_performance_observation",
            command,
        )

    def test_source_initialization_leaves_model_cache_absent(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e '/opt/ngr-v10/runtime/model-cache'", script)
        self.assertIn("mkdir '/opt/ngr-v10/runtime/source'", script)
        self.assertNotIn(
            "mkdir -p /opt/ngr-v10/runtime/model-cache",
            script,
        )
        for path in ("databases", "runs", "archive", "transport"):
            self.assertIn(f"/opt/ngr-v10/runtime/{path}", script)
        self.assertNotIn("knowledge.db", script)

    def test_model_copy_uses_frozen_v10_exclusive_verifier(self) -> None:
        with patch.object(
            observation.cache_freeze,
            "_container_model_copy_verify",
            return_value={"target_exclusive_create": True},
        ) as verifier:
            result = observation._container_model_copy(
                "/input/models",
                "/opt/ngr-v10/runtime/model-cache",
                "/opt/ngr-v10/runtime/model-verification.json",
            )
        self.assertTrue(result["target_exclusive_create"])
        self.assertEqual(
            verifier.call_args.args[2],
            "/opt/ngr-v10/runtime/source/tests/fixtures/"
            "github_cross_encoder_precision_v8.models.json",
        )

    def test_platform_evidence_names_cache_freeze_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "platform-report.json"
            observation._write_lifecycle_json_exclusive(
                path,
                {
                    "path_freeze_volume": observation.CACHE_FREEZE_VOLUME,
                    "path_freeze_volume_mounted": False,
                    "path_freeze_volume_read": False,
                },
            )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("path_freeze_volume", report)
        self.assertEqual(
            report["cache_freeze_volume"], observation.CACHE_FREEZE_VOLUME
        )
        self.assertFalse(report["cache_freeze_volume_mounted"])
        self.assertFalse(report["cache_freeze_volume_read"])
        self.assertFalse(report["cache_freeze_volume_reused"])

    def test_manifest_freezes_cache_predecessor_and_v8_contract(self) -> None:
        contract = observation._stored_freeze_contract(observation.ROOT)
        self.assertEqual(contract["cache_freeze_artifact_count"], 15)
        self.assertEqual(contract["predecessor_artifact_count"], 20)
        self.assertEqual(
            contract["cache_freeze_pass_sha256"],
            observation.CACHE_FREEZE_PASS_SHA256,
        )
        self.assertEqual(contract["accepted_image"]["id"], observation.IMAGE_ID)
        self.assertEqual(contract["expected_distribution_count"], 29)
        self.assertFalse(contract["cache_freeze_volume_mounted"])
        self.assertFalse(contract["cache_freeze_volume_read"])
        self.assertFalse(contract["cache_freeze_volume_copied"])
        self.assertFalse(contract["cache_freeze_volume_reused"])

    def test_development_error_is_terminal_and_never_opens_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            with (
                patch.object(
                    observation.lifecycle,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64
                    },
                ),
                patch.object(
                    observation.lifecycle, "_git_output", return_value="b" * 40
                ),
                patch.object(
                    observation.lifecycle,
                    "_remote_ci_green",
                    return_value={"preflight_evidence_commit": "b" * 40},
                ),
                patch.object(observation, "_sync_preflight_evidence"),
                patch.object(
                    observation.lifecycle,
                    "_hash_shared_database",
                    return_value="a" * 64,
                ),
                patch.object(
                    observation,
                    "_run_stage_host",
                    side_effect=RuntimeError("one-shot failure"),
                ) as stage,
                patch.object(observation, "_export_volume_evidence"),
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

    def test_preflight_error_finalizer_preserves_raw_and_forbids_retry(self) -> None:
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
            self.assertTrue(terminal["shared_database_post_error_hash_recorded"])
            self.assertTrue(terminal["shared_database_unchanged"])
            self.assertEqual(terminal["performance"], "not assessed")
            self.assertTrue(
                (evidence / "observation-evidence-manifest.json").is_file()
            )

    def test_holdout_opens_only_for_selected_all_gate_development(self) -> None:
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
                        observation.lifecycle,
                        "verify_preflight",
                        return_value={
                            "shared_database_sha256_before_preflight": "c" * 64
                        },
                    ),
                    patch.object(
                        observation.lifecycle,
                        "_git_output",
                        return_value="d" * 40,
                    ),
                    patch.object(
                        observation.lifecycle,
                        "_remote_ci_green",
                        return_value={"preflight_evidence_commit": "d" * 40},
                    ),
                    patch.object(observation, "_sync_preflight_evidence"),
                    patch.object(
                        observation.lifecycle,
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

    def test_worker_commands_fix_six_fresh_process_and_database_paths(self) -> None:
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
            patch.object(observation.lifecycle, "_run_logged", side_effect=logged),
            patch.object(observation, "_export_volume_evidence"),
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
        self.assertTrue(
            all(
                path.startswith("/opt/ngr-v10/runtime/databases/development/")
                for path in databases
            )
        )
        self.assertTrue(all("\\" not in path for path in databases))

    def test_frozen_rank_only_candidates_and_top5_are_unchanged(self) -> None:
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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v9_performance_observation as observation,
)


class CrossEncoderPrecisionV9PerformanceObservationTest(unittest.TestCase):
    def test_identity_worker_order_and_runtime_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "25790b5218ccc7a5741dbdf6a19d1f7723d7afeb",
        )
        self.assertEqual(observation.WSLC_VERSION, "2.9.4.0")
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v9-runtime"
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
            observation.serialize_container_path(Path("/opt/ngr-v9/runtime"))  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            spec = observation.host_bind_spec(
                Path(directory), "/input/source", mode="ro"
            )
            self.assertTrue(spec.endswith(":/input/source:ro"))
            with self.assertRaises(ValueError):
                observation.host_bind_spec(
                    Path(directory), "C:\\input\\source", mode="ro"
                )

    def test_container_command_is_offline_runtime_scoped_and_path_safe(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v9-development-baseline-primary",
        )
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertIn("--rm", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v9-runtime:/opt/ngr-v9/runtime",
        )
        self.assertNotIn(observation.PATH_FREEZE_VOLUME, "\n".join(command))
        self.assertNotIn(str(observation.SHARED_DATABASE), command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn(
            "neuron_graph_rag.cross_encoder_precision_v9_performance_observation",
            command,
        )
        for argument in command:
            if argument.startswith("github-cross-encoder-precision-v9-runtime:"):
                self.assertNotIn("\\", argument)

    def test_source_initialization_uses_distinct_posix_roots(self) -> None:
        script = observation._source_initialization_script()
        for path in (
            "/opt/ngr-v9/runtime/source",
            "/opt/ngr-v9/runtime/model-cache",
            "/opt/ngr-v9/runtime/databases",
            "/opt/ngr-v9/runtime/runs",
            "/opt/ngr-v9/runtime/archive",
            "/opt/ngr-v9/runtime/transport",
        ):
            self.assertIn(path, script)
        self.assertIn("test ! -e '/opt/ngr-v9/runtime/source'", script)
        self.assertNotIn("knowledge.db", script)

    def test_manifest_freezes_path_predecessor_and_v8_contract(self) -> None:
        contract = observation._stored_freeze_contract(observation.ROOT)
        self.assertEqual(contract["path_freeze_artifact_count"], 12)
        self.assertEqual(contract["v8_predecessor_artifact_count"], 29)
        self.assertEqual(contract["path_smoke_sha256"], observation.PATH_SMOKE_SHA256)
        self.assertEqual(contract["count_audit_sha256"], observation.COUNT_AUDIT_SHA256)
        self.assertEqual(contract["accepted_image"]["id"], observation.IMAGE_ID)
        self.assertEqual(contract["expected_distribution_count"], 29)
        self.assertEqual(contract["additional_image_build_count"], 0)
        self.assertEqual(contract["additional_runtime_content_report_count"], 0)
        self.assertEqual(contract["additional_attestation_report_count"], 0)
        self.assertFalse(contract["path_freeze_volume_mounted"])
        self.assertFalse(contract["path_freeze_volume_read"])

    def test_preflight_source_has_no_freeze_rerun_command(self) -> None:
        source = Path(observation.__file__).read_text(encoding="utf-8")
        preflight = source[
            source.index("def preflight(") : source.index("def _verify_hash_manifest(")
        ]
        self.assertNotIn('"build"', preflight)
        self.assertNotIn("runtime_content.py", preflight)
        self.assertNotIn("validate_runtime.py", preflight)
        self.assertNotIn(
            "named_volume_spec(PATH_FREEZE_VOLUME", preflight
        )

    def test_development_error_is_terminal_and_never_opens_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64
                    },
                ),
                patch.object(observation, "_git_output", return_value="b" * 40),
                patch.object(
                    observation,
                    "_remote_ci_green",
                    return_value={"preflight_evidence_commit": "b" * 40},
                ),
                patch.object(observation, "_sync_preflight_evidence"),
                patch.object(
                    observation, "_hash_shared_database", return_value="a" * 64
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
            self.assertTrue(
                (evidence / "observation-evidence-manifest.json").is_file()
            )

    def test_preflight_error_finalizer_preserves_raw_and_records_post_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            raw = {
                "protocol_id": observation.PROTOCOL_ID,
                "implementation_commit": "e" * 40,
                "error": "FileExistsError: dedicated ext4 model cache already exists",
                "runtime_volume_create_count": 1,
                "development_claim_count": 0,
                "holdout_claim_count": 0,
                "registered_query_execution_count": 0,
                "preflight_forward_inference_count": 0,
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
            self.assertEqual(terminal["development_claim_count"], 0)
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
                evidence = root / observation.EVIDENCE
                evidence.mkdir(parents=True)
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
                        observation,
                        "verify_preflight",
                        return_value={
                            "shared_database_sha256_before_preflight": "c" * 64
                        },
                    ),
                    patch.object(observation, "_git_output", return_value="d" * 40),
                    patch.object(
                        observation,
                        "_remote_ci_green",
                        return_value={"preflight_evidence_commit": "d" * 40},
                    ),
                    patch.object(observation, "_sync_preflight_evidence"),
                    patch.object(
                        observation,
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
            patch.object(observation, "_run_logged", side_effect=logged),
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
            all(path.startswith("/opt/ngr-v9/runtime/databases/development/") for path in databases)
        )
        self.assertTrue(all("\\" not in path for path in databases))

    def test_json_contract_is_utf8_without_replacement_character(self) -> None:
        raw = (observation.ROOT / observation.MANIFEST).read_bytes()
        value = json.loads(raw.decode("utf-8"))
        self.assertIsInstance(value, dict)
        self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

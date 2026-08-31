from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v19_performance_observation as observation,
)
from neuron_graph_rag import rank_observation_stage_contract


class CrossEncoderPrecisionV19PerformanceObservationTest(unittest.TestCase):
    def _copy_prebuild_contract(self, root: Path) -> None:
        for relative in (
            observation.MANIFEST,
            observation.SOURCE_IDENTITY,
            observation.OBSERVATION_AUDIT,
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((observation.ROOT / relative).read_bytes())
        manifest = json.loads((root / observation.MANIFEST).read_text("utf-8"))
        for relative in manifest["predecessor_immutable_sha256"]:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((observation.ROOT / relative).read_bytes())

    @staticmethod
    def _command(subcommand: str, returncode: int) -> dict[str, object]:
        return {
            "command": ["python", "-m", observation.MODULE, subcommand],
            "returncode": returncode,
        }

    def test_identity_workers_and_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "5106e341522bd6cd9d79a7de48800c607eedc455",
        )
        self.assertEqual(
            observation.MODULE,
            "neuron_graph_rag.cross_encoder_precision_v19_performance_observation",
        )
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v19-runtime"
        )
        self.assertEqual(len(observation.lifecycle.WORKERS), 6)
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
            self.assertEqual(observation.serialize_container_path(value), value.as_posix())

    def test_prebuild_freezes_exact_v18_artifact_closure(self) -> None:
        result = observation.validate_prebuild(observation.ROOT)
        self.assertEqual(result["status"], "prebuild_contract_valid")
        self.assertEqual(result["predecessor_artifact_count"], 26)
        self.assertEqual(result["protocol_artifact_count"], 23)
        self.assertEqual(result["corpus_document_count"], 24)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_v18_executed_artifacts_and_raw_evidence_remain_byte_exact(self) -> None:
        manifest = json.loads(
            (observation.ROOT / observation.MANIFEST).read_text("utf-8")
        )
        registry = manifest["predecessor_immutable_sha256"]
        self.assertEqual(len(registry), 26)
        for relative, expected in observation.PREDECESSOR_ANCHOR_SHA256.items():
            self.assertEqual(registry[relative], expected)
        self.assertEqual(
            registry[
                "tests/evidence/github_cross_encoder_precision_v18_observation/"
                "execution-error.json"
            ],
            "259ec86b35fd38d650ca921c649a5256d7e26c6cbd2686e2f906751a375af16f",
        )

    def test_verification_uses_literal_module_identity_and_no_local_full(self) -> None:
        commands = observation._verification_commands(observation.ROOT)
        self_audit = commands[-1]
        self.assertEqual(self_audit[-3:], ["-m", observation.MODULE, "audit"])
        self.assertNotIn("__main__", self_audit)
        self.assertFalse(any("unittest discover" in " ".join(row) for row in commands))

    def test_container_command_is_offline_and_mounts_only_v19_runtime(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v19-development-baseline-primary",
        )
        rendered = "\n".join(command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v19-runtime:/opt/ngr-v19/runtime",
        )
        for forbidden in (*observation.FORBIDDEN_VOLUMES.values(), "/opt/ngr-v8/runtime"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn(observation.MODULE, command)

    def test_source_initialization_preserves_exclusive_model_copy(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e '/opt/ngr-v19/runtime/model-cache'", script)
        self.assertIn("test ! -e '/opt/ngr-v8/runtime'", script)
        self.assertIn("mkdir '/opt/ngr-v19/runtime/source'", script)
        self.assertNotIn("/databases/development", script)
        self.assertNotIn("/runs/development", script)

    def test_stage_contract_exclusive_creates_exact_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = PurePosixPath(Path(directory).as_posix())
            databases = Path(str(root / "databases"))
            runs = Path(str(root / "runs"))
            databases.mkdir()
            runs.mkdir()
            contract = rank_observation_stage_contract.RankObservationStageContract(
                root / "databases", root / "runs"
            )
            value = contract.initialize_container_stage("development")
            self.assertEqual(value["stage_directory_create_count"], 2)
            self.assertTrue((databases / "development").is_dir())
            self.assertTrue((runs / "development").is_dir())
            self.assertEqual(contract.validate_initialization(value, "development"), value)
            with self.assertRaises(FileExistsError):
                contract.initialize_container_stage("development")

    def test_stage_contract_rejects_missing_root_and_unknown_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = PurePosixPath(Path(directory).as_posix())
            contract = rank_observation_stage_contract.RankObservationStageContract(
                root / "missing-databases", root / "missing-runs"
            )
            with self.assertRaises(FileNotFoundError):
                contract.initialize_container_stage("development")
            with self.assertRaises(ValueError):
                contract.initialize_container_stage("other")

    def test_worker_stage_initializes_before_claim_and_six_workers(self) -> None:
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
            if "stage-init" in command:
                database, output = observation.STAGE_CONTRACT.stage_paths("development")
                return json.dumps(
                    {
                        "protocol_boundary": "fresh-stage-directories",
                        "stage": "development",
                        "database_directory": str(database),
                        "output_directory": str(output),
                        "stage_directory_create_count": 2,
                        "exclusive_create": True,
                    }
                )
            if "finalize" in command:
                return '{"all_hard_gates_pass": false, "selected_candidate_id": null}'
            return "{}"

        with (
            patch.object(
                observation.lifecycle.lifecycle.lifecycle,
                "_run_logged",
                side_effect=logged,
            ),
            patch.object(observation.lifecycle.lifecycle, "_export_volume_evidence"),
        ):
            observation._run_stage_host("development", observation.ROOT, rows, claims)
        self.assertIn("stage-init", commands[0])
        self.assertIn("claim", commands[1])
        workers = [command for command in commands if "worker" in command]
        self.assertEqual(len(workers), 6)
        self.assertEqual(claims["development"], 1)
        self.assertTrue(
            all(
                command[command.index("--name") + 1].startswith("ngr-v19-")
                for command in workers
            )
        )

    def test_actual_counts_do_not_expand_failed_worker_to_planned_slots(self) -> None:
        report = {
            "development_claim_count": 1,
            "holdout_claim_count": 0,
            "commands": [
                self._command("stage-init", 0),
                self._command("claim", 0),
                self._command("worker", 1),
                self._command("fail-stage", 0),
            ],
        }
        counts = observation.STAGE_CONTRACT.execution_counts(report)
        self.assertEqual(counts["planned_worker_slot_count"], 6)
        self.assertEqual(counts["actual_worker_launch_count"], 1)
        self.assertEqual(counts["actual_successful_worker_count"], 0)
        self.assertEqual(counts["actual_observed_result_count"], 0)
        self.assertEqual(counts["actual_finalize_count"], 0)
        self.assertEqual(counts["stage_directory_initialization_count"], 1)

    def test_actual_counts_record_complete_development_and_holdout(self) -> None:
        commands = []
        for _stage in ("development", "holdout"):
            commands.append(self._command("stage-init", 0))
            commands.append(self._command("claim", 0))
            commands.extend(self._command("worker", 0) for _ in range(6))
            commands.append(self._command("finalize", 0))
        counts = observation.STAGE_CONTRACT.execution_counts(
            {
                "development_claim_count": 1,
                "holdout_claim_count": 1,
                "commands": commands,
            }
        )
        self.assertEqual(counts["planned_worker_slot_count"], 12)
        self.assertEqual(counts["actual_worker_launch_count"], 12)
        self.assertEqual(counts["actual_successful_worker_count"], 12)
        self.assertEqual(counts["actual_observed_result_count"], 12)
        self.assertEqual(counts["actual_finalize_count"], 2)
        self.assertEqual(counts["stage_directory_initialization_count"], 2)

    def test_terminal_error_count_audit_uses_actual_command_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            (evidence / "execution-error.json").write_text(
                json.dumps(
                    {
                        "protocol_id": observation.PROTOCOL_ID,
                        "development_claim_count": 1,
                        "holdout_claim_count": 0,
                        "retry_count": 0,
                        "shared_database_unchanged": True,
                        "commands": [
                            self._command("stage-init", 0),
                            self._command("claim", 0),
                            self._command("worker", 1),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            counts = observation.TERMINAL_AUDIT.fixate_terminal_evidence(root)
            self.assertEqual(counts["worker_process_count"], 1)
            self.assertEqual(counts["observed_result_count"], 0)
            self.assertEqual(counts["planned_worker_slot_count"], 6)
            self.assertEqual(counts["actual_worker_launch_count"], 1)
            self.assertEqual(counts["actual_observed_result_count"], 0)
            self.assertEqual(counts["actual_finalize_count"], 0)
            observation.TERMINAL_AUDIT._validate_counts(counts)
            invalid = {**counts, "observed_result_count": 6}
            with self.assertRaisesRegex(ValueError, "actual count"):
                observation.TERMINAL_AUDIT._validate_counts(invalid)

    def test_stage_init_dispatch_uses_fresh_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = PurePosixPath(Path(directory).as_posix())
            Path(str(root / "databases")).mkdir()
            Path(str(root / "runs")).mkdir()
            contract = rank_observation_stage_contract.RankObservationStageContract(
                root / "databases", root / "runs"
            )
            with patch.object(observation, "STAGE_CONTRACT", contract):
                value = observation.SPEC.dispatch_container_command(
                    "stage-init", stage="development"
                )
            self.assertEqual(value["stage_directory_create_count"], 2)
            self.assertTrue(value["exclusive_create"])

    def test_source_root_verifier_normalizes_actual_claim_root(self) -> None:
        identity = json.loads(
            (observation.ROOT / observation.SOURCE_IDENTITY).read_text("utf-8")
        )
        protocol = {"root": Path("/opt/ngr-v19/runtime/source")}

        def nested(
            _commit: str, value: dict[str, object], **_kwargs: object
        ) -> dict[str, object]:
            self.assertEqual(value["root"], "/opt/ngr-v19/runtime/frozen-source")
            return {"protocol_artifact_count": 23, "corpus_document_count": 24}

        result = observation.SOURCE_ROOT_SPEC.verify_protocol_commit(
            observation.V8_PROTOCOL_COMMIT,
            protocol,
            identity=identity,
            source=observation.CONTAINER_SOURCE,
            protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
            nested_verifier=nested,
        )
        self.assertTrue(result["source_root_propagation_exact"])

    def test_terminal_audit_scope_binds_v19_at_every_nested_layer(self) -> None:
        modules = (
            observation.lifecycle,
            observation.lifecycle.lifecycle,
            observation.lifecycle.lifecycle.lifecycle,
        )
        originals = tuple(module.PROTOCOL_ID for module in modules)
        with observation.TERMINAL_AUDIT.protocol_identity_scope():
            self.assertTrue(
                all(module.PROTOCOL_ID == observation.PROTOCOL_ID for module in modules)
            )
        self.assertEqual(tuple(module.PROTOCOL_ID for module in modules), originals)

    def test_evidence_absence_is_result_free_and_retry_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_prebuild_contract(root)
            result = observation.audit_evidence(root)
        self.assertEqual(result["status"], "preflight-not-run")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_json_contracts_are_utf8_without_replacement_character(self) -> None:
        for relative in (
            observation.MANIFEST,
            observation.SOURCE_IDENTITY,
            observation.OBSERVATION_AUDIT,
        ):
            raw = (observation.ROOT / relative).read_bytes()
            self.assertIsInstance(json.loads(raw.decode("utf-8")), dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

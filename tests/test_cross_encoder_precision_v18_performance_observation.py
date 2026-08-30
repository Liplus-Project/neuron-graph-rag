from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v18_performance_observation as observation,
)
from neuron_graph_rag import rank_observation_lifecycle, source_root_propagation


class CrossEncoderPrecisionV18PerformanceObservationTest(unittest.TestCase):
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

    def test_identity_workers_and_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "7a4b63d65c5abc84e7550856a965572837b238b0",
        )
        self.assertEqual(
            observation.MODULE,
            "neuron_graph_rag.cross_encoder_precision_v18_performance_observation",
        )
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v18-runtime"
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

    def test_prebuild_freezes_exact_v17_artifact_closure(self) -> None:
        result = observation.validate_prebuild(observation.ROOT)
        self.assertEqual(result["status"], "prebuild_contract_valid")
        self.assertEqual(result["predecessor_artifact_count"], 13)
        self.assertEqual(result["protocol_artifact_count"], 23)
        self.assertEqual(result["corpus_document_count"], 24)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_v17_executed_artifacts_remain_byte_exact(self) -> None:
        manifest = json.loads(
            (observation.ROOT / observation.MANIFEST).read_text("utf-8")
        )
        registry = manifest["predecessor_immutable_sha256"]
        self.assertEqual(len(registry), 13)
        self.assertEqual(
            registry[
                "tests/evidence/github_cross_encoder_precision_v17_observation/"
                "terminal-evidence-manifest.json"
            ],
            "37e4e0b2ddc4896436a76a5632db1ed9cccf0974d3f1c70ce4027fd73b354191",
        )
        self.assertEqual(
            registry["src/neuron_graph_rag/rank_observation_lifecycle.py"],
            "1f33578f988f365b7daab7202d89309955e9192581f02ae43607cfdcac0c3ff6",
        )

    def test_verification_uses_literal_module_identity_and_no_local_full(self) -> None:
        commands = observation._verification_commands(observation.ROOT)
        self_audit = commands[-1]
        self.assertEqual(self_audit[-3:], ["-m", observation.MODULE, "audit"])
        self.assertEqual(self_audit[self_audit.index("-m") + 1], observation.MODULE)
        self.assertNotIn("__main__", self_audit)
        self.assertFalse(any("unittest discover" in " ".join(row) for row in commands))

    def test_container_command_is_offline_and_mounts_only_v18_runtime(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v18-development-baseline-primary",
        )
        rendered = "\n".join(command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v18-runtime:/opt/ngr-v18/runtime",
        )
        for forbidden in (*observation.FORBIDDEN_VOLUMES.values(), "/opt/ngr-v8/runtime"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn(observation.MODULE, command)

    def test_source_initialization_preserves_exclusive_model_copy(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e '/opt/ngr-v18/runtime/model-cache'", script)
        self.assertIn("test ! -e '/opt/ngr-v8/runtime'", script)
        self.assertIn("mkdir '/opt/ngr-v18/runtime/source'", script)
        self.assertNotIn("mkdir -p /opt/ngr-v18/runtime/model-cache", script)

    def test_source_root_verifier_normalizes_actual_claim_root(self) -> None:
        identity = json.loads(
            (observation.ROOT / observation.SOURCE_IDENTITY).read_text("utf-8")
        )
        protocol = {"root": Path("/opt/ngr-v18/runtime/source")}

        def nested(
            _commit: str, value: dict[str, object], **_kwargs: object
        ) -> dict[str, object]:
            self.assertEqual(
                value["root"], "/opt/ngr-v18/runtime/frozen-source"
            )
            return {"protocol_artifact_count": 23, "corpus_document_count": 24}

        result = observation.SOURCE_ROOT_SPEC.verify_protocol_commit(
            observation.V8_PROTOCOL_COMMIT,
            protocol,
            identity=identity,
            source=observation.CONTAINER_SOURCE,
            protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
            nested_verifier=nested,
        )
        self.assertEqual(
            result["resolved_frozen_source_root"],
            "/opt/ngr-v18/runtime/frozen-source",
        )
        self.assertTrue(result["source_root_propagation_exact"])

    def test_configure_binds_v16_common_component_to_actual_graph(self) -> None:
        identity = {
            "identity_schema": "ngr.source-root-propagation/v1",
            "source_archive_commit": observation.FREEZE_COMMIT,
            "configured_claim_source_root": str(observation.CONTAINER_SOURCE),
            "configured_frozen_source_root": str(
                observation.CONTAINER_PROTOCOL_SOURCE
            ),
            "git_free_identity": {},
        }
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(
                rank_observation_lifecycle.RankObservationSpec,
                "source_identity",
                return_value=identity,
            ),
            patch.object(
                source_root_propagation.SourceRootFreezeSpec,
                "bind_verifier",
                return_value={"source_root_propagation_verifier_bound": True},
            ) as binder,
        ):
            observation._configure_container_harness()
        self.assertEqual(binder.call_args.kwargs["volume"], observation.VOLUME)
        self.assertEqual(
            binder.call_args.kwargs["source"], Path("/opt/ngr-v18/runtime/source")
        )
        self.assertEqual(
            binder.call_args.kwargs["protocol_source"],
            Path("/opt/ngr-v18/runtime/frozen-source"),
        )

    def test_terminal_audit_scope_binds_v18_at_every_nested_layer(self) -> None:
        modules = (
            observation.lifecycle,
            observation.lifecycle.lifecycle,
            observation.lifecycle.lifecycle.lifecycle,
        )
        originals = tuple(module.PROTOCOL_ID for module in modules)
        with observation.TERMINAL_AUDIT.protocol_identity_scope() as engine:
            self.assertIs(engine, modules[-1])
            self.assertTrue(
                all(module.PROTOCOL_ID == observation.PROTOCOL_ID for module in modules)
            )
        self.assertEqual(tuple(module.PROTOCOL_ID for module in modules), originals)

    def test_terminal_preflight_error_is_fixed_and_audited_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_prebuild_contract(root)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            raw = {
                "protocol_id": observation.PROTOCOL_ID,
                "commands": [
                    {
                        "command": ["wslc", "volume", "create", observation.VOLUME],
                        "returncode": 0,
                    },
                    {
                        "command": ["python", "model-copy-verify"],
                        "returncode": 0,
                    },
                ],
                "runtime_volume_create_count": 1,
                "preflight_forward_inference_count": 2,
                "development_claim_count": 0,
                "holdout_claim_count": 0,
                "result_count": 0,
                "retry_count": 0,
            }
            terminal = {
                "protocol_id": observation.PROTOCOL_ID,
                "status": "error",
                "shared_database_unchanged": True,
                "retry_count": 0,
                "same_protocol_retry_allowed": False,
                "performance": "not assessed",
            }
            (evidence / "preflight.error.json").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            (evidence / "preflight-terminal.json").write_text(
                json.dumps(terminal), encoding="utf-8"
            )
            counts = observation.TERMINAL_AUDIT.fixate_terminal_evidence(root)
            result = observation.audit_evidence(root)
            self.assertEqual(counts["runtime_volume_create_count"], 1)
            self.assertEqual(counts["model_cache_copy_count"], 1)
            self.assertEqual(counts["model_forward_inference_count"], 2)
            self.assertEqual(result["status"], "preflight-error")
            self.assertEqual(result["development_claim_count"], 0)
            self.assertEqual(result["holdout_claim_count"], 0)
            self.assertEqual(result["retry_count"], 0)
            with self.assertRaises(FileExistsError):
                observation.TERMINAL_AUDIT.fixate_terminal_evidence(root)

    def test_terminal_manifest_rejects_unregistered_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            payload = evidence / "payload.json"
            payload.write_text("{}", encoding="utf-8")
            digest = observation.lifecycle.sha256_file(payload)
            (evidence / "terminal-evidence-manifest.json").write_text(
                json.dumps(
                    {
                        "protocol_id": observation.PROTOCOL_ID,
                        "status": "error",
                        "files_sha256": {"payload.json": digest},
                    }
                ),
                encoding="utf-8",
            )
            observation.TERMINAL_AUDIT.verify_hash_manifest(
                evidence, "terminal-evidence-manifest.json", exact=True
            )
            (evidence / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set"):
                observation.TERMINAL_AUDIT.verify_hash_manifest(
                    evidence, "terminal-evidence-manifest.json", exact=True
                )

    def test_worker_stage_fixes_six_fresh_processes_and_databases(self) -> None:
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
                observation.lifecycle.lifecycle.lifecycle,
                "_run_logged",
                side_effect=logged,
            ),
            patch.object(observation.lifecycle.lifecycle, "_export_volume_evidence"),
        ):
            observation._run_stage_host(
                "development", observation.ROOT, rows, claims
            )
        workers = [command for command in commands if "worker" in command]
        self.assertEqual(len(workers), 6)
        self.assertEqual(claims["development"], 1)
        self.assertTrue(
            all(
                command[command.index("--name") + 1].startswith("ngr-v18-")
                for command in workers
            )
        )
        self.assertTrue(
            all(
                command[command.index("--database") + 1].startswith(
                    "/opt/ngr-v18/runtime/databases/development/"
                )
                for command in workers
            )
        )

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

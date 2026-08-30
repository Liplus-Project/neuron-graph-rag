from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v17_performance_observation as observation,
)
from neuron_graph_rag import rank_observation_lifecycle, source_root_propagation


class CrossEncoderPrecisionV17PerformanceObservationTest(unittest.TestCase):
    def test_identity_workers_and_paths_are_exact(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "041233ab6267e883fdf9d519609bbe615c79645b",
        )
        self.assertEqual(observation.WSLC_VERSION, "2.9.4.0")
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v17-runtime"
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

    def test_prebuild_freezes_exact_v16_artifact_closure(self) -> None:
        result = observation.validate_prebuild(observation.ROOT)
        self.assertEqual(result["status"], "prebuild_contract_valid")
        self.assertEqual(result["predecessor_artifact_count"], 16)
        self.assertEqual(result["protocol_artifact_count"], 23)
        self.assertEqual(result["corpus_document_count"], 24)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_container_command_is_offline_and_mounts_only_v17_runtime(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v17-development-baseline-primary",
        )
        rendered = "\n".join(command)
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v17-runtime:/opt/ngr-v17/runtime",
        )
        for forbidden in (*observation.FORBIDDEN_VOLUMES.values(), "/opt/ngr-v8/runtime"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn(
            "neuron_graph_rag.cross_encoder_precision_v17_performance_observation",
            command,
        )

    def test_source_initialization_preserves_exclusive_model_copy(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e '/opt/ngr-v17/runtime/model-cache'", script)
        self.assertIn("test ! -e '/opt/ngr-v8/runtime'", script)
        self.assertIn("mkdir '/opt/ngr-v17/runtime/source'", script)
        self.assertNotIn("mkdir -p /opt/ngr-v17/runtime/model-cache", script)
        for path in ("databases", "runs", "archive", "transport"):
            self.assertIn(f"/opt/ngr-v17/runtime/{path}", script)
        self.assertNotIn("knowledge.db", script)

    def test_source_root_verifier_normalizes_actual_claim_root(self) -> None:
        identity = json.loads(
            (observation.ROOT / observation.SOURCE_IDENTITY).read_text(
                encoding="utf-8"
            )
        )
        protocol = {"root": Path("/opt/ngr-v17/runtime/source")}

        def nested(
            commit: str,
            value: dict[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            self.assertEqual(commit, observation.V8_PROTOCOL_COMMIT)
            self.assertEqual(
                value["root"], "/opt/ngr-v17/runtime/frozen-source"
            )
            return {
                "protocol_artifact_count": 23,
                "corpus_document_count": 24,
            }

        result = observation.SOURCE_ROOT_SPEC.verify_protocol_commit(
            observation.V8_PROTOCOL_COMMIT,
            protocol,
            identity=identity,
            source=observation.CONTAINER_SOURCE,
            protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
            nested_verifier=nested,
        )
        self.assertEqual(
            result["observed_claim_source_root"],
            "/opt/ngr-v17/runtime/source",
        )
        self.assertEqual(
            result["resolved_frozen_source_root"],
            "/opt/ngr-v17/runtime/frozen-source",
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
        self.assertIs(
            binder.call_args.args[0], observation.lifecycle.lifecycle.predecessor
        )
        self.assertEqual(binder.call_args.kwargs["volume"], observation.VOLUME)
        self.assertEqual(
            binder.call_args.kwargs["source"], Path("/opt/ngr-v17/runtime/source")
        )
        self.assertEqual(
            binder.call_args.kwargs["protocol_source"],
            Path("/opt/ngr-v17/runtime/frozen-source"),
        )
        self.assertIs(binder.call_args.kwargs["identity"], identity)

    def test_v17_scope_is_thin_and_restores_frozen_v14_defaults(self) -> None:
        original = {
            "protocol": observation.lifecycle.PROTOCOL_ID,
            "root": observation.lifecycle.CONTAINER_ROOT,
            "configure": observation.lifecycle._configure_container_harness,
        }
        with observation._v17_scope():
            self.assertEqual(
                observation.lifecycle.PROTOCOL_ID, observation.PROTOCOL_ID
            )
            self.assertEqual(
                observation.lifecycle.CONTAINER_ROOT, observation.CONTAINER_ROOT
            )
            self.assertEqual(
                observation.lifecycle._configure_container_harness,
                observation._configure_container_harness,
            )
        self.assertEqual(observation.lifecycle.PROTOCOL_ID, original["protocol"])
        self.assertEqual(observation.lifecycle.CONTAINER_ROOT, original["root"])
        self.assertIs(
            observation.lifecycle._configure_container_harness,
            original["configure"],
        )

    def test_platform_report_names_every_forbidden_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "platform-report.json"
            observation._write_lifecycle_json_exclusive(
                path,
                {
                    "path_freeze_volume": (
                        observation.FORBIDDEN_VOLUMES[
                            "v16_source_root_propagation_freeze_volume"
                        ]
                    ),
                    "path_freeze_volume_mounted": False,
                    "path_freeze_volume_read": False,
                },
            )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("path_freeze_volume", report)
        for field in observation.FORBIDDEN_VOLUMES:
            self.assertFalse(report[f"{field}_mounted"])
            self.assertFalse(report[f"{field}_read"])
            self.assertFalse(report[f"{field}_reused"])
        self.assertFalse(
            report["predecessor_terminal_evidence_semantic_content_opened"]
        )
        self.assertFalse(report["predecessor_packet_reused"])
        self.assertEqual(
            observation._canonical_lifecycle_value(report),
            observation.lifecycle.canonical_sha256(
                {
                    "path_freeze_volume": (
                        observation.FORBIDDEN_VOLUMES[
                            "v16_source_root_propagation_freeze_volume"
                        ]
                    ),
                    "path_freeze_volume_mounted": False,
                    "path_freeze_volume_read": False,
                }
            ),
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
        names = {command[command.index("--name") + 1] for command in workers}
        databases = {
            command[command.index("--database") + 1] for command in workers
        }
        self.assertEqual(len(names), 6)
        self.assertEqual(len(databases), 6)
        self.assertTrue(all(name.startswith("ngr-v17-") for name in names))
        self.assertTrue(
            all(
                path.startswith("/opt/ngr-v17/runtime/databases/development/")
                for path in databases
            )
        )

    def test_preflight_verification_is_staged_without_local_full_suite(self) -> None:
        commands = observation._verification_commands(observation.ROOT)
        rendered = [" ".join(command) for command in commands]
        self.assertTrue(any("test_cross_encoder_precision_v17" in row for row in rendered))
        self.assertTrue(any("cross_encoder_precision_v16_observation audit" in row for row in rendered))
        self.assertFalse(any("unittest discover" in row for row in rendered))

    def test_evidence_absence_is_result_free_and_retry_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            with patch.object(
                observation.lifecycle.lifecycle.predecessor,
                "_stored_freeze_contract",
                return_value={
                    "accepted_image": {
                        "build": "build_a",
                        "id": observation.IMAGE_ID,
                        "tag": observation.IMAGE,
                    },
                    "runtime_content_sha256": "a" * 64,
                    "attestation_sha256": "b" * 64,
                    "fingerprint_sha256": "c" * 64,
                    "metadata_correspondence_sha256": "d" * 64,
                    "expected_distribution_count": 29,
                },
            ):
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

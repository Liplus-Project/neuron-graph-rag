from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path, PurePosixPath

from neuron_graph_rag import cross_encoder_precision_v11_observation as observation


class CrossEncoderPrecisionV11RootFreezeTest(unittest.TestCase):
    def test_container_paths_are_strict_posix_and_isolated(self) -> None:
        self.assertEqual(
            observation.serialize_container_path(observation.CONTAINER_ROOT),
            "/opt/ngr-v11/root-freeze",
        )
        self.assertEqual(
            observation.named_volume_spec(
                observation.ROOT_FREEZE_VOLUME, observation.CONTAINER_ROOT
            ),
            "github-cross-encoder-precision-v11-root-freeze:/opt/ngr-v11/root-freeze",
        )
        for value in (
            Path("/opt/ngr-v11/root-freeze"),
            "",
            ".",
            "opt/ngr-v11/root-freeze",
            "\\opt\\ngr-v11\\root-freeze",
            "C:\\opt\\ngr-v11\\root-freeze",
            "/opt/../ngr-v11/root-freeze",
        ):
            with (
                self.subTest(value=repr(value)),
                self.assertRaises((TypeError, ValueError)),
            ):
                observation.serialize_container_path(value)  # type: ignore[arg-type]

    def test_prebuild_freezes_complete_v10_closure_and_zero_counts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 23)
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertFalse(report["old_frozen_source_allowed"])
        for key in (
            "registered_query_execution_count",
            "model_cache_copy_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "development_claim_count",
            "holdout_claim_count",
            "observed_result_count",
            "shared_database_open_count",
        ):
            self.assertEqual(report[key], 0)
        self.assertEqual(report["performance"], "not assessed")

    def test_v10_failure_is_negative_fixture_and_v11_rebinds_base(self) -> None:
        terminal = json.loads(
            (observation.ROOT / observation.V10_TERMINAL_PATH).read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("/opt/ngr-v8/runtime/frozen-source", terminal["failure_cause"])
        old_hashes = observation._verify_predecessor_hashes(
            observation.ROOT, observation._manifest(observation.ROOT)
        )
        self.assertEqual(
            old_hashes[observation.V10_RAW_FAILURE_PATH.as_posix()],
            observation.V10_RAW_FAILURE_SHA256,
        )
        self.assertEqual(
            old_hashes[observation.V10_TERMINAL_PATH.as_posix()],
            observation.V10_TERMINAL_SHA256,
        )

    def test_binder_updates_wrapper_and_distinct_base_explicitly(self) -> None:
        evaluation_base = types.SimpleNamespace()
        evaluation = types.SimpleNamespace(_BASE=evaluation_base)
        base = types.SimpleNamespace()
        wrapper = types.SimpleNamespace(_BASE=base, evaluation=evaluation)
        calls: list[dict[str, str]] = []

        def bind_base() -> None:
            calls.append(
                {
                    "root": str(base.CONTAINER_ROOT),
                    "source": str(base.CONTAINER_SOURCE),
                    "cache": str(base.CONTAINER_CACHE),
                }
            )

        base._bind_container_harness = bind_base
        report = observation.bind_frozen_harness_root(
            wrapper,
            volume=observation.ROOT_FREEZE_VOLUME,
            root=PurePosixPath("/opt/ngr-v11/root-freeze"),
            source=PurePosixPath("/opt/ngr-v11/root-freeze/source"),
            cache=PurePosixPath("/opt/ngr-v11/root-freeze/model-cache"),
            protocol_source=PurePosixPath("/opt/ngr-v11/root-freeze/frozen-source"),
            evidence=observation.EVIDENCE,
        )
        self.assertTrue(report["wrapper_base_distinct"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["wrapper_binding"], report["base_binding"])
        self.assertEqual(
            report["base_binding"]["CONTAINER_ROOT"],
            "/opt/ngr-v11/root-freeze",
        )
        self.assertEqual(
            report["base_binding"]["CONTAINER_PROTOCOL_SOURCE"],
            "/opt/ngr-v11/root-freeze/frozen-source",
        )
        self.assertEqual(
            report["evaluation_base_root"],
            "/opt/ngr-v11/root-freeze/frozen-source",
        )

    def test_binder_rejects_old_or_out_of_root_paths(self) -> None:
        base = types.SimpleNamespace(_bind_container_harness=lambda: None)
        evaluation = types.SimpleNamespace(_BASE=types.SimpleNamespace())
        wrapper = types.SimpleNamespace(_BASE=base, evaluation=evaluation)
        with self.assertRaisesRegex(ValueError, "old v8 runtime root"):
            observation.bind_frozen_harness_root(
                wrapper,
                volume="test",
                root=PurePosixPath("/opt/ngr-v8/runtime"),
                source=PurePosixPath("/opt/ngr-v8/runtime/source"),
                cache=PurePosixPath("/opt/ngr-v8/runtime/model-cache"),
                protocol_source=PurePosixPath("/opt/ngr-v8/runtime/frozen-source"),
                evidence=Path("evidence"),
            )
        with self.assertRaisesRegex(ValueError, "below the parameterized root"):
            observation.bind_frozen_harness_root(
                wrapper,
                volume="test",
                root=PurePosixPath("/opt/ngr-v11/root-freeze"),
                source=PurePosixPath("/opt/ngr-v11/source"),
                cache=PurePosixPath("/opt/ngr-v11/root-freeze/model-cache"),
                protocol_source=PurePosixPath("/opt/ngr-v11/root-freeze/frozen-source"),
                evidence=Path("evidence"),
            )

    def test_verifier_reads_exact_protocol_and_corpus_from_new_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root-freeze"
            source = root / "source"
            protocol_source = root / "frozen-source"
            cache = root / "model-cache"
            output = root / "report.json"
            source.mkdir(parents=True)
            protocol_source.mkdir()
            artifact_registry: dict[str, str] = {}
            for index in range(23):
                relative = f"fixtures/artifact-{index:02d}.json"
                payload = f"artifact-{index}\n".encode()
                path = protocol_source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                artifact_registry[relative] = hashlib.sha256(payload).hexdigest()
            documents: list[dict[str, str]] = []
            for index in range(24):
                relative = f"docs/document-{index:02d}.md"
                payload = f"document-{index}\n".encode()
                path = protocol_source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                documents.append(
                    {
                        "path": relative,
                        "content_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            evaluation_base = types.SimpleNamespace()

            def load_protocol(_: Path) -> dict[str, object]:
                return {
                    "manifest": {"artifact_sha256": artifact_registry},
                    "corpus": {"commit": "c" * 40, "documents": documents},
                }

            evaluation = types.SimpleNamespace(
                _BASE=evaluation_base, load_protocol=load_protocol
            )
            base = types.SimpleNamespace()
            wrapper = types.SimpleNamespace(_BASE=base, evaluation=evaluation)

            def bind_base() -> None:
                def direct_git_bytes(_: Path, __: str, relative: str) -> bytes:
                    return (
                        base.CONTAINER_ROOT / "frozen-source" / relative
                    ).read_bytes()

                evaluation_base._git_bytes = direct_git_bytes

            base._bind_container_harness = bind_base
            report = observation.root_binding_verify(
                root,
                source,
                cache,
                protocol_source,
                output,
                wrapper=wrapper,
            )
            self.assertEqual(report["protocol_artifact_count"], 23)
            self.assertEqual(report["corpus_document_count"], 24)
            self.assertTrue(report["exact_protocol_bytes_verified"])
            self.assertTrue(report["exact_corpus_bytes_verified"])
            self.assertFalse(cache.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_commands_are_offline_and_never_use_v10_volumes_or_old_root(self) -> None:
        command = observation.root_binding_command()
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--volume"
        ]
        self.assertEqual(
            mounts,
            ["github-cross-encoder-precision-v11-root-freeze:/opt/ngr-v11/root-freeze"],
        )
        text = "\n".join(command)
        self.assertNotIn(observation.V10_RUNTIME_VOLUME, text)
        self.assertNotIn(observation.V10_CACHE_FREEZE_VOLUME, text)
        self.assertNotIn(str(observation.OLD_FROZEN_SOURCE), text)
        for forbidden in (
            "model-copy",
            "model-probe",
            "registered query",
            "worker",
            "sqlite",
        ):
            self.assertNotIn(forbidden, text.lower())
        for script in (
            observation._protocol_source_import_script(),
            observation._harness_source_import_script(),
        ):
            self.assertIn("test ! -e '/opt/ngr-v8/runtime/frozen-source'", script)
            self.assertNotIn("mkdir '/opt/ngr-v8/runtime/frozen-source'", script)
            self.assertNotIn("tar -xf - -C '/opt/ngr-v8/runtime/frozen-source'", script)

    def test_count_audit_is_exactly_once_and_result_free(self) -> None:
        rows = [
            {
                "command": [
                    "wslc",
                    "volume",
                    "create",
                    observation.ROOT_FREEZE_VOLUME,
                ]
            },
            {"command": observation.root_binding_command()},
        ]
        counts = observation._count_audit(
            status="pass",
            rows=rows,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        self.assertEqual(counts["root_freeze_volume_create_count"], 1)
        self.assertEqual(counts["root_binding_verifier_run_count"], 1)
        self.assertFalse(counts["v10_runtime_volume_mounted"])
        self.assertFalse(counts["v10_cache_freeze_volume_mounted"])
        for key in (
            "retry_count",
            "model_cache_copy_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "registered_query_execution_count",
            "development_claim_count",
            "holdout_claim_count",
            "observed_result_count",
            "shared_database_open_count",
        ):
            self.assertEqual(counts[key], 0)

    def test_audit_is_pending_or_valid_terminal_evidence(self) -> None:
        report = observation.audit_evidence()
        self.assertIn(
            report["status"], {"prebuild_ready_evidence_absent", "pass", "error"}
        )
        self.assertEqual(report["predecessor_artifact_count"], 23)
        self.assertEqual(report["registered_query_execution_count"], 0)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_json_contracts_are_utf8_without_replacement_character(self) -> None:
        for relative in (observation.MANIFEST, observation.RESULT_FREE_AUDIT):
            raw = (observation.ROOT / relative).read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertIsInstance(value, dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

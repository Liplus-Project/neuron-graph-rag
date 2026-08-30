from __future__ import annotations

import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import PurePosixPath
from unittest import mock

from neuron_graph_rag import cross_encoder_precision_v16_observation as observation
from neuron_graph_rag import source_root_propagation


class CrossEncoderPrecisionV16ObservationTest(unittest.TestCase):
    def _local_identity(self) -> dict[str, object]:
        identity = deepcopy(observation._source_identity(observation.ROOT))
        identity["configured_claim_source_root"] = "/opt/test/source"
        identity["configured_frozen_source_root"] = "/opt/test/frozen-source"
        return identity

    def _local_protocol(self, root: object = PurePosixPath("/opt/test/source")) -> dict:
        protocol = dict(observation.frozen_v8.evaluation.load_protocol(observation.ROOT))
        protocol["root"] = root
        return protocol

    def test_prebuild_contract_freezes_v15_terminal_artifacts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 14)
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertEqual(report["source_root_propagation_verifier_run_limit"], 1)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_exact_path_and_string_roots_map_to_frozen_source(self) -> None:
        for root in (
            PurePosixPath("/opt/test/source"),
            "/opt/test/source",
        ):
            with self.subTest(root=root):
                resolved = source_root_propagation.resolve_exact_source_root(
                    root,
                    configured_source=PurePosixPath("/opt/test/source"),
                    configured_frozen_source=PurePosixPath(
                        "/opt/test/frozen-source"
                    ),
                )
                self.assertEqual(resolved, PurePosixPath("/opt/test/frozen-source"))

    def test_wrong_relative_old_sibling_child_and_escape_roots_fail_closed(self) -> None:
        for root in (
            "source",
            "/opt/ngr-v8/runtime/frozen-source",
            "/opt/test/sibling",
            "/opt/test/source/child",
            "/opt/test/source/../frozen-source",
        ):
            with self.subTest(root=root), self.assertRaises(ValueError):
                source_root_propagation.resolve_exact_source_root(
                    root,
                    configured_source=PurePosixPath("/opt/test/source"),
                    configured_frozen_source=PurePosixPath(
                        "/opt/test/frozen-source"
                    ),
                )
        with self.assertRaises(TypeError):
            source_root_propagation.resolve_exact_source_root(
                16,
                configured_source=PurePosixPath("/opt/test/source"),
                configured_frozen_source=PurePosixPath("/opt/test/frozen-source"),
            )

    def test_normalize_protocol_root_preserves_payload_and_emits_frozen_string(self) -> None:
        protocol = {"root": PurePosixPath("/opt/test/source"), "marker": object()}
        normalized, observed, resolved = source_root_propagation.normalize_protocol_root(
            protocol,
            configured_source=PurePosixPath("/opt/test/source"),
            configured_frozen_source=PurePosixPath("/opt/test/frozen-source"),
        )
        self.assertEqual(observed, "/opt/test/source")
        self.assertEqual(resolved, PurePosixPath("/opt/test/frozen-source"))
        self.assertEqual(normalized["root"], "/opt/test/frozen-source")
        self.assertIs(normalized["marker"], protocol["marker"])
        self.assertIsInstance(protocol["root"], PurePosixPath)

    def test_resolver_accepts_actual_path_transport_and_delegates_frozen_root(self) -> None:
        delegated = {
            "protocol_artifact_count": 23,
            "corpus_document_count": 24,
            "exact_protocol_bytes_verified": True,
            "exact_corpus_bytes_verified": True,
            "container_git_executable_invocation_count": 0,
            "container_subprocess_invocation_count": 0,
        }
        with mock.patch.object(
            observation.lifecycle.git_free,
            "git_free_verify_protocol_commit",
            return_value=delegated,
        ) as verifier:
            report = observation.resolver_aware_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                self._local_protocol(),
                identity=self._local_identity(),
                source=PurePosixPath("/opt/test/source"),
                protocol_source=PurePosixPath("/opt/test/frozen-source"),
            )
        self.assertEqual(report["observed_claim_source_root_type"], "PurePosixPath")
        self.assertTrue(report["source_root_propagation_exact"])
        self.assertTrue(report["exact_protocol_bytes_verified"])
        normalized = verifier.call_args.args[1]
        self.assertEqual(normalized["root"], "/opt/test/frozen-source")

    def test_v8_loader_transports_root_as_a_path(self) -> None:
        root = observation.ROOT
        protocol = observation.frozen_v8.evaluation.load_protocol(root)
        self.assertIs(protocol["root"], root)
        self.assertEqual(
            source_root_propagation.protocol_root_text(root), root.as_posix()
        )

    def test_actual_path_transport_reaches_exact_byte_verifier(self) -> None:
        protocol = self._local_protocol()

        def verify_local_bytes(
            protocol_commit: str,
            normalized: dict,
            **kwargs: object,
        ) -> dict:
            self.assertEqual(normalized["root"], "/opt/test/frozen-source")
            local = dict(normalized)
            local["root"] = observation.ROOT
            identity = kwargs["identity"]
            assert isinstance(identity, dict)
            return observation.lifecycle.git_free.git_free_verify_protocol_commit(
                protocol_commit,
                local,
                identity=identity,
                protocol_source=observation.ROOT,
            )

        with mock.patch.object(
            subprocess, "run", side_effect=AssertionError("subprocess must not be used")
        ):
            report = observation.SPEC.verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=self._local_identity(),
                source=PurePosixPath("/opt/test/source"),
                protocol_source=PurePosixPath("/opt/test/frozen-source"),
                nested_verifier=verify_local_bytes,
            )
        self.assertTrue(report["source_root_propagation_exact"])
        self.assertTrue(report["exact_protocol_bytes_verified"])
        self.assertTrue(report["exact_corpus_bytes_verified"])
        self.assertEqual(report["observed_claim_source_root_type"], "PurePosixPath")

    def test_nested_git_free_verifier_checks_bytes_without_subprocess(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)["git_free_identity"]
        assert isinstance(identity, dict)
        with mock.patch.object(
            subprocess, "run", side_effect=AssertionError("subprocess must not be used")
        ):
            report = observation.lifecycle.git_free.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )
        self.assertTrue(report["exact_protocol_bytes_verified"])
        self.assertTrue(report["exact_corpus_bytes_verified"])

    def test_actual_six_surface_graph_receives_one_common_verifier(self) -> None:
        wrapper = observation.frozen_v8
        modules = tuple(source_root_propagation.verifier_surfaces(wrapper).values())
        snapshots = [(module, vars(module).copy()) for module in modules]
        try:
            report = observation.bind_source_root_propagation_verifier(
                wrapper,
                volume=observation.SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME,
                root=observation.CONTAINER_ROOT,
                source=observation.CONTAINER_SOURCE,
                cache=observation.CONTAINER_CACHE,
                protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
                evidence=observation.EVIDENCE,
                identity=observation._source_identity(observation.ROOT),
            )
            verifier = wrapper.verify_protocol_commit
            self.assertTrue(report["wrapper_base_distinct"])
            self.assertTrue(report["git_free_verifier_bound"])
            self.assertTrue(report["source_root_propagation_verifier_bound"])
            self.assertEqual(len(report["verifier_binding_surfaces"]), 6)
            for module in modules:
                self.assertIs(module.verify_protocol_commit, verifier)
        finally:
            for module, snapshot in snapshots:
                current = vars(module)
                for name in set(current) - set(snapshot):
                    del current[name]
                current.update(snapshot)

    def test_command_is_offline_and_uses_only_v16_volume(self) -> None:
        command = observation.source_root_propagation_command()
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--volume"
        ]
        self.assertEqual(
            mounts,
            [
                "github-cross-encoder-precision-v16-source-root-propagation-freeze:/opt/ngr-v16/source-root-propagation-freeze"
            ],
        )
        text = "\n".join(command)
        for forbidden in (
            observation.V10_RUNTIME_VOLUME,
            observation.V10_CACHE_FREEZE_VOLUME,
            observation.V11_ROOT_FREEZE_VOLUME,
            observation.V12_RUNTIME_VOLUME,
            observation.V13_COMMIT_FREEZE_VOLUME,
            observation.V14_RUNTIME_VOLUME,
            observation.V15_ROOT_NORMALIZATION_FREEZE_VOLUME,
            str(observation.OLD_FROZEN_SOURCE),
        ):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("subprocess", text)
        self.assertNotIn("--entrypoint\ngit", text)

    def test_count_audit_is_exactly_once_and_result_free(self) -> None:
        rows = [
            {
                "command": [
                    "wslc",
                    "volume",
                    "create",
                    observation.SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME,
                ]
            },
            {"command": observation.source_root_propagation_command()},
        ]
        counts = observation._count_audit(
            status="pass",
            rows=rows,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        self.assertEqual(
            counts["source_root_propagation_freeze_volume_create_count"], 1
        )
        self.assertEqual(counts["source_root_propagation_verifier_run_count"], 1)
        for key in (
            "retry_count",
            "model_cache_copy_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "registered_query_execution_count",
            "development_claim_count",
            "holdout_claim_count",
            "worker_process_count",
            "observed_result_count",
            "shared_database_open_count",
            "container_git_executable_invocation_count",
            "container_subprocess_invocation_count",
        ):
            self.assertEqual(counts[key], 0)
        for key in (
            "v10_runtime_volume_mounted",
            "v10_cache_freeze_volume_mounted",
            "v11_root_freeze_volume_mounted",
            "v12_runtime_volume_mounted",
            "v13_commit_freeze_volume_mounted",
            "v14_runtime_volume_mounted",
            "v15_root_normalization_freeze_volume_mounted",
        ):
            self.assertFalse(counts[key])

    def test_thin_lifecycle_composition_restores_v15_module(self) -> None:
        original = {
            "protocol": observation.lifecycle.PROTOCOL_ID,
            "root": observation.lifecycle.CONTAINER_ROOT,
            "writer": observation.lifecycle._write_evidence,
        }
        with mock.patch.object(
            observation.lifecycle,
            "run_root_normalization_freeze",
            return_value={"claim_source_root_verification_sha256": "a" * 64},
        ) as runner:
            result = observation.run_source_root_propagation_freeze()
        runner.assert_called_once_with(observation.ROOT)
        self.assertEqual(
            result["source_root_propagation_verification_sha256"], "a" * 64
        )
        self.assertNotIn("claim_source_root_verification_sha256", result)
        self.assertEqual(observation.lifecycle.PROTOCOL_ID, original["protocol"])
        self.assertEqual(observation.lifecycle.CONTAINER_ROOT, original["root"])
        self.assertIs(observation.lifecycle._write_evidence, original["writer"])

    def test_audit_is_pending_or_valid_terminal_evidence(self) -> None:
        report = observation.audit_evidence()
        self.assertIn(
            report["status"], {"prebuild_ready_evidence_absent", "pass", "error"}
        )
        self.assertEqual(report["predecessor_artifact_count"], 14)
        self.assertEqual(report["registered_query_execution_count"], 0)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_json_contracts_are_utf8_without_replacement_character(self) -> None:
        for relative in (
            observation.MANIFEST,
            observation.SOURCE_IDENTITY,
            observation.RESULT_FREE_AUDIT,
        ):
            raw = (observation.ROOT / relative).read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertIsInstance(value, dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

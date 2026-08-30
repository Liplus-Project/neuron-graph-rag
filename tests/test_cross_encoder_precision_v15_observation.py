from __future__ import annotations

import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import PurePosixPath
from unittest import mock

from neuron_graph_rag import cross_encoder_precision_v15_observation as observation


class CrossEncoderPrecisionV15ObservationTest(unittest.TestCase):
    def _local_identity(self) -> dict[str, object]:
        identity = deepcopy(observation._source_identity(observation.ROOT))
        identity["configured_claim_source_root"] = "/opt/test/source"
        identity["configured_frozen_source_root"] = "/opt/test/frozen-source"
        return identity

    def _local_protocol(self) -> dict[str, object]:
        protocol = dict(observation.frozen_v8.evaluation.load_protocol(observation.ROOT))
        protocol["root"] = "/opt/test/source"
        return protocol

    def test_prebuild_contract_freezes_v14_terminal_artifacts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 15)
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertEqual(report["root_normalization_verifier_run_limit"], 1)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_exact_claim_source_root_maps_to_frozen_source(self) -> None:
        resolved = observation.resolve_claim_source_root(
            "/opt/ngr-v15/root-normalization-freeze/source",
            configured_source=observation.CONTAINER_SOURCE,
            configured_protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
        )
        self.assertEqual(resolved, observation.CONTAINER_PROTOCOL_SOURCE)

    def test_wrong_relative_old_and_escape_roots_fail_closed(self) -> None:
        for root in (
            "source",
            "/opt/ngr-v8/runtime/frozen-source",
            "/opt/ngr-v15/root-normalization-freeze/sibling",
            "/opt/ngr-v15/root-normalization-freeze/source/../frozen-source",
            "/opt/ngr-v15/root-normalization-freeze/source/child",
        ):
            with self.subTest(root=root), self.assertRaises(ValueError):
                observation.resolve_claim_source_root(
                    root,
                    configured_source=observation.CONTAINER_SOURCE,
                    configured_protocol_source=observation.CONTAINER_PROTOCOL_SOURCE,
                )

    def test_resolver_aware_verifier_delegates_normalized_frozen_root(self) -> None:
        delegated = {
            "protocol_artifact_count": 23,
            "corpus_document_count": 24,
            "exact_protocol_bytes_verified": True,
            "exact_corpus_bytes_verified": True,
            "container_git_executable_invocation_count": 0,
            "container_subprocess_invocation_count": 0,
        }
        with mock.patch.object(
            observation.git_free,
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
        self.assertTrue(report["root_normalization_exact"])
        self.assertTrue(report["exact_protocol_bytes_verified"])
        self.assertTrue(report["exact_corpus_bytes_verified"])
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertEqual(report["container_git_executable_invocation_count"], 0)
        self.assertEqual(report["container_subprocess_invocation_count"], 0)
        normalized = verifier.call_args.args[1]
        self.assertEqual(normalized["root"], "/opt/test/frozen-source")

    def test_nested_git_free_verifier_rejects_wrong_commit(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)["git_free_identity"]
        assert isinstance(identity, dict)
        with self.assertRaisesRegex(ValueError, "commit identity mismatch"):
            observation.git_free.git_free_verify_protocol_commit(
                "0" * 40,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )

    def test_nested_git_free_verifier_checks_bytes_without_subprocess(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)["git_free_identity"]
        assert isinstance(identity, dict)
        with mock.patch.object(
            subprocess, "run", side_effect=AssertionError("subprocess must not be used")
        ):
            report = observation.git_free.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )
        self.assertTrue(report["exact_protocol_bytes_verified"])
        self.assertTrue(report["exact_corpus_bytes_verified"])

    def test_nested_git_free_verifier_rejects_tampered_identity(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = deepcopy(
            observation._source_identity(observation.ROOT)["git_free_identity"]
        )
        assert isinstance(identity, dict)
        identity["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest byte mismatch"):
            observation.git_free.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )

    def test_actual_six_surface_graph_receives_one_resolver_verifier(self) -> None:
        wrapper = observation.frozen_v8
        base = wrapper._BASE
        evaluation = wrapper.evaluation
        modules = (
            wrapper,
            base,
            evaluation,
            evaluation._BASE,
            base._v4,
            base._v4._BASE,
        )
        snapshots = [(module, vars(module).copy()) for module in modules]
        try:
            report = observation.bind_claim_source_root_verifier(
                wrapper,
                volume=observation.ROOT_NORMALIZATION_FREEZE_VOLUME,
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
            self.assertTrue(report["claim_source_root_resolver_bound"])
            self.assertEqual(len(report["verifier_binding_surfaces"]), 6)
            for module in modules:
                self.assertIs(module.verify_protocol_commit, verifier)
        finally:
            for module, snapshot in snapshots:
                current = vars(module)
                for name in set(current) - set(snapshot):
                    del current[name]
                current.update(snapshot)

    def test_command_is_offline_and_uses_only_v15_volume(self) -> None:
        command = observation.root_normalization_command()
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
                "github-cross-encoder-precision-v15-root-normalization-freeze:/opt/ngr-v15/root-normalization-freeze"
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
                    observation.ROOT_NORMALIZATION_FREEZE_VOLUME,
                ]
            },
            {"command": observation.root_normalization_command()},
        ]
        counts = observation._count_audit(
            status="pass",
            rows=rows,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        self.assertEqual(counts["root_normalization_freeze_volume_create_count"], 1)
        self.assertEqual(counts["root_normalization_verifier_run_count"], 1)
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
        ):
            self.assertFalse(counts[key])

    def test_v15_scope_restores_predecessor_module(self) -> None:
        original = {
            "protocol": observation.predecessor.PROTOCOL_ID,
            "root": observation.predecessor.CONTAINER_ROOT,
            "writer": observation.predecessor._write_evidence,
        }
        with observation._v15_scope():
            self.assertEqual(observation.predecessor.PROTOCOL_ID, observation.PROTOCOL_ID)
            self.assertEqual(
                observation.predecessor.CONTAINER_ROOT, observation.CONTAINER_ROOT
            )
            self.assertIs(
                observation.predecessor._write_evidence, observation._write_evidence
            )
        self.assertEqual(observation.predecessor.PROTOCOL_ID, original["protocol"])
        self.assertEqual(observation.predecessor.CONTAINER_ROOT, original["root"])
        self.assertIs(observation.predecessor._write_evidence, original["writer"])

    def test_audit_is_pending_or_valid_terminal_evidence(self) -> None:
        report = observation.audit_evidence()
        self.assertIn(
            report["status"], {"prebuild_ready_evidence_absent", "pass", "error"}
        )
        self.assertEqual(report["predecessor_artifact_count"], 15)
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

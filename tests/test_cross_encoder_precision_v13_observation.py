from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from neuron_graph_rag import cross_encoder_precision_v13_observation as observation


class CrossEncoderPrecisionV13ObservationTest(unittest.TestCase):
    def test_prebuild_contract_freezes_v12_terminal_artifacts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 15)
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertEqual(report["commit_identity_verifier_run_limit"], 1)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_git_free_verifier_accepts_exact_frozen_identity(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)
        with mock.patch.object(
            subprocess,
            "run",
            side_effect=AssertionError("subprocess must not be used"),
        ):
            report = observation.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )
        self.assertEqual(report["protocol_artifact_count"], 23)
        self.assertEqual(report["corpus_document_count"], 24)
        self.assertTrue(report["exact_protocol_bytes_verified"])
        self.assertTrue(report["exact_corpus_bytes_verified"])
        self.assertTrue(report["source_identity_complete"])
        self.assertEqual(report["container_git_executable_invocation_count"], 0)
        self.assertEqual(report["container_subprocess_invocation_count"], 0)

    def test_git_free_verifier_rejects_wrong_commit(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)
        with self.assertRaisesRegex(ValueError, "commit identity mismatch"):
            observation.git_free_verify_protocol_commit(
                "0" * 40,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )

    def test_git_free_verifier_rejects_incomplete_identity(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)
        identity.pop("manifest_sha256")
        with self.assertRaisesRegex(ValueError, "identity is incomplete"):
            observation.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )

    def test_actual_frozen_object_graph_receives_one_git_free_verifier(self) -> None:
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
            report = observation.bind_git_free_commit_verifier(
                wrapper,
                volume=observation.COMMIT_FREEZE_VOLUME,
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
            self.assertEqual(len(report["verifier_binding_surfaces"]), 6)
            for module in modules:
                self.assertIs(module.verify_protocol_commit, verifier)
            self.assertEqual(
                base.CONTAINER_PROTOCOL_SOURCE,
                observation.CONTAINER_PROTOCOL_SOURCE,
            )
            self.assertEqual(
                evaluation._BASE.ROOT,
                observation.CONTAINER_PROTOCOL_SOURCE,
            )
        finally:
            for module, snapshot in snapshots:
                current = vars(module)
                for name in set(current) - set(snapshot):
                    del current[name]
                current.update(snapshot)

    def test_source_identity_mismatch_fails_closed(self) -> None:
        protocol = observation.frozen_v8.evaluation.load_protocol(observation.ROOT)
        identity = observation._source_identity(observation.ROOT)
        identity["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest byte mismatch"):
            observation.git_free_verify_protocol_commit(
                observation.FROZEN_PROTOCOL_COMMIT,
                protocol,
                identity=identity,
                protocol_source=observation.ROOT,
            )

    def test_command_is_offline_and_uses_only_v13_volume(self) -> None:
        command = observation.commit_identity_command()
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
                (
                    "github-cross-encoder-precision-v13-commit-freeze:"
                    "/opt/ngr-v13/commit-freeze"
                )
            ],
        )
        text = "\n".join(command)
        for forbidden in (
            observation.V10_RUNTIME_VOLUME,
            observation.V10_CACHE_FREEZE_VOLUME,
            observation.V11_ROOT_FREEZE_VOLUME,
            observation.V12_RUNTIME_VOLUME,
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
                    observation.COMMIT_FREEZE_VOLUME,
                ]
            },
            {"command": observation.commit_identity_command()},
        ]
        counts = observation._count_audit(
            status="pass",
            rows=rows,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        self.assertEqual(counts["commit_freeze_volume_create_count"], 1)
        self.assertEqual(counts["commit_identity_verifier_run_count"], 1)
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
        ):
            self.assertFalse(counts[key])

    def test_v13_scope_restores_predecessor_module(self) -> None:
        original = {
            "protocol": observation.predecessor.PROTOCOL_ID,
            "root": observation.predecessor.CONTAINER_ROOT,
            "writer": observation.predecessor._write_evidence,
        }
        with observation._v13_scope():
            self.assertEqual(
                observation.predecessor.PROTOCOL_ID, observation.PROTOCOL_ID
            )
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

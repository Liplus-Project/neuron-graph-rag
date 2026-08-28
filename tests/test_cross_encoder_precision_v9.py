from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path, PurePosixPath

from neuron_graph_rag import cross_encoder_precision_v9_observation as observation


class CrossEncoderPrecisionV9PathFreezeTest(unittest.TestCase):
    def test_container_path_serializer_accepts_only_canonical_posix(self) -> None:
        self.assertEqual(
            observation.serialize_container_path(
                PurePosixPath("/opt/ngr-v9/path-freeze")
            ),
            "/opt/ngr-v9/path-freeze",
        )
        self.assertEqual(
            observation.serialize_container_path("/opt/ngr-v9/path-freeze"),
            "/opt/ngr-v9/path-freeze",
        )

    def test_container_path_serializer_rejects_host_and_ambiguous_paths(self) -> None:
        rejected: tuple[object, ...] = (
            Path("/opt/ngr-v9/path-freeze"),
            "",
            ".",
            "..",
            "opt/ngr-v9/path-freeze",
            "\\opt\\ngr-v8\\runtime",
            "C:\\opt\\ngr-v9\\path-freeze",
            "/C:/opt/ngr-v9/path-freeze",
            "//server/share",
            "/opt//ngr-v9/path-freeze",
            "/opt/./ngr-v9/path-freeze",
            "/opt/../ngr-v9/path-freeze",
            "/opt/ngr-v9/path-freeze\x00suffix",
        )
        for value in rejected:
            with self.subTest(value=repr(value)), self.assertRaises(
                (TypeError, ValueError)
            ):
                observation.serialize_container_path(value)  # type: ignore[arg-type]

    def test_named_volume_spec_is_structured_and_fail_closed(self) -> None:
        self.assertEqual(
            observation.named_volume_spec(
                observation.PATH_FREEZE_VOLUME,
                observation.CONTAINER_ROOT,
            ),
            "github-cross-encoder-precision-v9-path-freeze:/opt/ngr-v9/path-freeze",
        )
        self.assertEqual(
            observation.named_volume_spec(
                observation.PATH_FREEZE_VOLUME,
                observation.CONTAINER_ROOT,
                mode="ro",
            ),
            "github-cross-encoder-precision-v9-path-freeze:/opt/ngr-v9/path-freeze:ro",
        )
        for volume in ("", "Uppercase", "host/path", "name:other"):
            with self.subTest(volume=volume), self.assertRaises(ValueError):
                observation.named_volume_spec(volume, observation.CONTAINER_ROOT)
        with self.assertRaises(ValueError):
            observation.named_volume_spec(
                observation.PATH_FREEZE_VOLUME,
                observation.CONTAINER_ROOT,
                mode="shared",
            )

    def test_v8_failure_volume_is_a_negative_fixture(self) -> None:
        raw = json.loads((observation.ROOT / observation.V8_FAILURE_PATH).read_text(encoding="utf-8"))
        failed_spec = raw["commands"][-1]["command"][
            raw["commands"][-1]["command"].index("--volume") + 1
        ]
        self.assertEqual(
            failed_spec,
            "github-cross-encoder-precision-v8-runtime:\\opt\\ngr-v8\\runtime",
        )
        _, destination = failed_spec.split(":", 1)
        with self.assertRaises(ValueError):
            observation.named_volume_spec("github-cross-encoder-precision-v8-runtime", destination)

    def test_smoke_command_is_one_offline_named_volume_run(self) -> None:
        command = observation.smoke_command()
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command.count("--volume"), 1)
        self.assertEqual(
            command[command.index("--volume") + 1],
            "github-cross-encoder-precision-v9-path-freeze:/opt/ngr-v9/path-freeze",
        )
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--rm", command)
        self.assertIn("/bin/sh", command)
        self.assertIn(observation.IMAGE, command)
        joined = "\n".join(command).lower()
        for forbidden in (
            "github-cross-encoder-precision-v9-runtime",
            "sqlite",
            "knowledge.db",
            "model-cache",
            "transformers",
            "sentence-transformers",
            "registered query",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)
        self.assertIn("/proc/self/mountinfo", joined)
        self.assertIn("path-transport-v9.txt", joined)

    def test_smoke_stdout_binds_path_mount_and_sentinel(self) -> None:
        digest = hashlib.sha256(
            (observation.SENTINEL_VALUE + "\n").encode("utf-8")
        ).hexdigest()
        raw = (
            "container_path=/opt/ngr-v9/path-freeze\n"
            "mount_identity=42|0:71|/|/opt/ngr-v9/path-freeze|/dev/sdz\n"
            "sentinel_path=/opt/ngr-v9/path-freeze/sentinel/path-transport-v9.txt\n"
            f"sentinel_sha256={digest}\n"
        )
        report = observation._parse_smoke_stdout(raw)
        self.assertEqual(report["container_path"], "/opt/ngr-v9/path-freeze")
        with self.assertRaises(ValueError):
            observation._parse_smoke_stdout(raw.replace(digest, "0" * 64))

    def test_prebuild_manifest_freezes_v8_and_result_free_counts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 29)
        self.assertEqual(report["registered_query_execution_count"], 0)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)
        self.assertEqual(report["performance"], "not assessed")

    def test_audit_is_pending_or_valid_terminal_evidence(self) -> None:
        report = observation.audit_evidence()
        self.assertIn(
            report["status"],
            {"prebuild_ready_evidence_absent", "pass", "error"},
        )
        self.assertEqual(report["v8_failure_sha256"], observation.V8_FAILURE_SHA256)
        self.assertEqual(report["registered_query_execution_count"], 0)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_json_contracts_are_utf8_and_canonicalizable(self) -> None:
        for relative in (observation.MANIFEST, observation.RESULT_FREE_AUDIT):
            raw = (observation.ROOT / relative).read_bytes()
            value = json.loads(raw.decode("utf-8"))
            self.assertIsInstance(value, dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)
            self.assertEqual(json.loads(observation.canonical_json_bytes(value)), value)


if __name__ == "__main__":
    unittest.main()

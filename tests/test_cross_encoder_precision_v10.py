from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from neuron_graph_rag import cross_encoder_precision_v10_observation as observation


def _git_blob_id(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


class CrossEncoderPrecisionV10CacheFreezeTest(unittest.TestCase):
    def test_container_path_serializer_accepts_only_canonical_posix(self) -> None:
        self.assertEqual(
            observation.serialize_container_path(
                PurePosixPath("/opt/ngr-v10/cache-freeze/model-cache")
            ),
            "/opt/ngr-v10/cache-freeze/model-cache",
        )
        rejected: tuple[object, ...] = (
            Path("/opt/ngr-v10/cache-freeze"),
            "",
            ".",
            "opt/ngr-v10/cache-freeze",
            "\\opt\\ngr-v10\\cache-freeze",
            "C:\\opt\\ngr-v10\\cache-freeze",
            "/C:/opt/ngr-v10/cache-freeze",
            "//server/share",
            "/opt//ngr-v10/cache-freeze",
            "/opt/../ngr-v10/cache-freeze",
        )
        for value in rejected:
            with (
                self.subTest(value=repr(value)),
                self.assertRaises((TypeError, ValueError)),
            ):
                observation.serialize_container_path(value)  # type: ignore[arg-type]

    def test_volume_and_host_mount_specs_keep_path_types_separate(self) -> None:
        self.assertEqual(
            observation.named_volume_spec(
                observation.CACHE_FREEZE_VOLUME, observation.CONTAINER_ROOT
            ),
            "github-cross-encoder-precision-v10-cache-freeze:/opt/ngr-v10/cache-freeze",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            mount = observation.host_bind_spec(source, "/input/models", mode="ro")
            self.assertTrue(mount.endswith(":/input/models:ro"))
            with self.assertRaises(ValueError):
                observation.host_bind_spec(source, "C:\\input\\models", mode="ro")

    def test_v9_initialization_is_negative_fixture_for_v10_ownership(self) -> None:
        raw = json.loads(
            (observation.ROOT / observation.V9_RAW_FAILURE_PATH).read_text(
                encoding="utf-8"
            )
        )
        v9_initialization = next(
            row["command"][-1]
            for row in raw["commands"]
            if row["command"][:2] == ["wslc", "run"]
            and "mkdir -p /opt/ngr-v9/runtime/model-cache" in row["command"][-1]
        )
        self.assertIn("mkdir -p /opt/ngr-v9/runtime/model-cache", v9_initialization)
        v10_initialization = observation._source_initialization_script()
        self.assertNotIn(
            "mkdir '/opt/ngr-v10/cache-freeze/model-cache'", v10_initialization
        )
        self.assertNotIn(
            "mkdir -p /opt/ngr-v10/cache-freeze/model-cache", v10_initialization
        )
        self.assertIn(
            "test ! -e '/opt/ngr-v10/cache-freeze/model-cache'",
            v10_initialization,
        )
        for directory in ("source", "databases", "runs", "archive", "transport"):
            self.assertIn(f"/opt/ngr-v10/cache-freeze/{directory}", v10_initialization)

    def test_model_copy_command_is_one_offline_read_only_source_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = observation.model_copy_command(Path(directory))
        self.assertEqual(command[:2], ["wslc", "run"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--rm", command)
        mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--volume"
        ]
        self.assertEqual(len(mounts), 2)
        self.assertEqual(
            mounts[0],
            "github-cross-encoder-precision-v10-cache-freeze:/opt/ngr-v10/cache-freeze",
        )
        self.assertTrue(mounts[1].endswith(":/input/models:ro"))
        self.assertNotIn(observation.FUTURE_RUNTIME_VOLUME, "\n".join(command))
        self.assertIn("model-copy-verify", command)
        for forbidden in ("registered query", "model-probe", "worker", "sqlite"):
            self.assertNotIn(forbidden, "\n".join(command).lower())

    def test_model_verifier_rejects_preexisting_target_before_source_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            marker = cache / "keep.txt"
            marker.write_text("preexisting", encoding="utf-8")
            with self.assertRaisesRegex(
                FileExistsError, "dedicated ext4 model cache already exists"
            ):
                observation.model_copy_verify(
                    root / "missing-source",
                    cache,
                    root / "missing-models.json",
                    root / "output.json",
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preexisting")

    def test_model_verifier_exclusive_copy_and_post_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            cache = root / "cache"
            output = root / "report.json"
            models_path = root / "models.json"
            models = []
            payloads = (b"model-a\n", b"model-b\n")
            for index, payload in enumerate(payloads):
                model_id = f"owner/model-{index}"
                revision = f"revision-{index}"
                snapshot = (
                    source
                    / ("models--" + model_id.replace("/", "--"))
                    / "snapshots"
                    / revision
                )
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_bytes(payload)
                models.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "required_files": [
                            {
                                "path": "config.json",
                                "size": len(payload),
                                "git_blob_id": _git_blob_id(payload),
                                "lfs_sha256": None,
                            }
                        ],
                    }
                )
            models_path.write_text(json.dumps({"models": models}), encoding="utf-8")
            report = observation.model_copy_verify(source, cache, models_path, output)
            self.assertTrue(report["target_absent_before_exclusive_create"])
            self.assertTrue(report["target_exclusive_create"])
            self.assertTrue(report["all_required_files_byte_identical"])
            self.assertEqual(report["model_count"], 2)
            self.assertEqual(report["required_file_count"], 2)
            self.assertEqual(
                [
                    row["sha256"]
                    for model in report["source_models"]
                    for row in model["required_files"]
                ],
                [
                    row["sha256"]
                    for model in report["models"]
                    for row in model["required_files"]
                ],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_prebuild_freezes_v9_anchors_and_result_free_counts(self) -> None:
        report = observation.validate_prebuild()
        self.assertEqual(report["status"], "prebuild_contract_valid")
        self.assertEqual(report["predecessor_artifact_count"], 20)
        self.assertEqual(report["model_count"], 2)
        self.assertEqual(report["required_file_count"], 12)
        self.assertEqual(report["required_file_size"], 3427616927)
        self.assertFalse(report["model_cache_created_by_source_initialization"])
        for key in (
            "registered_query_execution_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "observed_result_count",
        ):
            self.assertEqual(report[key], 0)
        self.assertEqual(report["performance"], "not assessed")

    def test_count_audit_is_result_free_and_counts_verifier_once(self) -> None:
        rows = [
            {
                "command": [
                    "wslc",
                    "volume",
                    "create",
                    observation.CACHE_FREEZE_VOLUME,
                ]
            },
            {"command": observation.model_copy_command(observation.ROOT)},
        ]
        counts = observation._count_audit(
            status="pass",
            rows=rows,
            runtime_absent=True,
            predecessor_unchanged=True,
        )
        self.assertEqual(counts["cache_freeze_volume_create_count"], 1)
        self.assertEqual(counts["model_copy_verifier_run_count"], 1)
        self.assertEqual(counts["retry_count"], 0)
        self.assertEqual(counts["development_claim_count"], 0)
        self.assertEqual(counts["holdout_claim_count"], 0)
        self.assertEqual(counts["registered_query_execution_count"], 0)
        self.assertEqual(counts["model_forward_inference_count"], 0)
        self.assertEqual(counts["observed_result_count"], 0)
        self.assertEqual(counts["performance"], "not assessed")

    def test_audit_is_pending_or_valid_terminal_evidence(self) -> None:
        report = observation.audit_evidence()
        self.assertIn(
            report["status"], {"prebuild_ready_evidence_absent", "pass", "error"}
        )
        self.assertEqual(report["predecessor_artifact_count"], 20)
        self.assertEqual(report["registered_query_execution_count"], 0)
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)

    def test_json_contracts_are_utf8_without_replacement_character(self) -> None:
        for relative in (observation.MANIFEST, observation.RESULT_FREE_AUDIT):
            raw = (observation.ROOT / relative).read_bytes()
            value = json.loads(raw.decode("utf-8"))
            self.assertIsInstance(value, dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

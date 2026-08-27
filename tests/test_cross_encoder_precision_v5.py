from __future__ import annotations

import copy
import hashlib
import re
import unittest
from pathlib import Path

from neuron_graph_rag import cross_encoder_precision_v4_evaluation as v4_evaluation
from neuron_graph_rag import cross_encoder_precision_v5_evaluation as evaluation
from tests import test_cross_encoder_precision_v4 as v4_tests

ROOT = Path(__file__).resolve().parents[1]


class CrossEncoderPrecisionV5FreezeTest(v4_tests.CrossEncoderPrecisionV4FreezeTest):
    def setUp(self) -> None:
        self._previous_v4_evaluation = v4_tests.evaluation
        v4_tests.evaluation = evaluation
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        v4_tests.evaluation = self._previous_v4_evaluation

    def test_protocol_is_hashed_disjoint_bilingual_and_result_free(self) -> None:
        protocol = evaluation.load_protocol()
        self.assertEqual(len(protocol["corpus"]["documents"]), 24)
        self.assertEqual(
            [row["candidate_id"] for row in protocol["candidates"]["candidates"]],
            list(evaluation.CANDIDATE_IDS),
        )
        self.assertEqual(
            [row["gate_id"] for row in protocol["gate"]["gates"]],
            list(evaluation.GATE_IDS),
        )
        for stage in evaluation.STAGES:
            cases = protocol["queries"]["stages"][stage]
            for cohort in evaluation.COHORTS:
                self.assertEqual(
                    [row["language"] for row in cases if row["cohort"] == cohort],
                    ["en", "ja"],
                )
        audit = protocol["result_free_audit"]
        self.assertEqual(audit["freeze_registered_query_execution_count"], 0)
        self.assertEqual(audit["freeze_model_inference_count"], 0)
        self.assertEqual(audit["freeze_observed_result_count"], 0)
        self.assertEqual(
            audit["count_scope"],
            "v5 registered query/model inference/observed result counts only; "
            "historical v1/v2/v3/v4 observations excluded",
        )
        self.assertEqual(
            audit["container_acceptance_count_scope"],
            "final accepted v5 container contract only; pre-finalization iterative "
            "image builds and offline validations excluded",
        )
        self.assertEqual(audit["accepted_wslc_image_build_count"], 1)
        self.assertEqual(audit["accepted_offline_synthetic_validation_count"], 1)
        self.assertFalse(audit["predecessor_evidence_semantic_content_opened"])
        self.assertFalse(audit["model_cache_opened"])
        self.assertFalse(audit["model_weights_opened"])
        self.assertFalse(audit["model_forward_executed"])
        output_paths = [
            path
            for stage in protocol["manifest"]["outputs"].values()
            for path in stage.values()
        ]
        self.assertEqual(len(output_paths), len(set(output_paths)))
        self.assertTrue(
            all("github_cross_encoder_precision_v5" in path for path in output_paths)
        )
        self.assertTrue(
            all(
                f"github_cross_encoder_precision_v{version}" not in path
                for version in range(1, 5)
                for path in output_paths
            )
        )

    def test_result_free_audit_count_scopes_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        for name, mutate in (
            (
                "result scope",
                lambda row: row["result_free_audit"].update(count_scope="all runs"),
            ),
            (
                "container scope",
                lambda row: row["result_free_audit"].update(
                    container_acceptance_count_scope="all builds"
                ),
            ),
            (
                "accepted build count",
                lambda row: row["result_free_audit"].update(
                    accepted_wslc_image_build_count=2
                ),
            ),
            (
                "accepted validation count",
                lambda row: row["result_free_audit"].update(
                    accepted_offline_synthetic_validation_count=2
                ),
            ),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(protocol)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_protocol(tampered)

    def test_v4_v5_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        v5 = evaluation.load_protocol()
        v4 = v4_evaluation.load_protocol()
        for key in (
            "corpus",
            "queries",
            "gold",
            "models",
            "candidates",
            "gate",
            "result_schema",
        ):
            expected = copy.deepcopy(v4[key])
            expected["protocol_id"] = evaluation.PROTOCOL_ID
            self.assertEqual(expected, v5[key], key)
        self.assertEqual(
            {
                key: v4["platform"]["execution"][key]
                for key in (
                    "device",
                    "dtype",
                    "eval",
                    "inference_mode",
                    "batch_size",
                    "fresh_process",
                    "fresh_database",
                    "local_files_only",
                )
            },
            {
                key: v5["platform"]["execution"][key]
                for key in (
                    "device",
                    "dtype",
                    "eval",
                    "inference_mode",
                    "batch_size",
                    "fresh_process",
                    "fresh_database",
                    "local_files_only",
                )
            },
        )

        expected_v4_paths = {
            "docs/cross-encoder-precision-freeze-v4.md",
            "src/neuron_graph_rag/cross_encoder_precision_v4_evaluation.py",
            "src/neuron_graph_rag/cross_encoder_precision_v4_observation.py",
            "tests/test_cross_encoder_precision_v4.py",
            "tests/test_cross_encoder_precision_v4_observation.py",
            "tools/run_cross_encoder_precision_v4_wsl.sh",
            *{
                str(path.relative_to(ROOT)).replace("\\", "/")
                for path in (ROOT / "tests/fixtures").glob(
                    "github_cross_encoder_precision_v4.*"
                )
            },
            *{
                str(path.relative_to(ROOT)).replace("\\", "/")
                for path in (
                    ROOT / "tests/evidence/github_cross_encoder_precision_v4"
                ).rglob("*")
                if path.is_file()
            },
        }
        registry = v5["manifest"]["v4_immutable_sha256"]
        self.assertEqual(set(registry), expected_v4_paths)
        for relative, expected_hash in registry.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected_hash,
            )

    def test_dependency_versions_match_v4_and_exact_linux_artifacts(self) -> None:
        v5 = evaluation.load_protocol()
        v4 = v4_evaluation.load_protocol()

        def versions(lock: str) -> dict[str, str]:
            rows = dict(
                re.findall(
                    r"^([a-z0-9][a-z0-9._-]*)==([^ \\\n]+)",
                    lock,
                    flags=re.MULTILINE,
                )
            )
            direct = re.search(r"^torch @ .*torch-([0-9.]+)%2Bcpu-", lock, re.MULTILINE)
            if direct:
                rows["torch"] = f"{direct.group(1)}+cpu"
            return rows

        self.assertEqual(
            versions(v4["requirements_lock"]), versions(v5["requirements_lock"])
        )
        artifacts = v5["dependency_artifacts"]["artifacts"]
        self.assertEqual(len(artifacts), 26)
        self.assertEqual(len({row["name"] for row in artifacts}), 26)
        self.assertFalse(any(row["name"].startswith("nvidia-") for row in artifacts))
        self.assertFalse(any(row["name"] == "triton" for row in artifacts))
        torch = next(row for row in artifacts if row["name"] == "torch")
        self.assertEqual(torch["version"], "2.4.1+cpu")
        self.assertIn("download.pytorch.org/whl/cpu", torch["url"])

    def test_v2_v3_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        # Override the v4-only lock/source delta assertion. The successor delta
        # is covered against v4 directly, including the full v4 hash registry.
        self.test_v4_v5_semantic_diff_and_predecessor_byte_immutability()

    def test_linux_platform_contract_accepts_only_frozen_ext4_metadata(self) -> None:
        protocol = evaluation.load_protocol()
        valid = evaluation.build_synthetic_runtime_metadata(protocol)
        evaluation.validate_runtime_metadata(protocol, valid)
        for name, mutate in (
            ("Windows", lambda row: row.update(os="windows")),
            ("Python 3.14", lambda row: row.update(python_version="3.14.0")),
            ("arm64", lambda row: row.update(architecture="arm64")),
            ("image", lambda row: row.update(image_id="sha256:" + "f" * 64)),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(valid)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_runtime_metadata(protocol, tampered)

    def test_platform_paths_are_dedicated_exclusive_and_under_run_root(self) -> None:
        protocol = evaluation.load_protocol()
        for name, mutate in (
            (
                "Windows mount",
                lambda row: row["platform"]["run_root"].update(path="/mnt/c/ngr-v5"),
            ),
            (
                "host bind",
                lambda row: row["platform"]["run_root"].update(host_bind_mount=True),
            ),
            (
                "shared DB",
                lambda row: row["platform"]["run_root"].update(
                    shared_windows_database=True
                ),
            ),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(protocol)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_protocol(tampered)

    def test_container_contract_rejects_platform_image_and_routing_tamper(self) -> None:
        protocol = evaluation.load_protocol()
        for name, mutate in (
            (
                "tag only base",
                lambda row: row["platform"]["container"]["base_image"].update(
                    digest=""
                ),
            ),
            (
                "wrong architecture",
                lambda row: row["platform"]["container"].update(architecture="arm64"),
            ),
            (
                "wrong Python",
                lambda row: row["platform"]["python"].update(version="3.12.0"),
            ),
            (
                "wrong WSLC",
                lambda row: row["platform"]["wslc"].update(version="2.9.3.0"),
            ),
            (
                "online validation",
                lambda row: row["platform"]["container"]["validation_command"].remove(
                    "none"
                ),
            ),
            (
                "index fallback",
                lambda row: row["platform"]["resolver"].update(index_fallback=True),
            ),
            (
                "PyPI torch",
                lambda row: next(
                    item
                    for item in row["dependency_artifacts"]["artifacts"]
                    if item["name"] == "torch"
                ).update(url="https://files.pythonhosted.org/torch.whl"),
            ),
            (
                "artifact hash",
                lambda row: row["dependency_artifacts"]["artifacts"][0].update(
                    sha256="f" * 64
                ),
            ),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(protocol)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_protocol(tampered)

    def test_runtime_metadata_rejects_cuda_network_and_inference(self) -> None:
        protocol = evaluation.load_protocol()
        valid = evaluation.build_synthetic_runtime_metadata(protocol)
        evaluation.validate_runtime_metadata(protocol, valid)
        for name, mutate in (
            ("cuda", lambda row: row.update(cuda="12.1")),
            ("network", lambda row: row.update(network="enabled")),
            ("query", lambda row: row.update(registered_query_execution_count=1)),
            ("forward", lambda row: row.update(model_forward_count=1)),
            ("image", lambda row: row.update(image_id="sha256:" + "f" * 64)),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(valid)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_runtime_metadata(protocol, tampered)


if __name__ == "__main__":
    unittest.main()

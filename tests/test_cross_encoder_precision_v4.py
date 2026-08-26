from __future__ import annotations

import copy
import hashlib
import re
import unittest
from pathlib import Path

from neuron_graph_rag import cross_encoder_precision_v3_evaluation as v3_evaluation
from neuron_graph_rag import cross_encoder_precision_v4_evaluation as evaluation
from tests import test_cross_encoder_precision_v3 as v3_tests

ROOT = Path(__file__).resolve().parents[1]


class CrossEncoderPrecisionV4FreezeTest(v3_tests.CrossEncoderPrecisionV3FreezeTest):
    def setUp(self) -> None:
        self._previous_evaluation = v3_tests.evaluation
        v3_tests.evaluation = evaluation

    def tearDown(self) -> None:
        v3_tests.evaluation = self._previous_evaluation

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
        self.assertFalse(audit["model_cache_opened"])
        self.assertFalse(audit["model_weights_opened"])
        self.assertFalse(audit["venv_opened"])
        output_paths = [
            path
            for stage in protocol["manifest"]["outputs"].values()
            for path in stage.values()
        ]
        self.assertEqual(len(output_paths), len(set(output_paths)))
        self.assertTrue(
            all("github_cross_encoder_precision_v4" in path for path in output_paths)
        )
        self.assertTrue(
            all(
                "github_cross_encoder_precision_v1" not in path
                and "github_cross_encoder_precision_v2" not in path
                and "github_cross_encoder_precision_v3" not in path
                for path in output_paths
            )
        )

    def test_v2_v3_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        v4 = evaluation.load_protocol()
        v3 = v3_evaluation.load_protocol()
        for key in (
            "corpus",
            "queries",
            "gold",
            "models",
            "candidates",
            "gate",
            "result_schema",
        ):
            expected = copy.deepcopy(v3[key])
            expected["protocol_id"] = evaluation.PROTOCOL_ID
            self.assertEqual(expected, v4[key], key)
        self.assertEqual(
            (ROOT / "tests/fixtures/github_cross_encoder_precision_v3.requirements.in").read_bytes(),
            (ROOT / "tests/fixtures/github_cross_encoder_precision_v4.requirements.in").read_bytes(),
        )

        def versions(lock: str) -> dict[str, str]:
            return dict(
                re.findall(
                    r"^([a-z0-9][a-z0-9._-]*)==([^ \\\n]+)",
                    lock,
                    flags=re.MULTILINE,
                )
            )

        v3_versions = versions(v3["requirements_lock"])
        v4_versions = versions(v4["requirements_lock"])
        self.assertEqual(set(v3_versions) - set(v4_versions), {"colorama"})
        self.assertEqual(set(v4_versions) - set(v3_versions), set())
        self.assertEqual(
            {
                key: v4_versions[key].split("+", 1)[0]
                for key in set(v3_versions) & set(v4_versions)
            },
            {
                key: v3_versions[key].split("+", 1)[0]
                for key in set(v3_versions) & set(v4_versions)
            },
        )
        self.assertEqual(v4_versions["torch"], "2.4.1+cpu")
        self.assertFalse(any(name.startswith("nvidia-") for name in v4_versions))
        self.assertNotIn("triton", v4_versions)

        expected_v3_paths = {
            "docs/cross-encoder-precision-freeze-v3.md",
            "docs/cross-encoder-precision-observation-v3.md",
            "src/neuron_graph_rag/cross_encoder_precision_v3_evaluation.py",
            "src/neuron_graph_rag/cross_encoder_precision_v3_observation.py",
            "tests/test_cross_encoder_precision_v3.py",
            "tests/test_cross_encoder_precision_v3_observation.py",
            *{
                str(path.relative_to(ROOT)).replace("\\", "/")
                for path in (ROOT / "tests/fixtures").glob(
                    "github_cross_encoder_precision_v3.*"
                )
            },
            *{
                str(path.relative_to(ROOT)).replace("\\", "/")
                for path in (
                    ROOT / "tests/evidence/github_cross_encoder_precision_v3"
                ).rglob("*")
                if path.is_file()
            },
        }
        registry = v4["manifest"]["v3_immutable_sha256"]
        self.assertEqual(set(registry), expected_v3_paths)
        for relative, expected_hash in registry.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected_hash,
            )

    def test_linux_platform_contract_accepts_only_frozen_ext4_metadata(self) -> None:
        protocol = evaluation.load_protocol()
        valid = evaluation.build_synthetic_platform_metadata(protocol)
        evaluation.validate_runtime_platform(protocol, valid)
        for name, mutate in (
            ("Windows", lambda row: row.update(os="windows")),
            ("Python 3.14", lambda row: row.update(python_version="3.14.0")),
            ("mnt c", lambda row: row.update(run_root="/mnt/c/ngr-v4")),
            (
                "interpreter hash",
                lambda row: row.update(python_artifact_sha256="f" * 64),
            ),
            ("lock hash", lambda row: row.update(dependency_lock_sha256="f" * 64)),
            ("platform tag", lambda row: row.update(platform_tag="win_amd64")),
            ("filesystem", lambda row: row.update(run_root_filesystem="ntfs")),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(valid)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.validate_runtime_platform(protocol, tampered)

    def test_platform_paths_are_dedicated_exclusive_and_under_run_root(self) -> None:
        protocol = evaluation.load_protocol()
        valid = evaluation.build_synthetic_platform_metadata(protocol)
        outside = copy.deepcopy(valid)
        outside["paths"]["database"] = "/tmp/knowledge.db"
        with self.assertRaises(ValueError):
            evaluation.validate_runtime_platform(protocol, outside)
        duplicate = copy.deepcopy(valid)
        duplicate["paths"]["transport"] = duplicate["paths"]["runtime_result"]
        with self.assertRaises(ValueError):
            evaluation.validate_runtime_platform(protocol, duplicate)


if __name__ == "__main__":
    unittest.main()

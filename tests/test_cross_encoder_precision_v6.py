from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from neuron_graph_rag import cross_encoder_precision_v5_evaluation as v5_evaluation
from neuron_graph_rag import cross_encoder_precision_v6_evaluation as evaluation
from tests import test_cross_encoder_precision_v5 as v5_tests

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "containers/github_cross_encoder_precision_v6/runtime_content.py"
CONTENT_SPEC = importlib.util.spec_from_file_location("v6_runtime_content", CONTENT_PATH)
if CONTENT_SPEC is None or CONTENT_SPEC.loader is None:
    raise RuntimeError("unable to load v6 runtime content tool")
content = importlib.util.module_from_spec(CONTENT_SPEC)
CONTENT_SPEC.loader.exec_module(content)


def _file(path: str, body: bytes = b"same") -> dict[str, object]:
    return {
        "path": path,
        "type": "file",
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
    }


class RuntimeContentFingerprintV1Test(unittest.TestCase):
    def _report(self, entries: list[dict[str, object]]) -> dict[str, object]:
        return content.build_report(
            entries,
            dependency_artifact_registry_sha256="a" * 64,
            python_identity={
                "implementation": "CPython",
                "version": "3.11.15",
                "abi": "cp311",
            },
        )

    def test_metadata_and_enumeration_order_do_not_change_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            roots = [(root, PurePosixPath("root"))]
            before = content.build_report(
                content.collect_entries(roots),
                dependency_artifact_registry_sha256="a" * 64,
                python_identity={"implementation": "CPython", "version": "3.11.15", "abi": "cp311"},
            )
            os.utime(first, (1_000_000_000, 1_000_000_000))
            first.replace(root / "moved.txt")
            (root / "moved.txt").replace(first)
            after = content.build_report(
                reversed(content.collect_entries(roots)),
                dependency_artifact_registry_sha256="a" * 64,
                python_identity={"implementation": "CPython", "version": "3.11.15", "abi": "cp311"},
            )
        self.assertEqual(before, after)
        self.assertEqual(
            set(before["normalized_entries"][0]),
            {"path", "type", "content_sha256", "size"},
        )

    def test_content_symlink_missing_extra_and_registry_changes_are_detected(self) -> None:
        base = self._report([_file("root/a")])
        changed = self._report([_file("root/a", b"changed")])
        missing = self._report([])
        extra = self._report([_file("root/a"), _file("root/b")])
        symlink_a = self._report(
            [{"path": "root/link", "type": "symlink", "symlink_target": "a", "size": 1}]
        )
        symlink_b = self._report(
            [{"path": "root/link", "type": "symlink", "symlink_target": "b", "size": 1}]
        )
        dependency = content.build_report(
            [_file("root/a")],
            dependency_artifact_registry_sha256="b" * 64,
            python_identity={"implementation": "CPython", "version": "3.11.15", "abi": "cp311"},
        )
        for other in (changed, missing, extra, symlink_a, symlink_b, dependency):
            with self.subTest(other=other["fingerprint_sha256"]):
                self.assertNotEqual(base["fingerprint_sha256"], other["fingerprint_sha256"])

    def test_duplicate_traversal_and_case_collision_fail_closed(self) -> None:
        for name, entries in (
            ("duplicate", [_file("root/a"), _file("root/a")]),
            ("traversal", [_file("root/../a")]),
            ("backslash", [_file("root\\a")]),
            ("case", [_file("root/a"), _file("ROOT/A")]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._report(entries)

    def test_algorithm_base_and_exclusion_registry_are_frozen(self) -> None:
        report = self._report([_file("root/a")])
        for key, value in (
            ("algorithm_version", "ngr.wslc-runtime-content/v2"),
            ("base_digest", "sha256:" + "f" * 64),
        ):
            tampered = copy.deepcopy(report)
            tampered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                content.validate_report(tampered)
        original = content.exclusion_registry_sha256()
        previous = content.EXCLUSION_REGISTRY
        try:
            content.EXCLUSION_REGISTRY = (*previous, {"kind": "suffix", "value": ".new"})
            self.assertNotEqual(original, content.exclusion_registry_sha256())
        finally:
            content.EXCLUSION_REGISTRY = previous


class CrossEncoderPrecisionV6FreezeTest(v5_tests.CrossEncoderPrecisionV5FreezeTest):
    def setUp(self) -> None:
        self._previous_v5_evaluation = v5_tests.evaluation
        v5_tests.evaluation = evaluation
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        v5_tests.evaluation = self._previous_v5_evaluation

    def test_protocol_is_hashed_disjoint_bilingual_and_result_free(self) -> None:
        protocol = evaluation.load_protocol()
        audit = protocol["result_free_audit"]
        self.assertEqual(
            (
                audit["freeze_registered_query_execution_count"],
                audit["freeze_model_inference_count"],
                audit["freeze_observed_result_count"],
            ),
            (0, 0, 0),
        )
        self.assertEqual(audit["freeze_outcome"], "fail_closed_offline_attestation_not_exact")
        self.assertEqual(audit["one_shot_wslc_image_build_count"], 2)
        self.assertEqual(audit["runtime_content_report_count"], 2)
        self.assertEqual(audit["offline_attestation_report_count"], 2)
        self.assertEqual(audit["additional_wslc_image_build_count"], 0)
        self.assertEqual(audit["additional_offline_report_run_count"], 0)
        self.assertFalse(audit["accepted_image"])
        self.assertFalse(audit["successor_observation_allowed"])
        self.assertEqual(audit["performance"], "not assessed")
        self.assertFalse(audit["predecessor_evidence_semantic_content_opened"])
        self.assertFalse(audit["model_cache_opened"])
        self.assertFalse(audit["model_weights_opened"])
        self.assertFalse(audit["model_forward_executed"])
        outputs = [
            path
            for stage in protocol["manifest"]["outputs"].values()
            for path in stage.values()
        ]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertTrue(all(evaluation.STEM in path for path in outputs))

    def test_frozen_json_is_canonical_utf8(self) -> None:
        compact_suffixes = (
            ".attestation.build-a.json",
            ".attestation.build-b.json",
            ".runtime-content.build-a.json",
            ".runtime-content.build-b.json",
        )
        for path in (ROOT / "tests/fixtures").glob(f"{evaluation.STEM}.*.json"):
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            if path.name.endswith(compact_suffixes):
                self.assertEqual(raw, evaluation._CONTENT.canonical_json_bytes(value))
            else:
                self.assertEqual(
                    raw.decode("utf-8"),
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                )

    def test_result_free_audit_count_scopes_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        for name, key, value in (
            ("result scope", "count_scope", "all runs"),
            ("container scope", "container_attempt_count_scope", "all builds"),
            ("build count", "one_shot_wslc_image_build_count", 1),
            ("fingerprint count", "runtime_content_report_count", 1),
            ("attestation count", "offline_attestation_report_count", 1),
            ("additional build", "additional_wslc_image_build_count", 1),
            ("additional report", "additional_offline_report_run_count", 1),
            ("accepted image", "accepted_image", True),
            ("successor", "successor_observation_allowed", True),
        ):
            tampered = copy.deepcopy(protocol)
            tampered["result_free_audit"][key] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation.validate_protocol(tampered)

    def test_v5_v6_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        v6 = evaluation.load_protocol()
        v5 = v5_evaluation.load_protocol()
        for key in (
            "corpus", "queries", "gold", "models", "candidates", "gate", "result_schema"
        ):
            expected = copy.deepcopy(v5[key])
            expected["protocol_id"] = evaluation.PROTOCOL_ID
            self.assertEqual(expected, v6[key], key)
        for key in (
            "device", "dtype", "eval", "inference_mode", "batch_size",
            "fresh_process", "fresh_database", "local_files_only",
        ):
            self.assertEqual(v5["platform"]["execution"][key], v6["platform"]["execution"][key])
        registry = v6["manifest"]["v5_immutable_sha256"]
        expected_paths = {
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in ROOT.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (
                "cross_encoder_precision_v5" in path.as_posix()
                or "cross-encoder-precision-freeze-v5" in path.as_posix()
                or "cross-encoder-precision-observation-v5" in path.as_posix()
            )
        }
        self.assertEqual(set(registry), expected_paths)
        for relative, expected_hash in registry.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected_hash)

    def test_v4_v5_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        self.test_v5_v6_semantic_diff_and_predecessor_byte_immutability()

    def test_v2_v3_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        self.test_v5_v6_semantic_diff_and_predecessor_byte_immutability()

    def test_content_identity_matches_but_exact_attestation_fails_closed(self) -> None:
        protocol = evaluation.load_protocol()
        contract = protocol["platform"]["content_equivalence"]
        root = protocol["root"]
        content_a = evaluation._read_canonical_json(root / contract["runtime_content_build_a"]["path"])
        content_b = evaluation._read_canonical_json(root / contract["runtime_content_build_b"]["path"])
        attestation_a = evaluation._read_canonical_json(root / contract["attestation_build_a"]["path"])
        attestation_b = evaluation._read_canonical_json(root / contract["attestation_build_b"]["path"])
        self.assertEqual(content_a, content_b)
        self.assertEqual(attestation_a, attestation_b)
        extras, missing = evaluation.installed_distribution_delta(content_a, attestation_a)
        self.assertEqual(
            extras,
            [
                "pip-24.0.dist-info",
                "setuptools-79.0.1.dist-info",
                "wheel-0.46.3.dist-info",
            ],
        )
        self.assertEqual(missing, [])
        for image_id_a, image_id_b in (
            ("sha256:" + "a" * 64, "sha256:" + "b" * 64),
            ("sha256:" + "a" * 64, "sha256:" + "a" * 64),
        ):
            with self.assertRaises(evaluation.ExactInstalledDistributionError):
                evaluation.validate_content_equivalence(
                    content_a, content_b, attestation_a, attestation_b, protocol,
                    image_id_a=image_id_a, image_id_b=image_id_b,
                )
        tampered = copy.deepcopy(content_b)
        tampered["normalized_entries"][0]["size"] += 1
        with self.assertRaises(ValueError):
            evaluation.validate_content_equivalence(
                content_a, tampered, attestation_a, attestation_b, protocol,
                image_id_a="sha256:" + "a" * 64, image_id_b="sha256:" + "a" * 64,
            )

    def test_linux_platform_contract_accepts_only_frozen_ext4_metadata(self) -> None:
        self.test_content_identity_matches_but_exact_attestation_fails_closed()

    def test_container_contract_rejects_platform_image_and_routing_tamper(self) -> None:
        protocol = evaluation.load_protocol()
        for name, mutate in (
            ("tag only base", lambda row: row["platform"]["container"]["base_image"].update(digest="")),
            ("wrong architecture", lambda row: row["platform"]["container"].update(architecture="arm64")),
            ("wrong Python", lambda row: row["platform"]["python"].update(version="3.12.0")),
            ("wrong WSLC", lambda row: row["platform"]["wslc"].update(version="2.9.3.0")),
            ("index fallback", lambda row: row["platform"]["resolver"].update(index_fallback=True)),
            ("exclusion registry", lambda row: row["platform"]["content_equivalence"].update(exclusion_registry_sha256="f" * 64)),
            ("accepted image", lambda row: row["platform"]["container"].update(accepted_image="build_a")),
            ("successor", lambda row: row["platform"]["content_equivalence"].update(successor_observation_allowed=True)),
            ("PyPI torch", lambda row: next(item for item in row["dependency_artifacts"]["artifacts"] if item["name"] == "torch").update(url="https://files.pythonhosted.org/torch.whl")),
        ):
            tampered = copy.deepcopy(protocol)
            mutate(tampered)
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation.validate_protocol(tampered)

    def test_runtime_metadata_rejects_cuda_network_and_inference(self) -> None:
        protocol = evaluation.load_protocol()
        contract = protocol["platform"]["content_equivalence"]
        valid = evaluation._read_canonical_json(protocol["root"] / contract["attestation_build_a"]["path"])
        evaluation.validate_attestation(valid, protocol)
        for name, key, value in (
            ("cuda", "torch_cuda", "12.1"),
            ("network", "network", "enabled"),
            ("query", "registered_query_count", 1),
            ("forward", "model_forward_inference_count", 1),
            ("result", "observed_result_count", 1),
        ):
            tampered = copy.deepcopy(valid)
            tampered[key] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation.validate_attestation(tampered, protocol)


if __name__ == "__main__":
    unittest.main()

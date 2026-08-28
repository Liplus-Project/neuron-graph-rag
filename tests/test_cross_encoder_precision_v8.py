from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

from neuron_graph_rag import cross_encoder_precision_v8_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "containers/github_cross_encoder_precision_v8/runtime_content.py"
CONTENT_SPEC = importlib.util.spec_from_file_location(
    "v8_runtime_content", CONTENT_PATH
)
if CONTENT_SPEC is None or CONTENT_SPEC.loader is None:
    raise RuntimeError("unable to load v8 runtime content tool")
content = importlib.util.module_from_spec(CONTENT_SPEC)
CONTENT_SPEC.loader.exec_module(content)


def _file(path: str, body: bytes = b"same") -> dict[str, object]:
    return {
        "path": path,
        "type": "file",
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
    }


class CrossEncoderPrecisionV8FreezeTest(unittest.TestCase):
    def _synthetic(self) -> tuple[dict[str, object], dict[str, object]]:
        protocol = evaluation.load_protocol()
        registry_sha = protocol["platform"]["expected_distribution_registry"]["sha256"]
        entries: list[dict[str, object]] = []
        inventory: list[dict[str, str]] = []
        distributions: list[dict[str, str]] = []
        for row in protocol["expected_distributions"]["distributions"]:
            name, version = row["canonical_name"], row["version"]
            raw = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
            metadata_path = f"site-packages/{name}-{version}.dist-info/METADATA"
            entries.append(_file(metadata_path, raw))
            inventory.append(
                {
                    "metadata_path": metadata_path,
                    "name": name,
                    "version": version,
                    "canonical_name": name,
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            distributions.append(
                {"canonical_name": name, "name": name, "version": version}
            )
        report = content.build_report(
            entries,
            dependency_artifact_registry_sha256="a" * 64,
            expected_distribution_registry_sha256=registry_sha,
            filesystem_distributions=inventory,
            python_identity={
                "implementation": "CPython",
                "version": "3.11.15",
                "abi": "cp311",
            },
        )
        attestation = {
            "architecture": "amd64",
            "distributions": sorted(
                distributions, key=lambda row: row["canonical_name"]
            ),
            "expected_distribution_registry_sha256": registry_sha,
            "filesystem_probe": "exclusive-create",
            "forbidden_distributions": [],
            "model_forward_inference_count": 0,
            "network": "disabled",
            "observed_result_count": 0,
            "os": "linux",
            "python": {
                "abi": "cp311",
                "implementation": "CPython",
                "version": "3.11.15",
            },
            "registered_query_count": 0,
            "synthetic_tensor_probe": {
                "device": "cpu",
                "dtype": "float32",
                "output": [-1.0, 4.0],
            },
            "torch_cuda": None,
        }
        return report, attestation

    def test_pending_protocol_is_hashed_disjoint_bilingual_and_result_free(self) -> None:
        protocol = evaluation.load_protocol()
        evaluation.validate_protocol(protocol)
        audit = protocol["result_free_audit"]
        self.assertEqual(
            (
                audit["freeze_registered_query_execution_count"],
                audit["freeze_model_inference_count"],
                audit["freeze_observed_result_count"],
            ),
            (0, 0, 0),
        )
        self.assertEqual(audit["freeze_outcome"], evaluation.PENDING_OUTCOME)
        self.assertEqual(
            (
                audit["one_shot_wslc_image_build_count"],
                audit["runtime_content_report_count"],
                audit["offline_attestation_report_count"],
            ),
            (0, 0, 0),
        )
        self.assertFalse(audit["accepted_image"])
        self.assertFalse(audit["successor_observation_allowed"])
        self.assertEqual(len(protocol["expected_distributions"]["distributions"]), 29)
        self.assertEqual(len(protocol["corpus"]["documents"]), 24)
        for stage in evaluation.STAGES:
            self.assertEqual(len(protocol["queries"]["stages"][stage]), 8)

    def test_pending_one_shot_paths_are_absent_and_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        contract = protocol["platform"]["content_equivalence"]
        self.assertIsNone(contract["fingerprint_sha256"])
        self.assertIsNone(contract["attestation_sha256"])
        self.assertFalse(contract["exact_installed_distribution_set_attested"])
        self.assertFalse(contract["successor_observation_allowed"])
        self.assertIsNone(contract["metadata_correspondence_sha256"])
        for key in (
            "runtime_content_build_a",
            "runtime_content_build_b",
            "attestation_build_a",
            "attestation_build_b",
        ):
            self.assertFalse((ROOT / contract[key]["path"]).exists())

    def test_exact_29_accepts_order_and_different_image_ids(self) -> None:
        report, attestation = self._synthetic()
        reordered = content.build_report(
            reversed(report["normalized_entries"]),
            dependency_artifact_registry_sha256="a" * 64,
            expected_distribution_registry_sha256=report[
                "expected_distribution_registry_sha256"
            ],
            filesystem_distributions=reversed(report["filesystem_distributions"]),
            python_identity=report["python"],
        )
        self.assertEqual(report, reordered)
        evaluation.validate_content_equivalence(
            report,
            reordered,
            attestation,
            copy.deepcopy(attestation),
            evaluation.load_protocol(),
            image_id_a="sha256:" + "a" * 64,
            image_id_b="sha256:" + "b" * 64,
        )

    def test_three_inventory_axes_reject_independent_tamper(self) -> None:
        report, attestation = self._synthetic()
        protocol = evaluation.load_protocol()
        actual = copy.deepcopy(attestation)
        actual["distributions"][0]["version"] = "0"
        filesystem = copy.deepcopy(report)
        filesystem["filesystem_distributions"][0]["version"] = "0"
        payload = {k: v for k, v in filesystem.items() if k != "fingerprint_sha256"}
        filesystem["fingerprint_sha256"] = content.sha256_bytes(
            content.canonical_json_bytes(payload)
        )
        expected = copy.deepcopy(protocol)
        expected["expected_distributions"]["distributions"][0]["version"] = "0"
        cases = (
            ("importlib", report, actual, protocol),
            ("filesystem", filesystem, attestation, protocol),
            ("expected", report, attestation, expected),
        )
        for name, changed_report, changed_attestation, changed_protocol in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation.validate_content_equivalence(
                    changed_report,
                    report,
                    changed_attestation,
                    attestation,
                    changed_protocol,
                    image_id_a="sha256:" + "a" * 64,
                    image_id_b="sha256:" + "a" * 64,
                )

    def test_three_way_extra_missing_version_and_toolchain_omission_reject(
        self,
    ) -> None:
        report, attestation = self._synthetic()
        protocol = evaluation.load_protocol()
        for axis in ("actual", "filesystem", "expected"):
            for mutation in ("missing", "extra", "version", "omit-pip"):
                changed_report = copy.deepcopy(report)
                changed_attestation = copy.deepcopy(attestation)
                changed_protocol = copy.deepcopy(protocol)
                if axis == "actual":
                    rows = changed_attestation["distributions"]
                elif axis == "filesystem":
                    rows = changed_report["filesystem_distributions"]
                else:
                    rows = changed_protocol["expected_distributions"]["distributions"]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "extra":
                    if axis == "filesystem":
                        rows.append(
                            {
                                "metadata_path": "site-packages/extra-1.dist-info/METADATA",
                                "name": "extra",
                                "version": "1",
                                "canonical_name": "extra",
                                "metadata_sha256": "a" * 64,
                            }
                        )
                    elif axis == "expected":
                        rows.append(
                            {
                                "canonical_name": "extra",
                                "origin_class": "ml-runtime-artifact",
                                "version": "1",
                            }
                        )
                    else:
                        rows.append(
                            {"canonical_name": "extra", "name": "extra", "version": "1"}
                        )
                elif mutation == "version":
                    rows[0]["version"] = "0"
                else:
                    rows[:] = [
                        row for row in rows if row["canonical_name"] != "pip"
                    ]
                with self.subTest(axis=axis, mutation=mutation), self.assertRaises(
                    ValueError
                ):
                    evaluation.validate_exact_installed_distributions(
                        changed_report, changed_attestation, changed_protocol
                    )

    def test_extra_missing_version_duplicate_and_toolchain_omission_reject(
        self,
    ) -> None:
        report, attestation = self._synthetic()
        mutations = []
        missing = copy.deepcopy(attestation)
        missing["distributions"].pop()
        mutations.append(("missing", missing))
        extra = copy.deepcopy(attestation)
        extra["distributions"].append(
            {"canonical_name": "extra", "name": "extra", "version": "1"}
        )
        mutations.append(("extra", extra))
        mismatch = copy.deepcopy(attestation)
        mismatch["distributions"][0]["version"] = "0"
        mutations.append(("version", mismatch))
        duplicate = copy.deepcopy(attestation)
        duplicate["distributions"].append(copy.deepcopy(duplicate["distributions"][0]))
        mutations.append(("duplicate", duplicate))
        for tool in ("pip", "setuptools", "wheel"):
            omitted = copy.deepcopy(attestation)
            omitted["distributions"] = [
                row for row in omitted["distributions"] if row["canonical_name"] != tool
            ]
            mutations.append((f"omit-{tool}", omitted))
        for name, changed in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation.validate_content_equivalence(
                    report,
                    report,
                    changed,
                    attestation,
                    evaluation.load_protocol(),
                    image_id_a="sha256:" + "a" * 64,
                    image_id_b="sha256:" + "b" * 64,
                )

    def test_filesystem_metadata_rejects_absent_malformed_and_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = root / "absent-1.dist-info"
            absent.mkdir()
            with self.assertRaises(ValueError):
                content.collect_filesystem_distribution_inventory(root)
            absent.rmdir()
            malformed = root / "malformed-1.dist-info"
            malformed.mkdir()
            (malformed / "METADATA").write_text("Name: malformed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                content.collect_filesystem_distribution_inventory(root)
            (malformed / "METADATA").write_text(
                "Name: Foo.Bar\nVersion: 1\n", encoding="utf-8"
            )
            duplicate = root / "duplicate-1.dist-info"
            duplicate.mkdir()
            (duplicate / "METADATA").write_text(
                "Name: foo_bar\nVersion: 1\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                content.collect_filesystem_distribution_inventory(root)

    def test_filesystem_metadata_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "example-1.dist-info"
            dist.mkdir()
            metadata = dist / "METADATA"
            metadata.write_text("Name: example\nVersion: 1\n", encoding="utf-8")
            original = Path.is_symlink

            def pretend_metadata_symlink(path: Path) -> bool:
                return path == metadata or original(path)

            with mock.patch.object(Path, "is_symlink", pretend_metadata_symlink):
                with self.assertRaises(ValueError):
                    content.collect_filesystem_distribution_inventory(root)

    def test_entry_fingerprint_ignores_enumeration_and_stat_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"stable")
            roots = ((path, PurePosixPath("root/payload")),)
            first = content.collect_entries(roots)
            os.utime(path, (1_700_000_000, 1_700_000_000))
            second = content.collect_entries(reversed(roots))
            self.assertEqual(first, second)
            self.assertEqual(
                set(first[0]), {"path", "type", "content_sha256", "size"}
            )

    def test_top_level_correspondence_binds_regular_file_path_and_digest(self) -> None:
        raw = b"Name: example\nVersion: 1\n"
        path = "site-packages/example-1.dist-info/METADATA"
        inventory = [
            {
                "metadata_path": path,
                "name": "example",
                "version": "1",
                "canonical_name": "example",
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
        report = content.build_report(
            [_file(path, raw)],
            dependency_artifact_registry_sha256="a" * 64,
            expected_distribution_registry_sha256="b" * 64,
            filesystem_distributions=inventory,
            expected_distribution_count=1,
            actual_distribution_count=1,
            python_identity={
                "implementation": "CPython",
                "version": "3.11.15",
                "abi": "cp311",
            },
        )
        diagnostic = report["metadata_correspondence"]
        self.assertEqual(diagnostic["missing_normalized_metadata_paths"], [])
        self.assertEqual(
            diagnostic["extra_normalized_top_level_metadata_paths"], []
        )
        self.assertEqual(diagnostic["metadata_digest_mismatches"], [])
        self.assertEqual(diagnostic["nested_metadata_paths"], [])
        self.assertEqual(
            report["metadata_correspondence_sha256"],
            content.sha256_bytes(content.canonical_json_bytes(diagnostic)),
        )
        content.validate_report(report)

    def test_nested_metadata_remains_fingerprinted_but_not_installed(self) -> None:
        raw = b"Name: example\nVersion: 1\n"
        top = "site-packages/example-1.dist-info/METADATA"
        nested = "site-packages/vendor/nested-1.dist-info/METADATA"
        inventory = [
            {
                "metadata_path": top,
                "name": "example",
                "version": "1",
                "canonical_name": "example",
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
        report = content.build_report(
            [_file(nested, b"nested"), _file(top, raw)],
            dependency_artifact_registry_sha256="a" * 64,
            expected_distribution_registry_sha256="b" * 64,
            filesystem_distributions=inventory,
            expected_distribution_count=1,
            actual_distribution_count=1,
            python_identity={
                "implementation": "CPython",
                "version": "3.11.15",
                "abi": "cp311",
            },
        )
        diagnostic = report["metadata_correspondence"]
        self.assertEqual(diagnostic["nested_metadata_paths"], [nested])
        self.assertEqual(diagnostic["nested_metadata_count"], 1)
        self.assertIn(nested, [row["path"] for row in report["normalized_entries"]])
        self.assertNotIn(
            nested, [row["metadata_path"] for row in report["filesystem_distributions"]]
        )

    def test_missing_extra_digest_and_non_regular_reject_with_sorted_diagnostic(
        self,
    ) -> None:
        raw = b"Name: a\nVersion: 1\n"
        inv_path = "site-packages/z-1.dist-info/METADATA"
        extra_a = "site-packages/a-1.dist-info/METADATA"
        extra_b = "site-packages/b-1.dist-info/METADATA"
        inventory = [
            {
                "metadata_path": inv_path,
                "name": "z",
                "version": "1",
                "canonical_name": "z",
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
        cases = (
            ([], "missing_normalized_metadata_paths"),
            ([_file(inv_path, b"different")], "metadata_digest_mismatches"),
            (
                [_file(extra_b), _file(extra_a), _file(inv_path, raw)],
                "extra_normalized_top_level_metadata_paths",
            ),
            (
                [
                    {
                        "path": inv_path,
                        "type": "symlink",
                        "symlink_target": "x",
                        "size": 1,
                    }
                ],
                "missing_normalized_metadata_paths",
            ),
        )
        for entries, key in cases:
            with self.subTest(key=key), self.assertRaises(
                content.MetadataCorrespondenceError
            ) as caught:
                content.build_report(
                    entries,
                    dependency_artifact_registry_sha256="a" * 64,
                    expected_distribution_registry_sha256="b" * 64,
                    filesystem_distributions=inventory,
                    expected_distribution_count=1,
                    actual_distribution_count=1,
                )
            self.assertTrue(caught.exception.diagnostic[key])
            for value in caught.exception.diagnostic[key]:
                if isinstance(value, dict):
                    continue
                self.assertEqual(
                    caught.exception.diagnostic[key],
                    sorted(
                        caught.exception.diagnostic[key],
                        key=lambda item: item.encode("utf-8"),
                    ),
                )
                break
            canonical = content.canonical_json_bytes(caught.exception.diagnostic)
            self.assertEqual(
                json.loads(canonical.decode("utf-8")), caught.exception.diagnostic
            )

    def test_metadata_path_grammar_duplicate_case_collision_and_traversal_reject(
        self,
    ) -> None:
        base = {
            "name": "example",
            "version": "1",
            "canonical_name": "example",
            "metadata_sha256": "a" * 64,
        }
        for path in (
            "site-packages/.dist-info/METADATA",
            "site-packages/nested/example-1.dist-info/METADATA",
            "site-packages/example-1.dist-info/../METADATA",
            "site-packages\\example-1.dist-info\\METADATA",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                content.normalize_distribution_inventory(
                    [{**base, "metadata_path": path}]
                )
        exact = {**base, "metadata_path": "site-packages/example-1.dist-info/METADATA"}
        with self.assertRaises(ValueError):
            content.normalize_distribution_inventory([exact, exact])
        folded = {
            **base,
            "metadata_path": "site-packages/EXAMPLE-1.dist-info/METADATA",
        }
        with self.assertRaises(ValueError):
            content.normalize_distribution_inventory([exact, folded])

    def test_content_path_symlink_algorithm_base_and_registry_tamper_reject(
        self,
    ) -> None:
        report, attestation = self._synthetic()
        for key, value in (
            ("algorithm_version", "v1"),
            ("base_digest", "sha256:" + "f" * 64),
            ("expected_distribution_registry_sha256", "f" * 64),
        ):
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                content.validate_report(changed)
        for entries in (
            [_file("root/a"), _file("root/a")],
            [_file("root/../a")],
            [_file("root/a"), _file("ROOT/A")],
            [
                {
                    "path": "root/link",
                    "type": "symlink",
                    "symlink_target": "a",
                    "size": 2,
                }
            ],
        ):
            with self.assertRaises(ValueError):
                content.normalize_entries(entries)
        changed = copy.deepcopy(report)
        changed["normalized_entries"][0]["size"] += 1
        with self.assertRaises(ValueError):
            evaluation.validate_content_equivalence(
                report,
                changed,
                attestation,
                attestation,
                evaluation.load_protocol(),
                image_id_a="sha256:" + "a" * 64,
                image_id_b="sha256:" + "a" * 64,
            )

    def test_v7_v8_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        v8 = evaluation.load_protocol()
        for key, suffix in (
            ("corpus", "corpus"),
            ("queries", "queries"),
            ("gold", "gold"),
            ("models", "models"),
            ("candidates", "candidates"),
            ("gate", "gate"),
            ("result_schema", "result-schema"),
        ):
            expected = json.loads(
                (
                    ROOT
                    / f"tests/fixtures/github_cross_encoder_precision_v7.{suffix}.json"
                ).read_text(encoding="utf-8")
            )
            expected["protocol_id"] = evaluation.PROTOCOL_ID
            self.assertEqual(expected, v8[key], key)
        v7_lock = (
            ROOT / "tests/fixtures/github_cross_encoder_precision_v7.requirements.lock"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            evaluation._locked_versions(v8["requirements_lock"]),
            evaluation._locked_versions(v7_lock),
        )
        for relative, expected_hash in v8["manifest"]["v7_immutable_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected_hash,
            )

    def test_frozen_json_is_strict_utf8_without_mojibake(self) -> None:
        paths = list((ROOT / "tests/fixtures").glob(f"{evaluation.STEM}.*.json"))
        evidence = ROOT / "tests/evidence/github_cross_encoder_precision_v8"
        if evidence.exists():
            paths.extend(evidence.glob("*.json"))
        for path in paths:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="strict")
            json.loads(text)
            self.assertNotIn("\ufffd", text)
            self.assertNotIn(b"\r", raw)


if __name__ == "__main__":
    unittest.main()

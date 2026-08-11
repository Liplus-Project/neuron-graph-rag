from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.canonical_gate_evaluation import (
    _rollback_probe,
    _trajectory,
    canonical_gate_ids,
    prove_writer_verifier_round_trip,
    read_json,
    validate_gate_array,
    verify_identity_only_registry,
    verify_registered_result,
    write_observed_exclusive,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class CanonicalGateEvaluationTest(unittest.TestCase):
    @staticmethod
    def _placeholder_case() -> dict[str, object]:
        prefix = "roundtrip-probe"
        source = f"{prefix}-a-source"
        target = f"{prefix}-target"
        sibling = f"{prefix}-sibling"
        reverse = f"{prefix}-y-reverse"
        unrelated = f"{prefix}-u-unrelated"
        unrelated_target = f"{prefix}-v-unrelated-target"
        nodes = [
            (source, "placeholderquery relation seed"),
            (target, "credited placeholder conclusion"),
            (sibling, "sibling placeholder conclusion"),
            (f"{prefix}-z-direct", "placeholderquery direct lexical control"),
            (reverse, "reverse control"),
            (unrelated, "unrelated origin"),
            (unrelated_target, "unrelated target"),
        ]
        return {
            "case_id": "temporary-placeholder-case",
            "stage": "temporary",
            "stratum": "headroom",
            "query": "placeholderquery",
            "target_node_id": target,
            "direct_node_id": f"{prefix}-z-direct",
            "reverse_node_id": reverse,
            "nodes": [
                {
                    "node_id": node_id,
                    "text": text,
                    "source_url": f"https://example.invalid/temporary/{node_id}",
                }
                for node_id, text in nodes
            ],
            "edges": [
                {"source_id": source, "target_id": target, "edge_type": "probe_edge", "weight": 1.49},
                {"source_id": source, "target_id": sibling, "edge_type": "probe_edge", "weight": 1.55},
                {"source_id": unrelated, "target_id": unrelated_target, "edge_type": "probe_isolated", "weight": 0.73},
                {"source_id": reverse, "target_id": source, "edge_type": "probe_reverse", "weight": 0.67},
            ],
            "credited_edge": {"source_id": source, "target_id": target, "edge_type": "probe_edge"},
            "sibling_edge": {"source_id": source, "target_id": sibling, "edge_type": "probe_edge"},
            "unrelated_edge": {"source_id": unrelated, "target_id": unrelated_target, "edge_type": "probe_isolated"},
            "reverse_edge": {"source_id": reverse, "target_id": source, "edge_type": "probe_reverse"},
        }

    def test_registered_gate_ids_are_non_alphabetical_and_unique(self) -> None:
        gate_ids = canonical_gate_ids(FIXTURES / "canonical_evidence_gate_v1.gate.json")
        self.assertNotEqual(gate_ids, sorted(gate_ids))
        self.assertEqual(len(gate_ids), len(set(gate_ids)))

    def test_actual_writer_and_verifier_preserve_non_alphabetical_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prove_writer_verifier_round_trip(Path(directory) / "placeholder.observed.json")

    def test_exclusive_writer_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.observed.json"
            sentinel = b"existing immutable result\n"
            path.write_bytes(sentinel)
            with self.assertRaises(FileExistsError):
                write_observed_exclusive(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), sentinel)

    def test_verifier_rejects_mapping_reordering_duplicates_and_failure(self) -> None:
        expected = ["zulu", "alpha", "mike"]
        with self.assertRaises(ValueError):
            validate_gate_array(
                {gate_id: True for gate_id in expected}, expected, require_all_passed=True
            )
        with self.assertRaises(ValueError):
            validate_gate_array(
                [{"gate_id": item, "passed": True} for item in sorted(expected)],
                expected,
                require_all_passed=True,
            )
        with self.assertRaises(ValueError):
            validate_gate_array(
                [
                    {"gate_id": "zulu", "passed": True},
                    {"gate_id": "alpha", "passed": True},
                    {"gate_id": "alpha", "passed": True},
                ],
                expected,
                require_all_passed=True,
            )
        with self.assertRaises(ValueError):
            validate_gate_array(
                [
                    {"gate_id": "zulu", "passed": True},
                    {"gate_id": "alpha", "passed": False},
                    {"gate_id": "mike", "passed": True},
                ],
                expected,
                require_all_passed=True,
            )

    def test_registered_artifacts_are_result_free_or_immutable(self) -> None:
        manifest = read_json(FIXTURES / "canonical_evidence_gate_v1.manifest.json")
        manifest_relative = "tests/fixtures/canonical_evidence_gate_v1.manifest.json"
        frozen_commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", manifest_relative],
            cwd=ROOT,
            text=True,
        ).strip()
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{frozen_commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(
            commit_check.returncode,
            0,
            "the manifest-introducing commit must be present; CI uses fetch-depth 0",
        )
        evolving_surfaces = {
            "README.md",
            "tests/test_canonical_gate_evaluation.py",
        }
        for relative, expected_hash in manifest["artifacts"].items():
            if relative in evolving_surfaces:
                frozen_bytes = subprocess.check_output(
                    ["git", "show", f"{frozen_commit}:{relative}"],
                    cwd=ROOT,
                )
                actual = hashlib.sha256(frozen_bytes).hexdigest()
            else:
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected_hash, relative)
        for stage, relative in manifest["outputs"].items():
            output = ROOT / relative
            if output.exists():
                verify_registered_result(stage)

    def test_identity_audit_is_recomputed_from_identity_fields_only(self) -> None:
        fixture = read_json(FIXTURES / "canonical_evidence_gate_v1.fixture.json")
        registry = read_json(
            FIXTURES / "canonical_evidence_gate_v1.identity-registry.json"
        )
        audit = read_json(FIXTURES / "canonical_evidence_gate_v1.audit.json")
        checks = verify_identity_only_registry(fixture, registry)
        self.assertEqual(checks, audit["checks"])
        self.assertTrue(all(checks.values()))

    def test_placeholder_mechanics_cross_quorum_at_registered_boundary(self) -> None:
        case = self._placeholder_case()
        current = {"evidence_quorum": 1, "sibling_normalization": 0.0}
        combined = {"evidence_quorum": 3, "sibling_normalization": 1.0}
        baseline = _trajectory(case, combined, 0)
        first = _trajectory(case, current, 1)
        prequorum = _trajectory(case, combined, 2)
        threshold = _trajectory(case, combined, 3)
        self.assertEqual(first["ranking"]["target_rank"], 1)
        self.assertEqual(prequorum["after"], baseline["after"])
        self.assertEqual(prequorum["ranking"]["target_rank"], 2)
        self.assertEqual(threshold["ranking"]["target_rank"], 1)
        self.assertTrue(_rollback_probe(case, sibling_failure=False))
        self.assertTrue(_rollback_probe(case, sibling_failure=True))


if __name__ == "__main__":
    unittest.main()

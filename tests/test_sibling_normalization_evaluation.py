from __future__ import annotations

import unittest
from pathlib import Path

from neuron_graph_rag.sibling_normalization_evaluation import (
    validate_observed_outputs,
    validate_protocol,
)


class SiblingNormalizationEvaluationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_registered_artifacts_and_source_match_manifest(self) -> None:
        validated = validate_protocol(self.repo_root)

        self.assertEqual(
            validated["manifest"]["protocol"],
            "sibling-normalization-controlled-v1",
        )
        self.assertEqual(
            validated["manifest"]["evaluated_source"]["sha256"],
            validated["source_hash"],
        )

    def test_observed_output_state_respects_stage_order_and_hashes(self) -> None:
        validate_observed_outputs(self.repo_root)

    def test_only_registered_exclusive_outputs_use_protocol_output_prefix(self) -> None:
        manifest = validate_protocol(self.repo_root)["manifest"]
        expected = {
            str(path) for path in manifest["exclusive_outputs"].values()
        }
        actual = {
            str(path.relative_to(self.repo_root)).replace("\\", "/")
            for path in (self.repo_root / "tests" / "fixtures").glob(
                "sibling_normalization_controlled_v1.*.observed.json"
            )
        }

        self.assertTrue(actual <= expected)


if __name__ == "__main__":
    unittest.main()

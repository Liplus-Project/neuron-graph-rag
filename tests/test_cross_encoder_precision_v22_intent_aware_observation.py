from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from neuron_graph_rag import (
    cross_encoder_precision_v22_intent_aware_observation as scaffold,
)


class CrossEncoderPrecisionV22IntentAwareObservationTests(unittest.TestCase):
    def test_result_free_scaffold_validates_without_performance_execution(self) -> None:
        result = scaffold.validate_result_free()
        self.assertEqual(result["status"], "result-free-engine-scaffold-valid")
        self.assertEqual(result["performance"], "not assessed")
        self.assertEqual(result["model_execution_count"], 0)
        self.assertEqual(result["result_count"], 0)
        self.assertTrue(result["v21_source_and_evidence_immutable"])
        self.assertEqual(result["v21_protocol_artifact_count"], 9)
        self.assertEqual(result["v21_evidence_artifact_count"], 29)

    def test_composition_injects_protocol_paths_stages_and_models(self) -> None:
        spec = scaffold.ENGINE_SPEC
        self.assertEqual(spec.protocol_id, "github-ngr-cross-encoder-precision-v22")
        self.assertIn("v22", spec.fixture_paths.corpus.name)
        self.assertIn("v22", spec.fixture_paths.queries.name)
        self.assertIn("v22", spec.fixture_paths.gold.name)
        self.assertNotEqual(
            spec.stage_identity("development"), spec.stage_identity("holdout")
        )
        self.assertEqual([model.kind for model in spec.models], ["base", "v2-m3"])

    def test_scaffold_exposes_validate_only(self) -> None:
        source = inspect.getsource(scaffold.main)
        self.assertIn('choices=("validate",)', source)
        self.assertNotIn('choices=("run",)', source)
        self.assertNotIn("preflight", source)

    def test_v22_evidence_is_absent(self) -> None:
        contract = json.loads(
            (scaffold.ROOT / scaffold.CONTRACT).read_text(encoding="utf-8")
        )
        self.assertFalse((scaffold.ROOT / contract["evidence_path"]).exists())

    def test_composition_module_is_thin(self) -> None:
        source_path = Path(scaffold.__file__)
        self.assertLessEqual(len(source_path.read_text(encoding="utf-8").splitlines()), 130)

    def test_architecture_docs_keep_result_and_claim_boundaries(self) -> None:
        text = (
            scaffold.ROOT / "docs/intent-aware-observation-engine.md"
        ).read_text(encoding="utf-8")
        self.assertIn("workerはgoldを受け取らない", text)
        self.assertIn("protocol validity", text)
        self.assertIn("Issue #196", text)
        self.assertIn("performance evidence", text)
        self.assertNotIn("## Completion", text)


if __name__ == "__main__":
    unittest.main()

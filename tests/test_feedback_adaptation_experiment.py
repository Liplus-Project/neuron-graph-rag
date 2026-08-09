from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.feedback_adaptation_experiment import (
    _canonical_sha256,
    read_feedback_adaptation_manifest,
    run_feedback_adaptation_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_feedback_adaptation_experiment.manifest.json"
DEVELOPMENT_RESULT = FIXTURES / "d1_liplus_feedback_adaptation_experiment.development.result.json"
HOLDOUT_RESULT = FIXTURES / "d1_liplus_feedback_adaptation_experiment.holdout.result.json"


class FeedbackAdaptationFreezeTest(unittest.TestCase):
    def test_manifest_freezes_disjoint_split_and_registered_runs(self) -> None:
        manifest = read_feedback_adaptation_manifest(MANIFEST)
        self.assertEqual(manifest["candidate_id"], "trace-credited-feedback")
        self.assertEqual(manifest["registered_runs"]["development"], {"control": 1, "treatment": 2})
        self.assertTrue(manifest["registered_runs"]["holdout"]["conditional_on_development_gate"])
        development = json.loads((FIXTURES / manifest["development"]["gold"]).read_text(encoding="utf-8"))
        holdout = json.loads((FIXTURES / manifest["holdout"]["gold"]).read_text(encoding="utf-8"))
        self.assertNotEqual(development["feedback"]["query"], development["scoring_cases"][0]["query"])
        self.assertFalse({case["query"] for case in development["scoring_cases"]} & {case["query"] for case in holdout["scoring_cases"]})

    def test_result_files_are_exclusive_and_observed_files_are_not_recomputed(self) -> None:
        self.assertFalse(HOLDOUT_RESULT.exists() and not DEVELOPMENT_RESULT.exists())
        for path, stage in ((DEVELOPMENT_RESULT, "development"), (HOLDOUT_RESULT, "holdout")):
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["stage"], stage)
                self.assertEqual(result["manifest_sha256"], _canonical_sha256(MANIFEST))

    def test_failed_development_cannot_open_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed = Path(directory) / "development.json"
            failed.write_text(json.dumps({"manifest_sha256": _canonical_sha256(MANIFEST), "gate_passed": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stop rule forbids opening"):
                run_feedback_adaptation_holdout(MANIFEST, failed)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.feedback_adaptation_reproduction import (
    PATH_IDENTITY_FIELDS,
    _canonical_sha256,
    _experiment_config,
    project_relation_path,
    read_feedback_adaptation_reproduction_manifest,
    run_feedback_adaptation_reproduction_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_feedback_adaptation_reproduction_experiment.manifest.json"
DEVELOPMENT_RESULT = FIXTURES / "d1_liplus_feedback_adaptation_reproduction_experiment.development.result.json"
HOLDOUT_RESULT = FIXTURES / "d1_liplus_feedback_adaptation_reproduction_experiment.holdout.result.json"


class FeedbackAdaptationReproductionFreezeTest(unittest.TestCase):
    def test_path_identity_projects_away_runtime_fields(self) -> None:
        raw_step = {
            "source_id": "source",
            "target_id": "target",
            "edge_type": "mention",
            "edge_weight": 1.25,
            "factuality": 0.8,
            "activation": 0.5,
            "runtime_trace_id": "trace-ignored",
        }
        self.assertEqual(
            project_relation_path([raw_step]),
            [{"source_id": "source", "target_id": "target", "edge_type": "mention"}],
        )
        self.assertEqual(set(project_relation_path([raw_step])[0]), set(PATH_IDENTITY_FIELDS))

    def test_manifest_freezes_disjoint_split_and_registered_runs(self) -> None:
        manifest = read_feedback_adaptation_reproduction_manifest(MANIFEST)
        self.assertEqual(manifest["candidate_id"], "trace-credited-feedback-reproduction")
        self.assertEqual(manifest["registered_runs"]["development"], {"control": 1, "treatment": 2})
        self.assertTrue(manifest["registered_runs"]["holdout"]["conditional_on_development_gate"])
        development = json.loads((FIXTURES / manifest["development"]["gold"]).read_text(encoding="utf-8"))
        holdout = json.loads((FIXTURES / manifest["holdout"]["gold"]).read_text(encoding="utf-8"))
        self.assertFalse({case["query"] for case in development["scoring_cases"]} & {case["query"] for case in holdout["scoring_cases"]})

    def test_result_files_are_exclusive_before_observation(self) -> None:
        self.assertFalse(DEVELOPMENT_RESULT.exists())
        self.assertFalse(HOLDOUT_RESULT.exists())

    def test_failed_development_cannot_open_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed = Path(directory) / "development.json"
            failed.write_text(
                json.dumps({"manifest_sha256": _canonical_sha256(MANIFEST), "gate_passed": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Stop rule forbids opening"):
                run_feedback_adaptation_reproduction_holdout(MANIFEST, failed)

    def test_limit_is_a_runner_setting_not_an_engine_config_field(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        config, limit = _experiment_config(manifest)
        self.assertEqual(limit, 2)
        self.assertEqual(config.seed_count, 1)


if __name__ == "__main__":
    unittest.main()

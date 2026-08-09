from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.longitudinal_feedback_experiment import (
    HORIZONS,
    PATH_IDENTITY_FIELDS,
    _canonical_sha256,
    _experiment_config,
    project_relation_path,
    read_longitudinal_feedback_manifest,
    run_longitudinal_feedback_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_longitudinal_feedback_experiment.manifest.json"
DEVELOPMENT_RESULT = FIXTURES / "d1_liplus_longitudinal_feedback_experiment.development.result.json"


class LongitudinalFeedbackFreezeTest(unittest.TestCase):
    def test_path_identity_projects_away_runtime_fields(self) -> None:
        raw_step = {"source_id": "source", "target_id": "target", "edge_type": "supports", "edge_weight": 1.2, "factuality": 0.9, "activation": 0.4, "runtime_trace_id": "ignored"}
        projected = project_relation_path([raw_step])
        self.assertEqual(projected, [{"source_id": "source", "target_id": "target", "edge_type": "supports"}])
        self.assertEqual(set(projected[0]), set(PATH_IDENTITY_FIELDS))

    def test_manifest_freezes_six_disjoint_clusters_and_horizons(self) -> None:
        manifest = read_longitudinal_feedback_manifest(MANIFEST)
        clusters = [cluster for stage in ("development", "holdout") for cluster in manifest[stage]["clusters"]]
        self.assertEqual(len(clusters), 6)
        self.assertEqual({cluster["cohort"] for cluster in clusters}, {"headroom", "ceiling"})
        self.assertEqual(manifest["registered_runs"]["development"]["horizons"], list(HORIZONS))
        self.assertTrue(manifest["registered_runs"]["holdout"]["conditional_on_development_gate"])
        audit = json.loads((FIXTURES / manifest["contamination_audit"]["artifact"]).read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["prior_usage"], "fixture identifiers only; prior gold and observed result artifacts are not loaded")

    def test_registered_runner_refuses_unregistered_or_existing_output_without_evaluation(self) -> None:
        environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
        unregistered = subprocess.run([sys.executable, "tools/run_longitudinal_feedback_experiment.py", "development", "--manifest", str(MANIFEST), "--output", "unregistered.json"], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
        self.assertNotEqual(unregistered.returncode, 0)
        self.assertIn("Refusing unregistered output path", unregistered.stderr)
        if DEVELOPMENT_RESULT.exists():
            existing = subprocess.run([sys.executable, "tools/run_longitudinal_feedback_experiment.py", "development", "--manifest", str(MANIFEST), "--output", str(DEVELOPMENT_RESULT)], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("Refusing to overwrite an observed result", existing.stderr)

    def test_failed_development_cannot_open_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed = Path(directory) / "development.json"
            failed.write_text(json.dumps({"manifest_sha256": _canonical_sha256(MANIFEST), "gate_passed": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stop rule forbids opening"):
                run_longitudinal_feedback_holdout(MANIFEST, failed)

    def test_limit_is_a_runner_setting_not_an_engine_config_field(self) -> None:
        config, limit = _experiment_config(json.loads(MANIFEST.read_text(encoding="utf-8")))
        self.assertEqual(limit, 2)
        self.assertEqual(config.seed_count, 1)


if __name__ == "__main__":
    unittest.main()

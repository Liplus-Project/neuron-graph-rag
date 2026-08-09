from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.engine_feedback_trajectory import (
    _nondecreasing,
    _project_path,
    _read_frozen_artifacts,
    _verify_source_integrity,
    read_engine_feedback_trajectory_manifest,
    run_engine_feedback_trajectory_holdout,
    write_engine_feedback_trajectory_result,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "engine_feedback_trajectory_v3.manifest.json"


class EngineFeedbackTrajectoryFreezeTest(unittest.TestCase):
    def test_result_free_registration_is_complete(self) -> None:
        manifest = read_engine_feedback_trajectory_manifest(MANIFEST)

        self.assertEqual(
            manifest["source_corpus_commit"],
            "94c8bc250b7352e3009eeee1b353c3aec677bfb7",
        )
        self.assertEqual(manifest["checkpoints"], [0, 1, 3, 10])
        self.assertEqual(
            manifest["registered_runs"]["development"],
            {"control": 1, "treatment": 1},
        )
        self.assertTrue(
            manifest["registered_runs"]["holdout"][
                "conditional_on_development_gate"
            ]
        )
        self.assertEqual(
            set(manifest["result_paths"]), {"development", "holdout"}
        )

    def test_observed_development_is_immutable_and_holdout_remains_unopened(self) -> None:
        manifest = read_engine_feedback_trajectory_manifest(MANIFEST)
        development = FIXTURES / manifest["result_paths"]["development"]
        holdout = FIXTURES / manifest["result_paths"]["holdout"]

        self.assertTrue(development.exists())
        self.assertEqual(
            "sha256:" + hashlib.sha256(development.read_bytes()).hexdigest(),
            "sha256:ec3cb4f6bde411ed06a0e8d62cfbf04897438dd9b9d340f21f50a0a5945f46b0",
        )
        observed = json.loads(development.read_text(encoding="utf-8"))
        self.assertEqual(observed["stage"], "development")
        self.assertFalse(observed["gate_passed"])
        self.assertEqual(
            observed["holdout_status"],
            "not_opened_development_gate_failed",
        )
        self.assertFalse(holdout.exists())

    def test_source_hashes_and_explicit_links_match_the_frozen_split(self) -> None:
        manifest = read_engine_feedback_trajectory_manifest(MANIFEST)
        artifacts = _read_frozen_artifacts(MANIFEST, manifest)

        for stage, expected_count in (("development", 10), ("holdout", 5)):
            integrity = _verify_source_integrity(
                MANIFEST, artifacts["fixture"], stage
            )
            self.assertTrue(integrity["passed"])
            self.assertEqual(len(integrity["checks"]), expected_count)
            self.assertTrue(integrity["explicit_links"]["passed"])

    def test_path_projection_ignores_runtime_fields(self) -> None:
        path = [
            {
                "source_id": "source",
                "target_id": "target",
                "edge_type": "explicit_link",
                "edge_weight": 1.7,
                "factuality": 0.9,
            }
        ]
        self.assertEqual(
            _project_path(path),
            (("source", "target", "explicit_link"),),
        )

    def test_non_regression_is_monotonic_and_allows_ties(self) -> None:
        self.assertTrue(_nondecreasing([0.25, 0.25, 0.5, 1.0]))
        self.assertFalse(_nondecreasing([0.25, 0.5, 0.4, 1.0]))

    def test_holdout_cannot_open_after_failed_development(self) -> None:
        manifest = read_engine_feedback_trajectory_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            failed = Path(directory) / "development.json"
            failed.write_text(
                json.dumps(
                    {
                        "stage": "development",
                        "manifest_sha256": "not-the-frozen-manifest",
                        "gate_passed": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "registered development output"):
                run_engine_feedback_trajectory_holdout(MANIFEST, failed)
        self.assertFalse(
            (FIXTURES / manifest["result_paths"]["holdout"]).exists()
        )

    def test_result_writer_never_overwrites_observed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observed.json"
            output.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                write_engine_feedback_trajectory_result(output, {"gate_passed": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()

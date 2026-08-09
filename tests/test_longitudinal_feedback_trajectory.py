from __future__ import annotations

import unittest
from pathlib import Path

from neuron_graph_rag.longitudinal_feedback_trajectory import (
    ARTIFACTS,
    MANIFEST,
    OBSERVED,
    PROTOCOL,
    load_frozen_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LongitudinalFeedbackTrajectoryFreezeTest(unittest.TestCase):
    def test_v3_protocol_is_result_free_and_identity_isolated_before_execution(self) -> None:
        artifacts = load_frozen_protocol(REPOSITORY_ROOT)
        fixture = artifacts["longitudinal_feedback_trajectory_v3.fixture.json"]

        self.assertEqual(fixture["source_commit"], "94c8bc250b7352e3009eeee1b353c3aec677bfb7")
        self.assertEqual(set(fixture["roles"]), {"development", "holdout", "trajectory_audit"})
        self.assertEqual({role["cluster"] for role in fixture["roles"].values()}, {
            "signal-stability", "boundary-recovery", "evidence-continuity"
        })
        self.assertEqual({artifact["protocol"] for artifact in artifacts.values() if isinstance(artifact, dict) and "protocol" in artifact}, {PROTOCOL})

    def test_manifest_freezes_only_new_protocol_artifacts_and_output_paths(self) -> None:
        fixture_root = REPOSITORY_ROOT / "tests" / "fixtures"
        manifest = load_frozen_protocol(REPOSITORY_ROOT)[MANIFEST]

        self.assertEqual(set(manifest["artifact_hashes"]), set(ARTIFACTS))
        self.assertEqual(set(manifest["exclusive_output_paths"]), set(OBSERVED.values()))
        self.assertTrue(all(path.startswith("longitudinal_feedback_trajectory_v3.") for path in manifest["exclusive_output_paths"]))
        self.assertTrue(all((fixture_root / name).is_file() for name in ARTIFACTS))


if __name__ == "__main__":
    unittest.main()

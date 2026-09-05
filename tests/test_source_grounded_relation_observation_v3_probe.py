from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "probe_source_grounded_relation_observation_v3.py"


class SourceGroundedRelationObservationV3ProbeTests(unittest.TestCase):
    def test_exact_frozen_module_passes_before_and_after_observation(self) -> None:
        environment = dict(os.environ)
        python_path = str(ROOT / "src")
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        completed = subprocess.run(
            [sys.executable, str(PROBE), "--root", str(ROOT)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            self.fail(completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["status"], "whole-module-two-state-probe-valid"
        )
        outcomes = payload["outcomes"]
        self.assertEqual(outcomes["result_free"]["phase"], "result-free")
        observed = outcomes["synthetic_post_observation"]
        self.assertEqual(
            observed["development_closed"]["phase"], "development-closed"
        )
        self.assertEqual(
            observed["holdout_eligible"]["phase"], "holdout-eligible"
        )
        self.assertEqual(payload["real_queries_executed"], 0)
        self.assertIs(payload["shared_database_opened"], False)
        self.assertEqual(payload["persistent_artifacts_created"], 0)


if __name__ == "__main__":
    unittest.main()

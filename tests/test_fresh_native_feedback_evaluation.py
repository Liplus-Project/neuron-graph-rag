from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.fresh_native_feedback_evaluation import (
    _read_frozen_artifacts,
    _verify_source_integrity,
    read_fresh_native_feedback_manifest,
    run_fresh_native_feedback_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "fresh_native_feedback_v2.manifest.json"


class FreshNativeFeedbackFreezeTest(unittest.TestCase):
    def test_result_free_artifacts_are_frozen_and_disjoint(self) -> None:
        manifest = read_fresh_native_feedback_manifest(MANIFEST)

        self.assertEqual(manifest["candidate_id"], "fresh-native-credited-feedback-v2")
        self.assertEqual(manifest["registered_runs"]["development"], {"control": 1, "treatment": 1})
        self.assertTrue(manifest["registered_runs"]["holdout"]["conditional_on_development_gate"])
        for stage, name in manifest["result_paths"].items():
            self.assertTrue(name.startswith("fresh_native_feedback_v2."))
            path = FIXTURES / name
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["stage"], stage)
                self.assertEqual(result["manifest_sha256"], _manifest_hash())

    def test_portable_source_integrity_accepts_the_frozen_corpus(self) -> None:
        manifest = read_fresh_native_feedback_manifest(MANIFEST)
        artifacts = _read_frozen_artifacts(MANIFEST, manifest)

        integrity = _verify_source_integrity(MANIFEST, artifacts["fixture"])

        self.assertTrue(integrity["passed"])
        self.assertEqual(len(integrity["checks"]), 8)
        self.assertTrue(all(item["decision"].startswith(("raw_match", "newline_equivalent")) for item in integrity["checks"]))

    def test_holdout_cannot_open_after_a_failed_development_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed = Path(directory) / "development.json"
            failed.write_text(
                json.dumps({"stage": "development", "manifest_sha256": _manifest_hash(), "gate_passed": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Stop rule forbids opening"):
                run_fresh_native_feedback_holdout(MANIFEST, failed)

    def test_runner_never_overwrites_an_observed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observed.json"
            output.write_text("{}\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/run_fresh_native_feedback_evaluation.py",
                    "development",
                    "--manifest",
                    str(MANIFEST),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Refusing to overwrite an observed result", completed.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "{}\n")


def _manifest_hash() -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()

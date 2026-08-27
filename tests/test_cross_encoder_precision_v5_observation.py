from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import cross_encoder_precision_v5_observation as observation


class CrossEncoderPrecisionV5ObservationTest(unittest.TestCase):
    def test_frozen_identity_and_worker_order_are_exact(self) -> None:
        self.assertEqual(
            observation.PROTOCOL_COMMIT,
            "d5c25d7998d634cac0aa96511f59a9cce0b7725a",
        )
        self.assertEqual(observation.WSLC_VERSION, "2.9.4.0")
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v5-runtime"
        )
        self.assertEqual(
            observation.WORKERS,
            (
                ("baseline", "primary"),
                ("baseline", "replay"),
                ("base", "primary"),
                ("base", "replay"),
                ("v2-m3", "primary"),
                ("v2-m3", "replay"),
            ),
        )

    def test_container_command_is_offline_volume_scoped_and_fresh(self) -> None:
        command = observation._container_command(
            "worker",
            "--stage",
            "development",
            name="ngr-v5-development-baseline-primary",
        )
        self.assertIn("--rm", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn(
            f"{observation.VOLUME}:{observation.CONTAINER_ROOT}", command
        )
        self.assertNotIn(str(observation.SHARED_DATABASE), command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)

    def test_source_initialization_is_exclusive_and_distinct(self) -> None:
        script = observation._source_initialization_script()
        self.assertIn("test ! -e /opt/ngr-v5/runtime/source", script)
        self.assertIn("/opt/ngr-v5/runtime/databases", script)
        self.assertIn("/opt/ngr-v5/runtime/runs", script)
        self.assertNotIn("knowledge.db", script)

    def test_verify_preflight_rejects_nonzero_observation_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            preclaim = {"status": "clean"}
            model = {"models": []}
            dependency = {"artifact_count": 26}
            platform_report = {"wslc": observation.WSLC_VERSION}
            report = {
                "protocol_id": observation.PROTOCOL_ID,
                "protocol_commit": observation.PROTOCOL_COMMIT,
                "development_claim_count": 1,
                "registered_query_execution_count": 0,
                "observed_stage_inference_count": 0,
                "result_count": 0,
                "phase": {"development": "unobserved", "holdout": "unobserved"},
                "preclaim_sha256": observation.canonical_sha256(preclaim),
                "model_report_sha256": observation.canonical_sha256(model),
                "dependency_report_sha256": observation.canonical_sha256(dependency),
                "platform_report_sha256": observation.canonical_sha256(
                    platform_report
                ),
            }
            for name, value in (
                ("preclaim.json", preclaim),
                ("model-verification.json", model),
                ("dependency-report.json", dependency),
                ("platform-report.json", platform_report),
                ("preflight.json", report),
                ("preflight-commands.json", {"commands": [{"returncode": 0}]}),
            ):
                (evidence / name).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count"):
                observation.verify_preflight(root)

    def test_development_failure_never_opens_holdout_or_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64
                    },
                ),
                patch.object(observation, "_git_output", return_value="b" * 40),
                patch.object(
                    observation,
                    "_remote_ci_green",
                    return_value={"preflight_evidence_commit": "b" * 40},
                ),
                patch.object(observation, "_sync_preflight_evidence"),
                patch.object(
                    observation,
                    "_hash_shared_database",
                    return_value="a" * 64,
                ),
                patch.object(
                    observation,
                    "_run_stage_host",
                    side_effect=RuntimeError("one-shot failure"),
                ) as stage,
                patch.object(observation, "_run_logged"),
                patch.object(observation, "_export_volume_evidence"),
                self.assertRaisesRegex(RuntimeError, "one-shot failure"),
            ):
                observation.run_once(root)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[0], "development")
            failure = json.loads(
                (evidence / "execution-error.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["retry_count"], 0)
            self.assertEqual(failure["holdout_claim_count"], 0)

    def test_holdout_opens_only_after_all_hard_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / observation.EVIDENCE
            evidence.mkdir(parents=True)
            development = {
                "all_hard_gates_pass": False,
                "selected_candidate_id": None,
            }
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "c" * 64
                    },
                ),
                patch.object(observation, "_git_output", return_value="d" * 40),
                patch.object(
                    observation,
                    "_remote_ci_green",
                    return_value={"preflight_evidence_commit": "d" * 40},
                ),
                patch.object(observation, "_sync_preflight_evidence"),
                patch.object(
                    observation,
                    "_hash_shared_database",
                    return_value="c" * 64,
                ),
                patch.object(
                    observation, "_run_stage_host", return_value=development
                ) as stage,
            ):
                result = observation.run_once(root)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(result["execution"]["holdout_claim_count"], 0)


if __name__ == "__main__":
    unittest.main()

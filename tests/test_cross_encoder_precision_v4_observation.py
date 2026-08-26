from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from neuron_graph_rag import cross_encoder_precision_v4_observation as observation


class CrossEncoderPrecisionV4ObservationTest(unittest.TestCase):
    @staticmethod
    def _ci_green() -> dict[str, object]:
        return {
            "preflight_evidence_commit": "9" * 40,
        }

    def test_identity_linux_root_and_frozen_bindings_are_exact(self) -> None:
        self.assertEqual(
            observation.PROTOCOL_COMMIT,
            "a79e801483d656d401336198a5cc56887a286842",
        )
        self.assertEqual(
            observation.RUN_ROOT,
            Path("/home/hal/ngr-experiments/github_cross_encoder_precision_v4"),
        )
        self.assertEqual(observation.BATCH_SIZE, 8)
        self.assertIs(observation._BASE.load_protocol, observation.load_protocol)
        self.assertIs(observation._BASE._run_worker, observation._run_worker)
        self.assertIs(observation._BASE._run_stage_once, observation._run_stage_once)
        self.assertEqual(observation._BASE.PROTOCOL_ID, observation.PROTOCOL_ID)

    def test_locked_versions_preserve_cpu_torch_local_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "requirements.lock"
            lock.write_text(
                "torch==2.4.1+cpu \\\n+    --hash=sha256:aaa\ntransformers==4.44.2 \\\n+    --hash=sha256:bbb\n",
                encoding="utf-8",
            )
            self.assertEqual(
                observation._locked_versions(lock),
                {"torch": "2.4.1+cpu", "transformers": "4.44.2"},
            )

    def test_preclaim_separates_handoff_from_successor_attestation(self) -> None:
        value = observation._verify_preclaim(observation.ROOT)
        self.assertEqual(
            value["clean_handoff"]["previous_executor_state"][
                "development_claim_count"
            ],
            0,
        )
        self.assertTrue(value["successor_executor"]["distinct_from_previous_executor"])
        self.assertTrue(value["successor_executor"]["semantic_unread_attested"])
        self.assertFalse(
            value["successor_executor"]["forbidden_semantic_content_opened"]
        )

    def test_runner_creates_only_the_missing_parent_before_exclusive_root(self) -> None:
        runner = (
            observation.ROOT / "tools/run_cross_encoder_precision_v4_wsl.sh"
        ).read_text(encoding="utf-8")
        parent = 'mkdir -p "$(dirname "$RUN_ROOT")"'
        exclusive = 'mkdir "$RUN_ROOT" || return $?'
        self.assertLess(runner.index(parent), runner.index(exclusive))
        self.assertNotIn('mkdir -p "$RUN_ROOT"', runner)
        self.assertIn('test ! -e "$RUN_ROOT/bootstrap-failures.tsv"', runner)
        self.assertIn(
            'mv "$RUN_ROOT/bootstrap-commands.tsv" "$RUN_ROOT/bootstrap-failures.tsv"',
            runner,
        )

    def test_bootstrap_log_requires_hashes_zero_rc_and_contiguous_sequence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.tsv"
            command = "git checkout --detach abc"
            empty = observation.sha256_bytes(b"")
            path.write_text(
                "\t".join(
                    [
                        "1",
                        command,
                        "0",
                        empty,
                        empty,
                        observation.sha256_bytes(command.encode("utf-8")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = observation._parse_bootstrap_log(path)
            self.assertEqual(rows[0]["returncode"], 0)
            path.write_text(
                path.read_text(encoding="utf-8").replace("\t0\t", "\t1\t", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "did not complete"):
                observation._parse_bootstrap_log(path)
            failed = observation._parse_bootstrap_log(path, require_all_success=False)
            self.assertEqual(failed[0]["returncode"], 1)

    def test_worker_command_uses_v4_module_and_fresh_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            cache = external / "model-cache"
            stage_root = external / "runs/development"
            stage_root.mkdir(parents=True)
            cache.mkdir()
            output = stage_root / "baseline-primary.json"
            rows: list[dict[str, object]] = []

            def write_packet(*_args: object, **_kwargs: object) -> str:
                output.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
                return ""

            with patch.object(
                observation._BASE, "_run_logged", side_effect=write_packet
            ) as run_logged:
                value = observation._run_worker(
                    root,
                    external,
                    cache,
                    "development",
                    "baseline",
                    "primary",
                    stage_root,
                    rows,
                )
            self.assertEqual(value, {"status": "ok"})
            command = run_logged.call_args.args[0]
            self.assertIn(
                "neuron_graph_rag.cross_encoder_precision_v4_observation", command
            )
            self.assertNotIn(
                "neuron_graph_rag.cross_encoder_precision_v3_observation", command
            )

    def test_failed_development_is_archived_without_retry_or_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / observation.EVIDENCE).mkdir(parents=True)
            error = RuntimeError("one-shot failure")
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "e" * 64,
                        "implementation_commit": "f" * 40,
                    },
                ),
                patch.object(
                    observation, "_run_stage_once", side_effect=error
                ) as run_stage,
                patch.object(
                    observation,
                    "_verify_ci_green",
                    return_value=self._ci_green(),
                ),
                patch.object(
                    observation._BASE, "hash_file_shared", return_value="e" * 64
                ),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={
                        "development": "archived-error",
                        "holdout": "unobserved",
                    },
                ),
                patch.object(observation, "load_protocol", return_value={}),
                self.assertRaisesRegex(RuntimeError, "one-shot failure"),
            ):
                observation.run_conditional(root, root / "external", root / "cache")
            self.assertEqual(run_stage.call_args_list[0].args[0], "development")
            self.assertEqual(len(run_stage.call_args_list), 1)
            failure = json.loads(
                (root / observation.EVIDENCE / "execution-error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["retry_count"], 0)
            self.assertEqual(failure["holdout_claim_count"], 0)

    def test_holdout_opens_only_after_development_hard_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / observation.EVIDENCE).mkdir(parents=True)
            development = {
                "all_hard_gates_pass": True,
                "selected_candidate_id": "base-ce",
            }
            holdout = {
                "all_hard_gates_pass": True,
                "selected_candidate_id": "base-ce",
            }
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64,
                        "implementation_commit": "b" * 40,
                    },
                ),
                patch.object(
                    observation,
                    "_run_stage_once",
                    side_effect=[development, holdout],
                ) as run_stage,
                patch.object(
                    observation,
                    "_verify_ci_green",
                    return_value=self._ci_green(),
                ),
                patch.object(
                    observation._BASE, "hash_file_shared", return_value="a" * 64
                ),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={"development": "passed", "holdout": "passed"},
                ),
                patch.object(observation, "load_protocol", return_value={}),
            ):
                result = observation.run_conditional(
                    root, root / "external", root / "cache"
                )
            self.assertEqual(
                [row.args[0] for row in run_stage.call_args_list],
                ["development", "holdout"],
            )
            self.assertEqual(result["execution"]["claim_count"], 2)
            self.assertEqual(result["execution"]["retry_count"], 0)

    def test_selected_none_does_not_open_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / observation.EVIDENCE).mkdir(parents=True)
            development = {
                "all_hard_gates_pass": False,
                "selected_candidate_id": None,
            }
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "c" * 64,
                        "implementation_commit": "d" * 40,
                    },
                ),
                patch.object(
                    observation, "_run_stage_once", return_value=development
                ) as run_stage,
                patch.object(
                    observation,
                    "_verify_ci_green",
                    return_value=self._ci_green(),
                ),
                patch.object(
                    observation._BASE, "hash_file_shared", return_value="c" * 64
                ),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={"development": "failed", "holdout": "unobserved"},
                ),
                patch.object(observation, "load_protocol", return_value={}),
            ):
                result = observation.run_conditional(
                    root, root / "external", root / "cache"
                )
            self.assertEqual(
                run_stage.call_args_list,
                [
                    call(
                        "development",
                        root,
                        (root / "external").resolve(),
                        (root / "cache").resolve(),
                        [],
                    )
                ],
            )
            self.assertEqual(result["execution"]["holdout_claim_count"], 0)

    def test_holdout_error_records_claim_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / observation.EVIDENCE).mkdir(parents=True)
            development = {
                "all_hard_gates_pass": True,
                "selected_candidate_id": "candidate",
            }
            with (
                patch.object(
                    observation,
                    "verify_preflight",
                    return_value={
                        "shared_database_sha256_before_preflight": "a" * 64,
                        "implementation_commit": "b" * 40,
                    },
                ),
                patch.object(
                    observation,
                    "_verify_ci_green",
                    return_value=self._ci_green(),
                ),
                patch.object(
                    observation,
                    "_run_stage_once",
                    side_effect=[development, RuntimeError("holdout error")],
                ),
                patch.object(
                    observation._BASE, "hash_file_shared", return_value="a" * 64
                ),
                patch.object(
                    observation,
                    "verify_phase_state",
                    return_value={"development": "passed", "holdout": "archived-error"},
                ),
                patch.object(observation, "load_protocol", return_value={}),
                self.assertRaisesRegex(RuntimeError, "holdout error"),
            ):
                observation.run_conditional(root, root / "external", root / "cache")
            failure = json.loads(
                (root / observation.EVIDENCE / "execution-error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["development_claim_count"], 1)
            self.assertEqual(failure["holdout_claim_count"], 1)
            self.assertEqual(failure["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()

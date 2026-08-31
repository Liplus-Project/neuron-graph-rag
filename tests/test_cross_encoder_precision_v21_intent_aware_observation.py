from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from neuron_graph_rag import (
    cross_encoder_precision_v21_intent_aware_observation as observation,
)
from neuron_graph_rag import intent_aware_rank_fusion


class CrossEncoderPrecisionV21IntentAwareObservationTest(unittest.TestCase):
    @staticmethod
    def _fixture(relative: Path) -> dict[str, object]:
        return json.loads((observation.ROOT / relative).read_text(encoding="utf-8"))

    def _passing_candidate_inputs(
        self,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        corpus = self._fixture(observation.CORPUS)
        queries = observation._stage_rows(
            self._fixture(observation.QUERIES), "development"
        )
        gold = observation._stage_rows(
            self._fixture(observation.GOLD), "development"
        )
        paths = [row["path"] for row in corpus["documents"]]
        gold_by_id = {row["case_id"]: row for row in gold}
        baseline_cases = []
        candidate_cases = []
        for query in queries:
            row = gold_by_id[query["case_id"]]
            target = row["expected_path"] or row["forbidden_path"]
            ordered = [target, *[path for path in paths if path != target]]
            relation = (
                [observation._relation_path(row)]
                if row["cohort"] == "relation_linked"
                else []
            )
            baseline_hits = [
                {
                    "source_path": path,
                    "rank": rank,
                    "ngr_score": 1.0 / rank,
                    "source_sha256": "0" * 64,
                    "relation_paths": relation if path == row["expected_path"] else [],
                }
                for rank, path in enumerate(ordered, 1)
            ]
            baseline_cases.append(
                {
                    "case_id": query["case_id"],
                    "cohort": query["cohort"],
                    "ranked_hits": baseline_hits,
                }
            )
            intent = intent_aware_rank_fusion.decompose_query_intent(query["query"])
            signals = []
            for rank, path in enumerate(ordered, 1):
                expected = path == row["expected_path"]
                forbidden = path == row["forbidden_path"]
                signals.append(
                    {
                        "source_path": path,
                        "prefilter_rank": rank,
                        "prefilter_score": 1.0 / rank,
                        "positive_logit": 10.0 if expected else -10.0,
                        "exclusion_logits": [
                            10.0 if forbidden else -10.0
                            for _ in intent.exclusion_queries
                        ],
                        "relation_paths": relation if expected else [],
                    }
                )
            ranked = intent_aware_rank_fusion.fuse_intent_aware_ranks(
                query["query"], signals
            )
            candidate_cases.append(
                {
                    "case_id": query["case_id"],
                    "cohort": query["cohort"],
                    "query": query["query"],
                    "production_signals": signals,
                    "ranked_hits": ranked,
                }
            )
        baseline = {
            "cases": baseline_cases,
            "quality": observation._quality(baseline_cases, gold),
        }
        candidate = {
            "cases": candidate_cases,
            "quality": observation._quality(candidate_cases, gold),
        }
        return baseline, candidate, gold

    def test_identity_paths_and_volume_are_fresh(self) -> None:
        self.assertEqual(
            observation.FREEZE_COMMIT,
            "33b465c7422e8eeae1153e323a46a662a97f8fee",
        )
        self.assertEqual(
            observation.VOLUME, "github-cross-encoder-precision-v21-runtime"
        )
        for path in (
            observation.CONTAINER_ROOT,
            observation.CONTAINER_SOURCE,
            observation.CONTAINER_CACHE,
            observation.CONTAINER_DATABASES,
            observation.CONTAINER_RUNS,
            observation.CONTAINER_ARCHIVE,
            observation.CONTAINER_TRANSPORT,
        ):
            self.assertIsInstance(path, PurePosixPath)
            self.assertTrue(path.as_posix().startswith("/opt/ngr-v21/runtime"))

    def test_fresh_fixtures_are_disjoint_from_v19(self) -> None:
        result = observation._validate_protocol_fixtures(observation.ROOT)
        self.assertEqual(result["corpus_document_count"], 24)
        self.assertEqual(result["query_count"], 16)
        self.assertEqual(result["v19_query_text_reuse_count"], 0)
        self.assertEqual(result["v19_source_path_reuse_count"], 0)
        self.assertNotEqual(
            result["development_identity"], result["holdout_identity"]
        )

    def test_runtime_fixture_validation_does_not_open_v19_cases(self) -> None:
        with patch.object(observation, "_read_object", wraps=observation._read_object) as read:
            observation._validate_protocol_fixtures(
                observation.ROOT, verify_v19_disjoint=False
            )
        opened = {str(call.args[0]) for call in read.call_args_list}
        self.assertFalse(any("v8.queries" in path for path in opened))
        self.assertFalse(any("v8.corpus" in path for path in opened))

    def test_worker_source_has_no_gold_or_forbidden_input(self) -> None:
        source = inspect.getsource(observation._container_worker_v21)
        self.assertNotIn("GOLD", source)
        self.assertNotIn("forbidden", source.lower())
        self.assertIn("production_signals", source)

    def test_v20_gate_ownership_is_literal(self) -> None:
        gates = self._fixture(observation.V20_GATES)
        self.assertEqual(
            gates["protocol_validity_gates"], list(observation.PROTOCOL_GATE_IDS)
        )
        self.assertEqual(
            gates["candidate_controllable_gates"],
            list(observation.CANDIDATE_GATE_IDS),
        )

    def test_candidate_gates_keep_case_nonregression_and_strict_negative(self) -> None:
        baseline, candidate, gold = self._passing_candidate_inputs()
        gates = observation._candidate_gates(candidate, baseline, gold)
        self.assertEqual(
            [row["gate_id"] for row in gates],
            list(observation.CANDIDATE_GATE_IDS),
        )
        self.assertTrue(all(row["passed"] for row in gates))
        broken = json.loads(json.dumps(candidate))
        direct = next(
            row
            for row in broken["cases"]
            if row["cohort"] == "direct_lexical"
        )
        direct["ranked_hits"].reverse()
        broken["quality"] = observation._quality(broken["cases"], gold)
        failed = observation._candidate_gates(broken, baseline, gold)
        self.assertFalse(failed[0]["passed"])

    def test_protocol_gate_rows_precede_candidate_gate_rows(self) -> None:
        self.assertEqual(observation.PROTOCOL_GATE_IDS[0], "protocol-source-contract-integrity")
        self.assertEqual(
            observation.CANDIDATE_GATE_IDS[0],
            "positive-case-rank-non-regression",
        )
        self.assertTrue(set(observation.PROTOCOL_GATE_IDS).isdisjoint(observation.CANDIDATE_GATE_IDS))

    def test_container_command_is_offline_and_uses_only_v21_runtime(self) -> None:
        command = observation.SPEC.container_command(
            "worker", "--stage", "development"
        )
        rendered = "\n".join(command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn(
            "github-cross-encoder-precision-v21-runtime:/opt/ngr-v21/runtime",
            rendered,
        )
        self.assertIn(observation.MODULE, command)
        for forbidden in observation.FORBIDDEN_VOLUMES.values():
            self.assertNotIn(forbidden, rendered)

    def test_source_initialization_does_not_create_predecessor_roots(self) -> None:
        script = observation.SPEC.source_initialization_script()
        self.assertIn("/opt/ngr-v21/runtime/source", script)
        self.assertIn("test ! -e '/opt/ngr-v8/runtime'", script)
        self.assertNotIn("github-cross-encoder-precision-v19-runtime", script)

    def test_stage_host_orders_init_claim_workers_finalize(self) -> None:
        rows: list[dict[str, object]] = []
        claims = {"development": 0, "holdout": 0}
        commands: list[list[str]] = []

        def logged(
            command: list[str],
            _root: Path,
            _rows: list[dict[str, object]],
            **_kwargs: object,
        ) -> str:
            commands.append(command)
            if "stage-init" in command:
                database, output = observation.STAGE_CONTRACT.stage_paths(
                    "development"
                )
                return json.dumps(
                    {
                        "protocol_boundary": "fresh-stage-directories",
                        "stage": "development",
                        "database_directory": str(database),
                        "output_directory": str(output),
                        "stage_directory_create_count": 2,
                        "exclusive_create": True,
                    }
                )
            if "finalize" in command:
                return '{"all_hard_gates_pass": false, "selected_candidate_id": null}'
            return "{}"

        with (
            patch.object(
                observation.lifecycle.lifecycle.lifecycle,
                "_run_logged",
                side_effect=logged,
            ),
            patch.object(observation.lifecycle.lifecycle, "_export_volume_evidence"),
        ):
            observation.SPEC.run_stage_host(
                "development", observation.ROOT, rows, claims
            )
        self.assertIn("stage-init", commands[0])
        self.assertIn("claim", commands[1])
        self.assertEqual(sum("worker" in command for command in commands), 6)
        self.assertIn("finalize", commands[-1])
        self.assertEqual(claims["development"], 1)

    def test_actual_count_contract_separates_planned_and_launched(self) -> None:
        report = {
            "development_claim_count": 1,
            "holdout_claim_count": 0,
            "commands": [
                {"command": ["python", "-m", observation.MODULE, "stage-init"], "returncode": 0},
                {"command": ["python", "-m", observation.MODULE, "claim"], "returncode": 0},
                {"command": ["python", "-m", observation.MODULE, "worker"], "returncode": 1},
            ],
        }
        counts = observation.STAGE_CONTRACT.execution_counts(report)
        self.assertEqual(counts["planned_worker_slot_count"], 6)
        self.assertEqual(counts["actual_worker_launch_count"], 1)
        self.assertEqual(counts["actual_successful_worker_count"], 0)
        self.assertEqual(counts["actual_observed_result_count"], 0)

    def test_prebuild_contract_is_result_free(self) -> None:
        result = observation.validate_prebuild(observation.ROOT)
        self.assertEqual(result["status"], "prebuild_contract_valid")
        self.assertEqual(result["predecessor_artifact_count"], 39)
        self.assertEqual(result["v21_protocol_artifact_count"], 9)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_runner_has_exact_staged_actions(self) -> None:
        runner = (
            observation.ROOT
            / "tools/run_cross_encoder_precision_v21_observation_wslc.ps1"
        ).read_text(encoding="utf-8")
        for action in (
            "prebuild",
            "preflight",
            "verify-preflight",
            "run",
            "audit",
            "finalize-preflight-error",
        ):
            self.assertIn(f'"{action}"', runner)
        self.assertIn(observation.MODULE, runner)

    def test_evidence_absence_is_retry_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                observation.MANIFEST,
                observation.SOURCE_IDENTITY,
                observation.OBSERVATION_AUDIT,
                observation.CORPUS,
                observation.QUERIES,
                observation.GOLD,
                observation.V20_IDENTITIES,
                observation.V20_GATES,
                Path("tests/fixtures/github_cross_encoder_precision_v8.queries.json"),
                Path("tests/fixtures/github_cross_encoder_precision_v8.corpus.json"),
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((observation.ROOT / relative).read_bytes())
            manifest = self._fixture(observation.MANIFEST)
            for registry_name in (
                "predecessor_immutable_sha256",
                "v21_protocol_artifact_sha256",
            ):
                for relative in manifest[registry_name]:
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((observation.ROOT / relative).read_bytes())
            result = observation.audit_evidence(root)
        self.assertEqual(result["status"], "preflight-not-run")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["performance"], "not assessed")

    def test_docs_keep_claim_boundary(self) -> None:
        text = (
            observation.ROOT / "docs/cross-encoder-precision-observation-v21.md"
        ).read_text(encoding="utf-8")
        self.assertIn("protocol validity", text)
        self.assertIn("exactly once", text)
        self.assertIn("production performance", text)
        self.assertNotIn("## Completion", text)

    def test_json_contracts_are_utf8_without_replacement_character(self) -> None:
        for relative in (
            observation.MANIFEST,
            observation.SOURCE_IDENTITY,
            observation.OBSERVATION_AUDIT,
            observation.CORPUS,
            observation.QUERIES,
            observation.GOLD,
        ):
            raw = (observation.ROOT / relative).read_bytes()
            self.assertIsInstance(json.loads(raw.decode("utf-8", errors="strict")), dict)
            self.assertNotIn(b"\xef\xbf\xbd", raw)


if __name__ == "__main__":
    unittest.main()

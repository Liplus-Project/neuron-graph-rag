from __future__ import annotations

import copy
import gc
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import cross_encoder_precision_v2_evaluation as v2_evaluation
from neuron_graph_rag import cross_encoder_precision_v3_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[1]


class CrossEncoderPrecisionV3FreezeTest(unittest.TestCase):
    def _synthetic(self, root: Path) -> dict[str, object]:
        protocol = dict(evaluation.load_protocol())
        protocol["root"] = root
        return protocol

    def _claim(self, protocol: dict[str, object], stage: str = "development") -> bytes:
        path = evaluation._output_path(protocol, stage, "runtime_claim")
        evaluation.write_json_exclusive(
            path,
            {
                "protocol_id": evaluation.PROTOCOL_ID,
                "protocol_commit": "0" * 40,
                "stage": stage,
                "protocol_hashes": dict(protocol["manifest"]["artifact_sha256"]),
                "one_time_claim": True,
            },
        )
        return path.read_bytes()

    def test_protocol_is_hashed_disjoint_bilingual_and_result_free(self) -> None:
        protocol = evaluation.load_protocol()
        self.assertEqual(len(protocol["corpus"]["documents"]), 24)
        self.assertEqual(
            [row["candidate_id"] for row in protocol["candidates"]["candidates"]],
            list(evaluation.CANDIDATE_IDS),
        )
        self.assertEqual(
            [row["gate_id"] for row in protocol["gate"]["gates"]],
            list(evaluation.GATE_IDS),
        )
        for stage in evaluation.STAGES:
            cases = protocol["queries"]["stages"][stage]
            for cohort in evaluation.COHORTS:
                languages = [
                    row["language"] for row in cases if row["cohort"] == cohort
                ]
                self.assertEqual(languages, ["en", "ja"])
        self.assertEqual(
            protocol["result_free_audit"]["freeze_registered_query_execution_count"],
            0,
        )
        self.assertEqual(
            protocol["result_free_audit"]["freeze_model_inference_count"], 0
        )
        self.assertEqual(
            protocol["result_free_audit"]["freeze_observed_result_count"], 0
        )
        output_paths = [
            path
            for stage in protocol["manifest"]["outputs"].values()
            for path in stage.values()
        ]
        self.assertTrue(
            all("github_cross_encoder_precision_v3" in path for path in output_paths)
        )
        self.assertTrue(
            all(
                "github_cross_encoder_precision_v1" not in path
                and "github_cross_encoder_precision_v2" not in path
                for path in output_paths
            )
        )

    def test_frozen_json_is_canonical_utf8(self) -> None:
        for path in (ROOT / "tests/fixtures").glob(f"{evaluation.STEM}.*.json"):
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            )

    def test_codepoint_projection_and_empty_input(self) -> None:
        text = "a" * 479 + "\n" + "日" * 402
        chunks = evaluation.project_passages(text)
        self.assertEqual(
            [(row["start_codepoint"], row["end_codepoint"]) for row in chunks],
            [(0, 480), (400, 880), (800, 882)],
        )
        self.assertEqual(chunks[0]["text"][-1], "\n")
        self.assertEqual(chunks[0]["text"][-80:], chunks[1]["text"][:80])
        self.assertEqual(evaluation.project_passages(""), [])

    def test_synthetic_scorer_recomputes_four_candidates_and_eleven_gates(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            result = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
            evaluation.verify_result_payload(protocol, "development", result, claim)
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(result["selected_candidate_id"], evaluation.CANDIDATE_IDS[0])
        self.assertEqual(len(result["gates"]), 11)
        self.assertTrue(result["all_hard_gates_pass"])
        self.assertTrue(all(row["passed"] for row in result["gates"]))
        self.assertTrue(
            all(len(case["ranked_hits"]) == 24 for case in result["baseline"]["cases"])
        )
        self.assertTrue(
            all(
                len(case["ranked_hits"]) == 20
                for model in result["models"]
                for case in model["cases"]
            )
        )

    def test_all_negative_logits_return_five_and_uniform_shift_is_rank_invariant(
        self,
    ) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            valid = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
        baseline = evaluation._raw_baseline(valid["baseline"])
        models = [evaluation._raw_model(row) for row in valid["models"]]
        self._shift_logits(models, -10.0)
        negative = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, models
        )
        evaluation.verify_result_payload(protocol, "development", negative, claim)
        self.assertTrue(
            all(
                len(case["returned_source_paths"]) == 5
                for candidate in negative["candidates"]
                for case in candidate["cases"]
            )
        )
        shifted_models = copy.deepcopy(models)
        self._shift_logits(shifted_models, -100.0)
        shifted = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, shifted_models
        )
        for left, right in zip(
            negative["candidates"], shifted["candidates"], strict=True
        ):
            self.assertEqual(
                [case["returned_source_paths"] for case in left["cases"]],
                [case["returned_source_paths"] for case in right["cases"]],
            )
            self.assertEqual(
                [
                    [score["ce_rank"] for score in case["scores"]]
                    for case in left["cases"]
                ],
                [
                    [score["ce_rank"] for score in case["scores"]]
                    for case in right["cases"]
                ],
            )

    def test_negative_mixed_short_empty_and_ties_round_trip(self) -> None:
        protocol = evaluation.load_protocol()
        scenarios = (
            ("positive", 20, 10.0),
            ("negative", 20, -10.0),
            ("mixed", 20, 0.0),
            ("short", 4, 0.0),
            ("empty", 0, 0.0),
        )
        for name, hit_count, shift in scenarios:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                synthetic = dict(protocol)
                synthetic["root"] = Path(directory)
                claim = self._claim(synthetic)
                valid = evaluation.build_synthetic_evaluated_result(
                    protocol, "development", claim
                )
                baseline = evaluation._raw_baseline(valid["baseline"])
                models = [evaluation._raw_model(row) for row in valid["models"]]
                self._truncate_model_cases(models, hit_count)
                self._shift_logits(models, shift)
                result = evaluation.evaluate_result_payload(
                    protocol, "development", claim, baseline, models
                )
                evaluation.verify_result_payload(protocol, "development", result, claim)
                expected = min(5, hit_count)
                self.assertTrue(
                    all(
                        len(case["returned_source_paths"]) == expected
                        for candidate in result["candidates"]
                        for case in candidate["cases"]
                    )
                )

        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            valid = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
        baseline = evaluation._raw_baseline(valid["baseline"])
        models = [evaluation._raw_model(row) for row in valid["models"]]
        for model in models:
            hits = model["cases"][0]["ranked_hits"]
            paths = sorted((hits[0]["source_path"], hits[1]["source_path"]))
            for hit in hits[:2]:
                hit["raw_logit"] = 2.0
                for chunk in hit["chunks"]:
                    chunk["raw_logit"] = 2.0
        tied = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, models
        )
        evaluation.verify_result_payload(protocol, "development", tied, claim)
        for candidate in tied["candidates"]:
            if "-ce-" in candidate["candidate_id"]:
                self.assertEqual(
                    candidate["cases"][0]["returned_source_paths"][:2], paths
                )

        models = [evaluation._raw_model(row) for row in valid["models"]]
        for model in models:
            hits = model["cases"][0]["ranked_hits"]
            paths = sorted((hits[0]["source_path"], hits[1]["source_path"]))
            for hit, logit in zip(hits[:2], (9.0, 10.0), strict=True):
                hit["raw_logit"] = logit
                for chunk in hit["chunks"]:
                    chunk["raw_logit"] = logit
        rrf_tied = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, models
        )
        evaluation.verify_result_payload(protocol, "development", rrf_tied, claim)
        for candidate in rrf_tied["candidates"]:
            if "-rrf-" in candidate["candidate_id"]:
                self.assertEqual(
                    candidate["cases"][0]["returned_source_paths"][:2], paths
                )

    def test_empty_derived_and_gate_tampering_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            valid = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
        baseline = evaluation._raw_baseline(valid["baseline"])
        models = [evaluation._raw_model(row) for row in valid["models"]]
        self._truncate_model_cases(models, 0)
        result = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, models
        )
        for name, mutate in (
            (
                "derived empty paths",
                lambda value: value["candidates"][0]["cases"][0][
                    "returned_source_paths"
                ].append(protocol["corpus"]["documents"][0]["path"]),
            ),
            (
                "empty completeness gate",
                lambda value: value["candidates"][0]["gates"][6].update(passed=True),
            ),
        ):
            with self.subTest(name=name):
                tampered = copy.deepcopy(result)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    evaluation.verify_result_payload(
                        protocol, "development", tampered, claim
                    )
                del tampered
                gc.collect()

    def test_v2_v3_semantic_diff_and_predecessor_byte_immutability(self) -> None:
        v3 = evaluation.load_protocol()
        v2 = v2_evaluation.load_protocol()
        for key in ("corpus", "queries", "gold", "models", "result_schema"):
            left = copy.deepcopy(v2[key])
            left["protocol_id"] = evaluation.PROTOCOL_ID
            self.assertEqual(left, v3[key], key)
        self.assertEqual(v2["requirements_lock"], v3["requirements_lock"])

        candidates = copy.deepcopy(v2["candidates"])
        candidates["protocol_id"] = evaluation.PROTOCOL_ID
        candidates.pop("threshold_raw_logit")
        for row in candidates["candidates"]:
            row["candidate_id"] = row["candidate_id"].replace(
                "-threshold", "-rank-only"
            )
        self.assertEqual(candidates, v3["candidates"])

        gates = copy.deepcopy(v2["gate"])
        gates["protocol_id"] = evaluation.PROTOCOL_ID
        gates["gates"][8]["gate_id"] = "cross-encoder-fusion-rank-only-recomputation"
        self.assertEqual(gates, v3["gate"])

        expected_registries = {
            "v1_immutable_sha256": {
                "docs/cross-encoder-precision-freeze-v1.md",
                "src/neuron_graph_rag/cross_encoder_precision_evaluation.py",
                "tests/test_cross_encoder_precision.py",
                *{
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in (ROOT / "tests/fixtures").glob(
                        "github_cross_encoder_precision_v1.*"
                    )
                },
                *{
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in (
                        ROOT / "tests/evidence/github_cross_encoder_precision_v1"
                    ).rglob("*")
                    if path.is_file()
                },
            },
            "v2_immutable_sha256": {
                "docs/cross-encoder-precision-freeze-v2.md",
                "src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py",
                "tests/test_cross_encoder_precision_v2.py",
                *{
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in (ROOT / "tests/fixtures").glob(
                        "github_cross_encoder_precision_v2.*"
                    )
                },
                *{
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in (
                        ROOT / "tests/evidence/github_cross_encoder_precision_v2"
                    ).rglob("*")
                    if path.is_file()
                },
            },
        }
        for key, expected_paths in expected_registries.items():
            registry = v3["manifest"][key]
            self.assertEqual(set(registry), expected_paths)
            for relative, expected_hash in registry.items():
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                    expected_hash,
                )

    def test_baseline_missing_and_clean_negative_defects_are_separated(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            valid = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
        baseline = evaluation._raw_baseline(valid["baseline"])
        models = [evaluation._raw_model(row) for row in valid["models"]]
        positive_target = protocol["gold"]["stages"]["development"][0][
            "expected_paths"
        ][0]
        for surface in [
            baseline["cases"][0]["ranked_hits"],
            *[model["cases"][0]["ranked_hits"] for model in models],
        ]:
            hit = next(row for row in surface if row["source_path"] == positive_target)
            surface.remove(hit)
            surface.insert(5, hit)
            for rank, row in enumerate(surface, 1):
                row["rank"] = rank
                row["ngr_score"] = 1.0 / rank
        for model in models:
            hit = next(
                row
                for row in model["cases"][0]["ranked_hits"]
                if row["source_path"] == positive_target
            )
            for chunk in hit["chunks"]:
                chunk["raw_logit"] = -4.0
            hit["raw_logit"] = -4.0
        clean_forbidden = protocol["gold"]["stages"]["development"][6][
            "forbidden_paths"
        ][0]
        for surface in [
            baseline["cases"][6]["ranked_hits"],
            *[model["cases"][6]["ranked_hits"] for model in models],
        ]:
            hit = next(row for row in surface if row["source_path"] == clean_forbidden)
            surface.remove(hit)
            surface.insert(5, hit)
            for rank, row in enumerate(surface, 1):
                row["rank"] = rank
                row["ngr_score"] = 1.0 / rank
        rebuilt = evaluation.evaluate_result_payload(
            protocol, "development", claim, baseline, models
        )
        candidate = rebuilt["candidates"][0]
        self.assertTrue(candidate["gates"][3]["passed"])
        self.assertFalse(candidate["gates"][6]["passed"])
        self.assertTrue(candidate["gates"][5]["passed"])

    def test_raw_and_derived_tampering_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic)
            valid = evaluation.build_synthetic_evaluated_result(
                protocol, "development", claim
            )
        mutators = {
            "extra field": lambda value: value.update(extra=True),
            "claim hash": lambda value: value.update(claim_sha256="f" * 64),
            "model revision": lambda value: value["models"][0].update(
                revision="0" * 40
            ),
            "prefilter order": lambda value: value["models"][0]["cases"][0][
                "ranked_hits"
            ].reverse(),
            "chunk max": lambda value: value["models"][0]["cases"][0]["ranked_hits"][
                0
            ].update(raw_logit=-9.0),
            "chunk text hash": lambda value: value["models"][0]["cases"][0][
                "ranked_hits"
            ][0]["chunks"][0].update(text_sha256="f" * 64),
            "chunk bounds": lambda value: value["models"][0]["cases"][0]["ranked_hits"][
                0
            ]["chunks"][0].update(start_codepoint=999999, end_codepoint=1000000),
            "chunk cardinality": lambda value: value["models"][0]["cases"][0][
                "ranked_hits"
            ][0]["chunks"].pop(),
            "pair metric": lambda value: value["models"][0]["metrics"].update(
                pair_count=1
            ),
            "window metric": lambda value: value["models"][0]["metrics"].update(
                window_count=1
            ),
            "NGR non-finite": lambda value: value["baseline"]["cases"][0][
                "ranked_hits"
            ][0].update(ngr_score=float("nan")),
            "NGR order": lambda value: value["baseline"]["cases"][0]["ranked_hits"][
                0
            ].update(ngr_score=0.01),
            "relation shape": lambda value: value["baseline"]["cases"][4][
                "ranked_hits"
            ][0]["relation_paths"][0].update(extra=True),
            "relation edge": lambda value: value["baseline"]["cases"][4]["ranked_hits"][
                0
            ]["relation_paths"][0].update(edge_type="unfrozen"),
            "RRF score": lambda value: value["candidates"][0]["cases"][0]["scores"][
                0
            ].update(final_score=-1.0),
            "CE score": lambda value: value["candidates"][1]["cases"][0]["scores"][
                0
            ].update(final_score=-1.0),
            "CE rank": lambda value: value["candidates"][1]["cases"][0]["scores"][
                0
            ].update(ce_rank=99),
            "tie-break order": lambda value: value["candidates"][1]["cases"][0][
                "scores"
            ].reverse(),
            "returned path": lambda value: value["candidates"][0]["cases"][0][
                "returned_source_paths"
            ].reverse(),
            "gate derived field": lambda value: value["candidates"][0]["gates"][8][
                "details"
            ].update(recomputed=False),
            "gate bool": lambda value: value["gates"][0].update(passed=1),
            "selection": lambda value: value.update(
                selected_candidate_id=evaluation.CANDIDATE_IDS[1]
            ),
            "status": lambda value: value.update(status="failed"),
            "state": lambda value: value["models"][0]["state"].update(
                edge_sha256_after="f" * 64
            ),
            "state hash type": lambda value: value["models"][0]["state"].update(
                sqlite_sha256_before=7, sqlite_sha256_after=7
            ),
            "state count type": lambda value: value["models"][0]["state"].update(
                feedback_count_before=True
            ),
            "database identity": lambda value: value["models"][1]["state"].update(
                fresh_database_id=value["models"][0]["state"]["fresh_database_id"]
            ),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(valid)
                mutate(tampered)
                with self.assertRaises((ValueError, TypeError)):
                    evaluation.verify_result_payload(
                        protocol, "development", tampered, claim
                    )
                del tampered
                gc.collect()

    def test_lifecycle_round_trips_every_phase_without_permanent_unobserved_assertion(
        self,
    ) -> None:
        states = evaluation.prove_archive_round_trip()
        self.assertEqual(
            states["unobserved"],
            {"development": "unobserved", "holdout": "unobserved"},
        )
        self.assertEqual(states["development-passed"]["development"], "archived-passed")
        self.assertEqual(states["holdout-passed"]["holdout"], "archived-passed")
        self.assertEqual(states["development-failed"]["development"], "archived-failed")
        self.assertEqual(states["development-error"]["development"], "archived-error")

    def test_duplicate_overwrite_rerun_and_holdout_after_failure_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                evaluation.load_protocol(), "development", claim
            )
            result_path = evaluation._output_path(
                protocol, "development", "runtime_result"
            )
            evaluation.write_json_exclusive(result_path, result)
            with self.assertRaises(FileExistsError):
                evaluation.write_json_exclusive(result_path, result)
            evaluation._archive_stage(protocol, "development")
            with self.assertRaises(FileExistsError):
                evaluation._assert_stage_can_start(protocol, "development")
            with self.assertRaises(FileExistsError):
                evaluation._archive_stage(protocol, "development")

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                evaluation.load_protocol(), "development", claim
            )
            for model in result["models"]:
                model["state"]["replay_ranking_sha256"] = "f" * 64
            failed = evaluation.evaluate_result_payload(
                evaluation.load_protocol(),
                "development",
                claim,
                evaluation._raw_baseline(result["baseline"]),
                [evaluation._raw_model(row) for row in result["models"]],
            )
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                failed,
            )
            evaluation._archive_stage(protocol, "development")
            with self.assertRaisesRegex(ValueError, "closed"):
                evaluation._assert_stage_can_start(protocol, "holdout")

    def test_lifecycle_claim_error_transport_and_cross_exclusivity_fail_closed(
        self,
    ) -> None:
        source = evaluation.load_protocol()

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                source, "development", claim
            )
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                result,
            )
            with (
                patch.object(evaluation, "load_protocol", return_value=protocol),
                self.assertRaises(FileExistsError),
            ):
                evaluation.write_stage_error("development", "must reject")

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                source, "development", claim
            )
            with patch.object(evaluation, "load_protocol", return_value=protocol):
                evaluation.write_stage_error("development", "synthetic failure")
                with self.assertRaises(FileExistsError):
                    evaluation.write_stage_result("development", result)

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            claim_path = evaluation._output_path(
                protocol, "development", "runtime_claim"
            )
            tampered_claim = json.loads(claim.decode("utf-8"))
            tampered_claim["one_time_claim"] = False
            claim_path.write_text(json.dumps(tampered_claim), encoding="utf-8")
            result = evaluation.build_synthetic_evaluated_result(
                source, "development", claim
            )
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                result,
            )
            with self.assertRaisesRegex(ValueError, "claim"):
                evaluation._archive_stage(protocol, "development")

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            error_path = evaluation._output_path(
                protocol, "development", "runtime_error"
            )
            evaluation.write_json_exclusive(
                error_path,
                {
                    "protocol_id": evaluation.PROTOCOL_ID,
                    "stage": "development",
                    "claim_sha256": evaluation.sha256_bytes(claim),
                    "error": "synthetic",
                    "extra": True,
                },
            )
            with self.assertRaisesRegex(ValueError, "error evidence"):
                evaluation._archive_stage(protocol, "development")

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            self._claim(protocol)
            with patch.object(evaluation, "load_protocol", return_value=protocol):
                evaluation.write_stage_error("development", "synthetic failure")
            evaluation._archive_stage(protocol, "development")
            error_path = evaluation._output_path(
                protocol, "development", "archive_error"
            )
            error_value = evaluation.read_json(error_path)
            error_value["claim_sha256"] = "f" * 64
            error_path.write_text(json.dumps(error_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "error evidence"):
                evaluation.verify_phase_state(protocol)

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                source, "development", claim
            )
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                result,
            )
            transport_path = evaluation._archive_stage(protocol, "development")
            transport = evaluation.read_json(transport_path)
            transport["stage_execution_count"] = 2
            transport_path.write_text(
                json.dumps(transport, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "transport"):
                evaluation.verify_phase_state(protocol)
            with self.assertRaises((ValueError, FileExistsError)):
                evaluation._assert_stage_can_start(protocol, "holdout")

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            claim = self._claim(protocol)
            result = evaluation.build_synthetic_evaluated_result(
                source, "development", claim
            )
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                result,
            )
            evaluation._archive_stage(protocol, "development")
            claim_path = evaluation._output_path(
                protocol, "development", "archive_claim"
            )
            claim_value = evaluation.read_json(claim_path)
            claim_value["one_time_claim"] = False
            claim_path.write_text(json.dumps(claim_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim"):
                evaluation.verify_phase_state(protocol)

        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic(Path(directory))
            incomplete = evaluation._output_path(
                protocol, "development", "archive_result"
            )
            evaluation.write_json_exclusive(incomplete, {"all_hard_gates_pass": True})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                evaluation._assert_stage_can_start(protocol, "holdout")

    def test_default_dependency_and_configuration_surfaces_are_unchanged(self) -> None:
        manifest = evaluation.load_protocol()["manifest"]
        registry = manifest["predecessor_immutable_sha256"]
        for relative in (
            "pyproject.toml",
            "src/neuron_graph_rag/retrieval.py",
            "tests/fixtures/github_precision_control_v1.manifest.json",
            "tests/fixtures/github_retrieval_parity_v1.manifest.json",
        ):
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                registry[relative],
            )

    def test_only_freeze_manifest_introduction_commit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init")
            git("config", "core.autocrlf", "false")
            git("config", "user.email", "cross-encoder@example.invalid")
            git("config", "user.name", "Cross Encoder Test")
            (root / "parent.txt").write_text("parent\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "parent")
            artifact = root / "artifact.txt"
            artifact.write_text("frozen\n", encoding="utf-8")
            manifest = (
                root / "tests/fixtures/github_cross_encoder_precision_v3.manifest.json"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "artifact_sha256": {
                            "artifact.txt": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest()
                        }
                    }
                ),
                encoding="utf-8",
            )
            git("add", ".")
            git("commit", "-m", "freeze")
            freeze = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", freeze)
            evaluation.verify_protocol_commit(
                freeze, {"root": root, "manifest_path": manifest}
            )
            (root / "later.txt").write_text("later\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "later")
            later = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", later)
            with self.assertRaisesRegex(ValueError, "introduced"):
                evaluation.verify_protocol_commit(
                    later, {"root": root, "manifest_path": manifest}
                )

    @staticmethod
    def _truncate_model_cases(models: list[dict[str, object]], hit_count: int) -> None:
        for model in models:
            for case in model["cases"]:
                del case["ranked_hits"][hit_count:]
            count = sum(
                len(hit["chunks"])
                for case in model["cases"]
                for hit in case["ranked_hits"]
            )
            model["metrics"]["pair_count"] = count
            model["metrics"]["window_count"] = count

    @staticmethod
    def _shift_logits(models: list[dict[str, object]], amount: float) -> None:
        for model in models:
            for case in model["cases"]:
                for hit in case["ranked_hits"]:
                    hit["raw_logit"] += amount
                    for chunk in hit["chunks"]:
                        chunk["raw_logit"] += amount


if __name__ == "__main__":
    unittest.main()

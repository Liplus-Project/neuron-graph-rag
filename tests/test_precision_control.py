from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import EngineConfig, NeuronGraphRAG, PrecisionControl
from neuron_graph_rag import precision_control_evaluation as evaluation
from neuron_graph_rag.models import ActivationPath, DocumentNode, PathStep, SearchHit
from neuron_graph_rag.precision_control import apply_precision_control

ROOT = Path(__file__).resolve().parents[1]


class PrecisionControlMechanicsTest(unittest.TestCase):
    def _engine(self, control: PrecisionControl | None = None) -> NeuronGraphRAG:
        engine = NeuronGraphRAG(
            config=EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                entry_weight=0.5,
                graph_weight=0.5,
                seed_count=1,
                max_hops=1,
                precision_control=control,
            )
        )
        engine.add_document("source", "alpha source", metadata={"path": "source.md"})
        engine.add_document(
            "linked", "unrelated linked", metadata={"path": "linked.md"}
        )
        engine.add_document("noise", "unrelated noise", metadata={"path": "noise.md"})
        engine.add_edge("source", "linked", "informs", weight=0.8)
        return engine

    def test_default_search_and_explicit_none_are_exactly_equivalent(self) -> None:
        with self._engine() as default, self._engine(None) as explicit_none:
            left = default.search("alpha", limit=3, now=100.0)
            right = explicit_none.search("alpha", limit=3, now=100.0)
        self.assertEqual(left.hits, right.hits)
        self.assertEqual(left.diagnostics, right.diagnostics)
        self.assertNotIn("precision_control", left.hits[0].explain())
        self.assertNotIn("precision_control", left.diagnostics)

    def test_opt_in_filters_after_ranking_without_mutating_scores_or_state(
        self,
    ) -> None:
        control = PrecisionControl("floor", minimum_final_score=0.2)
        with self._engine() as baseline, self._engine(control) as filtered:
            before_edge = filtered.store.list_edges()
            baseline_trace = baseline.search("alpha", limit=3, now=100.0)
            filtered_trace = filtered.search("alpha", limit=3, now=100.0)
            after_edge = filtered.store.list_edges()
            baseline_activation = {
                node.node_id: baseline.store.activation(node.node_id)
                for node in baseline.store.list_nodes()
            }
            filtered_activation = {
                node.node_id: filtered.store.activation(node.node_id)
                for node in filtered.store.list_nodes()
            }
            feedback_count = filtered.store.count_feedback()
        baseline_scores = {
            hit.node.node_id: hit.final_score for hit in baseline_trace.hits
        }
        for hit in filtered_trace.hits:
            self.assertEqual(hit.final_score, baseline_scores[hit.node.node_id])
        self.assertEqual(before_edge, after_edge)
        self.assertEqual(baseline_activation, filtered_activation)
        self.assertEqual(feedback_count, 0)
        self.assertIn("precision_control", filtered_trace.diagnostics)

    def test_explanation_is_recomputable_and_keeps_source_and_relation_path(
        self,
    ) -> None:
        path = ActivationPath(
            seed_id="seed",
            contribution=0.4,
            steps=(PathStep("seed", "target", "informs", 0.8, 1.0),),
        )
        hit = SearchHit(
            node=DocumentNode("target", "text", {"path": "target.md"}),
            sparse_score=0.2,
            dense_score=0.1,
            entry_score=0.3,
            graph_activation=0.4,
            final_score=0.6,
            paths=(path,),
            normalized_graph_activation=0.5,
        )
        lower = SearchHit(
            node=DocumentNode("lower", "text", {"path": "lower.md"}),
            sparse_score=0.1,
            dense_score=0.0,
            entry_score=0.1,
            graph_activation=0.0,
            final_score=0.2,
        )
        control = PrecisionControl(
            "combined",
            minimum_final_score=0.15,
            minimum_top_score_ratio=0.4,
            maximum_top_score_margin=0.4,
            require_entry_graph_signal_agreement=True,
        )
        annotated, accepted = apply_precision_control((hit, lower), control)
        explanation = annotated[0].explain()
        decision = explanation["precision_control"]
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["pre_filter_rank"], 1)
        self.assertEqual(decision["pre_filter_score"], hit.final_score)
        self.assertEqual(
            decision["top_score_ratio"],
            decision["pre_filter_score"] / decision["top_score"],
        )
        self.assertEqual(
            decision["top_score_margin"],
            decision["top_score"] - decision["pre_filter_score"],
        )
        self.assertEqual(explanation["paths"][0]["steps"][0]["edge_type"], "informs")
        self.assertEqual(decision["source_provenance"]["path"], "target.md")
        self.assertEqual(accepted[0].node.metadata["path"], "target.md")
        self.assertFalse(annotated[1].precision_control["accepted"])

    def test_invalid_or_empty_controls_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            PrecisionControl("empty")
        with self.assertRaises(ValueError):
            PrecisionControl("bad", minimum_final_score=1.1)
        with self.assertRaises(ValueError):
            PrecisionControl.from_mapping({"candidate_id": "bad", "query": "no"})


class PrecisionControlFreezeTest(unittest.TestCase):
    def _synthetic_protocol(self, root: Path) -> dict[str, object]:
        protocol = evaluation.load_protocol()
        synthetic = dict(protocol)
        synthetic["root"] = root
        return synthetic

    def _claim(self, protocol: dict[str, object], stage: str) -> bytes:
        hashes = dict(protocol["manifest"]["artifact_sha256"])
        path = evaluation._output_path(protocol, stage, "runtime_claim")
        evaluation.write_json_exclusive(
            path,
            {
                "protocol_id": evaluation.PROTOCOL_ID,
                "protocol_commit": "0" * 40,
                "stage": stage,
                "protocol_hashes": hashes,
                "one_time_claim": True,
            },
        )
        return path.read_bytes()

    def _result(
        self, protocol: dict[str, object], stage: str, claim: bytes, passed: bool
    ) -> dict[str, object]:
        result = evaluation.build_synthetic_evaluated_result(protocol, stage, claim)
        if passed:
            return result
        baseline_raw = {
            "baseline_id": result["baseline"]["baseline_id"],
            "cases": copy.deepcopy(result["baseline"]["cases"]),
            "state": copy.deepcopy(result["baseline"]["state"]),
        }
        candidate_raws = [
            {
                "candidate_id": item["candidate_id"],
                "cases": copy.deepcopy(item["cases"]),
                "explanations": copy.deepcopy(item["explanations"]),
                "state": copy.deepcopy(item["state"]),
            }
            for item in result["candidates"]
        ]
        baseline_raw["state"]["replay_ranking_sha256"] = "f" * 64
        return evaluation.evaluate_result_payload(
            protocol, stage, claim, baseline_raw, candidate_raws
        )

    def test_frozen_protocol_is_complete_disjoint_hashed_and_result_free(self) -> None:
        protocol = evaluation.load_protocol()
        self.assertEqual(len(protocol["corpus"]["documents"]), 20)
        self.assertEqual(len(protocol["candidates"]["candidates"]), 5)
        self.assertEqual(
            [item["gate_id"] for item in protocol["gate"]["gates"]],
            list(evaluation.GATE_IDS),
        )
        self.assertEqual(
            evaluation.verify_phase_state(protocol),
            {
                "development": "unobserved",
                "holdout": "unobserved",
            },
        )
        self.assertEqual(
            protocol["result_free_audit"]["registered_query_execution_count"], 0
        )

    def test_all_freeze_json_is_canonical_utf8(self) -> None:
        for path in (ROOT / "tests/fixtures").glob(f"{evaluation.STEM}.*.json"):
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            )

    def test_audit_and_archive_probe_never_execute_registered_search(self) -> None:
        with patch.object(
            NeuronGraphRAG,
            "search",
            side_effect=AssertionError("registered query execution forbidden"),
        ):
            evaluation.load_protocol()
            phases = evaluation.prove_archive_round_trip()
        self.assertEqual(phases["development"], "archived")

    def test_duplicate_observation_overwrite_and_development_rerun_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic_protocol(Path(directory))
            claim = self._claim(protocol, "development")
            result_path = evaluation._output_path(
                protocol, "development", "runtime_result"
            )
            result = self._result(protocol, "development", claim, True)
            evaluation.write_json_exclusive(result_path, result)
            with self.assertRaises(FileExistsError):
                evaluation.write_json_exclusive(result_path, result)
            evaluation._archive_stage(protocol, "development")
            with self.assertRaises(FileExistsError):
                evaluation._assert_stage_can_start(protocol, "development")
            with self.assertRaises(FileExistsError):
                evaluation._archive_stage(protocol, "development")

    def test_holdout_is_rejected_after_failed_development_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self._synthetic_protocol(Path(directory))
            claim = self._claim(protocol, "development")
            result = self._result(protocol, "development", claim, False)
            evaluation.write_json_exclusive(
                evaluation._output_path(protocol, "development", "runtime_result"),
                result,
            )
            evaluation._archive_stage(protocol, "development")
            with self.assertRaisesRegex(ValueError, "closed"):
                evaluation._assert_stage_can_start(protocol, "holdout")

    def test_result_shape_and_gate_types_fail_closed(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic, "development")
            valid = self._result(synthetic, "development", claim, True)
            evaluation.verify_result_payload(synthetic, "development", valid, claim)
            for mutate in (
                lambda value: value.update(extra=True),
                lambda value: value["gates"].append({}),
                lambda value: value["gates"].__setitem__(0, "not-an-object"),
                lambda value: value["gates"][0].update(hard=False),
                lambda value: value["gates"][0].update(passed=1),
                lambda value: value["candidates"][0].update(all_hard_gates_pass=1),
                lambda value: (
                    value["candidates"][0].update(all_hard_gates_pass=False),
                    value["candidates"][1].update(all_hard_gates_pass=True),
                ),
            ):
                tampered = copy.deepcopy(valid)
                mutate(tampered)
                with self.assertRaises((ValueError, AttributeError)):
                    evaluation.verify_result_payload(
                        synthetic, "development", tampered, claim
                    )

    def test_evaluator_recomputes_nonempty_metrics_gates_and_selection(self) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic, "development")
            passed = self._result(synthetic, "development", claim, True)
            evaluation.verify_result_payload(synthetic, "development", passed, claim)
            self.assertEqual(len(passed["baseline"]["cases"]), 8)
            self.assertEqual(len(passed["baseline"]["cohorts"]), 4)
            self.assertTrue(
                all(
                    len(case["ranked_hits"]) == 20
                    for case in passed["baseline"]["cases"]
                )
            )
            self.assertEqual(passed["selected_candidate_id"], "absolute-floor-025")
            self.assertTrue(passed["all_hard_gates_pass"])
            self.assertTrue(all(gate["passed"] for gate in passed["gates"]))

            failed = self._result(synthetic, "development", claim, False)
            evaluation.verify_result_payload(synthetic, "development", failed, claim)
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["selected_candidate_id"])
            self.assertFalse(failed["gates"][2]["passed"])

    def test_raw_result_metric_provenance_state_and_selection_tamper_fail_closed(
        self,
    ) -> None:
        protocol = evaluation.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            synthetic = dict(protocol)
            synthetic["root"] = Path(directory)
            claim = self._claim(synthetic, "development")
            valid = self._result(synthetic, "development", claim, True)
            forbidden = protocol["gold"]["stages"]["development"][6]["forbidden_paths"][
                0
            ]
            mutators = {
                "empty cases": lambda value: value["baseline"]["cases"].clear(),
                "case id": lambda value: value["baseline"]["cases"][0].update(
                    case_id="tampered"
                ),
                "raw rank": lambda value: value["baseline"]["cases"][0]["ranked_hits"][
                    0
                ].update(rank=2),
                "case rank regression": lambda value: value["candidates"][0]["cases"][
                    0
                ]["returned_source_paths"].reverse(),
                "cohort mrr": lambda value: value["candidates"][0]["cohorts"][0].update(
                    mrr=0.0
                ),
                "negative forbidden": lambda value: value["candidates"][0]["cases"][6][
                    "returned_source_paths"
                ].append(forbidden),
                "relation provenance": lambda value: value["candidates"][0]["cases"][4][
                    "ranked_hits"
                ][0]["relation_paths"].clear(),
                "explanation decision": lambda value: value["candidates"][0][
                    "explanations"
                ][0]["decisions"][0].update(accepted=False),
                "state invariance": lambda value: value["candidates"][0][
                    "state"
                ].update(score_sha256="f" * 64),
                "fresh database isolation": lambda value: value["candidates"][1][
                    "state"
                ].update(
                    fresh_database_id=value["candidates"][0]["state"][
                        "fresh_database_id"
                    ]
                ),
                "candidate gate": lambda value: value["candidates"][0]["gates"][
                    0
                ].update(passed=False),
                "candidate summary": lambda value: value["candidates"][0].update(
                    all_hard_gates_pass=False
                ),
                "global gate": lambda value: value["gates"][0].update(passed=False),
                "selection": lambda value: value.update(
                    selected_candidate_id="top-ratio-055"
                ),
                "status": lambda value: value.update(status="failed"),
            }
            for name, mutate in mutators.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(valid)
                    mutate(tampered)
                    with self.assertRaises(ValueError):
                        evaluation.verify_result_payload(
                            synthetic, "development", tampered, claim
                        )

    def test_only_manifest_introduction_commit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return completed.stdout.strip()

            git("init")
            git("config", "core.autocrlf", "false")
            git("config", "user.email", "precision@example.invalid")
            git("config", "user.name", "Precision Test")
            (root / "parent.txt").write_text("parent\n", encoding="utf-8")
            git("add", "parent.txt")
            git("commit", "-m", "parent")
            artifact = root / "frozen.txt"
            artifact.write_text("frozen\n", encoding="utf-8")
            manifest_path = (
                root / "tests/fixtures/github_precision_control_v1.manifest.json"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifact_sha256": {
                            "frozen.txt": hashlib.sha256(
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
            protocol = {"root": root, "manifest_path": manifest_path}
            evaluation.verify_protocol_commit(freeze, protocol)
            (root / "later.txt").write_text("later\n", encoding="utf-8")
            git("add", "later.txt")
            git("commit", "-m", "later")
            later = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", later)
            with self.assertRaisesRegex(ValueError, "introduced"):
                evaluation.verify_protocol_commit(later, protocol)


if __name__ == "__main__":
    unittest.main()

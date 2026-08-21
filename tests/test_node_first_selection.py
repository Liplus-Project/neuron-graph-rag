from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.node_first_selection import (
    _canonical_checkout_sha256,
    _json_document_bytes,
    aggregate_node_first_results,
    capture_single_response,
    generate_node_first_stage,
    parse_single_response,
    preflight_node_first_capture,
    read_node_first_manifest,
    validate_case_packet,
    validate_single_response,
    validate_stage_packet,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_channels_node_first_experiment.manifest.json"


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _hit(node_id: str, rank: int, *, relation: bool) -> dict[str, object]:
    hit: dict[str, object] = {
        "rank": rank,
        "node_id": node_id,
        "title": node_id,
        "content": f"content for {node_id}",
        "source_metadata": {"doc_path": node_id},
    }
    if relation:
        raw = [
            {
                "source_id": "seed",
                "target_id": node_id,
                "edge_type": "mention",
                "edge_weight": 1.0,
                "factuality": 1.0,
            }
        ]
        hit["paths"] = [
            {
                "raw_steps": raw,
                "projected_steps": [
                    {
                        "source_id": "seed",
                        "target_id": node_id,
                        "edge_type": "mention",
                    }
                ],
            }
        ]
    return hit


def _case(index: int) -> dict[str, object]:
    node_id = f"node-{index}"
    return {
        "case_id": f"case-{index:04d}",
        "query": f"query {index}",
        "lanes": {
            "lexical": {
                "trace_id": f"lex-{index}",
                "hits": [
                    _hit(node_id, 1, relation=False),
                    _hit(f"lex-alt-{index}", 2, relation=False),
                ],
            },
            "relation": {
                "trace_id": f"rel-{index}",
                "hits": [
                    _hit(node_id, 1, relation=True),
                    _hit(f"rel-alt-{index}", 2, relation=True),
                ],
            },
        },
        "agreement_node_ids": [node_id],
        "edge_weight_audit": {
            "before_sha256": "sha256:edges",
            "after_sha256": "sha256:edges",
            "unchanged": True,
        },
    }


def _common() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "d1-liplus-node-first-blind-selection-v3",
        "stage": "development",
        "protocol_manifest_canonical_sha256": "sha256:manifest",
        "judge_prompt_canonical_sha256": "sha256:prompt",
        "lane_semantics": {
            "lexical": "lexical",
            "relation": "relation",
            "rank_scope": "within lane",
        },
        "response_schema": {
            "case_id": "single opaque case",
            "selected_channel": "enum",
            "trace_id": "trace",
            "node_id": "node",
            "rationale": "reason",
        },
    }


def _case_packet(index: int) -> dict[str, object]:
    return {**_common(), "case": _case(index)}


def _response(index: int, channel: str = "lexical") -> dict[str, object]:
    return {
        "case_id": f"case-{index:04d}",
        "selected_channel": channel,
        "trace_id": f"{'lex' if channel == 'lexical' else 'rel'}-{index}",
        "node_id": f"node-{index}",
        "rationale": "synthetic evidence",
    }


class NodeFirstFreezeContractTest(unittest.TestCase):
    def test_runbook_preflight_stage_packet_matches_manifest(self) -> None:
        manifest = read_node_first_manifest(MANIFEST)
        stage_packet = manifest["artifact_paths"]["development"]["stage_packet"]
        runbook = (
            ROOT / "docs" / "node-first-blind-selection-experiment.md"
        ).read_text(encoding="utf-8")

        self.assertIn(f"--stage-packet {stage_packet}", runbook)

    def test_manifest_freezes_prior_versions_queries_and_twelve_gates(self) -> None:
        manifest = read_node_first_manifest(MANIFEST)
        versions = {entry["version"] for entry in manifest["frozen_prior_artifacts"]}
        self.assertEqual(versions, {"v1", "v2"})
        self.assertEqual(len(manifest["frozen_prior_artifacts"]), 24)
        self.assertEqual(len(manifest["gate"]), 12)
        self.assertEqual(manifest["case_count"], 4)
        self.assertEqual(manifest["judges_per_case"], 3)
        self.assertEqual(manifest["relation_case_id"], "case-0003")
        self.assertIn("linked from", manifest["query_overrides"]["development"]["case-0003"])
        self.assertIn("linked from", manifest["query_overrides"]["holdout"]["case-0003"])

    def test_holdout_stops_before_opening_on_development_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "development.json"
            result.write_text(
                json.dumps({"stage": "development", "gate_passed": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Stop rule forbids"):
                generate_node_first_stage(
                    MANIFEST, "holdout", development_result_path=result
                )

    def test_canonical_hash_allows_uniform_checkout_newlines_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_bytes(b"alpha\nbeta\n")
            expected = _canonical_checkout_sha256(path)
            path.write_bytes(b"alpha\r\nbeta\r\n")
            self.assertEqual(_canonical_checkout_sha256(path), expected)
            path.write_bytes(b"alpha\r\nbeta\n")
            with self.assertRaisesRegex(ValueError, "Mixed"):
                _canonical_checkout_sha256(path)


class NodeFirstPacketContractTest(unittest.TestCase):
    def test_case_packet_contains_exactly_one_case_without_scores(self) -> None:
        packet = _case_packet(1)
        validate_case_packet(packet)
        encoded = json.dumps(packet)
        self.assertIn('"case"', encoded)
        self.assertNotIn('"cases"', encoded)
        self.assertNotIn("score", encoded)
        self.assertNotIn("expected_node_id", encoded)

    def test_stage_packet_contains_four_cases_and_single_case_contract(self) -> None:
        cases = [_case(index) for index in range(1, 5)]
        packet = {
            **_common(),
            "cases": cases,
            "case_packet_artifacts": [
                {
                    "case_id": case["case_id"],
                    "path": f"case-{index}.json",
                    "sha256": "sha256:case",
                }
                for index, case in enumerate(cases, start=1)
            ],
            "invocation_contract": {
                "cases_per_invocation": 1,
                "fresh_judges_per_case": 3,
                "context_reuse": "forbidden",
            },
        }
        validate_stage_packet(packet)
        packet["invocation_contract"]["cases_per_invocation"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_stage_packet(packet)

    def test_single_response_rejects_arrays_other_cases_and_cross_lane_trace(self) -> None:
        packet = _case_packet(1)
        with self.assertRaisesRegex(ValueError, "single JSON object"):
            parse_single_response(json.dumps([_response(1)]))
        wrong_case = _response(2)
        with self.assertRaisesRegex(ValueError, "only case"):
            validate_single_response(packet, wrong_case)
        wrong_trace = _response(1)
        wrong_trace["trace_id"] = "rel-1"
        with self.assertRaisesRegex(ValueError, "trace does not belong"):
            validate_single_response(packet, wrong_trace)

    def test_capture_preserves_one_raw_and_parsed_response(self) -> None:
        packet = _case_packet(1)
        raw = json.dumps(_response(1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_bytes(_json_document_bytes(packet))
            artifact = capture_single_response(
                path,
                raw,
                judge_id="judge-1",
                model="synthetic-model",
                agent_type="synthetic",
                executed_at="2026-08-02T00:00:00Z",
            )
        self.assertEqual(artifact["case_id"], "case-0001")
        self.assertEqual(artifact["raw_response"], raw)
        self.assertEqual(artifact["parsed_response"], _response(1))


class NodeFirstAggregationTest(unittest.TestCase):
    def test_synthetic_twelve_invocations_allow_channel_split_on_correct_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--initial-branch=main"], cwd=root, check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Node First Test"],
                cwd=root, check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "node-first@example.invalid"],
                cwd=root, check=True,
            )
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            prompt = fixtures / "prompt.txt"
            prompt.write_text(
                "packetには一つのcaseしかありません\n"
                "laneをまたいで数値比較しない\n",
                encoding="utf-8",
            )
            prior_v1 = root / "prior-v1.txt"
            prior_v2 = root / "prior-v2.txt"
            prior_v1.write_text("v1\n", encoding="utf-8")
            prior_v2.write_text("v2\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "prior-v1.txt", "prior-v2.txt"],
                cwd=root, check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "test: freeze prior artifacts"],
                cwd=root, check=True, capture_output=True,
            )
            prior_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            gold_cases = []
            for index in range(1, 5):
                gold: dict[str, object] = {
                    "cohort": "relation" if index == 3 else "direct_lookup",
                    "expected_node_id": f"node-{index}",
                }
                if index == 3:
                    gold["expected_path"] = [
                        {
                            "source_id": "seed",
                            "target_id": "node-3",
                            "edge_type": "mention",
                        }
                    ]
                gold_cases.append(gold)
            gold = fixtures / "development.gold.json"
            gold.write_text(json.dumps({"cases": gold_cases}), encoding="utf-8")
            v1_manifest = fixtures / "v1.json"
            v1_manifest.write_text(
                json.dumps(
                    {
                        "development": {"gold": gold.name},
                        "holdout": {"gold": "unused.gold.json"},
                    }
                ),
                encoding="utf-8",
            )
            artifact_paths = {
                split: {
                    "stage_packet": f"tests/fixtures/{split}.stage.json",
                    "case_packet_template": f"tests/fixtures/{split}.{{case_id}}.json",
                    "response_template": f"tests/fixtures/{split}.{{case_id}}.judge-{{judge_number}}.json",
                    "result": f"tests/fixtures/{split}.result.json",
                }
                for split in ("development", "holdout")
            }
            manifest_path = fixtures / "manifest.json"
            manifest_payload = {
                "schema_version": 1,
                "experiment_id": "d1-liplus-node-first-blind-selection-v3",
                "case_count": 4,
                "judges_per_case": 3,
                "relation_case_id": "case-0003",
                "gate": [
                    "packet_and_prompt_exclude_answer_fields",
                    "all_responses_valid_and_trace_bound",
                    "node_majority_exists_for_every_case",
                    "majority_selects_every_expected_node",
                    "relation_task_selects_edge_target",
                    "selected_relation_paths_match",
                    "one_case_per_invocation",
                    "abstain_is_never_majority",
                    "search_and_scoring_do_not_change_edges",
                    "node_correctness_is_independent_of_channel",
                    "v1_v2_artifacts_match_frozen_bytes",
                    "observed_artifact_overwrite_is_refused",
                ],
                "frozen_prior_artifacts": [
                    {"path": "prior-v1.txt", "sha256": _sha256_bytes(prior_v1.read_bytes()), "version": "v1"},
                    {"path": "prior-v2.txt", "sha256": _sha256_bytes(prior_v2.read_bytes()), "version": "v2"},
                ],
                "frozen_prior_commit": prior_commit,
                "judge_prompt": {
                    "path": "tests/fixtures/prompt.txt",
                    "canonical_sha256": _canonical_checkout_sha256(prompt),
                },
                "artifact_paths": artifact_paths,
                "query_overrides": {"development": {}, "holdout": {}},
                "stop_rule": {"refuse_overwrite": True},
                "v1_manifest": "tests/fixtures/v1.json",
            }
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            common = _common()
            common["protocol_manifest_canonical_sha256"] = _canonical_checkout_sha256(manifest_path)
            common["judge_prompt_canonical_sha256"] = _canonical_checkout_sha256(prompt)
            cases = [_case(index) for index in range(1, 5)]
            refs = []
            for case in cases:
                packet = {**common, "case": case}
                refs.append(
                    {
                        "case_id": case["case_id"],
                        "path": f"tests/fixtures/{case['case_id']}.json",
                        "sha256": _sha256_bytes(_json_document_bytes(packet)),
                    }
                )
            stage = {
                **common,
                "cases": cases,
                "case_packet_artifacts": refs,
                "invocation_contract": {
                    "cases_per_invocation": 1,
                    "fresh_judges_per_case": 3,
                    "context_reuse": "forbidden",
                },
            }
            stage_path = fixtures / "development.stage.json"
            stage_path.write_bytes(_json_document_bytes(stage))
            case_paths = []
            for case in cases:
                case_path = fixtures / f"{case['case_id']}.json"
                case_path.write_bytes(
                    _json_document_bytes({**common, "case": case})
                )
                case_paths.append(case_path)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "test: register node-first manifest"],
                cwd=root, check=True, capture_output=True,
            )
            preflighted_case = preflight_node_first_capture(
                manifest_path, stage_path, case_paths[0]
            )
            self.assertEqual(preflighted_case["case"]["case_id"], "case-0001")
            invalid_stage = dict(stage)
            invalid_stage["case_packet_artifacts"] = list(
                stage["case_packet_artifacts"]
            )
            invalid_stage["case_packet_artifacts"][0] = dict(
                invalid_stage["case_packet_artifacts"][0]
            )
            invalid_stage["case_packet_artifacts"][0]["sha256"] = "sha256:invalid"
            stage_path.write_bytes(_json_document_bytes(invalid_stage))
            with self.assertRaisesRegex(ValueError, "reference hash"):
                preflight_node_first_capture(manifest_path, stage_path)
            stage_path.write_bytes(_json_document_bytes(stage))
            preflight = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "run_node_first_selection.py"),
                    "preflight-capture",
                    "--manifest",
                    str(manifest_path),
                    "--stage-packet",
                    str(stage_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            response_paths = []
            for case_index in range(1, 5):
                for judge_number in range(1, 4):
                    channel = (
                        "relation"
                        if case_index == 3 and judge_number == 3
                        else "lexical"
                    )
                    parsed = _response(case_index, channel)
                    raw = json.dumps(parsed)
                    artifact = {
                        "schema_version": 1,
                        "experiment_id": common["experiment_id"],
                        "stage": "development",
                        "case_id": parsed["case_id"],
                        "case_packet_sha256": refs[case_index - 1]["sha256"],
                        "judge_id": f"case-{case_index}-judge-{judge_number}",
                        "model": "synthetic-model",
                        "agent_type": "synthetic",
                        "executed_at": f"2026-08-02T00:00:{case_index}{judge_number}Z",
                        "raw_response": raw,
                        "parsed_response": parsed,
                    }
                    path = fixtures / f"response-{case_index}-{judge_number}.json"
                    path.write_bytes(_json_document_bytes(artifact))
                    response_paths.append(path)
            result = aggregate_node_first_results(
                manifest_path, stage_path, response_paths
            )
            self.assertTrue(result["gate_passed"])
            relation_case = next(
                case for case in result["cases"] if case["case_id"] == "case-0003"
            )
            self.assertEqual(relation_case["majority_node_id"], "node-3")
            self.assertEqual(
                relation_case["channel_vote_counts"],
                {"lexical": 2, "relation": 1},
            )
            self.assertEqual(result["metrics"]["majority_node_accuracy"], 1.0)
            self.assertEqual(result["metrics"]["union_oracle_gap"], 0.0)
            self.assertNotIn("majority_channel_accuracy", result["metrics"])

            raw_response = fixtures / "raw-response.json"
            raw_response.write_text(json.dumps(_response(1)), encoding="utf-8")
            refused_output = fixtures / "refused-response.json"
            case_paths[0].write_bytes(b"{}")
            refused_capture = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "run_node_first_selection.py"),
                    "capture-response",
                    "--manifest",
                    str(manifest_path),
                    "--stage-packet",
                    str(stage_path),
                    "--case-packet",
                    str(case_paths[0]),
                    "--raw-response",
                    str(raw_response),
                    "--judge-id",
                    "new-judge",
                    "--model",
                    "synthetic-model",
                    "--agent-type",
                    "synthetic",
                    "--output",
                    str(refused_output),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(refused_capture.returncode, 0)
            self.assertIn(
                "Frozen case packet differs from stage reference",
                refused_capture.stderr,
            )
            self.assertFalse(refused_output.exists())
            case_paths[0].write_bytes(
                _json_document_bytes({**common, "case": cases[0]})
            )

            valid_artifact = json.loads(response_paths[0].read_text(encoding="utf-8"))
            invalid_artifact = dict(valid_artifact)
            invalid_artifact["stage"] = "holdout"
            response_paths[0].write_bytes(_json_document_bytes(invalid_artifact))
            with self.assertRaisesRegex(ValueError, "stage differs"):
                aggregate_node_first_results(
                    manifest_path, stage_path, response_paths
                )

            response_paths[0].write_bytes(_json_document_bytes(valid_artifact))
            invalid_stage = json.loads(stage_path.read_text(encoding="utf-8"))
            invalid_stage["case_packet_artifacts"][0]["sha256"] = "sha256:invalid"
            stage_path.write_bytes(_json_document_bytes(invalid_stage))
            with self.assertRaisesRegex(ValueError, "reference hash"):
                aggregate_node_first_results(
                    manifest_path, stage_path, response_paths
                )

    def test_observed_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_exclusive(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"second": True})


if __name__ == "__main__":
    unittest.main()

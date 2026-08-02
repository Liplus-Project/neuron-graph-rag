from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.blind_selection import (
    _evaluate_majority,
    _matches_frozen_text_bytes,
    aggregate_blind_results,
    capture_judge_response,
    generate_blind_packet,
    project_path,
    read_blind_manifest,
    validate_judge_responses,
    validate_packet,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "d1_liplus_channels_blind_experiment.manifest.json"
DEVELOPMENT_PACKET = (
    FIXTURES / "d1_liplus_channels_blind.development.packet.json"
)
DEVELOPMENT_JUDGE_1 = (
    FIXTURES / "d1_liplus_channels_blind.development.judge-1.json"
)
DEVELOPMENT_JUDGE_2 = (
    FIXTURES / "d1_liplus_channels_blind.development.judge-2.json"
)
DEVELOPMENT_JUDGE_3_RAW = (
    FIXTURES / "d1_liplus_channels_blind.development.judge-3.raw.json"
)
DEVELOPMENT_RESULT = (
    FIXTURES / "d1_liplus_channels_blind.development.result.json"
)
HOLDOUT_PACKET = FIXTURES / "d1_liplus_channels_blind.holdout.packet.json"
HOLDOUT_RESULT = FIXTURES / "d1_liplus_channels_blind.holdout.result.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
        hit["paths"] = [{"raw_steps": raw, "projected_steps": project_path(raw)}]
    return hit


def _packet() -> dict[str, object]:
    cases = []
    for index in range(1, 5):
        cases.append(
            {
                "case_id": f"case-{index:04d}",
                "query": f"query {index}",
                "lanes": {
                    "lexical": {
                        "trace_id": f"lex-{index}",
                        "hits": [
                            _hit(f"lex-node-{index}", 1, relation=False),
                            _hit(f"shared-{index}", 2, relation=False),
                        ],
                    },
                    "relation": {
                        "trace_id": f"rel-{index}",
                        "hits": [
                            _hit(f"rel-node-{index}", 1, relation=True),
                            _hit(f"shared-{index}", 2, relation=True),
                        ],
                    },
                },
                "agreement_node_ids": [f"shared-{index}"],
                "edge_weight_audit": {
                    "before_sha256": "sha256:before",
                    "after_sha256": "sha256:before",
                    "unchanged": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": "synthetic",
        "stage": "development",
        "protocol_manifest_sha256": "sha256:manifest",
        "judge_prompt_sha256": "sha256:prompt",
        "lane_semantics": {
            "lexical": "lexical",
            "relation": "relation",
            "rank_scope": "within lane",
        },
        "response_schema": {
            "case_id": "opaque",
            "selected_channel": "enum",
            "trace_id": "trace",
            "node_id": "node",
            "rationale": "reason",
        },
        "cases": cases,
    }


def _responses(packet: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case_id": case["case_id"],
            "selected_channel": "lexical",
            "trace_id": case["lanes"]["lexical"]["trace_id"],
            "node_id": case["lanes"]["lexical"]["hits"][0]["node_id"],
            "rationale": "direct wording",
        }
        for case in packet["cases"]
    ]


class BlindFreezeContractTest(unittest.TestCase):
    def test_manifest_freezes_v1_bytes_prompt_and_twelve_gates(self) -> None:
        manifest = read_blind_manifest(MANIFEST)
        self.assertEqual(manifest["frozen_v1_baseline_commit"], "b15e27882f013bd895032e6edd15489eb5206926")
        self.assertEqual(len(manifest["frozen_v1_bytes"]), 11)
        self.assertEqual(len(manifest["gate"]), 12)
        self.assertEqual(manifest["judge_count"], 3)
        self.assertTrue(manifest["stop_rule"]["refuse_overwrite"])

    def test_v1_hash_allows_only_exact_lf_crlf_checkout_transform(self) -> None:
        frozen = b"alpha\r\nbeta\r\n"
        expected = "sha256:" + hashlib.sha256(frozen).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.txt"
            path.write_bytes(frozen)
            self.assertTrue(_matches_frozen_text_bytes(path, expected))
            path.write_bytes(b"alpha\nbeta\n")
            self.assertTrue(_matches_frozen_text_bytes(path, expected))
            path.write_bytes(b"alpha changed\nbeta\n")
            self.assertFalse(_matches_frozen_text_bytes(path, expected))
            path.write_bytes(b"alpha\r\nbeta\n")
            self.assertFalse(_matches_frozen_text_bytes(path, expected))
            path.write_bytes(b"alpha\rbeta\r")
            self.assertFalse(_matches_frozen_text_bytes(path, expected))
            expected_lf = "sha256:" + hashlib.sha256(b"alpha\nbeta\n").hexdigest()
            path.write_bytes(frozen)
            self.assertTrue(_matches_frozen_text_bytes(path, expected_lf))

    def test_holdout_packet_stops_before_opening_on_development_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "development.json"
            result.write_text(
                json.dumps(
                    {
                        "stage": "development",
                        "gate_passed": False,
                        "manifest_sha256": "not-used-on-failure",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Stop rule forbids"):
                generate_blind_packet(
                    MANIFEST, "holdout", development_result_path=result
                )


class BlindObservedResultAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))

    def test_invalid_third_response_stops_before_aggregation(self) -> None:
        self.assertEqual(
            self.result["freeze_commit"],
            "062c1314c1b63ee34b8963b980b2c51eb2c3e9a0",
        )
        self.assertEqual(
            self.result["failure"],
            {
                "phase": "capture-response",
                "judge_id": "blind-dev-judge-3",
                "error": "Judge must answer every packet case exactly once",
                "expected_response_count": 4,
                "observed_response_count": 3,
                "missing_case_ids": ["case-0003"],
            },
        )
        self.assertEqual(self.result["execution"]["valid_captured_responses"], 2)
        self.assertEqual(self.result["execution"]["invalid_responses"], 1)
        self.assertFalse(self.result["execution"]["aggregate_executed"])
        self.assertFalse(self.result["execution"]["retry_performed"])
        self.assertFalse(self.result["execution"]["replacement_judge_used"])
        self.assertFalse(self.result["gate_passed"])
        self.assertEqual(
            self.result["metrics_status"],
            "not_computed_invalid_judge_response",
        )

    def test_result_hashes_preserved_observed_artifacts_without_reading_them(self) -> None:
        self.assertEqual(
            self.result["packet_sha256"], _sha256(DEVELOPMENT_PACKET)
        )
        expected_paths = {
            DEVELOPMENT_JUDGE_1.name: DEVELOPMENT_JUDGE_1,
            DEVELOPMENT_JUDGE_2.name: DEVELOPMENT_JUDGE_2,
            DEVELOPMENT_JUDGE_3_RAW.name: DEVELOPMENT_JUDGE_3_RAW,
        }
        for artifact in self.result["judge_artifacts"]:
            path = expected_paths[artifact["path"]]
            self.assertEqual(artifact["sha256"], _sha256(path))

    def test_stop_rule_keeps_holdout_unopened(self) -> None:
        self.assertEqual(
            self.result["holdout_status"],
            "not_opened_invalid_judge_response",
        )
        self.assertFalse(HOLDOUT_PACKET.exists())
        self.assertFalse(HOLDOUT_RESULT.exists())


class BlindPacketValidationTest(unittest.TestCase):
    def test_packet_carries_independent_traces_without_scores(self) -> None:
        packet = _packet()
        validate_packet(packet)
        encoded = json.dumps(packet)
        self.assertNotIn("score", encoded)
        self.assertNotIn("intended_channel", encoded)
        self.assertNotIn("expected_node_id", encoded)

    def test_forbidden_answer_field_is_rejected_recursively(self) -> None:
        packet = _packet()
        packet["cases"][0]["cohort"] = "relation"
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            validate_packet(packet)

    def test_zero_hop_and_changed_projection_are_rejected(self) -> None:
        zero_hop = _packet()
        zero_hop["cases"][0]["lanes"]["relation"]["hits"][0]["paths"][0]["raw_steps"] = []
        with self.assertRaisesRegex(ValueError, "zero-hop"):
            validate_packet(zero_hop)
        changed = _packet()
        changed["cases"][0]["lanes"]["relation"]["hits"][0]["paths"][0]["projected_steps"][0]["edge_type"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_packet(changed)

    def test_relation_hit_without_a_path_is_rejected(self) -> None:
        packet = _packet()
        packet["cases"][0]["lanes"]["relation"]["hits"][0]["paths"] = []
        with self.assertRaisesRegex(ValueError, "require edge paths"):
            validate_packet(packet)


class BlindJudgeResponseTest(unittest.TestCase):
    def test_response_capture_preserves_raw_and_parsed_with_metadata(self) -> None:
        packet = _packet()
        raw = json.dumps({"responses": _responses(packet)})
        with tempfile.TemporaryDirectory() as directory:
            packet_path = Path(directory) / "packet.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            artifact = capture_judge_response(
                packet_path,
                raw,
                judge_id="judge-1",
                model="test-model",
                agent_type="default",
                executed_at="2026-08-02T00:00:00Z",
            )
        self.assertEqual(artifact["raw_response"], raw)
        self.assertEqual(artifact["parsed_responses"], _responses(packet))
        self.assertEqual(artifact["model"], "test-model")

    def test_cross_lane_trace_or_node_is_rejected(self) -> None:
        packet = _packet()
        responses = _responses(packet)
        responses[0]["trace_id"] = "rel-1"
        with self.assertRaisesRegex(ValueError, "trace does not belong"):
            validate_judge_responses(packet, responses)
        responses = _responses(packet)
        responses[0]["node_id"] = "rel-node-1"
        with self.assertRaisesRegex(ValueError, "node does not belong"):
            validate_judge_responses(packet, responses)

    def test_abstention_requires_null_trace_and_node(self) -> None:
        packet = _packet()
        responses = _responses(packet)
        responses[0].update(
            {"selected_channel": "abstain", "trace_id": None, "node_id": None}
        )
        validate_judge_responses(packet, responses)
        responses[0]["trace_id"] = "lex-1"
        with self.assertRaisesRegex(ValueError, "requires null"):
            validate_judge_responses(packet, responses)

    def test_response_requires_exact_top_level_object(self) -> None:
        packet = _packet()
        raw = json.dumps(_responses(packet))
        with tempfile.TemporaryDirectory() as directory:
            packet_path = Path(directory) / "packet.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen JSON object"):
                capture_judge_response(
                    packet_path,
                    raw,
                    judge_id="judge-1",
                    model="test-model",
                    agent_type="default",
                )


class BlindAggregationRuleTest(unittest.TestCase):
    def test_synthetic_three_judge_aggregation_passes_frozen_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            prompt = fixtures / "prompt.txt"
            prompt.write_text("laneをまたいで数値比較しない\n", encoding="utf-8")
            baseline = root / "baseline.txt"
            baseline.write_text("frozen\n", encoding="utf-8")
            gold_cases = []
            packet = _packet()
            packet["experiment_id"] = "d1-liplus-blind-channel-selection-v2"
            for index, case in enumerate(packet["cases"], start=1):
                relation = index == 4
                gold: dict[str, object] = {
                    "cohort": "relation" if relation else "direct_lookup",
                    "expected_node_id": (
                        f"rel-node-{index}" if relation else f"lex-node-{index}"
                    ),
                    "intended_channel": "relation" if relation else "lexical",
                }
                if relation:
                    gold["expected_path"] = [
                        {
                            "source_id": "seed",
                            "target_id": f"rel-node-{index}",
                            "edge_type": "mention",
                        }
                    ]
                gold_cases.append(gold)
            gold_path = fixtures / "development.gold.json"
            gold_path.write_text(
                json.dumps({"cases": gold_cases}), encoding="utf-8"
            )
            v1_manifest = fixtures / "v1.json"
            v1_manifest.write_text(
                json.dumps(
                    {
                        "development": {"gold": gold_path.name},
                        "holdout": {"gold": "unused-holdout.gold.json"},
                    }
                ),
                encoding="utf-8",
            )
            manifest = fixtures / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "d1-liplus-blind-channel-selection-v2",
                        "gate": [f"gate-{index}" for index in range(12)],
                        "frozen_v1_bytes": [
                            {"path": "baseline.txt", "sha256": _sha256(baseline)}
                        ],
                        "frozen_absent_paths": [],
                        "judge_prompt": {
                            "path": "tests/fixtures/prompt.txt",
                            "sha256": _sha256(prompt),
                        },
                        "stop_rule": {"refuse_overwrite": True},
                        "v1_manifest": "tests/fixtures/v1.json",
                    }
                ),
                encoding="utf-8",
            )
            packet["protocol_manifest_sha256"] = _sha256(manifest)
            packet["judge_prompt_sha256"] = _sha256(prompt)
            packet_path = fixtures / "packet.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")

            response_paths = []
            for judge_index in range(3):
                parsed = []
                for case_index, case in enumerate(packet["cases"], start=1):
                    channel = "relation" if case_index == 4 else "lexical"
                    parsed.append(
                        {
                            "case_id": case["case_id"],
                            "selected_channel": channel,
                            "trace_id": case["lanes"][channel]["trace_id"],
                            "node_id": case["lanes"][channel]["hits"][0]["node_id"],
                            "rationale": "synthetic evidence",
                        }
                    )
                raw = json.dumps({"responses": parsed})
                artifact_path = fixtures / f"judge-{judge_index}.json"
                artifact_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "packet_sha256": _sha256(packet_path),
                            "judge_id": f"judge-{judge_index}",
                            "model": "synthetic-model",
                            "agent_type": "synthetic",
                            "executed_at": f"2026-08-02T00:00:0{judge_index}Z",
                            "raw_response": raw,
                            "parsed_responses": parsed,
                        }
                    ),
                    encoding="utf-8",
                )
                response_paths.append(artifact_path)

            result = aggregate_blind_results(
                manifest, packet_path, response_paths
            )
            self.assertTrue(result["gate_passed"])
            self.assertEqual(result["metrics"]["selected_node_mrr"], 1.0)
            self.assertEqual(result["metrics"]["union_oracle_gap"], 0.0)
            self.assertEqual(result["holdout_status"], "eligible_for_single_packet")

    def test_two_matching_votes_form_majority_and_project_path(self) -> None:
        case = _packet()["cases"][0]
        gold = {
            "cohort": "relation",
            "expected_node_id": "rel-node-1",
            "intended_channel": "relation",
            "expected_path": [
                {
                    "source_id": "seed",
                    "target_id": "rel-node-1",
                    "edge_type": "mention",
                }
            ],
        }
        rel = {
            "case_id": "case-0001",
            "selected_channel": "relation",
            "trace_id": "rel-1",
            "node_id": "rel-node-1",
            "rationale": "edge path",
        }
        lex = {
            "case_id": "case-0001",
            "selected_channel": "lexical",
            "trace_id": "lex-1",
            "node_id": "lex-node-1",
            "rationale": "terms",
        }
        evaluated = _evaluate_majority(
            case,
            gold,
            [{"case-0001": rel}, {"case-0001": copy.deepcopy(rel)}, {"case-0001": lex}],
        )
        self.assertTrue(evaluated["has_majority"])
        self.assertTrue(evaluated["node_correct"])
        self.assertTrue(evaluated["channel_correct"])
        self.assertTrue(evaluated["path_matched"])

    def test_three_different_votes_aggregate_to_abstain(self) -> None:
        case = _packet()["cases"][0]
        gold = {
            "cohort": "direct_lookup",
            "expected_node_id": "lex-node-1",
            "intended_channel": "lexical",
        }
        votes = [
            {"selected_channel": "lexical", "node_id": "lex-node-1"},
            {"selected_channel": "relation", "node_id": "rel-node-1"},
            {"selected_channel": "abstain", "node_id": None},
        ]
        maps = [
            {"case-0001": {"case_id": "case-0001", "trace_id": None, "rationale": "x", **vote}}
            for vote in votes
        ]
        evaluated = _evaluate_majority(case, gold, maps)
        self.assertFalse(evaluated["has_majority"])
        self.assertEqual(evaluated["selected_channel"], "abstain")
        self.assertIsNone(evaluated["selected_node_id"])

    def test_observed_artifact_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            write_json_exclusive(output, {"first": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(output, {"second": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"first": True})


if __name__ == "__main__":
    unittest.main()

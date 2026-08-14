from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.feedback_policy_comparison_evaluation import (
    MANIFEST_PATH,
    _evaluate_gates,
    _run_arm,
    prove_writer_verifier_round_trip,
    read_json,
    verify_registered_result,
    write_observed_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
STEM = "feedback_policy_comparison_v1"


def _fixture(name: str) -> dict[str, object]:
    return read_json(FIXTURES / f"{STEM}.{name}.json")


class FeedbackPolicyComparisonEvaluationTest(unittest.TestCase):
    def test_result_free_artifacts_are_canonical_and_complete(self) -> None:
        names = (
            "fixture",
            "gold",
            "schedule",
            "gate",
            "result-schema",
            "result-free-audit",
        )
        for name in names:
            path = FIXTURES / f"{STEM}.{name}.json"
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="strict"))
            self.assertNotIn(b"\r", raw)
            self.assertEqual(
                raw.decode("utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            self.assertEqual(
                payload["protocol_id"], "policycmp85-feedback-policy-comparison-v1"
            )

        audit = _fixture("result-free-audit")
        self.assertTrue(audit["result_free"])
        self.assertTrue(audit["placeholder_round_trip_passed"])
        self.assertFalse(audit["placeholder_output_registered"])

        manifest = read_json(MANIFEST_PATH)
        for relative, expected in manifest["artifact_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected,
                relative,
            )

    def test_source_projection_and_split_identity_are_frozen(self) -> None:
        fixture = _fixture("fixture")
        source = json.loads(
            (
                ROOT
                / "corpora"
                / "feedback-policy-comparison-v1"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        identity: dict[str, set[str]] = {}
        for stage in ("development", "holdout"):
            split = fixture["splits"][stage]
            source_split = source["splits"][stage]
            self.assertEqual(len(split["nodes"]), 8)
            self.assertEqual(len(split["edges"]), 6)
            self.assertEqual(
                {(item["node_id"], item["path"], item["raw_sha256"]) for item in split["nodes"]},
                {
                    (item["node_id"], item["path"], item["raw_sha256"])
                    for item in source_split["documents"]
                },
            )
            identity[stage] = {
                split["split_id"],
                *(item["node_id"] for item in split["nodes"]),
                *(item["path"] for item in split["nodes"]),
                *(item["source_url"] for item in split["nodes"]),
                *(
                    "|".join((item["source_id"], item["target_id"], item["edge_type"]))
                    for item in split["edges"]
                ),
            }
        self.assertTrue(identity["development"].isdisjoint(identity["holdout"]))

    def test_arms_schedule_gold_gates_and_schema_are_frozen(self) -> None:
        schedule = _fixture("schedule")
        gold = _fixture("gold")
        gate = _fixture("gate")
        schema = _fixture("result-schema")
        self.assertEqual(schedule["checkpoints"], [0, 1, 3, 10])
        self.assertEqual(
            [item["arm_id"] for item in schedule["arms"]],
            ["control", "used_q3_s1", "confirmed_r05_s1"],
        )
        self.assertEqual(
            schedule["arms"][1]["engine_overrides"],
            {
                "relation_feedback_evidence_quorum": 3,
                "confirmed_outcome_reinforcement": False,
                "confirmation_decay_ratio": None,
            },
        )
        self.assertEqual(
            schedule["arms"][2]["engine_overrides"],
            {
                "relation_feedback_evidence_quorum": 1,
                "confirmed_outcome_reinforcement": True,
                "confirmation_decay_ratio": 0.5,
            },
        )
        for stage in ("development", "holdout"):
            self.assertEqual(
                [item["cohort"] for item in gold["splits"][stage]],
                ["confirmed-use", "corrected-use"],
            )
        self.assertEqual(
            [item["gate_id"] for item in gate["gates"]],
            [
                "protocol-integrity",
                "confirmed-diminishing",
                "used-quorum-boundary",
                "corrected-isolation",
                "confirmed-headroom",
                "checkpoint-10-safety",
                "mutation-locality",
            ],
        )
        self.assertEqual(schema["field_reorder_after_observation"], "prohibited")

    def test_exclusive_writer_and_placeholder_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "probe.json"
            prove_writer_verifier_round_trip(probe)
            self.assertFalse(probe.exists())
            output = root / "exclusive.json"
            write_observed_exclusive(output, {"value": 1})
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_observed_exclusive(output, {"value": 2})
            self.assertEqual(output.read_bytes(), before)

    def test_placeholder_mechanics_probe_passes_registered_gate_shapes(self) -> None:
        split = {"nodes": [], "edges": []}
        texts = {}
        cases = []
        for cohort, outcome in (("confirmed-use", "confirmed"), ("corrected-use", "corrected")):
            prefix = f"probe-{cohort}"
            node_ids = {
                role: f"{prefix}-{role}" for role in ("source", "route", "terminal", "sibling")
            }
            query = f"Probe {cohort} source"
            direct = f"Probe {cohort} terminal"
            for role, node_id in node_ids.items():
                split["nodes"].append(
                    {
                        "node_id": node_id,
                        "source_url": f"https://example.invalid/{node_id}",
                        "path": f"temporary/{node_id}.md",
                    }
                )
                texts[node_id] = query if role == "source" else (
                    direct if role == "terminal" else f"Placeholder {cohort} {role}"
                )
            split["edges"].extend(
                [
                    {
                        "source_id": node_ids["source"],
                        "target_id": node_ids["route"],
                        "edge_type": "mention",
                        "weight": 0.8,
                    },
                    {
                        "source_id": node_ids["source"],
                        "target_id": node_ids["sibling"],
                        "edge_type": "mention",
                        "weight": 0.9,
                    },
                    {
                        "source_id": node_ids["route"],
                        "target_id": node_ids["terminal"],
                        "edge_type": "mention",
                        "weight": 1.3,
                    },
                ]
            )
            used = node_ids["terminal"] if cohort == "confirmed-use" else node_ids["sibling"]
            non_target = node_ids["sibling"] if cohort == "confirmed-use" else node_ids["route"]
            cases.append(
                {
                    "case_id": prefix,
                    "cohort": cohort,
                    "cluster_node_ids": list(node_ids.values()),
                    "relation_query": query,
                    "direct_query": direct,
                    "reverse_query": direct,
                    "expected_terminal_node_id": node_ids["terminal"],
                    "used_node_id": used,
                    "non_target_sibling_node_id": non_target,
                    "unrelated_node_id": (
                        "probe-corrected-use-terminal"
                        if cohort == "confirmed-use"
                        else "probe-confirmed-use-terminal"
                    ),
                    "direct_node_id": node_ids["terminal"],
                    "reverse_expected_node_id": node_ids["source"],
                    "outcome": outcome,
                    "outcome_summary": "placeholder result-free mechanics probe",
                    "limit": 8,
                    "allowed_mutation_edges": [
                        "|".join((edge["source_id"], edge["target_id"], edge["edge_type"]))
                        for edge in split["edges"][-3:]
                        if cohort == "confirmed-use" or edge["source_id"] == node_ids["source"]
                    ],
                }
            )
        schedule = _fixture("schedule")
        observed = {}
        for arm in schedule["arms"]:
            result = _run_arm(arm, schedule["base_engine_config"], split, cases, texts)
            result["semantic_sha256"] = "placeholder"
            result["fresh_clone_replay"] = True
            observed[result["arm_id"]] = result
        gates = _evaluate_gates({"passed": True}, cases, observed)
        self.assertTrue(all(gates.values()), gates)

    def test_registered_outputs_are_verified_without_reexecution(self) -> None:
        if not MANIFEST_PATH.exists():
            self.skipTest("freeze manifest has not been assembled")
        manifest = read_json(MANIFEST_PATH)
        for stage in ("development", "holdout"):
            output = ROOT / manifest["outputs"][stage]
            if output.exists():
                verify_registered_result(stage)


if __name__ == "__main__":
    unittest.main()

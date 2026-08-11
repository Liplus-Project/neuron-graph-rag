from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.rank_elasticity import (
    SCHEMA_VERSION,
    _diagnose,
    _rank_deltas,
    read_rank_elasticity_schedule,
    run_rank_elasticity,
    write_rank_elasticity_result,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FIXTURE = FIXTURES / "rank_elasticity_v1.fixture.json"
SCHEDULE = FIXTURES / "rank_elasticity_v1.schedule.json"


def _create_source_database(path: Path) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with NeuronGraphRAG(path) as engine:
        for document in fixture["documents"]:
            engine.add_document(
                document["node_id"],
                document["text"],
                metadata=document["metadata"],
            )
        for edge in fixture["edges"]:
            engine.add_edge(
                edge["source_id"],
                edge["target_id"],
                edge["edge_type"],
                weight=edge["weight"],
                factuality=edge["factuality"],
            )


class RankElasticityTest(unittest.TestCase):
    def test_schedule_fixes_ceiling_threshold_and_three_control_roles(self) -> None:
        schedule = read_rank_elasticity_schedule(SCHEDULE)
        self.assertEqual(schedule["schema_version"], SCHEMA_VERSION)
        self.assertEqual(schedule["checkpoints"], [0, 1, 3, 5, 10])
        self.assertEqual(
            {scenario["scenario_id"] for scenario in schedule["scenarios"]},
            {"max-normalization-ceiling", "credited-edge-threshold"},
        )
        for scenario in schedule["scenarios"]:
            self.assertEqual(
                {case["role"] for case in scenario["cases"]},
                {
                    "relation_target",
                    "direct_control",
                    "lexical_control",
                    "directional_negative_control",
                },
            )

    def test_fresh_clone_simulation_is_deterministic_and_source_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.sqlite"
            _create_source_database(database)
            before = database.read_bytes()

            first = run_rank_elasticity(database, SCHEDULE)
            second = run_rank_elasticity(database, SCHEDULE)

            self.assertEqual(first, second)
            self.assertEqual(database.read_bytes(), before)
            self.assertTrue(
                first["source_database"]["unchanged_after_simulation"]
            )
            self.assertEqual(
                first["source_database"]["counts"],
                {
                    "nodes": 10,
                    "edges": 4,
                    "retrievals": 0,
                    "success_feedback": 0,
                    "source_use_state": 0,
                    "delayed_outcomes": 0,
                },
            )

    def test_ceiling_and_threshold_diagnostics_preserve_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.sqlite"
            _create_source_database(database)
            result = run_rank_elasticity(database, SCHEDULE)
        scenarios = {
            scenario["scenario_id"]: scenario for scenario in result["scenarios"]
        }

        ceiling = scenarios["max-normalization-ceiling"]
        self.assertEqual(
            ceiling["diagnosis"]["classification"],
            "edge_changed_but_rank_unchanged",
        )
        self.assertTrue(ceiling["diagnosis"]["fusion_side_ceiling"])
        self.assertIsNone(ceiling["diagnosis"]["rank_flip_threshold"])
        ceiling_target = next(
            case for case in ceiling["cases"] if case["role"] == "relation_target"
        )
        self.assertEqual(
            {checkpoint["rank"] for checkpoint in ceiling_target["checkpoints"]},
            {1},
        )
        self.assertGreater(
            ceiling_target["checkpoints"][-1]["scores"]["graph_raw"],
            ceiling_target["checkpoints"][0]["scores"]["graph_raw"],
        )
        self.assertEqual(
            ceiling_target["checkpoints"][-1]["scores"]["graph_normalized"],
            ceiling_target["checkpoints"][0]["scores"]["graph_normalized"],
        )

        threshold = scenarios["credited-edge-threshold"]
        self.assertEqual(
            threshold["diagnosis"]["classification"], "rank_flip_threshold"
        )
        self.assertEqual(threshold["diagnosis"]["rank_flip_threshold"], 1)
        threshold_target = next(
            case for case in threshold["cases"] if case["role"] == "relation_target"
        )
        self.assertLess(threshold_target["checkpoints"][0]["adjacent_margin"], 0.0)
        self.assertGreater(threshold_target["checkpoints"][-1]["adjacent_margin"], 0.0)

        for scenario in scenarios.values():
            controls = [
                case for case in scenario["cases"] if case["role"] != "relation_target"
            ]
            self.assertEqual(len(controls), 3)
            self.assertTrue(
                all(case["rank_stable_through_schedule"] for case in controls)
            )
            self.assertTrue(
                all(
                    checkpoint["non_target_churn"]["changed_node_count"] == 0
                    for case in controls
                    for checkpoint in case["checkpoints"]
                )
            )

    def test_checkpoint_replays_only_its_registered_feedback_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.sqlite"
            _create_source_database(database)
            result = run_rank_elasticity(database, SCHEDULE)
        for scenario in result["scenarios"]:
            target = next(
                case for case in scenario["cases"] if case["role"] == "relation_target"
            )
            for checkpoint in target["checkpoints"]:
                count = checkpoint["feedback_count"]
                if count == 0:
                    self.assertEqual(checkpoint["changed_edges"], [])
                else:
                    self.assertEqual(len(checkpoint["changed_edges"]), 1)
                    self.assertEqual(
                        checkpoint["changed_edges"][0]["after_reinforced_count"],
                        count,
                    )
                self.assertIn("entry", checkpoint["scores"])
                self.assertIn("graph_raw", checkpoint["scores"])
                self.assertIn("graph_normalized", checkpoint["scores"])
                self.assertIn("final", checkpoint["scores"])
                self.assertTrue(checkpoint["top_k_rank_delta"])

    def test_top_k_rank_delta_includes_entries_and_exits(self) -> None:
        deltas = {
            item["node_id"]: item
            for item in _rank_deltas(
                {"stable": 1, "departing": 2},
                {"stable": 1, "arriving": 2},
            )
        }
        self.assertEqual(deltas["stable"]["top_k_status"], "retained")
        self.assertEqual(deltas["stable"]["delta"], 0)
        self.assertEqual(deltas["departing"]["top_k_status"], "left")
        self.assertEqual(deltas["departing"]["current_rank"], None)
        self.assertLess(deltas["departing"]["delta"], 0)
        self.assertEqual(deltas["arriving"]["top_k_status"], "entered")
        self.assertEqual(deltas["arriving"]["baseline_rank"], None)
        self.assertGreater(deltas["arriving"]["delta"], 0)

    def test_diagnosis_classifies_regression_before_an_earlier_flip(self) -> None:
        records = [
            {
                "feedback_count": count,
                "rank": rank,
                "scores": {"graph_normalized": 0.5, "final": 0.5},
            }
            for count, rank in ((0, 2), (1, 1), (3, 3))
        ]
        diagnosis = _diagnose(
            {"checkpoints": records},
            [{"changed_edges": []} for _ in records],
        )

        self.assertEqual(diagnosis["classification"], "rank_regression")
        self.assertEqual(diagnosis["rank_flip_threshold"], 1)
        self.assertEqual(diagnosis["rank_regression_first_checkpoint"], 3)
        self.assertFalse(diagnosis["rank_stable_through_schedule"])

    def test_schedule_rejects_a_missing_control_role(self) -> None:
        schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(schedule)
        invalid["scenarios"][0]["cases"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every control role"):
                read_rank_elasticity_schedule(path)

    def test_result_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            write_rank_elasticity_result(output, {"schema_version": SCHEMA_VERSION})
            original = output.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                write_rank_elasticity_result(output, {"changed": True})
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_result_writer_exclusively_creates_under_a_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            barrier = threading.Barrier(2)
            original_exists = Path.exists

            def synchronized_exists(path: Path) -> bool:
                if path == output:
                    barrier.wait(timeout=5)
                    return False
                return original_exists(path)

            def attempt(value: str) -> tuple[str, str]:
                try:
                    write_rank_elasticity_result(output, {"writer": value})
                except ValueError:
                    return "refused", value
                return "written", value

            with (
                patch.object(Path, "exists", synchronized_exists),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                outcomes = list(executor.map(attempt, ("first", "second")))

            written = [value for status, value in outcomes if status == "written"]
            refused = [value for status, value in outcomes if status == "refused"]
            self.assertEqual(len(written), 1)
            self.assertEqual(len(refused), 1)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"writer": written[0]},
            )


if __name__ == "__main__":
    unittest.main()

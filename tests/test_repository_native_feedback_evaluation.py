from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.repository_native_feedback_evaluation import (
    ProtocolStop,
    assert_valid_development_result,
    audit_prior_identities,
    build_stage_result,
    load_corpus,
    write_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]


class RepositoryNativeFeedbackEvaluationTest(unittest.TestCase):
    def test_development_result_is_deterministic_and_improves_relation_mrr(self) -> None:
        first = build_stage_result("development", ROOT, allow_holdout=False)
        second = build_stage_result("development", ROOT, allow_holdout=False)
        self.assertEqual(first, second)
        evaluation = first["evaluation"]
        self.assertGreater(
            evaluation["treatment_relation_mrr"], evaluation["baseline_relation_mrr"]
        )
        self.assertEqual(evaluation["control_mutated_edges"], [])
        self.assertEqual(
            set(map(tuple, evaluation["treatment_mutated_edges"])),
            set(map(tuple, evaluation["credited_path"])),
        )

    def test_holdout_requires_development_gate(self) -> None:
        with self.assertRaisesRegex(ProtocolStop, "holdout is unavailable"):
            build_stage_result("holdout", ROOT, allow_holdout=False)
        result = build_stage_result("holdout", ROOT, allow_holdout=True)
        self.assertEqual(result["evaluation"]["split"], "holdout")

    def test_prior_identity_overlap_stops_before_output(self) -> None:
        documents, edges, _ = load_corpus(ROOT / "corpora" / "repository-native-controlled-v2")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = directory / "old.json"
            fixture.write_text(json.dumps({"node_id": "v2-dev-0"}), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolStop, "prior identity overlap"):
                audit_prior_identities(directory, documents, edges)
            self.assertFalse((directory / "development.json").exists())

    def test_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "development.json"
            write_exclusive(output, {"説明": "最初の観測"})
            with self.assertRaisesRegex(ProtocolStop, "exclusive output"):
                write_exclusive(output, {"説明": "再計算"})

    def test_holdout_accepts_only_an_intact_development_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            development = build_stage_result("development", ROOT, allow_holdout=False)
            output = Path(temporary) / "development.result.json"
            write_exclusive(output, development)
            assert_valid_development_result(output)
            altered = json.loads(output.read_text(encoding="utf-8"))
            altered["stage"] = "holdout"
            output.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolStop, "another stage|output hash mismatch"):
                assert_valid_development_result(output)

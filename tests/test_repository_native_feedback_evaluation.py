from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuron_graph_rag.repository_native_feedback_evaluation import (
    ProtocolStop,
    assert_valid_development_result,
    audit_prior_identities,
    build_stage_result,
    load_corpus,
    write_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]


def test_development_result_is_deterministic_and_improves_relation_mrr() -> None:
    first = build_stage_result("development", ROOT, allow_holdout=False)
    second = build_stage_result("development", ROOT, allow_holdout=False)
    assert first == second
    evaluation = first["evaluation"]
    assert evaluation["treatment_relation_mrr"] > evaluation["baseline_relation_mrr"]
    assert evaluation["control_mutated_edges"] == []
    assert set(map(tuple, evaluation["treatment_mutated_edges"])) == set(
        map(tuple, evaluation["credited_path"])
    )


def test_holdout_requires_development_gate() -> None:
    with pytest.raises(ProtocolStop, match="holdout is unavailable"):
        build_stage_result("holdout", ROOT, allow_holdout=False)
    result = build_stage_result("holdout", ROOT, allow_holdout=True)
    assert result["evaluation"]["split"] == "holdout"


def test_prior_identity_overlap_stops_before_output(tmp_path: Path) -> None:
    documents, edges, _ = load_corpus(ROOT / "corpora" / "repository-native-controlled-v2")
    fixture = tmp_path / "old.json"
    fixture.write_text(json.dumps({"node_id": "v2-dev-0"}), encoding="utf-8")
    with pytest.raises(ProtocolStop, match="prior identity overlap"):
        audit_prior_identities(tmp_path, documents, edges)
    output = tmp_path / "development.json"
    assert not output.exists()


def test_output_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "development.json"
    write_exclusive(output, {"説明": "最初の観測"})
    with pytest.raises(ProtocolStop, match="exclusive output"):
        write_exclusive(output, {"説明": "再計算"})


def test_holdout_accepts_only_an_intact_development_result(tmp_path: Path) -> None:
    development = build_stage_result("development", ROOT, allow_holdout=False)
    output = tmp_path / "development.result.json"
    write_exclusive(output, development)
    assert_valid_development_result(output)
    altered = json.loads(output.read_text(encoding="utf-8"))
    altered["stage"] = "holdout"
    output.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ProtocolStop, match="another stage|output hash mismatch"):
        assert_valid_development_result(output)

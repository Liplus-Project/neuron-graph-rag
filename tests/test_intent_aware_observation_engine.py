from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import intent_aware_observation_engine as shared
from neuron_graph_rag import intent_aware_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
V21_EVIDENCE = Path(
    "tests/evidence/github_cross_encoder_precision_v21_observation"
)
V21_CORPUS = Path("tests/fixtures/github_cross_encoder_precision_v21.corpus.json")
V21_QUERIES = Path("tests/fixtures/github_cross_encoder_precision_v21.queries.json")
V21_GOLD = Path("tests/fixtures/github_cross_encoder_precision_v21.gold.json")
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")


def _read(relative: Path) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _v21_spec() -> shared.IntentAwareObservationSpec:
    return shared.IntentAwareObservationSpec(
        protocol_id="github-ngr-cross-encoder-precision-v21",
        fixture_paths=shared.ObservationFixturePaths(
            corpus=V21_CORPUS,
            queries=V21_QUERIES,
            gold=V21_GOLD,
            model_registry=MODEL_REGISTRY,
        ),
        stage_identities=(
            (
                "development",
                "github-ngr-cross-encoder-precision-v20-development-7b6b4f9d",
            ),
            (
                "holdout",
                "github-ngr-cross-encoder-precision-v20-holdout-c31e958a",
            ),
        ),
        models=(
            shared.ModelIdentity(
                kind="base",
                candidate_id="base-intent-aware",
                model_id="BAAI/bge-reranker-base",
                revision="2cfc18c9415c912f9d8155881c133215df768a70",
            ),
            shared.ModelIdentity(
                kind="v2-m3",
                candidate_id="v2-m3-intent-aware",
                model_id="BAAI/bge-reranker-v2-m3",
                revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            ),
        ),
        baseline_evidence_id="current-ngr-prefilter",
    )


def _raw(stage: str) -> dict[tuple[str, str], dict[str, object]]:
    root = V21_EVIDENCE / "raw" / stage
    return {
        (kind, replay): _read(root / f"{kind}-{replay}.json")
        for kind in ("baseline", "base", "v2-m3")
        for replay in ("primary", "replay")
    }


def _behavior_projection(value: dict[str, object]) -> dict[str, object]:
    baseline = value["baseline"]
    candidates = value["candidates"]
    assert isinstance(baseline, dict)
    assert isinstance(candidates, list)
    return {
        "protocol_validity_gates": value["protocol_validity_gates"],
        "baseline": {
            key: baseline[key]
            for key in ("candidate_id", "cases", "quality")
        },
        "candidates": [
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "cases",
                    "quality",
                    "gates",
                    "failed_hard_gate_ids",
                    "all_candidate_gates_pass",
                )
            }
            for candidate in candidates
        ],
        "selected_candidate_id": value["selected_candidate_id"],
        "all_hard_gates_pass": value["all_hard_gates_pass"],
    }


class IntentAwareObservationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = shared.IntentAwareObservationEngine(_v21_spec())

    def test_worker_api_has_no_gold_input(self) -> None:
        signature = inspect.signature(self.engine.build_worker_cases)
        self.assertNotIn("gold", signature.parameters)
        source = inspect.getsource(self.engine.build_worker_cases)
        self.assertNotIn("gold", source.lower())

    def test_shared_engine_does_not_own_lifecycle_or_transport(self) -> None:
        source = inspect.getsource(shared)
        for forbidden in (
            "RankObservationStageContract",
            "ActualCount",
            "dispatch_container_command",
            "run_once",
            "_copy_stage_artifacts",
            "terminal-evidence-manifest",
            '"transport"',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_worker_loader_does_not_open_gold_and_finalizer_does(self) -> None:
        opened: list[Path] = []

        def reader(path: Path) -> dict[str, object]:
            opened.append(path)
            return json.loads(path.read_text(encoding="utf-8"))

        engine = shared.IntentAwareObservationEngine(
            _v21_spec(), object_reader=reader
        )
        fixture = engine.load_worker_fixture(ROOT, "development")
        self.assertEqual(len(fixture.documents), 24)
        self.assertNotIn(ROOT / V21_GOLD, opened)
        engine.load_finalizer_fixture(ROOT, "development")
        self.assertIn(ROOT / V21_GOLD, opened)

    def test_worker_uses_positive_clause_and_production_signals_only(self) -> None:
        fixture = shared.WorkerFixture(
            repository="example/repository",
            documents=(
                {"path": "docs/alpha.md", "text": "alpha"},
                {"path": "docs/beta.md", "text": "beta"},
            ),
            relationships=(),
            queries=(
                {
                    "case_id": "negative",
                    "cohort": "negative_control",
                    "language": "en",
                    "query": "find alpha without beta",
                },
            ),
        )
        searched: list[str] = []
        scored: list[str] = []

        def prefilter(query: str, limit: int) -> list[dict[str, object]]:
            searched.append(query)
            self.assertEqual(limit, 2)
            return [
                {
                    "source_path": "docs/alpha.md",
                    "rank": 1,
                    "ngr_score": 1.0,
                    "relation_paths": [],
                },
                {
                    "source_path": "docs/beta.md",
                    "rank": 2,
                    "ngr_score": 0.5,
                    "relation_paths": [],
                },
            ]

        def score(
            query: str,
            rows: object,
            documents: object,
            model: shared.ModelIdentity,
        ) -> tuple[list[dict[str, object]], int]:
            del rows, documents
            scored.append(query)
            self.assertEqual(model.kind, "base")
            return [
                {
                    "source_path": "docs/alpha.md",
                    "raw_logit": 6.0 if query == "find alpha" else -6.0,
                },
                {
                    "source_path": "docs/beta.md",
                    "raw_logit": -6.0 if query == "find alpha" else 6.0,
                },
            ], 2

        result = self.engine.build_worker_cases(
            fixture,
            "base",
            prefilter_search=prefilter,
            score_query=score,
        )
        self.assertEqual(searched, ["find alpha"])
        self.assertEqual(scored, ["find alpha", "beta"])
        self.assertEqual(result["pair_count"], 4)
        case = result["cases"][0]
        self.assertEqual(
            set(case["production_signals"][0]),
            shared.EXPECTED_PRODUCTION_SIGNAL_KEYS,
        )
        self.assertEqual(case["ranked_hits"][0]["source_path"], "docs/alpha.md")

    def _assert_v21_worker_case_parity(self, stage: str, kind: str) -> None:
        fixture = self.engine.load_worker_fixture(ROOT, stage)
        raw = _raw(stage)
        baseline_cases = raw[("baseline", "primary")]["cases"]
        assert isinstance(baseline_cases, list)
        expected = raw[(kind, "primary")]
        expected_cases = expected["cases"]
        assert isinstance(expected_cases, list)
        prefilter_index = 0

        def prefilter(query: str, limit: int) -> list[dict[str, object]]:
            nonlocal prefilter_index
            fixture_query = fixture.queries[prefilter_index]
            intent = intent_aware_rank_fusion.decompose_query_intent(
                str(fixture_query["query"])
            )
            self.assertEqual(query, intent.positive_query)
            self.assertEqual(limit, len(fixture.documents))
            baseline_case = baseline_cases[prefilter_index]
            assert isinstance(baseline_case, dict)
            prefilter_index += 1
            return [dict(row) for row in baseline_case["ranked_hits"]]

        score_calls: list[tuple[str, list[dict[str, object]]]] = []
        if kind != "baseline":
            for query, case in zip(fixture.queries, expected_cases, strict=True):
                assert isinstance(case, dict)
                signals = case["production_signals"]
                assert isinstance(signals, list)
                intent = intent_aware_rank_fusion.decompose_query_intent(
                    str(query["query"])
                )
                score_calls.append(
                    (
                        intent.positive_query,
                        [
                            {
                                "source_path": row["source_path"],
                                "raw_logit": row["positive_logit"],
                            }
                            for row in signals
                        ],
                    )
                )
                for index, exclusion in enumerate(intent.exclusion_queries):
                    score_calls.append(
                        (
                            exclusion,
                            [
                                {
                                    "source_path": row["source_path"],
                                    "raw_logit": row["exclusion_logits"][index],
                                }
                                for row in signals
                            ],
                        )
                    )
        score_index = 0

        def score(
            query: str,
            rows: object,
            documents: object,
            model: shared.ModelIdentity,
        ) -> tuple[list[dict[str, object]], int]:
            nonlocal score_index
            del rows
            self.assertEqual(model.kind, kind)
            self.assertEqual(documents, fixture.documents)
            expected_query, scores = score_calls[score_index]
            score_index += 1
            self.assertEqual(query, expected_query)
            return scores, len(fixture.documents)

        actual = self.engine.build_worker_cases(
            fixture,
            kind,
            prefilter_search=prefilter,
            score_query=None if kind == "baseline" else score,
        )
        self.assertEqual(actual["cases"], expected_cases)
        self.assertEqual(prefilter_index, len(fixture.queries))
        self.assertEqual(score_index, len(score_calls))
        metrics = expected["metrics"]
        assert isinstance(metrics, dict)
        self.assertEqual(actual["pair_count"], metrics["pair_count"])
        self.assertEqual(actual["model_id"], expected["model_id"])
        self.assertEqual(actual["revision"], expected["revision"])

    def test_v21_worker_case_parity_without_model_execution(self) -> None:
        for stage in shared.STAGES:
            for kind in ("baseline", "base", "v2-m3"):
                with self.subTest(stage=stage, kind=kind):
                    self._assert_v21_worker_case_parity(stage, kind)

    def test_v20_weights_signals_and_gate_ids_are_api_invariants(self) -> None:
        changed = _v21_spec()
        object.__setattr__(
            changed,
            "fusion_config",
            intent_aware_rank_fusion.IntentAwareFusionConfig(
                positive_logit_weight=0.5
            ),
        )
        with self.assertRaisesRegex(ValueError, "fusion weights"):
            changed.validate()
        self.assertEqual(self.engine.spec.protocol_gate_ids, shared.PROTOCOL_GATE_IDS)
        self.assertEqual(
            self.engine.spec.candidate_gate_ids, shared.CANDIDATE_GATE_IDS
        )
        self.assertEqual(
            intent_aware_rank_fusion.PRODUCTION_SIGNAL_KEYS,
            shared.EXPECTED_PRODUCTION_SIGNAL_KEYS,
        )

    def test_protocol_failure_prevents_candidate_gate_evaluation(self) -> None:
        fixture = self.engine.load_finalizer_fixture(ROOT, "development")
        claim = _read(V21_EVIDENCE / "development.claim.json")
        with patch.object(
            self.engine,
            "candidate_gates",
            side_effect=AssertionError("candidate gates ran before protocol validity"),
        ):
            result = self.engine.finalize_stage(
                "development",
                claim=claim,
                claim_sha256="0" * 64,
                raw=_raw("development"),
                fixture=fixture,
                validity=shared.ProtocolValidityInputs(False, True, 24),
            )
        self.assertFalse(result["protocol_validity_pass"])
        self.assertFalse(result["candidate_gates_evaluated"])
        self.assertTrue(
            all(not candidate["gates"] for candidate in result["candidates"])
        )

    def _assert_development_raw_rejected(
        self,
        raw: dict[tuple[str, str], dict[str, object]],
        message: str,
    ) -> None:
        fixture = self.engine.load_finalizer_fixture(ROOT, "development")
        claim = _read(V21_EVIDENCE / "development.claim.json")
        with self.assertRaisesRegex(ValueError, message):
            self.engine.finalize_stage(
                "development",
                claim=claim,
                claim_sha256="0" * 64,
                raw=raw,
                fixture=fixture,
                validity=shared.ProtocolValidityInputs(True, True, 24),
            )

    def test_finalizer_rejects_replay_model_revision_mismatch(self) -> None:
        raw = copy.deepcopy(_raw("development"))
        raw[("base", "replay")]["revision"] = "wrong-revision"
        self._assert_development_raw_rejected(raw, "model identity mismatch")

    def test_finalizer_rejects_baseline_model_identity(self) -> None:
        for key in (("baseline", "primary"), ("baseline", "replay")):
            for field in ("model_id", "revision"):
                with self.subTest(key=key, field=field):
                    raw = copy.deepcopy(_raw("development"))
                    raw[key][field] = "unexpected"
                    self._assert_development_raw_rejected(
                        raw, "model identity mismatch"
                    )

    def test_finalizer_rejects_worker_packet_identity_mismatch(self) -> None:
        mutations = (
            (("base", "primary"), "protocol_id", "wrong-protocol"),
            (("baseline", "replay"), "stage", "holdout"),
            (("base", "primary"), "kind", "v2-m3"),
            (("v2-m3", "replay"), "replay", "primary"),
        )
        for key, field, value in mutations:
            with self.subTest(key=key, field=field):
                raw = copy.deepcopy(_raw("development"))
                raw[key][field] = value
                self._assert_development_raw_rejected(
                    raw, f"{field} identity mismatch"
                )

    def test_finalizer_requires_exact_worker_packet_set(self) -> None:
        missing = copy.deepcopy(_raw("development"))
        del missing[("base", "replay")]
        self._assert_development_raw_rejected(missing, "packet set mismatch")

        extra = copy.deepcopy(_raw("development"))
        extra[("base", "shadow")] = copy.deepcopy(extra[("base", "replay")])
        self._assert_development_raw_rejected(extra, "packet set mismatch")

    def test_v21_reference_behavior_parity_without_model_execution(self) -> None:
        for stage in shared.STAGES:
            with self.subTest(stage=stage):
                expected = _read(V21_EVIDENCE / f"{stage}.observed.json")
                claim = _read(V21_EVIDENCE / f"{stage}.claim.json")
                fixture = self.engine.load_finalizer_fixture(ROOT, stage)
                actual = self.engine.finalize_stage(
                    stage,
                    claim=claim,
                    claim_sha256=str(expected["claim_sha256"]),
                    raw=_raw(stage),
                    fixture=fixture,
                    validity=shared.ProtocolValidityInputs(True, True, 24),
                )
                self.assertEqual(
                    _behavior_projection(actual), _behavior_projection(expected)
                )
                projection = json.dumps(_behavior_projection(actual), sort_keys=True)
                self.assertNotIn("latency_ms", projection)
                self.assertNotIn("peak_rss_bytes", projection)
                self.assertNotIn('"metrics"', projection)
                self.assertNotIn('"performance"', projection)


if __name__ == "__main__":
    unittest.main()

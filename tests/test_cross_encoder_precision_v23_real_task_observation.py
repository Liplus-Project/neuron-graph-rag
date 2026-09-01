from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path

from neuron_graph_rag import cross_encoder_precision_v19_performance_observation as v19
from neuron_graph_rag import cross_encoder_precision_v23_real_task_observation as v23
from neuron_graph_rag import intent_aware_observation_engine as shared
from neuron_graph_rag import intent_aware_observation_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: Path) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class CrossEncoderPrecisionV23RealTaskObservationTests(unittest.TestCase):
    def test_four_arms_and_exact_worker_slots_are_frozen(self) -> None:
        self.assertEqual(
            v23.ENGINE_SPEC.worker_kinds(),
            ("default", "baseline", "base", "v2-m3"),
        )
        self.assertEqual(len(v23.WORKERS), 8)
        self.assertEqual(v23.STAGE_CONTRACT.worker_slots_per_stage, 8)
        self.assertEqual(
            v23.ENGINE_SPEC.baseline_evidence_id,
            "original-full-query-ngr-default",
        )
        self.assertEqual(v23.ENGINE_SPEC.baseline_query_mode, "full")
        self.assertEqual(
            v23.ENGINE_SPEC.ablation_arms,
            (
                shared.RetrievalArmIdentity(
                    "baseline",
                    "positive-clause-ngr-prefilter-ablation",
                    "positive",
                ),
            ),
        )

    def test_selection_policy_is_literal_and_not_candidate_order(self) -> None:
        self.assertEqual(
            v23.ENGINE_SPEC.selection_policy,
            "lowest-development-primary-latency",
        )
        source = inspect.getsource(shared.IntentAwareObservationEngine.finalize_stage)
        self.assertIn('candidate["metrics"]["primary"]', source)
        self.assertIn("min(passing, key=latency_key)", source)

    def test_real_corpus_provenance_and_content_hashes_are_complete(self) -> None:
        corpus = _read(v23.CORPUS)
        provenance = corpus["acquisition_provenance"]
        documents = corpus["documents"]
        assert isinstance(provenance, dict)
        assert isinstance(documents, list)
        self.assertEqual(provenance["commit"], v23.FREEZE_COMMIT)
        self.assertEqual(provenance["read_only"], True)
        self.assertEqual(len(documents), 12)
        for row in documents:
            assert isinstance(row, dict)
            path = str(row["path"])
            observed = hashlib.sha256(str(row["text"]).encode()).hexdigest()
            self.assertEqual(provenance["content_sha256"][path], observed)

    def test_development_holdout_and_predecessor_separation_validate(self) -> None:
        result = v23._validate_protocol_fixtures(ROOT)
        self.assertEqual(result["query_count"], 16)
        self.assertEqual(result["corpus_document_count"], 12)
        self.assertEqual(result["v19_v21_exact_query_reuse_count"], 0)
        self.assertLess(
            result["v19_v21_max_normalized_similarity"],
            result["v19_v21_similarity_limit_exclusive"],
        )

    def test_worker_surface_cannot_load_gold(self) -> None:
        source = inspect.getsource(v23._container_worker)
        self.assertNotIn("GOLD", source)
        self.assertNotIn("load_finalizer_fixture", source)
        signature = inspect.signature(v23._container_worker)
        self.assertNotIn("gold", signature.parameters)

    def test_version_module_reuses_engine_and_existing_lifecycle(self) -> None:
        version_source = inspect.getsource(v23)
        runtime_source = inspect.getsource(runtime)
        self.assertLessEqual(len(version_source.splitlines()), 300)
        self.assertTrue(
            issubclass(v23.V23RankObservationSpec, v19.V19RankObservationSpec)
        )
        self.assertIn("self.engine.build_worker_cases", runtime_source)
        self.assertIn("self.engine.finalize_stage", runtime_source)
        self.assertNotIn("github-ngr-cross-encoder-precision-v23", runtime_source)
        for duplicate in (
            "def _quality(",
            "def _candidate_gates(",
            "def _protocol_gates(",
        ):
            self.assertNotIn(duplicate, version_source)
            self.assertNotIn(duplicate, runtime_source)

    def test_prebuild_contract_is_result_free_before_one_shot(self) -> None:
        result = v23.validate_prebuild(ROOT)
        self.assertEqual(result["status"], "prebuild_contract_valid")
        self.assertEqual(result["corpus_document_count"], 12)
        self.assertEqual(result["development_claim_count"], 0)
        self.assertEqual(result["holdout_claim_count"], 0)
        self.assertEqual(result["observed_result_count"], 0)
        self.assertEqual(result["performance"], "not assessed")


if __name__ == "__main__":
    unittest.main()

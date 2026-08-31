from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from neuron_graph_rag import cross_encoder_precision_v20_intent_aware_freeze as v20
from neuron_graph_rag.intent_aware_rank_fusion import (
    IntentAwareFusionConfig,
    decompose_query_intent,
    fuse_intent_aware_ranks,
)

ROOT = Path(__file__).resolve().parents[1]


def _signal(
    source_path: str,
    rank: int,
    score: float,
    positive: float,
    exclusions: list[float] | None = None,
    relation_paths: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "source_path": source_path,
        "prefilter_rank": rank,
        "prefilter_score": score,
        "positive_logit": positive,
        "exclusion_logits": exclusions or [],
        "relation_paths": relation_paths or [],
    }


class IntentAwareRankFusionTests(unittest.TestCase):
    def test_english_exclusion_intent_is_decomposed(self) -> None:
        intent = decompose_query_intent(
            "find cache freeze docs without v8 runtime excluding benchmark results"
        )
        self.assertEqual(intent.positive_query, "find cache freeze docs")
        self.assertEqual(
            intent.exclusion_queries, ("v8 runtime", "benchmark results")
        )

    def test_japanese_exclusion_and_relation_intent_are_decomposed(self) -> None:
        intent = decompose_query_intent("関連するcache文書、v8 runtimeを除外")
        self.assertEqual(intent.positive_query, "関連するcache文書")
        self.assertEqual(intent.exclusion_queries, ("v8 runtime",))
        self.assertTrue(intent.relation_requested)

    def test_exclusion_only_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive retrieval intent"):
            decompose_query_intent("exclude v8 runtime")

    def test_evaluation_only_signal_keys_are_rejected(self) -> None:
        signal = _signal("docs/a.md", 1, 1.0, 1.0)
        signal["forbidden_paths"] = ["docs/a.md"]
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            fuse_intent_aware_ranks("cache", [signal])

    def test_exclusion_logit_count_must_match_query_decomposition(self) -> None:
        with self.assertRaisesRegex(ValueError, "clause count"):
            fuse_intent_aware_ranks(
                "cache without v8",
                [_signal("docs/a.md", 1, 1.0, 1.0, [])],
            )

    def test_exclusion_penalty_uses_logits_without_gold(self) -> None:
        rows = [
            _signal("docs/excluded.md", 1, 1.0, 3.0, [6.0]),
            _signal("docs/clean.md", 2, 0.8, 2.0, [-6.0]),
        ]
        ranked = fuse_intent_aware_ranks("cache without v8", rows)
        self.assertEqual(ranked[0]["source_path"], "docs/clean.md")

    def test_relation_bonus_preserves_exact_paths(self) -> None:
        path = {
            "seed_path": "docs/seed.md",
            "target_path": "docs/related.md",
            "edge_type": "informs",
            "step_count": 1,
        }
        rows = [
            _signal("docs/plain.md", 1, 1.0, 1.0),
            _signal("docs/related.md", 2, 0.9, 1.0, relation_paths=[path]),
        ]
        ranked = fuse_intent_aware_ranks("related cache decision", rows)
        self.assertEqual(ranked[0]["source_path"], "docs/related.md")
        self.assertEqual(ranked[0]["relation_paths"], [path])

    def test_relation_path_target_must_match_candidate(self) -> None:
        path = {
            "seed_path": "docs/seed.md",
            "target_path": "docs/other.md",
            "edge_type": "informs",
            "step_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "target"):
            fuse_intent_aware_ranks(
                "related cache",
                [_signal("docs/a.md", 1, 1.0, 1.0, relation_paths=[path])],
            )

    def test_prefilter_identities_and_ranks_are_validated_before_fusion(self) -> None:
        duplicate = [
            _signal("docs/a.md", 1, 1.0, 1.0),
            _signal("docs/a.md", 1, 0.5, 0.5),
        ]
        with self.assertRaisesRegex(ValueError, "source identities"):
            fuse_intent_aware_ranks("cache", duplicate)

    def test_ties_use_source_path_and_top_k_is_frozen(self) -> None:
        rows = [
            _signal(f"docs/{letter}.md", index, 1.0, 1.0)
            for index, letter in enumerate("fedcba", 1)
        ]
        ranked = fuse_intent_aware_ranks(
            "cache",
            rows,
            config=IntentAwareFusionConfig(prefilter_rank_weight=0.0),
        )
        self.assertEqual(
            [row["source_path"] for row in ranked],
            ["docs/a.md", "docs/b.md", "docs/c.md", "docs/d.md", "docs/e.md"],
        )


class IntentAwareFreezeTests(unittest.TestCase):
    def test_prebuild_contract_is_result_free(self) -> None:
        report = v20.validate_prebuild(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["model_forward_inference_count"], 0)
        self.assertEqual(report["observed_result_count"], 0)
        self.assertEqual(report["performance"], "not assessed")

    def test_v19_predecessor_registry_is_exact_and_immutable(self) -> None:
        manifest = json.loads((ROOT / v20.MANIFEST).read_text(encoding="utf-8"))
        registry = manifest["predecessor_immutable_sha256"]
        self.assertEqual(len(registry), 30)
        self.assertTrue(
            all("v19" in path or "rank_observation_stage_contract.py" in path for path in registry)
        )
        for relative, expected in registry.items():
            self.assertEqual(v20.sha256_file(ROOT / relative), expected)

    def test_gate_ownership_is_disjoint(self) -> None:
        contract = json.loads((ROOT / v20.GATE_OWNERSHIP).read_text(encoding="utf-8"))
        protocol = set(contract["protocol_validity_gates"])
        candidate = set(contract["candidate_controllable_gates"])
        self.assertFalse(protocol & candidate)
        self.assertIn("relation-source-edge-only-provenance", protocol)
        self.assertNotIn("relation-source-edge-only-provenance", candidate)

    def test_positive_case_non_regression_is_not_relaxed(self) -> None:
        contract = json.loads((ROOT / v20.GATE_OWNERSHIP).read_text(encoding="utf-8"))
        self.assertEqual(
            contract["positive_case_non_regression"],
            {
                "comparator": "candidate_rank <= baseline_rank",
                "when": "baseline expected source is in top_k",
                "missing_candidate": "fail",
                "cohort_average_can_mask_case_regression": False,
            },
        )

    def test_future_identities_are_fresh_and_v19_is_diagnostic_only(self) -> None:
        identities = json.loads(
            (ROOT / v20.FUTURE_IDENTITIES).read_text(encoding="utf-8")
        )
        self.assertNotEqual(identities["development"], identities["holdout"])
        self.assertEqual(identities["v19_case_role"], "diagnostic-design-input-only")
        self.assertFalse(identities["v19_result_packet_reuse_allowed"])

    def test_result_free_counts_are_all_zero(self) -> None:
        audit = json.loads((ROOT / v20.RESULT_FREE_AUDIT).read_text(encoding="utf-8"))
        for key, value in audit.items():
            if key.endswith("_count"):
                self.assertEqual(value, 0, key)

    def test_docs_keep_claim_boundary(self) -> None:
        text = (ROOT / "docs/cross-encoder-precision-observation-v20.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("performanceは`not assessed`", text)
        self.assertIn("gold expected path", text)
        self.assertIn("positive per-case rank non-regression", text)

    def test_manifest_mutation_fails_closed(self) -> None:
        manifest = json.loads((ROOT / v20.MANIFEST).read_text(encoding="utf-8"))
        mutated = copy.deepcopy(manifest)
        mutated["performance"] = "observed"
        self.assertNotEqual(mutated, manifest)


if __name__ == "__main__":
    unittest.main()

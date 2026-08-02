from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import COHORTS, _edge_key, _metrics, _rank, _simple_path
from .d1_fixture import load_fixture, read_fixture
from .engine import EngineConfig, NeuronGraphRAG


EXPERIMENT_SCHEMA_VERSION = 1
SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS = {1, 2, 3, 4}


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
        raise ValueError("Unsupported dynamics experiment schema version")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Experiment manifest must contain variants")
    if len(variants) > int(manifest["maximum_variants"]):
        raise ValueError("Experiment variant count exceeds the frozen maximum")
    ids = [str(variant.get("id", "")) for variant in variants]
    if any(not variant_id for variant_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Experiment variant ids must be non-empty and unique")
    if ids[0] != "current" or variants[0]["family"] != "current_positive_additive":
        raise ValueError("The first experiment variant must be current")
    base = manifest_path.parent
    for split_name in ("development", "holdout"):
        split = manifest[split_name]
        for field in ("fixture", "gold"):
            actual = _canonical_sha256(base / split[field])
            if actual != split[f"{field}_sha256"]:
                raise ValueError(
                    f"Frozen {split_name} {field} hash mismatch: {actual}"
                )
    for split_name in ("development", "holdout"):
        split = manifest[split_name]
        if "provenance" in split and (
            _canonical_sha256(base / split["provenance"])
            != split["provenance_sha256"]
        ):
            raise ValueError(f"Frozen {split_name} provenance hash mismatch")
    development_paths = _doc_paths(base / manifest["development"]["fixture"])
    holdout_paths = _doc_paths(base / manifest["holdout"]["fixture"])
    overlap = sorted(development_paths & holdout_paths)
    if overlap:
        raise ValueError(f"Development and holdout doc paths overlap: {overlap!r}")
    if schema_version in {2, 3, 4}:
        expected_baselines = (
            ["current", "recurrent-balanced"]
            if schema_version == 2
            else (["current", "bm25-only"] if schema_version == 3 else ["current"])
        )
        if ids[: len(expected_baselines)] != expected_baselines:
            raise ValueError(
                f"Schema v{schema_version} requires its frozen baselines"
            )
        if len(variants) != int(manifest["maximum_variants"]):
            raise ValueError("Experiment requires the exact frozen variant count")
        audit = manifest["contamination_audit"]
        audit_path = base / audit["artifact"]
        if _canonical_sha256(audit_path) != audit["artifact_sha256"]:
            raise ValueError("Frozen contamination audit hash mismatch")
        if not _read_json(audit_path).get("passed"):
            raise ValueError("Frozen contamination audit did not pass")
        candidate_ids = [
            str(item) for item in manifest["selection_rule"]["candidate_ids"]
        ]
        if ids != [*manifest["baselines"], *candidate_ids]:
            raise ValueError("Experiment variant order differs from the frozen roles")
        audit_payload = _read_json(audit_path)
        audit_inputs = audit_payload["inputs"]
        for split_name in ("development", "holdout"):
            split = manifest[split_name]
            for field in ("fixture", "gold"):
                if audit_inputs[f"{split_name}_{field}_sha256"] != split[
                    f"{field}_sha256"
                ]:
                    raise ValueError("Contamination audit input differs from manifest")
            _validate_gold_membership(base / split["fixture"], base / split["gold"])
    return manifest


def run_development(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    split = manifest["development"]
    base = manifest_path.parent
    gold = _read_gold(base / split["gold"])
    variants = [
        _evaluate_variant(
            base / split["fixture"],
            gold,
            _variant_config(manifest, variant),
            variant,
            int(manifest["shared_config"]["limit"]),
        )
        for variant in manifest["variants"]
    ]
    selection = (
        _select_local_development(variants)
        if int(manifest["schema_version"]) == 2
        else (
            _select_anchored_development(variants)
            if int(manifest["schema_version"]) == 3
            else (
                _select_fusion_development(variants)
                if int(manifest["schema_version"]) == 4
                else _select_development(variants)
            )
        )
    )
    return {
        "schema_version": int(manifest["schema_version"]),
        "experiment_id": manifest["experiment_id"],
        "stage": "development",
        "manifest_sha256": _canonical_sha256(manifest_path),
        "inputs": {
            "fixture_sha256": split["fixture_sha256"],
            "gold_sha256": split["gold_sha256"],
        },
        "variant_count": len(variants),
        "variants": variants,
        "selection": selection,
        "holdout_status": (
            "not_opened_candidate_selected"
            if selection["selected_variant_id"] != "current"
            else "not_opened_no_candidate"
        ),
    }


def run_holdout(
    manifest_path: str | Path,
    development_result_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    development_result_path = Path(development_result_path)
    manifest = read_manifest(manifest_path)
    development = _read_json(development_result_path)
    expected_manifest_hash = _canonical_sha256(manifest_path)
    if development.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("Development result does not match the frozen manifest")
    selected_id = str(development["selection"]["selected_variant_id"])
    if selected_id == "current":
        raise ValueError("Stop rule forbids opening holdout when no candidate passed")
    variant_by_id = {str(item["id"]): item for item in manifest["variants"]}
    if selected_id not in variant_by_id:
        raise ValueError("Development result selected an unknown variant")

    split = manifest["holdout"]
    base = manifest_path.parent
    gold = _read_gold(base / split["gold"])
    schema_version = int(manifest["schema_version"])
    holdout_variant_ids = (
        ("current", "recurrent-balanced", selected_id)
        if schema_version == 2
        else (
            ("current", "bm25-only", selected_id)
            if schema_version == 3
            else ("current", selected_id)
        )
    )
    evaluated = []
    for variant_id in holdout_variant_ids:
        variant = variant_by_id[variant_id]
        evaluated.append(
            _evaluate_variant(
                base / split["fixture"],
                gold,
                _variant_config(manifest, variant),
                variant,
                int(manifest["shared_config"]["limit"]),
            )
        )
    current = evaluated[0]
    selected = evaluated[-1]
    paths_match = _paths_match(selected)
    feedback_isolated = _feedback_isolated(selected)
    if schema_version == 2:
        recurrent = evaluated[1]
        gate = _local_candidate_gate(selected, current, recurrent)
        no_cohort_regression = gate["direct_non_regression"] and gate[
            "negative_non_regression"
        ]
        adopted = all(gate.values())
    elif schema_version == 3:
        gate = _anchored_candidate_gate(selected, current)
        no_cohort_regression = gate["direct_non_regression"] and gate[
            "negative_non_regression"
        ]
        adopted = all(gate.values())
    elif schema_version == 4:
        gate = _fusion_candidate_gate(selected, current)
        no_cohort_regression = gate["direct_non_regression"] and gate[
            "negative_non_regression"
        ]
        adopted = all(gate.values())
    else:
        no_cohort_regression = all(
            selected["metrics"]["cohorts"][cohort]["mean_reciprocal_rank"]
            >= current["metrics"]["cohorts"][cohort]["mean_reciprocal_rank"]
            for cohort in COHORTS
        )
        gate = {
            "no_cohort_regression": no_cohort_regression,
            "all_relation_paths_match": paths_match,
            "feedback_isolated": feedback_isolated,
        }
        adopted = all(gate.values())
    return {
        "schema_version": int(manifest["schema_version"]),
        "experiment_id": manifest["experiment_id"],
        "stage": "holdout",
        "manifest_sha256": expected_manifest_hash,
        "development_result_sha256": _canonical_sha256(development_result_path),
        "holdout_open_count": 1,
        "inputs": {
            "fixture_sha256": split["fixture_sha256"],
            "gold_sha256": split["gold_sha256"],
            "provenance_sha256": split["provenance_sha256"],
        },
        "variants": evaluated,
        "decision": {
            "selected_variant_id": selected_id,
            "no_cohort_regression": no_cohort_regression,
            "all_relation_paths_match": paths_match,
            "feedback_isolated": feedback_isolated,
            "gate": gate,
            "adopted": adopted,
            "default_variant_id": selected_id if adopted else "current",
        },
    }


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evaluate_variant(
    fixture_path: Path,
    gold: dict[str, Any],
    config: EngineConfig,
    variant: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        for case in gold["cases"]:
            trace = engine.search(case["query"], limit=limit, now=1_000.0)
            target = next(
                hit for hit in trace.hits if hit.node.node_id == case["expected_node_id"]
            )
            observed_paths = [_simple_path(path) for path in target.explain()["paths"]]
            expected_path = case.get("expected_path", [])
            cases.append(
                {
                    "id": case["id"],
                    "cohort": case["cohort"],
                    "acceptable_rank": int(case["acceptable_rank"]),
                    "rank": _rank(trace, case["expected_node_id"]),
                    "expected_path": expected_path,
                    "observed_paths": observed_paths,
                    "path_matched": (
                        any(path["steps"] == expected_path for path in observed_paths)
                        if expected_path
                        else None
                    ),
                    "scores": target.explain()["scores"],
                    "ranks": target.explain()["ranks"],
                    "fusion": target.explain()["fusion"],
                    "ranked_hits": [
                        {
                            "node_id": hit.node.node_id,
                            "scores": hit.explain()["scores"],
                            "ranks": hit.explain()["ranks"],
                            "fusion": hit.explain()["fusion"],
                        }
                        for hit in trace.hits
                    ],
                    "formula_recomputed": _trace_formula_recomputable(
                        trace, config
                    ),
                    "diagnostics": trace.diagnostics,
                }
            )
    feedback = _run_feedback(
        fixture_path,
        gold["cases"],
        config,
        limit,
        _feedback_case_id(gold["cases"]),
    )
    metrics = _metrics(cases)
    diagnostics = {
        "mean_expansions": sum(
            int(case["diagnostics"]["expansions"]) for case in cases
        )
        / len(cases),
        "mean_steps": sum(int(case["diagnostics"]["steps"]) for case in cases)
        / len(cases),
        "mean_activation_total": sum(
            float(case["diagnostics"]["activation_total"]) for case in cases
        )
        / len(cases),
        "mean_active_path_count": sum(
            int(case["diagnostics"].get("active_path_count", 0)) for case in cases
        )
        / len(cases),
        "competition_set_count": sum(
            len(case["diagnostics"].get("competition_sets", [])) for case in cases
        ),
        "stop_reasons": dict(
            sorted(Counter(case["diagnostics"]["stop_reason"] for case in cases).items())
        ),
        "entry_anchor_invariant": all(
            bool(case["diagnostics"].get("entry_anchor_invariant"))
            for case in cases
        ),
        "graph_signal_excludes_zero_hop": all(
            case["diagnostics"].get("graph_signal_excludes_zero_hop") is not False
            for case in cases
        ),
        "final_order_recomputable": all(
            bool(case["diagnostics"].get("final_order_recomputable"))
            and bool(case["formula_recomputed"])
            and _case_order_recomputable(case)
            for case in cases
        ),
        "graph_normalizations": sorted(
            {
                str(case["diagnostics"].get("graph_normalization"))
                for case in cases
            }
        ),
        "final_fusion_strategies": sorted(
            {
                str(case["diagnostics"].get("final_fusion_strategy"))
                for case in cases
            }
        ),
    }
    return {
        "id": variant["id"],
        "family": variant["family"],
        "parameters": variant["parameters"],
        "structural_complexity": int(variant["structural_complexity"]),
        "metrics": metrics,
        "cases": cases,
        "explanations": [
            {
                "id": case["id"],
                "matched": case["path_matched"],
                "expected_path": case["expected_path"],
                "observed_paths": case["observed_paths"],
            }
            for case in cases
            if case["cohort"] == "relation"
        ],
        "feedback": feedback,
        "diagnostics": diagnostics,
    }


def _run_feedback(
    fixture_path: Path,
    cases: list[dict[str, Any]],
    config: EngineConfig,
    limit: int,
    feedback_case_id: str,
) -> dict[str, Any]:
    case_by_id = {str(case["id"]): case for case in cases}
    feedback_case = case_by_id[feedback_case_id]
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before_ranks: dict[str, int] = {}
        feedback_trace = None
        for case in cases:
            trace = engine.search(case["query"], limit=limit, now=2_000.0)
            before_ranks[case["id"]] = _rank(trace, case["expected_node_id"])
            if case["id"] == feedback_case_id:
                feedback_trace = trace
        if feedback_trace is None:
            raise ValueError("Feedback trace was not produced")
        before_edges = {_edge_key(edge): edge.weight for edge in engine.store.list_edges()}
        receipt = engine.record_success(
            feedback_trace.trace_id,
            [feedback_case["expected_node_id"]],
            now=2_001.0,
        )
        after_edges = {_edge_key(edge): edge.weight for edge in engine.store.list_edges()}
        changed_edges = [
            {
                "source_id": key[0],
                "target_id": key[1],
                "edge_type": key[2],
                "old_weight": before_edges[key],
                "new_weight": after_edges[key],
            }
            for key in sorted(before_edges)
            if before_edges[key] != after_edges[key]
        ]
        credited_edges = [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "old_weight": edge.old_weight,
                "new_weight": edge.new_weight,
            }
            for edge in receipt.reinforced_edges
        ]
        credited_keys = {
            (edge["source_id"], edge["target_id"], edge["edge_type"])
            for edge in credited_edges
        }
        uncredited = [
            edge
            for edge in changed_edges
            if (edge["source_id"], edge["target_id"], edge["edge_type"])
            not in credited_keys
        ]
        after_ranks: dict[str, int] = {}
        for case in cases:
            trace = engine.search(case["query"], limit=limit, now=2_002.0)
            after_ranks[case["id"]] = _rank(trace, case["expected_node_id"])
        non_target_changes = [
            {
                "id": case_id,
                "before_rank": before_ranks[case_id],
                "after_rank": after_ranks[case_id],
            }
            for case_id in sorted(before_ranks)
            if case_id != feedback_case_id
            and before_ranks[case_id] != after_ranks[case_id]
        ]
        return {
            "case_id": feedback_case_id,
            "target_node_id": feedback_case["expected_node_id"],
            "target_rank_before": before_ranks[feedback_case_id],
            "target_rank_after": after_ranks[feedback_case_id],
            "credited_edges": credited_edges,
            "changed_edges": changed_edges,
            "uncredited_edge_changes": uncredited,
            "non_target_rank_changes": non_target_changes,
        }


def _select_development(variants: list[dict[str, Any]]) -> dict[str, Any]:
    current = variants[0]
    current_relation = _cohort_mrr(current, "relation")
    current_negative = _cohort_mrr(current, "negative_control")
    eligible: list[dict[str, Any]] = []
    for variant in variants:
        relation = _cohort_mrr(variant, "relation")
        negative = _cohort_mrr(variant, "negative_control")
        variant["relative_to_current"] = {
            "relation_mrr_delta": relation - current_relation,
            "negative_control_mrr_delta": negative - current_negative,
        }
        variant["candidate_gate_passed"] = (
            variant["id"] != "current"
            and relation >= current_relation
            and negative >= current_negative
            and (relation > current_relation or negative > current_negative)
        )
        if variant["candidate_gate_passed"]:
            eligible.append(variant)

    pareto: list[dict[str, Any]] = []
    for candidate in eligible:
        dominated_by = [
            other["id"]
            for other in eligible
            if other is not candidate and _dominates(other, candidate)
        ]
        candidate["dominated_by"] = sorted(dominated_by)
        candidate["pareto_status"] = "dominated" if dominated_by else "frontier"
        if not dominated_by:
            pareto.append(candidate)
    for variant in variants:
        if "pareto_status" not in variant:
            variant["pareto_status"] = "failed_gate"
            variant["dominated_by"] = []

    if not pareto:
        return {
            "selected_variant_id": "current",
            "reason": "no_variant_passed_candidate_gate",
            "eligible_variant_ids": [],
            "pareto_frontier_variant_ids": [],
        }
    selected = sorted(
        pareto,
        key=lambda variant: (
            -min(
                _cohort_mrr(variant, cohort)
                for cohort in COHORTS
            ),
            float(variant["diagnostics"]["mean_expansions"]),
            int(variant["structural_complexity"]),
            str(variant["id"]),
        ),
    )[0]
    return {
        "selected_variant_id": selected["id"],
        "reason": "predeclared_pareto_and_tie_break",
        "eligible_variant_ids": sorted(variant["id"] for variant in eligible),
        "pareto_frontier_variant_ids": sorted(variant["id"] for variant in pareto),
    }


def _select_local_development(
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(variant["id"]): variant for variant in variants}
    current = by_id["current"]
    recurrent = by_id["recurrent-balanced"]
    candidates: list[dict[str, Any]] = []
    for variant in variants:
        variant["relative_to_baselines"] = {
            "relation_mrr_delta_from_current": (
                _cohort_mrr(variant, "relation")
                - _cohort_mrr(current, "relation")
            ),
            "relation_mrr_delta_from_recurrent_balanced": (
                _cohort_mrr(variant, "relation")
                - _cohort_mrr(recurrent, "relation")
            ),
            "direct_mrr_delta_from_current": (
                _cohort_mrr(variant, "direct_lookup")
                - _cohort_mrr(current, "direct_lookup")
            ),
            "negative_mrr_delta_from_current": (
                _cohort_mrr(variant, "negative_control")
                - _cohort_mrr(current, "negative_control")
            ),
        }
        gate = _local_candidate_gate(variant, current, recurrent)
        variant["candidate_gate"] = gate
        variant["candidate_gate_passed"] = (
            variant["id"] not in {"current", "recurrent-balanced"}
            and all(gate.values())
        )
        if variant["candidate_gate_passed"]:
            candidates.append(variant)

    if not candidates:
        return {
            "selected_variant_id": "current",
            "reason": "no_local_variant_passed_frozen_gate",
            "eligible_variant_ids": [],
        }
    selected = sorted(
        candidates,
        key=lambda variant: (
            -min(_cohort_mrr(variant, cohort) for cohort in COHORTS),
            -_cohort_mrr(variant, "relation"),
            float(variant["diagnostics"]["mean_expansions"]),
            int(variant["structural_complexity"]),
            str(variant["id"]),
        ),
    )[0]
    return {
        "selected_variant_id": selected["id"],
        "reason": "predeclared_local_gate_and_tie_break",
        "eligible_variant_ids": sorted(
            str(variant["id"]) for variant in candidates
        ),
    }


def _local_candidate_gate(
    candidate: dict[str, Any],
    current: dict[str, Any],
    recurrent: dict[str, Any],
) -> dict[str, bool]:
    relation = _cohort_mrr(candidate, "relation")
    return {
        "relation_strictly_above_current": relation
        > _cohort_mrr(current, "relation"),
        "relation_strictly_above_recurrent_balanced": relation
        > _cohort_mrr(recurrent, "relation"),
        "direct_non_regression": _cohort_mrr(candidate, "direct_lookup")
        >= _cohort_mrr(current, "direct_lookup"),
        "negative_non_regression": _cohort_mrr(candidate, "negative_control")
        >= _cohort_mrr(current, "negative_control"),
        "all_relation_paths_match": _paths_match(candidate),
        "feedback_isolated": _feedback_isolated(candidate),
    }


def _select_anchored_development(
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(variant["id"]): variant for variant in variants}
    current = by_id["current"]
    candidates: list[dict[str, Any]] = []
    for variant in variants:
        variant["relative_to_current"] = {
            f"{cohort}_mrr_delta": (
                _cohort_mrr(variant, cohort) - _cohort_mrr(current, cohort)
            )
            for cohort in COHORTS
        }
        gate = _anchored_candidate_gate(variant, current)
        variant["candidate_gate"] = gate
        variant["candidate_gate_passed"] = (
            variant["id"] not in {"current", "bm25-only"}
            and all(gate.values())
        )
        if variant["candidate_gate_passed"]:
            candidates.append(variant)

    if not candidates:
        return {
            "selected_variant_id": "current",
            "reason": "no_anchored_variant_passed_frozen_gate",
            "eligible_variant_ids": [],
        }
    selected = sorted(
        candidates,
        key=lambda variant: (
            -_cohort_mrr(variant, "relation"),
            -min(_cohort_mrr(variant, cohort) for cohort in COHORTS),
            float(variant["diagnostics"]["mean_expansions"]),
            int(variant["structural_complexity"]),
            str(variant["id"]),
        ),
    )[0]
    return {
        "selected_variant_id": selected["id"],
        "reason": "predeclared_anchored_gate_and_tie_break",
        "eligible_variant_ids": sorted(
            str(variant["id"]) for variant in candidates
        ),
    }


def _anchored_candidate_gate(
    candidate: dict[str, Any], current: dict[str, Any]
) -> dict[str, bool]:
    return {
        "relation_strictly_above_current": _cohort_mrr(candidate, "relation")
        > _cohort_mrr(current, "relation"),
        "direct_non_regression": _cohort_mrr(candidate, "direct_lookup")
        >= _cohort_mrr(current, "direct_lookup"),
        "negative_non_regression": _cohort_mrr(candidate, "negative_control")
        >= _cohort_mrr(current, "negative_control"),
        "all_relation_paths_match": _paths_match(candidate),
        "feedback_isolated": _feedback_isolated(candidate),
        "entry_anchor_invariant": bool(
            candidate["diagnostics"].get("entry_anchor_invariant")
        ),
        "graph_signal_excludes_zero_hop": bool(
            candidate["diagnostics"].get("graph_signal_excludes_zero_hop")
        ),
    }


def _select_fusion_development(
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    current = next(variant for variant in variants if variant["id"] == "current")
    candidates: list[dict[str, Any]] = []
    for variant in variants:
        improved_cases = _individually_improved_cases(variant, current)
        variant["relative_to_current"] = {
            f"{cohort}_mrr_delta": (
                _cohort_mrr(variant, cohort) - _cohort_mrr(current, cohort)
            )
            for cohort in COHORTS
        }
        variant["relative_to_current"]["individually_improved_case_ids"] = (
            improved_cases
        )
        gate = _fusion_candidate_gate(variant, current)
        variant["candidate_gate"] = gate
        variant["candidate_gate_passed"] = (
            variant["id"] != "current" and all(gate.values())
        )
        if variant["candidate_gate_passed"]:
            candidates.append(variant)

    if not candidates:
        return {
            "selected_variant_id": "current",
            "reason": "no_fusion_variant_passed_frozen_gate",
            "eligible_variant_ids": [],
        }
    selected = sorted(
        candidates,
        key=lambda variant: (
            -_cohort_mrr(variant, "relation"),
            -min(_cohort_mrr(variant, cohort) for cohort in COHORTS),
            -len(_individually_improved_cases(variant, current)),
            float(variant["diagnostics"]["mean_expansions"]),
            int(variant["structural_complexity"]),
            str(variant["id"]),
        ),
    )[0]
    return {
        "selected_variant_id": selected["id"],
        "reason": "predeclared_fusion_gate_and_tie_break",
        "eligible_variant_ids": sorted(
            str(variant["id"]) for variant in candidates
        ),
    }


def _fusion_candidate_gate(
    candidate: dict[str, Any], current: dict[str, Any]
) -> dict[str, bool]:
    current_ranks = {str(case["id"]): int(case["rank"]) for case in current["cases"]}
    candidate_cases = {
        str(case["id"]): case for case in candidate["cases"]
    }
    control_case_non_regression = all(
        int(case["rank"]) <= current_ranks[case_id]
        for case_id, case in candidate_cases.items()
        if case["cohort"] in {"direct_lookup", "negative_control"}
    )
    relation_case_improvement = any(
        int(case["rank"]) < current_ranks[case_id]
        for case_id, case in candidate_cases.items()
        if case["cohort"] == "relation"
    )
    return {
        "relation_strictly_above_current": _cohort_mrr(candidate, "relation")
        > _cohort_mrr(current, "relation"),
        "direct_non_regression": _cohort_mrr(candidate, "direct_lookup")
        >= _cohort_mrr(current, "direct_lookup"),
        "negative_non_regression": _cohort_mrr(candidate, "negative_control")
        >= _cohort_mrr(current, "negative_control"),
        "individual_control_rank_non_regression": control_case_non_regression,
        "individual_relation_rank_improvement": relation_case_improvement,
        "all_relation_paths_match": _paths_match(candidate),
        "feedback_isolated": _feedback_isolated(candidate),
        "entry_anchor_invariant": bool(
            candidate["diagnostics"].get("entry_anchor_invariant")
        ),
        "graph_signal_excludes_zero_hop": bool(
            candidate["diagnostics"].get("graph_signal_excludes_zero_hop")
        ),
        "final_order_recomputable": bool(
            candidate["diagnostics"].get("final_order_recomputable")
        ),
    }


def _individually_improved_cases(
    candidate: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    current_ranks = {str(case["id"]): int(case["rank"]) for case in current["cases"]}
    return sorted(
        str(case["id"])
        for case in candidate["cases"]
        if int(case["rank"]) < current_ranks[str(case["id"])]
    )


def _case_order_recomputable(case: dict[str, Any]) -> bool:
    ranked_hits = case.get("ranked_hits", [])
    if not ranked_hits:
        return False
    for hit in ranked_hits:
        fusion = hit["fusion"]
        if abs(
            float(fusion["entry_component"])
            + float(fusion["graph_component"])
            - float(fusion["final"])
        ) > 1e-12:
            return False
    observed = [str(hit["node_id"]) for hit in ranked_hits]
    recomputed = [
        str(hit["node_id"])
        for hit in sorted(
            ranked_hits,
            key=lambda hit: (
                -float(hit["fusion"]["final"]),
                str(hit["node_id"]),
            ),
        )
    ]
    return observed == recomputed


def _trace_formula_recomputable(trace: Any, config: EngineConfig) -> bool:
    positive_graph_count = int(
        trace.diagnostics.get("positive_graph_node_count", 0)
    )
    recomputed: list[tuple[str, float]] = []
    for hit in trace.hits:
        if config.final_fusion_strategy == "linear":
            total_weight = config.entry_weight + config.graph_weight
            entry_component = (
                hit.entry_score * config.entry_weight / total_weight
            )
            graph_component = (
                hit.normalized_graph_activation
                * config.graph_weight
                / total_weight
            )
        else:
            entry_component = config.entry_weight / (
                config.rrf_k + hit.entry_rank
            )
            graph_component = 0.0
            if hit.graph_rank is not None:
                graph_component = config.graph_weight * (
                    1.0 / (config.rrf_k + hit.graph_rank)
                    - 1.0
                    / (
                        config.rrf_k
                        + positive_graph_count
                        + 1
                    )
                )
        if (
            abs(entry_component - hit.entry_fusion_component) > 1e-12
            or abs(graph_component - hit.graph_fusion_component) > 1e-12
            or abs(entry_component + graph_component - hit.final_score) > 1e-12
        ):
            return False
        recomputed.append((hit.node.node_id, entry_component + graph_component))
    expected_order = [
        node_id
        for node_id, _ in sorted(
            recomputed, key=lambda item: (-item[1], item[0])
        )
    ]
    return expected_order == [hit.node.node_id for hit in trace.hits]


def _paths_match(variant: dict[str, Any]) -> bool:
    return bool(variant["explanations"]) and all(
        item["matched"] for item in variant["explanations"]
    )


def _feedback_isolated(variant: dict[str, Any]) -> bool:
    feedback = variant["feedback"]
    return (
        bool(feedback["credited_edges"])
        and not feedback["uncredited_edge_changes"]
        and not feedback["non_target_rank_changes"]
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_axes = (
        _cohort_mrr(left, "relation"),
        _cohort_mrr(left, "negative_control"),
    )
    right_axes = (
        _cohort_mrr(right, "relation"),
        _cohort_mrr(right, "negative_control"),
    )
    return all(a >= b for a, b in zip(left_axes, right_axes, strict=True)) and any(
        a > b for a, b in zip(left_axes, right_axes, strict=True)
    )


def _cohort_mrr(variant: dict[str, Any], cohort: str) -> float:
    return float(variant["metrics"]["cohorts"][cohort]["mean_reciprocal_rank"])


def _variant_config(
    manifest: dict[str, Any], variant: dict[str, Any]
) -> EngineConfig:
    values = {
        key: value
        for key, value in manifest["shared_config"].items()
        if key != "limit"
    }
    values["activation_strategy"] = variant["family"]
    values.update(variant["parameters"])
    return EngineConfig(**values)


def _read_gold(path: Path) -> dict[str, Any]:
    gold = _read_json(path)
    cases = gold.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Experiment gold must contain cases")
    counts = Counter(str(case.get("cohort", "")) for case in cases)
    if set(counts) != set(COHORTS) or any(counts[cohort] == 0 for cohort in COHORTS):
        raise ValueError("Experiment gold must cover every cohort")
    for case in cases:
        if not str(case.get("source_url", "")).startswith("https://github.com/"):
            raise ValueError(f"Gold case {case.get('id')} lacks a public source URL")
    return gold


def _validate_gold_membership(fixture_path: Path, gold_path: Path) -> None:
    fixture = read_fixture(fixture_path)
    node_ids = {str(node["node_id"]) for node in fixture["nodes"]}
    edges = {
        (str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"]))
        for edge in fixture["edges"]
    }
    gold = _read_gold(gold_path)
    counts = Counter(str(case["cohort"]) for case in gold["cases"])
    if len(set(counts.values())) != 1:
        raise ValueError("Schema v2 gold cohorts must be balanced")
    for case in gold["cases"]:
        if str(case["expected_node_id"]) not in node_ids:
            raise ValueError(f"Gold target is outside fixture: {case['id']}")
        for step in case.get("expected_path", []):
            key = (
                str(step["source_id"]),
                str(step["target_id"]),
                str(step["edge_type"]),
            )
            if key not in edges:
                raise ValueError(f"Gold path edge is outside fixture: {case['id']}")


def _feedback_case_id(cases: list[dict[str, Any]]) -> str:
    return str(next(case["id"] for case in cases if case["cohort"] == "relation"))


def _doc_paths(path: Path) -> set[str]:
    fixture = read_fixture(path)
    return {
        str(node.get("metadata", {}).get("doc_path", ""))
        for node in fixture["nodes"]
        if str(node.get("metadata", {}).get("doc_path", ""))
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(path: Path) -> str:
    value = _read_json(path)
    canonical = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()

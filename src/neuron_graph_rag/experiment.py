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


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
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
    holdout = manifest["holdout"]
    if _canonical_sha256(base / holdout["provenance"]) != holdout["provenance_sha256"]:
        raise ValueError("Frozen holdout provenance hash mismatch")
    development_paths = _doc_paths(base / manifest["development"]["fixture"])
    holdout_paths = _doc_paths(base / manifest["holdout"]["fixture"])
    overlap = sorted(development_paths & holdout_paths)
    if overlap:
        raise ValueError(f"Development and holdout doc paths overlap: {overlap!r}")
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
    selection = _select_development(variants)
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
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
    evaluated = []
    for variant_id in ("current", selected_id):
        variant = variant_by_id[variant_id]
        evaluated.append(
            _evaluate_variant(
                base / split["fixture"],
                gold,
                _variant_config(manifest, variant),
                variant,
                len(read_fixture(base / split["fixture"])["nodes"]),
            )
        )
    current, selected = evaluated
    no_cohort_regression = all(
        selected["metrics"]["cohorts"][cohort]["mean_reciprocal_rank"]
        >= current["metrics"]["cohorts"][cohort]["mean_reciprocal_rank"]
        for cohort in COHORTS
    )
    paths_match = all(item["matched"] for item in selected["explanations"])
    feedback = selected["feedback"]
    feedback_isolated = (
        bool(feedback["credited_edges"])
        and not feedback["uncredited_edge_changes"]
        and not feedback["non_target_rank_changes"]
    )
    adopted = no_cohort_regression and paths_match and feedback_isolated
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
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
        "stop_reasons": dict(
            sorted(Counter(case["diagnostics"]["stop_reason"] for case in cases).items())
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

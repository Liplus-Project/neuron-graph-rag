from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .d1_fixture import load_fixture, read_fixture
from .engine import EngineConfig, NeuronGraphRAG


LONGITUDINAL_FEEDBACK_SCHEMA_VERSION = 1
HORIZONS = (0, 1, 2, 3)
PATH_IDENTITY_FIELDS = ("source_id", "target_id", "edge_type")


def project_relation_path(steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Discard runtime-only relation-step fields before identity comparison."""
    return [{field: str(step[field]) for field in PATH_IDENTITY_FIELDS} for step in steps]


def read_longitudinal_feedback_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != LONGITUDINAL_FEEDBACK_SCHEMA_VERSION:
        raise ValueError("Unsupported longitudinal feedback schema version")
    if manifest.get("candidate_id") != "trace-credited-longitudinal-feedback":
        raise ValueError("Longitudinal feedback candidate is not frozen")
    base = manifest_path.parent
    seen_clusters: set[str] = set()
    for stage in ("development", "holdout"):
        clusters = manifest.get(stage, {}).get("clusters")
        if not isinstance(clusters, list) or len(clusters) < 3:
            raise ValueError(f"{stage} must freeze at least three corpus clusters")
        cohorts = {str(cluster.get("cohort")) for cluster in clusters}
        if not {"headroom", "ceiling"} <= cohorts:
            raise ValueError(f"{stage} must contain headroom and ceiling cohorts")
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id"))
            if not cluster_id or cluster_id in seen_clusters:
                raise ValueError("Cluster identifiers must be globally disjoint")
            seen_clusters.add(cluster_id)
            for field in ("fixture", "gold", "provenance"):
                artifact = base / str(cluster.get(field, ""))
                if _canonical_sha256(artifact) != cluster.get(f"{field}_sha256"):
                    raise ValueError(f"Frozen {stage} {cluster_id} {field} hash mismatch")
            _validate_cluster(
                cluster_id,
                str(cluster.get("cohort")),
                base / str(cluster["fixture"]),
                base / str(cluster["gold"]),
                base / str(cluster["provenance"]),
            )
    audit = manifest.get("contamination_audit", {})
    audit_path = base / str(audit.get("artifact", ""))
    if _canonical_sha256(audit_path) != audit.get("artifact_sha256"):
        raise ValueError("Frozen longitudinal contamination audit hash mismatch")
    if not _read_json(audit_path).get("passed"):
        raise ValueError("Frozen longitudinal contamination audit did not pass")
    return manifest


def run_longitudinal_feedback_development(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_longitudinal_feedback_manifest(manifest_path)
    result = _evaluate_split(manifest_path, manifest, "development")
    passed = all(result["gate"].values())
    return {
        "schema_version": LONGITUDINAL_FEEDBACK_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "development",
        "manifest_sha256": _canonical_sha256(manifest_path),
        **result,
        "gate_passed": passed,
        "holdout_status": "not_opened_candidate_selected" if passed else "not_opened_no_candidate",
    }


def run_longitudinal_feedback_holdout(
    path: str | Path, development_result_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_longitudinal_feedback_manifest(manifest_path)
    development_path = Path(development_result_path)
    development = _read_json(development_path)
    if not development.get("gate_passed"):
        raise ValueError("Stop rule forbids opening longitudinal feedback holdout")
    expected = manifest_path.parent / str(manifest["outputs"]["development"])
    if development_path.resolve() != expected.resolve():
        raise ValueError("Holdout requires the registered development result path")
    if development.get("manifest_sha256") != _canonical_sha256(manifest_path):
        raise ValueError("Development result does not match frozen longitudinal manifest")
    result = _evaluate_split(manifest_path, manifest, "holdout")
    passed = all(result["gate"].values())
    return {
        "schema_version": LONGITUDINAL_FEEDBACK_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "stage": "holdout",
        "manifest_sha256": _canonical_sha256(manifest_path),
        "development_result_sha256": _canonical_sha256(development_path),
        "holdout_open_count": 1,
        **result,
        "decision": {
            "adopted": False,
            "reason": "frozen_longitudinal_gate_passed_no_default_or_generalization_claim"
            if passed
            else "longitudinal_feedback_failed_frozen_holdout_gate",
        },
    }


def write_longitudinal_feedback_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evaluate_split(path: Path, manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    base = path.parent
    config, limit = _experiment_config(manifest)
    clusters: list[dict[str, Any]] = []
    for cluster in manifest[stage]["clusters"]:
        fixture = base / str(cluster["fixture"])
        gold = _read_json(base / str(cluster["gold"]))
        control = _run_cluster(fixture, gold, config, limit, mutate=False)
        treatment = _run_cluster(fixture, gold, config, limit, mutate=True)
        replay = _run_cluster(fixture, gold, config, limit, mutate=True)
        clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cohort": cluster["cohort"],
                "inputs": {field: cluster[f"{field}_sha256"] for field in ("fixture", "gold", "provenance")},
                "control": control,
                "treatment": treatment,
                "replay": replay,
            }
        )
    headroom = [cluster for cluster in clusters if cluster["cohort"] == "headroom"]
    ceiling = [cluster for cluster in clusters if cluster["cohort"] == "ceiling"]
    metrics = {
        "horizon_relation_mrr": {
            str(horizon): {
                "control": _aggregate_mrr(clusters, "control", horizon),
                "treatment": _aggregate_mrr(clusters, "treatment", horizon),
                "headroom_control": _aggregate_mrr(headroom, "control", horizon),
                "headroom_treatment": _aggregate_mrr(headroom, "treatment", horizon),
                "ceiling_control": _aggregate_mrr(ceiling, "control", horizon),
                "ceiling_treatment": _aggregate_mrr(ceiling, "treatment", horizon),
            }
            for horizon in HORIZONS
        }
    }
    final = metrics["horizon_relation_mrr"]["3"]
    gate = {
        "fixed_horizons_present": all(_has_horizons(cluster["control"]) and _has_horizons(cluster["treatment"]) for cluster in clusters),
        "headroom_h0_control_is_below_ceiling": all(_relation_mrr(cluster["control"], 0) < 1.0 for cluster in headroom),
        "ceiling_h0_control_equals_ceiling": all(_relation_mrr(cluster["control"], 0) == 1.0 for cluster in ceiling),
        "headroom_final_aggregate_mrr_strictly_improves": final["headroom_treatment"] > final["headroom_control"],
        "headroom_clusters_do_not_regress": all(
            _relation_mrr(cluster["treatment"], 3) >= _relation_mrr(cluster["control"], 3) for cluster in headroom
        ),
        "ceiling_clusters_do_not_regress": all(
            _relation_mrr(cluster["treatment"], 3) >= _relation_mrr(cluster["control"], 3) for cluster in ceiling
        ),
        "projected_paths_match_at_every_horizon": all(_paths_match(cluster["treatment"]) for cluster in clusters),
        "direct_lexical_controls_do_not_regress": all(_controls_do_not_regress(cluster, "direct_lexical") for cluster in clusters),
        "directional_negative_controls_remain_empty": all(_negative_controls_remain_empty(cluster) for cluster in clusters),
        "credited_only_mutation_and_control_no_mutation": all(
            cluster["treatment"]["feedback"]["credited_only"] and cluster["control"]["feedback"]["no_edge_mutation"]
            for cluster in clusters
        ),
        "same_schedule_replay_is_deterministic": all(cluster["treatment"] == cluster["replay"] for cluster in clusters),
        "contamination_audit_passed": True,
    }
    return {
        "clusters": clusters,
        "metrics": metrics,
        "ceiling_policy": {
            "headroom": "strict final-horizon aggregate MRR improvement with per-cluster non-regression",
            "ceiling": "non-regression only; a pass does not evidence an additional rank improvement",
            "ceiling_rank_improvement_observable": any(
                _relation_mrr(cluster["treatment"], 3) > _relation_mrr(cluster["control"], 3) for cluster in ceiling
            ),
        },
        "gate": gate,
    }


def _run_cluster(
    fixture_path: Path, gold: dict[str, Any], config: EngineConfig, limit: int, *, mutate: bool
) -> dict[str, Any]:
    schedule = {int(event["horizon"]): event for event in gold["feedback_schedule"]}
    credited: list[dict[str, Any]] = []
    with NeuronGraphRAG(config=config) as engine:
        load_fixture(engine, fixture_path)
        before = _edge_snapshot(engine)
        horizons: list[dict[str, Any]] = []
        for horizon in HORIZONS:
            if horizon:
                event = schedule[horizon]
                channels = engine.search_channels(event["query"], limit=limit, now=event["now"])
                if event["channel"] != "relation" or event["used_node_id"] not in {hit.node.node_id for hit in channels.relation.hits}:
                    raise ValueError("Frozen feedback event was not retrieved by its relation trace")
                if mutate:
                    receipt = engine.record_success(
                        channels.relation.trace_id, [event["used_node_id"]], now=float(event["now"]) + 0.1
                    )
                    credited.extend(_reinforced(edge) for edge in receipt.reinforced_edges)
                else:
                    engine.store.apply_success_feedback(
                        f"control-{horizon}", channels.relation.trace_id, float(event["now"]) + 0.1,
                        [event["used_node_id"]], (),
                    )
            horizons.append({"horizon": horizon, "cases": [_score_case(engine, case, limit) for case in gold["scoring_cases"]]})
        after = _edge_snapshot(engine)
    changed = _edge_changes(before, after)
    credited_keys = {(edge["source_id"], edge["target_id"], edge["edge_type"]) for edge in credited}
    return {
        "horizons": horizons,
        "feedback": {
            "recorded": True,
            "feedback_count": len(schedule),
            "credited_edges": credited,
            "edge_changes": changed,
            "no_edge_mutation": not changed,
            "credited_only": bool(credited) and all(
                (edge["source_id"], edge["target_id"], edge["edge_type"]) in credited_keys for edge in changed
            ),
        },
    }


def _score_case(engine: NeuronGraphRAG, case: dict[str, Any], limit: int) -> dict[str, Any]:
    channels = engine.search_channels(case["query"], limit=limit, now=case["now"])
    expected = str(case["expected_node_id"])
    lexical_hits = [hit.node.node_id for hit in channels.lexical.hits]
    relation_hits = [hit.node.node_id for hit in channels.relation.hits]
    observed_paths = [
        path["steps"]
        for hit in channels.relation.hits
        if hit.node.node_id == expected
        for path in hit.explain()["paths"]
    ]
    projected_paths = [project_relation_path(path) for path in observed_paths]
    expected_path = case.get("expected_path", [])
    return {
        "id": case["id"],
        "role": case["role"],
        "expected_node_id": expected,
        "lexical_rank": _rank(lexical_hits, expected, limit),
        "relation_rank": _rank(relation_hits, expected, limit),
        "relation_found": expected in relation_hits,
        "relation_hits": relation_hits,
        "path_matched": any(path == expected_path for path in projected_paths)
        if expected_path
        else bool(case.get("expected_relation_empty")) and not relation_hits,
        "observed_paths": observed_paths,
        "projected_observed_paths": projected_paths,
    }


def _validate_cluster(cluster_id: str, cohort: str, fixture_path: Path, gold_path: Path, provenance_path: Path) -> None:
    fixture = read_fixture(fixture_path)
    gold = _read_json(gold_path)
    provenance = _read_json(provenance_path)
    if gold.get("cluster_id") != cluster_id or provenance.get("cluster_id") != cluster_id:
        raise ValueError("Cluster artifact identities do not agree")
    if cohort not in {"headroom", "ceiling"} or gold.get("cohort") != cohort:
        raise ValueError("Cluster cohort is invalid")
    nodes = {str(node["node_id"]) for node in fixture["nodes"]}
    edges = {(str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"])) for edge in fixture["edges"]}
    schedule = gold.get("feedback_schedule")
    if not isinstance(schedule, list) or [event.get("horizon") for event in schedule] != [1, 2, 3]:
        raise ValueError("Feedback schedule must freeze horizons one through three")
    cases = gold.get("scoring_cases")
    if not isinstance(cases, list) or {case.get("role") for case in cases} != {"relation", "direct_lexical", "directional_negative"}:
        raise ValueError("Gold must freeze relation and both controls")
    feedback_queries = {str(event.get("query")) for event in schedule}
    if feedback_queries & {str(case.get("query")) for case in cases}:
        raise ValueError("Feedback queries must be separate from scoring queries")
    for event in schedule:
        if event.get("channel") != "relation" or str(event.get("used_node_id")) not in nodes:
            raise ValueError("Feedback event is invalid")
    for case in cases:
        if str(case.get("expected_node_id")) not in nodes or not str(case.get("source_url", "")).startswith("https://github.com/"):
            raise ValueError("Gold case target or source URL is invalid")
        for step in case.get("expected_path", []):
            if set(step) != set(PATH_IDENTITY_FIELDS):
                raise ValueError("Gold path identity must contain endpoint and edge type only")
            if tuple(str(step[field]) for field in PATH_IDENTITY_FIELDS) not in edges:
                raise ValueError("Expected relation path is outside fixture")


def _experiment_config(manifest: dict[str, Any]) -> tuple[EngineConfig, int]:
    values = dict(manifest["shared_config"])
    return EngineConfig(**{key: value for key, value in values.items() if key != "limit"}), int(values["limit"])


def _aggregate_mrr(clusters: list[dict[str, Any]], arm: str, horizon: int) -> float:
    ranks = [
        case["relation_rank"]
        for cluster in clusters
        for row in cluster[arm]["horizons"]
        if row["horizon"] == horizon
        for case in row["cases"]
        if case["role"] == "relation"
    ]
    return _mrr(ranks)


def _relation_mrr(group: dict[str, Any], horizon: int) -> float:
    return _mrr([case["relation_rank"] for row in group["horizons"] if row["horizon"] == horizon for case in row["cases"] if case["role"] == "relation"])


def _has_horizons(group: dict[str, Any]) -> bool:
    return [row["horizon"] for row in group["horizons"]] == list(HORIZONS)


def _paths_match(group: dict[str, Any]) -> bool:
    return all(case["path_matched"] for row in group["horizons"] for case in row["cases"] if case["role"] == "relation")


def _controls_do_not_regress(cluster: dict[str, Any], role: str) -> bool:
    for left, right in zip(cluster["control"]["horizons"], cluster["treatment"]["horizons"], strict=True):
        left_case = next(case for case in left["cases"] if case["role"] == role)
        right_case = next(case for case in right["cases"] if case["role"] == role)
        if right_case["lexical_rank"] > left_case["lexical_rank"]:
            return False
    return True


def _negative_controls_remain_empty(cluster: dict[str, Any]) -> bool:
    return all(
        case["relation_hits"] == []
        for arm in ("control", "treatment")
        for row in cluster[arm]["horizons"]
        for case in row["cases"]
        if case["role"] == "directional_negative"
    )


def _rank(hits: list[str], node_id: str, limit: int) -> int:
    return next((index for index, value in enumerate(hits, start=1) if value == node_id), limit + 1)


def _mrr(ranks: list[int]) -> float:
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _edge_snapshot(engine: NeuronGraphRAG) -> dict[tuple[str, str, str], tuple[float, int]]:
    return {(edge.source_id, edge.target_id, edge.edge_type): (edge.weight, edge.reinforced_count) for edge in engine.store.list_edges()}


def _edge_changes(before: dict[tuple[str, str, str], tuple[float, int]], after: dict[tuple[str, str, str], tuple[float, int]]) -> list[dict[str, Any]]:
    return [{"source_id": key[0], "target_id": key[1], "edge_type": key[2], "before_weight": before[key][0], "after_weight": after[key][0], "before_reinforced_count": before[key][1], "after_reinforced_count": after[key][1]} for key in sorted(before) if before[key] != after[key]]


def _reinforced(edge: Any) -> dict[str, Any]:
    return {"source_id": edge.source_id, "target_id": edge.target_id, "edge_type": edge.edge_type, "old_weight": edge.old_weight, "new_weight": edge.new_weight}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_sha256(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256((json.dumps(_read_json(Path(path)), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()

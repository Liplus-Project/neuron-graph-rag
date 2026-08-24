from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import engine as engine_module
from .engine import EngineConfig, NeuronGraphRAG
from .models import SearchHit
from .precision_control import PrecisionControl
from .precision_control_evaluation import (
    PROTOCOL_ID,
    ROOT,
    archive_stage,
    evaluate_result_payload,
    load_protocol,
    register_stage_claim,
    sha256_bytes,
    verify_phase_state,
    verify_protocol_commit,
    verify_result_payload,
    write_json_exclusive,
    write_stage_result,
)

PROTOCOL_COMMIT = "c1577cad5753bdafe9abf301bb60b1787a64927f"
OBSERVATION_ARCHIVE = "tests/evidence/github_precision_control_observation_v1"
ERROR_FIELDS = (
    "protocol_id",
    "protocol_commit",
    "stage",
    "claim_sha256",
    "exception_type",
    "message",
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def preflight(root: Path = ROOT) -> dict[str, Any]:
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    lifecycle_phase = verify_phase_state(protocol)
    phase = verify_observation_phase(protocol)
    registered_stage_count = sum(value == "archived" for value in phase.values())
    for stage in ("development", "holdout"):
        _reject_failed_retry(root, stage)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "phase": phase,
        "lifecycle_phase": lifecycle_phase,
        "registered_stage_execution_count": registered_stage_count,
        "post_observation_stage_reexecution_count": 0,
        "shared_database_opened": False,
        "existing_experiment_database_opened": False,
        "github_rag_mcp_called": False,
        "feedback_or_outcome_recorded": False,
    }


def run_stage(stage: str, root: Path = ROOT) -> dict[str, Any]:
    result = _execute_registered_stage(stage, root)
    relocate_stage_archive(stage, root)
    verify_observation_phase(load_protocol(root))
    return result


def _execute_registered_stage(stage: str, root: Path = ROOT) -> dict[str, Any]:
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    _reject_observed_retry(root, stage)
    _reject_failed_retry(root, stage)
    claim_path = register_stage_claim(stage, PROTOCOL_COMMIT, root)
    claim_raw = claim_path.read_bytes()
    try:
        documents = _load_documents(protocol)
        baseline_raw = _execute_baseline(protocol, stage, documents)
        candidate_raws = _execute_candidates(protocol, stage, documents)
        result = evaluate_result_payload(
            protocol,
            stage,
            claim_raw,
            baseline_raw,
            candidate_raws,
        )
        verify_result_payload(protocol, stage, result, claim_raw)
        write_stage_result(stage, result, root)
        archive_stage(stage, root)
        verify_phase_state(load_protocol(root))
        return result
    except BaseException as error:
        _archive_error(root, stage, claim_path, claim_raw, error)
        raise


def run_conditional(root: Path = ROOT) -> dict[str, Any]:
    development = _execute_registered_stage("development", root)
    outcome: dict[str, Any] = {"development": development, "holdout": None}
    if development["all_hard_gates_pass"] is True:
        outcome["holdout"] = _execute_registered_stage("holdout", root)
        relocate_stage_archive("holdout", root)
    relocate_stage_archive("development", root)
    protocol = load_protocol(root)
    verify_phase_state(protocol)
    verify_observation_phase(protocol)
    return outcome


def relocate_stage_archive(stage: str, root: Path = ROOT) -> Path:
    protocol = load_protocol(root)
    outputs = _mapping(_mapping(_mapping(protocol, "manifest"), "outputs"), stage)
    lifecycle_paths = {
        "claim": root / _string(outputs, "archive_claim"),
        "result": root / _string(outputs, "archive_result"),
        "lifecycle_transport": root / _string(outputs, "transport"),
    }
    runtime_paths: dict[str, str | None] = {
        "claim": _string(outputs, "runtime_claim"),
        "result": _string(outputs, "runtime_result"),
        "lifecycle_transport": None,
    }
    final_paths = _final_archive_paths(root, stage)
    if final_paths["archive_transport"].exists():
        raise FileExistsError(f"{stage} observation archive already exists")
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError(f"{stage} observation archive is partial")
    if not all(path.exists() for path in lifecycle_paths.values()):
        raise ValueError("complete lifecycle archive is required before relocation")

    raw = {key: path.read_bytes() for key, path in lifecycle_paths.items()}
    result = json.loads(raw["result"].decode("utf-8", errors="strict"))
    verify_result_payload(protocol, stage, result, raw["claim"])
    lifecycle_transport = json.loads(
        raw["lifecycle_transport"].decode("utf-8", errors="strict")
    )
    if [row.get("sha256") for row in _rows(lifecycle_transport, "files")] != [
        sha256_bytes(raw["claim"]),
        sha256_bytes(raw["result"]),
    ]:
        raise ValueError("lifecycle transport does not bind claim/result bytes")

    final_paths["claim"].parent.mkdir(parents=True, exist_ok=True)
    for key in ("claim", "result", "lifecycle_transport"):
        os.replace(lifecycle_paths[key], final_paths[key])
        if final_paths[key].read_bytes() != raw[key]:
            raise ValueError(f"byte identity failed during archival: {key}")
    write_json_exclusive(
        final_paths["archive_transport"],
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "stage": stage,
            "reason": (
                "phase-boundary archival avoids the frozen absence-test and "
                "observation-output path collision"
            ),
            "runtime_verified": True,
            "files": [
                {
                    "kind": key,
                    "runtime_path": runtime_paths[key],
                    "lifecycle_archive_path": str(
                        lifecycle_paths[key].relative_to(root)
                    ).replace("\\", "/"),
                    "final_archive_path": str(
                        final_paths[key].relative_to(root)
                    ).replace("\\", "/"),
                    "sha256": sha256_bytes(raw[key]),
                    "byte_identity": final_paths[key].read_bytes() == raw[key],
                }
                for key in ("claim", "result", "lifecycle_transport")
            ],
        },
    )
    return final_paths["archive_transport"]


def verify_observation_phase(protocol: Mapping[str, Any]) -> dict[str, str]:
    root = _path(protocol, "root")
    phases: dict[str, str] = {}
    for stage in ("development", "holdout"):
        paths = _final_archive_paths(root, stage)
        if not any(path.exists() for path in paths.values()):
            phases[stage] = "unobserved"
            continue
        if not all(path.exists() for path in paths.values()):
            raise ValueError(f"{stage} observation archive is incomplete")
        raw = {key: path.read_bytes() for key, path in paths.items()}
        result = json.loads(raw["result"].decode("utf-8", errors="strict"))
        verify_result_payload(protocol, stage, result, raw["claim"])
        transport = json.loads(
            raw["archive_transport"].decode("utf-8", errors="strict")
        )
        if (
            transport.get("protocol_id") != PROTOCOL_ID
            or transport.get("protocol_commit") != PROTOCOL_COMMIT
            or transport.get("stage") != stage
            or transport.get("runtime_verified") is not True
        ):
            raise ValueError("observation archive transport identity mismatch")
        rows = _rows(transport, "files")
        keys = ("claim", "result", "lifecycle_transport")
        outputs = _mapping(_mapping(_mapping(protocol, "manifest"), "outputs"), stage)
        runtime_paths = (
            _string(outputs, "runtime_claim"),
            _string(outputs, "runtime_result"),
            None,
        )
        if len(rows) != len(keys):
            raise ValueError("observation archive transport cardinality mismatch")
        for key, runtime_path, row in zip(keys, runtime_paths, rows, strict=True):
            if (
                row.get("kind") != key
                or row.get("runtime_path") != runtime_path
                or row.get("sha256") != sha256_bytes(raw[key])
                or row.get("byte_identity") is not True
                or row.get("final_archive_path")
                != str(paths[key].relative_to(root)).replace("\\", "/")
            ):
                raise ValueError("observation archive transport hash mismatch")
        lifecycle_transport = json.loads(
            raw["lifecycle_transport"].decode("utf-8", errors="strict")
        )
        if [row.get("sha256") for row in _rows(lifecycle_transport, "files")] != [
            sha256_bytes(raw["claim"]),
            sha256_bytes(raw["result"]),
        ]:
            raise ValueError("lifecycle transport claim/result hash mismatch")
        phases[stage] = "archived"
    if phases["holdout"] == "archived":
        development = json.loads(
            _final_archive_paths(root, "development")["result"].read_text(
                encoding="utf-8", errors="strict"
            )
        )
        if development.get("all_hard_gates_pass") is not True:
            raise ValueError("holdout archive requires passing development")
    return phases


def _load_documents(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    corpus = _mapping(protocol, "corpus")
    repository = _string(corpus, "repository")
    commit = _string(corpus, "commit")
    rows: list[dict[str, Any]] = []
    for document in _rows(corpus, "documents"):
        path = _string(document, "path")
        raw = _git_bytes(_path(protocol, "root"), commit, path)
        expected = _string(document, "content_sha256")
        if sha256_bytes(raw) != expected:
            raise ValueError(f"source content hash mismatch: {path}")
        rows.append(
            {
                "node_id": f"github:{repository}:doc:{path}",
                "text": raw.decode("utf-8", errors="strict"),
                "metadata": {
                    "repository": repository,
                    "commit": commit,
                    "path": path,
                    "content_sha256": expected,
                },
            }
        )
    return rows


def _execute_baseline(
    protocol: Mapping[str, Any], stage: str, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = _execute_arm(protocol, stage, documents, EngineConfig())
    replay = _execute_arm(protocol, stage, documents, EngineConfig())
    return {
        "baseline_id": "current-ngr",
        "cases": primary["cases"],
        "state": _combine_state(primary, replay),
    }


def _execute_candidates(
    protocol: Mapping[str, Any], stage: str, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    candidates = _rows(_mapping(protocol, "candidates"), "candidates")
    for candidate in candidates:
        control = PrecisionControl.from_mapping(dict(candidate))
        primary = _execute_arm(
            protocol,
            stage,
            documents,
            EngineConfig(precision_control=control),
        )
        replay = _execute_arm(
            protocol,
            stage,
            documents,
            EngineConfig(precision_control=control),
        )
        result.append(
            {
                "candidate_id": control.candidate_id,
                "cases": primary["cases"],
                "explanations": primary["explanations"],
                "state": _combine_state(primary, replay),
            }
        )
    return result


def _execute_arm(
    protocol: Mapping[str, Any],
    stage: str,
    documents: list[dict[str, Any]],
    config: EngineConfig,
) -> dict[str, Any]:
    root = _path(protocol, "root")
    with tempfile.TemporaryDirectory(prefix="ngr-pc-observation-") as directory:
        database = Path(directory) / "observation.sqlite3"
        database_id = hashlib.sha256(
            str(database.resolve()).encode("utf-8")
        ).hexdigest()
        with NeuronGraphRAG(database, config=config) as engine:
            _index_corpus(engine, protocol, documents)
            edges_before = _edge_state(engine)
            feedback_before = engine.store.count_feedback()
            cases: list[dict[str, Any]] = []
            explanations: list[dict[str, Any]] = []
            queries = _rows(_mapping(_mapping(protocol, "queries"), "stages"), stage)
            request = _mapping(_mapping(protocol, "queries"), "request")
            limit = _integer(request, "limit")
            now = _number(request, "now")
            for query in queries:
                with _capture_constructed_hits() as captured:
                    trace = engine.search(_string(query, "query"), limit=limit, now=now)
                ranked = sorted(
                    captured,
                    key=lambda hit: (-hit.final_score, hit.node.node_id),
                )
                if len(ranked) != len(documents):
                    raise ValueError("pre-filter capture did not cover the corpus")
                hit_rows = [
                    _hit_row(engine, hit, rank)
                    for rank, hit in enumerate(ranked, start=1)
                ]
                case = {
                    "case_id": _string(query, "case_id"),
                    "cohort": _string(query, "cohort"),
                    "ranked_hits": hit_rows,
                    "returned_source_paths": [_source_path(hit) for hit in trace.hits],
                }
                cases.append(case)
                if config.precision_control is not None:
                    explanations.append(
                        _explanation_row(
                            config.precision_control,
                            case["case_id"],
                            hit_rows,
                            trace.diagnostics,
                        )
                    )
            edges_after = _edge_state(engine)
            feedback_after = engine.store.count_feedback()
            activation = _activation_state(engine)
        return {
            "database_id": database_id,
            "cases": cases,
            "explanations": explanations,
            "ranking_sha256": canonical_sha256(
                [
                    {
                        "case_id": case["case_id"],
                        "pre_filter": [
                            hit["source_path"] for hit in case["ranked_hits"]
                        ],
                        "returned": case["returned_source_paths"],
                    }
                    for case in cases
                ]
            ),
            "score_sha256": canonical_sha256(
                [
                    {
                        "case_id": case["case_id"],
                        "scores": [
                            {
                                "source_path": hit["source_path"],
                                "final_score": hit["final_score"],
                                "entry_score": hit["entry_score"],
                                "normalized_graph_score": hit["normalized_graph_score"],
                            }
                            for hit in case["ranked_hits"]
                        ],
                    }
                    for case in cases
                ]
            ),
            "activation_sha256": canonical_sha256(activation),
            "edge_sha256_before": canonical_sha256(edges_before),
            "edge_sha256_after": canonical_sha256(edges_after),
            "feedback_count_before": feedback_before,
            "feedback_count_after": feedback_after,
            "root": root,
        }


def _index_corpus(
    engine: NeuronGraphRAG,
    protocol: Mapping[str, Any],
    documents: list[dict[str, Any]],
) -> None:
    for document in documents:
        engine.add_document(
            str(document["node_id"]),
            str(document["text"]),
            metadata=dict(_mapping(document, "metadata")),
        )
    repository = _string(_mapping(protocol, "corpus"), "repository")
    for relation in _rows(_mapping(protocol, "corpus"), "relationships"):
        source = f"github:{repository}:doc:{_string(relation, 'source_path')}"
        target = f"github:{repository}:doc:{_string(relation, 'target_path')}"
        engine.add_edge(source, target, _string(relation, "edge_type"))


@contextmanager
def _capture_constructed_hits() -> Iterator[list[SearchHit]]:
    original = engine_module.SearchHit
    captured: list[SearchHit] = []

    def recording_search_hit(*args: Any, **kwargs: Any) -> SearchHit:
        hit = original(*args, **kwargs)
        captured.append(hit)
        return hit

    engine_module.SearchHit = recording_search_hit  # type: ignore[misc]
    try:
        yield captured
    finally:
        engine_module.SearchHit = original  # type: ignore[misc]


def _hit_row(engine: NeuronGraphRAG, hit: SearchHit, rank: int) -> dict[str, Any]:
    metadata = hit.node.metadata
    relation_paths: list[dict[str, Any]] = []
    for path in hit.paths:
        if len(path.steps) != 1:
            continue
        step = path.steps[0]
        relation_paths.append(
            {
                "seed_path": _source_path(engine.store.get_node(path.seed_id)),
                "target_path": _source_path(hit),
                "edge_type": step.edge_type,
                "step_count": 1,
            }
        )
    return {
        "source_path": _source_path(hit),
        "rank": rank,
        "final_score": hit.final_score,
        "entry_score": hit.entry_score,
        "normalized_graph_score": hit.normalized_graph_activation,
        "source_provenance": {
            "repository": metadata["repository"],
            "commit": metadata["commit"],
            "path": metadata["path"],
            "content_sha256": metadata["content_sha256"],
        },
        "relation_paths": relation_paths,
    }


def _explanation_row(
    control: PrecisionControl,
    case_id: str,
    hit_rows: list[dict[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    precision = _mapping(diagnostics, "precision_control")
    raw_decisions = precision.get("decisions")
    if not isinstance(raw_decisions, list) or len(raw_decisions) != len(hit_rows):
        raise ValueError("actual precision decisions do not cover the corpus")
    decisions: list[dict[str, Any]] = []
    for hit, raw in zip(hit_rows, raw_decisions, strict=True):
        if not isinstance(raw, dict):
            raise TypeError("actual precision decision must be an object")
        decision = dict(raw)
        if decision.get("candidate_id") != control.candidate_id:
            raise ValueError("actual precision decision candidate mismatch")
        decision["source_path"] = hit["source_path"]
        decisions.append(decision)
    return {"case_id": case_id, "decisions": decisions}


def _combine_state(
    primary: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "fresh_database_id": primary["database_id"],
        "replay_database_id": replay["database_id"],
        "ranking_sha256": primary["ranking_sha256"],
        "replay_ranking_sha256": replay["ranking_sha256"],
        "score_sha256": primary["score_sha256"],
        "replay_score_sha256": replay["score_sha256"],
        "activation_sha256": primary["activation_sha256"],
        "replay_activation_sha256": replay["activation_sha256"],
        "edge_sha256_before": primary["edge_sha256_before"],
        "edge_sha256_after": primary["edge_sha256_after"],
        "feedback_count_before": primary["feedback_count_before"],
        "feedback_count_after": primary["feedback_count_after"],
    }


def _edge_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    return [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "factuality": edge.factuality,
            "reinforced_count": edge.reinforced_count,
        }
        for edge in engine.store.list_edges()
    ]


def _activation_state(engine: NeuronGraphRAG) -> list[dict[str, Any]]:
    rows = []
    for node in engine.store.list_nodes():
        activation = engine.store.activation(node.node_id)
        rows.append(
            {
                "node_id": node.node_id,
                "activation": None
                if activation is None
                else {"value": activation[0], "updated_at": activation[1]},
            }
        )
    return rows


def _source_path(value: Any) -> str:
    node = value.node if hasattr(value, "node") else value
    path = node.metadata.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("source path metadata is required")
    return path


def _reject_failed_retry(root: Path, stage: str) -> None:
    if _error_path(root, stage).exists() or _failed_claim_path(root, stage).exists():
        raise FileExistsError(f"{stage} has already failed and cannot be retried")


def _reject_observed_retry(root: Path, stage: str) -> None:
    if any(path.exists() for path in _final_archive_paths(root, stage).values()):
        raise FileExistsError(
            f"{stage} has already been observed and cannot be retried"
        )


def _final_archive_paths(root: Path, stage: str) -> dict[str, Path]:
    base = root / OBSERVATION_ARCHIVE
    return {
        "claim": base / f"{stage}.claim.json",
        "result": base / f"{stage}.observed.json",
        "lifecycle_transport": base / f"{stage}.lifecycle-transport.json",
        "archive_transport": base / f"{stage}.archive-transport.json",
    }


def _archive_error(
    root: Path,
    stage: str,
    claim_path: Path,
    claim_raw: bytes,
    error: BaseException,
) -> None:
    error_path = _error_path(root, stage)
    failed_claim = _failed_claim_path(root, stage)
    if failed_claim.exists() or error_path.exists():
        return
    failed_claim.parent.mkdir(parents=True, exist_ok=True)
    if claim_path.exists():
        os.replace(claim_path, failed_claim)
    write_json_exclusive(
        error_path,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "stage": stage,
            "claim_sha256": sha256_bytes(claim_raw),
            "exception_type": type(error).__name__,
            "message": str(error),
        },
    )


def _error_path(root: Path, stage: str) -> Path:
    return root / OBSERVATION_ARCHIVE / f"{stage}.error.json"


def _failed_claim_path(root: Path, stage: str) -> Path:
    return root / OBSERVATION_ARCHIVE / f"{stage}.failed.claim.json"


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TypeError(f"{key} must be an object")
    return item


def _rows(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, dict) for row in item):
        raise ValueError(f"{key} must be an array of objects")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"{key} must be an integer")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(item)


def _path(value: Mapping[str, Any], key: str) -> Path:
    item = value.get(key)
    if not isinstance(item, Path):
        raise TypeError(f"{key} must be a path")
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen precision-control observation exactly once"
    )
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("stage", nargs="?", choices=("development", "holdout"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        if args.stage is not None:
            parser.error("preflight does not accept a stage")
        result = preflight()
    else:
        if args.stage is None:
            result = run_conditional()
        else:
            result = run_stage(args.stage)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

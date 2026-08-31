from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_observation as worker_base
from . import cross_encoder_precision_v19_performance_observation as predecessor
from . import (
    intent_aware_rank_fusion,
    rank_observation_stage_contract,
    source_root_propagation,
)

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v21"
FREEZE_COMMIT = "33b465c7422e8eeae1153e323a46a662a97f8fee"
V8_PROTOCOL_COMMIT = predecessor.V8_PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MODULE = "neuron_graph_rag.cross_encoder_precision_v21_intent_aware_observation"
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v21_observation.manifest.json"
)
SOURCE_IDENTITY = Path(
    "tests/fixtures/github_cross_encoder_precision_v21.source-identity.json"
)
OBSERVATION_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v21.observation-audit.json"
)
CORPUS = Path("tests/fixtures/github_cross_encoder_precision_v21.corpus.json")
QUERIES = Path("tests/fixtures/github_cross_encoder_precision_v21.queries.json")
GOLD = Path("tests/fixtures/github_cross_encoder_precision_v21.gold.json")
V20_IDENTITIES = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.future-identities.json"
)
V20_GATES = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.gate-ownership.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v21_observation")

lifecycle = predecessor.lifecycle
source_root_freeze = predecessor.source_root_freeze
IMAGE = predecessor.IMAGE
IMAGE_ID = predecessor.IMAGE_ID
WSLC_VERSION = predecessor.WSLC_VERSION
VOLUME = "github-cross-encoder-precision-v21-runtime"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v21/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"
CONTAINER_MODEL_REGISTRY = (
    CONTAINER_SOURCE / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
)

FORBIDDEN_VOLUMES = {
    **predecessor.FORBIDDEN_VOLUMES,
    "v19_runtime_volume": predecessor.VOLUME,
}

PREDECESSOR_ANCHOR_SHA256 = {
    "src/neuron_graph_rag/cross_encoder_precision_v19_performance_observation.py": (
        "e9766f1c9b16a7ab9bf0e8cfa7eca7c0a4a1f1bea8fb29750687e4cacb7af231"
    ),
    "src/neuron_graph_rag/intent_aware_rank_fusion.py": (
        "c73a9e18c1e71d2fc08b2c9ea098bf9cfe388ed143da1a0062dd79a5d211b720"
    ),
    "tests/fixtures/github_cross_encoder_precision_v20.gate-ownership.json": (
        "ae57aee84977ceb033f450ee56e15e43ce74786443fd8621f36b33ee0b092af7"
    ),
    "tests/evidence/github_cross_encoder_precision_v19_observation/"
    "terminal-evidence-manifest.json": (
        "953c856ec0160e56843e6bf5ab461a280fe7305e8b4501253c7da922c0c98907"
    ),
}

PROTOCOL_GATE_IDS = (
    "protocol-source-contract-integrity",
    "identity-separation",
    "baseline-prefilter-validity",
    "relation-source-edge-only-provenance",
    "production-signal-only",
    "default-surface-immutability",
)
CANDIDATE_GATE_IDS = (
    "positive-case-rank-non-regression",
    "positive-cohort-mrr-hit-at-5-non-regression",
    "negative-non-worsening-and-aggregate-strict-improvement",
    "positive-expected-source-top-5-completeness",
    "intent-aware-fusion-rank-only-recomputation",
    "relation-path-preservation",
)
WORKERS = predecessor.lifecycle.WORKERS


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"v21 JSON must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return lifecycle.sha256_file(path)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            str(Path("src/neuron_graph_rag") / Path(MODULE.split(".")[-1] + ".py")),
            "tests/test_cross_encoder_precision_v21_intent_aware_observation.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v19_performance_observation",
            "tests.test_cross_encoder_precision_v20_intent_aware_freeze",
            "tests.test_cross_encoder_precision_v21_intent_aware_observation",
        ],
        [str(python), "-m", predecessor.MODULE, "audit"],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v20_intent_aware_freeze",
            "audit",
        ],
        [str(python), "-m", MODULE, "audit"],
    )


def _stage_rows(value: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    stages = value.get("stages")
    if not isinstance(stages, Mapping) or stage not in stages:
        raise ValueError(f"v21 stage is missing: {stage}")
    selected = stages[stage]
    if isinstance(selected, Mapping):
        selected = selected.get("cases")
    if not isinstance(selected, list) or not all(
        isinstance(row, dict) for row in selected
    ):
        raise TypeError(f"v21 stage rows must be objects: {stage}")
    return [dict(row) for row in selected]


def _validate_protocol_fixtures(
    root: Path, *, verify_v19_disjoint: bool = True
) -> dict[str, Any]:
    corpus = _read_object(root / CORPUS)
    queries = _read_object(root / QUERIES)
    gold = _read_object(root / GOLD)
    identities = _read_object(root / V20_IDENTITIES)
    gates = _read_object(root / V20_GATES)
    values = (corpus, queries, gold)
    if any(value.get("protocol_id") != PROTOCOL_ID for value in values):
        raise ValueError("v21 fixture protocol identity mismatch")
    documents = corpus.get("documents")
    if not isinstance(documents, list) or len(documents) != 24:
        raise ValueError("v21 corpus must contain exactly 24 fresh documents")
    paths: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            raise TypeError("v21 corpus documents must be objects")
        path = document.get("path")
        text = document.get("text")
        if (
            not isinstance(path, str)
            or not path.startswith("fresh/v21/")
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("v21 corpus document identity mismatch")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("v21 corpus paths must be unique")
    relationships = corpus.get("relationships")
    if not isinstance(relationships, list) or len(relationships) != 4:
        raise ValueError("v21 relation cardinality mismatch")
    for relation in relationships:
        if (
            not isinstance(relation, dict)
            or set(relation) != {"source_path", "target_path", "edge_type"}
            or relation.get("source_path") not in paths
            or relation.get("target_path") not in paths
            or relation.get("edge_type") != "informs"
        ):
            raise ValueError("v21 relation fixture mismatch")
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    expected_identities = {
        "development": identities.get("development"),
        "holdout": identities.get("holdout"),
    }
    if len(set(expected_identities.values())) != 2:
        raise ValueError("v21 stage identities must be fresh and separated")
    query_stages = queries.get("stages")
    if not isinstance(query_stages, Mapping):
        raise TypeError("v21 query stages must be an object")
    for stage in ("development", "holdout"):
        case_prefix = "v21-dev-" if stage == "development" else "v21-holdout-"
        stage_value = query_stages.get(stage)
        if (
            not isinstance(stage_value, Mapping)
            or stage_value.get("identity") != expected_identities[stage]
        ):
            raise ValueError(f"v21 query identity mismatch: {stage}")
        rows = _stage_rows(queries, stage)
        if len(rows) != 8:
            raise ValueError(f"v21 query count mismatch: {stage}")
        cohorts = [row.get("cohort") for row in rows]
        if sorted(cohorts) != sorted(
            ["direct_lexical", "semantic_paraphrase", "relation_linked", "negative_control"]
            * 2
        ):
            raise ValueError(f"v21 cohort cardinality mismatch: {stage}")
        for row in rows:
            case_id = row.get("case_id")
            query = row.get("query")
            if (
                not isinstance(case_id, str)
                or not case_id.startswith(case_prefix)
                or not isinstance(query, str)
                or not query.strip()
            ):
                raise ValueError("v21 query row mismatch")
            query_ids.add(case_id)
            query_texts.add(query)
    if len(query_ids) != 16 or len(query_texts) != 16:
        raise ValueError("v21 query identities and text must be unique")
    for stage in ("development", "holdout"):
        query_rows = {row["case_id"]: row for row in _stage_rows(queries, stage)}
        gold_rows = _stage_rows(gold, stage)
        if {row.get("case_id") for row in gold_rows} != set(query_rows):
            raise ValueError(f"v21 query/gold identity mismatch: {stage}")
        for row in gold_rows:
            expected = row.get("expected_path")
            forbidden = row.get("forbidden_path")
            if (expected is None) == (forbidden is None):
                raise ValueError("v21 gold must choose expected xor forbidden")
            selected = expected if expected is not None else forbidden
            if selected not in paths:
                raise ValueError("v21 gold path is outside the fresh corpus")
            if row.get("cohort") != query_rows[row["case_id"]].get("cohort"):
                raise ValueError("v21 query/gold cohort mismatch")
    if verify_v19_disjoint:
        v19_queries = _read_object(
            root / "tests/fixtures/github_cross_encoder_precision_v8.queries.json"
        )
        v19_corpus = _read_object(
            root / "tests/fixtures/github_cross_encoder_precision_v8.corpus.json"
        )
        old_queries = {
            row["query"]
            for stage in ("development", "holdout")
            for row in _stage_rows(v19_queries, stage)
        }
        old_paths = {
            row["path"]
            for row in v19_corpus.get("documents", [])
            if isinstance(row, dict)
        }
        if query_texts & old_queries or set(paths) & old_paths:
            raise ValueError("v21 must not reuse v19 query text or source identity")
    if gates.get("protocol_validity_gates") != list(PROTOCOL_GATE_IDS):
        raise ValueError("v21 protocol gate ownership diverged from v20")
    if gates.get("candidate_controllable_gates") != list(CANDIDATE_GATE_IDS):
        raise ValueError("v21 candidate gate ownership diverged from v20")
    return {
        "corpus_document_count": len(paths),
        "query_count": len(query_ids),
        "development_identity": expected_identities["development"],
        "holdout_identity": expected_identities["holdout"],
        "v19_query_text_reuse_count": 0,
        "v19_source_path_reuse_count": 0,
    }


SOURCE_ROOT_SPEC = source_root_propagation.SourceRootFreezeSpec(
    protocol_id=PROTOCOL_ID,
    phase="performance-observation",
    predecessor_merge_commit=FREEZE_COMMIT,
    frozen_protocol_commit=V8_PROTOCOL_COMMIT,
    root=ROOT,
    manifest_path=MANIFEST,
    source_identity_path=SOURCE_IDENTITY,
    audit_path=OBSERVATION_AUDIT,
    evidence_path=EVIDENCE,
    image=IMAGE,
    image_id=IMAGE_ID,
    wslc_version=WSLC_VERSION,
    freeze_volume=VOLUME,
    future_runtime_volume=VOLUME,
    container_root=CONTAINER_ROOT,
    container_source=CONTAINER_SOURCE,
    container_cache=CONTAINER_CACHE,
    container_frozen_source=CONTAINER_PROTOCOL_SOURCE,
    container_report=CONTAINER_ROOT / "source-root-propagation-verification.json",
    container_source_identity=CONTAINER_SOURCE / SOURCE_IDENTITY.as_posix(),
    old_frozen_source=source_root_freeze.OLD_FROZEN_SOURCE,
    predecessor_artifact_count=39,
    identity_schema="ngr.source-root-propagation/v1",
    evidence_stem="observation",
    report_name="source-root-propagation-verification.json",
    forbidden_volumes=FORBIDDEN_VOLUMES,
    read_json=lifecycle.read_json,
    sha256_file=lifecycle.sha256_file,
    canonical_sha256=lifecycle.canonical_sha256,
    write_json_exclusive=lifecycle._write_json_exclusive,
)

STAGE_CONTRACT = rank_observation_stage_contract.RankObservationStageContract(
    container_databases=CONTAINER_DATABASES,
    container_runs=CONTAINER_RUNS,
    worker_slots_per_stage=len(WORKERS),
)


def _documents(root: Path) -> list[dict[str, str]]:
    corpus = _read_object(root / CORPUS)
    return [
        {
            "path": str(row["path"]),
            "text": str(row["text"]),
            "sha256": worker_base.sha256_bytes(str(row["text"]).encode("utf-8")),
        }
        for row in corpus["documents"]
    ]


def _index_fresh_corpus(
    engine: worker_base.NeuronGraphRAG,
    root: Path,
    documents: Sequence[Mapping[str, str]],
) -> None:
    corpus = _read_object(root / CORPUS)
    repository = str(corpus["repository"])
    for row in documents:
        engine.add_document(
            f"github:{repository}:doc:{row['path']}",
            row["text"],
            metadata={
                "repository": repository,
                "commit": FREEZE_COMMIT,
                "path": row["path"],
                "content_sha256": row["sha256"],
            },
        )
    for relation in corpus["relationships"]:
        engine.add_edge(
            f"github:{repository}:doc:{relation['source_path']}",
            f"github:{repository}:doc:{relation['target_path']}",
            str(relation["edge_type"]),
        )


def _model_spec(root: Path, kind: str) -> dict[str, Any]:
    registry = _read_object(
        root / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
    )
    models = registry.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("v21 frozen model registry mismatch")
    index = {"base": 0, "v2-m3": 1}.get(kind)
    if index is None or not isinstance(models[index], dict):
        raise ValueError(f"v21 worker kind mismatch: {kind}")
    return dict(models[index])


def _state_payload(
    engine: worker_base.NeuronGraphRAG,
    database: Path,
    cases: list[dict[str, Any]],
    edge_before: str,
    feedback_before: int,
    sqlite_before: str,
) -> dict[str, Any]:
    edge_after = worker_base.canonical_sha256(worker_base._edge_state(engine))
    feedback_after = engine.store.count_feedback()
    sqlite_after = worker_base.canonical_sha256(
        worker_base._static_sqlite_state(engine)
    )
    activation = worker_base.canonical_sha256(worker_base._activation_state(engine))
    return {
        "database_id": worker_base.sha256_bytes(
            str(database.resolve()).encode("utf-8")
        ),
        "cases": cases,
        "ranking_sha256": worker_base.canonical_sha256(cases),
        "activation_sha256": activation,
        "edge_sha256_before": edge_before,
        "edge_sha256_after": edge_after,
        "feedback_count_before": feedback_before,
        "feedback_count_after": feedback_after,
        "sqlite_sha256_before": sqlite_before,
        "sqlite_sha256_after": sqlite_after,
    }


def _container_worker_v21(
    stage: str, kind: str, replay: str, database: Path, output: Path
) -> dict[str, Any]:
    if stage not in {"development", "holdout"} or (kind, replay) not in WORKERS:
        raise ValueError("v21 worker identity is not frozen")
    if database.exists() or output.exists():
        raise FileExistsError("v21 worker DB/output must be fresh")
    for path in (database.parent, output.parent, Path(str(CONTAINER_CACHE))):
        path.resolve().relative_to(Path(str(CONTAINER_ROOT)))
    root = Path(str(CONTAINER_SOURCE))
    _validate_protocol_fixtures(root, verify_v19_disjoint=False)
    documents = _documents(root)
    queries = _stage_rows(_read_object(root / QUERIES), stage)
    started = time.perf_counter()
    model_spec = None if kind == "baseline" else _model_spec(root, kind)
    model_runtime = (
        None
        if model_spec is None
        else worker_base._load_model(model_spec, Path(str(CONTAINER_CACHE)))
    )
    pair_count = 0
    with worker_base.NeuronGraphRAG(
        database, config=worker_base.EngineConfig()
    ) as engine:
        _index_fresh_corpus(engine, root, documents)
        edge_before = worker_base.canonical_sha256(worker_base._edge_state(engine))
        feedback_before = engine.store.count_feedback()
        sqlite_before = worker_base.canonical_sha256(
            worker_base._static_sqlite_state(engine)
        )
        cases: list[dict[str, Any]] = []
        for query in queries:
            query_text = str(query["query"])
            intent = intent_aware_rank_fusion.decompose_query_intent(query_text)
            trace = engine.search(
                intent.positive_query,
                limit=len(documents),
                now=worker_base.OBSERVATION_NOW,
            )
            prefilter = [
                worker_base._baseline_hit_row(hit, rank)
                for rank, hit in enumerate(trace.hits, start=1)
            ]
            if len(prefilter) != len(documents):
                raise ValueError("v21 prefilter did not rank the complete fresh corpus")
            if model_runtime is None:
                cases.append(
                    {
                        "case_id": query["case_id"],
                        "cohort": query["cohort"],
                        "ranked_hits": prefilter,
                    }
                )
                continue
            positive_rows, pairs = worker_base._score_case(
                intent.positive_query, prefilter, documents, model_runtime
            )
            pair_count += pairs
            exclusion_rows = []
            for exclusion in intent.exclusion_queries:
                scored, pairs = worker_base._score_case(
                    exclusion, prefilter, documents, model_runtime
                )
                exclusion_rows.append({row["source_path"]: row for row in scored})
                pair_count += pairs
            positive = {row["source_path"]: row for row in positive_rows}
            signals = [
                {
                    "source_path": row["source_path"],
                    "prefilter_rank": row["rank"],
                    "prefilter_score": row["ngr_score"],
                    "positive_logit": positive[row["source_path"]]["raw_logit"],
                    "exclusion_logits": [
                        scored[row["source_path"]]["raw_logit"]
                        for scored in exclusion_rows
                    ],
                    "relation_paths": row["relation_paths"],
                }
                for row in prefilter
            ]
            ranked = intent_aware_rank_fusion.fuse_intent_aware_ranks(
                query_text, signals
            )
            cases.append(
                {
                    "case_id": query["case_id"],
                    "cohort": query["cohort"],
                    "production_signals": signals,
                    "ranked_hits": ranked,
                }
            )
        payload = _state_payload(
            engine,
            database,
            cases,
            edge_before,
            feedback_before,
            sqlite_before,
        )
    import psutil

    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "stage": stage,
            "kind": kind,
            "replay": replay,
            "container_id": os.environ.get(
                "NGR_V21_CONTAINER_IDENTITY", socket.gethostname()
            ),
            "container_process_pid": os.getpid(),
            "model_id": None if model_spec is None else model_spec["model_id"],
            "revision": None if model_spec is None else model_spec["revision"],
            "metrics": {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "peak_rss_bytes": psutil.Process().memory_info().rss,
                "cache_bytes": worker_base._tree_bytes(Path(str(CONTAINER_CACHE))),
                "pair_count": pair_count,
            },
        }
    )
    _write_json_exclusive(output, payload)
    return payload


def _combine_state(primary: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    return worker_base._combine_state(primary, replay)


def _returned_paths(case: Mapping[str, Any]) -> list[str]:
    hits = case.get("ranked_hits")
    if not isinstance(hits, list):
        raise TypeError("v21 ranked hits must be a list")
    return [str(row["source_path"]) for row in hits[:5] if isinstance(row, Mapping)]


def _quality(
    cases: Sequence[Mapping[str, Any]], gold_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_case = {str(row["case_id"]): row for row in cases}
    cohort_values: dict[str, list[float]] = {
        "direct_lexical": [],
        "semantic_paraphrase": [],
        "relation_linked": [],
    }
    forbidden_count = 0
    case_rows = []
    for gold in gold_rows:
        case_id = str(gold["case_id"])
        paths = _returned_paths(by_case[case_id])
        expected = gold.get("expected_path")
        forbidden = gold.get("forbidden_path")
        rank = paths.index(expected) + 1 if expected in paths else None
        forbidden_present = forbidden in paths if forbidden is not None else False
        if expected is not None:
            cohort_values[str(gold["cohort"])].append(
                0.0 if rank is None else 1.0 / rank
            )
        forbidden_count += int(forbidden_present)
        case_rows.append(
            {
                "case_id": case_id,
                "cohort": gold["cohort"],
                "expected_rank": rank,
                "forbidden_top_5": forbidden_present,
            }
        )
    cohorts = {
        name: {
            "mrr": sum(values) / len(values),
            "hit_at_5": sum(value > 0.0 for value in values) / len(values),
        }
        for name, values in cohort_values.items()
    }
    return {
        "cohorts": cohorts,
        "negative_forbidden_top_5_count": forbidden_count,
        "cases": case_rows,
    }


def _state_is_immutable(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("ranking_sha256") == state.get("replay_ranking_sha256")
        and state.get("activation_sha256") == state.get("replay_activation_sha256")
        and state.get("edge_sha256_before") == state.get("edge_sha256_after")
        and state.get("feedback_count_before")
        == state.get("feedback_count_after")
        == 0
        and state.get("sqlite_sha256_before") == state.get("sqlite_sha256_after")
        and state.get("fresh_database_id") != state.get("replay_database_id")
    )


def _gate_rows(ids: Sequence[str], values: Sequence[bool]) -> list[dict[str, Any]]:
    return [
        {"gate_id": gate_id, "hard": True, "passed": bool(value)}
        for gate_id, value in zip(ids, values, strict=True)
    ]


def _relation_path(gold: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed_path": gold["relation_seed_path"],
        "target_path": gold["expected_path"],
        "edge_type": gold["relation_edge_type"],
        "step_count": 1,
    }


def _protocol_gates(
    root: Path,
    baseline_cases: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixture_ok = bool(
        _validate_protocol_fixtures(root, verify_v19_disjoint=False)
    )
    identities = _read_object(root / V20_IDENTITIES)
    identity_ok = identities["development"] != identities["holdout"]
    documents = _documents(root)
    baseline_ok = all(
        len(case.get("ranked_hits", [])) == len(documents)
        and len(set(_returned_paths(case))) == 5
        for case in baseline_cases
    )
    baseline_by_case = {str(row["case_id"]): row for row in baseline_cases}
    relation_ok = True
    for gold in gold_rows:
        if gold.get("cohort") != "relation_linked":
            continue
        hits = baseline_by_case[str(gold["case_id"])]["ranked_hits"]
        target = next(
            (row for row in hits if row["source_path"] == gold["expected_path"]),
            None,
        )
        relation_ok &= bool(target and _relation_path(gold) in target["relation_paths"])
    production_ok = True
    for candidate in candidate_rows:
        for case in candidate["cases"]:
            for signal in case["production_signals"]:
                production_ok &= set(signal) == intent_aware_rank_fusion.PRODUCTION_SIGNAL_KEYS
                lowered = " ".join(signal).lower()
                production_ok &= not any(
                    token in lowered
                    for token in ("gold", "expected", "forbidden", "label", "relevance")
                )
    immutable = all(_state_is_immutable(state) for state in states)
    return _gate_rows(
        PROTOCOL_GATE_IDS,
        (fixture_ok, identity_ok, baseline_ok, relation_ok, production_ok, immutable),
    )


def _candidate_gates(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gold_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_cases = {str(row["case_id"]): row for row in baseline["cases"]}
    candidate_cases = {str(row["case_id"]): row for row in candidate["cases"]}
    positive_nonregression = True
    completeness = True
    negative_nonworse = True
    relation_preserved = True
    recomputed = True
    baseline_forbidden = 0
    candidate_forbidden = 0
    for gold in gold_rows:
        case_id = str(gold["case_id"])
        baseline_paths = _returned_paths(baseline_cases[case_id])
        candidate_paths = _returned_paths(candidate_cases[case_id])
        expected = gold.get("expected_path")
        forbidden = gold.get("forbidden_path")
        if expected is not None:
            baseline_rank = (
                baseline_paths.index(expected) + 1 if expected in baseline_paths else None
            )
            candidate_rank = (
                candidate_paths.index(expected) + 1 if expected in candidate_paths else None
            )
            if baseline_rank is not None and (
                candidate_rank is None or candidate_rank > baseline_rank
            ):
                positive_nonregression = False
            completeness &= candidate_rank is not None
        else:
            baseline_present = forbidden in baseline_paths
            candidate_present = forbidden in candidate_paths
            baseline_forbidden += int(baseline_present)
            candidate_forbidden += int(candidate_present)
            negative_nonworse &= not (candidate_present and not baseline_present)
        case = candidate_cases[case_id]
        query = str(case["query"])
        expected_ranking = intent_aware_rank_fusion.fuse_intent_aware_ranks(
            query, case["production_signals"]
        )
        recomputed &= expected_ranking == case["ranked_hits"]
        signals = {
            row["source_path"]: row for row in case["production_signals"]
        }
        relation_preserved &= all(
            row["relation_paths"] == signals[row["source_path"]]["relation_paths"]
            for row in case["ranked_hits"]
        )
        if gold.get("cohort") == "relation_linked" and expected in candidate_paths:
            hit = next(
                row for row in case["ranked_hits"] if row["source_path"] == expected
            )
            relation_preserved &= _relation_path(gold) in hit["relation_paths"]
    baseline_quality = baseline["quality"]
    candidate_quality = candidate["quality"]
    cohort_nonregression = all(
        candidate_quality["cohorts"][cohort][metric]
        >= baseline_quality["cohorts"][cohort][metric]
        for cohort in ("direct_lexical", "semantic_paraphrase", "relation_linked")
        for metric in ("mrr", "hit_at_5")
    )
    negative_gate = (
        negative_nonworse
        and candidate_forbidden <= baseline_forbidden
        and candidate_forbidden < baseline_forbidden
    )
    return _gate_rows(
        CANDIDATE_GATE_IDS,
        (
            positive_nonregression,
            cohort_nonregression,
            negative_gate,
            completeness,
            recomputed,
            relation_preserved,
        ),
    )


def _container_claim_v21(stage: str) -> dict[str, Any]:
    root = Path(str(CONTAINER_SOURCE))
    contract = _validate_protocol_fixtures(root, verify_v19_disjoint=False)
    query = _read_object(root / QUERIES)
    stage_value = query["stages"][stage]
    claim = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "stage_identity": stage_value["identity"],
        "query_fixture_sha256": _sha256_file(root / QUERIES),
        "corpus_fixture_sha256": _sha256_file(root / CORPUS),
        "query_count": len(stage_value["cases"]),
        "v19_query_text_reuse_count": contract["v19_query_text_reuse_count"],
        "v19_source_path_reuse_count": contract["v19_source_path_reuse_count"],
        "retry_count": 0,
    }
    if stage == "holdout":
        development = _read_object(
            Path(str(CONTAINER_ARCHIVE)) / "development.observed.json"
        )
        selected = development.get("selected_candidate_id")
        if not isinstance(selected, str):
            raise ValueError("v21 holdout requires a selected development candidate")
        claim["selected_candidate_id"] = selected
    path = Path(str(CONTAINER_RUNS / stage / "claim.json"))
    _write_json_exclusive(path, claim)
    return claim


def _copy_stage_artifacts(
    stage: str,
    claim_path: Path,
    result_path: Path,
    worker_paths: Sequence[Path],
) -> dict[str, Any]:
    evidence = Path(str(CONTAINER_SOURCE / EVIDENCE.as_posix()))
    raw_root = evidence / "raw" / stage
    raw_root.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}
    claim_target = evidence / f"{stage}.claim.json"
    result_target = evidence / f"{stage}.observed.json"
    for source, target in ((claim_path, claim_target), (result_path, result_target)):
        raw = source.read_bytes()
        _write_bytes_exclusive(target, raw)
        files[str(target.relative_to(evidence)).replace("\\", "/")] = worker_base.sha256_bytes(raw)
    for source in worker_paths:
        target = raw_root / source.name
        raw = source.read_bytes()
        _write_bytes_exclusive(target, raw)
        files[str(target.relative_to(evidence)).replace("\\", "/")] = worker_base.sha256_bytes(raw)
    transport = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "status": "complete",
        "files": dict(sorted(files.items())),
        "byte_identity_verified": True,
        "retry_count": 0,
    }
    transport_path = evidence / f"{stage}.transport.json"
    _write_json_exclusive(transport_path, transport)
    return transport


def _container_finalize_v21(stage: str) -> dict[str, Any]:
    root = Path(str(CONTAINER_SOURCE))
    query_rows = _stage_rows(_read_object(root / QUERIES), stage)
    gold_rows = _stage_rows(_read_object(root / GOLD), stage)
    run_root = Path(str(CONTAINER_RUNS / stage))
    claim_path = run_root / "claim.json"
    claim = _read_object(claim_path)
    raw = {
        (kind, replay): _read_object(run_root / f"{kind}-{replay}.json")
        for kind, replay in WORKERS
    }
    baseline_primary = raw[("baseline", "primary")]
    baseline_replay = raw[("baseline", "replay")]
    if baseline_primary["cases"] != baseline_replay["cases"]:
        raise ValueError("v21 baseline replay cases differ")
    baseline = {
        "candidate_id": "current-ngr-prefilter",
        "cases": baseline_primary["cases"],
        "quality": _quality(baseline_primary["cases"], gold_rows),
        "state": _combine_state(baseline_primary, baseline_replay),
        "metrics": {
            "primary": baseline_primary["metrics"],
            "replay": baseline_replay["metrics"],
        },
    }
    candidates = []
    states = [baseline["state"]]
    query_by_id = {str(row["case_id"]): str(row["query"]) for row in query_rows}
    for kind in ("base", "v2-m3"):
        primary = raw[(kind, "primary")]
        replay = raw[(kind, "replay")]
        if primary["cases"] != replay["cases"]:
            raise ValueError(f"v21 {kind} replay cases differ")
        cases = []
        for case in primary["cases"]:
            cases.append({**case, "query": query_by_id[str(case["case_id"])]})
        state = _combine_state(primary, replay)
        states.append(state)
        candidates.append(
            {
                "candidate_id": f"{kind}-intent-aware",
                "model_id": primary["model_id"],
                "revision": primary["revision"],
                "cases": cases,
                "quality": _quality(cases, gold_rows),
                "state": state,
                "metrics": {
                    "primary": primary["metrics"],
                    "replay": replay["metrics"],
                },
            }
        )
    protocol_gates = _protocol_gates(
        root, baseline["cases"], candidates, states, gold_rows
    )
    protocol_pass = all(row["passed"] for row in protocol_gates)
    selected: str | None = None
    for candidate in candidates:
        gates = (
            _candidate_gates(candidate, baseline, gold_rows)
            if protocol_pass
            else []
        )
        candidate["gates"] = gates
        candidate["failed_hard_gate_ids"] = [
            row["gate_id"] for row in gates if not row["passed"]
        ]
        candidate["all_candidate_gates_pass"] = bool(gates) and all(
            row["passed"] for row in gates
        )
        eligible = stage == "development" or candidate["candidate_id"] == claim.get(
            "selected_candidate_id"
        )
        if selected is None and eligible and candidate["all_candidate_gates_pass"]:
            selected = str(candidate["candidate_id"])
    if stage == "holdout" and selected != claim.get("selected_candidate_id"):
        selected = None
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "stage_identity": claim["stage_identity"],
        "protocol_validity_gates": protocol_gates,
        "protocol_validity_pass": protocol_pass,
        "candidate_gates_evaluated": protocol_pass,
        "baseline": baseline,
        "candidates": candidates,
        "selected_candidate_id": selected,
        "all_hard_gates_pass": selected is not None,
        "performance": "assessed",
        "retry_count": 0,
        "claim_sha256": _sha256_file(claim_path),
    }
    archive = Path(str(CONTAINER_ARCHIVE)) / f"{stage}.observed.json"
    _write_json_exclusive(archive, result)
    worker_paths = [run_root / f"{kind}-{replay}.json" for kind, replay in WORKERS]
    _copy_stage_artifacts(stage, claim_path, archive, worker_paths)
    return result


def _container_fail_stage_v21(stage: str, message: str) -> dict[str, Any]:
    run_root = Path(str(CONTAINER_RUNS / stage))
    claim_path = run_root / "claim.json"
    if not claim_path.is_file():
        raise FileNotFoundError("v21 failed stage claim is unavailable")
    error = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "error": message,
        "retry_count": 0,
        "same_protocol_retry_allowed": False,
        "performance": "not assessed",
        "claim_sha256": _sha256_file(claim_path),
    }
    archive = Path(str(CONTAINER_ARCHIVE)) / f"{stage}.error.json"
    _write_json_exclusive(archive, error)
    evidence = Path(str(CONTAINER_SOURCE / EVIDENCE.as_posix()))
    _write_bytes_exclusive(evidence / f"{stage}.claim.json", claim_path.read_bytes())
    _write_bytes_exclusive(evidence / f"{stage}.error.json", archive.read_bytes())
    worker_paths = [
        run_root / f"{kind}-{replay}.json"
        for kind, replay in WORKERS
        if (run_root / f"{kind}-{replay}.json").is_file()
    ]
    raw_root = evidence / "raw" / stage
    raw_root.mkdir(parents=True, exist_ok=False)
    files = {
        f"{stage}.claim.json": _sha256_file(evidence / f"{stage}.claim.json"),
        f"{stage}.error.json": _sha256_file(evidence / f"{stage}.error.json"),
    }
    for source in worker_paths:
        target = raw_root / source.name
        shutil.copyfile(source, target)
        files[f"raw/{stage}/{source.name}"] = _sha256_file(target)
    transport = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "status": "error",
        "files": dict(sorted(files.items())),
        "byte_identity_verified": True,
        "retry_count": 0,
    }
    _write_json_exclusive(evidence / f"{stage}.transport.json", transport)
    return error


class V21RankObservationSpec(predecessor.V19RankObservationSpec):
    def validate_prebuild(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        result = super().validate_prebuild(project_root)
        protocol = _validate_protocol_fixtures(project_root)
        manifest = self.manifest(project_root)
        registry = manifest.get("v21_protocol_artifact_sha256")
        if not isinstance(registry, dict) or len(registry) != 9:
            raise ValueError("v21 protocol artifact registry cardinality mismatch")
        for relative, expected in registry.items():
            if (
                not isinstance(relative, str)
                or not isinstance(expected, str)
                or _sha256_file(project_root / relative) != expected
            ):
                raise ValueError(f"v21 protocol artifact changed: {relative}")
        if (
            manifest.get("v19_cases_reused_as_performance_evidence") is not False
            or manifest.get("v19_result_packets_opened") is not False
            or manifest.get("v20_fusion_contract_changed") is not False
            or manifest.get("fresh_protocol_root") != str(CONTAINER_ROOT)
        ):
            raise ValueError("v21 predecessor and freshness boundary mismatch")
        return {
            **result,
            **protocol,
            "v21_protocol_artifact_count": len(registry),
        }

    def run_stage_host(
        self,
        stage: str,
        root: Path,
        rows: list[dict[str, object]],
        claim_counts: dict[str, int],
    ) -> dict[str, object]:
        engine = lifecycle.lifecycle.lifecycle
        initialized = json.loads(
            engine._run_logged(
                self.container_command("stage-init", "--stage", stage), root, rows
            )
        )
        STAGE_CONTRACT.validate_initialization(initialized, stage)
        engine._run_logged(self.container_command("claim", "--stage", stage), root, rows)
        claim_counts[stage] += 1
        for kind, replay in WORKERS:
            identity = f"ngr-v21-{stage}-{kind}-{replay}"
            command = self.container_command(
                "worker",
                "--stage",
                stage,
                "--kind",
                kind,
                "--replay",
                replay,
                "--database",
                str(self.container_databases / stage / f"{kind}-{replay}.sqlite3"),
                "--output",
                str(self.container_runs / stage / f"{kind}-{replay}.json"),
                name=identity,
            )
            insert_at = command.index("--workdir")
            command[insert_at:insert_at] = [
                "--env",
                f"NGR_V21_CONTAINER_IDENTITY={identity}",
            ]
            engine._run_logged(command, root, rows)
        result = json.loads(
            engine._run_logged(
                self.container_command("finalize", "--stage", stage), root, rows
            )
        )
        lifecycle.lifecycle._export_volume_evidence(root, rows)
        return result

    def dispatch_container_command(
        self, command: str, **arguments: str
    ) -> dict[str, Any]:
        if command == "stage-init":
            return STAGE_CONTRACT.initialize_container_stage(arguments["stage"])
        if command == "claim":
            return _container_claim_v21(arguments["stage"])
        if command == "worker":
            return _container_worker_v21(
                arguments["stage"],
                arguments["kind"],
                arguments["replay"],
                Path(arguments["database"]),
                Path(arguments["output"]),
            )
        if command == "finalize":
            return _container_finalize_v21(arguments["stage"])
        if command == "fail-stage":
            return _container_fail_stage_v21(
                arguments["stage"], arguments["message"]
            )
        return super().dispatch_container_command(command, **arguments)


SPEC = V21RankObservationSpec(
    protocol_id=PROTOCOL_ID,
    freeze_commit=FREEZE_COMMIT,
    root=ROOT,
    manifest_path=MANIFEST,
    source_identity_path=SOURCE_IDENTITY,
    audit_path=OBSERVATION_AUDIT,
    evidence_path=EVIDENCE,
    module_name=MODULE,
    runtime_volume=VOLUME,
    container_root=CONTAINER_ROOT,
    predecessor_artifact_count=39,
    predecessor_anchor_sha256=PREDECESSOR_ANCHOR_SHA256,
    forbidden_volumes=FORBIDDEN_VOLUMES,
    source_root_spec=SOURCE_ROOT_SPEC,
    verification_commands_factory=_verification_commands,
)
TERMINAL_AUDIT = (
    rank_observation_stage_contract.RankObservationActualCountTerminalAudit(
        SPEC, STAGE_CONTRACT
    )
)

serialize_container_path = lifecycle.serialize_container_path
validate_prebuild = SPEC.validate_prebuild
verify_preflight = SPEC.verify_preflight


def preflight(root: Path = ROOT, model_cache: Path | None = None) -> dict[str, object]:
    try:
        return SPEC.preflight(root, model_cache)
    except BaseException:
        evidence = root / EVIDENCE
        if (evidence / "preflight.error.json").is_file() and not (
            evidence / "preflight-terminal.json"
        ).exists():
            SPEC.finalize_preflight_error(root)
        if (evidence / "preflight-terminal.json").is_file() and not (
            evidence / "terminal-evidence-manifest.json"
        ).exists():
            TERMINAL_AUDIT.fixate_terminal_evidence(root)
        raise


def finalize_preflight_error(root: Path = ROOT) -> dict[str, object]:
    evidence = root / EVIDENCE
    result = (
        SPEC.finalize_preflight_error(root)
        if not (evidence / "preflight-terminal.json").exists()
        else lifecycle.read_json(evidence / "preflight-terminal.json")
    )
    if not (evidence / "terminal-evidence-manifest.json").exists():
        TERMINAL_AUDIT.fixate_terminal_evidence(root)
    return result


def run_once(root: Path = ROOT) -> dict[str, object]:
    try:
        result = SPEC.run_once(root)
    except BaseException:
        if (root / EVIDENCE / "observation-evidence-manifest.json").is_file():
            TERMINAL_AUDIT.fixate_terminal_evidence(root)
        raise
    TERMINAL_AUDIT.fixate_terminal_evidence(root)
    return result


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    result = TERMINAL_AUDIT.audit_evidence(root)
    evidence = root / EVIDENCE
    for stage in ("development", "holdout"):
        observed = evidence / f"{stage}.observed.json"
        if not observed.is_file():
            continue
        value = _read_object(observed)
        if (
            value.get("protocol_id") != PROTOCOL_ID
            or value.get("stage") != stage
            or value.get("retry_count") != 0
            or not isinstance(value.get("protocol_validity_gates"), list)
            or not isinstance(value.get("candidates"), list)
        ):
            raise ValueError(f"v21 observed evidence mismatch: {stage}")
        result[f"{stage}_protocol_validity_pass"] = value[
            "protocol_validity_pass"
        ]
        result[f"{stage}_selected_candidate_id"] = value[
            "selected_candidate_id"
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe intent-aware WSLC rank benchmark v21 once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prebuild",
        "preflight",
        "verify-preflight",
        "run",
        "audit",
        "finalize-preflight-error",
        "dependency-report",
    ):
        commands.add_parser(name)
    copy = commands.add_parser("model-copy-verify")
    copy.add_argument("--source-cache", required=True)
    copy.add_argument("--cache", required=True)
    copy.add_argument("--output", required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", required=True)
    read = commands.add_parser("read-json")
    read.add_argument("path")
    stage_init = commands.add_parser("stage-init")
    stage_init.add_argument("--stage", required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("--stage", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--stage", required=True)
    worker.add_argument("--kind", required=True)
    worker.add_argument("--replay", required=True)
    worker.add_argument("--database", required=True)
    worker.add_argument("--output", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--stage", required=True)
    failure = commands.add_parser("fail-stage")
    failure.add_argument("--stage", required=True)
    failure.add_argument("--message", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prebuild":
        result = validate_prebuild()
    elif arguments.command == "preflight":
        result = preflight()
    elif arguments.command == "verify-preflight":
        result = verify_preflight()
    elif arguments.command == "run":
        result = run_once()
    elif arguments.command == "audit":
        result = audit_evidence()
    elif arguments.command == "finalize-preflight-error":
        result = finalize_preflight_error()
    else:
        values = vars(arguments)
        result = SPEC.dispatch_container_command(
            arguments.command,
            **{
                key: value
                for key, value in values.items()
                if key != "command" and value is not None
            },
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

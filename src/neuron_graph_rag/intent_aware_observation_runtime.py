from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_observation as worker_base
from . import cross_encoder_precision_v19_performance_observation as predecessor
from . import (
    intent_aware_observation_engine,
    rank_observation_stage_contract,
    source_root_propagation,
)

JsonObject = dict[str, Any]
FixtureValidation = Callable[[Path], Mapping[str, Any]]


def _read_object(path: Path) -> JsonObject:
    return intent_aware_observation_engine.read_object(path)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _stage_rows(value: Mapping[str, Any], stage: str) -> list[JsonObject]:
    selected = value.get("stages", {}).get(stage)
    if isinstance(selected, Mapping):
        selected = selected.get("cases")
    if not isinstance(selected, list) or not all(
        isinstance(row, dict) for row in selected
    ):
        raise TypeError(f"intent-aware runtime stage rows must be objects: {stage}")
    return [dict(row) for row in selected]


def _normalized_query(value: str) -> str:
    characters = (
        character.casefold() if character.isalnum() else " " for character in value
    )
    return " ".join("".join(characters).split())


@dataclass(frozen=True)
class RealCorpusFixtureContract:
    engine: intent_aware_observation_engine.IntentAwareObservationEngine
    corpus_path: Path
    predecessor_query_paths: tuple[Path, ...]
    expected_provenance: tuple[tuple[str, Any], ...]
    document_count: int
    cohort_cardinality: tuple[tuple[str, int], ...]
    maximum_predecessor_similarity: float
    separation_result_prefix: str = "predecessor"
    relationship_edge_type: str = "informs"

    def _query_separation(
        self, root: Path, current: Sequence[str]
    ) -> dict[str, Any]:
        old = []
        for relative in self.predecessor_query_paths:
            fixture = _read_object(root / relative)
            old.extend(
                str(row["query"])
                for stage in intent_aware_observation_engine.STAGES
                for row in _stage_rows(fixture, stage)
            )
        normalized_current = [_normalized_query(value) for value in current]
        normalized_old = [_normalized_query(value) for value in old]
        maximum = max(
            SequenceMatcher(None, fresh, prior).ratio()
            for fresh in normalized_current
            for prior in normalized_old
        )
        if (
            set(normalized_current) & set(normalized_old)
            or maximum >= self.maximum_predecessor_similarity
        ):
            raise ValueError(
                "intent-aware queries copy, paraphrase, or closely transform predecessors"
            )
        prefix = self.separation_result_prefix
        return {
            f"{prefix}_exact_query_reuse_count": 0,
            f"{prefix}_max_normalized_similarity": maximum,
            f"{prefix}_similarity_limit_exclusive": (
                self.maximum_predecessor_similarity
            ),
        }

    def validate_worker(self, root: Path) -> dict[str, Any]:
        development = self.engine.load_worker_fixture(root, "development")
        holdout = self.engine.load_worker_fixture(root, "holdout")
        corpus = _read_object(root / self.corpus_path)
        provenance = corpus.get("acquisition_provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError("real corpus acquisition provenance must be an object")
        if any(
            provenance.get(key) != value for key, value in self.expected_provenance
        ):
            raise ValueError("real corpus acquisition provenance mismatch")
        paths = [str(row["path"]) for row in development.documents]
        hashes = provenance.get("content_sha256")
        if (
            len(paths) != self.document_count
            or len(set(paths)) != self.document_count
            or provenance.get("source_paths") != paths
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(paths)
        ):
            raise ValueError("real corpus path registry mismatch")
        for row in development.documents:
            observed = worker_base.sha256_bytes(str(row["text"]).encode("utf-8"))
            if hashes.get(row["path"]) != observed:
                raise ValueError(f"real corpus content hash mismatch: {row['path']}")
        if (
            development.documents != holdout.documents
            or development.relationships != holdout.relationships
        ):
            raise ValueError("development and holdout corpus differ")
        for relation in development.relationships:
            if (
                relation["source_path"] not in paths
                or relation["target_path"] not in paths
                or relation["edge_type"] != self.relationship_edge_type
            ):
                raise ValueError("real corpus relationship mismatch")
        queries = [*development.queries, *holdout.queries]
        if len({row["case_id"] for row in queries}) != len(queries) or len(
            {row["query"] for row in queries}
        ) != len(queries):
            raise ValueError("real query identities and text must be unique")
        expected_cohorts = dict(self.cohort_cardinality)
        for stage, rows in (
            ("development", development.queries),
            ("holdout", holdout.queries),
        ):
            observed = {
                name: sum(row["cohort"] == name for row in rows)
                for name in expected_cohorts
            }
            if observed != expected_cohorts:
                raise ValueError(f"real cohort cardinality mismatch: {stage}")
        commit = dict(self.expected_provenance).get("commit")
        return {
            "corpus_document_count": len(paths),
            "query_count": len(queries),
            "corpus_commit": commit,
            "corpus_fixture_sha256": predecessor.lifecycle.sha256_file(
                root / self.corpus_path
            ),
            **self._query_separation(
                root, [str(row["query"]) for row in queries]
            ),
        }

    def validate_protocol(self, root: Path) -> dict[str, Any]:
        contract = self.validate_worker(root)
        for stage in intent_aware_observation_engine.STAGES:
            finalizer = self.engine.load_finalizer_fixture(root, stage)
            gold_by_id = {str(row["case_id"]): row for row in finalizer.gold}
            corpus_paths = {
                str(row["path"])
                for row in self.engine.load_worker_fixture(root, stage).documents
            }
            for query in finalizer.queries:
                gold = gold_by_id[str(query["case_id"])]
                expected = gold.get("expected_path")
                forbidden = gold.get("forbidden_path")
                if (expected is None) == (forbidden is None):
                    raise ValueError("gold must select expected xor forbidden")
                selected = expected if expected is not None else forbidden
                if selected not in corpus_paths:
                    raise ValueError("gold path is outside the real corpus")
                if gold.get("cohort") != query.get("cohort"):
                    raise ValueError("query and gold cohorts differ")
                if query["cohort"] == "relation_linked" and (
                    gold.get("relation_seed_path") not in corpus_paths
                    or gold.get("relation_edge_type")
                    != self.relationship_edge_type
                ):
                    raise ValueError("relation gold is incomplete")
        return contract


@dataclass(frozen=True)
class IntentAwareRuntimeConfig:
    engine: intent_aware_observation_engine.IntentAwareObservationEngine
    fixture_contract: RealCorpusFixtureContract
    protocol_id: str
    freeze_commit: str
    root: Path
    module_name: str
    manifest_path: Path
    source_identity_path: Path
    audit_path: Path
    evidence_path: Path
    model_registry_path: Path
    runtime_volume: str
    container_root: PurePosixPath
    predecessor_artifact_count: int
    predecessor_anchor_sha256: Mapping[str, str]
    forbidden_volumes: Mapping[str, str]
    verification_commands_factory: Callable[[Path], tuple[list[str], ...]]
    protocol_artifact_registry_field: str
    protocol_artifact_count: int
    manifest_boundaries: tuple[tuple[str, Any], ...]
    container_identity_environment: str
    container_identity_prefix: str


@dataclass(frozen=True)
class IntentAwareRankObservationSpec(predecessor.V19RankObservationSpec):
    runtime: IntentAwareObservationRuntime = field(compare=False, repr=False)

    def validate_prebuild(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        result = super().validate_prebuild(project_root)
        protocol = self.runtime.config.fixture_contract.validate_protocol(project_root)
        manifest = self.manifest(project_root)
        field_name = self.runtime.config.protocol_artifact_registry_field
        registry = manifest.get(field_name)
        if (
            not isinstance(registry, dict)
            or len(registry) != self.runtime.config.protocol_artifact_count
        ):
            raise ValueError("intent-aware protocol artifact cardinality mismatch")
        for relative, expected in registry.items():
            if (
                not isinstance(relative, str)
                or not isinstance(expected, str)
                or predecessor.lifecycle.sha256_file(project_root / relative)
                != expected
            ):
                raise ValueError(f"intent-aware protocol artifact changed: {relative}")
        if any(
            manifest.get(key) != value
            for key, value in self.runtime.config.manifest_boundaries
        ):
            raise ValueError("intent-aware observation manifest boundary mismatch")
        return {
            **result,
            **protocol,
            "intent_aware_protocol_artifact_count": len(registry),
        }

    def run_stage_host(
        self,
        stage: str,
        root: Path,
        rows: list[dict[str, object]],
        claim_counts: dict[str, int],
    ) -> dict[str, object]:
        return self.runtime.run_stage_host(stage, root, rows, claim_counts)

    def dispatch_container_command(
        self, command: str, **arguments: str
    ) -> dict[str, Any]:
        return self.runtime.dispatch_container_command(command, **arguments)


class IntentAwareObservationRuntime:
    """Reusable worker/finalizer adapter around the existing rank lifecycle."""

    def __init__(self, config: IntentAwareRuntimeConfig) -> None:
        self.config = config
        self.engine = config.engine
        self.workers = tuple(
            (kind, replay)
            for kind in self.engine.spec.worker_kinds()
            for replay in ("primary", "replay")
        )
        container_source = config.container_root / "source"
        container_cache = config.container_root / "model-cache"
        container_protocol_source = config.container_root / "frozen-source"
        self.source_root_spec = source_root_propagation.SourceRootFreezeSpec(
            protocol_id=config.protocol_id,
            phase="performance-observation",
            predecessor_merge_commit=config.freeze_commit,
            frozen_protocol_commit=predecessor.V8_PROTOCOL_COMMIT,
            root=config.root,
            manifest_path=config.manifest_path,
            source_identity_path=config.source_identity_path,
            audit_path=config.audit_path,
            evidence_path=config.evidence_path,
            image=predecessor.IMAGE,
            image_id=predecessor.IMAGE_ID,
            wslc_version=predecessor.WSLC_VERSION,
            freeze_volume=config.runtime_volume,
            future_runtime_volume=config.runtime_volume,
            container_root=config.container_root,
            container_source=container_source,
            container_cache=container_cache,
            container_frozen_source=container_protocol_source,
            container_report=(
                config.container_root / "source-root-propagation-verification.json"
            ),
            container_source_identity=(
                container_source / config.source_identity_path.as_posix()
            ),
            old_frozen_source=predecessor.source_root_freeze.OLD_FROZEN_SOURCE,
            predecessor_artifact_count=config.predecessor_artifact_count,
            identity_schema="ngr.source-root-propagation/v1",
            evidence_stem="observation",
            report_name="source-root-propagation-verification.json",
            forbidden_volumes=config.forbidden_volumes,
            read_json=predecessor.lifecycle.read_json,
            sha256_file=predecessor.lifecycle.sha256_file,
            canonical_sha256=predecessor.lifecycle.canonical_sha256,
            write_json_exclusive=predecessor.lifecycle._write_json_exclusive,
        )
        self.stage_contract = (
            rank_observation_stage_contract.RankObservationStageContract(
                container_databases=config.container_root / "databases",
                container_runs=config.container_root / "runs",
                worker_slots_per_stage=len(self.workers),
            )
        )
        self.spec = IntentAwareRankObservationSpec(
            protocol_id=config.protocol_id,
            freeze_commit=config.freeze_commit,
            root=config.root,
            manifest_path=config.manifest_path,
            source_identity_path=config.source_identity_path,
            audit_path=config.audit_path,
            evidence_path=config.evidence_path,
            module_name=config.module_name,
            runtime_volume=config.runtime_volume,
            container_root=config.container_root,
            predecessor_artifact_count=config.predecessor_artifact_count,
            predecessor_anchor_sha256=config.predecessor_anchor_sha256,
            forbidden_volumes=config.forbidden_volumes,
            source_root_spec=self.source_root_spec,
            verification_commands_factory=config.verification_commands_factory,
            runtime=self,
        )
        self.terminal_audit = (
            rank_observation_stage_contract.RankObservationActualCountTerminalAudit(
                self.spec, self.stage_contract
            )
        )

    @property
    def container_source(self) -> PurePosixPath:
        return self.config.container_root / "source"

    @property
    def container_cache(self) -> PurePosixPath:
        return self.config.container_root / "model-cache"

    @property
    def container_databases(self) -> PurePosixPath:
        return self.config.container_root / "databases"

    @property
    def container_runs(self) -> PurePosixPath:
        return self.config.container_root / "runs"

    @property
    def container_archive(self) -> PurePosixPath:
        return self.config.container_root / "archive"

    def _documents(self, root: Path) -> list[dict[str, str]]:
        fixture = self.engine.load_worker_fixture(root, "development")
        return [
            {
                "path": str(row["path"]),
                "text": str(row["text"]),
                "sha256": worker_base.sha256_bytes(
                    str(row["text"]).encode("utf-8")
                ),
            }
            for row in fixture.documents
        ]

    def _index_corpus(
        self,
        graph: worker_base.NeuronGraphRAG,
        fixture: intent_aware_observation_engine.WorkerFixture,
        documents: Sequence[Mapping[str, str]],
    ) -> None:
        for row in documents:
            graph.add_document(
                f"github:{fixture.repository}:doc:{row['path']}",
                row["text"],
                metadata={
                    "repository": fixture.repository,
                    "commit": self.config.freeze_commit,
                    "path": row["path"],
                    "content_sha256": row["sha256"],
                },
            )
        for relation in fixture.relationships:
            graph.add_edge(
                f"github:{fixture.repository}:doc:{relation['source_path']}",
                f"github:{fixture.repository}:doc:{relation['target_path']}",
                str(relation["edge_type"]),
            )

    def _model_spec(self, root: Path, kind: str) -> dict[str, Any]:
        model = self.engine.spec.model(kind)
        registry = _read_object(root / self.config.model_registry_path).get("models")
        if not isinstance(registry, list):
            raise TypeError("model registry must be a list")
        selected = next(
            (
                row
                for row in registry
                if isinstance(row, dict)
                and row.get("model_id") == model.model_id
                and row.get("revision") == model.revision
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"frozen model identity missing: {kind}")
        return dict(selected)

    @staticmethod
    def _state_payload(
        graph: worker_base.NeuronGraphRAG,
        database: Path,
        cases: list[dict[str, Any]],
        edge_before: str,
        feedback_before: int,
        sqlite_before: str,
    ) -> dict[str, Any]:
        return {
            "database_id": worker_base.sha256_bytes(
                str(database.resolve()).encode()
            ),
            "cases": cases,
            "ranking_sha256": worker_base.canonical_sha256(cases),
            "activation_sha256": worker_base.canonical_sha256(
                worker_base._activation_state(graph)
            ),
            "edge_sha256_before": edge_before,
            "edge_sha256_after": worker_base.canonical_sha256(
                worker_base._edge_state(graph)
            ),
            "feedback_count_before": feedback_before,
            "feedback_count_after": graph.store.count_feedback(),
            "sqlite_sha256_before": sqlite_before,
            "sqlite_sha256_after": worker_base.canonical_sha256(
                worker_base._static_sqlite_state(graph)
            ),
        }

    def container_worker(
        self, stage: str, kind: str, replay: str, database: Path, output: Path
    ) -> dict[str, Any]:
        if (
            stage not in intent_aware_observation_engine.STAGES
            or (kind, replay) not in self.workers
        ):
            raise ValueError("intent-aware worker identity is not frozen")
        if database.exists() or output.exists():
            raise FileExistsError("intent-aware worker DB/output must be fresh")
        for path in (database.parent, output.parent, Path(str(self.container_cache))):
            path.resolve().relative_to(Path(str(self.config.container_root)))
        root = Path(str(self.container_source))
        fixture = self.engine.load_worker_fixture(root, stage)
        documents = self._documents(root)
        started = time.perf_counter()
        model_spec = None
        model_runtime = None
        if kind in {model.kind for model in self.engine.spec.models}:
            model_spec = self._model_spec(root, kind)
            model_runtime = worker_base._load_model(
                model_spec, Path(str(self.container_cache))
            )
        with worker_base.NeuronGraphRAG(
            database, config=worker_base.EngineConfig()
        ) as graph:
            self._index_corpus(graph, fixture, documents)
            edge_before = worker_base.canonical_sha256(
                worker_base._edge_state(graph)
            )
            feedback_before = graph.store.count_feedback()
            sqlite_before = worker_base.canonical_sha256(
                worker_base._static_sqlite_state(graph)
            )

            def prefilter(query: str, limit: int) -> list[dict[str, Any]]:
                trace = graph.search(
                    query, limit=limit, now=worker_base.OBSERVATION_NOW
                )
                return [
                    worker_base._baseline_hit_row(hit, rank)
                    for rank, hit in enumerate(trace.hits, start=1)
                ]

            def score(
                query: str,
                rows: Sequence[Mapping[str, Any]],
                source_documents: Sequence[Mapping[str, Any]],
                model: intent_aware_observation_engine.ModelIdentity,
            ) -> tuple[Sequence[Mapping[str, Any]], int]:
                if model.kind != kind or model_runtime is None:
                    raise ValueError("scorer model identity mismatch")
                return worker_base._score_case(
                    query, rows, source_documents, model_runtime
                )

            worker_result = self.engine.build_worker_cases(
                fixture,
                kind,
                prefilter_search=prefilter,
                score_query=score if model_runtime is not None else None,
            )
            payload = self._state_payload(
                graph,
                database,
                worker_result["cases"],
                edge_before,
                feedback_before,
                sqlite_before,
            )
        import psutil

        payload.update(
            {
                "protocol_id": self.config.protocol_id,
                "stage": stage,
                "kind": kind,
                "replay": replay,
                "container_id": os.environ.get(
                    self.config.container_identity_environment,
                    socket.gethostname(),
                ),
                "container_process_pid": os.getpid(),
                "model_id": worker_result["model_id"],
                "revision": worker_result["revision"],
                "metrics": {
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "peak_rss_bytes": psutil.Process().memory_info().rss,
                    "cache_bytes": worker_base._tree_bytes(
                        Path(str(self.container_cache))
                    ),
                    "pair_count": worker_result["pair_count"],
                },
            }
        )
        _write_json_exclusive(output, payload)
        return payload

    def container_claim(self, stage: str) -> dict[str, Any]:
        root = Path(str(self.container_source))
        contract = self.config.fixture_contract.validate_worker(root)
        fixture = self.engine.load_worker_fixture(root, stage)
        claim: dict[str, Any] = {
            "protocol_id": self.config.protocol_id,
            "stage": stage,
            "stage_identity": self.engine.spec.stage_identity(stage),
            "query_fixture_sha256": predecessor.lifecycle.sha256_file(
                root / self.engine.spec.fixture_paths.queries
            ),
            "corpus_fixture_sha256": predecessor.lifecycle.sha256_file(
                root / self.engine.spec.fixture_paths.corpus
            ),
            "corpus_commit": contract["corpus_commit"],
            "query_count": len(fixture.queries),
            "selection_policy": self.engine.spec.selection_policy,
            "retry_count": 0,
        }
        if stage == "holdout":
            development = _read_object(
                Path(str(self.container_archive)) / "development.observed.json"
            )
            selected = development.get("selected_candidate_id")
            if not isinstance(selected, str):
                raise ValueError("holdout requires a passing development selection")
            claim["selected_candidate_id"] = selected
        _write_json_exclusive(
            Path(str(self.container_runs / stage / "claim.json")), claim
        )
        return claim

    def _copy_stage_artifacts(
        self,
        stage: str,
        claim_path: Path,
        result_path: Path,
        worker_paths: Sequence[Path],
    ) -> dict[str, Any]:
        evidence = Path(
            str(self.container_source / self.config.evidence_path.as_posix())
        )
        raw_root = evidence / "raw" / stage
        raw_root.mkdir(parents=True, exist_ok=False)
        files: dict[str, str] = {}
        for source, target in (
            (claim_path, evidence / f"{stage}.claim.json"),
            (result_path, evidence / f"{stage}.observed.json"),
        ):
            raw = source.read_bytes()
            _write_bytes_exclusive(target, raw)
            files[target.relative_to(evidence).as_posix()] = (
                worker_base.sha256_bytes(raw)
            )
        for source in worker_paths:
            target = raw_root / source.name
            raw = source.read_bytes()
            _write_bytes_exclusive(target, raw)
            files[target.relative_to(evidence).as_posix()] = (
                worker_base.sha256_bytes(raw)
            )
        transport = {
            "protocol_id": self.config.protocol_id,
            "stage": stage,
            "status": "complete",
            "files": dict(sorted(files.items())),
            "byte_identity_verified": True,
            "retry_count": 0,
        }
        _write_json_exclusive(evidence / f"{stage}.transport.json", transport)
        return transport

    def container_finalize(self, stage: str) -> dict[str, Any]:
        root = Path(str(self.container_source))
        run_root = Path(str(self.container_runs / stage))
        claim_path = run_root / "claim.json"
        claim = _read_object(claim_path)
        raw = {
            (kind, replay): _read_object(run_root / f"{kind}-{replay}.json")
            for kind, replay in self.workers
        }
        result = self.engine.finalize_stage(
            stage,
            claim=claim,
            claim_sha256=predecessor.lifecycle.sha256_file(claim_path),
            raw=raw,
            fixture=self.engine.load_finalizer_fixture(root, stage),
            validity=intent_aware_observation_engine.ProtocolValidityInputs(
                fixture_contract_valid=bool(
                    self.config.fixture_contract.validate_worker(root)
                ),
                identity_separation_valid=(
                    self.engine.spec.stage_identity("development")
                    != self.engine.spec.stage_identity("holdout")
                ),
                document_count=len(
                    self.engine.load_worker_fixture(root, stage).documents
                ),
            ),
        )
        archive = Path(str(self.container_archive)) / f"{stage}.observed.json"
        _write_json_exclusive(archive, result)
        self._copy_stage_artifacts(
            stage,
            claim_path,
            archive,
            [
                run_root / f"{kind}-{replay}.json"
                for kind, replay in self.workers
            ],
        )
        return result

    def container_fail_stage(self, stage: str, message: str) -> dict[str, Any]:
        run_root = Path(str(self.container_runs / stage))
        claim_path = run_root / "claim.json"
        if not claim_path.is_file():
            raise FileNotFoundError("failed stage claim is unavailable")
        error = {
            "protocol_id": self.config.protocol_id,
            "stage": stage,
            "error": message,
            "retry_count": 0,
            "same_protocol_retry_allowed": False,
            "performance": "not assessed",
            "claim_sha256": predecessor.lifecycle.sha256_file(claim_path),
        }
        archive = Path(str(self.container_archive)) / f"{stage}.error.json"
        _write_json_exclusive(archive, error)
        evidence = Path(
            str(self.container_source / self.config.evidence_path.as_posix())
        )
        _write_bytes_exclusive(
            evidence / f"{stage}.claim.json", claim_path.read_bytes()
        )
        _write_bytes_exclusive(
            evidence / f"{stage}.error.json", archive.read_bytes()
        )
        workers = [
            run_root / f"{kind}-{replay}.json"
            for kind, replay in self.workers
            if (run_root / f"{kind}-{replay}.json").is_file()
        ]
        raw_root = evidence / "raw" / stage
        raw_root.mkdir(parents=True, exist_ok=False)
        files = {
            f"{stage}.claim.json": predecessor.lifecycle.sha256_file(
                evidence / f"{stage}.claim.json"
            ),
            f"{stage}.error.json": predecessor.lifecycle.sha256_file(
                evidence / f"{stage}.error.json"
            ),
        }
        for source in workers:
            target = raw_root / source.name
            shutil.copyfile(source, target)
            files[f"raw/{stage}/{source.name}"] = (
                predecessor.lifecycle.sha256_file(target)
            )
        _write_json_exclusive(
            evidence / f"{stage}.transport.json",
            {
                "protocol_id": self.config.protocol_id,
                "stage": stage,
                "status": "error",
                "files": dict(sorted(files.items())),
                "byte_identity_verified": True,
                "retry_count": 0,
            },
        )
        return error

    def run_stage_host(
        self,
        stage: str,
        root: Path,
        rows: list[dict[str, object]],
        claim_counts: dict[str, int],
    ) -> dict[str, object]:
        runner = predecessor.lifecycle.lifecycle.lifecycle
        initialized = json.loads(
            runner._run_logged(
                self.spec.container_command("stage-init", "--stage", stage),
                root,
                rows,
            )
        )
        self.stage_contract.validate_initialization(initialized, stage)
        runner._run_logged(
            self.spec.container_command("claim", "--stage", stage), root, rows
        )
        claim_counts[stage] += 1
        for kind, replay in self.workers:
            identity = f"{self.config.container_identity_prefix}-{stage}-{kind}-{replay}"
            command = self.spec.container_command(
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
                f"{self.config.container_identity_environment}={identity}",
            ]
            runner._run_logged(command, root, rows)
        result = json.loads(
            runner._run_logged(
                self.spec.container_command("finalize", "--stage", stage),
                root,
                rows,
            )
        )
        predecessor.lifecycle.lifecycle._export_volume_evidence(root, rows)
        return result

    def dispatch_container_command(
        self, command: str, **arguments: str
    ) -> dict[str, Any]:
        if command == "stage-init":
            return self.stage_contract.initialize_container_stage(arguments["stage"])
        if command == "claim":
            return self.container_claim(arguments["stage"])
        if command == "worker":
            return self.container_worker(
                arguments["stage"],
                arguments["kind"],
                arguments["replay"],
                Path(arguments["database"]),
                Path(arguments["output"]),
            )
        if command == "finalize":
            return self.container_finalize(arguments["stage"])
        if command == "fail-stage":
            return self.container_fail_stage(
                arguments["stage"], arguments["message"]
            )
        return super(IntentAwareRankObservationSpec, self.spec).dispatch_container_command(
            command, **arguments
        )

    def preflight(
        self, root: Path | None = None, model_cache: Path | None = None
    ) -> dict[str, object]:
        project_root = self.config.root if root is None else root
        try:
            return self.spec.preflight(project_root, model_cache)
        except BaseException:
            evidence = project_root / self.config.evidence_path
            if (evidence / "preflight.error.json").is_file() and not (
                evidence / "preflight-terminal.json"
            ).exists():
                self.spec.finalize_preflight_error(project_root)
            if (evidence / "preflight-terminal.json").is_file() and not (
                evidence / "terminal-evidence-manifest.json"
            ).exists():
                self.terminal_audit.fixate_terminal_evidence(project_root)
            raise

    def finalize_preflight_error(
        self, root: Path | None = None
    ) -> dict[str, object]:
        project_root = self.config.root if root is None else root
        evidence = project_root / self.config.evidence_path
        result = (
            self.spec.finalize_preflight_error(project_root)
            if not (evidence / "preflight-terminal.json").exists()
            else predecessor.lifecycle.read_json(
                evidence / "preflight-terminal.json"
            )
        )
        if not (evidence / "terminal-evidence-manifest.json").exists():
            self.terminal_audit.fixate_terminal_evidence(project_root)
        return result

    def run_once(self, root: Path | None = None) -> dict[str, object]:
        project_root = self.config.root if root is None else root
        try:
            result = self.spec.run_once(project_root)
        except BaseException:
            if (
                project_root
                / self.config.evidence_path
                / "observation-evidence-manifest.json"
            ).is_file():
                self.terminal_audit.fixate_terminal_evidence(project_root)
            raise
        self.terminal_audit.fixate_terminal_evidence(project_root)
        return result

    def audit_evidence(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.config.root if root is None else root
        result = self.terminal_audit.audit_evidence(project_root)
        evidence = project_root / self.config.evidence_path
        for stage in intent_aware_observation_engine.STAGES:
            observed = evidence / f"{stage}.observed.json"
            if not observed.is_file():
                continue
            value = _read_object(observed)
            if (
                value.get("protocol_id") != self.config.protocol_id
                or value.get("stage") != stage
                or value.get("retry_count") != 0
                or value.get("selection_policy")
                != self.engine.spec.selection_policy
            ):
                raise ValueError(f"observed evidence mismatch: {stage}")
            result[f"{stage}_protocol_validity_pass"] = value[
                "protocol_validity_pass"
            ]
            result[f"{stage}_selected_candidate_id"] = value[
                "selected_candidate_id"
            ]
        return result

    def main(self, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(
            description="Observe intent-aware rank fusion on fixed retrieval tasks"
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
        for name in ("stage-init", "claim", "finalize"):
            command = commands.add_parser(name)
            command.add_argument("--stage", required=True)
        worker = commands.add_parser("worker")
        worker.add_argument("--stage", required=True)
        worker.add_argument("--kind", required=True)
        worker.add_argument("--replay", required=True)
        worker.add_argument("--database", required=True)
        worker.add_argument("--output", required=True)
        failure = commands.add_parser("fail-stage")
        failure.add_argument("--stage", required=True)
        failure.add_argument("--message", required=True)
        arguments = parser.parse_args(argv)
        if arguments.command == "prebuild":
            result = self.spec.validate_prebuild()
        elif arguments.command == "preflight":
            result = self.preflight()
        elif arguments.command == "verify-preflight":
            result = self.spec.verify_preflight()
        elif arguments.command == "run":
            result = self.run_once()
        elif arguments.command == "audit":
            result = self.audit_evidence()
        elif arguments.command == "finalize-preflight-error":
            result = self.finalize_preflight_error()
        else:
            values = vars(arguments)
            result = self.dispatch_container_command(
                arguments.command,
                **{
                    key: str(value)
                    for key, value in values.items()
                    if key != "command"
                },
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

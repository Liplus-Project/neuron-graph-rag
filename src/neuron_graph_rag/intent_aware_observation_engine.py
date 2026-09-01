from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cross_encoder_precision_observation as observation_support
from . import intent_aware_rank_fusion

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
EXPECTED_PRODUCTION_SIGNAL_KEYS = frozenset(
    {
        "source_path",
        "prefilter_rank",
        "prefilter_score",
        "positive_logit",
        "exclusion_logits",
        "relation_paths",
    }
)
COHORTS = ("direct_lexical", "semantic_paraphrase", "relation_linked")
STAGES = ("development", "holdout")

JsonObject = dict[str, Any]
ReadObject = Callable[[Path], JsonObject]
PrefilterSearch = Callable[[str, int], Sequence[Mapping[str, Any]]]
ScoreQuery = Callable[
    [str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], "ModelIdentity"],
    tuple[Sequence[Mapping[str, Any]], int],
]


def read_object(path: Path) -> JsonObject:
    value = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"intent-aware observation JSON must be an object: {path}")
    return value


@dataclass(frozen=True)
class ObservationFixturePaths:
    corpus: Path
    queries: Path
    gold: Path
    model_registry: Path


@dataclass(frozen=True)
class ModelIdentity:
    kind: str
    candidate_id: str
    model_id: str
    revision: str


@dataclass(frozen=True)
class RetrievalArmIdentity:
    kind: str
    evidence_id: str
    query_mode: str


@dataclass(frozen=True)
class IntentAwareObservationSpec:
    protocol_id: str
    fixture_paths: ObservationFixturePaths
    stage_identities: tuple[tuple[str, str], ...]
    models: tuple[ModelIdentity, ...]
    baseline_evidence_id: str = "positive-clause-ngr-prefilter"
    baseline_kind: str = "baseline"
    baseline_query_mode: str = "positive"
    ablation_arms: tuple[RetrievalArmIdentity, ...] = ()
    selection_policy: str = "first-passing"
    protocol_gate_ids: tuple[str, ...] = PROTOCOL_GATE_IDS
    candidate_gate_ids: tuple[str, ...] = CANDIDATE_GATE_IDS
    fusion_config: intent_aware_rank_fusion.IntentAwareFusionConfig = field(
        default_factory=intent_aware_rank_fusion.IntentAwareFusionConfig
    )

    def stage_identity(self, stage: str) -> str:
        identities = dict(self.stage_identities)
        if stage not in identities:
            raise ValueError(f"unsupported intent-aware observation stage: {stage}")
        return identities[stage]

    def model(self, kind: str) -> ModelIdentity:
        for value in self.models:
            if value.kind == kind:
                return value
        raise ValueError(f"unsupported intent-aware observation model kind: {kind}")

    def retrieval_arm(self, kind: str) -> RetrievalArmIdentity:
        if kind == self.baseline_kind:
            return RetrievalArmIdentity(
                kind=kind,
                evidence_id=self.baseline_evidence_id,
                query_mode=self.baseline_query_mode,
            )
        for value in self.ablation_arms:
            if value.kind == kind:
                return value
        raise ValueError(f"unsupported retrieval arm kind: {kind}")

    def worker_kinds(self) -> tuple[str, ...]:
        return (
            self.baseline_kind,
            *(arm.kind for arm in self.ablation_arms),
            *(model.kind for model in self.models),
        )

    def validate(self) -> None:
        if (
            not self.protocol_id.strip()
            or not self.baseline_evidence_id.strip()
            or not self.baseline_kind.strip()
        ):
            raise ValueError("protocol and baseline identities must be non-empty")
        identities = dict(self.stage_identities)
        if (
            tuple(identities) != STAGES
            or len(set(identities.values())) != len(STAGES)
            or any(not value.strip() for value in identities.values())
        ):
            raise ValueError("development and holdout identities must be distinct")
        paths = (
            self.fixture_paths.corpus,
            self.fixture_paths.queries,
            self.fixture_paths.gold,
            self.fixture_paths.model_registry,
        )
        if len(set(paths)) != len(paths) or any(path.is_absolute() for path in paths):
            raise ValueError("fixture paths must be distinct repository-relative paths")
        if not self.models:
            raise ValueError("at least one model identity is required")
        for attribute in ("kind", "candidate_id", "model_id", "revision"):
            values = [getattr(model, attribute) for model in self.models]
            if len(set(values)) != len(values) or any(
                not value.strip() for value in values
            ):
                raise ValueError(
                    f"model {attribute} values must be unique and non-empty"
                )
        retrieval_arms = (
            RetrievalArmIdentity(
                self.baseline_kind,
                self.baseline_evidence_id,
                self.baseline_query_mode,
            ),
            *self.ablation_arms,
        )
        for attribute in ("kind", "evidence_id"):
            values = [getattr(arm, attribute) for arm in retrieval_arms]
            if len(set(values)) != len(values) or any(
                not value.strip() for value in values
            ):
                raise ValueError(
                    f"retrieval arm {attribute} values must be unique and non-empty"
                )
        if any(arm.query_mode not in {"full", "positive"} for arm in retrieval_arms):
            raise ValueError("retrieval arm query mode must be full or positive")
        if len(set(self.worker_kinds())) != len(self.worker_kinds()):
            raise ValueError("worker kinds must be unique")
        if self.selection_policy not in {
            "first-passing",
            "lowest-development-primary-latency",
        }:
            raise ValueError("unsupported candidate selection policy")
        if self.protocol_gate_ids != PROTOCOL_GATE_IDS:
            raise ValueError("protocol gate IDs differ from the v20 contract")
        if self.candidate_gate_ids != CANDIDATE_GATE_IDS:
            raise ValueError("candidate gate IDs differ from the v20 contract")
        if self.fusion_config != intent_aware_rank_fusion.IntentAwareFusionConfig():
            raise ValueError("fusion weights differ from the v20 contract")
        if (
            intent_aware_rank_fusion.PRODUCTION_SIGNAL_KEYS
            != EXPECTED_PRODUCTION_SIGNAL_KEYS
        ):
            raise ValueError("production signal schema differs from the v20 contract")


@dataclass(frozen=True)
class WorkerFixture:
    repository: str
    documents: tuple[JsonObject, ...]
    relationships: tuple[JsonObject, ...]
    queries: tuple[JsonObject, ...]


@dataclass(frozen=True)
class FinalizerFixture:
    queries: tuple[JsonObject, ...]
    gold: tuple[JsonObject, ...]


@dataclass(frozen=True)
class ProtocolValidityInputs:
    fixture_contract_valid: bool
    identity_separation_valid: bool
    document_count: int


class IntentAwareObservationEngine:
    """Pure worker/finalizer behavior without lifecycle or evidence ownership."""

    def __init__(
        self,
        spec: IntentAwareObservationSpec,
        *,
        object_reader: ReadObject = read_object,
    ) -> None:
        spec.validate()
        self.spec = spec
        self._read_object = object_reader

    @staticmethod
    def _stage_rows(value: Mapping[str, Any], stage: str) -> list[JsonObject]:
        stages = value.get("stages")
        if not isinstance(stages, Mapping) or stage not in stages:
            raise ValueError(f"stage fixture is unavailable: {stage}")
        raw = stages[stage]
        if isinstance(raw, Mapping):
            raw = raw.get("cases")
        if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
            raise TypeError(f"stage rows must be objects: {stage}")
        return [dict(row) for row in raw]

    def load_worker_fixture(self, root: Path, stage: str) -> WorkerFixture:
        """Load corpus/query input only; the gold path is intentionally unreachable."""
        self.spec.stage_identity(stage)
        corpus = self._read_object(root / self.spec.fixture_paths.corpus)
        query_fixture = self._read_object(root / self.spec.fixture_paths.queries)
        if corpus.get("protocol_id") != self.spec.protocol_id:
            raise ValueError("corpus protocol identity mismatch")
        if query_fixture.get("protocol_id") != self.spec.protocol_id:
            raise ValueError("query protocol identity mismatch")
        stage_value = query_fixture["stages"][stage]
        if stage_value.get("identity") != self.spec.stage_identity(stage):
            raise ValueError("query stage identity mismatch")
        documents = corpus.get("documents")
        relationships = corpus.get("relationships")
        if not isinstance(documents, list) or not isinstance(relationships, list):
            raise TypeError("corpus documents and relationships must be lists")
        if any(set(row) != {"path", "text"} for row in documents):
            raise ValueError("worker document schema contains non-production fields")
        if any(
            set(row) != {"source_path", "target_path", "edge_type"}
            for row in relationships
        ):
            raise ValueError(
                "worker relationship schema contains non-production fields"
            )
        queries = self._stage_rows(query_fixture, stage)
        if any(
            set(row) != {"case_id", "cohort", "language", "query"} for row in queries
        ):
            raise ValueError("worker query schema contains evaluation-only fields")
        repository = corpus.get("repository")
        if not isinstance(repository, str) or not repository.strip():
            raise TypeError("corpus repository must be a non-empty string")
        return WorkerFixture(
            repository=repository,
            documents=tuple(dict(row) for row in documents),
            relationships=tuple(dict(row) for row in relationships),
            queries=tuple(queries),
        )

    def load_finalizer_fixture(self, root: Path, stage: str) -> FinalizerFixture:
        """Load evaluation-only gold solely on the finalizer surface."""
        self.spec.stage_identity(stage)
        query_fixture = self._read_object(root / self.spec.fixture_paths.queries)
        gold_fixture = self._read_object(root / self.spec.fixture_paths.gold)
        if (
            query_fixture.get("protocol_id") != self.spec.protocol_id
            or gold_fixture.get("protocol_id") != self.spec.protocol_id
        ):
            raise ValueError("finalizer fixture protocol identity mismatch")
        queries = self._stage_rows(query_fixture, stage)
        gold = self._stage_rows(gold_fixture, stage)
        if {row["case_id"] for row in queries} != {row["case_id"] for row in gold}:
            raise ValueError("query/gold case identities differ")
        return FinalizerFixture(queries=tuple(queries), gold=tuple(gold))

    def build_worker_cases(
        self,
        fixture: WorkerFixture,
        kind: str,
        *,
        prefilter_search: PrefilterSearch,
        score_query: ScoreQuery | None = None,
    ) -> dict[str, Any]:
        try:
            retrieval_arm = self.spec.retrieval_arm(kind)
        except ValueError:
            retrieval_arm = None
        model = None if retrieval_arm is not None else self.spec.model(kind)
        if (model is None) != (score_query is None):
            raise ValueError(
                "retrieval arms have no scorer and model candidates require one"
            )
        pair_count = 0
        cases: list[JsonObject] = []
        document_count = len(fixture.documents)
        for query in fixture.queries:
            query_text = str(query["query"])
            intent = intent_aware_rank_fusion.decompose_query_intent(query_text)
            search_query = (
                query_text
                if retrieval_arm is not None and retrieval_arm.query_mode == "full"
                else intent.positive_query
            )
            prefilter = [
                dict(row) for row in prefilter_search(search_query, document_count)
            ]
            if len(prefilter) != document_count:
                raise ValueError("prefilter must rank the complete protocol corpus")
            if model is None:
                cases.append(
                    {
                        "case_id": query["case_id"],
                        "cohort": query["cohort"],
                        "ranked_hits": prefilter,
                    }
                )
                continue
            assert score_query is not None
            positive_rows, pairs = score_query(
                intent.positive_query, prefilter, fixture.documents, model
            )
            pair_count += pairs
            exclusion_rows = []
            for exclusion in intent.exclusion_queries:
                scored, pairs = score_query(
                    exclusion, prefilter, fixture.documents, model
                )
                exclusion_rows.append({str(row["source_path"]): row for row in scored})
                pair_count += pairs
            positive = {str(row["source_path"]): row for row in positive_rows}
            signals = [
                {
                    "source_path": row["source_path"],
                    "prefilter_rank": row["rank"],
                    "prefilter_score": row["ngr_score"],
                    "positive_logit": positive[str(row["source_path"])]["raw_logit"],
                    "exclusion_logits": [
                        values[str(row["source_path"])]["raw_logit"]
                        for values in exclusion_rows
                    ],
                    "relation_paths": row["relation_paths"],
                }
                for row in prefilter
            ]
            intent_aware_rank_fusion.validate_prefilter_signals(signals, intent=intent)
            ranked = intent_aware_rank_fusion.fuse_intent_aware_ranks(
                query_text, signals, config=self.spec.fusion_config
            )
            cases.append(
                {
                    "case_id": query["case_id"],
                    "cohort": query["cohort"],
                    "production_signals": signals,
                    "ranked_hits": ranked,
                }
            )
        return {
            "cases": cases,
            "pair_count": pair_count,
            "model_id": None if model is None else model.model_id,
            "revision": None if model is None else model.revision,
        }

    @staticmethod
    def _returned_paths(case: Mapping[str, Any]) -> list[str]:
        hits = case.get("ranked_hits")
        if not isinstance(hits, list):
            raise TypeError("ranked hits must be a list")
        return [str(row["source_path"]) for row in hits[:5] if isinstance(row, Mapping)]

    def quality(
        self,
        cases: Sequence[Mapping[str, Any]],
        gold_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        by_case = {str(row["case_id"]): row for row in cases}
        cohort_values: dict[str, list[float]] = {name: [] for name in COHORTS}
        forbidden_count = 0
        case_rows = []
        for gold in gold_rows:
            case_id = str(gold["case_id"])
            paths = self._returned_paths(by_case[case_id])
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
        if any(not values for values in cohort_values.values()):
            raise ValueError("every positive quality cohort requires at least one case")
        return {
            "cohorts": {
                name: {
                    "mrr": sum(values) / len(values),
                    "hit_at_5": sum(value > 0.0 for value in values) / len(values),
                }
                for name, values in cohort_values.items()
            },
            "negative_forbidden_top_5_count": forbidden_count,
            "cases": case_rows,
        }

    @staticmethod
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

    @staticmethod
    def _gate_rows(ids: Sequence[str], values: Sequence[bool]) -> list[dict[str, Any]]:
        return [
            {"gate_id": gate_id, "hard": True, "passed": bool(value)}
            for gate_id, value in zip(ids, values, strict=True)
        ]

    @staticmethod
    def _relation_path(gold: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "seed_path": gold["relation_seed_path"],
            "target_path": gold["expected_path"],
            "edge_type": gold["relation_edge_type"],
            "step_count": 1,
        }

    def protocol_gates(
        self,
        validity: ProtocolValidityInputs,
        baseline_cases: Sequence[Mapping[str, Any]],
        candidate_rows: Sequence[Mapping[str, Any]],
        states: Sequence[Mapping[str, Any]],
        gold_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        baseline_ok = all(
            len(case.get("ranked_hits", [])) == validity.document_count
            and len(set(self._returned_paths(case))) == 5
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
            relation_ok &= bool(
                target and self._relation_path(gold) in target["relation_paths"]
            )
        production_ok = True
        for candidate in candidate_rows:
            for case in candidate["cases"]:
                for signal in case["production_signals"]:
                    production_ok &= set(signal) == EXPECTED_PRODUCTION_SIGNAL_KEYS
                    lowered = " ".join(signal).lower()
                    production_ok &= not any(
                        token in lowered
                        for token in (
                            "gold",
                            "expected",
                            "forbidden",
                            "label",
                            "relevance",
                        )
                    )
        immutable = all(self._state_is_immutable(state) for state in states)
        return self._gate_rows(
            self.spec.protocol_gate_ids,
            (
                validity.fixture_contract_valid,
                validity.identity_separation_valid,
                baseline_ok,
                relation_ok,
                production_ok,
                immutable,
            ),
        )

    def candidate_gates(
        self,
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
            baseline_paths = self._returned_paths(baseline_cases[case_id])
            candidate_paths = self._returned_paths(candidate_cases[case_id])
            expected = gold.get("expected_path")
            forbidden = gold.get("forbidden_path")
            if expected is not None:
                baseline_rank = (
                    baseline_paths.index(expected) + 1
                    if expected in baseline_paths
                    else None
                )
                candidate_rank = (
                    candidate_paths.index(expected) + 1
                    if expected in candidate_paths
                    else None
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
            expected_ranking = intent_aware_rank_fusion.fuse_intent_aware_ranks(
                str(case["query"]),
                case["production_signals"],
                config=self.spec.fusion_config,
            )
            recomputed &= expected_ranking == case["ranked_hits"]
            signals = {row["source_path"]: row for row in case["production_signals"]}
            relation_preserved &= all(
                row["relation_paths"] == signals[row["source_path"]]["relation_paths"]
                for row in case["ranked_hits"]
            )
            if gold.get("cohort") == "relation_linked" and expected in candidate_paths:
                hit = next(
                    row for row in case["ranked_hits"] if row["source_path"] == expected
                )
                relation_preserved &= self._relation_path(gold) in hit["relation_paths"]
        baseline_quality = baseline["quality"]
        candidate_quality = candidate["quality"]
        cohort_nonregression = all(
            candidate_quality["cohorts"][cohort][metric]
            >= baseline_quality["cohorts"][cohort][metric]
            for cohort in COHORTS
            for metric in ("mrr", "hit_at_5")
        )
        negative_gate = (
            negative_nonworse
            and candidate_forbidden <= baseline_forbidden
            and candidate_forbidden < baseline_forbidden
        )
        return self._gate_rows(
            self.spec.candidate_gate_ids,
            (
                positive_nonregression,
                cohort_nonregression,
                negative_gate,
                completeness,
                recomputed,
                relation_preserved,
            ),
        )

    def _validate_worker_packets(
        self,
        raw: Mapping[tuple[str, str], Mapping[str, Any]],
        stage: str,
    ) -> None:
        expected_keys = {
            (kind, replay)
            for kind in self.spec.worker_kinds()
            for replay in ("primary", "replay")
        }
        actual_keys = set(raw)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys, key=repr)
            raise ValueError(
                f"worker packet set mismatch: missing={missing!r}, extra={extra!r}"
            )
        for kind, replay in sorted(expected_keys):
            packet = raw[(kind, replay)]
            for identity_field, expected in (
                ("protocol_id", self.spec.protocol_id),
                ("stage", stage),
                ("kind", kind),
                ("replay", replay),
            ):
                if packet.get(identity_field) != expected:
                    raise ValueError(
                        f"{kind} {replay} worker packet {identity_field} "
                        "identity mismatch"
                    )
            try:
                self.spec.retrieval_arm(kind)
                model = None
            except ValueError:
                model = self.spec.model(kind)
            expected_model_id = None if model is None else model.model_id
            expected_revision = None if model is None else model.revision
            if (
                "model_id" not in packet
                or "revision" not in packet
                or packet["model_id"] != expected_model_id
                or packet["revision"] != expected_revision
            ):
                raise ValueError(
                    f"{kind} {replay} worker packet model identity mismatch"
                )

    def finalize_stage(
        self,
        stage: str,
        *,
        claim: Mapping[str, Any],
        claim_sha256: str,
        raw: Mapping[tuple[str, str], Mapping[str, Any]],
        fixture: FinalizerFixture,
        validity: ProtocolValidityInputs,
    ) -> dict[str, Any]:
        if claim.get("protocol_id") != self.spec.protocol_id:
            raise ValueError("claim protocol identity mismatch")
        if claim.get("stage_identity") != self.spec.stage_identity(stage):
            raise ValueError("claim stage identity mismatch")
        self._validate_worker_packets(raw, stage)
        baseline_key = self.spec.baseline_kind
        baseline_primary = raw[(baseline_key, "primary")]
        baseline_replay = raw[(baseline_key, "replay")]
        if baseline_primary["cases"] != baseline_replay["cases"]:
            raise ValueError("baseline replay cases differ")
        baseline = {
            "candidate_id": self.spec.baseline_evidence_id,
            "cases": baseline_primary["cases"],
            "quality": self.quality(baseline_primary["cases"], fixture.gold),
            "state": observation_support._combine_state(
                baseline_primary, baseline_replay
            ),
            "metrics": {
                "primary": baseline_primary["metrics"],
                "replay": baseline_replay["metrics"],
            },
        }
        ablations = []
        for arm in self.spec.ablation_arms:
            primary = raw[(arm.kind, "primary")]
            replay = raw[(arm.kind, "replay")]
            if primary["cases"] != replay["cases"]:
                raise ValueError(f"{arm.kind} replay cases differ")
            ablations.append(
                {
                    "candidate_id": arm.evidence_id,
                    "query_mode": arm.query_mode,
                    "cases": primary["cases"],
                    "quality": self.quality(primary["cases"], fixture.gold),
                    "state": observation_support._combine_state(primary, replay),
                    "metrics": {
                        "primary": primary["metrics"],
                        "replay": replay["metrics"],
                    },
                }
            )
        candidates = []
        states = [baseline["state"]]
        states.extend(arm["state"] for arm in ablations)
        query_by_id = {
            str(row["case_id"]): str(row["query"]) for row in fixture.queries
        }
        for model in self.spec.models:
            primary = raw[(model.kind, "primary")]
            replay = raw[(model.kind, "replay")]
            if primary["cases"] != replay["cases"]:
                raise ValueError(f"{model.kind} replay cases differ")
            cases = [
                {**case, "query": query_by_id[str(case["case_id"])]}
                for case in primary["cases"]
            ]
            state = observation_support._combine_state(primary, replay)
            states.append(state)
            candidates.append(
                {
                    "candidate_id": model.candidate_id,
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "cases": cases,
                    "quality": self.quality(cases, fixture.gold),
                    "state": state,
                    "metrics": {
                        "primary": primary["metrics"],
                        "replay": replay["metrics"],
                    },
                }
            )
        protocol_gates = self.protocol_gates(
            validity, baseline["cases"], candidates, states, fixture.gold
        )
        protocol_pass = all(row["passed"] for row in protocol_gates)
        passing: list[dict[str, Any]] = []
        for candidate in candidates:
            gates = (
                self.candidate_gates(candidate, baseline, fixture.gold)
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
            if eligible and candidate["all_candidate_gates_pass"]:
                passing.append(candidate)
        selected: str | None = None
        if stage == "holdout" or self.spec.selection_policy == "first-passing":
            if passing:
                selected = str(passing[0]["candidate_id"])
        elif passing:

            def latency_key(candidate: Mapping[str, Any]) -> tuple[float, str]:
                latency = candidate["metrics"]["primary"].get("latency_ms")
                if (
                    not isinstance(latency, (int, float))
                    or isinstance(latency, bool)
                    or not math.isfinite(float(latency))
                    or latency < 0
                ):
                    raise ValueError("candidate primary latency is invalid")
                return float(latency), str(candidate["candidate_id"])

            selected = str(min(passing, key=latency_key)["candidate_id"])
        return {
            "protocol_id": self.spec.protocol_id,
            "stage": stage,
            "stage_identity": claim["stage_identity"],
            "protocol_validity_gates": protocol_gates,
            "protocol_validity_pass": protocol_pass,
            "candidate_gates_evaluated": protocol_pass,
            "baseline": baseline,
            "ablations": ablations,
            "candidates": candidates,
            "selection_policy": self.spec.selection_policy,
            "selected_candidate_id": selected,
            "all_hard_gates_pass": selected is not None,
            "performance": "assessed",
            "retry_count": 0,
            "claim_sha256": claim_sha256,
        }

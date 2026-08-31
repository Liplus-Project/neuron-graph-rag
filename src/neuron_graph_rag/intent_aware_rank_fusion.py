from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PRODUCTION_SIGNAL_KEYS = frozenset(
    {
        "source_path",
        "prefilter_rank",
        "prefilter_score",
        "positive_logit",
        "exclusion_logits",
        "relation_paths",
    }
)
RELATION_PATH_KEYS = frozenset(
    {"seed_path", "target_path", "edge_type", "step_count"}
)
_LEAKAGE_KEY_PARTS = ("gold", "expected", "forbidden", "label", "relevance")
_EXCLUSION_MARKER = re.compile(
    r"(?i:\b(?:without|excluding|exclude|except|avoid|not)\b)"
    r"|(?:除外)"
)
_JA_SUFFIX_EXCLUSION = re.compile(
    r"(?P<clause>[^,、;]+?)(?:を除外|を除く|は含めない)"
)
_RELATION_MARKERS = (
    "related",
    "relation",
    "linked",
    "depends",
    "dependency",
    "supersedes",
    "関連",
    "関係",
    "依存",
    "継承",
    "置換",
)


def _clean_clause(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n,;:、。"))


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _reject_leakage_keys(value: Mapping[str, Any], *, scope: str) -> None:
    for key in value:
        lowered = key.lower()
        if any(part in lowered for part in _LEAKAGE_KEY_PARTS):
            raise ValueError(f"{scope} contains evaluation-only key: {key}")


@dataclass(frozen=True)
class QueryIntent:
    original_query: str
    positive_query: str
    exclusion_queries: tuple[str, ...]
    relation_requested: bool


@dataclass(frozen=True)
class RelationPath:
    seed_path: str
    target_path: str
    edge_type: str
    step_count: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_path: str
    ) -> RelationPath:
        _reject_leakage_keys(value, scope="relation path")
        if set(value) != RELATION_PATH_KEYS:
            raise ValueError("relation path keys must match the production schema")
        strings = {}
        for key in ("seed_path", "target_path", "edge_type"):
            item = value.get(key)
            if not isinstance(item, str) or not item.strip():
                raise TypeError(f"relation path {key} must be a non-empty string")
            strings[key] = item
        step_count = value.get("step_count")
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count < 1
        ):
            raise TypeError("relation path step_count must be a positive integer")
        if strings["target_path"] != source_path:
            raise ValueError("relation path target must match its candidate source")
        return cls(step_count=step_count, **strings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_path": self.seed_path,
            "target_path": self.target_path,
            "edge_type": self.edge_type,
            "step_count": self.step_count,
        }


@dataclass(frozen=True)
class RankSignal:
    source_path: str
    prefilter_rank: int
    prefilter_score: float
    positive_logit: float
    exclusion_logits: tuple[float, ...]
    relation_paths: tuple[RelationPath, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RankSignal:
        _reject_leakage_keys(value, scope="rank signal")
        if set(value) != PRODUCTION_SIGNAL_KEYS:
            raise ValueError("rank signal keys must match the production schema")
        source_path = value.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise TypeError("source_path must be a non-empty string")
        prefilter_rank = value.get("prefilter_rank")
        if (
            isinstance(prefilter_rank, bool)
            or not isinstance(prefilter_rank, int)
            or prefilter_rank < 1
        ):
            raise TypeError("prefilter_rank must be a positive integer")
        raw_exclusions = value.get("exclusion_logits")
        if not isinstance(raw_exclusions, list):
            raise TypeError("exclusion_logits must be a list")
        raw_paths = value.get("relation_paths")
        if not isinstance(raw_paths, list):
            raise TypeError("relation_paths must be a list")
        paths = tuple(
            RelationPath.from_mapping(path, source_path=source_path)
            for path in raw_paths
            if isinstance(path, Mapping)
        )
        if len(paths) != len(raw_paths):
            raise TypeError("relation_paths entries must be objects")
        return cls(
            source_path=source_path,
            prefilter_rank=prefilter_rank,
            prefilter_score=_finite_number(
                value.get("prefilter_score"), "prefilter_score"
            ),
            positive_logit=_finite_number(
                value.get("positive_logit"), "positive_logit"
            ),
            exclusion_logits=tuple(
                _finite_number(item, "exclusion logit") for item in raw_exclusions
            ),
            relation_paths=paths,
        )


@dataclass(frozen=True)
class IntentAwareFusionConfig:
    prefilter_rank_weight: float = 0.15
    prefilter_score_weight: float = 0.15
    positive_logit_weight: float = 1.0
    exclusion_penalty_weight: float = 1.0
    relation_path_bonus: float = 0.25
    rrf_constant: int = 60
    top_k: int = 5

    def validate(self) -> None:
        for name in (
            "prefilter_rank_weight",
            "prefilter_score_weight",
            "positive_logit_weight",
            "exclusion_penalty_weight",
            "relation_path_bonus",
        ):
            if _finite_number(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.rrf_constant < 1 or self.top_k < 1:
            raise ValueError("rrf_constant and top_k must be positive")


def decompose_query_intent(query_text: str) -> QueryIntent:
    if not isinstance(query_text, str) or not query_text.strip():
        raise TypeError("query_text must be a non-empty string")
    original = _clean_clause(query_text)
    suffix_exclusions = [
        _clean_clause(match.group("clause"))
        for match in _JA_SUFFIX_EXCLUSION.finditer(original)
    ]
    remaining = _clean_clause(_JA_SUFFIX_EXCLUSION.sub("", original))
    matches = list(_EXCLUSION_MARKER.finditer(remaining))
    if not matches and not suffix_exclusions:
        return QueryIntent(
            original_query=original,
            positive_query=original,
            exclusion_queries=(),
            relation_requested=any(
                marker in original.casefold() for marker in _RELATION_MARKERS
            ),
        )
    positive = (
        _clean_clause(remaining[: matches[0].start()]) if matches else remaining
    )
    if not positive:
        raise ValueError("an exclusion-only query has no positive retrieval intent")
    exclusions = list(suffix_exclusions)
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(remaining)
        )
        clause = _clean_clause(remaining[match.end() : end])
        if not clause:
            raise ValueError("each exclusion marker must have a clause")
        exclusions.append(clause)
    return QueryIntent(
        original_query=original,
        positive_query=positive,
        exclusion_queries=tuple(exclusions),
        relation_requested=any(
            marker in original.casefold() for marker in _RELATION_MARKERS
        ),
    )


def validate_prefilter_signals(
    signals: Sequence[Mapping[str, Any]], *, intent: QueryIntent
) -> tuple[RankSignal, ...]:
    if not signals:
        raise ValueError("at least one prefilter signal is required")
    parsed = tuple(RankSignal.from_mapping(value) for value in signals)
    sources = [row.source_path for row in parsed]
    ranks = [row.prefilter_rank for row in parsed]
    if len(set(sources)) != len(sources):
        raise ValueError("prefilter source identities must be unique")
    if len(set(ranks)) != len(ranks):
        raise ValueError("prefilter ranks must be unique")
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("prefilter ranks must be contiguous from one")
    for row in parsed:
        if len(row.exclusion_logits) != len(intent.exclusion_queries):
            raise ValueError(
                "exclusion logits must match decomposed exclusion clause count"
            )
    return parsed


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def fuse_intent_aware_ranks(
    query_text: str,
    signals: Sequence[Mapping[str, Any]],
    *,
    config: IntentAwareFusionConfig | None = None,
) -> list[dict[str, Any]]:
    intent = decompose_query_intent(query_text)
    rows = validate_prefilter_signals(signals, intent=intent)
    selected_config = config or IntentAwareFusionConfig()
    selected_config.validate()
    scores = [row.prefilter_score for row in rows]
    low, high = min(scores), max(scores)
    normalized = {
        row.source_path: 0.5
        if high == low
        else (row.prefilter_score - low) / (high - low)
        for row in rows
    }
    fused = []
    for row in rows:
        rank_component = selected_config.prefilter_rank_weight / (
            selected_config.rrf_constant + row.prefilter_rank
        )
        score_component = (
            selected_config.prefilter_score_weight * normalized[row.source_path]
        )
        positive_component = (
            selected_config.positive_logit_weight * _sigmoid(row.positive_logit)
        )
        exclusion_component = (
            selected_config.exclusion_penalty_weight
            * max((_sigmoid(value) for value in row.exclusion_logits), default=0.0)
        )
        relation_component = (
            selected_config.relation_path_bonus
            if intent.relation_requested and row.relation_paths
            else 0.0
        )
        final_score = (
            rank_component
            + score_component
            + positive_component
            - exclusion_component
            + relation_component
        )
        fused.append(
            {
                "source_path": row.source_path,
                "final_score": final_score,
                "prefilter_rank": row.prefilter_rank,
                "prefilter_score": row.prefilter_score,
                "positive_logit": row.positive_logit,
                "exclusion_logits": list(row.exclusion_logits),
                "relation_paths": [path.to_dict() for path in row.relation_paths],
                "components": {
                    "prefilter_rank": rank_component,
                    "prefilter_score": score_component,
                    "positive_intent": positive_component,
                    "exclusion_penalty": exclusion_component,
                    "relation_preservation": relation_component,
                },
            }
        )
    return sorted(
        fused,
        key=lambda row: (-float(row["final_score"]), str(row["source_path"])),
    )[: selected_config.top_k]

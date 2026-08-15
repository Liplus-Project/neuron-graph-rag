from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from typing import Any, Mapping

from .evidence_feedback import EngineConfig, NeuronGraphRAG


FEEDBACK_CONFIG_FIELDS = (
    "feedback_learning_rate",
    "sibling_feedback_normalization",
    "maximum_edge_weight",
    "relation_feedback_evidence_quorum",
    "confirmed_outcome_reinforcement",
    "confirmation_decay_ratio",
)
SEARCH_SURFACES = ("combined", "relation")


def config_fingerprint(value: Mapping[str, Any]) -> str:
    canonical = (
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def effective_config(config: EngineConfig) -> dict[str, dict[str, Any]]:
    raw = asdict(config)
    names = {field.name for field in fields(EngineConfig)}
    if set(raw) != names:
        raise RuntimeError("EngineConfig serialization is incomplete")
    return {
        "retrieval": {
            name: raw[name]
            for name in sorted(names - set(FEEDBACK_CONFIG_FIELDS))
        },
        "feedback": {
            name: raw[name]
            for name in sorted(FEEDBACK_CONFIG_FIELDS)
        },
    }


def effective_config_provenance(config: EngineConfig) -> dict[str, Any]:
    value = effective_config(config)
    return {
        "effective_config": value,
        "retrieval_config_fingerprint": config_fingerprint(value["retrieval"]),
        "feedback_config_fingerprint": config_fingerprint(value["feedback"]),
        "full_config_fingerprint": config_fingerprint(value),
    }


def effective_search_surface(config: EngineConfig) -> str:
    if (
        config.confirmed_outcome_reinforcement
        or config.sibling_feedback_normalization > 0.0
    ):
        return "relation"
    return "combined"


def search_with_surface(
    engine: NeuronGraphRAG,
    query: str,
    *,
    limit: int,
    search_surface: str,
    now: float | None = None,
) -> Any:
    if search_surface == "combined":
        return engine.search(query, limit=limit, now=now)
    if search_surface == "relation":
        return engine.search_channels(query, limit=limit, now=now).relation
    raise ValueError(f"unknown search surface: {search_surface}")

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .engine import EngineConfig as BaseEngineConfig
from .engine import NeuronGraphRAG as BaseNeuronGraphRAG
from .models import (
    FeedbackEvidence,
    FeedbackReceipt,
    NormalizedSiblingEdge,
    ReinforcedEdge,
)
from .retrieval import DenseEncoder


@dataclass(frozen=True, slots=True)
class EngineConfig(BaseEngineConfig):
    relation_feedback_evidence_quorum: int = 1
    confirmed_outcome_reinforcement: bool = False
    confirmation_decay_ratio: float | None = None
    soft_start_feedback_reinforcement: bool = False
    soft_start_feedback_ratio: float | None = None

    def __post_init__(self) -> None:
        BaseEngineConfig.__post_init__(self)
        if (
            isinstance(self.relation_feedback_evidence_quorum, bool)
            or not isinstance(self.relation_feedback_evidence_quorum, int)
            or self.relation_feedback_evidence_quorum < 1
        ):
            raise ValueError(
                "relation_feedback_evidence_quorum must be a positive integer"
            )
        if not isinstance(self.confirmed_outcome_reinforcement, bool):
            raise TypeError("confirmed_outcome_reinforcement must be a boolean")
        if not isinstance(self.soft_start_feedback_reinforcement, bool):
            raise TypeError("soft_start_feedback_reinforcement must be a boolean")
        if (
            self.confirmed_outcome_reinforcement
            and self.soft_start_feedback_reinforcement
        ):
            raise ValueError(
                "confirmed-only and soft-start feedback reinforcement are mutually exclusive"
            )
        outcome_candidate = (
            self.confirmed_outcome_reinforcement
            or self.soft_start_feedback_reinforcement
        )
        if outcome_candidate:
            if (
                isinstance(self.confirmation_decay_ratio, bool)
                or not isinstance(self.confirmation_decay_ratio, (int, float))
                or not math.isfinite(float(self.confirmation_decay_ratio))
                or not 0.0 < float(self.confirmation_decay_ratio) < 1.0
            ):
                raise ValueError(
                    "confirmation_decay_ratio must be explicitly set between 0 and 1"
                )
        elif self.confirmation_decay_ratio is not None:
            raise ValueError(
                "confirmation_decay_ratio requires confirmed-only or soft-start "
                "feedback reinforcement"
            )
        if self.soft_start_feedback_reinforcement:
            if self.relation_feedback_evidence_quorum != 1:
                raise ValueError(
                    "soft-start feedback reinforcement cannot be combined with a "
                    "hard evidence quorum"
                )
            if (
                isinstance(self.soft_start_feedback_ratio, bool)
                or not isinstance(self.soft_start_feedback_ratio, (int, float))
                or not math.isfinite(float(self.soft_start_feedback_ratio))
                or not 0.0 < float(self.soft_start_feedback_ratio) < 1.0
            ):
                raise ValueError(
                    "soft_start_feedback_ratio must be explicitly set between 0 and 1"
                )
        elif self.soft_start_feedback_ratio is not None:
            raise ValueError(
                "soft_start_feedback_ratio requires soft_start_feedback_reinforcement"
            )


class NeuronGraphRAG(BaseNeuronGraphRAG):
    config: EngineConfig

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        config: EngineConfig | None = None,
        dense_encoder: DenseEncoder | None = None,
    ) -> None:
        super().__init__(
            database,
            config=config or EngineConfig(),
            dense_encoder=dense_encoder,
        )

    def record_success(
        self,
        trace_id: str,
        used_node_ids: Iterable[str],
        *,
        now: datetime | float | None = None,
    ) -> FeedbackReceipt:
        ordered_node_ids = tuple(dict.fromkeys(used_node_ids))
        if not ordered_node_ids:
            raise ValueError("At least one used node is required")
        timestamp = self._timestamp(now)

        selected_paths: list[dict[str, object]] = []
        for node_id in ordered_node_ids:
            try:
                paths = self.store.retrieval_paths(trace_id, node_id)
            except KeyError as error:
                raise ValueError(
                    f"Successful node {node_id} was not retrieved by trace {trace_id}"
                ) from error
            if paths:
                selected_path = max(
                    paths,
                    key=lambda path: (
                        float(path["contribution"]),
                        str(path["seed_id"]),
                    ),
                )
                if selected_path["steps"]:
                    selected_paths.append(selected_path)

        feedback_id = uuid.uuid4().hex
        unique_edges: dict[tuple[str, str, str], float] = {}
        for path in selected_paths:
            contribution = min(1.0, max(0.1, float(path["contribution"])))
            for step in path["steps"]:
                key = (
                    str(step["source_id"]),
                    str(step["target_id"]),
                    str(step["edge_type"]),
                )
                unique_edges[key] = max(unique_edges.get(key, 0.0), contribution)

        updates = (
            (
                source_id,
                target_id,
                edge_type,
                self.config.feedback_learning_rate * contribution,
                self.config.maximum_edge_weight,
            )
            for (source_id, target_id, edge_type), contribution in sorted(
                unique_edges.items()
            )
        )
        channel = self.store.retrieval_channel(trace_id)
        normalization_sets: tuple[
            tuple[str, tuple[tuple[str, str, str], ...], float], ...
        ] = ()
        if (
            channel == "relation"
            and self.config.sibling_feedback_normalization > 0.0
            and unique_edges
        ):
            credited_keys = set(unique_edges)
            normalization_sets = tuple(
                (
                    source_id,
                    tuple(
                        (edge.source_id, edge.target_id, edge.edge_type)
                        for edge in self.store.outgoing_edges(source_id)
                        if (edge.source_id, edge.target_id, edge.edge_type)
                        not in credited_keys
                    ),
                    self.config.sibling_feedback_normalization,
                )
                for source_id in sorted({key[0] for key in credited_keys})
            )
        stored_reinforced, stored_normalized, stored_evidence = (
            self.store.apply_evidence_gated_success_feedback(
                feedback_id,
                trace_id,
                timestamp,
                ordered_node_ids,
                updates,
                normalization_sets,
                evidence_quorum=self.config.relation_feedback_evidence_quorum,
            )
        )
        reinforced = tuple(
            ReinforcedEdge(
                source_id,
                target_id,
                edge_type,
                old_weight,
                new_weight,
            )
            for source_id, target_id, edge_type, old_weight, new_weight in stored_reinforced
        )
        normalized = tuple(
            NormalizedSiblingEdge(
                source_id,
                target_id,
                edge_type,
                old_weight,
                new_weight,
            )
            for source_id, target_id, edge_type, old_weight, new_weight in stored_normalized
        )
        evidence = tuple(
            FeedbackEvidence(
                source_id,
                target_id,
                edge_type,
                count,
                quorum,
                activated,
            )
            for source_id, target_id, edge_type, count, quorum, activated in stored_evidence
        )
        return FeedbackReceipt(
            feedback_id,
            trace_id,
            ordered_node_ids,
            reinforced,
            channel,
            normalized,
            evidence,
        )

    def record_soft_start(
        self,
        trace_id: str,
        used_node_ids: Iterable[str],
        *,
        now: datetime | float | None = None,
    ) -> FeedbackReceipt:
        """Apply the one-time provisional part of a soft-start relation schedule."""
        if not self.config.soft_start_feedback_reinforcement:
            raise ValueError("soft-start feedback reinforcement is not enabled")
        ordered_node_ids = tuple(dict.fromkeys(used_node_ids))
        if not ordered_node_ids:
            raise ValueError("At least one used node is required")
        timestamp = self._timestamp(now)
        plan = self._relation_feedback_plan(
            trace_id, ordered_node_ids, require_candidate_marker=False
        )
        feedback_id = uuid.uuid4().hex
        stored_reinforced = self.store.apply_soft_start_feedback(
            feedback_id,
            trace_id,
            timestamp,
            ordered_node_ids,
            plan["updates"],
            soft_start_ratio=float(self.config.soft_start_feedback_ratio),
            decay_ratio=float(self.config.confirmation_decay_ratio),
        )
        return FeedbackReceipt(
            feedback_id,
            trace_id,
            ordered_node_ids,
            tuple(
                ReinforcedEdge(
                    source_id,
                    target_id,
                    edge_type,
                    old_weight,
                    new_weight,
                )
                for source_id, target_id, edge_type, old_weight, new_weight in stored_reinforced
            ),
            self.store.retrieval_channel(trace_id),
        )

    def confirmed_outcome_plan(
        self, trace_id: str, used_node_ids: Iterable[str]
    ) -> dict[str, Any]:
        """Build a relation-only credited plan for an opt-in confirmed outcome."""
        if not (
            self.config.confirmed_outcome_reinforcement
            or self.config.soft_start_feedback_reinforcement
        ):
            raise ValueError("confirmed outcome reinforcement is not enabled")
        return self._relation_feedback_plan(
            trace_id, used_node_ids, require_candidate_marker=True
        )

    def _relation_feedback_plan(
        self,
        trace_id: str,
        used_node_ids: Iterable[str],
        *,
        require_candidate_marker: bool,
    ) -> dict[str, Any]:
        ordered_node_ids = tuple(dict.fromkeys(used_node_ids))
        if not ordered_node_ids:
            raise ValueError("At least one used node is required")
        if self.store.retrieval_channel(trace_id) != "relation":
            return {"updates": (), "normalization_sets": (), "credited_paths": ()}
        eligible_node_ids = (
            tuple(
                node_id
                for node_id in ordered_node_ids
                if self.store.is_confirmed_candidate_use(trace_id, node_id)
            )
            if require_candidate_marker
            else ordered_node_ids
        )

        selected_paths: list[tuple[str, dict[str, object]]] = []
        for node_id in eligible_node_ids:
            try:
                paths = self.store.retrieval_paths(trace_id, node_id)
            except KeyError as error:
                raise ValueError(
                    f"Confirmed node {node_id} was not retrieved by trace {trace_id}"
                ) from error
            if not paths:
                continue
            selected_path = max(
                paths,
                key=lambda path: (
                    float(path["contribution"]),
                    str(path["seed_id"]),
                ),
            )
            if selected_path["steps"]:
                selected_paths.append((node_id, selected_path))

        unique_edges: dict[tuple[str, str, str], float] = {}
        credited_paths: list[dict[str, object]] = []
        for node_id, path in selected_paths:
            contribution = min(1.0, max(0.1, float(path["contribution"])))
            steps = tuple(
                {
                    "source_id": str(step["source_id"]),
                    "target_id": str(step["target_id"]),
                    "edge_type": str(step["edge_type"]),
                    "edge_weight": float(step["edge_weight"]),
                    "factuality": float(step["factuality"]),
                }
                for step in path["steps"]
            )
            credited_paths.append({"node_id": node_id, "steps": steps})
            for step in steps:
                key = (
                    str(step["source_id"]),
                    str(step["target_id"]),
                    str(step["edge_type"]),
                )
                unique_edges[key] = max(unique_edges.get(key, 0.0), contribution)

        updates = tuple(
            (
                source_id,
                target_id,
                edge_type,
                self.config.feedback_learning_rate * contribution,
                self.config.maximum_edge_weight,
            )
            for (source_id, target_id, edge_type), contribution in sorted(
                unique_edges.items()
            )
        )
        normalization_sets: tuple[
            tuple[str, tuple[tuple[str, str, str], ...], float], ...
        ] = ()
        if self.config.sibling_feedback_normalization > 0.0 and unique_edges:
            credited_keys = set(unique_edges)
            normalization_sets = tuple(
                (
                    source_id,
                    tuple(
                        (edge.source_id, edge.target_id, edge.edge_type)
                        for edge in self.store.outgoing_edges(source_id)
                        if (edge.source_id, edge.target_id, edge.edge_type)
                        not in credited_keys
                    ),
                    self.config.sibling_feedback_normalization,
                )
                for source_id in sorted({key[0] for key in credited_keys})
            )
        return {
            "updates": updates,
            "normalization_sets": normalization_sets,
            "credited_paths": tuple(credited_paths),
        }

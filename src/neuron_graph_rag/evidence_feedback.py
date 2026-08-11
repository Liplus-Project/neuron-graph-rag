from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .engine import EngineConfig as BaseEngineConfig
from .engine import NeuronGraphRAG as BaseNeuronGraphRAG
from .models import (
    FeedbackEvidence,
    FeedbackReceipt,
    NormalizedSiblingEdge,
    ReinforcedEdge,
)


@dataclass(frozen=True, slots=True)
class EngineConfig(BaseEngineConfig):
    relation_feedback_evidence_quorum: int = 1

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


class NeuronGraphRAG(BaseNeuronGraphRAG):
    config: EngineConfig

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
            self.store.apply_success_feedback(
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

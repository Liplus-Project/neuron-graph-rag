from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from datetime import datetime

from .engine import NeuronGraphRAG
from .models import (
    FeedbackContractError,
    FeedbackReceipt,
    NormalizedSiblingEdge,
    OutcomeReceipt,
    ReinforcedEdge,
    SourceUseEvent,
    SourceUseEventReceipt,
    SourceUseReceipt,
)


class FeedbackLedger:
    """Transport-neutral source-use and delayed-outcome domain API."""

    def __init__(self, engine: NeuronGraphRAG) -> None:
        self.engine = engine

    def record_source_use(
        self,
        trace_id: str,
        events: Iterable[SourceUseEvent],
        *,
        idempotency_key: str,
        now: datetime | float | None = None,
    ) -> SourceUseReceipt:
        ordered_events = tuple(events)
        if not ordered_events:
            raise ValueError("At least one source-use event is required")
        if any(event.stage not in {"selected", "validated", "used"} for event in ordered_events):
            raise ValueError("Unknown source-use stage")
        timestamp = self._timestamp(now)
        current = self.engine.store.source_use_stages(trace_id)
        simulated = dict(current)
        stage_order = {"retrieved": 0, "selected": 1, "validated": 2, "used": 3}
        newly_used_candidates: list[str] = []
        for event in ordered_events:
            prior = simulated.get(event.node_id, "retrieved")
            if stage_order[event.stage] == stage_order[prior] + 1:
                simulated[event.node_id] = event.stage
                if event.stage == "used" and event.node_id not in newly_used_candidates:
                    newly_used_candidates.append(event.node_id)

        updates, normalization_sets, channel = self._feedback_update_plan(
            trace_id, tuple(newly_used_candidates)
        )
        payload_json = json.dumps(
            {
                "trace_id": trace_id,
                "events": [
                    {"node_id": event.node_id, "stage": event.stage}
                    for event in ordered_events
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        stored = self.engine.store.record_source_use(
            idempotency_key=idempotency_key,
            payload_json=payload_json,
            receipt_id=uuid.uuid4().hex,
            feedback_id=uuid.uuid4().hex,
            trace_id=trace_id,
            created_at=timestamp,
            events=tuple((event.node_id, event.stage) for event in ordered_events),
            edge_updates=updates,
            normalization_sets=normalization_sets,
            channel=channel,
        )
        feedback_data = stored["feedback"]
        feedback = None
        if feedback_data is not None:
            feedback = FeedbackReceipt(
                str(feedback_data["feedback_id"]),
                str(feedback_data["trace_id"]),
                tuple(str(item) for item in feedback_data["used_node_ids"]),
                tuple(
                    ReinforcedEdge(
                        str(edge["source_id"]),
                        str(edge["target_id"]),
                        str(edge["edge_type"]),
                        float(edge["old_weight"]),
                        float(edge["new_weight"]),
                    )
                    for edge in feedback_data["reinforced_edges"]
                ),
                None if feedback_data["channel"] is None else str(feedback_data["channel"]),
                tuple(
                    NormalizedSiblingEdge(
                        str(edge["source_id"]),
                        str(edge["target_id"]),
                        str(edge["edge_type"]),
                        float(edge["old_weight"]),
                        float(edge["new_weight"]),
                    )
                    for edge in feedback_data["normalized_sibling_edges"]
                ),
            )
        return SourceUseReceipt(
            str(stored["receipt_id"]),
            str(stored["trace_id"]),
            tuple(
                SourceUseEventReceipt(
                    str(event["node_id"]), str(event["stage"]), bool(event["changed"])
                )
                for event in stored["events"]
            ),
            tuple(str(node_id) for node_id in stored["newly_used_node_ids"]),
            feedback,
        )

    def record_outcome(
        self,
        trace_id: str,
        node_ids: Iterable[str],
        outcome: str,
        summary: str,
        *,
        idempotency_key: str,
        external_ref: str | None = None,
        now: datetime | float | None = None,
    ) -> OutcomeReceipt:
        ordered_node_ids = tuple(node_ids)
        if not ordered_node_ids:
            raise ValueError("At least one source node is required")
        if len(set(ordered_node_ids)) != len(ordered_node_ids):
            raise ValueError("Outcome node IDs must be unique")
        if outcome not in {"confirmed", "corrected", "rolled_back", "superseded"}:
            raise ValueError("Unknown outcome")
        payload_json = json.dumps(
            {
                "trace_id": trace_id,
                "node_ids": ordered_node_ids,
                "outcome": outcome,
                "summary": summary,
                "external_ref": external_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        stored = self.engine.store.record_outcome(
            idempotency_key=idempotency_key,
            payload_json=payload_json,
            outcome_id=uuid.uuid4().hex,
            trace_id=trace_id,
            node_ids=ordered_node_ids,
            outcome=outcome,
            summary=summary,
            external_ref=external_ref,
            recorded_at=self._timestamp(now),
        )
        return OutcomeReceipt(
            str(stored["outcome_id"]),
            str(stored["trace_id"]),
            tuple(str(item) for item in stored["node_ids"]),
            str(stored["outcome"]),
            float(stored["recorded_at"]),
            bool(stored["reinforcement_applied"]),
        )

    def _feedback_update_plan(
        self, trace_id: str, used_node_ids: tuple[str, ...]
    ) -> tuple[
        tuple[tuple[str, str, str, float, float], ...],
        tuple[tuple[str, tuple[tuple[str, str, str], ...], float], ...],
        str | None,
    ]:
        selected_paths: list[dict[str, object]] = []
        for node_id in used_node_ids:
            try:
                paths = self.engine.store.retrieval_paths(trace_id, node_id)
            except KeyError as error:
                raise FeedbackContractError(
                    "node_not_in_trace", f"node {node_id} was not returned by this trace"
                ) from error
            if paths:
                selected_path = max(
                    paths,
                    key=lambda path: (float(path["contribution"]), str(path["seed_id"])),
                )
                if selected_path["steps"]:
                    selected_paths.append(selected_path)
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
        config = self.engine.config
        updates = tuple(
            (
                source_id,
                target_id,
                edge_type,
                config.feedback_learning_rate * contribution,
                config.maximum_edge_weight,
            )
            for (source_id, target_id, edge_type), contribution in sorted(unique_edges.items())
        )
        channel = self.engine.store.retrieval_channel(trace_id)
        normalization_sets: tuple[
            tuple[str, tuple[tuple[str, str, str], ...], float], ...
        ] = ()
        if channel == "relation" and config.sibling_feedback_normalization > 0.0 and unique_edges:
            credited_keys = set(unique_edges)
            normalization_sets = tuple(
                (
                    source_id,
                    tuple(
                        (edge.source_id, edge.target_id, edge.edge_type)
                        for edge in self.engine.store.outgoing_edges(source_id)
                        if (edge.source_id, edge.target_id, edge.edge_type) not in credited_keys
                    ),
                    config.sibling_feedback_normalization,
                )
                for source_id in sorted({key[0] for key in credited_keys})
            )
        return updates, normalization_sets, channel

    @staticmethod
    def _timestamp(value: datetime | float | None) -> float:
        if value is None:
            return time.time()
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from datetime import datetime

from .engine import NeuronGraphRAG
from .models import (
    ConfirmedEdge,
    CreditedPath,
    FeedbackEvidence,
    FeedbackReceipt,
    NormalizedSiblingEdge,
    OutcomeReceipt,
    PathStep,
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
        confirmed_only = bool(
            getattr(self.engine.config, "confirmed_outcome_reinforcement", False)
        )
        soft_start = bool(
            getattr(self.engine.config, "soft_start_feedback_reinforcement", False)
        )
        if soft_start:
            apply_feedback = lambda node_ids: self.engine.record_soft_start(
                trace_id, node_ids, now=timestamp
            )
        elif confirmed_only:
            apply_feedback = None
        else:
            apply_feedback = lambda node_ids: self.engine.record_success(
                trace_id, node_ids, now=timestamp
            )
        stored = self.engine.store.record_source_use(
            idempotency_key=idempotency_key,
            payload_json=payload_json,
            receipt_id=uuid.uuid4().hex,
            trace_id=trace_id,
            created_at=timestamp,
            events=tuple((event.node_id, event.stage) for event in ordered_events),
            apply_feedback=apply_feedback,
            confirmation_candidate=confirmed_only or soft_start,
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
                tuple(
                    FeedbackEvidence(
                        str(item["source_id"]),
                        str(item["target_id"]),
                        str(item["edge_type"]),
                        int(item["count"]),
                        int(item["quorum"]),
                        bool(item["activated"]),
                    )
                    for item in feedback_data.get("evidence", [])
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
        outcome_id = uuid.uuid4().hex
        recorded_at = self._timestamp(now)
        candidate_enabled = bool(
            getattr(self.engine.config, "confirmed_outcome_reinforcement", False)
            or getattr(self.engine.config, "soft_start_feedback_reinforcement", False)
        )
        if candidate_enabled and outcome == "confirmed":
            plan = self.engine.confirmed_outcome_plan(trace_id, ordered_node_ids)
            record_confirmed = (
                self.engine.store.record_soft_start_confirmed_outcome
                if getattr(
                    self.engine.config, "soft_start_feedback_reinforcement", False
                )
                else self.engine.store.record_confirmed_outcome
            )
            candidate_options = (
                {"soft_start_ratio": float(self.engine.config.soft_start_feedback_ratio)}
                if getattr(
                    self.engine.config, "soft_start_feedback_reinforcement", False
                )
                else {}
            )
            stored = record_confirmed(
                idempotency_key=idempotency_key,
                payload_json=payload_json,
                outcome_id=outcome_id,
                trace_id=trace_id,
                node_ids=ordered_node_ids,
                summary=summary,
                external_ref=external_ref,
                recorded_at=recorded_at,
                decay_ratio=float(self.engine.config.confirmation_decay_ratio),
                edge_updates=plan["updates"],
                normalization_sets=plan["normalization_sets"],
                credited_paths=plan["credited_paths"],
                **candidate_options,
            )
        else:
            stored = self.engine.store.record_outcome(
                idempotency_key=idempotency_key,
                payload_json=payload_json,
                outcome_id=outcome_id,
                trace_id=trace_id,
                node_ids=ordered_node_ids,
                outcome=outcome,
                summary=summary,
                external_ref=external_ref,
                recorded_at=recorded_at,
            )
        return OutcomeReceipt(
            str(stored["outcome_id"]),
            str(stored["trace_id"]),
            tuple(str(item) for item in stored["node_ids"]),
            str(stored["outcome"]),
            float(stored["recorded_at"]),
            bool(stored["reinforcement_applied"]),
            tuple(
                ConfirmedEdge(
                    str(item["source_id"]),
                    str(item["target_id"]),
                    str(item["edge_type"]),
                    int(item["confirmation_count"]),
                    float(item["multiplier"]),
                    float(item["actual_delta"]),
                    float(item["old_weight"]),
                    float(item["new_weight"]),
                )
                for item in stored.get("confirmations", [])
            ),
            tuple(
                CreditedPath(
                    str(path["node_id"]),
                    tuple(
                        PathStep(
                            str(step["source_id"]),
                            str(step["target_id"]),
                            str(step["edge_type"]),
                            float(step["edge_weight"]),
                            float(step["factuality"]),
                        )
                        for step in path["steps"]
                    ),
                )
                for path in stored.get("credited_paths", [])
            ),
            tuple(
                NormalizedSiblingEdge(
                    str(edge["source_id"]),
                    str(edge["target_id"]),
                    str(edge["edge_type"]),
                    float(edge["old_weight"]),
                    float(edge["new_weight"]),
                )
                for edge in stored.get("normalized_sibling_edges", [])
            ),
        )

    @staticmethod
    def _timestamp(value: datetime | float | None) -> float:
        if value is None:
            return time.time()
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

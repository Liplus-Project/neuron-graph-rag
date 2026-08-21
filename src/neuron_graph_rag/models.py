from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentNode:
    node_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class TypedEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    factuality: float = 1.0
    reinforced_count: int = 0


@dataclass(frozen=True, slots=True)
class PathStep:
    source_id: str
    target_id: str
    edge_type: str
    edge_weight: float
    factuality: float


@dataclass(frozen=True, slots=True)
class ActivationPath:
    seed_id: str
    contribution: float
    steps: tuple[PathStep, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchHit:
    node: DocumentNode
    sparse_score: float
    dense_score: float
    entry_score: float
    graph_activation: float
    final_score: float
    paths: tuple[ActivationPath, ...] = ()
    sparse_raw_score: float = 0.0
    dense_raw_score: float = 0.0
    normalized_graph_activation: float = 0.0
    entry_anchor_before_competition: float = 0.0
    entry_anchor_after_competition: float = 0.0
    entry_rank: int = 0
    graph_rank: int | None = None
    entry_fusion_component: float = 0.0
    graph_fusion_component: float = 0.0
    final_fusion_strategy: str = "linear"
    graph_normalization: str = "max"

    def explain(self) -> dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "scores": {
                "sparse": self.sparse_score,
                "sparse_raw": self.sparse_raw_score,
                "dense": self.dense_score,
                "dense_raw": self.dense_raw_score,
                "entry": self.entry_score,
                "entry_anchor_before_competition": (
                    self.entry_anchor_before_competition
                ),
                "entry_anchor_after_competition": (
                    self.entry_anchor_after_competition
                ),
                "graph_activation": self.graph_activation,
                "graph_activation_normalized": (
                    self.normalized_graph_activation
                ),
                "final": self.final_score,
            },
            "ranks": {
                "entry": self.entry_rank,
                "graph": self.graph_rank,
            },
            "fusion": {
                "strategy": self.final_fusion_strategy,
                "graph_normalization": self.graph_normalization,
                "entry_component": self.entry_fusion_component,
                "graph_component": self.graph_fusion_component,
                "final": self.final_score,
            },
            "paths": [
                {
                    "seed_id": path.seed_id,
                    "contribution": path.contribution,
                    "kind": "graph" if path.steps else "entry_zero_hop",
                    "steps": [
                        {
                            "source_id": step.source_id,
                            "target_id": step.target_id,
                            "edge_type": step.edge_type,
                            "edge_weight": step.edge_weight,
                            "factuality": step.factuality,
                        }
                        for step in path.steps
                    ],
                }
                for path in self.paths
            ],
        }


@dataclass(frozen=True, slots=True)
class SearchTrace:
    trace_id: str
    query: str
    created_at: float
    hits: tuple[SearchHit, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchChannelHit:
    channel: str
    node: DocumentNode
    rank: int
    channel_score: float
    sparse_score: float
    sparse_raw_score: float
    dense_score: float = 0.0
    dense_raw_score: float = 0.0
    entry_score: float = 0.0
    graph_activation: float = 0.0
    paths: tuple[ActivationPath, ...] = ()

    def explain(self) -> dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "channel": self.channel,
            "rank": self.rank,
            "scores": {
                "channel": self.channel_score,
                "bm25": self.sparse_score,
                "bm25_raw": self.sparse_raw_score,
                "dense": self.dense_score,
                "dense_raw": self.dense_raw_score,
                "entry": self.entry_score,
                "graph_activation": self.graph_activation,
            },
            "paths": [
                {
                    "seed_id": path.seed_id,
                    "contribution": path.contribution,
                    "kind": "graph",
                    "steps": [
                        {
                            "source_id": step.source_id,
                            "target_id": step.target_id,
                            "edge_type": step.edge_type,
                            "edge_weight": step.edge_weight,
                            "factuality": step.factuality,
                        }
                        for step in path.steps
                    ],
                }
                for path in self.paths
            ],
        }


@dataclass(frozen=True, slots=True)
class SearchChannelTrace:
    trace_id: str
    query: str
    created_at: float
    channel: str
    hits: tuple[SearchChannelHit, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchChannelsResult:
    query: str
    lexical: SearchChannelTrace
    relation: SearchChannelTrace
    agreement_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReinforcedEdge:
    source_id: str
    target_id: str
    edge_type: str
    old_weight: float
    new_weight: float


@dataclass(frozen=True, slots=True)
class NormalizedSiblingEdge:
    source_id: str
    target_id: str
    edge_type: str
    old_weight: float
    new_weight: float


@dataclass(frozen=True, slots=True)
class FeedbackEvidence:
    source_id: str
    target_id: str
    edge_type: str
    count: int
    quorum: int
    activated: bool


@dataclass(frozen=True, slots=True)
class FeedbackReceipt:
    feedback_id: str
    trace_id: str
    used_node_ids: tuple[str, ...]
    reinforced_edges: tuple[ReinforcedEdge, ...]
    channel: str | None = None
    normalized_sibling_edges: tuple[NormalizedSiblingEdge, ...] = ()
    evidence: tuple[FeedbackEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceUseEvent:
    node_id: str
    stage: str


@dataclass(frozen=True, slots=True)
class SourceUseEventReceipt:
    node_id: str
    stage: str
    changed: bool


@dataclass(frozen=True, slots=True)
class SourceUseReceipt:
    receipt_id: str
    trace_id: str
    events: tuple[SourceUseEventReceipt, ...]
    newly_used_node_ids: tuple[str, ...]
    feedback: FeedbackReceipt | None


@dataclass(frozen=True, slots=True)
class ConfirmedEdge:
    source_id: str
    target_id: str
    edge_type: str
    confirmation_count: int
    multiplier: float
    actual_delta: float
    old_weight: float
    new_weight: float


@dataclass(frozen=True, slots=True)
class CreditedPath:
    node_id: str
    steps: tuple[PathStep, ...]


@dataclass(frozen=True, slots=True)
class ContributionMutation:
    mutation_role: str
    source_id: str
    target_id: str
    edge_type: str
    actual_delta: float
    old_weight: float
    new_weight: float


@dataclass(frozen=True, slots=True)
class ReversedContribution:
    contribution_id: str
    contribution_kind: str
    source_record_id: str
    source_id: str
    target_id: str
    edge_type: str
    credited_delta: float
    mutations: tuple[ContributionMutation, ...]


@dataclass(frozen=True, slots=True)
class DormancyChange:
    source_id: str
    target_id: str
    edge_type: str
    old_dormant: bool
    new_dormant: bool


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    outcome_id: str
    trace_id: str
    node_ids: tuple[str, ...]
    outcome: str
    recorded_at: float
    reinforcement_applied: bool = False
    confirmations: tuple[ConfirmedEdge, ...] = ()
    credited_paths: tuple[CreditedPath, ...] = ()
    normalized_sibling_edges: tuple[NormalizedSiblingEdge, ...] = ()
    deactivation_applied: bool = False
    reversed_contributions: tuple[ReversedContribution, ...] = ()
    dormancy_changes: tuple[DormancyChange, ...] = ()
    reactivated_edges: tuple[DormancyChange, ...] = ()


class FeedbackContractError(ValueError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable

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
class ReinforcedEdge:
    source_id: str
    target_id: str
    edge_type: str
    old_weight: float
    new_weight: float


@dataclass(frozen=True, slots=True)
class FeedbackReceipt:
    feedback_id: str
    trace_id: str
    used_node_ids: tuple[str, ...]
    reinforced_edges: tuple[ReinforcedEdge, ...]

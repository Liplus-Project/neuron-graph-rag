from __future__ import annotations

import math
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import (
    DocumentNode,
    FeedbackReceipt,
    ReinforcedEdge,
    SearchHit,
    SearchTrace,
    TypedEdge,
)
from .dynamics import DynamicsSettings, propagate
from .retrieval import BM25Retriever, DenseEncoder, DenseRetriever, normalize_scores
from .storage import SQLiteStore


@dataclass(frozen=True, slots=True)
class EngineConfig:
    sparse_weight: float = 0.55
    dense_weight: float = 0.45
    entry_weight: float = 0.55
    graph_weight: float = 0.45
    seed_count: int = 3
    max_hops: int = 2
    hop_decay: float = 0.70
    activation_half_life_seconds: float = 3600.0
    feedback_learning_rate: float = 0.20
    maximum_edge_weight: float = 2.0
    maximum_activation: float = 10.0
    max_paths_per_node: int = 4
    max_propagation_expansions: int = 10_000
    activation_strategy: str = "current_positive_additive"
    activation_budget: float = 1.0
    inhibition_ratio: float = 0.0
    inhibition_top_k: int = 0
    query_transmission_floor: float = 0.4
    query_transmission_power: float = 1.0
    recurrent_steps: int = 3
    recurrent_decay: float = 0.5
    convergence_tolerance: float = 1e-9
    max_active_paths_per_node: int = 4
    use_dense_retrieval: bool = True
    use_graph_propagation: bool = True
    graph_normalization: str = "max"
    final_fusion_strategy: str = "linear"
    rrf_k: int = 60

    def __post_init__(self) -> None:
        for name in (
            "sparse_weight",
            "dense_weight",
            "entry_weight",
            "graph_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if self.sparse_weight + self.dense_weight <= 0.0:
            raise ValueError("At least one entry retrieval weight must be positive")
        if self.entry_weight + self.graph_weight <= 0.0:
            raise ValueError("At least one final ranking weight must be positive")
        if self.seed_count < 1 or self.max_hops < 0:
            raise ValueError("seed_count must be positive and max_hops non-negative")
        if not 0.0 <= self.hop_decay <= 1.0:
            raise ValueError("hop_decay must be between 0 and 1")
        if self.activation_half_life_seconds <= 0.0:
            raise ValueError("activation half-life must be positive")
        if self.feedback_learning_rate <= 0.0:
            raise ValueError("feedback learning rate must be positive")
        if self.activation_strategy not in {
            "current_positive_additive",
            "finite_activation_budget",
            "lateral_inhibition",
            "query_conditioned_transmission",
            "recurrent_competition",
            "local_neighbor_competition",
            "local_neighbor_query_competition",
            "local_neighbor_path_competition",
            "local_neighbor_query_path_competition",
            "anchored_local_competition",
            "anchored_local_query_competition",
        }:
            raise ValueError("Unknown activation_strategy")
        if self.activation_budget <= 0.0:
            raise ValueError("activation_budget must be positive")
        if not 0.0 <= self.inhibition_ratio < 1.0:
            raise ValueError("inhibition_ratio must be in [0, 1)")
        if self.inhibition_top_k < 0:
            raise ValueError("inhibition_top_k must be non-negative")
        if not 0.0 <= self.query_transmission_floor <= 1.0:
            raise ValueError("query_transmission_floor must be between 0 and 1")
        if self.query_transmission_power <= 0.0:
            raise ValueError("query_transmission_power must be positive")
        if self.recurrent_steps < 1:
            raise ValueError("recurrent_steps must be positive")
        if not 0.0 <= self.recurrent_decay <= 1.0:
            raise ValueError("recurrent_decay must be between 0 and 1")
        if self.convergence_tolerance < 0.0:
            raise ValueError("convergence_tolerance must be non-negative")
        if self.max_active_paths_per_node < 1:
            raise ValueError("max_active_paths_per_node must be positive")
        if not self.use_dense_retrieval and self.dense_weight != 0.0:
            raise ValueError("dense_weight must be zero when dense retrieval is disabled")
        if not self.use_graph_propagation and self.graph_weight != 0.0:
            raise ValueError("graph_weight must be zero when graph propagation is disabled")
        if self.graph_normalization not in {"max", "none", "l1_mass"}:
            raise ValueError("Unknown graph_normalization")
        if self.final_fusion_strategy not in {"linear", "rrf"}:
            raise ValueError("Unknown final_fusion_strategy")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")


class NeuronGraphRAG:
    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        config: EngineConfig | None = None,
        dense_encoder: DenseEncoder | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.store = SQLiteStore(database)
        self.sparse_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(dense_encoder)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> NeuronGraphRAG:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_document(
        self,
        node_id: str,
        text: str,
        *,
        metadata: dict[str, object] | None = None,
        confidence: float = 1.0,
    ) -> DocumentNode:
        if not node_id or not text.strip():
            raise ValueError("node_id and text are required")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        node = DocumentNode(node_id, text, dict(metadata or {}), confidence)
        self.store.upsert_node(node)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        factuality: float = 1.0,
    ) -> TypedEdge:
        if not edge_type:
            raise ValueError("edge_type is required")
        if weight < 0.0:
            raise ValueError("weight must not be negative")
        if not 0.0 <= factuality <= 1.0:
            raise ValueError("factuality must be between 0 and 1")
        self.store.get_node(source_id)
        self.store.get_node(target_id)
        edge = TypedEdge(source_id, target_id, edge_type, weight, factuality)
        self.store.upsert_edge(edge)
        return edge

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        now: datetime | float | None = None,
    ) -> SearchTrace:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        nodes = self.store.list_nodes()
        if not nodes:
            raise ValueError("Cannot search an empty corpus")
        timestamp = self._timestamp(now)

        sparse_raw = self.sparse_retriever.score(query, nodes)
        dense_raw = (
            self.dense_retriever.score(query, nodes)
            if self.config.use_dense_retrieval
            else {node.node_id: 0.0 for node in nodes}
        )
        sparse = normalize_scores(sparse_raw)
        dense = normalize_scores(dense_raw)
        entry = {
            node.node_id: self._weighted_average(
                sparse[node.node_id],
                dense[node.node_id],
                self.config.sparse_weight,
                self.config.dense_weight,
            )
            for node in nodes
        }
        seed_ids = [
            node_id
            for node_id, score in sorted(
                entry.items(), key=lambda item: (-item[1], item[0])
            )[: self.config.seed_count]
            if score > 0.0
        ]
        if self.config.use_graph_propagation:
            propagation = propagate(
                query=query,
                seed_ids=seed_ids,
                entry=entry,
                nodes={node.node_id: node for node in nodes},
                outgoing_edges=self.store.outgoing_edges,
                settings=DynamicsSettings(
                    strategy=self.config.activation_strategy,
                    max_hops=self.config.max_hops,
                    hop_decay=self.config.hop_decay,
                    max_expansions=self.config.max_propagation_expansions,
                    activation_budget=self.config.activation_budget,
                    inhibition_ratio=self.config.inhibition_ratio,
                    inhibition_top_k=self.config.inhibition_top_k,
                    query_transmission_floor=self.config.query_transmission_floor,
                    query_transmission_power=self.config.query_transmission_power,
                    recurrent_steps=self.config.recurrent_steps,
                    recurrent_decay=self.config.recurrent_decay,
                    convergence_tolerance=self.config.convergence_tolerance,
                    max_active_paths_per_node=self.config.max_active_paths_per_node,
                ),
            )
            graph_activation, paths = propagation.activation, propagation.paths
            propagation_diagnostics = propagation.diagnostics.as_dict()
        else:
            graph_activation, paths = {}, {}
            propagation_diagnostics = {
                "strategy": "graph_disabled",
                "steps": 0,
                "expansions": 0,
                "activation_total": 0.0,
                "converged": True,
                "stop_reason": "graph_disabled",
                "active_path_count": 0,
                "competition_sets": [],
            }
        graph_values = {
            node.node_id: graph_activation.get(node.node_id, 0.0) for node in nodes
        }
        normalized_activation = self._normalize_graph_activation(
            graph_values, self.config.graph_normalization
        )
        self._record_activation(graph_activation, timestamp)

        entry_ranks = self._rank_positive_or_all(entry, positive_only=False)
        graph_ranks = self._rank_positive_or_all(
            graph_values, positive_only=True
        )
        positive_graph_count = len(graph_ranks)

        hits = []
        for node in nodes:
            entry_component, graph_component = self._fusion_components(
                entry=entry[node.node_id],
                normalized_graph=normalized_activation[node.node_id],
                entry_rank=entry_ranks[node.node_id],
                graph_rank=graph_ranks.get(node.node_id),
                positive_graph_count=positive_graph_count,
            )
            hits.append(SearchHit(
                node=node,
                sparse_score=sparse[node.node_id],
                dense_score=dense[node.node_id],
                entry_score=entry[node.node_id],
                graph_activation=graph_activation.get(node.node_id, 0.0),
                final_score=entry_component + graph_component,
                paths=tuple(
                    sorted(
                        paths.get(node.node_id, ()),
                        key=lambda path: (-path.contribution, path.seed_id),
                    )[: self.config.max_paths_per_node]
                ),
                sparse_raw_score=sparse_raw[node.node_id],
                dense_raw_score=dense_raw[node.node_id],
                normalized_graph_activation=normalized_activation[node.node_id],
                entry_anchor_before_competition=entry[node.node_id],
                entry_anchor_after_competition=entry[node.node_id],
                entry_rank=entry_ranks[node.node_id],
                graph_rank=graph_ranks.get(node.node_id),
                entry_fusion_component=entry_component,
                graph_fusion_component=graph_component,
                final_fusion_strategy=self.config.final_fusion_strategy,
                graph_normalization=self.config.graph_normalization,
            ))
        hits.sort(key=lambda hit: (-hit.final_score, hit.node.node_id))
        selected_hits = tuple(hits[:limit])
        trace_id = uuid.uuid4().hex
        self.store.create_retrieval(
            trace_id,
            query,
            timestamp,
            (
                {
                    "node_id": hit.node.node_id,
                    "rank": rank,
                    "sparse_score": hit.sparse_score,
                    "dense_score": hit.dense_score,
                    "entry_score": hit.entry_score,
                    "graph_activation": hit.graph_activation,
                    "final_score": hit.final_score,
                    "paths": hit.explain()["paths"],
                }
                for rank, hit in enumerate(selected_hits, start=1)
            ),
        )
        propagation_diagnostics.update(
            {
                "use_dense_retrieval": self.config.use_dense_retrieval,
                "use_graph_propagation": self.config.use_graph_propagation,
                "entry_anchor_invariant": all(
                    hit.entry_anchor_before_competition
                    == hit.entry_anchor_after_competition
                    for hit in hits
                ),
                "graph_signal_excludes_zero_hop": all(
                    path.steps
                    for node_paths in paths.values()
                    for path in node_paths
                ),
                "graph_normalization": self.config.graph_normalization,
                "final_fusion_strategy": self.config.final_fusion_strategy,
                "rrf_k": self.config.rrf_k,
                "positive_graph_node_count": positive_graph_count,
                "final_order_recomputable": all(
                    hit.final_score
                    == hit.entry_fusion_component + hit.graph_fusion_component
                    for hit in hits
                ),
            }
        )
        return SearchTrace(
            trace_id,
            query,
            timestamp,
            selected_hits,
            propagation_diagnostics,
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
        reinforced = [
            ReinforcedEdge(
                source_id,
                target_id,
                edge_type,
                old_weight,
                new_weight,
            )
            for source_id, target_id, edge_type, old_weight, new_weight in (
                self.store.apply_success_feedback(
                    feedback_id,
                    trace_id,
                    timestamp,
                    ordered_node_ids,
                    updates,
                )
            )
        ]
        return FeedbackReceipt(
            feedback_id,
            trace_id,
            ordered_node_ids,
            tuple(reinforced),
        )

    def activation(
        self, node_id: str, *, now: datetime | float | None = None
    ) -> float:
        self.store.get_node(node_id)
        state = self.store.activation(node_id)
        if state is None:
            return 0.0
        value, updated_at = state
        return self._decayed(value, updated_at, self._timestamp(now))

    def _record_activation(
        self, activation: dict[str, float], timestamp: float
    ) -> None:
        with self.store.transaction():
            for node_id, current in activation.items():
                state = self.store.activation(node_id)
                previous = (
                    self._decayed(state[0], state[1], timestamp) if state else 0.0
                )
                self.store.set_activation(
                    node_id,
                    min(self.config.maximum_activation, previous + current),
                    timestamp,
                )

    def _decayed(self, value: float, updated_at: float, now: float) -> float:
        elapsed = max(0.0, now - updated_at)
        return value * math.pow(
            0.5, elapsed / self.config.activation_half_life_seconds
        )

    @staticmethod
    def _weighted_average(
        left: float, right: float, left_weight: float, right_weight: float
    ) -> float:
        total_weight = left_weight + right_weight
        return (left * left_weight + right * right_weight) / total_weight

    @staticmethod
    def _normalize_graph_activation(
        scores: dict[str, float], strategy: str
    ) -> dict[str, float]:
        positive = {key: max(0.0, value) for key, value in scores.items()}
        if strategy == "none":
            return positive
        if strategy == "max":
            return normalize_scores(positive)
        if strategy == "l1_mass":
            total = sum(positive.values())
            if total <= 0.0:
                return {key: 0.0 for key in positive}
            return {key: value / total for key, value in positive.items()}
        raise ValueError(f"Unknown graph normalization: {strategy}")

    @staticmethod
    def _rank_positive_or_all(
        scores: dict[str, float], *, positive_only: bool
    ) -> dict[str, int]:
        ordered = sorted(
            (
                (node_id, score)
                for node_id, score in scores.items()
                if not positive_only or score > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            node_id: rank
            for rank, (node_id, _) in enumerate(ordered, start=1)
        }

    def _fusion_components(
        self,
        *,
        entry: float,
        normalized_graph: float,
        entry_rank: int,
        graph_rank: int | None,
        positive_graph_count: int,
    ) -> tuple[float, float]:
        if self.config.final_fusion_strategy == "linear":
            total_weight = self.config.entry_weight + self.config.graph_weight
            return (
                entry * self.config.entry_weight / total_weight,
                normalized_graph * self.config.graph_weight / total_weight,
            )
        entry_component = self.config.entry_weight / (
            self.config.rrf_k + entry_rank
        )
        graph_component = 0.0
        if graph_rank is not None:
            graph_component = self.config.graph_weight * (
                1.0 / (self.config.rrf_k + graph_rank)
                - 1.0
                / (
                    self.config.rrf_k
                    + positive_graph_count
                    + 1
                )
            )
        return entry_component, graph_component

    @staticmethod
    def _timestamp(value: datetime | float | None) -> float:
        if value is None:
            return time.time()
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

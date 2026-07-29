from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import (
    ActivationPath,
    DocumentNode,
    FeedbackReceipt,
    PathStep,
    ReinforcedEdge,
    SearchHit,
    SearchTrace,
    TypedEdge,
)
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
        dense_raw = self.dense_retriever.score(query, nodes)
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
        graph_activation, paths = self._propagate(seed_ids, entry)
        normalized_activation = normalize_scores(
            {node.node_id: graph_activation.get(node.node_id, 0.0) for node in nodes}
        )
        self._record_activation(graph_activation, timestamp)

        hits = [
            SearchHit(
                node=node,
                sparse_score=sparse[node.node_id],
                dense_score=dense[node.node_id],
                entry_score=entry[node.node_id],
                graph_activation=graph_activation.get(node.node_id, 0.0),
                final_score=self._weighted_average(
                    entry[node.node_id],
                    normalized_activation[node.node_id],
                    self.config.entry_weight,
                    self.config.graph_weight,
                ),
                paths=tuple(
                    sorted(
                        paths.get(node.node_id, ()),
                        key=lambda path: (-path.contribution, path.seed_id),
                    )[: self.config.max_paths_per_node]
                ),
            )
            for node in nodes
        ]
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
        return SearchTrace(trace_id, query, timestamp, selected_hits)

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

    def _propagate(
        self, seed_ids: list[str], entry: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, list[ActivationPath]]]:
        activation: dict[str, float] = defaultdict(float)
        paths: dict[str, list[ActivationPath]] = defaultdict(list)
        queue: deque[
            tuple[str, str, float, tuple[PathStep, ...], frozenset[str], int]
        ] = deque()
        for seed_id in seed_ids:
            contribution = entry[seed_id]
            activation[seed_id] += contribution
            path = ActivationPath(seed_id, contribution)
            paths[seed_id].append(path)
            queue.append(
                (seed_id, seed_id, contribution, (), frozenset({seed_id}), 0)
            )

        expansions = 0
        while queue and expansions < self.config.max_propagation_expansions:
            seed_id, current_id, contribution, steps, visited, depth = queue.popleft()
            if depth >= self.config.max_hops:
                continue
            for edge in self.store.outgoing_edges(current_id):
                if edge.target_id in visited:
                    continue
                next_contribution = (
                    contribution
                    * edge.weight
                    * edge.factuality
                    * self.config.hop_decay
                )
                if next_contribution <= 0.0:
                    continue
                step = PathStep(
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.factuality,
                )
                next_steps = steps + (step,)
                activation[edge.target_id] += next_contribution
                paths[edge.target_id].append(
                    ActivationPath(seed_id, next_contribution, next_steps)
                )
                queue.append(
                    (
                        seed_id,
                        edge.target_id,
                        next_contribution,
                        next_steps,
                        visited | {edge.target_id},
                        depth + 1,
                    )
                )
                expansions += 1
                if expansions >= self.config.max_propagation_expansions:
                    break
        return dict(activation), dict(paths)

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
    def _timestamp(value: datetime | float | None) -> float:
        if value is None:
            return time.time()
        if isinstance(value, datetime):
            return value.timestamp()
        return float(value)

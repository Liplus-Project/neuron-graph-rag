from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import DocumentNode, TypedEdge


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0)
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL CHECK(weight >= 0.0),
                factuality REAL NOT NULL CHECK(factuality BETWEEN 0.0 AND 1.0),
                reinforced_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (source_id, target_id, edge_type)
            );

            CREATE TABLE IF NOT EXISTS activation_state (
                node_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
                value REAL NOT NULL CHECK(value >= 0.0),
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrievals (
                trace_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrieval_results (
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                rank INTEGER NOT NULL,
                sparse_score REAL NOT NULL,
                dense_score REAL NOT NULL,
                entry_score REAL NOT NULL,
                graph_activation REAL NOT NULL,
                final_score REAL NOT NULL,
                paths_json TEXT NOT NULL,
                PRIMARY KEY (trace_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS retrieval_channels (
                trace_id TEXT PRIMARY KEY REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                channel TEXT NOT NULL CHECK(channel IN ('lexical', 'relation'))
            );

            CREATE TABLE IF NOT EXISTS success_feedback (
                feedback_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS success_nodes (
                feedback_id TEXT NOT NULL REFERENCES success_feedback(feedback_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                PRIMARY KEY (feedback_id, node_id)
            );
            """
        )
        self.connection.commit()

    def upsert_node(self, node: DocumentNode) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO nodes(node_id, text, metadata_json, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    text = excluded.text,
                    metadata_json = excluded.metadata_json,
                    confidence = excluded.confidence
                """,
                (
                    node.node_id,
                    node.text,
                    json.dumps(node.metadata, sort_keys=True, ensure_ascii=False),
                    node.confidence,
                ),
            )

    def upsert_edge(self, edge: TypedEdge) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO edges(
                    source_id, target_id, edge_type, weight, factuality, reinforced_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                    weight = excluded.weight,
                    factuality = excluded.factuality
                """,
                (
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.factuality,
                    edge.reinforced_count,
                ),
            )

    def get_node(self, node_id: str) -> DocumentNode:
        row = self.connection.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown node: {node_id}")
        return self._node_from_row(row)

    def list_nodes(self) -> list[DocumentNode]:
        rows = self.connection.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
        return [self._node_from_row(row) for row in rows]

    def list_edges(self) -> list[TypedEdge]:
        rows = self.connection.execute(
            "SELECT * FROM edges ORDER BY source_id, target_id, edge_type"
        ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def outgoing_edges(self, node_id: str) -> list[TypedEdge]:
        rows = self.connection.execute(
            """
            SELECT * FROM edges
            WHERE source_id = ?
            ORDER BY target_id, edge_type
            """,
            (node_id,),
        ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def edge(self, source_id: str, target_id: str, edge_type: str) -> TypedEdge:
        row = self.connection.execute(
            """
            SELECT * FROM edges
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            (source_id, target_id, edge_type),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown edge: {source_id} -> {target_id} ({edge_type})")
        return self._edge_from_row(row)

    def activation(self, node_id: str) -> tuple[float, float] | None:
        row = self.connection.execute(
            "SELECT value, updated_at FROM activation_state WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return float(row["value"]), float(row["updated_at"])

    def set_activation(self, node_id: str, value: float, updated_at: float) -> None:
        self.connection.execute(
            """
            INSERT INTO activation_state(node_id, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (node_id, value, updated_at),
        )

    def create_retrieval(
        self,
        trace_id: str,
        query: str,
        created_at: float,
        result_rows: Iterable[dict[str, Any]],
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO retrievals(trace_id, query, created_at) VALUES (?, ?, ?)",
                (trace_id, query, created_at),
            )
            for row in result_rows:
                connection.execute(
                    """
                    INSERT INTO retrieval_results(
                        trace_id, node_id, rank, sparse_score, dense_score,
                        entry_score, graph_activation, final_score, paths_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        row["node_id"],
                        row["rank"],
                        row["sparse_score"],
                        row["dense_score"],
                        row["entry_score"],
                        row["graph_activation"],
                        row["final_score"],
                        json.dumps(row["paths"], sort_keys=True),
                    ),
                )

    def create_channel_retrieval(
        self,
        trace_id: str,
        query: str,
        created_at: float,
        channel: str,
        result_rows: Iterable[dict[str, Any]],
    ) -> None:
        if channel not in {"lexical", "relation"}:
            raise ValueError(f"Unknown retrieval channel: {channel}")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO retrievals(trace_id, query, created_at) VALUES (?, ?, ?)",
                (trace_id, query, created_at),
            )
            connection.execute(
                "INSERT INTO retrieval_channels(trace_id, channel) VALUES (?, ?)",
                (trace_id, channel),
            )
            for row in result_rows:
                connection.execute(
                    """
                    INSERT INTO retrieval_results(
                        trace_id, node_id, rank, sparse_score, dense_score,
                        entry_score, graph_activation, final_score, paths_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        row["node_id"],
                        row["rank"],
                        row["sparse_score"],
                        row["dense_score"],
                        row["entry_score"],
                        row["graph_activation"],
                        row["channel_score"],
                        json.dumps(row["paths"], sort_keys=True),
                    ),
                )

    def retrieval_channel(self, trace_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT channel FROM retrieval_channels WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        return None if row is None else str(row["channel"])

    def retrieval_paths(self, trace_id: str, node_id: str) -> list[dict[str, Any]]:
        row = self.connection.execute(
            """
            SELECT paths_json FROM retrieval_results
            WHERE trace_id = ? AND node_id = ?
            """,
            (trace_id, node_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Node {node_id} was not returned by trace {trace_id}")
        return list(json.loads(row["paths_json"]))

    def apply_success_feedback(
        self,
        feedback_id: str,
        trace_id: str,
        created_at: float,
        used_node_ids: Iterable[str],
        edge_updates: Iterable[tuple[str, str, str, float, float]],
        normalization_sets: Iterable[
            tuple[str, tuple[tuple[str, str, str], ...], float]
        ] = (),
    ) -> tuple[
        list[tuple[str, str, str, float, float]],
        list[tuple[str, str, str, float, float]],
    ]:
        reinforced: list[tuple[str, str, str, float, float]] = []
        normalized: list[tuple[str, str, str, float, float]] = []
        with self.transaction() as connection:
            trace = connection.execute(
                "SELECT 1 FROM retrievals WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                raise KeyError(f"Unknown trace: {trace_id}")
            connection.execute(
                """
                INSERT INTO success_feedback(feedback_id, trace_id, created_at)
                VALUES (?, ?, ?)
                """,
                (feedback_id, trace_id, created_at),
            )
            for node_id in used_node_ids:
                exists = connection.execute(
                    """
                    SELECT 1 FROM retrieval_results
                    WHERE trace_id = ? AND node_id = ?
                    """,
                    (trace_id, node_id),
                ).fetchone()
                if exists is None:
                    raise ValueError(
                        f"Successful node {node_id} was not retrieved by trace {trace_id}"
                    )
                connection.execute(
                    """
                    INSERT INTO success_nodes(feedback_id, node_id) VALUES (?, ?)
                    """,
                    (feedback_id, node_id),
                )
            reinforced_increase_by_source: dict[str, float] = {}
            for source_id, target_id, edge_type, increment, maximum in edge_updates:
                row = connection.execute(
                    """
                    SELECT weight FROM edges
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if row is None:
                    raise KeyError(
                        f"Unknown edge: {source_id} -> {target_id} ({edge_type})"
                    )
                old_weight = float(row["weight"])
                new_weight = max(
                    old_weight,
                    min(maximum, old_weight + increment),
                )
                connection.execute(
                    """
                    UPDATE edges
                    SET weight = ?, reinforced_count = reinforced_count + 1
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (new_weight, source_id, target_id, edge_type),
                )
                reinforced.append(
                    (source_id, target_id, edge_type, old_weight, new_weight)
                )
                reinforced_increase_by_source[source_id] = (
                    reinforced_increase_by_source.get(source_id, 0.0)
                    + (new_weight - old_weight)
                )
            for source_id, sibling_keys, ratio in normalization_sets:
                total_increase = reinforced_increase_by_source.get(source_id, 0.0)
                if total_increase <= 0.0 or not sibling_keys:
                    continue
                reduction = total_increase * ratio / len(sibling_keys)
                for sibling_source, target_id, edge_type in sibling_keys:
                    row = connection.execute(
                        """
                        SELECT weight FROM edges
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (sibling_source, target_id, edge_type),
                    ).fetchone()
                    if row is None:
                        raise KeyError(
                            f"Unknown edge: {sibling_source} -> {target_id} ({edge_type})"
                        )
                    old_weight = float(row["weight"])
                    new_weight = max(0.0, old_weight - reduction)
                    if new_weight < old_weight:
                        connection.execute(
                            """
                            UPDATE edges SET weight = ?
                            WHERE source_id = ? AND target_id = ? AND edge_type = ?
                            """,
                            (new_weight, sibling_source, target_id, edge_type),
                        )
                        normalized.append(
                            (sibling_source, target_id, edge_type, old_weight, new_weight)
                        )
        return reinforced, normalized

    def count_retrievals(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM retrievals").fetchone()
        return int(row["count"])

    def count_feedback(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM success_feedback"
        ).fetchone()
        return int(row["count"])

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> DocumentNode:
        return DocumentNode(
            node_id=str(row["node_id"]),
            text=str(row["text"]),
            metadata=dict(json.loads(row["metadata_json"])),
            confidence=float(row["confidence"]),
        )

    @staticmethod
    def _edge_from_row(row: sqlite3.Row) -> TypedEdge:
        return TypedEdge(
            source_id=str(row["source_id"]),
            target_id=str(row["target_id"]),
            edge_type=str(row["edge_type"]),
            weight=float(row["weight"]),
            factuality=float(row["factuality"]),
            reinforced_count=int(row["reinforced_count"]),
        )

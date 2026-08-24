from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import DocumentNode, FeedbackContractError, FeedbackReceipt, TypedEdge

FILE_BUSY_TIMEOUT_MILLISECONDS = 5_000

JUDGMENT_RELATION_TYPE_SEEDS = (
    (
        "depends_on",
        "この結論は target が真であり続けることを前提にする。",
    ),
    (
        "refines",
        "target を置き換えずに、その範囲を狭めるか運用可能にする。",
    ),
    (
        "supersedes",
        "この結論が target を現在の状態として置き換える。",
    ),
    (
        "conflicts_with",
        "二つの現在の結論は同時に適用できない。",
    ),
    (
        "informs",
        "target に関連する判断材料を与えるが、target を制約しない。",
    ),
)
JUDGMENT_RELATION_TYPE_NAMESPACE = "ngr.decision_structure"
JUDGMENT_RELATION_TYPE_SEED_PROVENANCE = json.dumps(
    {"document": "docs/Decision-Structure.md", "section": "Edge vocabulary"},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
JUDGMENT_RELATION_TYPE_SEED_TIMESTAMP = "2026-08-24T00:00:00.000000Z"


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        file_backed = self.path != ":memory:"
        self.connection = sqlite3.connect(
            self.path,
            timeout=FILE_BUSY_TIMEOUT_MILLISECONDS / 1_000,
        )
        self.connection.row_factory = sqlite3.Row
        if file_backed:
            journal_mode = self.connection.execute(
                "PRAGMA journal_mode = WAL"
            ).fetchone()[0]
            if str(journal_mode).lower() != "wal":
                self.connection.close()
                raise RuntimeError("file-backed SQLite database did not enable WAL mode")
            self.connection.execute(
                f"PRAGMA busy_timeout = {FILE_BUSY_TIMEOUT_MILLISECONDS}"
            )
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._transaction_depth = 0
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        outermost = self._transaction_depth == 0
        self._transaction_depth += 1
        try:
            yield self.connection
            if outermost:
                self.connection.commit()
        except Exception:
            if outermost:
                self.connection.rollback()
            raise
        finally:
            self._transaction_depth -= 1

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

            CREATE TABLE IF NOT EXISTS relation_feedback_evidence (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                feedback_id TEXT NOT NULL REFERENCES success_feedback(feedback_id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type, trace_id),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS source_use_state (
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                stage TEXT NOT NULL CHECK(stage IN ('selected', 'validated', 'used')),
                updated_at REAL NOT NULL,
                PRIMARY KEY (trace_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS confirmed_source_uses (
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                PRIMARY KEY (trace_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS feedback_requests (
                idempotency_key TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                result_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS delayed_outcomes (
                outcome_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                outcome TEXT NOT NULL CHECK(outcome IN ('confirmed', 'corrected', 'rolled_back', 'superseded')),
                summary TEXT NOT NULL,
                external_ref TEXT,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS delayed_outcome_nodes (
                outcome_id TEXT NOT NULL REFERENCES delayed_outcomes(outcome_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                PRIMARY KEY (outcome_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS confirmed_edge_state (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                confirmation_count INTEGER NOT NULL CHECK(confirmation_count >= 1),
                base_increment REAL NOT NULL CHECK(base_increment > 0.0),
                initial_weight REAL NOT NULL CHECK(initial_weight >= 0.0),
                decay_ratio REAL NOT NULL CHECK(decay_ratio > 0.0 AND decay_ratio < 1.0),
                geometric_maximum REAL NOT NULL CHECK(geometric_maximum >= initial_weight),
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS confirmed_relation_feedback (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                outcome_id TEXT NOT NULL REFERENCES delayed_outcomes(outcome_id) ON DELETE CASCADE,
                confirmation_count INTEGER NOT NULL CHECK(confirmation_count >= 1),
                multiplier REAL NOT NULL CHECK(multiplier > 0.0),
                actual_delta REAL NOT NULL CHECK(actual_delta >= 0.0),
                old_weight REAL NOT NULL CHECK(old_weight >= 0.0),
                new_weight REAL NOT NULL CHECK(new_weight >= old_weight),
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type, trace_id),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS soft_start_edge_state (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                confirmation_count INTEGER NOT NULL CHECK(confirmation_count >= 0),
                base_increment REAL NOT NULL CHECK(base_increment > 0.0),
                initial_weight REAL NOT NULL CHECK(initial_weight >= 0.0),
                soft_start_ratio REAL NOT NULL
                    CHECK(soft_start_ratio > 0.0 AND soft_start_ratio < 1.0),
                decay_ratio REAL NOT NULL CHECK(decay_ratio > 0.0 AND decay_ratio < 1.0),
                geometric_maximum REAL NOT NULL CHECK(geometric_maximum >= initial_weight),
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS soft_start_relation_feedback (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                feedback_id TEXT NOT NULL
                    REFERENCES success_feedback(feedback_id) ON DELETE CASCADE,
                actual_delta REAL NOT NULL CHECK(actual_delta >= 0.0),
                old_weight REAL NOT NULL CHECK(old_weight >= 0.0),
                new_weight REAL NOT NULL CHECK(new_weight >= old_weight),
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type, trace_id),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback_contributions (
                contribution_id TEXT PRIMARY KEY,
                contribution_kind TEXT NOT NULL
                    CHECK(contribution_kind IN ('soft_start_provisional', 'soft_start_confirmation')),
                source_record_id TEXT NOT NULL,
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                baseline_weight REAL NOT NULL CHECK(baseline_weight >= 0.0),
                credited_delta REAL NOT NULL CHECK(credited_delta > 0.0),
                reinforced_count_delta INTEGER NOT NULL CHECK(reinforced_count_delta >= 0),
                active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
                reversed_by_outcome_id TEXT REFERENCES delayed_outcomes(outcome_id),
                created_at REAL NOT NULL,
                UNIQUE(contribution_kind, source_record_id, source_id, target_id, edge_type),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback_contribution_mutations (
                contribution_id TEXT NOT NULL
                    REFERENCES feedback_contributions(contribution_id) ON DELETE CASCADE,
                mutation_role TEXT NOT NULL CHECK(mutation_role IN ('credited', 'sibling')),
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                actual_delta REAL NOT NULL CHECK(actual_delta != 0.0),
                PRIMARY KEY (
                    contribution_id, mutation_role, source_id, target_id, edge_type
                ),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback_edge_journal_state (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                baseline_weight REAL NOT NULL CHECK(baseline_weight >= 0.0),
                baseline_reinforced_count INTEGER NOT NULL
                    CHECK(baseline_reinforced_count >= 0),
                maximum_weight REAL NOT NULL CHECK(maximum_weight >= baseline_weight),
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relation_edge_dormancy (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                dormant INTEGER NOT NULL CHECK(dormant IN (0, 1)),
                outcome_id TEXT NOT NULL REFERENCES delayed_outcomes(outcome_id),
                trace_id TEXT NOT NULL REFERENCES retrievals(trace_id) ON DELETE CASCADE,
                updated_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id, target_id, edge_type)
                    REFERENCES edges(source_id, target_id, edge_type) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS judgments (
                judgment_id TEXT PRIMARY KEY,
                current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
                lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active', 'archived')),
                superseded_by TEXT UNIQUE REFERENCES judgments(judgment_id),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS judgment_revisions (
                judgment_id TEXT NOT NULL REFERENCES judgments(judgment_id) ON DELETE CASCADE,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                statement TEXT NOT NULL,
                rationale TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (judgment_id, revision)
            );

            CREATE TABLE IF NOT EXISTS judgment_relation_types (
                relation_type TEXT PRIMARY KEY,
                current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
                lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active', 'deprecated')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS judgment_relation_type_revisions (
                relation_type TEXT NOT NULL
                    REFERENCES judgment_relation_types(relation_type) ON DELETE RESTRICT,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                definition TEXT NOT NULL,
                namespace TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active', 'deprecated')),
                created_at TEXT NOT NULL,
                PRIMARY KEY (relation_type, revision)
            );

            CREATE TABLE IF NOT EXISTS judgment_relations (
                source_id TEXT NOT NULL REFERENCES judgments(judgment_id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES judgments(judgment_id) ON DELETE RESTRICT,
                relation_type TEXT NOT NULL,
                relation_type_revision INTEGER,
                assertion_kind TEXT NOT NULL DEFAULT 'explicit'
                    CHECK(assertion_kind = 'explicit'),
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type),
                CHECK(source_id <> target_id)
            );
            """
        )
        relation_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(judgment_relations)")
        }
        if "relation_type_revision" not in relation_columns:
            self.connection.execute(
                "ALTER TABLE judgment_relations ADD COLUMN relation_type_revision INTEGER"
            )
        if "assertion_kind" not in relation_columns:
            self.connection.execute(
                "ALTER TABLE judgment_relations ADD COLUMN assertion_kind TEXT NOT NULL "
                "DEFAULT 'explicit' CHECK(assertion_kind = 'explicit')"
            )
        try:
            with self.transaction() as connection:
                for relation_type, definition in JUDGMENT_RELATION_TYPE_SEEDS:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO judgment_relation_types(
                            relation_type, current_revision, lifecycle, created_at, updated_at
                        ) VALUES (?, 1, 'active', ?, ?)
                        """,
                        (
                            relation_type,
                            JUDGMENT_RELATION_TYPE_SEED_TIMESTAMP,
                            JUDGMENT_RELATION_TYPE_SEED_TIMESTAMP,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO judgment_relation_type_revisions(
                            relation_type, revision, definition, namespace,
                            provenance_json, lifecycle, created_at
                        ) VALUES (?, 1, ?, ?, ?, 'active', ?)
                        """,
                        (
                            relation_type,
                            definition,
                            JUDGMENT_RELATION_TYPE_NAMESPACE,
                            JUDGMENT_RELATION_TYPE_SEED_PROVENANCE,
                            JUDGMENT_RELATION_TYPE_SEED_TIMESTAMP,
                        ),
                    )
                    seeded_revision = connection.execute(
                        """
                        SELECT definition, namespace, provenance_json, lifecycle
                        FROM judgment_relation_type_revisions
                        WHERE relation_type = ? AND revision = 1
                        """,
                        (relation_type,),
                    ).fetchone()
                    expected = (
                        definition,
                        JUDGMENT_RELATION_TYPE_NAMESPACE,
                        JUDGMENT_RELATION_TYPE_SEED_PROVENANCE,
                        "active",
                    )
                    if seeded_revision is None or tuple(seeded_revision) != expected:
                        raise RuntimeError(
                            f"canonical relation type seed conflict: {relation_type}"
                        )
                    current = connection.execute(
                        """
                        SELECT registry.current_revision, registry.lifecycle,
                               revision.lifecycle AS revision_lifecycle
                        FROM judgment_relation_types AS registry
                        LEFT JOIN judgment_relation_type_revisions AS revision
                          ON revision.relation_type = registry.relation_type
                         AND revision.revision = registry.current_revision
                        WHERE registry.relation_type = ?
                        """,
                        (relation_type,),
                    ).fetchone()
                    if (
                        current is None
                        or current["revision_lifecycle"] is None
                        or current["lifecycle"] != current["revision_lifecycle"]
                    ):
                        raise RuntimeError(
                            f"invalid current relation type revision: {relation_type}"
                        )
                    connection.execute(
                        """
                        UPDATE judgment_relations
                        SET relation_type_revision = 1, assertion_kind = 'explicit'
                        WHERE relation_type = ? AND relation_type_revision IS NULL
                        """,
                        (relation_type,),
                    )
        except Exception:
            self.connection.rollback()
            raise
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
        rows = self.connection.execute(
            """
            SELECT nodes.* FROM nodes
            LEFT JOIN judgments ON judgments.judgment_id = nodes.node_id
            WHERE judgments.judgment_id IS NULL OR judgments.lifecycle = 'active'
            ORDER BY nodes.node_id
            """
        ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def list_edges(self) -> list[TypedEdge]:
        rows = self.connection.execute(
            """
            SELECT edges.* FROM edges
            LEFT JOIN judgments AS source_judgment
              ON source_judgment.judgment_id = edges.source_id
            LEFT JOIN judgments AS target_judgment
              ON target_judgment.judgment_id = edges.target_id
            WHERE (source_judgment.judgment_id IS NULL OR source_judgment.lifecycle = 'active')
              AND (target_judgment.judgment_id IS NULL OR target_judgment.lifecycle = 'active')
            ORDER BY edges.source_id, edges.target_id, edges.edge_type
            """
        ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def outgoing_edges(self, node_id: str) -> list[TypedEdge]:
        rows = self.connection.execute(
            """
            SELECT edges.* FROM edges
            LEFT JOIN relation_edge_dormancy AS dormancy
              ON dormancy.source_id = edges.source_id
             AND dormancy.target_id = edges.target_id
             AND dormancy.edge_type = edges.edge_type
            WHERE edges.source_id = ? AND COALESCE(dormancy.dormant, 0) = 0
              AND NOT EXISTS (
                SELECT 1 FROM judgments
                WHERE judgments.judgment_id = edges.target_id
                  AND judgments.lifecycle <> 'active'
              )
            ORDER BY edges.target_id, edges.edge_type
            """,
            (node_id,),
        ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def edge_has_active_contribution(
        self, source_id: str, target_id: str, edge_type: str
    ) -> bool:
        return self.connection.execute(
            """
            SELECT 1 FROM feedback_contributions
            WHERE source_id = ? AND target_id = ? AND edge_type = ? AND active = 1
            LIMIT 1
            """,
            (source_id, target_id, edge_type),
        ).fetchone() is not None

    def edge_is_dormant(
        self, source_id: str, target_id: str, edge_type: str
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT dormant FROM relation_edge_dormancy
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            (source_id, target_id, edge_type),
        ).fetchone()
        return row is not None and bool(row["dormant"])

    @staticmethod
    def _contribution_id(
        contribution_kind: str,
        source_record_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> str:
        value = "\0".join(
            (contribution_kind, source_record_id, source_id, target_id, edge_type)
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _ensure_edge_journal_state(
        cls,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        edge_type: str,
        baseline_weight: float,
        baseline_reinforced_count: int,
        maximum_weight: float,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO feedback_edge_journal_state(
                source_id, target_id, edge_type, baseline_weight,
                baseline_reinforced_count, maximum_weight
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                target_id,
                edge_type,
                baseline_weight,
                baseline_reinforced_count,
                max(baseline_weight, maximum_weight),
            ),
        )

    @classmethod
    def _insert_contribution(
        cls,
        connection: sqlite3.Connection,
        *,
        contribution_kind: str,
        source_record_id: str,
        trace_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        baseline_weight: float,
        edge_weight_before: float,
        edge_reinforced_count_before: int,
        maximum_weight: float,
        credited_delta: float,
        created_at: float,
        sibling_mutations: Iterable[
            tuple[str, str, str, float, float, int]
        ] = (),
    ) -> str | None:
        if credited_delta <= 0.0:
            return None
        contribution_id = cls._contribution_id(
            contribution_kind, source_record_id, source_id, target_id, edge_type
        )
        cls._ensure_edge_journal_state(
            connection,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            baseline_weight=edge_weight_before,
            baseline_reinforced_count=edge_reinforced_count_before,
            maximum_weight=maximum_weight,
        )
        connection.execute(
            """
            INSERT INTO feedback_contributions(
                contribution_id, contribution_kind, source_record_id, trace_id,
                source_id, target_id, edge_type, baseline_weight, credited_delta,
                reinforced_count_delta, active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?)
            """,
            (
                contribution_id,
                contribution_kind,
                source_record_id,
                trace_id,
                source_id,
                target_id,
                edge_type,
                baseline_weight,
                credited_delta,
                created_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO feedback_contribution_mutations(
                contribution_id, mutation_role, source_id, target_id, edge_type,
                actual_delta
            ) VALUES (?, 'credited', ?, ?, ?, ?)
            """,
            (contribution_id, source_id, target_id, edge_type, credited_delta),
        )
        for (
            sibling_source,
            sibling_target,
            sibling_type,
            reduction,
            sibling_old_weight,
            sibling_reinforced_count,
        ) in sibling_mutations:
            if reduction <= 0.0:
                continue
            cls._ensure_edge_journal_state(
                connection,
                source_id=sibling_source,
                target_id=sibling_target,
                edge_type=sibling_type,
                baseline_weight=sibling_old_weight,
                baseline_reinforced_count=sibling_reinforced_count,
                maximum_weight=maximum_weight,
            )
            connection.execute(
                """
                INSERT INTO feedback_contribution_mutations(
                    contribution_id, mutation_role, source_id, target_id, edge_type,
                    actual_delta
                ) VALUES (?, 'sibling', ?, ?, ?, ?)
                """,
                (
                    contribution_id,
                    sibling_source,
                    sibling_target,
                    sibling_type,
                    -reduction,
                ),
            )
        return contribution_id

    @staticmethod
    def _rebuild_edge_from_active_journal(
        connection: sqlite3.Connection,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> tuple[float, int]:
        state = connection.execute(
            """
            SELECT * FROM feedback_edge_journal_state
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            (source_id, target_id, edge_type),
        ).fetchone()
        if state is None:
            raise KeyError("journaled edge state is absent")
        active_delta = float(
            connection.execute(
                """
                SELECT COALESCE(SUM(mutation.actual_delta), 0.0)
                FROM feedback_contribution_mutations AS mutation
                JOIN feedback_contributions AS contribution
                  ON contribution.contribution_id = mutation.contribution_id
                WHERE mutation.source_id = ? AND mutation.target_id = ?
                  AND mutation.edge_type = ? AND contribution.active = 1
                """,
                (source_id, target_id, edge_type),
            ).fetchone()[0]
        )
        active_reinforced_count = int(
            connection.execute(
                """
                SELECT COALESCE(SUM(contribution.reinforced_count_delta), 0)
                FROM feedback_contribution_mutations AS mutation
                JOIN feedback_contributions AS contribution
                  ON contribution.contribution_id = mutation.contribution_id
                WHERE mutation.source_id = ? AND mutation.target_id = ?
                  AND mutation.edge_type = ? AND mutation.mutation_role = 'credited'
                  AND contribution.active = 1
                """,
                (source_id, target_id, edge_type),
            ).fetchone()[0]
        )
        new_weight = min(
            float(state["maximum_weight"]),
            max(0.0, float(state["baseline_weight"]) + active_delta),
        )
        new_count = int(state["baseline_reinforced_count"]) + active_reinforced_count
        connection.execute(
            """
            UPDATE edges SET weight = ?, reinforced_count = ?
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            (new_weight, new_count, source_id, target_id, edge_type),
        )
        return new_weight, new_count

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

    def source_use_stages(self, trace_id: str) -> dict[str, str]:
        if self.connection.execute(
            "SELECT 1 FROM retrievals WHERE trace_id = ?", (trace_id,)
        ).fetchone() is None:
            raise FeedbackContractError("unknown_trace", "trace handle does not exist")
        return {
            str(row["node_id"]): str(row["stage"])
            for row in self.connection.execute(
                "SELECT node_id, stage FROM source_use_state WHERE trace_id = ?",
                (trace_id,),
            )
        }

    def is_confirmed_candidate_use(self, trace_id: str, node_id: str) -> bool:
        return self.connection.execute(
            """
            SELECT 1 FROM confirmed_source_uses
            WHERE trace_id = ? AND node_id = ?
            """,
            (trace_id, node_id),
        ).fetchone() is not None

    def count_outcomes(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM delayed_outcomes"
        ).fetchone()
        return int(row["count"])

    def count_confirmations(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM confirmed_relation_feedback"
        ).fetchone()
        return int(row["count"])

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

    def apply_evidence_gated_success_feedback(
        self,
        feedback_id: str,
        trace_id: str,
        created_at: float,
        used_node_ids: Iterable[str],
        edge_updates: Iterable[tuple[str, str, str, float, float]],
        normalization_sets: Iterable[
            tuple[str, tuple[tuple[str, str, str], ...], float]
        ] = (),
        *,
        evidence_quorum: int = 1,
    ) -> tuple[
        list[tuple[str, str, str, float, float]],
        list[tuple[str, str, str, float, float]],
        list[tuple[str, str, str, int, int, bool]],
    ]:
        if (
            isinstance(evidence_quorum, bool)
            or not isinstance(evidence_quorum, int)
            or evidence_quorum < 1
        ):
            raise ValueError("evidence_quorum must be a positive integer")
        updates = tuple(edge_updates)
        reinforced: list[tuple[str, str, str, float, float]] = []
        normalized: list[tuple[str, str, str, float, float]] = []
        evidence: list[tuple[str, str, str, int, int, bool]] = []
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
            for source_id, target_id, edge_type, increment, maximum in updates:
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
                inserted = connection.execute(
                    """
                    INSERT INTO relation_feedback_evidence(
                        source_id, target_id, edge_type, trace_id, feedback_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, edge_type, trace_id) DO NOTHING
                    """,
                    (
                        source_id,
                        target_id,
                        edge_type,
                        trace_id,
                        feedback_id,
                        created_at,
                    ),
                ).rowcount == 1
                count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM relation_feedback_evidence
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (source_id, target_id, edge_type),
                    ).fetchone()[0]
                )
                activated = inserted and count >= evidence_quorum
                evidence.append(
                    (
                        source_id,
                        target_id,
                        edge_type,
                        count,
                        evidence_quorum,
                        activated,
                    )
                )
                if not activated:
                    continue
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
        return reinforced, normalized, evidence

    def apply_soft_start_feedback(
        self,
        feedback_id: str,
        trace_id: str,
        created_at: float,
        used_node_ids: Iterable[str],
        edge_updates: Iterable[tuple[str, str, str, float, float]],
        *,
        soft_start_ratio: float,
        decay_ratio: float,
    ) -> list[tuple[str, str, str, float, float]]:
        """Atomically apply the first provisional update for each credited edge."""
        updates = tuple(edge_updates)
        reinforced: list[tuple[str, str, str, float, float]] = []
        with self.transaction() as connection:
            self._require_trace(connection, trace_id)
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
                    "INSERT INTO success_nodes(feedback_id, node_id) VALUES (?, ?)",
                    (feedback_id, node_id),
                )
            for source_id, target_id, edge_type, base_increment, maximum in updates:
                duplicate = connection.execute(
                    """
                    SELECT 1 FROM soft_start_relation_feedback
                    WHERE source_id = ? AND target_id = ? AND edge_type = ? AND trace_id = ?
                    """,
                    (source_id, target_id, edge_type, trace_id),
                ).fetchone()
                if duplicate is not None:
                    continue
                edge = connection.execute(
                    """
                    SELECT weight, reinforced_count FROM edges
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if edge is None:
                    raise KeyError(
                        f"Unknown edge: {source_id} -> {target_id} ({edge_type})"
                    )
                old_weight = float(edge["weight"])
                confirmed_state = connection.execute(
                    """
                    SELECT 1 FROM confirmed_edge_state
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if confirmed_state is not None:
                    raise FeedbackContractError(
                        "confirmation_policy_conflict",
                        "soft-start cannot replace a persisted confirmed-only edge schedule",
                    )
                state = connection.execute(
                    """
                    SELECT * FROM soft_start_edge_state
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                actual_delta = 0.0
                new_weight = old_weight
                if state is None:
                    geometric_maximum = min(
                        maximum, old_weight + base_increment / (1.0 - decay_ratio)
                    )
                    new_weight = max(
                        old_weight,
                        min(
                            maximum,
                            geometric_maximum,
                            old_weight + base_increment * soft_start_ratio,
                        ),
                    )
                    actual_delta = new_weight - old_weight
                    connection.execute(
                        """
                        UPDATE edges
                        SET weight = ?, reinforced_count = reinforced_count + 1
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (new_weight, source_id, target_id, edge_type),
                    )
                    connection.execute(
                        """
                        INSERT INTO soft_start_edge_state(
                            source_id, target_id, edge_type, confirmation_count,
                            base_increment, initial_weight, soft_start_ratio,
                            decay_ratio, geometric_maximum
                        ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            target_id,
                            edge_type,
                            base_increment,
                            old_weight,
                            soft_start_ratio,
                            decay_ratio,
                            geometric_maximum,
                        ),
                    )
                    reinforced.append(
                        (source_id, target_id, edge_type, old_weight, new_weight)
                    )
                else:
                    if not math.isclose(
                        float(state["soft_start_ratio"]),
                        soft_start_ratio,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ) or not math.isclose(
                        float(state["decay_ratio"]),
                        decay_ratio,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise FeedbackContractError(
                            "confirmation_policy_conflict",
                            "soft-start ratio or confirmation decay differs from the "
                            "persisted edge schedule",
                        )
                connection.execute(
                    """
                    INSERT INTO soft_start_relation_feedback(
                        source_id, target_id, edge_type, trace_id, feedback_id,
                        actual_delta, old_weight, new_weight, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        target_id,
                        edge_type,
                        trace_id,
                        feedback_id,
                        actual_delta,
                        old_weight,
                        new_weight,
                        created_at,
                    ),
                )
                self._insert_contribution(
                    connection,
                    contribution_kind="soft_start_provisional",
                    source_record_id=feedback_id,
                    trace_id=trace_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    baseline_weight=(
                        old_weight if state is None else float(state["initial_weight"])
                    ),
                    edge_weight_before=old_weight,
                    edge_reinforced_count_before=int(edge["reinforced_count"]),
                    maximum_weight=maximum,
                    credited_delta=actual_delta,
                    created_at=created_at,
                )
        return reinforced

    def record_source_use(
        self,
        *,
        idempotency_key: str,
        payload_json: str,
        receipt_id: str,
        trace_id: str,
        created_at: float,
        events: tuple[tuple[str, str], ...],
        apply_feedback: Callable[[tuple[str, ...]], FeedbackReceipt] | None,
        confirmation_candidate: bool = False,
    ) -> dict[str, Any]:
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            replay = self._idempotent_replay(
                connection, idempotency_key, "record_source_use", payload_hash
            )
            if replay is not None:
                return replay
            self._require_trace(connection, trace_id)
            allowed = {
                str(row["node_id"])
                for row in connection.execute(
                    "SELECT node_id FROM retrieval_results WHERE trace_id = ?", (trace_id,)
                )
            }
            state = {
                str(row["node_id"]): str(row["stage"])
                for row in connection.execute(
                    "SELECT node_id, stage FROM source_use_state WHERE trace_id = ?",
                    (trace_id,),
                )
            }
            order = {"retrieved": 0, "selected": 1, "validated": 2, "used": 3}
            event_results: list[dict[str, Any]] = []
            newly_used: list[str] = []
            for node_id, stage in events:
                if node_id not in allowed:
                    raise FeedbackContractError(
                        "node_not_in_trace", f"node {node_id} was not returned by this trace"
                    )
                current = state.get(node_id, "retrieved")
                if order[stage] < order[current] or order[stage] > order[current] + 1:
                    raise FeedbackContractError(
                        "invalid_stage_transition",
                        f"node {node_id} cannot transition from {current} to {stage}",
                    )
                changed = stage != current
                if changed:
                    state[node_id] = stage
                    if stage == "used" and node_id not in newly_used:
                        newly_used.append(node_id)
                event_results.append({"node_id": node_id, "stage": stage, "changed": changed})

            feedback: dict[str, Any] | None = None
            if newly_used and apply_feedback is not None:
                receipt = apply_feedback(tuple(newly_used))
                feedback = {
                    "feedback_id": receipt.feedback_id,
                    "trace_id": receipt.trace_id,
                    "used_node_ids": list(receipt.used_node_ids),
                    "reinforced_edges": [
                        {
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "edge_type": edge.edge_type,
                            "old_weight": edge.old_weight,
                            "new_weight": edge.new_weight,
                        }
                        for edge in receipt.reinforced_edges
                    ],
                    "channel": receipt.channel,
                    "normalized_sibling_edges": [
                        {
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "edge_type": edge.edge_type,
                            "old_weight": edge.old_weight,
                            "new_weight": edge.new_weight,
                        }
                        for edge in receipt.normalized_sibling_edges
                    ],
                    "evidence": [
                        {
                            "source_id": item.source_id,
                            "target_id": item.target_id,
                            "edge_type": item.edge_type,
                            "count": item.count,
                            "quorum": item.quorum,
                            "activated": item.activated,
                        }
                        for item in receipt.evidence
                    ],
                }
            for node_id, stage in state.items():
                if stage != "retrieved":
                    connection.execute(
                        """
                        INSERT INTO source_use_state(trace_id, node_id, stage, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(trace_id, node_id) DO UPDATE SET
                            stage = excluded.stage, updated_at = excluded.updated_at
                        """,
                        (trace_id, node_id, stage, created_at),
                    )
            if confirmation_candidate:
                for node_id in newly_used:
                    connection.execute(
                        """
                        INSERT INTO confirmed_source_uses(trace_id, node_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (trace_id, node_id, created_at),
                    )
            result = {
                "receipt_id": receipt_id,
                "trace_id": trace_id,
                "events": event_results,
                "newly_used_node_ids": newly_used,
                "feedback": feedback,
            }
            self._save_idempotent_result(
                connection, idempotency_key, "record_source_use", payload_hash, result
            )
            return result

    def record_confirmed_outcome(
        self,
        *,
        idempotency_key: str,
        payload_json: str,
        outcome_id: str,
        trace_id: str,
        node_ids: tuple[str, ...],
        summary: str,
        external_ref: str | None,
        recorded_at: float,
        decay_ratio: float,
        edge_updates: tuple[tuple[str, str, str, float, float], ...],
        normalization_sets: tuple[
            tuple[str, tuple[tuple[str, str, str], ...], float], ...
        ],
        credited_paths: tuple[dict[str, object], ...],
    ) -> dict[str, Any]:
        """Atomically record a confirmed outcome and its diminishing edge updates."""
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        confirmations: list[dict[str, Any]] = []
        normalized: list[dict[str, Any]] = []
        with self.transaction() as connection:
            replay = self._idempotent_replay(
                connection, idempotency_key, "record_outcome", payload_hash
            )
            if replay is not None:
                return replay
            self._require_trace(connection, trace_id)
            self._require_used_nodes(connection, trace_id, node_ids)
            connection.execute(
                """
                INSERT INTO delayed_outcomes(
                    outcome_id, trace_id, outcome, summary, external_ref, recorded_at
                ) VALUES (?, ?, 'confirmed', ?, ?, ?)
                """,
                (outcome_id, trace_id, summary, external_ref, recorded_at),
            )
            for node_id in node_ids:
                connection.execute(
                    "INSERT INTO delayed_outcome_nodes(outcome_id, node_id) VALUES (?, ?)",
                    (outcome_id, node_id),
                )

            reinforced_increase_by_source: dict[str, float] = {}
            for source_id, target_id, edge_type, base_increment, maximum in edge_updates:
                duplicate = connection.execute(
                    """
                    SELECT 1 FROM confirmed_relation_feedback
                    WHERE source_id = ? AND target_id = ? AND edge_type = ? AND trace_id = ?
                    """,
                    (source_id, target_id, edge_type, trace_id),
                ).fetchone()
                if duplicate is not None:
                    continue
                edge = connection.execute(
                    """
                    SELECT weight FROM edges
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if edge is None:
                    raise KeyError(
                        f"Unknown edge: {source_id} -> {target_id} ({edge_type})"
                    )
                old_weight = float(edge["weight"])
                state = connection.execute(
                    """
                    SELECT * FROM confirmed_edge_state
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if state is None:
                    soft_start_state = connection.execute(
                        """
                        SELECT 1 FROM soft_start_edge_state
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (source_id, target_id, edge_type),
                    ).fetchone()
                    if soft_start_state is not None:
                        raise FeedbackContractError(
                            "confirmation_policy_conflict",
                            "confirmed-only cannot replace a persisted soft-start edge schedule",
                        )
                    confirmation_count = 1
                    stored_base_increment = base_increment
                    geometric_maximum = min(
                        maximum, old_weight + base_increment / (1.0 - decay_ratio)
                    )
                else:
                    stored_ratio = float(state["decay_ratio"])
                    if not math.isclose(
                        stored_ratio, decay_ratio, rel_tol=0.0, abs_tol=1e-15
                    ):
                        raise FeedbackContractError(
                            "confirmation_policy_conflict",
                            "confirmation decay ratio differs from the persisted edge schedule",
                        )
                    confirmation_count = int(state["confirmation_count"]) + 1
                    stored_base_increment = float(state["base_increment"])
                    geometric_maximum = float(state["geometric_maximum"])
                multiplier = decay_ratio ** (confirmation_count - 1)
                new_weight = max(
                    old_weight,
                    min(
                        maximum,
                        geometric_maximum,
                        old_weight + stored_base_increment * multiplier,
                    ),
                )
                actual_delta = new_weight - old_weight
                connection.execute(
                    """
                    UPDATE edges
                    SET weight = ?, reinforced_count = reinforced_count + 1
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (new_weight, source_id, target_id, edge_type),
                )
                connection.execute(
                    """
                    INSERT INTO confirmed_edge_state(
                        source_id, target_id, edge_type, confirmation_count,
                        base_increment, initial_weight, decay_ratio, geometric_maximum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                        confirmation_count = excluded.confirmation_count
                    """,
                    (
                        source_id,
                        target_id,
                        edge_type,
                        confirmation_count,
                        stored_base_increment,
                        old_weight if state is None else float(state["initial_weight"]),
                        decay_ratio,
                        geometric_maximum,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO confirmed_relation_feedback(
                        source_id, target_id, edge_type, trace_id, outcome_id,
                        confirmation_count, multiplier, actual_delta,
                        old_weight, new_weight, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        target_id,
                        edge_type,
                        trace_id,
                        outcome_id,
                        confirmation_count,
                        multiplier,
                        actual_delta,
                        old_weight,
                        new_weight,
                        recorded_at,
                    ),
                )
                confirmations.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type,
                        "confirmation_count": confirmation_count,
                        "multiplier": multiplier,
                        "actual_delta": actual_delta,
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                    }
                )
                reinforced_increase_by_source[source_id] = (
                    reinforced_increase_by_source.get(source_id, 0.0) + actual_delta
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
                            {
                                "source_id": sibling_source,
                                "target_id": target_id,
                                "edge_type": edge_type,
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                            }
                        )
            result = {
                "outcome_id": outcome_id,
                "trace_id": trace_id,
                "node_ids": list(node_ids),
                "outcome": "confirmed",
                "recorded_at": recorded_at,
                "reinforcement_applied": bool(confirmations),
                "confirmations": confirmations,
                "credited_paths": list(credited_paths),
                "normalized_sibling_edges": normalized,
            }
            self._save_idempotent_result(
                connection, idempotency_key, "record_outcome", payload_hash, result
            )
            return result

    def record_soft_start_confirmed_outcome(
        self,
        *,
        idempotency_key: str,
        payload_json: str,
        outcome_id: str,
        trace_id: str,
        node_ids: tuple[str, ...],
        summary: str,
        external_ref: str | None,
        recorded_at: float,
        decay_ratio: float,
        soft_start_ratio: float,
        edge_updates: tuple[tuple[str, str, str, float, float], ...],
        normalization_sets: tuple[
            tuple[str, tuple[tuple[str, str, str], ...], float], ...
        ],
        credited_paths: tuple[dict[str, object], ...],
    ) -> dict[str, Any]:
        """Atomically record confirmation of a persisted soft-start schedule."""
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        confirmations: list[dict[str, Any]] = []
        normalized: list[dict[str, Any]] = []
        contribution_specs: list[dict[str, Any]] = []
        sibling_reductions_by_source: dict[
            str, list[tuple[str, str, str, float, float, int]]
        ] = {}
        reactivated: list[dict[str, Any]] = []
        with self.transaction() as connection:
            replay = self._idempotent_replay(
                connection, idempotency_key, "record_outcome", payload_hash
            )
            if replay is not None:
                return replay
            self._require_trace(connection, trace_id)
            self._require_used_nodes(connection, trace_id, node_ids)
            connection.execute(
                """
                INSERT INTO delayed_outcomes(
                    outcome_id, trace_id, outcome, summary, external_ref, recorded_at
                ) VALUES (?, ?, 'confirmed', ?, ?, ?)
                """,
                (outcome_id, trace_id, summary, external_ref, recorded_at),
            )
            for node_id in node_ids:
                connection.execute(
                    "INSERT INTO delayed_outcome_nodes(outcome_id, node_id) VALUES (?, ?)",
                    (outcome_id, node_id),
                )

            reinforced_increase_by_source: dict[str, float] = {}
            for source_id, target_id, edge_type, base_increment, maximum in edge_updates:
                duplicate = connection.execute(
                    """
                    SELECT 1 FROM confirmed_relation_feedback
                    WHERE source_id = ? AND target_id = ? AND edge_type = ? AND trace_id = ?
                    """,
                    (source_id, target_id, edge_type, trace_id),
                ).fetchone()
                if duplicate is not None:
                    continue
                dormant = connection.execute(
                    """
                    SELECT dormant FROM relation_edge_dormancy
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if dormant is not None and bool(dormant["dormant"]):
                    connection.execute(
                        """
                        UPDATE relation_edge_dormancy
                        SET dormant = 0, outcome_id = ?, trace_id = ?, updated_at = ?
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (
                            outcome_id,
                            trace_id,
                            recorded_at,
                            source_id,
                            target_id,
                            edge_type,
                        ),
                    )
                    reactivated.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "edge_type": edge_type,
                            "old_dormant": True,
                            "new_dormant": False,
                        }
                    )
                edge = connection.execute(
                    """
                    SELECT weight, reinforced_count FROM edges
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if edge is None:
                    raise KeyError(
                        f"Unknown edge: {source_id} -> {target_id} ({edge_type})"
                    )
                confirmed_only_state = connection.execute(
                    """
                    SELECT 1 FROM confirmed_edge_state
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if confirmed_only_state is not None:
                    raise FeedbackContractError(
                        "confirmation_policy_conflict",
                        "soft-start cannot replace a persisted confirmed-only edge schedule",
                    )
                state = connection.execute(
                    """
                    SELECT * FROM soft_start_edge_state
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (source_id, target_id, edge_type),
                ).fetchone()
                if state is None:
                    continue
                if not math.isclose(
                    float(state["soft_start_ratio"]),
                    soft_start_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ) or not math.isclose(
                    float(state["decay_ratio"]),
                    decay_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise FeedbackContractError(
                        "confirmation_policy_conflict",
                        "soft-start ratio or confirmation decay differs from the "
                        "persisted edge schedule",
                    )
                old_weight = float(edge["weight"])
                confirmation_count = int(state["confirmation_count"]) + 1
                stored_base_increment = float(state["base_increment"])
                if confirmation_count == 1:
                    multiplier = 1.0 - soft_start_ratio
                    target_weight = min(
                        maximum,
                        float(state["geometric_maximum"]),
                        float(state["initial_weight"]) + stored_base_increment,
                    )
                    new_weight = max(old_weight, target_weight)
                else:
                    multiplier = decay_ratio ** (confirmation_count - 1)
                    new_weight = max(
                        old_weight,
                        min(
                            maximum,
                            float(state["geometric_maximum"]),
                            old_weight + stored_base_increment * multiplier,
                        ),
                    )
                actual_delta = new_weight - old_weight
                connection.execute(
                    """
                    UPDATE edges
                    SET weight = ?, reinforced_count = reinforced_count + 1
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (new_weight, source_id, target_id, edge_type),
                )
                connection.execute(
                    """
                    UPDATE soft_start_edge_state SET confirmation_count = ?
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    (confirmation_count, source_id, target_id, edge_type),
                )
                connection.execute(
                    """
                    INSERT INTO confirmed_relation_feedback(
                        source_id, target_id, edge_type, trace_id, outcome_id,
                        confirmation_count, multiplier, actual_delta,
                        old_weight, new_weight, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id, target_id, edge_type, trace_id, outcome_id,
                        confirmation_count, multiplier, actual_delta,
                        old_weight, new_weight, recorded_at,
                    ),
                )
                confirmations.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type,
                        "confirmation_count": confirmation_count,
                        "multiplier": multiplier,
                        "actual_delta": actual_delta,
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                    }
                )
                contribution_specs.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type,
                        "baseline_weight": float(state["initial_weight"]),
                        "edge_weight_before": old_weight,
                        "edge_reinforced_count_before": int(
                            edge["reinforced_count"]
                        ),
                        "maximum_weight": maximum,
                        "credited_delta": actual_delta,
                    }
                )
                reinforced_increase_by_source[source_id] = (
                    reinforced_increase_by_source.get(source_id, 0.0) + actual_delta
                )

            for source_id, sibling_keys, ratio in normalization_sets:
                total_increase = reinforced_increase_by_source.get(source_id, 0.0)
                if total_increase <= 0.0 or not sibling_keys:
                    continue
                reduction = total_increase * ratio / len(sibling_keys)
                for sibling_source, target_id, edge_type in sibling_keys:
                    row = connection.execute(
                        """
                        SELECT weight, reinforced_count FROM edges
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
                            {
                                "source_id": sibling_source,
                                "target_id": target_id,
                                "edge_type": edge_type,
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                            }
                        )
                        sibling_reductions_by_source.setdefault(source_id, []).append(
                            (
                                sibling_source,
                                target_id,
                                edge_type,
                                old_weight - new_weight,
                                old_weight,
                                int(row["reinforced_count"]),
                            )
                        )
            for spec in contribution_specs:
                source_id = str(spec["source_id"])
                total = reinforced_increase_by_source.get(source_id, 0.0)
                share = float(spec["credited_delta"]) / total if total > 0.0 else 0.0
                self._insert_contribution(
                    connection,
                    contribution_kind="soft_start_confirmation",
                    source_record_id=outcome_id,
                    trace_id=trace_id,
                    source_id=source_id,
                    target_id=str(spec["target_id"]),
                    edge_type=str(spec["edge_type"]),
                    baseline_weight=float(spec["baseline_weight"]),
                    edge_weight_before=float(spec["edge_weight_before"]),
                    edge_reinforced_count_before=int(
                        spec["edge_reinforced_count_before"]
                    ),
                    maximum_weight=float(spec["maximum_weight"]),
                    credited_delta=float(spec["credited_delta"]),
                    created_at=recorded_at,
                    sibling_mutations=tuple(
                        (
                            sibling_source,
                            sibling_target,
                            sibling_type,
                            reduction * share,
                            sibling_old_weight,
                            sibling_reinforced_count,
                        )
                        for (
                            sibling_source,
                            sibling_target,
                            sibling_type,
                            reduction,
                            sibling_old_weight,
                            sibling_reinforced_count,
                        )
                        in sibling_reductions_by_source.get(source_id, [])
                    ),
                )
            result = {
                "outcome_id": outcome_id,
                "trace_id": trace_id,
                "node_ids": list(node_ids),
                "outcome": "confirmed",
                "recorded_at": recorded_at,
                "reinforcement_applied": bool(confirmations),
                "confirmations": confirmations,
                "credited_paths": list(credited_paths),
                "normalized_sibling_edges": normalized,
                "reactivated_edges": reactivated,
            }
            self._save_idempotent_result(
                connection, idempotency_key, "record_outcome", payload_hash, result
            )
            return result

    def record_outcome(
        self,
        *,
        idempotency_key: str,
        payload_json: str,
        outcome_id: str,
        trace_id: str,
        node_ids: tuple[str, ...],
        outcome: str,
        summary: str,
        external_ref: str | None,
        recorded_at: float,
    ) -> dict[str, Any]:
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            replay = self._idempotent_replay(
                connection, idempotency_key, "record_outcome", payload_hash
            )
            if replay is not None:
                return replay
            self._require_trace(connection, trace_id)
            self._require_used_nodes(connection, trace_id, node_ids)
            connection.execute(
                """
                INSERT INTO delayed_outcomes(
                    outcome_id, trace_id, outcome, summary, external_ref, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (outcome_id, trace_id, outcome, summary, external_ref, recorded_at),
            )
            for node_id in node_ids:
                connection.execute(
                    "INSERT INTO delayed_outcome_nodes(outcome_id, node_id) VALUES (?, ?)",
                    (outcome_id, node_id),
                )
            result = {
                "outcome_id": outcome_id,
                "trace_id": trace_id,
                "node_ids": list(node_ids),
                "outcome": outcome,
                "recorded_at": recorded_at,
                "reinforcement_applied": False,
            }
            self._save_idempotent_result(
                connection, idempotency_key, "record_outcome", payload_hash, result
            )
            return result

    def record_deactivation_outcome(
        self,
        *,
        idempotency_key: str,
        payload_json: str,
        outcome_id: str,
        trace_id: str,
        node_ids: tuple[str, ...],
        outcome: str,
        summary: str,
        external_ref: str | None,
        recorded_at: float,
        credited_paths: tuple[dict[str, object], ...],
    ) -> dict[str, Any]:
        """Atomically reverse attributable contributions or change edge dormancy."""
        if outcome not in {"corrected", "rolled_back", "superseded"}:
            raise ValueError("deactivation outcome is not supported")
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            replay = self._idempotent_replay(
                connection, idempotency_key, "record_outcome", payload_hash
            )
            if replay is not None:
                return replay
            self._require_trace(connection, trace_id)
            self._require_used_nodes(connection, trace_id, node_ids)
            connection.execute(
                """
                INSERT INTO delayed_outcomes(
                    outcome_id, trace_id, outcome, summary, external_ref, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (outcome_id, trace_id, outcome, summary, external_ref, recorded_at),
            )
            for node_id in node_ids:
                connection.execute(
                    "INSERT INTO delayed_outcome_nodes(outcome_id, node_id) VALUES (?, ?)",
                    (outcome_id, node_id),
                )

            credited_keys = {
                (
                    str(step["source_id"]),
                    str(step["target_id"]),
                    str(step["edge_type"]),
                )
                for path in credited_paths
                for step in path["steps"]
            }
            reversed_contributions: list[dict[str, Any]] = []
            dormancy_changes: list[dict[str, Any]] = []
            if outcome in {"corrected", "rolled_back"}:
                for source_id, target_id, edge_type in sorted(credited_keys):
                    contributions = connection.execute(
                        """
                        SELECT * FROM feedback_contributions
                        WHERE trace_id = ? AND source_id = ? AND target_id = ?
                          AND edge_type = ? AND active = 1
                        ORDER BY created_at DESC, contribution_id DESC
                        """,
                        (trace_id, source_id, target_id, edge_type),
                    ).fetchall()
                    for contribution in contributions:
                        mutations: list[dict[str, Any]] = []
                        mutation_rows = connection.execute(
                            """
                            SELECT * FROM feedback_contribution_mutations
                            WHERE contribution_id = ?
                            ORDER BY mutation_role, source_id, target_id, edge_type
                            """,
                            (contribution["contribution_id"],),
                        ).fetchall()
                        old_weights: dict[tuple[str, str, str], float] = {}
                        for mutation in mutation_rows:
                            edge = connection.execute(
                                """
                                SELECT weight FROM edges
                                WHERE source_id = ? AND target_id = ? AND edge_type = ?
                                """,
                                (
                                    mutation["source_id"],
                                    mutation["target_id"],
                                    mutation["edge_type"],
                                ),
                            ).fetchone()
                            if edge is None:
                                raise KeyError("journaled contribution edge is absent")
                            identity = (
                                str(mutation["source_id"]),
                                str(mutation["target_id"]),
                                str(mutation["edge_type"]),
                            )
                            old_weights[identity] = float(edge["weight"])
                        connection.execute(
                            """
                            UPDATE feedback_contributions
                            SET active = 0, reversed_by_outcome_id = ?
                            WHERE contribution_id = ?
                            """,
                            (outcome_id, contribution["contribution_id"]),
                        )
                        for mutation in mutation_rows:
                            identity = (
                                str(mutation["source_id"]),
                                str(mutation["target_id"]),
                                str(mutation["edge_type"]),
                            )
                            old_weight = old_weights[identity]
                            new_weight, _ = self._rebuild_edge_from_active_journal(
                                connection, *identity
                            )
                            mutations.append(
                                {
                                    "mutation_role": str(mutation["mutation_role"]),
                                    "source_id": str(mutation["source_id"]),
                                    "target_id": str(mutation["target_id"]),
                                    "edge_type": str(mutation["edge_type"]),
                                    "actual_delta": float(mutation["actual_delta"]),
                                    "old_weight": old_weight,
                                    "new_weight": new_weight,
                                }
                            )
                        reversed_contributions.append(
                            {
                                "contribution_id": str(contribution["contribution_id"]),
                                "contribution_kind": str(contribution["contribution_kind"]),
                                "source_record_id": str(contribution["source_record_id"]),
                                "source_id": str(contribution["source_id"]),
                                "target_id": str(contribution["target_id"]),
                                "edge_type": str(contribution["edge_type"]),
                                "credited_delta": float(contribution["credited_delta"]),
                                "mutations": mutations,
                            }
                        )
            else:
                for source_id, target_id, edge_type in sorted(credited_keys):
                    attributable = connection.execute(
                        """
                        SELECT 1 FROM feedback_contributions
                        WHERE trace_id = ? AND source_id = ? AND target_id = ?
                          AND edge_type = ?
                        LIMIT 1
                        """,
                        (trace_id, source_id, target_id, edge_type),
                    ).fetchone()
                    if attributable is None:
                        continue
                    current = connection.execute(
                        """
                        SELECT dormant FROM relation_edge_dormancy
                        WHERE source_id = ? AND target_id = ? AND edge_type = ?
                        """,
                        (source_id, target_id, edge_type),
                    ).fetchone()
                    old_dormant = current is not None and bool(current["dormant"])
                    connection.execute(
                        """
                        INSERT INTO relation_edge_dormancy(
                            source_id, target_id, edge_type, dormant,
                            outcome_id, trace_id, updated_at
                        ) VALUES (?, ?, ?, 1, ?, ?, ?)
                        ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                            dormant = 1,
                            outcome_id = excluded.outcome_id,
                            trace_id = excluded.trace_id,
                            updated_at = excluded.updated_at
                        """,
                        (
                            source_id, target_id, edge_type,
                            outcome_id, trace_id, recorded_at,
                        ),
                    )
                    if not old_dormant:
                        dormancy_changes.append(
                            {
                                "source_id": source_id,
                                "target_id": target_id,
                                "edge_type": edge_type,
                                "old_dormant": False,
                                "new_dormant": True,
                            }
                        )
            result = {
                "outcome_id": outcome_id,
                "trace_id": trace_id,
                "node_ids": list(node_ids),
                "outcome": outcome,
                "recorded_at": recorded_at,
                "reinforcement_applied": False,
                "deactivation_applied": bool(
                    reversed_contributions or dormancy_changes
                ),
                "confirmations": [],
                "credited_paths": list(credited_paths),
                "normalized_sibling_edges": [],
                "reversed_contributions": reversed_contributions,
                "dormancy_changes": dormancy_changes,
                "reactivated_edges": [],
            }
            self._save_idempotent_result(
                connection, idempotency_key, "record_outcome", payload_hash, result
            )
            return result

    @staticmethod
    def _require_used_nodes(
        connection: sqlite3.Connection, trace_id: str, node_ids: tuple[str, ...]
    ) -> None:
        for node_id in node_ids:
            row = connection.execute(
                """
                SELECT stage FROM source_use_state
                WHERE trace_id = ? AND node_id = ?
                """,
                (trace_id, node_id),
            ).fetchone()
            if row is None or str(row["stage"]) != "used":
                raise FeedbackContractError(
                    "source_not_used", f"node {node_id} is not marked used for this trace"
                )

    @staticmethod
    def _require_trace(connection: sqlite3.Connection, trace_id: str) -> None:
        if connection.execute(
            "SELECT 1 FROM retrievals WHERE trace_id = ?", (trace_id,)
        ).fetchone() is None:
            raise FeedbackContractError("unknown_trace", "trace handle does not exist")

    @staticmethod
    def _idempotent_replay(
        connection: sqlite3.Connection,
        key: str,
        operation: str,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT operation, payload_hash, result_json FROM feedback_requests WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if str(row["operation"]) != operation or str(row["payload_hash"]) != payload_hash:
            raise FeedbackContractError(
                "idempotency_conflict", "idempotency key was already used with a different request"
            )
        return dict(json.loads(row["result_json"]))

    @staticmethod
    def _save_idempotent_result(
        connection: sqlite3.Connection,
        key: str,
        operation: str,
        payload_hash: str,
        result: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO feedback_requests(idempotency_key, operation, payload_hash, result_json) VALUES (?, ?, ?, ?)",
            (key, operation, payload_hash, json.dumps(result, sort_keys=True, ensure_ascii=False)),
        )

    def count_retrievals(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM retrievals").fetchone()
        return int(row["count"])

    def count_feedback(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM success_feedback"
        ).fetchone()
        return int(row["count"])

    def count_feedback_evidence(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM relation_feedback_evidence"
        ).fetchone()
        return int(row["count"])

    def feedback_evidence_count(
        self, source_id: str, target_id: str, edge_type: str
    ) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count FROM relation_feedback_evidence
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
            """,
            (source_id, target_id, edge_type),
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

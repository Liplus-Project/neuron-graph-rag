from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Iterable

from .models import DocumentNode, TypedEdge
from .storage import SQLiteStore


_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_RELATION = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class JudgmentContractError(ValueError):
    """A fail-closed domain contract violation."""


class JudgmentGraph:
    """Atomic domain API for the canonical SQLite judgment graph."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _identity(value: str) -> str:
        if not isinstance(value, str) or _ID.fullmatch(value) is None:
            raise JudgmentContractError("invalid judgment identity")
        return value

    @staticmethod
    def _text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise JudgmentContractError(f"{name} is required")
        return value.strip()

    @staticmethod
    def _provenance(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise JudgmentContractError("provenance must be a non-empty object")
        try:
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise JudgmentContractError("provenance must be deterministic JSON") from error
        return value

    @staticmethod
    def _relations(
        relations: Iterable[dict[str, str]], *, source_id: str
    ) -> tuple[tuple[str, str], ...]:
        normalized: list[tuple[str, str]] = []
        for relation in relations:
            if not isinstance(relation, dict) or set(relation) != {"target_id", "relation_type"}:
                raise JudgmentContractError("relation requires target_id and relation_type")
            target = JudgmentGraph._identity(relation["target_id"])
            kind = relation["relation_type"]
            if not isinstance(kind, str) or _RELATION.fullmatch(kind) is None:
                raise JudgmentContractError("invalid relation type")
            if target == source_id:
                raise JudgmentContractError("self relations are not allowed")
            normalized.append((target, kind))
        if len(set(normalized)) != len(normalized):
            raise JudgmentContractError("duplicate relation")
        return tuple(sorted(normalized))

    def add(
        self,
        judgment_id: str,
        statement: str,
        rationale: str,
        provenance: dict[str, Any],
        *,
        relations: Iterable[dict[str, str]] = (),
    ) -> dict[str, Any]:
        judgment_id = self._identity(judgment_id)
        statement = self._text(statement, "statement")
        rationale = self._text(rationale, "rationale")
        provenance = self._provenance(provenance)
        normalized = self._relations(relations, source_id=judgment_id)
        now = self._now()
        with self.store.transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM judgments WHERE judgment_id = ?", (judgment_id,)
            ).fetchone():
                raise JudgmentContractError("judgment already exists")
            self._require_targets(connection, normalized)
            connection.execute(
                "INSERT INTO judgments VALUES (?, 1, 'active', NULL, ?, ?)",
                (judgment_id, now, now),
            )
            self._insert_revision(connection, judgment_id, 1, statement, rationale, provenance, now)
            self._replace_relations(connection, judgment_id, normalized, now)
            self._sync_retrieval(connection, judgment_id, statement, rationale, provenance, normalized)
        return self.get(judgment_id)

    def update(
        self,
        judgment_id: str,
        statement: str,
        rationale: str,
        provenance: dict[str, Any],
        *,
        expected_revision: int,
        relations: Iterable[dict[str, str]] = (),
    ) -> dict[str, Any]:
        judgment_id = self._identity(judgment_id)
        statement = self._text(statement, "statement")
        rationale = self._text(rationale, "rationale")
        provenance = self._provenance(provenance)
        normalized = self._relations(relations, source_id=judgment_id)
        now = self._now()
        with self.store.transaction() as connection:
            row = self._require(connection, judgment_id)
            if row["lifecycle"] != "active" or row["superseded_by"] is not None:
                raise JudgmentContractError("only current active judgments can be updated")
            if expected_revision != row["current_revision"]:
                raise JudgmentContractError("stale expected_revision")
            self._require_targets(connection, normalized)
            revision = expected_revision + 1
            self._insert_revision(connection, judgment_id, revision, statement, rationale, provenance, now)
            connection.execute(
                "UPDATE judgments SET current_revision = ?, updated_at = ? WHERE judgment_id = ?",
                (revision, now, judgment_id),
            )
            self._replace_relations(connection, judgment_id, normalized, now)
            self._sync_retrieval(connection, judgment_id, statement, rationale, provenance, normalized)
        return self.get(judgment_id)

    def supersede(
        self,
        predecessor_id: str,
        successor_id: str,
        statement: str,
        rationale: str,
        provenance: dict[str, Any],
        *,
        expected_revision: int,
        relations: Iterable[dict[str, str]] = (),
    ) -> dict[str, Any]:
        predecessor_id = self._identity(predecessor_id)
        successor_id = self._identity(successor_id)
        if predecessor_id == successor_id:
            raise JudgmentContractError("successor must have a new identity")
        statement = self._text(statement, "statement")
        rationale = self._text(rationale, "rationale")
        provenance = self._provenance(provenance)
        normalized = self._relations(relations, source_id=successor_id)
        normalized = tuple(sorted((*normalized, (predecessor_id, "supersedes"))))
        now = self._now()
        with self.store.transaction() as connection:
            old = self._require(connection, predecessor_id)
            if old["lifecycle"] != "active" or old["superseded_by"] is not None:
                raise JudgmentContractError("predecessor already inactive or superseded")
            if old["current_revision"] != expected_revision:
                raise JudgmentContractError("stale expected_revision")
            if connection.execute("SELECT 1 FROM judgments WHERE judgment_id = ?", (successor_id,)).fetchone():
                raise JudgmentContractError("successor already exists")
            self._require_targets(connection, tuple(item for item in normalized if item[0] != predecessor_id))
            connection.execute(
                "INSERT INTO judgments VALUES (?, 1, 'active', NULL, ?, ?)",
                (successor_id, now, now),
            )
            self._insert_revision(connection, successor_id, 1, statement, rationale, provenance, now)
            self._replace_relations(connection, successor_id, normalized, now)
            self._sync_retrieval(connection, successor_id, statement, rationale, provenance, normalized)
            connection.execute(
                "UPDATE judgments SET lifecycle = 'archived', superseded_by = ?, updated_at = ? WHERE judgment_id = ?",
                (successor_id, now, predecessor_id),
            )
        return self.get(successor_id)

    def archive(self, judgment_id: str, *, expected_revision: int) -> dict[str, Any]:
        return self._lifecycle(judgment_id, expected_revision, "archived")

    def restore(self, judgment_id: str, *, expected_revision: int) -> dict[str, Any]:
        return self._lifecycle(judgment_id, expected_revision, "active")

    def _lifecycle(self, judgment_id: str, expected_revision: int, lifecycle: str) -> dict[str, Any]:
        judgment_id = self._identity(judgment_id)
        with self.store.transaction() as connection:
            row = self._require(connection, judgment_id)
            if row["current_revision"] != expected_revision:
                raise JudgmentContractError("stale expected_revision")
            if lifecycle == "active" and row["superseded_by"] is not None:
                raise JudgmentContractError("superseded judgments cannot be restored")
            connection.execute(
                "UPDATE judgments SET lifecycle = ?, updated_at = ? WHERE judgment_id = ?",
                (lifecycle, self._now(), judgment_id),
            )
        return self.get(judgment_id)

    def hard_delete(self, judgment_id: str, *, expected_revision: int) -> None:
        judgment_id = self._identity(judgment_id)
        with self.store.transaction() as connection:
            row = self._require(connection, judgment_id)
            if row["current_revision"] != expected_revision or row["lifecycle"] != "archived":
                raise JudgmentContractError("hard delete requires an archived current revision")
            inbound = connection.execute(
                "SELECT 1 FROM judgment_relations WHERE target_id = ? LIMIT 1", (judgment_id,)
            ).fetchone()
            if inbound or row["superseded_by"] is not None:
                raise JudgmentContractError("hard delete candidate has retained graph history")
            connection.execute("DELETE FROM nodes WHERE node_id = ?", (judgment_id,))
            connection.execute("DELETE FROM judgments WHERE judgment_id = ?", (judgment_id,))

    def get(self, judgment_id: str) -> dict[str, Any]:
        judgment_id = self._identity(judgment_id)
        row = self.store.connection.execute(
            """
            SELECT j.*, r.statement, r.rationale, r.provenance_json
            FROM judgments j JOIN judgment_revisions r
              ON r.judgment_id = j.judgment_id AND r.revision = j.current_revision
            WHERE j.judgment_id = ?
            """, (judgment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown judgment: {judgment_id}")
        relations = self.store.connection.execute(
            "SELECT target_id, relation_type FROM judgment_relations WHERE source_id = ? ORDER BY target_id, relation_type",
            (judgment_id,),
        ).fetchall()
        return {
            "judgment_id": row["judgment_id"], "revision": row["current_revision"],
            "statement": row["statement"], "rationale": row["rationale"],
            "provenance": json.loads(row["provenance_json"]), "lifecycle": row["lifecycle"],
            "superseded_by": row["superseded_by"],
            "relations": [dict(item) for item in relations],
        }

    def export(self) -> dict[str, Any]:
        ids = [row[0] for row in self.store.connection.execute("SELECT judgment_id FROM judgments ORDER BY judgment_id")]
        return {"format": "ngr-judgment-graph/v1", "judgments": [self.get(item) for item in ids]}

    def import_graph(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("format") != "ngr-judgment-graph/v1":
            raise JudgmentContractError("unsupported judgment graph format")
        records = payload.get("judgments")
        if not isinstance(records, list):
            raise JudgmentContractError("judgments must be a list")
        ids = {record.get("judgment_id") for record in records if isinstance(record, dict)}
        if len(ids) != len(records) or None in ids:
            raise JudgmentContractError("judgment identities must be unique")
        for record in records:
            for relation in record.get("relations", []):
                if relation.get("target_id") not in ids:
                    raise JudgmentContractError("dangling relation")
        now = self._now()
        with self.store.transaction() as connection:
            for record in sorted(records, key=lambda item: item["judgment_id"]):
                judgment_id = self._identity(record["judgment_id"])
                revision = record.get("revision")
                if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                    raise JudgmentContractError("invalid revision")
                statement = self._text(record["statement"], "statement")
                rationale = self._text(record["rationale"], "rationale")
                provenance = self._provenance(record["provenance"])
                lifecycle = record.get("lifecycle", "active")
                if lifecycle not in {"active", "archived"}:
                    raise JudgmentContractError("invalid lifecycle")
                connection.execute(
                    "INSERT INTO judgments VALUES (?, ?, ?, NULL, ?, ?)",
                    (judgment_id, revision, lifecycle, now, now),
                )
                self._insert_revision(connection, judgment_id, revision, statement, rationale, provenance, now)
                self._sync_retrieval(connection, judgment_id, statement, rationale, provenance, ())
            for record in sorted(records, key=lambda item: item["judgment_id"]):
                relations = self._relations(record.get("relations", []), source_id=record["judgment_id"])
                self._require_targets(connection, relations)
                self._replace_relations(connection, record["judgment_id"], relations, now)
                self._sync_retrieval(
                    connection, record["judgment_id"], record["statement"], record["rationale"],
                    record["provenance"], relations,
                )
            for record in sorted(records, key=lambda item: item["judgment_id"]):
                successor = record.get("superseded_by")
                if successor is not None and successor not in ids:
                    raise JudgmentContractError("unknown successor")
                connection.execute(
                    "UPDATE judgments SET superseded_by = ? WHERE judgment_id = ?",
                    (successor, record["judgment_id"]),
                )

    @staticmethod
    def _require(connection: Any, judgment_id: str) -> Any:
        row = connection.execute("SELECT * FROM judgments WHERE judgment_id = ?", (judgment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown judgment: {judgment_id}")
        return row

    @staticmethod
    def _require_targets(connection: Any, relations: tuple[tuple[str, str], ...]) -> None:
        for target, _ in relations:
            if connection.execute("SELECT 1 FROM judgments WHERE judgment_id = ?", (target,)).fetchone() is None:
                raise JudgmentContractError(f"unknown relation target: {target}")

    @staticmethod
    def _insert_revision(connection: Any, judgment_id: str, revision: int, statement: str, rationale: str, provenance: dict[str, Any], now: str) -> None:
        connection.execute(
            "INSERT INTO judgment_revisions VALUES (?, ?, ?, ?, ?, ?)",
            (judgment_id, revision, statement, rationale, json.dumps(provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":")), now),
        )

    @staticmethod
    def _replace_relations(connection: Any, source_id: str, relations: tuple[tuple[str, str], ...], now: str) -> None:
        connection.execute("DELETE FROM judgment_relations WHERE source_id = ?", (source_id,))
        connection.executemany("INSERT INTO judgment_relations VALUES (?, ?, ?, ?)", [(source_id, target, kind, now) for target, kind in relations])

    @staticmethod
    def _sync_retrieval(connection: Any, judgment_id: str, statement: str, rationale: str, provenance: dict[str, Any], relations: tuple[tuple[str, str], ...]) -> None:
        metadata = {"kind": "judgment", "provenance": provenance}
        connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, 1.0) ON CONFLICT(node_id) DO UPDATE SET text=excluded.text, metadata_json=excluded.metadata_json, confidence=1.0",
            (judgment_id, f"{statement}\n\n{rationale}", json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
        )
        connection.execute("DELETE FROM edges WHERE source_id = ?", (judgment_id,))
        connection.executemany("INSERT INTO edges(source_id,target_id,edge_type,weight,factuality) VALUES (?,?,?,1.0,1.0)", [(judgment_id, target, kind) for target, kind in relations])

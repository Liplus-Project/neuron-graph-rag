from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Iterable

from .models import DocumentNode, TypedEdge
from .ontology import RelationTypeRegistry
from .retrieval import BM25Retriever, DenseRetriever, normalize_scores
from .storage import SQLiteStore


_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_RELATION = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class JudgmentContractError(ValueError):
    """A fail-closed domain contract violation."""


class JudgmentGraph:
    """Atomic domain API for the canonical SQLite judgment graph."""

    def __init__(
        self,
        store: SQLiteStore,
        *,
        sparse_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_weight: float = 0.55,
        dense_weight: float = 0.45,
        use_dense_retrieval: bool = True,
    ) -> None:
        self.store = store
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.use_dense_retrieval = use_dense_retrieval
        self.relation_types = RelationTypeRegistry(store)

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
            if row["lifecycle"] == lifecycle:
                raise JudgmentContractError(f"judgment is already {lifecycle}")
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
        return self.get_judgment(judgment_id)

    def get_judgment(self, judgment_id: str) -> dict[str, Any]:
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
            """
            SELECT target_id, relation_type, relation_type_revision, assertion_kind
            FROM judgment_relations WHERE source_id = ?
            ORDER BY target_id, relation_type
            """,
            (judgment_id,),
        ).fetchall()
        warnings = []
        seen_warning_types: set[str] = set()
        for relation in relations:
            relation_type = relation["relation_type"]
            if relation_type in seen_warning_types:
                continue
            relation_warnings = self.relation_types.validate(relation_type)
            if relation_warnings:
                seen_warning_types.add(relation_type)
                warnings.extend(relation_warnings)
        return {
            "judgment_id": row["judgment_id"], "revision": row["current_revision"],
            "statement": row["statement"], "rationale": row["rationale"],
            "provenance": json.loads(row["provenance_json"]), "lifecycle": row["lifecycle"],
            "superseded_by": row["superseded_by"],
            "relations": [dict(item) for item in relations],
            "advisory_warnings": warnings,
        }

    def register_relation_type(
        self,
        relation_type: str,
        definition: str,
        namespace: str,
        provenance: dict[str, Any],
        *,
        expected_revision: int,
        lifecycle: str = "active",
    ) -> dict[str, Any]:
        return self.relation_types.register(
            relation_type,
            definition,
            namespace,
            provenance,
            expected_revision=expected_revision,
            lifecycle=lifecycle,
        )

    def get_relation_type(
        self, relation_type: str, *, revision: int | None = None
    ) -> dict[str, Any]:
        return self.relation_types.get(relation_type, revision=revision)

    def list_relation_types(
        self, *, include_deprecated: bool = True
    ) -> list[dict[str, Any]]:
        return self.relation_types.list(include_deprecated=include_deprecated)

    def validate_relation_type(self, relation_type: str) -> list[dict[str, Any]]:
        return self.relation_types.validate(relation_type)

    def search_judgments(
        self,
        query: str,
        *,
        limit: int = 5,
        include_archived: bool = False,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        query = self._text(query, "query")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise JudgmentContractError("limit must be an integer from 1 through 100")
        if not isinstance(include_archived, bool):
            raise JudgmentContractError("include_archived must be a boolean")
        repository = self._repository(repository)

        rows = self.store.connection.execute(
            "SELECT judgment_id FROM judgments ORDER BY judgment_id"
        ).fetchall()
        candidates: list[tuple[dict[str, Any], DocumentNode]] = []
        for row in rows:
            judgment = self.get_judgment(row["judgment_id"])
            if judgment["lifecycle"] != "active" and not include_archived:
                continue
            if repository is not None and not self._repository_matches(
                judgment["provenance"], repository
            ):
                continue
            candidates.append((judgment, self.store.get_node(judgment["judgment_id"])))
        if not candidates:
            return []

        nodes = [node for _, node in candidates]
        sparse_raw = self.sparse_retriever.score(query, nodes)
        dense_raw = (
            self.dense_retriever.score(query, nodes)
            if self.use_dense_retrieval
            else {node.node_id: 0.0 for node in nodes}
        )
        sparse = normalize_scores(sparse_raw)
        dense = normalize_scores(dense_raw)
        weight_total = self.sparse_weight + self.dense_weight
        results: list[dict[str, Any]] = []
        for judgment, node in candidates:
            score = (
                sparse[node.node_id] * self.sparse_weight
                + dense[node.node_id] * self.dense_weight
            ) / weight_total
            results.append(
                {
                    **judgment,
                    "score": score,
                    "explanation": {
                        "sparse_score": sparse[node.node_id],
                        "dense_score": dense[node.node_id],
                        "sparse_weight": self.sparse_weight,
                        "dense_weight": self.dense_weight,
                    },
                }
            )
        results.sort(key=lambda item: (-item["score"], item["judgment_id"]))
        return results[:limit]

    def traverse_judgments(
        self,
        judgment_id: str,
        *,
        direction: str = "outgoing",
        relation_type: str | None = None,
        max_hops: int = 1,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        root = self.get_judgment(judgment_id)
        if direction not in {"incoming", "outgoing", "both"}:
            raise JudgmentContractError("direction must be incoming, outgoing, or both")
        if relation_type is not None:
            if not isinstance(relation_type, str) or _RELATION.fullmatch(relation_type) is None:
                raise JudgmentContractError("invalid relation type")
        if (
            isinstance(max_hops, bool)
            or not isinstance(max_hops, int)
            or not 1 <= max_hops <= 32
        ):
            raise JudgmentContractError("max_hops must be an integer from 1 through 32")
        if not isinstance(include_archived, bool):
            raise JudgmentContractError("include_archived must be a boolean")
        if root["lifecycle"] != "active" and not include_archived:
            raise JudgmentContractError("archived root requires include_archived")

        visited = {root["judgment_id"]}
        frontier = [root["judgment_id"]]
        traversed: list[dict[str, Any]] = []
        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for current_id in frontier:
                edges = self._traversal_edges(
                    current_id, direction=direction, relation_type=relation_type
                )
                for (
                    source_id,
                    target_id,
                    kind,
                    type_revision,
                    assertion_kind,
                    edge_direction,
                    neighbor_id,
                ) in edges:
                    if neighbor_id in visited:
                        continue
                    judgment = self.get_judgment(neighbor_id)
                    if judgment["lifecycle"] != "active" and not include_archived:
                        continue
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    traversed.append(
                        {
                            "hop": hop,
                            "direction": edge_direction,
                            "relation": {
                                "source_id": source_id,
                                "target_id": target_id,
                                "relation_type": kind,
                                "relation_type_revision": type_revision,
                                "assertion_kind": assertion_kind,
                            },
                            "judgment": judgment,
                        }
                    )
            frontier = sorted(next_frontier)
            if not frontier:
                break
        traversed.sort(
            key=lambda item: (
                item["hop"],
                item["judgment"]["judgment_id"],
                item["relation"]["source_id"],
                item["relation"]["target_id"],
                item["relation"]["relation_type"],
                item["direction"],
            )
        )
        return traversed

    def _traversal_edges(
        self,
        judgment_id: str,
        *,
        direction: str,
        relation_type: str | None,
    ) -> list[tuple[str, str, str, int | None, str, str, str]]:
        edges: list[tuple[str, str, str, int | None, str, str, str]] = []
        if direction in {"outgoing", "both"}:
            rows = self.store.connection.execute(
                """
                SELECT source_id, target_id, relation_type,
                       relation_type_revision, assertion_kind
                FROM judgment_relations WHERE source_id = ?
                ORDER BY target_id, relation_type
                """,
                (judgment_id,),
            ).fetchall()
            edges.extend(
                (
                    row["source_id"], row["target_id"], row["relation_type"],
                    row["relation_type_revision"], row["assertion_kind"],
                    "outgoing", row["target_id"],
                )
                for row in rows
                if relation_type is None or row["relation_type"] == relation_type
            )
        if direction in {"incoming", "both"}:
            rows = self.store.connection.execute(
                """
                SELECT source_id, target_id, relation_type,
                       relation_type_revision, assertion_kind
                FROM judgment_relations WHERE target_id = ?
                ORDER BY source_id, relation_type
                """,
                (judgment_id,),
            ).fetchall()
            edges.extend(
                (
                    row["source_id"], row["target_id"], row["relation_type"],
                    row["relation_type_revision"], row["assertion_kind"],
                    "incoming", row["source_id"],
                )
                for row in rows
                if relation_type is None or row["relation_type"] == relation_type
            )
        return sorted(edges, key=lambda item: (item[6], item[2], item[5], item[0], item[1]))

    @staticmethod
    def _repository(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise JudgmentContractError("repository must be a string")
        normalized = value.strip()
        if not normalized or len(normalized) > 256 or any(ord(character) < 32 for character in normalized):
            raise JudgmentContractError("repository is invalid")
        return normalized

    @staticmethod
    def _repository_matches(provenance: dict[str, Any], repository: str) -> bool:
        stored = provenance.get("repository")
        if not isinstance(stored, str):
            return False
        return stored == repository or stored.rsplit("/", 1)[-1] == repository

    def export(self) -> dict[str, Any]:
        ids = [
            row[0]
            for row in self.store.connection.execute(
                "SELECT judgment_id FROM judgments ORDER BY judgment_id"
            )
        ]
        relation_types = []
        for registry in self.store.connection.execute(
            "SELECT * FROM judgment_relation_types ORDER BY relation_type"
        ):
            revisions = []
            for revision in self.store.connection.execute(
                """
                SELECT * FROM judgment_relation_type_revisions
                WHERE relation_type = ? ORDER BY revision
                """,
                (registry["relation_type"],),
            ):
                revisions.append(
                    {
                        "revision": revision["revision"],
                        "definition": revision["definition"],
                        "namespace": revision["namespace"],
                        "provenance": json.loads(revision["provenance_json"]),
                        "lifecycle": revision["lifecycle"],
                        "created_at": revision["created_at"],
                    }
                )
            relation_types.append(
                {
                    "relation_type": registry["relation_type"],
                    "current_revision": registry["current_revision"],
                    "lifecycle": registry["lifecycle"],
                    "created_at": registry["created_at"],
                    "updated_at": registry["updated_at"],
                    "revisions": revisions,
                }
            )
        return {
            "format": "ngr-judgment-graph/v2",
            "relation_types": relation_types,
            "judgments": [self.get(item) for item in ids],
        }

    def import_graph(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("format") not in {
            "ngr-judgment-graph/v1",
            "ngr-judgment-graph/v2",
        }:
            raise JudgmentContractError("unsupported judgment graph format")
        format_version = payload["format"]
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
        relation_type_records = payload.get("relation_types", [])
        if format_version == "ngr-judgment-graph/v2" and not isinstance(
            relation_type_records, list
        ):
            raise JudgmentContractError("relation_types must be a list")
        now = self._now()
        with self.store.transaction() as connection:
            if format_version == "ngr-judgment-graph/v2":
                self._import_relation_types(connection, relation_type_records)
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
                relation_payload = record.get("relations", [])
                relations = self._relations(
                    [
                        {
                            "target_id": relation["target_id"],
                            "relation_type": relation["relation_type"],
                        }
                        for relation in relation_payload
                    ],
                    source_id=record["judgment_id"],
                )
                self._require_targets(connection, relations)
                bindings = None
                if format_version == "ngr-judgment-graph/v2":
                    bindings = self._import_relation_bindings(
                        connection, relation_payload
                    )
                self._replace_relations(
                    connection,
                    record["judgment_id"],
                    relations,
                    now,
                    bindings=bindings,
                )
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
    def _import_relation_types(connection: Any, records: list[Any]) -> None:
        seen: set[str] = set()
        if any(not isinstance(record, dict) for record in records):
            raise JudgmentContractError("relation type record must be an object")
        for record in sorted(records, key=lambda item: item.get("relation_type", "")):
            relation_type = record.get("relation_type")
            if not isinstance(relation_type, str) or _RELATION.fullmatch(relation_type) is None:
                raise JudgmentContractError("invalid relation type")
            if relation_type in seen:
                raise JudgmentContractError("relation type identities must be unique")
            seen.add(relation_type)
            current_revision = record.get("current_revision")
            lifecycle = record.get("lifecycle")
            created_at = record.get("created_at")
            updated_at = record.get("updated_at")
            revisions = record.get("revisions")
            if (
                isinstance(current_revision, bool)
                or not isinstance(current_revision, int)
                or current_revision < 1
                or lifecycle not in {"active", "deprecated"}
                or not isinstance(created_at, str)
                or not created_at
                or not isinstance(updated_at, str)
                or not updated_at
                or not isinstance(revisions, list)
            ):
                raise JudgmentContractError("invalid relation type registry record")
            existing = connection.execute(
                "SELECT * FROM judgment_relation_types WHERE relation_type = ?",
                (relation_type,),
            ).fetchone()
            existing_revision_numbers = {
                row["revision"]
                for row in connection.execute(
                    """
                    SELECT revision FROM judgment_relation_type_revisions
                    WHERE relation_type = ?
                    """,
                    (relation_type,),
                )
            }
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO judgment_relation_types(
                        relation_type, current_revision, lifecycle, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        relation_type,
                        current_revision,
                        lifecycle,
                        created_at,
                        updated_at,
                    ),
                )
            revision_numbers: set[int] = set()
            for revision in sorted(revisions, key=lambda item: item.get("revision", 0)):
                if not isinstance(revision, dict):
                    raise JudgmentContractError("relation type revision must be an object")
                number = revision.get("revision")
                definition = revision.get("definition")
                namespace = revision.get("namespace")
                provenance = revision.get("provenance")
                revision_lifecycle = revision.get("lifecycle")
                revision_created_at = revision.get("created_at")
                if (
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or number < 1
                    or number in revision_numbers
                    or not isinstance(definition, str)
                    or not definition.strip()
                    or not isinstance(namespace, str)
                    or not namespace
                    or not isinstance(provenance, dict)
                    or not provenance
                    or revision_lifecycle not in {"active", "deprecated"}
                    or not isinstance(revision_created_at, str)
                    or not revision_created_at
                ):
                    raise JudgmentContractError("invalid relation type revision")
                revision_numbers.add(number)
                encoded = json.dumps(
                    provenance,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                existing_revision = connection.execute(
                    """
                    SELECT definition, namespace, provenance_json, lifecycle, created_at
                    FROM judgment_relation_type_revisions
                    WHERE relation_type = ? AND revision = ?
                    """,
                    (relation_type, number),
                ).fetchone()
                values = (
                    definition.strip(),
                    namespace,
                    encoded,
                    revision_lifecycle,
                    revision_created_at,
                )
                if existing_revision is None:
                    connection.execute(
                        """
                        INSERT INTO judgment_relation_type_revisions(
                            relation_type, revision, definition, namespace,
                            provenance_json, lifecycle, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (relation_type, number, *values),
                    )
                elif tuple(existing_revision) != values:
                    raise JudgmentContractError(
                        f"relation type revision conflict: {relation_type}@{number}"
                    )
            if current_revision not in revision_numbers:
                raise JudgmentContractError("current relation type revision is missing")
            if revision_numbers != set(range(1, current_revision + 1)):
                raise JudgmentContractError(
                    "relation type revisions must be contiguous through current_revision"
                )
            if not existing_revision_numbers <= revision_numbers:
                raise JudgmentContractError(
                    "destination has newer relation type revision history"
                )
            if existing is not None:
                expected_registry = (
                    current_revision,
                    lifecycle,
                    created_at,
                    updated_at,
                )
                actual_registry = (
                    existing["current_revision"],
                    existing["lifecycle"],
                    existing["created_at"],
                    existing["updated_at"],
                )
                if existing["current_revision"] == current_revision:
                    if actual_registry != expected_registry:
                        raise JudgmentContractError(
                            f"relation type registry conflict: {relation_type}"
                        )
                elif existing["current_revision"] < current_revision:
                    if existing["created_at"] != created_at:
                        raise JudgmentContractError(
                            f"relation type registry origin conflict: {relation_type}"
                        )
                    connection.execute(
                        """
                        UPDATE judgment_relation_types
                        SET current_revision = ?, lifecycle = ?, created_at = ?, updated_at = ?
                        WHERE relation_type = ?
                        """,
                        (*expected_registry, relation_type),
                    )
                else:
                    raise JudgmentContractError(
                        "destination has newer relation type registry state"
                    )

    @staticmethod
    def _import_relation_bindings(
        connection: Any, relations: list[Any]
    ) -> dict[tuple[str, str], tuple[int | None, str]]:
        bindings: dict[tuple[str, str], tuple[int | None, str]] = {}
        for relation in relations:
            if not isinstance(relation, dict):
                raise JudgmentContractError("relation must be an object")
            revision = relation.get("relation_type_revision")
            assertion_kind = relation.get("assertion_kind")
            if (
                (revision is not None and (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 1
                ))
                or assertion_kind != "explicit"
            ):
                raise JudgmentContractError("invalid relation binding")
            relation_type = relation["relation_type"]
            if revision is not None and connection.execute(
                """
                SELECT 1 FROM judgment_relation_type_revisions
                WHERE relation_type = ? AND revision = ?
                """,
                (relation_type, revision),
            ).fetchone() is None:
                raise JudgmentContractError("unknown relation type revision binding")
            bindings[(relation["target_id"], relation_type)] = (
                revision,
                assertion_kind,
            )
        return bindings

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
    def _replace_relations(
        connection: Any,
        source_id: str,
        relations: tuple[tuple[str, str], ...],
        now: str,
        *,
        bindings: dict[tuple[str, str], tuple[int | None, str]] | None = None,
    ) -> None:
        connection.execute("DELETE FROM judgment_relations WHERE source_id = ?", (source_id,))
        rows = []
        for target, kind in relations:
            if bindings is None:
                registry = connection.execute(
                    """
                    SELECT current_revision FROM judgment_relation_types
                    WHERE relation_type = ?
                    """,
                    (kind,),
                ).fetchone()
                revision = None if registry is None else registry["current_revision"]
                assertion_kind = "explicit"
            else:
                revision, assertion_kind = bindings[(target, kind)]
            rows.append(
                (source_id, target, kind, revision, assertion_kind, now)
            )
        connection.executemany(
            """
            INSERT INTO judgment_relations(
                source_id, target_id, relation_type, relation_type_revision,
                assertion_kind, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    @staticmethod
    def _sync_retrieval(connection: Any, judgment_id: str, statement: str, rationale: str, provenance: dict[str, Any], relations: tuple[tuple[str, str], ...]) -> None:
        metadata = {"kind": "judgment", "provenance": provenance}
        connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, 1.0) ON CONFLICT(node_id) DO UPDATE SET text=excluded.text, metadata_json=excluded.metadata_json, confidence=1.0",
            (judgment_id, f"{statement}\n\n{rationale}", json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
        )
        connection.execute("DELETE FROM edges WHERE source_id = ?", (judgment_id,))
        connection.executemany("INSERT INTO edges(source_id,target_id,edge_type,weight,factuality) VALUES (?,?,?,1.0,1.0)", [(judgment_id, target, kind) for target, kind in relations])

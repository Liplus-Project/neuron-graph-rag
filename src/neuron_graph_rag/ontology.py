from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from .storage import SQLiteStore


_RELATION = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


class RelationTypeContractError(ValueError):
    """A fail-closed relation type registry contract violation."""


class RelationTypeRegistry:
    """Atomic domain API for judgment relation semantics."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _relation_type(value: str) -> str:
        if not isinstance(value, str) or _RELATION.fullmatch(value) is None:
            raise RelationTypeContractError("invalid relation type")
        return value

    @staticmethod
    def _text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RelationTypeContractError(f"{name} is required")
        return value.strip()

    @staticmethod
    def _namespace(value: str) -> str:
        if not isinstance(value, str) or _NAMESPACE.fullmatch(value) is None:
            raise RelationTypeContractError("invalid namespace")
        return value

    @staticmethod
    def _provenance(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise RelationTypeContractError("provenance must be a non-empty object")
        try:
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise RelationTypeContractError(
                "provenance must be deterministic JSON"
            ) from error
        return value

    def register(
        self,
        relation_type: str,
        definition: str,
        namespace: str,
        provenance: dict[str, Any],
        *,
        expected_revision: int,
        lifecycle: str = "active",
    ) -> dict[str, Any]:
        relation_type = self._relation_type(relation_type)
        definition = self._text(definition, "definition")
        namespace = self._namespace(namespace)
        provenance = self._provenance(provenance)
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise RelationTypeContractError(
                "expected_revision must be a non-negative integer"
            )
        if lifecycle not in {"active", "deprecated"}:
            raise RelationTypeContractError("invalid relation type lifecycle")
        now = self._now()
        encoded_provenance = json.dumps(
            provenance,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self.store.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM judgment_relation_types WHERE relation_type = ?",
                (relation_type,),
            ).fetchone()
            if current is None:
                if expected_revision != 0:
                    raise RelationTypeContractError("stale expected_revision")
                revision = 1
                connection.execute(
                    """
                    INSERT INTO judgment_relation_types(
                        relation_type, current_revision, lifecycle, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (relation_type, revision, lifecycle, now, now),
                )
            else:
                if current["current_revision"] != expected_revision:
                    raise RelationTypeContractError("stale expected_revision")
                revision = expected_revision + 1
                connection.execute(
                    """
                    UPDATE judgment_relation_types
                    SET current_revision = ?, lifecycle = ?, updated_at = ?
                    WHERE relation_type = ?
                    """,
                    (revision, lifecycle, now, relation_type),
                )
            connection.execute(
                """
                INSERT INTO judgment_relation_type_revisions(
                    relation_type, revision, definition, namespace,
                    provenance_json, lifecycle, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_type,
                    revision,
                    definition,
                    namespace,
                    encoded_provenance,
                    lifecycle,
                    now,
                ),
            )
        return self.get(relation_type)

    def get(
        self, relation_type: str, *, revision: int | None = None
    ) -> dict[str, Any]:
        relation_type = self._relation_type(relation_type)
        current = self.store.connection.execute(
            "SELECT * FROM judgment_relation_types WHERE relation_type = ?",
            (relation_type,),
        ).fetchone()
        if current is None:
            raise KeyError(f"Unknown relation type: {relation_type}")
        selected_revision = current["current_revision"] if revision is None else revision
        if (
            isinstance(selected_revision, bool)
            or not isinstance(selected_revision, int)
            or selected_revision < 1
        ):
            raise RelationTypeContractError("revision must be a positive integer")
        row = self.store.connection.execute(
            """
            SELECT * FROM judgment_relation_type_revisions
            WHERE relation_type = ? AND revision = ?
            """,
            (relation_type, selected_revision),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Unknown relation type revision: {relation_type}@{selected_revision}"
            )
        return {
            "relation_type": row["relation_type"],
            "revision": row["revision"],
            "definition": row["definition"],
            "namespace": row["namespace"],
            "provenance": json.loads(row["provenance_json"]),
            "lifecycle": row["lifecycle"],
            "created_at": row["created_at"],
            "is_current": row["revision"] == current["current_revision"],
        }

    def list(
        self, *, include_deprecated: bool = True
    ) -> list[dict[str, Any]]:
        if not isinstance(include_deprecated, bool):
            raise RelationTypeContractError("include_deprecated must be a boolean")
        query = "SELECT relation_type FROM judgment_relation_types"
        parameters: tuple[object, ...] = ()
        if not include_deprecated:
            query += " WHERE lifecycle = ?"
            parameters = ("active",)
        query += " ORDER BY relation_type"
        return [
            self.get(row["relation_type"])
            for row in self.store.connection.execute(query, parameters)
        ]

    def revisions(self, relation_type: str) -> list[dict[str, Any]]:
        relation_type = self._relation_type(relation_type)
        if self.store.connection.execute(
            "SELECT 1 FROM judgment_relation_types WHERE relation_type = ?",
            (relation_type,),
        ).fetchone() is None:
            raise KeyError(f"Unknown relation type: {relation_type}")
        revisions = self.store.connection.execute(
            """
            SELECT revision FROM judgment_relation_type_revisions
            WHERE relation_type = ? ORDER BY revision
            """,
            (relation_type,),
        )
        return [self.get(relation_type, revision=row["revision"]) for row in revisions]

    def validate(self, relation_type: str) -> list[dict[str, Any]]:
        relation_type = self._relation_type(relation_type)
        row = self.store.connection.execute(
            """
            SELECT current_revision, lifecycle FROM judgment_relation_types
            WHERE relation_type = ?
            """,
            (relation_type,),
        ).fetchone()
        if row is None:
            return [
                {
                    "code": "unknown_relation_type",
                    "message": "relation type is not registered; the assertion remains advisory and is preserved",
                    "relation_type": relation_type,
                    "relation_type_revision": None,
                    "lifecycle": "unknown",
                }
            ]
        if row["lifecycle"] == "deprecated":
            return [
                {
                    "code": "deprecated_relation_type",
                    "message": "relation type is deprecated; the assertion remains advisory and is preserved",
                    "relation_type": relation_type,
                    "relation_type_revision": row["current_revision"],
                    "lifecycle": "deprecated",
                }
            ]
        return []

    def current_revision(self, relation_type: str) -> int | None:
        relation_type = self._relation_type(relation_type)
        row = self.store.connection.execute(
            """
            SELECT current_revision FROM judgment_relation_types
            WHERE relation_type = ?
            """,
            (relation_type,),
        ).fetchone()
        return None if row is None else int(row["current_revision"])

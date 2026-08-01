from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import NeuronGraphRAG


FIXTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FixtureLoadResult:
    node_count: int
    edge_count: int


def read_fixture(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported fixture schema version: {fixture.get('schema_version')!r}"
        )
    if not isinstance(fixture.get("nodes"), list) or not isinstance(
        fixture.get("edges"), list
    ):
        raise ValueError("Fixture must contain nodes and edges arrays")
    return fixture


def load_fixture(
    engine: NeuronGraphRAG, path: str | Path
) -> FixtureLoadResult:
    fixture = read_fixture(path)
    node_ids: set[str] = set()
    for record in fixture["nodes"]:
        node_id = str(record["node_id"])
        if node_id in node_ids:
            raise ValueError(f"Duplicate fixture node_id: {node_id}")
        node_ids.add(node_id)
        engine.add_document(
            node_id,
            str(record["text"]),
            metadata=dict(record.get("metadata", {})),
            confidence=float(record.get("confidence", 1.0)),
        )

    edge_keys: set[tuple[str, str, str]] = set()
    for record in fixture["edges"]:
        source_id = str(record["source_id"])
        target_id = str(record["target_id"])
        edge_type = str(record["edge_type"])
        if source_id not in node_ids or target_id not in node_ids:
            raise ValueError(
                "Fixture edge endpoints must both be present: "
                f"{source_id} -> {target_id}"
            )
        key = (source_id, target_id, edge_type)
        if key in edge_keys:
            raise ValueError(f"Duplicate fixture edge: {key!r}")
        edge_keys.add(key)
        engine.add_edge(
            source_id,
            target_id,
            edge_type,
            weight=float(record.get("weight", 1.0)),
            factuality=float(record.get("factuality", 1.0)),
        )

    return FixtureLoadResult(len(node_ids), len(edge_keys))

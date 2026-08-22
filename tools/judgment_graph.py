from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_graph(database: Path, output: Path) -> None:
    with NeuronGraphRAG(database) as engine:
        _write_json(output, engine.judgments.export())


def import_graph(database: Path, source: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    with NeuronGraphRAG(database) as engine:
        engine.judgments.import_graph(payload)


def backup(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backup: {destination}")
    with closing(sqlite3.connect(source)) as source_connection, closing(sqlite3.connect(destination)) as destination_connection:
        source_connection.backup(destination_connection)


def restore(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite database: {destination}")
    with closing(sqlite3.connect(source)) as source_connection, closing(sqlite3.connect(destination)) as destination_connection:
        if source_connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("backup integrity check failed")
        source_connection.backup(destination_connection)


def integrity(database: Path) -> dict[str, object]:
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        sqlite_status = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = [dict(row) for row in connection.execute("PRAGMA foreign_key_check")]
        dangling = connection.execute(
            """
            SELECT source_id, target_id, relation_type FROM judgment_relations
            WHERE source_id NOT IN (SELECT judgment_id FROM judgments)
               OR target_id NOT IN (SELECT judgment_id FROM judgments)
            ORDER BY source_id, target_id, relation_type
            """
        ).fetchall()
        duplicate_successors = connection.execute(
            """
            SELECT superseded_by, COUNT(*) AS predecessor_count FROM judgments
            WHERE superseded_by IS NOT NULL GROUP BY superseded_by HAVING COUNT(*) > 1
            """
        ).fetchall()
    result = {
        "sqlite_integrity": sqlite_status,
        "foreign_key_violations": foreign_keys,
        "dangling_relations": [dict(row) for row in dangling],
        "duplicate_successors": [dict(row) for row in duplicate_successors],
    }
    if sqlite_status != "ok" or foreign_keys or dangling or duplicate_successors:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Operate the canonical SQLite judgment graph")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "import", "backup", "restore"):
        command = subparsers.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("destination", type=Path)
    check = subparsers.add_parser("integrity")
    check.add_argument("database", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "export":
        export_graph(arguments.source, arguments.destination)
    elif arguments.command == "import":
        import_graph(arguments.destination, arguments.source)
    elif arguments.command == "backup":
        backup(arguments.source, arguments.destination)
    elif arguments.command == "restore":
        restore(arguments.source, arguments.destination)
    else:
        print(json.dumps(integrity(arguments.database), sort_keys=True))


if __name__ == "__main__":
    main()

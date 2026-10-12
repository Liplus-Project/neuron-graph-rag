from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.decision_wiki_import import WikiSource
from neuron_graph_rag.decision_wiki_incremental import apply_plan, build_plan, public_plan

try:
    from tools.import_decision_wikis import _commit
    from tools.judgment_graph import backup, integrity
except ModuleNotFoundError:  # Direct script execution adds tools/ rather than the repository root.
    from import_decision_wikis import _commit
    from judgment_graph import backup, integrity


def _read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)


def dry_run(sources: Iterable[WikiSource], database: Path) -> dict[str, Any]:
    """Plan against a read-only connection. Nothing is written."""
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")
    with closing(_read_only(database)) as connection:
        return public_plan(build_plan(connection, sources))


def backup_path_for(database: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return database.with_name(f"{database.stem}.pre-wiki-increment-{stamp}{database.suffix}")


def write(sources: Iterable[WikiSource], database: Path) -> dict[str, Any]:
    """Back up the database next to itself, then apply the plan in one transaction.

    The database file is opened in place; it is never rebuilt or replaced. Any failure,
    including the post-write integrity check, rolls the whole increment back.
    """
    sources = tuple(sources)
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")
    with closing(_read_only(database)) as connection:
        preview = build_plan(connection, sources)
    if not preview["operations"]:
        return {**public_plan(preview), "written": [], "backup": None}
    backup_file = backup_path_for(database)
    backup(database, backup_file)
    with closing(sqlite3.connect(backup_file)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"backup integrity check failed: {backup_file}")
    with NeuronGraphRAG(database) as engine:
        with engine.store.transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")  # plan and writes see one state
            plan = build_plan(connection, sources)
            written = apply_plan(engine.judgments, plan)
            integrity(connection)
    return {**public_plan(plan), "written": written, "backup": str(backup_file)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Decision Structure Wiki changes into an existing judgment store. "
        "Dry run by default; --write applies the plan after a SQLite backup."
    )
    parser.add_argument("--liplus-wiki", type=Path, required=True)
    parser.add_argument("--ngr-wiki", type=Path, required=True)
    parser.add_argument("--liplus-index", type=Path)
    parser.add_argument("--ngr-index", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="apply the plan (default: dry run)")
    arguments = parser.parse_args(argv)
    sources = (
        WikiSource("Liplus-Project/liplus-language", arguments.liplus_wiki, arguments.liplus_index or arguments.liplus_wiki / "Decision-Structure.md", _commit(arguments.liplus_wiki)),
        WikiSource("Liplus-Project/neuron-graph-rag", arguments.ngr_wiki, arguments.ngr_index or arguments.ngr_wiki / "Decision-Structure.md", _commit(arguments.ngr_wiki)),
    )
    try:
        result = write(sources, arguments.database) if arguments.write else dry_run(sources, arguments.database)
    except FileNotFoundError as error:
        parser.error(str(error))
    result["mode"] = "write" if arguments.write else "dry-run"
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

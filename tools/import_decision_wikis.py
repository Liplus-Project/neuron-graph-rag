from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Callable, Iterable

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.decision_wiki_import import (
    WikiSource,
    build_payload,
    deterministic_json,
    import_atomically,
)
try:
    from tools.judgment_graph import backup, integrity
except ModuleNotFoundError:  # Direct script execution adds tools/ rather than the repository root.
    from judgment_graph import backup, integrity


def _commit(clone: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    temporary.unlink()


def publish_bundle(
    sources: Iterable[WikiSource],
    database: Path,
    export_path: Path,
    backup_path: Path,
    manifest_path: Path,
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> None:
    outputs = (database, export_path, backup_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output: " + ", ".join(existing))
    temporary = {path: _temporary_sibling(path) for path in outputs}
    published: list[Path] = []
    try:
        payload, manifest = build_payload(sources)
        import_atomically(temporary[database], payload)
        check = integrity(temporary[database])
        if failure_hook:
            failure_hook("after_database")

        with NeuronGraphRAG(temporary[database]) as engine:
            first_export = deterministic_json(engine.judgments.export())
            second_export = deterministic_json(engine.judgments.export())
        if first_export != second_export:
            raise RuntimeError("deterministic export verification failed")
        temporary[export_path].write_bytes(first_export)
        if failure_hook:
            failure_hook("after_export")

        backup(temporary[database], temporary[backup_path])
        with closing(sqlite3.connect(temporary[backup_path])) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("backup integrity check failed")
        if failure_hook:
            failure_hook("after_backup")

        manifest["integrity"] = check
        manifest["export_sha256"] = hashlib.sha256(first_export).hexdigest()
        temporary[manifest_path].write_bytes(deterministic_json(manifest))
        if failure_hook:
            failure_hook("before_publish")

        for destination in outputs:
            _publish_exclusive(temporary[destination], destination)
            published.append(destination)
            if failure_hook:
                failure_hook(f"published:{destination.name}")
    except BaseException:
        for path in published:
            if path.exists():
                path.unlink()
        raise
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import indexed Decision Structure Wiki entries")
    parser.add_argument("--liplus-wiki", type=Path, required=True)
    parser.add_argument("--ngr-wiki", type=Path, required=True)
    parser.add_argument("--liplus-index", type=Path)
    parser.add_argument("--ngr-index", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--export", dest="export_path", type=Path, required=True)
    parser.add_argument("--backup", dest="backup_path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    sources = (
        WikiSource("Liplus-Project/liplus-language", arguments.liplus_wiki, arguments.liplus_index or arguments.liplus_wiki / "Decision-Structure.md", _commit(arguments.liplus_wiki)),
        WikiSource("Liplus-Project/neuron-graph-rag", arguments.ngr_wiki, arguments.ngr_index or arguments.ngr_wiki / "Decision-Structure.md", _commit(arguments.ngr_wiki)),
    )
    try:
        publish_bundle(
            sources, arguments.database, arguments.export_path,
            arguments.backup_path, arguments.manifest,
        )
    except FileExistsError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

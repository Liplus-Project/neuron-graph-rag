from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.decision_wiki_import import (
    WikiSource,
    build_payload,
    deterministic_json,
    import_atomically,
)
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
    outputs = (arguments.database, arguments.export_path, arguments.backup_path, arguments.manifest)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error("refusing to overwrite existing output: " + ", ".join(existing))
    sources = (
        WikiSource("Liplus-Project/liplus-language", arguments.liplus_wiki, arguments.liplus_index or arguments.liplus_wiki / "Decision-Structure.md", _commit(arguments.liplus_wiki)),
        WikiSource("Liplus-Project/neuron-graph-rag", arguments.ngr_wiki, arguments.ngr_index or arguments.ngr_wiki / "Decision-Structure.md", _commit(arguments.ngr_wiki)),
    )
    payload, manifest = build_payload(sources)
    import_atomically(arguments.database, payload)
    check = integrity(arguments.database)
    with NeuronGraphRAG(arguments.database) as engine:
        exported = deterministic_json(engine.judgments.export())
    if exported != deterministic_json(json.loads(exported)):
        raise RuntimeError("deterministic export verification failed")
    arguments.export_path.write_bytes(exported)
    backup(arguments.database, arguments.backup_path)
    with closing(sqlite3.connect(arguments.backup_path)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("backup integrity check failed")
    manifest["integrity"] = check
    manifest["export_sha256"] = __import__("hashlib").sha256(exported).hexdigest()
    arguments.manifest.write_bytes(deterministic_json(manifest))


if __name__ == "__main__":
    main()

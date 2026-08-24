from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.database_home import (
    DatabaseResolution,
    migrate_database,
    prepare_database,
    resolve_database,
    resolve_database_path,
)
from neuron_graph_rag.storage import (
    FILE_BUSY_TIMEOUT_MILLISECONDS,
    SQLiteStore,
)


class DatabaseHomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_path_precedence_and_home_expansion(self) -> None:
        environment = {"NGR_DATABASE": "~/environment.db"}
        command_line = resolve_database(
            "~/command-line.db", environ=environment, home=self.home
        )
        from_environment = resolve_database(None, environ=environment, home=self.home)
        default = resolve_database(None, environ={}, home=self.home)

        self.assertEqual(command_line.path, self.home / "command-line.db")
        self.assertEqual(command_line.source, "command_line")
        self.assertEqual(from_environment.path, self.home / "environment.db")
        self.assertEqual(from_environment.source, "environment")
        self.assertEqual(default.path, self.home / ".ngrdb" / "knowledge.db")
        self.assertEqual(default.source, "default")
        self.assertEqual(
            resolve_database_path(None, environ={"NGR_DATABASE": ""}, home=self.home),
            default.path,
        )

    def test_only_default_resolution_creates_database_home(self) -> None:
        default = resolve_database(None, environ={}, home=self.home)
        explicit = DatabaseResolution(self.root / "custom" / "knowledge.db", "command_line")

        self.assertEqual(prepare_database(default), default.path)
        self.assertTrue(default.path.parent.is_dir())
        self.assertEqual(prepare_database(explicit), explicit.path)
        self.assertFalse(explicit.path.parent.exists())

    def test_file_backed_store_uses_wal_timeout_and_foreign_keys(self) -> None:
        database = self.root / "policy.sqlite"
        with SQLiteStore(database) as store:
            self.assertEqual(
                store.connection.execute("PRAGMA journal_mode").fetchone()[0],
                "wal",
            )
            self.assertEqual(
                store.connection.execute("PRAGMA busy_timeout").fetchone()[0],
                FILE_BUSY_TIMEOUT_MILLISECONDS,
            )
            self.assertEqual(
                store.connection.execute("PRAGMA foreign_keys").fetchone()[0],
                1,
            )

        with SQLiteStore(":memory:") as memory:
            self.assertEqual(
                memory.connection.execute("PRAGMA journal_mode").fetchone()[0],
                "memory",
            )

    def test_busy_timeout_serializes_two_local_writers(self) -> None:
        database = self.root / "contention.sqlite"
        first = SQLiteStore(database)
        second_ready = threading.Event()
        start_second_write = threading.Event()
        result: list[object] = []

        def write_from_second_connection() -> None:
            try:
                with SQLiteStore(database) as second:
                    second_ready.set()
                    start_second_write.wait(timeout=2)
                    with second.transaction() as connection:
                        connection.execute(
                            "INSERT INTO nodes VALUES (?, ?, ?, ?)",
                            ("second", "second writer", "{}", 1.0),
                        )
                result.append("ok")
            except Exception as error:  # pragma: no cover - failure detail assertion
                result.append(error)

        worker = threading.Thread(target=write_from_second_connection)
        worker.start()
        self.assertTrue(second_ready.wait(timeout=2))
        first.connection.execute("BEGIN IMMEDIATE")
        first.connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?)",
            ("first", "first writer", "{}", 1.0),
        )
        start_second_write.set()
        time.sleep(0.1)
        first.connection.commit()
        worker.join(timeout=2)
        first.close()

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, ["ok"])
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                2,
            )

    def test_migration_creates_checked_backup_and_destination(self) -> None:
        source = self.root / "legacy" / "knowledge.db"
        source.parent.mkdir()
        with NeuronGraphRAG(source) as engine:
            engine.add_document("shared", "shared database home")
        destination = self.root / "new-home" / "knowledge.db"
        backup = self.root / "backup" / "knowledge.before-migration.db"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        result = migrate_database(source, destination, backup)

        self.assertEqual(result.source, source.resolve())
        self.assertTrue(source.exists())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
        self.assertTrue(destination.exists())
        self.assertTrue(backup.exists())
        for database in (source, destination, backup):
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT text FROM nodes WHERE node_id = 'shared'"
                    ).fetchone()[0],
                    "shared database home",
                )

    def test_migration_refuses_existing_outputs_before_writing(self) -> None:
        source = self.root / "source.db"
        with NeuronGraphRAG(source):
            pass
        destination = self.root / "destination.db"
        destination.write_bytes(b"existing destination")
        backup = self.root / "backup.db"

        with self.assertRaises(FileExistsError):
            migrate_database(source, destination, backup)

        self.assertEqual(destination.read_bytes(), b"existing destination")
        self.assertFalse(backup.exists())

    def test_migration_rejects_corrupt_source_without_outputs(self) -> None:
        source = self.root / "corrupt.db"
        source.write_bytes(b"not a sqlite database")
        destination = self.root / "destination.db"
        backup = self.root / "backup.db"

        with self.assertRaises(sqlite3.DatabaseError):
            migrate_database(source, destination, backup)

        self.assertFalse(destination.exists())
        self.assertFalse(backup.exists())

    def test_migration_cli_requires_explicit_paths_and_reports_integrity(self) -> None:
        source = self.root / "source.db"
        with NeuronGraphRAG(source) as engine:
            engine.add_document("cli", "migration CLI")
        destination = self.root / "destination.db"
        backup = self.root / "backup.db"
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(repository / "src"), environment.get("PYTHONPATH", "")],
            )
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(repository / "tools" / "migrate_database.py"),
                "--source",
                str(source),
                "--destination",
                str(destination),
                "--backup",
                str(backup),
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["sqlite_integrity"], "ok")
        self.assertEqual(Path(payload["destination"]), destination.resolve())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag.database_home import prepare_database, resolve_database
from neuron_graph_rag import home_migration as migration
from neuron_graph_rag.user_config import config_path, ensure_config, ngr_home, read_config, resolve_shared, write_config


class SharedHomeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        runtime = patch.object(migration, "_runtime_processes", return_value=False)
        runtime.start()
        self.addCleanup(runtime.stop)

    def source(self):
        source = self.home / ".ngrdb/knowledge.db"
        source.parent.mkdir()
        with closing(sqlite3.connect(source)) as connection:
            connection.execute("CREATE TABLE records (text TEXT)")
            connection.execute("INSERT INTO records VALUES ('migration data')")
            connection.commit()
        return source

    def test_precedence_relative_json_and_working_directory(self):
        write_config({"database": "db/config.db", "port": 8912,
                      "cuda_cache": "cache/shortlist.db", "cuda_e5_snapshot": "models/e5",
                      "cuda_v2_m3_snapshot": "models/v2", "cuda_device": 2}, home=self.home)
        first = resolve_shared(home=self.home, environ={})
        elsewhere = self.home / "elsewhere"
        elsewhere.mkdir()
        original = Path.cwd()
        try:
            os.chdir(elsewhere)
            second = resolve_shared(home=self.home, environ={})
        finally:
            os.chdir(original)
        self.assertEqual(first, second)
        self.assertEqual(first.database, self.home / ".ngr/db/config.db")
        self.assertEqual(first.cuda_cache, self.home / ".ngr/cache/shortlist.db")
        self.assertEqual(first.cuda_identity()["cuda_device"], 2)
        self.assertEqual(resolve_shared(home=self.home, environ={"NGR_PORT": "9000"}).port, 9000)
        explicit = resolve_shared(argparse.Namespace(database="~/explicit.db", port=9001),
                                  home=self.home, environ={"NGR_DATABASE": "~/env.db", "NGR_PORT": "9000"})
        self.assertEqual(explicit.database, self.home / "explicit.db")
        self.assertEqual(explicit.port, 9001)

    def test_invalid_config_rejected_even_when_cli_overrides(self):
        config_path(self.home).parent.mkdir()
        for value in ('{"token":"secret"}', '{"port":true}', '{"port":65536}',
                      '{"database":""}', '{"unknown":1}', '[]', '{broken'):
            config_path(self.home).write_text(value, encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_shared(argparse.Namespace(database="~/explicit.db"), home=self.home)
        with self.assertRaises(ValueError):
            write_config({"cuda_cache": ""}, home=self.home)

    def test_cuda_is_opt_in_and_must_be_complete(self):
        self.assertEqual(resolve_shared(home=self.home, environ={}).cuda_identity(), {})
        write_config({"cuda_cache": "cache.db"}, home=self.home)
        with self.assertRaises(ValueError):
            resolve_shared(home=self.home, environ={})

    def test_new_config_is_created_without_replacing_an_existing_file(self):
        ensure_config(home=self.home)
        self.assertEqual(read_config(self.home), {})
        write_config({"port": 8912}, home=self.home)
        original = config_path(self.home).read_bytes()
        ensure_config(home=self.home)
        self.assertEqual(config_path(self.home).read_bytes(), original)

    def test_legacy_default_refuses_shadow_database_and_explicit_database_is_preserved(self):
        self.source()
        with self.assertRaises(ValueError):
            prepare_database(resolve_database(home=self.home, environ={}))
        self.assertFalse((ngr_home(self.home) / "db").exists())
        explicit = resolve_database("~/managed.db", home=self.home, environ={})
        self.assertEqual(explicit.path, self.home / "managed.db")

    def test_migration_success_and_rerun_keep_source_and_backup(self):
        source = self.source()
        original = source.read_bytes()
        result = migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertTrue(migration.migration_completed(self.home))
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue(Path(result["backup"]).is_file())
        database = resolve_shared(home=self.home, environ={}).database
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(connection.execute("SELECT text FROM records").fetchone()[0], "migration data")
        self.assertEqual(migration.migrate_home(home=self.home, confirm_stopped=True)["status"], "complete")
        self.assertFalse((ngr_home(self.home) / "home-migration.pending").exists())

    def test_migration_requires_stopped_processes_and_sqlite_writers(self):
        source = self.source()
        with self.assertRaises(ValueError):
            migration.migrate_home(home=self.home)
        with patch.object(migration, "_runtime_processes", return_value=True):
            with self.assertRaises(RuntimeError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        writer = sqlite3.connect(source)
        try:
            writer.execute("BEGIN IMMEDIATE")
            with self.assertRaises(sqlite3.OperationalError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        finally:
            writer.rollback()
            writer.close()
        self.assertFalse((ngr_home(self.home) / "db/knowledge.db").exists())

    def test_runtime_restart_before_cutover_is_rejected_and_recoverable(self):
        self.source()
        with patch.object(migration, "_runtime_processes", side_effect=[False, False, True]):
            with self.assertRaises(RuntimeError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertFalse(migration.migration_completed(self.home))
        self.assertFalse(config_path(self.home).exists())
        migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertTrue(migration.migration_completed(self.home))

    def test_configuration_save_failure_recovers_without_overwriting_copies(self):
        self.source()
        with patch.object(migration, "write_config", side_effect=OSError("save failed")):
            with self.assertRaises(OSError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        target = ngr_home(self.home) / "db/knowledge.db"
        original = target.read_bytes()
        with self.assertRaises(ValueError):
            resolve_shared(home=self.home, environ={})
        migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertEqual(target.read_bytes(), original)

    def test_receipt_failure_after_copy_recovers_from_prepared_record(self):
        self.source()
        atomic = migration.atomic_json
        count = 0
        def fail_second(path, value):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("receipt failed")
            return atomic(path, value)
        with patch.object(migration, "atomic_json", side_effect=fail_second):
            with self.assertRaises(OSError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        receipt = json.loads((ngr_home(self.home) / "home-migration.json").read_text())
        self.assertEqual(receipt["status"], "planned")
        migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertTrue(migration.migration_completed(self.home))

    def test_unrecorded_conflicts_corruption_and_changed_recovery_output_are_retained(self):
        self.source()
        destination = ngr_home(self.home) / "db/knowledge.db"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertEqual(destination.read_bytes(), b"existing")
        destination.unlink()
        with patch.object(migration, "write_config", side_effect=OSError):
            with self.assertRaises(OSError):
                migration.migrate_home(home=self.home, confirm_stopped=True)
        destination.write_bytes(b"changed output")
        with self.assertRaises(RuntimeError):
            migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertEqual(destination.read_bytes(), b"changed output")

    def test_explicit_databases_are_not_migrated(self):
        self.source()
        write_config({"database": "~/managed.db"}, home=self.home)
        with self.assertRaises(ValueError):
            migration.migrate_home(home=self.home, confirm_stopped=True)
        self.assertFalse((ngr_home(self.home) / "db/knowledge.db").exists())


if __name__ == "__main__":
    unittest.main()

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

from neuron_graph_rag.config_provenance import effective_config_provenance, effective_search_surface
from neuron_graph_rag.database_home import prepare_database, resolve_database
from neuron_graph_rag.evidence_feedback import EngineConfig
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

    SOFT_START = {"soft_start_feedback_reinforcement": True, "soft_start_feedback_ratio": 0.25,
                  "confirmation_decay_ratio": 0.5}

    def test_feedback_policy_stays_default_off_without_explicit_config(self):
        default_provenance = effective_config_provenance(EngineConfig())
        for document in ({}, {"soft_start_feedback_reinforcement": False,
                              "outcome_driven_feedback_deactivation": False,
                              "sibling_feedback_normalization": 0}):
            write_config(document, home=self.home)
            resolved = resolve_shared(home=self.home, environ={})
            self.assertEqual(resolved.feedback, {})
            self.assertEqual(resolved.feedback_identity(), {})
            self.assertEqual(resolved.engine_config(), EngineConfig())
            self.assertEqual(effective_config_provenance(resolved.engine_config()), default_provenance)

    def test_feedback_policy_is_enabled_by_config_and_reported_in_provenance(self):
        write_config({**self.SOFT_START, "outcome_driven_feedback_deactivation": True}, home=self.home)
        resolved = resolve_shared(home=self.home, environ={})
        expected = {**self.SOFT_START, "outcome_driven_feedback_deactivation": True}
        self.assertEqual(resolved.feedback, expected)
        self.assertEqual(resolved.feedback_identity(), {"feedback": expected})
        config = resolved.engine_config()
        self.assertTrue(config.soft_start_feedback_reinforcement)
        self.assertEqual(config.soft_start_feedback_ratio, 0.25)
        self.assertEqual(config.confirmation_decay_ratio, 0.5)
        self.assertTrue(config.outcome_driven_feedback_deactivation)
        self.assertFalse(config.confirmed_outcome_reinforcement)
        feedback = effective_config_provenance(config)["effective_config"]["feedback"]
        for name, value in expected.items():
            self.assertEqual(feedback[name], value)
        self.assertNotEqual(effective_config_provenance(config)["feedback_config_fingerprint"],
                            effective_config_provenance(EngineConfig())["feedback_config_fingerprint"])
        write_config(self.SOFT_START, home=self.home)
        soft_only = resolve_shared(home=self.home, environ={})
        self.assertEqual(soft_only.feedback, self.SOFT_START)
        self.assertFalse(soft_only.engine_config().outcome_driven_feedback_deactivation)
        self.assertEqual(soft_only.engine_config().sibling_feedback_normalization, 0.0)

    def test_sibling_normalization_is_independent_of_soft_start(self):
        for document, expected in (({"sibling_feedback_normalization": 1}, {"sibling_feedback_normalization": 1.0}),
                                   ({"sibling_feedback_normalization": 0.5}, {"sibling_feedback_normalization": 0.5}),
                                   ({**self.SOFT_START, "outcome_driven_feedback_deactivation": True,
                                     "sibling_feedback_normalization": 1.0},
                                    {**self.SOFT_START, "outcome_driven_feedback_deactivation": True,
                                     "sibling_feedback_normalization": 1.0})):
            with self.subTest(document=document):
                write_config(document, home=self.home)
                resolved = resolve_shared(home=self.home, environ={})
                self.assertEqual(resolved.feedback, expected)
                self.assertEqual(resolved.feedback_identity(), {"feedback": expected})
                config = resolved.engine_config()
                self.assertEqual(config.sibling_feedback_normalization, expected["sibling_feedback_normalization"])
                provenance = effective_config_provenance(config)
                self.assertEqual(provenance["effective_config"]["feedback"]["sibling_feedback_normalization"],
                                 expected["sibling_feedback_normalization"])
                self.assertEqual(effective_search_surface(config), "relation")

    def test_feedback_policy_is_read_from_config_only(self):
        environment = {"NGR_SOFT_START_FEEDBACK_REINFORCEMENT": "true", "NGR_SOFT_START_FEEDBACK_RATIO": "0.25",
                       "NGR_CONFIRMATION_DECAY_RATIO": "0.5", "NGR_OUTCOME_DRIVEN_FEEDBACK_DEACTIVATION": "true"}
        command_line = argparse.Namespace(**{name: True for name in self.SOFT_START})
        self.assertEqual(resolve_shared(command_line, home=self.home, environ=environment).feedback, {})

    def test_invalid_feedback_policy_is_rejected_like_the_stdio_cli(self):
        invalid = (
            {"soft_start_feedback_reinforcement": True, "soft_start_feedback_ratio": 0.25},
            {"soft_start_feedback_reinforcement": True, "confirmation_decay_ratio": 0.5},
            {"soft_start_feedback_ratio": 0.25},
            {"confirmation_decay_ratio": 0.5},
            {"outcome_driven_feedback_deactivation": True},
            {**self.SOFT_START, "soft_start_feedback_reinforcement": False},
            {**self.SOFT_START, "soft_start_feedback_reinforcement": "true"},
            {**self.SOFT_START, "soft_start_feedback_reinforcement": 1},
            {**self.SOFT_START, "outcome_driven_feedback_deactivation": "yes"},
            {**self.SOFT_START, "soft_start_feedback_ratio": 0},
            {**self.SOFT_START, "soft_start_feedback_ratio": 1},
            {**self.SOFT_START, "soft_start_feedback_ratio": 1.5},
            {**self.SOFT_START, "soft_start_feedback_ratio": True},
            {**self.SOFT_START, "soft_start_feedback_ratio": "0.25"},
            {**self.SOFT_START, "confirmation_decay_ratio": -0.5},
            {**self.SOFT_START, "confirmation_decay_ratio": None},
            {"sibling_feedback_normalization": -0.1},
            {"sibling_feedback_normalization": 1.01},
            {"sibling_feedback_normalization": True},
            {"sibling_feedback_normalization": "1.0"},
            {"sibling_feedback_normalization": None},
        )
        for document in invalid:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    write_config(document, home=self.home)
                config_path(self.home).parent.mkdir(exist_ok=True)
                config_path(self.home).write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    resolve_shared(home=self.home, environ={})
        for text in ('{"soft_start_feedback_reinforcement": true, "soft_start_feedback_ratio": NaN, '
                     '"confirmation_decay_ratio": 0.5}',
                     '{"soft_start_feedback_reinforcement": true, "soft_start_feedback_ratio": 0.25, '
                     '"confirmation_decay_ratio": Infinity}',
                     '{"sibling_feedback_normalization": NaN}'):
            config_path(self.home).write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_shared(home=self.home, environ={})
        config_path(self.home).write_text(json.dumps({"confirmed_outcome_reinforcement": True,
                                                      "confirmation_decay_ratio": 0.5}), encoding="utf-8")
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

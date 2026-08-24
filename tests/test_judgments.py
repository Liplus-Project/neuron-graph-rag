from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from neuron_graph_rag import JudgmentContractError, NeuronGraphRAG
from tools.judgment_graph import backup, integrity, restore


PROVENANCE = {"source": "issue:116", "actor": "test"}


class JudgmentGraphTests(unittest.TestCase):
    @staticmethod
    def _persistent_state(engine: NeuronGraphRAG) -> dict[str, list[tuple[object, ...]]]:
        tables = [
            row[0]
            for row in engine.store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: sorted(
                (tuple(row) for row in engine.store.connection.execute(f'SELECT * FROM "{table}"')),
                key=repr,
            )
            for table in tables
        }

    def test_lifecycle_supersede_and_audit_history(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("base", "Use the old path", "Initial decision", PROVENANCE)
            updated = engine.judgments.update(
                "base", "Use the old path", "Clarified decision", PROVENANCE,
                expected_revision=1,
            )
            self.assertEqual(updated["revision"], 2)
            successor = engine.judgments.supersede(
                "base", "successor", "Use the new path", "Evidence changed", PROVENANCE,
                expected_revision=2,
            )
            self.assertEqual(
                successor["relations"],
                [
                    {
                        "target_id": "base",
                        "relation_type": "supersedes",
                        "relation_type_revision": 1,
                        "assertion_kind": "explicit",
                    }
                ],
            )
            self.assertEqual(engine.judgments.get("base")["lifecycle"], "archived")
            self.assertEqual(engine.judgments.get("base")["superseded_by"], "successor")
            self.assertEqual([node.node_id for node in engine.store.list_nodes()], ["successor"])
            with self.assertRaises(JudgmentContractError):
                engine.judgments.supersede(
                    "base", "other", "Other", "No", PROVENANCE, expected_revision=2
                )
            with self.assertRaises(JudgmentContractError):
                engine.judgments.restore("base", expected_revision=2)

    def test_dangling_relation_and_stale_update_roll_back_atomically(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("one", "One", "Reason", PROVENANCE)
            with self.assertRaises(JudgmentContractError):
                engine.judgments.update(
                    "one", "Changed", "Reason", PROVENANCE, expected_revision=1,
                    relations=[{"target_id": "missing", "relation_type": "supports"}],
                )
            self.assertEqual(engine.judgments.get("one")["revision"], 1)
            self.assertEqual(engine.judgments.get("one")["statement"], "One")
            with self.assertRaises(JudgmentContractError):
                engine.judgments.update(
                    "one", "Changed", "Reason", PROVENANCE, expected_revision=2
                )

    def test_archive_is_logical_and_hard_delete_is_explicitly_restricted(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("candidate", "Candidate", "Reason", PROVENANCE)
            engine.judgments.archive("candidate", expected_revision=1)
            self.assertEqual(engine.store.list_nodes(), [])
            self.assertEqual(engine.judgments.get("candidate")["statement"], "Candidate")
            engine.judgments.restore("candidate", expected_revision=1)
            self.assertEqual([node.node_id for node in engine.store.list_nodes()], ["candidate"])
            engine.judgments.archive("candidate", expected_revision=1)
            engine.judgments.hard_delete("candidate", expected_revision=1)
            with self.assertRaises(KeyError):
                engine.judgments.get("candidate")

    def test_hard_delete_failure_boundaries_preserve_graph_atomically(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("active", "Active", "Reason", PROVENANCE)
            before = engine.judgments.export()
            with self.assertRaises(JudgmentContractError):
                engine.judgments.hard_delete("active", expected_revision=1)
            self.assertEqual(engine.judgments.export(), before)

            engine.judgments.add("referenced", "Referenced", "Reason", PROVENANCE)
            engine.judgments.add(
                "source", "Source", "Reason", PROVENANCE,
                relations=[{"target_id": "referenced", "relation_type": "supports"}],
            )
            engine.judgments.archive("referenced", expected_revision=1)
            before = engine.judgments.export()
            with self.assertRaises(JudgmentContractError):
                engine.judgments.hard_delete("referenced", expected_revision=1)
            self.assertEqual(engine.judgments.export(), before)

            engine.judgments.add("predecessor", "Old", "Reason", PROVENANCE)
            engine.judgments.supersede(
                "predecessor", "successor", "New", "Reason", PROVENANCE,
                expected_revision=1,
            )
            before = engine.judgments.export()
            with self.assertRaises(JudgmentContractError):
                engine.judgments.hard_delete("predecessor", expected_revision=1)
            self.assertEqual(engine.judgments.export(), before)

    def test_lifecycle_no_op_fails_without_touching_timestamp(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("state", "State", "Reason", PROVENANCE)
            active_timestamp = engine.store.connection.execute(
                "SELECT updated_at FROM judgments WHERE judgment_id = 'state'"
            ).fetchone()[0]
            with self.assertRaisesRegex(JudgmentContractError, "already active"):
                engine.judgments.restore("state", expected_revision=1)
            self.assertEqual(
                engine.store.connection.execute(
                    "SELECT updated_at FROM judgments WHERE judgment_id = 'state'"
                ).fetchone()[0],
                active_timestamp,
            )
            engine.judgments.archive("state", expected_revision=1)
            archived_timestamp = engine.store.connection.execute(
                "SELECT updated_at FROM judgments WHERE judgment_id = 'state'"
            ).fetchone()[0]
            with self.assertRaisesRegex(JudgmentContractError, "already archived"):
                engine.judgments.archive("state", expected_revision=1)
            self.assertEqual(
                engine.store.connection.execute(
                    "SELECT updated_at FROM judgments WHERE judgment_id = 'state'"
                ).fetchone()[0],
                archived_timestamp,
            )

    def test_integrity_rejects_inconsistent_supersession_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutation in ("active_predecessor", "missing_relation", "wrong_successor"):
                with self.subTest(mutation=mutation):
                    database = root / f"{mutation}.sqlite"
                    with NeuronGraphRAG(database) as engine:
                        engine.judgments.add("old", "Old", "Reason", PROVENANCE)
                        engine.judgments.supersede(
                            "old", "new", "New", "Reason", PROVENANCE,
                            expected_revision=1,
                        )
                        if mutation == "active_predecessor":
                            engine.store.connection.execute(
                                "UPDATE judgments SET lifecycle = 'active' WHERE judgment_id = 'old'"
                            )
                        elif mutation == "missing_relation":
                            engine.store.connection.execute(
                                "DELETE FROM judgment_relations WHERE source_id = 'new' AND target_id = 'old' AND relation_type = 'supersedes'"
                            )
                        else:
                            engine.judgments.add("other", "Other", "Reason", PROVENANCE)
                            engine.store.connection.execute(
                                "UPDATE judgments SET superseded_by = 'other' WHERE judgment_id = 'old'"
                            )
                        engine.store.connection.commit()
                    with self.assertRaisesRegex(RuntimeError, "supersession_inconsistencies"):
                        integrity(database)

    def test_integrity_rejects_invalid_relation_revision_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "invalid-binding.sqlite"
            with NeuronGraphRAG(database) as engine:
                engine.judgments.add("target", "Target", "Reason", PROVENANCE)
                engine.judgments.add(
                    "source",
                    "Source",
                    "Reason",
                    PROVENANCE,
                    relations=[
                        {"target_id": "target", "relation_type": "depends_on"}
                    ],
                )
                engine.store.connection.execute(
                    """
                    UPDATE judgment_relations SET relation_type_revision = 999
                    WHERE source_id = 'source'
                    """
                )
                engine.store.connection.commit()
            with self.assertRaisesRegex(RuntimeError, "relation_binding_inconsistencies"):
                integrity(database)

    def test_v1_import_binds_known_relation_without_losing_compatibility(self) -> None:
        payload = {
            "format": "ngr-judgment-graph/v1",
            "judgments": [
                {
                    "judgment_id": "source",
                    "revision": 1,
                    "statement": "Source",
                    "rationale": "Reason",
                    "provenance": PROVENANCE,
                    "lifecycle": "active",
                    "superseded_by": None,
                    "relations": [
                        {"target_id": "target", "relation_type": "depends_on"}
                    ],
                },
                {
                    "judgment_id": "target",
                    "revision": 1,
                    "statement": "Target",
                    "rationale": "Reason",
                    "provenance": PROVENANCE,
                    "lifecycle": "active",
                    "superseded_by": None,
                    "relations": [],
                },
            ],
        }
        with NeuronGraphRAG() as engine:
            engine.judgments.import_graph(payload)
            relation = engine.judgments.get("source")["relations"][0]
            self.assertEqual(relation["relation_type_revision"], 1)
            self.assertEqual(relation["assertion_kind"], "explicit")

    def test_deterministic_round_trip_and_backup_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.sqlite"
            exported = root / "graph.json"
            imported = root / "imported.sqlite"
            backup_path = root / "backup.sqlite"
            restored = root / "restored.sqlite"
            with NeuronGraphRAG(database) as engine:
                engine.judgments.register_relation_type(
                    "supports",
                    "Source supports target.",
                    "test.relations",
                    {"source": "test"},
                    expected_revision=0,
                )
                engine.judgments.add("a", "Alpha", "Reason A", PROVENANCE)
                engine.judgments.add(
                    "b", "Beta", "Reason B", PROVENANCE,
                    relations=[{"target_id": "a", "relation_type": "supports"}],
                )
                engine.judgments.register_relation_type(
                    "supports",
                    "Source formerly supported target.",
                    "test.relations",
                    {"source": "test", "revision": 2},
                    expected_revision=1,
                    lifecycle="deprecated",
                )
                expected = engine.judgments.export()
            exported.write_text(
                json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
            with NeuronGraphRAG(imported) as engine:
                engine.judgments.import_graph(json.loads(exported.read_text(encoding="utf-8")))
                self.assertEqual(engine.judgments.export(), expected)
            backup(database, backup_path)
            restore(backup_path, restored)
            self.assertEqual(integrity(restored)["sqlite_integrity"], "ok")
            with NeuronGraphRAG(restored) as engine:
                self.assertEqual(engine.judgments.export(), expected)

    def test_seeded_relation_types_and_advisory_unknown_and_deprecated_warnings(self) -> None:
        with NeuronGraphRAG() as engine:
            seeded = engine.judgments.list_relation_types()
            self.assertEqual(
                [item["relation_type"] for item in seeded],
                [
                    "conflicts_with", "depends_on", "informs", "refines",
                    "supersedes",
                ],
            )
            self.assertTrue(all(item["revision"] == 1 for item in seeded))
            self.assertEqual(
                engine.judgments.validate_relation_type("depends_on"), []
            )
            engine.judgments.add("target", "Target", "Reason", PROVENANCE)
            unknown = engine.judgments.add(
                "unknown-source",
                "Unknown",
                "Free relation types remain writable",
                PROVENANCE,
                relations=[{"target_id": "target", "relation_type": "supports"}],
            )
            self.assertEqual(
                unknown["relations"][0]["relation_type_revision"], None
            )
            self.assertEqual(
                unknown["advisory_warnings"][0]["code"],
                "unknown_relation_type",
            )
            deprecated = engine.judgments.register_relation_type(
                "supports",
                "Source provides support for target.",
                "test.relations",
                {"source": "test"},
                expected_revision=0,
                lifecycle="deprecated",
            )
            self.assertEqual(deprecated["revision"], 1)
            warned = engine.judgments.get("unknown-source")
            self.assertEqual(
                warned["advisory_warnings"][0]["code"],
                "deprecated_relation_type",
            )
            self.assertIsNone(warned["relations"][0]["relation_type_revision"])

    def test_relation_type_revision_check_and_validation_are_atomic_and_read_only(self) -> None:
        with NeuronGraphRAG() as engine:
            created = engine.judgments.register_relation_type(
                "supports",
                "Source supports target.",
                "test.relations",
                {"source": "test"},
                expected_revision=0,
            )
            self.assertEqual(created["revision"], 1)
            before_stale = self._persistent_state(engine)
            with self.assertRaisesRegex(ValueError, "stale expected_revision"):
                engine.judgments.register_relation_type(
                    "supports",
                    "Changed",
                    "test.relations",
                    {"source": "test"},
                    expected_revision=0,
                )
            self.assertEqual(self._persistent_state(engine), before_stale)
            revised = engine.judgments.register_relation_type(
                "supports",
                "Source supports target with current evidence.",
                "test.relations",
                {"source": "test", "revision": 2},
                expected_revision=1,
                lifecycle="deprecated",
            )
            self.assertEqual(revised["revision"], 2)
            before_reads = self._persistent_state(engine)
            self.assertEqual(
                engine.judgments.validate_relation_type("supports")[0]["code"],
                "deprecated_relation_type",
            )
            self.assertEqual(
                engine.judgments.validate_relation_type("unregistered")[0]["code"],
                "unknown_relation_type",
            )
            with self.assertRaises(KeyError):
                engine.judgments.get_relation_type("unregistered")
            self.assertEqual(self._persistent_state(engine), before_reads)

    def test_seeded_relation_type_can_be_revised_and_reopened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "revised-seed.sqlite"
            with NeuronGraphRAG(database) as engine:
                revised = engine.judgments.register_relation_type(
                    "depends_on",
                    "The source depends on the target under a revised contract.",
                    "ngr.decision_structure",
                    {"source": "test"},
                    expected_revision=1,
                    lifecycle="deprecated",
                )
                self.assertEqual(revised["revision"], 2)
            with NeuronGraphRAG(database) as reopened:
                current = reopened.judgments.get_relation_type("depends_on")
                self.assertEqual(current["revision"], 2)
                self.assertEqual(current["lifecycle"], "deprecated")
                original = reopened.judgments.get_relation_type(
                    "depends_on", revision=1
                )
                self.assertTrue(original["definition"].startswith("この結論は"))

    def test_existing_schema_migration_preserves_and_binds_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.sqlite"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE judgments (
                        judgment_id TEXT PRIMARY KEY,
                        current_revision INTEGER NOT NULL,
                        lifecycle TEXT NOT NULL,
                        superseded_by TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE judgment_revisions (
                        judgment_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        statement TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (judgment_id, revision)
                    );
                    CREATE TABLE judgment_relations (
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        relation_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (source_id, target_id, relation_type)
                    );
                    INSERT INTO judgments VALUES
                        ('source', 1, 'active', NULL, 't', 't'),
                        ('target', 1, 'active', NULL, 't', 't');
                    INSERT INTO judgment_revisions VALUES
                        ('source', 1, 'Source', 'Reason', '{"source":"legacy"}', 't'),
                        ('target', 1, 'Target', 'Reason', '{"source":"legacy"}', 't');
                    INSERT INTO judgment_relations VALUES
                        ('source', 'target', 'depends_on', 't');
                    """
                )
            with NeuronGraphRAG(database) as engine:
                relation = engine.judgments.get("source")["relations"][0]
                self.assertEqual(relation["relation_type_revision"], 1)
                self.assertEqual(relation["assertion_kind"], "explicit")
                self.assertEqual(engine.judgments.validate_relation_type("depends_on"), [])
            with NeuronGraphRAG(database) as restarted:
                self.assertEqual(
                    restarted.judgments.get("source")["relations"][0], relation
                )

    def test_read_api_search_get_filters_and_never_mutates(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add(
                "one:cache-policy",
                "Prefer bounded cache invalidation",
                "The fallback must remain deterministic",
                {"repository": "Liplus-Project/one", "source": "wiki"},
            )
            engine.judgments.add(
                "two:cache-policy",
                "Archive the legacy cache policy",
                "The replacement is active",
                {"repository": "Liplus-Project/two", "source": "wiki"},
            )
            engine.judgments.archive("two:cache-policy", expected_revision=1)
            before = self._persistent_state(engine)

            default_results = engine.judgments.search_judgments("cache policy")
            self.assertEqual(
                [item["judgment_id"] for item in default_results],
                ["one:cache-policy"],
            )
            self.assertEqual(default_results[0]["revision"], 1)
            self.assertIn("score", default_results[0])
            self.assertEqual(
                set(default_results[0]["explanation"]),
                {"sparse_score", "dense_score", "sparse_weight", "dense_weight"},
            )
            archived = engine.judgments.search_judgments(
                "cache policy",
                include_archived=True,
                repository="two",
            )
            self.assertEqual([item["judgment_id"] for item in archived], ["two:cache-policy"])
            exact = engine.judgments.get_judgment("two:cache-policy")
            self.assertEqual(exact["lifecycle"], "archived")
            with self.assertRaises(JudgmentContractError):
                engine.judgments.search_judgments(" ")

            self.assertEqual(self._persistent_state(engine), before)

    def test_read_api_traversal_is_filtered_cycle_safe_and_deterministic(self) -> None:
        with NeuronGraphRAG() as engine:
            engine.judgments.add("a", "Alpha", "Reason", PROVENANCE)
            engine.judgments.add(
                "b", "Beta", "Reason", PROVENANCE,
                relations=[{"target_id": "a", "relation_type": "supports"}],
            )
            engine.judgments.add(
                "c", "Gamma", "Reason", PROVENANCE,
                relations=[{"target_id": "b", "relation_type": "depends_on"}],
            )
            engine.judgments.update(
                "a", "Alpha", "Reason", PROVENANCE, expected_revision=1,
                relations=[{"target_id": "c", "relation_type": "informs"}],
            )
            before = self._persistent_state(engine)

            outgoing = engine.judgments.traverse_judgments("a", max_hops=3)
            self.assertEqual(
                [(item["hop"], item["judgment"]["judgment_id"]) for item in outgoing],
                [(1, "c"), (2, "b")],
            )
            incoming = engine.judgments.traverse_judgments(
                "a", direction="incoming", relation_type="supports", max_hops=3
            )
            self.assertEqual(
                [(item["hop"], item["judgment"]["judgment_id"]) for item in incoming],
                [(1, "b")],
            )
            both = engine.judgments.traverse_judgments("a", direction="both", max_hops=3)
            self.assertEqual(
                [(item["hop"], item["judgment"]["judgment_id"]) for item in both],
                [(1, "b"), (1, "c")],
            )
            engine.judgments.archive("b", expected_revision=1)
            archived_state = self._persistent_state(engine)
            self.assertEqual(engine.judgments.traverse_judgments("a", direction="incoming"), [])
            included = engine.judgments.traverse_judgments(
                "a", direction="incoming", max_hops=2, include_archived=True
            )
            self.assertEqual(
                [(item["hop"], item["judgment"]["judgment_id"]) for item in included],
                [(1, "b"), (2, "c")],
            )
            self.assertEqual(self._persistent_state(engine), archived_state)
            self.assertNotEqual(before, archived_state)


if __name__ == "__main__":
    unittest.main()

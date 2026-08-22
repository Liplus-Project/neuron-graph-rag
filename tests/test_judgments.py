from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import JudgmentContractError, NeuronGraphRAG
from tools.judgment_graph import backup, integrity, restore


PROVENANCE = {"source": "issue:116", "actor": "test"}


class JudgmentGraphTests(unittest.TestCase):
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
                [{"target_id": "base", "relation_type": "supersedes"}],
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

    def test_deterministic_round_trip_and_backup_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.sqlite"
            exported = root / "graph.json"
            imported = root / "imported.sqlite"
            backup_path = root / "backup.sqlite"
            restored = root / "restored.sqlite"
            with NeuronGraphRAG(database) as engine:
                engine.judgments.add("a", "Alpha", "Reason A", PROVENANCE)
                engine.judgments.add(
                    "b", "Beta", "Reason B", PROVENANCE,
                    relations=[{"target_id": "a", "relation_type": "supports"}],
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


if __name__ == "__main__":
    unittest.main()

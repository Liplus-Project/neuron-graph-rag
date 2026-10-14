from __future__ import annotations

import unittest
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

from neuron_graph_rag import FeedbackLedger, JudgmentContractError, NeuronGraphRAG, SourceUseEvent
from neuron_graph_rag.decision_wiki_incremental import apply_plan, build_plan, scope_plan
from neuron_graph_rag.decision_wiki_import import DecisionWikiImportError
from tests import test_decision_wiki_incremental as incremental
from tests.test_decision_wiki_incremental import _link
from tests import test_judgments as domain
from tests.test_judgments import PROVENANCE
from tools.import_decision_wikis_incremental import dry_run, write
from tools.judgment_graph import integrity


class StoredSuccessorTests(unittest.TestCase):
    def populate(self, engine):
        for identity in ("left", "right", "base"):
            engine.judgments.add(identity, identity, "Reason", PROVENANCE)
        engine.judgments.add(
            "successor", "Existing conclusion", "Existing rationale", PROVENANCE,
            relations=[{"target_id": "base", "relation_type": "informs"}],
        )

    def test_preserves_history_learning_bindings_and_current_payload(self):
        with NeuronGraphRAG() as engine:
            self.populate(engine)
            trace = engine.search("Existing conclusion", limit=4)
            FeedbackLedger(engine).record_source_use(
                trace.trace_id,
                [SourceUseEvent("successor", stage) for stage in ("selected", "validated", "used")],
                idempotency_key="preserved-use",
            )
            engine.judgments.register_relation_type(
                "informs", "Revised meaning", "test", PROVENANCE, expected_revision=1,
            )
            connection = engine.store.connection
            connection.execute("UPDATE edges SET weight=2.7, factuality=0.4 WHERE source_id='successor'")
            connection.execute("UPDATE nodes SET confidence=0.6 WHERE node_id='successor'")
            connection.commit()
            before = domain.JudgmentGraphTests._persistent_state(engine)
            current = engine.judgments.get("successor")
            result = engine.judgments.reconcile_supersedes(
                "successor", [("left", 1), ("right", 1)], expected_revision=1,
            )
            self.assertEqual(result["revision"], 2)
            for key in ("statement", "rationale", "provenance"):
                self.assertEqual(current[key], result[key])
            after = domain.JudgmentGraphTests._persistent_state(engine)
            for table, rows in before.items():
                if table not in {"judgments", "judgment_revisions", "judgment_relations", "edges"}:
                    self.assertEqual(after[table], rows, table)
                elif table != "judgments":
                    self.assertTrue(set(rows) <= set(after[table]), table)
            for identity in ("left", "right"):
                old = engine.judgments.get(identity)
                self.assertEqual((old["lifecycle"], old["superseded_by"], old["revision"]),
                                 ("archived", "successor", 1))
            integrity(connection)
            engine.judgments.reconcile_supersedes(
                "successor", [("left", 1), ("right", 1)], expected_revision=2,
            )
            self.assertEqual(domain.JudgmentGraphTests._persistent_state(engine), after)
            self.assertEqual({item["judgment_id"] for item in engine.judgments.search_judgments("left")},
                             {"successor", "base"})

    def test_rejects_bad_inputs_and_conflicts_atomically(self):
        with NeuronGraphRAG() as engine:
            self.populate(engine)
            engine.judgments.add("rival", "Rival", "Reason", PROVENANCE)
            engine.judgments.supersede("base", "taker", "Taker", "Reason", PROVENANCE, expected_revision=1)
            before = domain.JudgmentGraphTests._persistent_state(engine)
            cases = [
                ([], 1, "required"),
                ([("successor", 1)], 1, "self"),
                ([("left", 1), ("left", 1)], 1, "duplicate"),
                ([("left", 1), ("right", 2)], 1, "stale predecessor"),
                ([("left", 1)], 2, "stale successor"),
                ([("left", 1), ("base", 1)], 1, "another successor"),
            ]
            for predecessors, revision, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(JudgmentContractError, message):
                        engine.judgments.reconcile_supersedes(
                            "successor", predecessors, expected_revision=revision,
                        )
                    self.assertEqual(domain.JudgmentGraphTests._persistent_state(engine), before)

    def test_cycle_and_inactive_successor_are_rejected(self):
        with NeuronGraphRAG() as engine:
            self.populate(engine)
            engine.judgments.update("left", "Left", "Reason", PROVENANCE, expected_revision=1,
                                    relations=[{"target_id": "successor", "relation_type": "supersedes"}])
            before = domain.JudgmentGraphTests._persistent_state(engine)
            with self.assertRaisesRegex(JudgmentContractError, "cycle"):
                engine.judgments.reconcile_supersedes("successor", [("left", 2)], expected_revision=1)
            self.assertEqual(domain.JudgmentGraphTests._persistent_state(engine), before)
            engine.judgments.archive("successor", expected_revision=1)
            with self.assertRaisesRegex(JudgmentContractError, "current active"):
                engine.judgments.reconcile_supersedes("successor", [("right", 1)], expected_revision=1)

    def test_exception_during_multi_predecessor_write_rolls_back_every_table(self):
        with NeuronGraphRAG() as engine:
            self.populate(engine)
            engine.store.connection.execute(
                "CREATE TRIGGER reject_right BEFORE UPDATE ON judgments WHEN NEW.judgment_id='right' "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
            before = domain.JudgmentGraphTests._persistent_state(engine)
            with self.assertRaisesRegex(Exception, "injected failure"):
                engine.judgments.reconcile_supersedes(
                    "successor", [("left", 1), ("right", 1)], expected_revision=1,
                )
            self.assertEqual(domain.JudgmentGraphTests._persistent_state(engine), before)

    def test_another_writer_cannot_change_versions_between_validation_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "concurrency.sqlite"
            with NeuronGraphRAG(database) as engine:
                self.populate(engine)
                competing = sqlite3.connect(database, timeout=0.01)
                require = engine.judgments._require
                attempted = []

                def read(connection, identity):
                    row = require(connection, identity)
                    if identity == "successor":
                        with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                            competing.execute("UPDATE judgments SET updated_at='competing' WHERE judgment_id='left'")
                        competing.rollback()
                        attempted.append(identity)
                    return row

                try:
                    with mock.patch.object(engine.judgments, "_require", side_effect=read):
                        engine.judgments.reconcile_supersedes("successor", [("left", 1)], expected_revision=1)
                    self.assertEqual(attempted, ["successor"])
                    integrity(engine.store.connection)
                finally:
                    competing.close()


class LatePredecessorImportTests(unittest.TestCase):
    setUp = incremental.IncrementalImportTests.setUp
    tearDown = incremental.IncrementalImportTests.tearDown
    _judgment = incremental.IncrementalImportTests._judgment
    _restate = incremental.IncrementalImportTests._restate

    def successor_first(self):
        self.wiki.put("successor", "Successor", "- **supersedes** " + _link("late"),
                      "- **supersedes** " + _link("later"), "- **supersedes** " + _link("late"))
        write((self.wiki.source(),), self.database)
        self.wiki.put("late", "Late predecessor", state="superseded")
        self.wiki.put("later", "Later predecessor", state="superseded")

    def test_late_predecessors_reconcile_after_add_before_archive(self):
        self.successor_first()
        result = write((self.wiki.source(),), self.database)
        self.assertEqual([(item["action"], item["judgment_id"]) for item in result["written"]],
                         [("add", "one:late"), ("add", "one:later"), ("reconcile_supersedes", "one:successor")])
        for identity in ("one:late", "one:later"):
            old = self._judgment(identity)
            self.assertEqual((old["lifecycle"], old["superseded_by"]), ("archived", "one:successor"))
        self.assertEqual(dry_run((self.wiki.source(),), self.database)["operations"], [])
        integrity(self.database)

    def test_revised_predecessor_expected_revision_tracks_the_write_order(self):
        self.successor_first()
        with NeuronGraphRAG(self.database) as engine:
            plan = build_plan(engine.store.connection, (self.wiki.source(),))
            entry = plan["_entries"]["one:late"]
            engine.judgments.add("one:late", entry.statement, entry.page, entry.provenance())
        self.wiki.put("late", "Late predecessor revised", state="superseded")
        plan = dry_run((self.wiki.source(),), self.database)
        reconciliation = next(item for item in plan["operations"] if item["action"] == "reconcile_supersedes")
        self.assertIn({"judgment_id": "one:late", "expected_revision": 2}, reconciliation["predecessors"])
        write((self.wiki.source(),), self.database)
        self.assertEqual(self._judgment("one:late")["revision"], 2)

    def test_scoped_plan_omits_unrelated_edge_additions(self):
        self.successor_first()
        self._restate("one:old", [])
        with NeuronGraphRAG(self.database) as engine:
            with engine.store.transaction() as connection:
                full = build_plan(connection, (self.wiki.source(),))
                self.assertEqual(full["counts"]["add_edges"], 1)
                scoped = scope_plan(full, ("one:late", "one:later"))
                self.assertEqual(scoped["counts"]["add_edges"], 0)
                self.assertEqual(scoped["counts"]["reconcile_supersedes"], 1)
                apply_plan(engine.judgments, scoped)
                integrity(connection)
            again = scope_plan(build_plan(engine.store.connection, (self.wiki.source(),)), ("one:late", "one:later"))
            self.assertEqual(again["operations"], [])
            self.assertEqual(engine.judgments.get("one:old")["relations"], [])

    def test_stale_reconciliation_rolls_back_prior_additions(self):
        self.successor_first()
        with NeuronGraphRAG(self.database) as engine:
            plan = build_plan(engine.store.connection, (self.wiki.source(),))
            plan["operations"][-1]["expected_revision"] += 1
            before = domain.JudgmentGraphTests._persistent_state(engine)
            with self.assertRaisesRegex(JudgmentContractError, "stale successor"):
                with engine.store.transaction():
                    apply_plan(engine.judgments, plan)
            self.assertEqual(domain.JudgmentGraphTests._persistent_state(engine), before)

    def test_wiki_cycle_crossing_a_stored_successor_is_blocked(self):
        self.successor_first()
        self.wiki.put("late", "Late predecessor", "- **supersedes** " + _link("successor"))
        plan = dry_run((self.wiki.source(),), self.database)
        self.assertFalse(any(item["judgment_id"] in {"one:late", "one:successor"}
                             for item in plan["operations"]))
        self.assertEqual({item["judgment_id"] for item in plan["blocked"]},
                         {"one:late", "one:successor"})

    def test_scope_rejects_a_required_stored_successor_that_cannot_reconcile(self):
        self.successor_first()
        with NeuronGraphRAG(self.database) as engine:
            engine.judgments.archive("one:successor", expected_revision=1)
            plan = build_plan(engine.store.connection, (self.wiki.source(),))
            with self.assertRaisesRegex(DecisionWikiImportError, "unresolved supersession"):
                scope_plan(plan, ("one:late", "one:later"))
            engine.judgments.restore("one:successor", expected_revision=1)
            engine.judgments.update("one:successor", "Store changed", "R", {"source": "dialogue"},
                                    expected_revision=1)
            self.wiki.put("successor", "Wiki changed", "- **supersedes** " + _link("late"))
            plan = build_plan(engine.store.connection, (self.wiki.source(),))
            with self.assertRaisesRegex(DecisionWikiImportError, "unresolved supersession"):
                scope_plan(plan, ("one:late", "one:later"))


if __name__ == "__main__":
    unittest.main()

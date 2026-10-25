from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path

from neuron_graph_rag import EngineConfig, FeedbackLedger, NeuronGraphRAG, SourceUseEvent
from neuron_graph_rag import evidence_feedback
from neuron_graph_rag.cli import expire_nodes, main as cli_main
from neuron_graph_rag.cpu_shortlist_retrieval import (
    E5_MODEL_ID,
    E5_REVISION,
    RERANKER_MODEL_ID,
    RERANKER_REVISION,
    CacheStaleError,
    CpuShortlistRetriever,
    attach_cpu_shortlist_retriever,
)
from neuron_graph_rag.storage import SQLiteStore

DAY = 86_400.0
NODE_TABLES = (
    "activation_state",
    "retrieval_results",
    "success_nodes",
    "source_use_state",
    "confirmed_source_uses",
    "delayed_outcome_nodes",
)


def _config() -> EngineConfig:
    return EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        entry_weight=0.5,
        graph_weight=0.5,
        seed_count=1,
        max_hops=2,
    )


def _populate(engine: NeuronGraphRAG, *, now: float) -> None:
    engine.add_document("hub", "alpha hub fact", now=now)
    engine.add_document("kept", "bravo kept fact", now=now)
    engine.add_document("stale", "zulu stale fact", now=now)
    engine.add_edge("hub", "kept", "supports", weight=0.5)
    engine.add_edge("hub", "stale", "mentions", weight=0.5)
    engine.add_edge("stale", "kept", "mentions", weight=0.5)


class FakeE5:
    model_id = E5_MODEL_ID
    revision = E5_REVISION

    def embed_query(self, query):
        return (1.0, 0.0)

    def embed_passages(self, passages):
        return [(1.0, 0.0) if "zulu" in passage else (0.0, 1.0) for passage in passages]


class FakeReranker:
    model_id = RERANKER_MODEL_ID
    revision = RERANKER_REVISION

    def score_pairs(self, pairs):
        return [1.0 for _ in pairs]


class NodeUsageClockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NeuronGraphRAG(config=_config())
        _populate(self.engine, now=1_000.0)

    def tearDown(self) -> None:
        self.engine.close()

    def test_new_node_records_creation_and_no_use(self) -> None:
        self.assertEqual(
            self.engine.store.node_usage("kept"),
            {"created_at": 1_000.0, "last_used_at": None},
        )

    def test_reingest_keeps_creation_time_and_use(self) -> None:
        trace = self.engine.search("bravo", limit=3, now=2_000.0)
        self.engine.record_success(trace.trace_id, ["kept"], now=2_001.0)
        self.engine.add_document("kept", "bravo kept fact revised", now=9_000.0)
        self.assertEqual(
            self.engine.store.node_usage("kept"),
            {"created_at": 1_000.0, "last_used_at": 2_001.0},
        )

    def test_success_feedback_updates_last_used(self) -> None:
        trace = self.engine.search("alpha", limit=3, now=2_000.0)
        self.engine.record_success(trace.trace_id, ["kept"], now=2_500.0)
        self.assertEqual(self.engine.store.node_usage("kept")["last_used_at"], 2_500.0)
        self.assertIsNone(self.engine.store.node_usage("hub")["last_used_at"])
        self.assertIsNone(self.engine.store.node_usage("stale")["last_used_at"])

    def test_older_feedback_does_not_move_clock_backwards(self) -> None:
        first = self.engine.search("bravo", limit=3, now=2_000.0)
        second = self.engine.search("bravo", limit=3, now=2_001.0)
        self.engine.record_success(second.trace_id, ["kept"], now=5_000.0)
        self.engine.record_success(first.trace_id, ["kept"], now=3_000.0)
        self.assertEqual(self.engine.store.node_usage("kept")["last_used_at"], 5_000.0)

    def test_retrieval_does_not_update_last_used(self) -> None:
        self.engine.search("alpha bravo zulu", limit=3, now=2_000.0)
        self.engine.search_channels("alpha bravo zulu", limit=3, now=2_001.0)
        for node_id in ("hub", "kept", "stale"):
            self.assertIsNone(self.engine.store.node_usage(node_id)["last_used_at"])

    def test_source_use_only_used_stage_updates_last_used(self) -> None:
        ledger = FeedbackLedger(self.engine)
        trace = self.engine.search("alpha", limit=3, now=2_000.0)
        ledger.record_source_use(
            trace.trace_id,
            [SourceUseEvent("kept", "selected"), SourceUseEvent("kept", "validated")],
            idempotency_key="partial",
            now=2_100.0,
        )
        self.assertIsNone(self.engine.store.node_usage("kept")["last_used_at"])
        ledger.record_source_use(
            trace.trace_id,
            [SourceUseEvent("kept", "used")],
            idempotency_key="used",
            now=2_200.0,
        )
        self.assertEqual(self.engine.store.node_usage("kept")["last_used_at"], 2_200.0)

    def test_confirmed_only_policy_used_stage_updates_last_used(self) -> None:
        with evidence_feedback.NeuronGraphRAG(
            config=evidence_feedback.EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                seed_count=1,
                max_hops=2,
                confirmed_outcome_reinforcement=True,
                confirmation_decay_ratio=0.5,
            )
        ) as engine:
            _populate(engine, now=1_000.0)
            ledger = FeedbackLedger(engine)
            trace = engine.search("alpha", limit=3, now=2_000.0)
            for stage in ("selected", "validated", "used"):
                ledger.record_source_use(
                    trace.trace_id,
                    [SourceUseEvent("kept", stage)],
                    idempotency_key=stage,
                    now=2_300.0,
                )
            self.assertEqual(engine.store.count_feedback(), 0)
            self.assertEqual(engine.store.node_usage("kept")["last_used_at"], 2_300.0)

    def test_soft_start_policy_used_stage_updates_last_used(self) -> None:
        with evidence_feedback.NeuronGraphRAG(
            config=evidence_feedback.EngineConfig(
                sparse_weight=1.0,
                dense_weight=0.0,
                seed_count=1,
                max_hops=2,
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
            )
        ) as engine:
            _populate(engine, now=1_000.0)
            ledger = FeedbackLedger(engine)
            trace = engine.search_channels("alpha", limit=3, now=2_000.0).relation
            node_id = trace.hits[0].node.node_id
            for stage in ("selected", "validated", "used"):
                ledger.record_source_use(
                    trace.trace_id,
                    [SourceUseEvent(node_id, stage)],
                    idempotency_key=stage,
                    now=2_400.0,
                )
            self.assertEqual(engine.store.node_usage(node_id)["last_used_at"], 2_400.0)


class NodeExpiryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NeuronGraphRAG(config=_config())
        _populate(self.engine, now=0.0)
        trace = self.engine.search("alpha bravo", limit=3, now=20 * DAY)
        self.engine.record_success(trace.trace_id, ["kept", "hub"], now=20 * DAY)
        self.now = 40 * DAY

    def tearDown(self) -> None:
        self.engine.close()

    def _count(self, table: str, node_id: str) -> int:
        column = "node_id"
        return int(
            self.engine.store.connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (node_id,)
            ).fetchone()[0]
        )

    def test_dry_run_lists_without_deleting(self) -> None:
        result = self.engine.expire_unused_nodes(30, now=self.now, dry_run=True)
        self.assertEqual([item["node_id"] for item in result["expired"]], ["stale"])
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.engine.store.get_node("stale").node_id, "stale")

    def test_expiry_counts_from_last_use_or_creation(self) -> None:
        result = self.engine.expire_unused_nodes(30, now=self.now)
        self.assertEqual(result["expired_count"], 1)
        self.assertEqual(
            result["expired"],
            [{"node_id": "stale", "created_at": 0.0, "last_used_at": None}],
        )
        self.assertEqual(
            [node.node_id for node in self.engine.store.list_nodes()], ["hub", "kept"]
        )
        # Used 20 days before now: not expired at 30 days, expired at 10 days.
        later = self.engine.expire_unused_nodes(10, now=self.now)
        self.assertEqual(
            [item["node_id"] for item in later["expired"]], ["hub", "kept"]
        )

    def test_expired_node_leaves_every_retrieval_surface(self) -> None:
        before = self.engine.search("zulu", limit=3, now=self.now - 1)
        stale_hit = next(hit for hit in before.hits if hit.node.node_id == "stale")
        self.assertGreater(stale_hit.sparse_score, 0.0)
        self.assertGreater(stale_hit.dense_score, 0.0)
        self.assertGreater(self._count("retrieval_results", "stale"), 0)
        self.engine.expire_unused_nodes(30, now=self.now)

        with self.assertRaises(KeyError):
            self.engine.store.get_node("stale")
        for table in NODE_TABLES:
            self.assertEqual(self._count(table, "stale"), 0, table)
        edges = self.engine.store.connection.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = 'stale' OR target_id = 'stale'"
        ).fetchone()[0]
        self.assertEqual(edges, 0)
        self.assertEqual(
            self.engine.store.connection.execute("PRAGMA foreign_key_check").fetchall(), []
        )

        after = self.engine.search("zulu", limit=5, now=self.now + 1)
        self.assertEqual(sorted(hit.node.node_id for hit in after.hits), ["hub", "kept"])
        self.assertTrue(all(hit.sparse_raw_score == 0.0 for hit in after.hits))
        nodes = self.engine.store.list_nodes()
        self.assertEqual(
            set(self.engine.sparse_retriever.score("zulu", nodes)), {"hub", "kept"}
        )
        self.assertEqual(
            set(self.engine.dense_retriever.score("zulu stale fact", nodes)), {"hub", "kept"}
        )

        channels = self.engine.search_channels("zulu", limit=5, now=self.now + 3)
        self.assertNotIn("stale", [hit.node.node_id for hit in channels.lexical.hits])
        self.assertNotIn("stale", [hit.node.node_id for hit in channels.relation.hits])

    def test_failure_rolls_back_the_whole_expiry(self) -> None:
        self.engine.add_document("stale2", "yankee stale fact", now=0.0)
        self.engine.store.connection.executescript(
            """
            CREATE TRIGGER fail_second_delete BEFORE DELETE ON nodes
            WHEN OLD.node_id = 'stale2'
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END;
            """
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.engine.expire_unused_nodes(30, now=self.now)
        self.assertFalse(self.engine.store.connection.in_transaction)
        ids = [node.node_id for node in self.engine.store.list_nodes()]
        self.assertEqual(ids, ["hub", "kept", "stale", "stale2"])
        self.assertEqual(
            self.engine.store.connection.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id = 'stale' OR target_id = 'stale'"
            ).fetchone()[0],
            2,
        )

    def test_judgment_nodes_are_not_expired(self) -> None:
        self.engine.judgments.add(
            "decision", "Keep judgments", "Own lifecycle", {"source": "issue:308", "actor": "test"}
        )
        self.engine.store.connection.execute(
            "UPDATE nodes SET created_at = 0.0 WHERE node_id = 'decision'"
        )
        self.engine.store.connection.commit()
        result = self.engine.expire_unused_nodes(30, now=self.now)
        self.assertNotIn("decision", [item["node_id"] for item in result["expired"]])
        self.assertEqual(self.engine.judgments.get("decision")["judgment_id"], "decision")

    def test_invalid_age_is_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.engine.expire_unused_nodes(value, now=self.now)

    def test_attached_shortlist_cache_is_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = CpuShortlistRetriever(
                Path(directory) / "shortlist.db", FakeE5(), FakeReranker()
            )
            attach_cpu_shortlist_retriever(self.engine, retriever)
            self.engine.update_cpu_shortlist_cache()
            result = self.engine.expire_unused_nodes(30, now=self.now)
            self.assertEqual(result["pruned_caches"], {"cpu_shortlist_retriever": 1})
            trace = self.engine.search_cpu_shortlist("zulu")
            self.assertNotIn("stale", [hit.node.node_id for hit in trace.hits])

    def test_unpruned_shortlist_cache_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = CpuShortlistRetriever(
                Path(directory) / "shortlist.db", FakeE5(), FakeReranker()
            )
            retriever.update_cache(self.engine.store.list_nodes())
            self.engine.expire_unused_nodes(30, now=self.now)
            with self.assertRaisesRegex(CacheStaleError, "extra=\\['stale'\\]"):
                retriever.search("zulu", self.engine.store.list_nodes())


class LegacyDatabaseMigrationTest(unittest.TestCase):
    def test_existing_nodes_gain_usage_columns_and_creation_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE nodes (
                        node_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0)
                    );
                    INSERT INTO nodes VALUES ('old', 'legacy text', '{}', 1.0);
                    """
                )
            with SQLiteStore(database) as store:
                columns = [
                    row["name"] for row in store.connection.execute("PRAGMA table_info(nodes)")
                ]
                self.assertEqual(columns[-2:], ["created_at", "last_used_at"])
                usage = store.node_usage("old")
                self.assertIsNotNone(usage["created_at"])
                self.assertIsNone(usage["last_used_at"])
                first_stamp = usage["created_at"]
            with SQLiteStore(database) as store:
                self.assertEqual(store.node_usage("old")["created_at"], first_stamp)
                result = store.expire_unused_nodes(
                    cutoff=first_stamp - 1.0, now=first_stamp + 1.0
                )
                self.assertEqual(result["expired_count"], 0)


class ExpireNodesCliTest(unittest.TestCase):
    def test_cli_dry_run_then_delete_on_explicit_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "knowledge.db"
            with NeuronGraphRAG(database) as engine:
                engine.add_document("old", "old fact", now=0.0)
                engine.add_document("fresh", "fresh fact")
            dry = expire_nodes(str(database), 30, dry_run=True, home=directory)
            self.assertEqual([item["node_id"] for item in dry["expired"]], ["old"])
            self.assertEqual(dry["database_source"], "command_line")

            output = io.StringIO()
            with redirect_stdout(output):
                code = cli_main(
                    ["expire-nodes", "--database", str(database), "--older-than-days", "30"]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["expired_count"], 1)
            with NeuronGraphRAG(database) as engine:
                self.assertEqual(
                    [node.node_id for node in engine.store.list_nodes()], ["fresh"]
                )

    def test_cli_rejects_missing_database_and_bad_age(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.db"
            with self.assertRaises(FileNotFoundError):
                expire_nodes(str(missing), 30, home=directory)
            self.assertFalse(missing.exists())
            for arguments in (
                ["expire-nodes", "--database", str(missing), "--older-than-days", "30"],
                ["expire-nodes", "--database", str(missing), "--older-than-days", "0"],
            ):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli_main(arguments)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()

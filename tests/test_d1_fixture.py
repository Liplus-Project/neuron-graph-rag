from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.d1_fixture import load_fixture, read_fixture
from tools.acquire_d1_fixture import (
    build_coverage_query,
    redact,
    redact_final_payloads,
    transform,
    validate_read_only_sql,
)
from tools.compare_d1_provenance import compare_reports


FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "d1_liplus_wiki.json"
PROVENANCE = FIXTURES / "d1_liplus_wiki.provenance.json"


class ReadOnlyGuardTest(unittest.TestCase):
    def test_allows_one_select_or_with_query(self) -> None:
        self.assertEqual(validate_read_only_sql(" SELECT 1; "), "SELECT 1")
        self.assertEqual(
            validate_read_only_sql("WITH rows AS (SELECT 1) SELECT * FROM rows"),
            "WITH rows AS (SELECT 1) SELECT * FROM rows",
        )
        self.assertEqual(
            validate_read_only_sql("SELECT * FROM docs WHERE slug = 'C.-Update'"),
            "SELECT * FROM docs WHERE slug = 'C.-Update'",
        )

    def test_rejects_writes_administration_and_multiple_statements(self) -> None:
        for sql in (
            "DELETE FROM search_docs",
            "WITH x AS (SELECT 1) DELETE FROM search_docs",
            "PRAGMA table_info(search_docs)",
            "SELECT 1; SELECT 2",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate_read_only_sql(sql)

    def test_redacts_common_credential_shapes_deterministically(self) -> None:
        source = {
            "content": "Authorization: Bearer abcdefghijklmnop1234",
            "nested": ["GITHUB_TOKEN=abcdefghijklmnop1234"],
        }

        first, first_count = redact(source)
        second, second_count = redact(source)

        self.assertEqual(first, second)
        self.assertEqual(first_count, second_count)
        self.assertEqual(first_count, 2)
        self.assertNotIn("abcdefghijklmnop1234", json.dumps(first))


class FixtureTransformTest(unittest.TestCase):
    def test_coverage_does_not_count_empty_commit_sha_sentinel(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        self.addCleanup(connection.close)
        connection.execute(
            "CREATE TABLE search_docs "
            "(repo TEXT, type TEXT, updated_at TEXT, commit_sha TEXT)"
        )
        connection.executemany(
            "INSERT INTO search_docs VALUES (?, ?, ?, ?)",
            [
                ("owner/repo", "wiki_doc", "2026-01-01", ""),
                ("owner/repo", "wiki_doc", "2026-01-02", ""),
                ("owner/repo", "diff", "2026-01-01", "abc"),
                ("owner/repo", "diff", "2026-01-02", "abc"),
                ("owner/repo", "diff", "2026-01-03", "def"),
            ],
        )

        rows = connection.execute(
            build_coverage_query(["owner/repo"], ["diff", "wiki_doc"])
        ).fetchall()
        counts = {row["type"]: row["distinct_commit_count"] for row in rows}

        self.assertEqual(counts, {"diff": 2, "wiki_doc": 0})

    def test_final_payload_redaction_covers_source_and_known_gap(self) -> None:
        secret = "GITHUB_TOKEN=abcdefghijklmnop1234"
        fixture = {
            "source": {"repositories": [secret]},
            "nodes": [],
            "edges": [],
        }
        report = {
            "known_gaps": [secret],
            "result": {"redactions": 0},
        }

        redacted_fixture, redacted_report = redact_final_payloads(fixture, report)
        combined = json.dumps([redacted_fixture, redacted_report])

        self.assertNotIn("abcdefghijklmnop1234", combined)
        self.assertEqual(combined.count("[REDACTED_SECRET]"), 2)
        self.assertEqual(redacted_report["result"]["fixture_redactions"], 1)
        self.assertEqual(redacted_report["result"]["provenance_redactions"], 1)
        self.assertEqual(redacted_report["result"]["redactions"], 2)

    def test_missing_edge_endpoints_are_counted_without_fabricating_nodes(self) -> None:
        row = {
            "vector_id": "node-a",
            "repo": "owner/repo",
            "type": "doc",
            "state": None,
            "labels": None,
            "milestone": None,
            "assignees": None,
            "updated_at": "2026-01-01T00:00:00Z",
            "number": None,
            "tag_name": None,
            "doc_path": "README.md",
            "commit_sha": "abc123",
            "file_path": None,
            "file_status": None,
            "commit_date": None,
            "commit_author": None,
            "tokenizer_kind": "natural",
            "content": "public documentation",
            "indexed_at": "2026-01-01T00:00:01Z",
        }
        edge = {
            "src_vector_id": "node-a",
            "dst_vector_id": "missing",
            "repo": "owner/repo",
            "src_slug": "README",
            "dst_slug": "missing",
            "edge_kind": "mention",
            "updated_at": "2026-01-01T00:00:02Z",
        }

        fixture, statistics = transform([row], [edge])

        self.assertEqual(len(fixture["nodes"]), 1)
        self.assertEqual(fixture["edges"], [])
        self.assertEqual(statistics["edges_one_endpoint_missing"], 1)
        self.assertEqual(fixture["nodes"][0]["metadata"]["vector_id"], "node-a")

    def test_provenance_comparison_exposes_count_commit_and_time_deltas(self) -> None:
        before = {
            "coverage": [
                {
                    "repo": "owner/repo",
                    "type": "diff",
                    "source_count": 2,
                    "distinct_commit_count": 1,
                    "newest_updated_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        after = {
            "coverage": [
                {
                    "repo": "owner/repo",
                    "type": "diff",
                    "source_count": 5,
                    "distinct_commit_count": 3,
                    "newest_updated_at": "2026-01-03T00:00:00Z",
                }
            ]
        }

        change = compare_reports(before, after)["coverage_changes"][0]

        self.assertEqual(change["source_count_delta"], 3)
        self.assertEqual(change["distinct_commit_count_delta"], 2)
        self.assertTrue(change["newest_extended"])


class RealCorpusFixtureTest(unittest.TestCase):
    def test_committed_fixture_is_canonical_and_has_complete_edge_endpoints(self) -> None:
        fixture = read_fixture(FIXTURE)
        node_ids = {node["node_id"] for node in fixture["nodes"]}
        canonical = json.dumps(
            fixture, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"

        self.assertEqual(FIXTURE.read_text(encoding="utf-8"), canonical)
        self.assertEqual(len(node_ids), 6)
        self.assertEqual(len(fixture["edges"]), 2)
        self.assertEqual(
            {node["metadata"]["type"] for node in fixture["nodes"]},
            {"diff", "wiki_doc"},
        )
        diff_dates = [
            node["metadata"]["commit_date"]
            for node in fixture["nodes"]
            if node["metadata"]["type"] == "diff"
        ]
        self.assertEqual(diff_dates, sorted(diff_dates))
        self.assertTrue(all(diff_dates))
        for node in fixture["nodes"]:
            self.assertEqual(node["confidence"], 1.0)
            self.assertEqual(node["metadata"]["vector_id"], node["node_id"])
            self.assertIn("updated_at", node["metadata"])
            self.assertIn("source_url", node["metadata"])
            self.assertNotIn("content", node["metadata"])
            self.assertNotIn("content_fts", node["metadata"])
        for edge in fixture["edges"]:
            self.assertIn(edge["source_id"], node_ids)
            self.assertIn(edge["target_id"], node_ids)
            self.assertEqual(edge["edge_type"], "mention")
            self.assertEqual(
                edge["metadata"]["source_record"]["edge_kind"], "mention"
            )

    def test_provenance_records_coverage_gap_state_and_zero_writes(self) -> None:
        report = json.loads(PROVENANCE.read_text(encoding="utf-8"))

        self.assertEqual(
            {row["type"] for row in report["coverage"]}, {"diff", "wiki_doc"}
        )
        coverage = {row["type"]: row for row in report["coverage"]}
        self.assertEqual(coverage["wiki_doc"]["distinct_commit_count"], 0)
        self.assertGreater(coverage["diff"]["distinct_commit_count"], 0)
        self.assertEqual(report["result"]["nodes_included"], 6)
        self.assertEqual(report["result"]["edges_included"], 2)
        self.assertEqual(report["result"]["redactions"], 0)
        self.assertEqual(report["result"]["fixture_redactions"], 0)
        self.assertEqual(report["result"]["provenance_redactions"], 0)
        self.assertIsInstance(report["known_gaps"], list)
        self.assertTrue(
            all(isinstance(value, str) and value for value in report["known_gaps"])
        )
        self.assertTrue(
            all(value == 0 for value in report["read_only_evidence"]["rows_written"])
        )
        self.assertTrue(
            all(value is False for value in report["read_only_evidence"]["changed_db"])
        )

    def test_real_fixture_exercises_ingest_graph_search_time_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with NeuronGraphRAG(
                Path(directory) / "fixture.db",
                config=EngineConfig(
                    sparse_weight=1.0,
                    dense_weight=0.0,
                    seed_count=1,
                    max_hops=1,
                    hop_decay=0.8,
                ),
            ) as engine:
                loaded = load_fixture(engine, FIXTURE)
                trace = engine.search("Purpose Declaration", limit=3, now=1_000.0)
                target = next(
                    hit
                    for hit in trace.hits
                    if hit.node.metadata["doc_path"] == "6.-Adapter"
                )
                before = engine.store.edge(
                    target.paths[0].steps[0].source_id,
                    target.paths[0].steps[0].target_id,
                    "mention",
                ).weight
                receipt = engine.record_success(
                    trace.trace_id, [target.node.node_id], now=1_001.0
                )
                after = engine.store.edge(
                    target.paths[0].steps[0].source_id,
                    target.paths[0].steps[0].target_id,
                    "mention",
                ).weight

                self.assertEqual((loaded.node_count, loaded.edge_count), (6, 2))
                self.assertGreater(target.graph_activation, 0.0)
                self.assertEqual(target.paths[0].steps[0].edge_type, "mention")
                self.assertLess(
                    trace.hits[0].node.metadata["updated_at"],
                    "2026-08-01T00:00:00Z",
                )
                self.assertTrue(receipt.reinforced_edges)
                self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
